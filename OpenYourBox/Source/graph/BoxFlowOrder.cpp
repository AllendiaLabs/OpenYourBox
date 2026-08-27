#include "BoxFlowOrder.h"

#include <algorithm>
#include <stack>
#include <unordered_set>

namespace openyourbox::graph {
namespace {
/**
 * @brief Sort key for one sibling box after flow ranks are assigned.
 */
struct BoxSortKey {
  /** @brief Longest-path rank of the box's strongly connected component. */
  int rank = 0;
  /** @brief User-visible label used for the alphabetical tie-break. */
  juce::String name;
  /** @brief Stable identifier used when names compare equal. */
  std::int32_t id = 0;
};

/**
 * @brief Returns true when @p left should appear before @p right.
 * @param left Left sort key.
 * @param right Right sort key.
 */
bool boxSortKeyLess(const BoxSortKey &left, const BoxSortKey &right) {
  if (left.rank != right.rank)
    return left.rank < right.rank;
  const auto nameCmp = left.name.compareIgnoreCase(right.name);
  if (nameCmp != 0)
    return nameCmp < 0;
  return left.id < right.id;
}

/**
 * @brief Packs a directed pair into a unique-edge key.
 * @param source Upstream identifier.
 * @param destination Downstream identifier.
 */
std::uint64_t packDirectedPair(std::int32_t source, std::int32_t destination) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32) |
         static_cast<std::uint32_t>(destination);
}

/**
 * @brief Tarjan state for contracting sibling-link strongly connected components.
 */
struct TarjanState {
  /** @brief Adjacency list in dense index space. */
  const std::vector<std::vector<std::size_t>> *outgoing = nullptr;
  /** @brief DFS discovery index, or -1 before visit. */
  std::vector<int> dfsIndex;
  /** @brief Lowest discovery index reachable in the current stack. */
  std::vector<int> lowLink;
  /** @brief True while the node remains on the Tarjan stack. */
  std::vector<char> onStack;
  /** @brief Tarjan node stack. */
  std::stack<std::size_t> stack;
  /** @brief Component id for each dense node index. */
  std::vector<int> componentOf;
  /** @brief Next DFS discovery index. */
  int nextDfsIndex = 0;
  /** @brief Next strongly connected component id. */
  int nextComponent = 0;

  /**
   * @brief Recursively assigns strongly connected component ids from @p node.
   * @param node Dense node index to visit.
   */
  void connect(std::size_t node) {
    dfsIndex[node] = lowLink[node] = nextDfsIndex++;
    stack.push(node);
    onStack[node] = 1;
    for (const auto next : (*outgoing)[node]) {
      if (dfsIndex[next] < 0) {
        connect(next);
        lowLink[node] = std::min(lowLink[node], lowLink[next]);
      } else if (onStack[next] != 0) {
        lowLink[node] = std::min(lowLink[node], dfsIndex[next]);
      }
    }
    if (lowLink[node] != dfsIndex[node])
      return;
    const auto component = nextComponent++;
    while (true) {
      const auto member = stack.top();
      stack.pop();
      onStack[member] = 0;
      componentOf[member] = component;
      if (member == node)
        break;
    }
  }
};
} // namespace

std::vector<std::int32_t> orderBoxesByFlowRank(
    std::vector<std::int32_t> boxIds,
    const std::unordered_map<std::int32_t, juce::String> &names,
    const std::vector<BoxFlowEdge> &edges) {
  if (boxIds.empty())
    return boxIds;

  std::unordered_map<std::int32_t, std::size_t> indexOf;
  indexOf.reserve(boxIds.size());
  for (std::size_t index = 0; index < boxIds.size(); ++index)
    indexOf.emplace(boxIds[index], index);

  std::vector<std::vector<std::size_t>> outgoing(boxIds.size());
  std::unordered_set<std::uint64_t> uniqueEdges;
  for (const auto &edge : edges) {
    if (edge.source == edge.destination)
      continue;
    const auto source = indexOf.find(edge.source);
    const auto destination = indexOf.find(edge.destination);
    if (source == indexOf.end() || destination == indexOf.end())
      continue;
    if (!uniqueEdges.insert(packDirectedPair(edge.source, edge.destination))
             .second)
      continue;
    outgoing[source->second].push_back(destination->second);
  }

  const auto count = boxIds.size();
  TarjanState tarjan;
  tarjan.outgoing = &outgoing;
  tarjan.dfsIndex.assign(count, -1);
  tarjan.lowLink.assign(count, 0);
  tarjan.onStack.assign(count, 0);
  tarjan.componentOf.assign(count, -1);
  for (std::size_t node = 0; node < count; ++node) {
    if (tarjan.dfsIndex[node] < 0)
      tarjan.connect(node);
  }

  const auto componentCount = static_cast<std::size_t>(tarjan.nextComponent);
  std::vector<std::vector<int>> componentOutgoing(componentCount);
  std::vector<int> componentIndegree(componentCount, 0);
  std::unordered_set<std::uint64_t> componentEdges;
  for (std::size_t node = 0; node < count; ++node) {
    for (const auto next : outgoing[node]) {
      const auto from = tarjan.componentOf[node];
      const auto to = tarjan.componentOf[next];
      if (from == to)
        continue;
      if (!componentEdges.insert(packDirectedPair(from, to)).second)
        continue;
      componentOutgoing[static_cast<std::size_t>(from)].push_back(to);
      ++componentIndegree[static_cast<std::size_t>(to)];
    }
  }

  std::vector<int> componentRank(componentCount, 0);
  std::vector<int> remaining = componentIndegree;
  std::vector<int> ready;
  ready.reserve(componentCount);
  for (int component = 0; component < tarjan.nextComponent; ++component) {
    if (remaining[static_cast<std::size_t>(component)] == 0)
      ready.push_back(component);
  }
  for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
    const auto component = ready[cursor];
    const auto from = static_cast<std::size_t>(component);
    for (const auto next : componentOutgoing[from]) {
      const auto to = static_cast<std::size_t>(next);
      componentRank[to] =
          std::max(componentRank[to], componentRank[from] + 1);
      if (--remaining[to] == 0)
        ready.push_back(next);
    }
  }

  std::vector<BoxSortKey> keys;
  keys.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    BoxSortKey key;
    key.id = boxIds[index];
    key.rank = componentRank[static_cast<std::size_t>(
        tarjan.componentOf[index])];
    const auto found = names.find(key.id);
    if (found != names.end())
      key.name = found->second;
    keys.push_back(std::move(key));
  }
  std::sort(keys.begin(), keys.end(), boxSortKeyLess);

  std::vector<std::int32_t> ordered;
  ordered.reserve(count);
  for (const auto &key : keys)
    ordered.push_back(key.id);
  return ordered;
}
} // namespace openyourbox::graph
