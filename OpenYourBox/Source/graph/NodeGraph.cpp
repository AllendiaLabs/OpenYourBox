#include "NodeGraph.h"
#include "BoxFlowOrder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <torch/script.h>

namespace {
void migrateLegacyMixerNode(openyourbox::graph::GraphNode &node,
                            const juce::String &storedType);

void normalizeMergeNodeProperties(openyourbox::graph::GraphNode &node);

void refreshPropagatedPinShapes(openyourbox::graph::NodeGraph &graph);
void refreshPropagatedPinShapesCore(openyourbox::graph::NodeGraph &graph);
void refreshOutputRepeatShapes(openyourbox::graph::NodeGraph &graph);

std::string utilityNewInputIncompatibilityMessage(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &merge, std::int32_t newSourcePinId);
std::string utilityInputsIncompatibilityMessage(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &merge);

/**
 * @brief Returns the 0-based input pin index of @p pinId on @p node.
 * @param node Element owning the pin.
 * @param pinId Destination pin identifier.
 * @return Matching index, or 0 when @p pinId is not an input of @p node.
 */
int inputPinIndexForId(const openyourbox::graph::GraphNode &node,
                       std::int32_t pinId) {
  for (int index = 0; index < static_cast<int>(node.inputs.size()); ++index) {
    if (node.inputs[static_cast<std::size_t>(index)].id == pinId)
      return index;
  }
  return 0;
}

/**
 * @brief Output pins that continue a signal entering @p node at @p entryPinId.
 *
 * Group Input/Output hubs pair lane i to lane i. Control inputs have no
 * through-path. Other pins continue to every non-control output so internal
 * latent hops can be walked when a selection deletes several boxes at once.
 * @param node Element being traversed.
 * @param entryPinId Input pin that received an incoming cable.
 */
std::vector<std::int32_t>
throughOutputPinIds(const openyourbox::graph::GraphNode &node,
                    std::int32_t entryPinId) {
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isGroupBoundaryType;
  using openyourbox::graph::resolveCollapsedPin;
  entryPinId = resolveCollapsedPin(entryPinId);
  if (isGroupBoundaryType(node.type)) {
    for (std::size_t index = 0; index < node.inputs.size(); ++index) {
      if (node.inputs[index].id != entryPinId)
        continue;
      if (index < node.outputs.size())
        return {node.outputs[index].id};
      return {};
    }
    return {};
  }
  const openyourbox::graph::Pin *entry = nullptr;
  for (const auto &pin : node.inputs) {
    if (pin.id == entryPinId) {
      entry = &pin;
      break;
    }
  }
  if (entry == nullptr || isControlInputPin(*entry))
    return {};
  std::vector<std::int32_t> outputs;
  outputs.reserve(node.outputs.size());
  for (const auto &pin : node.outputs)
    outputs.push_back(pin.id);
  return outputs;
}

/**
 * @brief Source/destination pin pairs that bypass a deleted node set.
 *
 * Each external incoming cable is walked through the cut along through-pins
 * until it exits to a surviving destination. When that walk finds no exit,
 * a 1-in or 1-out boundary still skip-wires (Audio In feeding the cut and
 * Audio Out leaving it, even if interiors are disconnected). Two-or-more
 * inputs with two-or-more outputs stay walk-only so lanes are not
 * cross-wired. Duplicate pairs are omitted.
 * @param graph Graph owning topology.
 * @param removedNodeIds Nodes that will be deleted, including group interiors.
 */
std::vector<std::pair<std::int32_t, std::int32_t>>
collectBypassPinPairs(const openyourbox::graph::NodeGraph &graph,
                      const std::unordered_set<std::int32_t> &removedNodeIds) {
  using openyourbox::graph::resolveCollapsedPin;
  std::unordered_set<std::int32_t> removedPins;
  for (const auto nodeId : removedNodeIds) {
    const auto *node = graph.findNode(nodeId);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs)
      removedPins.insert(pin.id);
    for (const auto &pin : node->outputs)
      removedPins.insert(pin.id);
  }

  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  std::unordered_set<std::uint64_t> seen;
  const auto addPair = [&](std::int32_t sourcePin, std::int32_t destPin) {
    sourcePin = resolveCollapsedPin(sourcePin);
    destPin = resolveCollapsedPin(destPin);
    if (sourcePin == destPin)
      return;
    const auto key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sourcePin))
         << 32) |
        static_cast<std::uint32_t>(destPin);
    if (!seen.insert(key).second)
      return;
    pairs.emplace_back(sourcePin, destPin);
  };

  std::vector<std::int32_t> incomingSources;
  std::vector<std::int32_t> outgoingDests;
  std::unordered_set<std::int32_t> seenIncoming;
  std::unordered_set<std::int32_t> seenOutgoing;
  for (const auto &link : graph.getLinks()) {
    const auto sourcePin = resolveCollapsedPin(link.sourcePinId);
    const auto destPin = resolveCollapsedPin(link.destinationPinId);
    const auto destRemoved = removedPins.count(destPin) != 0;
    const auto sourceRemoved = removedPins.count(sourcePin) != 0;
    if (!sourceRemoved && destRemoved && seenIncoming.insert(sourcePin).second)
      incomingSources.push_back(sourcePin);
    if (sourceRemoved && !destRemoved && seenOutgoing.insert(destPin).second)
      outgoingDests.push_back(destPin);
  }

  for (const auto &link : graph.getLinks()) {
    const auto sourcePin = resolveCollapsedPin(link.sourcePinId);
    const auto destPin = resolveCollapsedPin(link.destinationPinId);
    if (removedPins.count(destPin) == 0 || removedPins.count(sourcePin) != 0)
      continue;
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visitedPins;
    pending.push(destPin);
    while (!pending.empty()) {
      const auto pinId = pending.front();
      pending.pop();
      if (!visitedPins.insert(pinId).second)
        continue;
      const auto ownerId = graph.findNodeForPin(pinId);
      if (!ownerId.has_value() || removedNodeIds.count(*ownerId) == 0) {
        addPair(sourcePin, pinId);
        continue;
      }
      const auto *node = graph.findNode(*ownerId);
      if (node == nullptr)
        continue;
      for (const auto outPin : throughOutputPinIds(*node, pinId)) {
        for (const auto &hop : graph.getLinks()) {
          if (resolveCollapsedPin(hop.sourcePinId) != outPin)
            continue;
          pending.push(resolveCollapsedPin(hop.destinationPinId));
        }
      }
    }
  }

  std::unordered_set<std::int32_t> walkedSources;
  for (const auto &pair : pairs)
    walkedSources.insert(pair.first);
  const auto singleIncoming = incomingSources.size() == 1;
  const auto singleOutgoing = outgoingDests.size() == 1;
  for (const auto sourcePin : incomingSources) {
    if (walkedSources.count(sourcePin) != 0)
      continue;
    if (singleIncoming) {
      for (const auto destPin : outgoingDests)
        addPair(sourcePin, destPin);
    } else if (singleOutgoing) {
      addPair(sourcePin, outgoingDests.front());
    }
  }
  return pairs;
}

/**
 * @brief Collects every node and nested group owned by @p groupId.
 * @param graph Graph owning membership.
 * @param groupId Ancestor group.
 * @param nodeIds Destination node identifiers.
 * @param groupIds Destination group identifiers, including @p groupId.
 */
void collectGroupSubtree(const openyourbox::graph::NodeGraph &graph,
                         std::int32_t groupId,
                         std::unordered_set<std::int32_t> &nodeIds,
                         std::unordered_set<std::int32_t> &groupIds);

/**
 * @brief Returns the user-visible name of a node or group box.
 * @param graph Graph owning labels.
 * @param boxId Node or group identifier.
 */
juce::String boxDisplayName(const openyourbox::graph::NodeGraph &graph,
                            std::int32_t boxId) {
  if (const auto *group = graph.findGroup(boxId))
    return juce::String(group->name);
  if (const auto *node = graph.findNode(boxId))
    return juce::String(node->label);
  return {};
}

/**
 * @brief Returns true when @p boxId may appear in structure or library trees.
 * @param graph Graph owning nodes and groups.
 * @param boxId Node or group identifier.
 */
bool isStructureBox(const openyourbox::graph::NodeGraph &graph,
                    std::int32_t boxId) {
  if (const auto *node = graph.findNode(boxId))
    return !openyourbox::graph::isFixedIoType(node->type) &&
           !openyourbox::graph::isGroupBoundaryType(node->type);
  return graph.findGroup(boxId) != nullptr;
}

/**
 * @brief Collects every node id owned by @p boxId, including nested groups.
 * @param graph Graph owning nodes and groups.
 * @param boxId Node or group identifier.
 * @param nodeIds Destination node identifiers.
 */
void collectNodesUnderBox(const openyourbox::graph::NodeGraph &graph,
                          std::int32_t boxId,
                          std::unordered_set<std::int32_t> &nodeIds) {
  if (graph.findNode(boxId) != nullptr) {
    nodeIds.insert(boxId);
    return;
  }
  std::unordered_set<std::int32_t> groupIds;
  collectGroupSubtree(graph, boxId, nodeIds, groupIds);
}

/**
 * @brief Maps a node to the sibling box that owns it in @p candidates.
 * @param graph Graph owning membership.
 * @param candidates Displayed sibling node and group identifiers.
 * @param nodeId Endpoint node resolved from a link pin.
 */
std::optional<std::int32_t>
owningCandidateBox(const openyourbox::graph::NodeGraph &graph,
                   const std::unordered_set<std::int32_t> &candidates,
                   std::int32_t nodeId) {
  if (candidates.count(nodeId) != 0)
    return nodeId;
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr)
    return std::nullopt;

  auto groupId = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (groupId.has_value()) {
    if (candidates.count(*groupId) != 0)
      return *groupId;
    const auto *group = graph.findGroup(*groupId);
    if (group == nullptr || !visiting.insert(*groupId).second)
      break;
    groupId = group->parentGroupId;
  }
  return std::nullopt;
}

/**
 * @brief Collects nodes whose links can order @p candidates at one scope.
 * @param graph Graph owning nodes and membership.
 * @param scopeGroupId Parent group, or empty for the graph root.
 * @param candidates Displayed sibling node and group identifiers.
 * @param nodeIds Destination node identifiers.
 */
void collectScopeFlowNodes(const openyourbox::graph::NodeGraph &graph,
                           std::optional<std::int32_t> scopeGroupId,
                           const std::unordered_set<std::int32_t> &candidates,
                           std::unordered_set<std::int32_t> &nodeIds) {
  for (const auto boxId : candidates)
    collectNodesUnderBox(graph, boxId, nodeIds);
  for (const auto &node : graph.getNodes()) {
    if (scopeGroupId.has_value()) {
      if (node.parentGroupId == scopeGroupId)
        nodeIds.insert(node.id);
    } else if (!node.parentGroupId.has_value()) {
      nodeIds.insert(node.id);
    }
  }
}

/**
 * @brief Returns true when @p id is a member of @p group, including nested groups.
 * @param graph Graph owning groups and nodes.
 * @param groupId Ancestor group.
 * @param id Node or nested group identifier.
 */
bool groupOwnsId(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
                 std::int32_t id) {
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return false;
  for (const auto member : group->memberIds) {
    if (member == id)
      return true;
    if (graph.findGroup(member) != nullptr && groupOwnsId(graph, member, id))
      return true;
  }
  return false;
}

/**
 * @brief Walks parent groups from @p groupId toward the canvas root.
 * @param graph Graph owning groups.
 * @param groupId Starting group.
 * @return Depth 1 for a root group.
 */
int groupDepth(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId) {
  int depth = 0;
  auto current = groupId;
  std::unordered_set<std::int32_t> visiting;
  while (const auto *group = graph.findGroup(current)) {
    if (!visiting.insert(current).second)
      break;
    ++depth;
    if (!group->parentGroupId.has_value())
      break;
    current = *group->parentGroupId;
  }
  return depth;
}

/**
 * @brief Collects leaf node ids owned by @p groupId into @p out.
 * @param graph Graph owning groups and nodes.
 * @param groupId Ancestor group.
 * @param out Destination list.
 *
 * Walks @c memberIds and also any node whose @c parentGroupId is @p groupId so
 * compile-time repeat clones remain visible to outer unrolls.
 */
void collectLeaves(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
                   std::vector<std::int32_t> &out) {
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return;
  std::unordered_set<std::int32_t> seen;
  const auto append = [&](std::int32_t nodeId) {
    if (seen.insert(nodeId).second)
      out.push_back(nodeId);
  };
  for (const auto member : group->memberIds) {
    if (graph.findNode(member) != nullptr)
      append(member);
    else
      collectLeaves(graph, member, out);
  }
  for (const auto &node : graph.getNodes()) {
    if (node.parentGroupId.has_value() && *node.parentGroupId == groupId)
      append(node.id);
  }
}

/**
 * @brief Returns true when @p nodeId is a descendant of @p groupId.
 * @param graph Graph owning parent links.
 * @param nodeId Candidate node.
 * @param groupId Ancestor group.
 */
bool nodeIsInsideGroup(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t nodeId, std::int32_t groupId) {
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr)
    return false;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (*parent == groupId)
      return true;
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = graph.findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  return false;
}

/**
 * @brief Records a materialized repeat as a member of its parent group.
 * @param graph Expanded compile graph.
 * @param parentGroupId Group that owns the cloned node.
 * @param cloneId New node identifier.
 */
void registerCloneMembership(openyourbox::graph::NodeGraph &graph,
                             std::optional<std::int32_t> parentGroupId,
                             std::int32_t cloneId) {
  if (!parentGroupId.has_value())
    return;
  auto *parent = graph.findGroup(*parentGroupId);
  if (parent == nullptr)
    return;
  if (std::find(parent->memberIds.begin(), parent->memberIds.end(), cloneId) ==
      parent->memberIds.end())
    parent->memberIds.push_back(cloneId);
}

/**
 * @brief Collects every node and nested group owned by @p groupId.
 * @param graph Graph owning membership.
 * @param groupId Ancestor group.
 * @param nodeIds Destination node identifiers.
 * @param groupIds Destination group identifiers, including @p groupId.
 */
void collectGroupSubtree(const openyourbox::graph::NodeGraph &graph,
                         std::int32_t groupId,
                         std::unordered_set<std::int32_t> &nodeIds,
                         std::unordered_set<std::int32_t> &groupIds) {
  if (!groupIds.insert(groupId).second)
    return;
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return;
  for (const auto member : group->memberIds) {
    if (graph.findNode(member) != nullptr)
      nodeIds.insert(member);
    else
      collectGroupSubtree(graph, member, nodeIds, groupIds);
  }
}

/**
 * @brief Collects the same processing-node set that Train snapshots and absorbs.
 *
 * Starts from armed weighted nodes, then walks every live processing node
 * reachable from Audio Input so unarmed RAVE elements (PQMF, Utility, Math,
 * Activation, Noise Synth) stay in the trained module and the Gold replacement.
 * @param graph Graph to inspect.
 * @return Unique live processing node identifiers.
 */
std::unordered_set<std::int32_t> collectTrainSnapshotNodeIds(
    const openyourbox::graph::NodeGraph &graph) {
  using openyourbox::graph::NodeState;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::isControlSourceType;
  std::unordered_set<std::int32_t> selected;
  for (const auto nodeId : graph.getArmedTrainableNodeIds())
    selected.insert(nodeId);
  std::int32_t inputId = 0;
  for (const auto &node : graph.getNodes()) {
    if (node.type == NodeType::audioInput)
      inputId = node.id;
  }
  if (inputId == 0)
    return selected;
  std::queue<std::int32_t> pending;
  pending.push(inputId);
  std::unordered_set<std::int32_t> visited;
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (!visited.insert(current).second)
      continue;
    const auto *node = graph.findNode(current);
    if (node != nullptr && !isControlSourceType(node->type) &&
        node->state == NodeState::liveBlue)
      selected.insert(current);
    for (const auto &link : graph.getLinks()) {
      const auto source = graph.findNodeForPin(link.sourcePinId);
      const auto destination = graph.findNodeForPin(link.destinationPinId);
      if (source.has_value() && *source == current && destination.has_value())
        pending.push(*destination);
    }
  }
  return selected;
}

/**
 * @brief True when every non-hub member of @p groupId is in @p selected.
 *
 * Nested groups count as absorbed only when they themselves qualify. A Knob
 * or other control source inside the group blocks whole-group absorb.
 * @param graph Graph owning membership.
 * @param groupId Candidate group.
 * @param selected Processing nodes that will be replaced by Gold.
 * @param memo Recursion cache, group id → result.
 */
bool groupIsFullyAbsorbed(
    const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
    const std::unordered_set<std::int32_t> &selected,
    std::unordered_map<std::int32_t, bool> &memo) {
  using openyourbox::graph::isControlSourceType;
  using openyourbox::graph::isGroupBoundaryType;
  if (const auto found = memo.find(groupId); found != memo.end())
    return found->second;
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return false;
  bool hasContent = false;
  for (const auto member : group->memberIds) {
    if (const auto *node = graph.findNode(member)) {
      if (isGroupBoundaryType(node->type))
        continue;
      if (isControlSourceType(node->type) || selected.count(member) == 0) {
        memo[groupId] = false;
        return false;
      }
      hasContent = true;
    } else if (graph.findGroup(member) != nullptr) {
      if (!groupIsFullyAbsorbed(graph, member, selected, memo)) {
        memo[groupId] = false;
        return false;
      }
      hasContent = true;
    }
  }
  memo[groupId] = hasContent;
  return hasContent;
}

/**
 * @brief True when @p nodeId sits in @p absorbedGroups or a descendant.
 * @param graph Graph owning parent links.
 * @param nodeId Processing node.
 * @param absorbedGroups Groups that will be deleted with the chain.
 */
bool nodeCoveredByAbsorbedGroup(
    const openyourbox::graph::NodeGraph &graph, std::int32_t nodeId,
    const std::unordered_set<std::int32_t> &absorbedGroups) {
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr)
    return false;
  auto current = node->parentGroupId;
  while (current.has_value()) {
    if (absorbedGroups.count(*current) != 0)
      return true;
    const auto *group = graph.findGroup(*current);
    if (group == nullptr)
      break;
    current = group->parentGroupId;
  }
  return false;
}

/**
 * @brief Signal I/O used to stack independent group repeats in series.
 *
 * Uses the group's interface pins (unconnected internally) rather than
 * currently attached external cables, and omits TCN/BlackBox control inputs
 * so optional conditioning does not block a legal audio through-path.
 * @param graph Graph owning membership and pins.
 * @param groupId Target group.
 * @param inputs Destination list of chainable external inputs.
 * @param outputs Destination list of chainable external outputs.
 */
void appendSerialChainPorts(
    const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
    std::vector<openyourbox::graph::GroupBoundaryPort> &inputs,
    std::vector<openyourbox::graph::GroupBoundaryPort> &outputs) {
  using openyourbox::graph::PinKind;
  using openyourbox::graph::isControlInputPin;
  for (const auto &port : graph.groupInterfacePorts(groupId)) {
    const auto *pin = graph.findPin(port.memberPinId);
    if (pin != nullptr && isControlInputPin(*pin))
      continue;
    if (port.kind == PinKind::input)
      inputs.push_back(port);
    else
      outputs.push_back(port);
  }
}

/**
 * @brief Returns the parent group of a node or nested group.
 * @param graph Graph owning membership.
 * @param memberId Node or group identifier.
 */
std::optional<std::int32_t>
parentGroupOf(const openyourbox::graph::NodeGraph &graph, std::int32_t memberId) {
  if (const auto *node = graph.findNode(memberId))
    return node->parentGroupId;
  if (const auto *group = graph.findGroup(memberId))
    return group->parentGroupId;
  return std::nullopt;
}

/**
 * @brief Axis-aligned bounds of a node or nested group.
 * @param graph Graph owning layout.
 * @param memberId Node or group identifier.
 */
std::optional<std::pair<juce::Point<float>, juce::Point<float>>>
memberBounds(const openyourbox::graph::NodeGraph &graph, std::int32_t memberId) {
  if (const auto *node = graph.findNode(memberId))
    return std::make_pair(node->position, node->size);
  if (const auto *group = graph.findGroup(memberId))
    return std::make_pair(group->position, group->size);
  return std::nullopt;
}

/**
 * @brief Writes the stored position of a node or nested group.
 * @param graph Graph to mutate.
 * @param memberId Node or group identifier.
 * @param position New stored position.
 */
void setItemStoredPosition(openyourbox::graph::NodeGraph &graph,
                           std::int32_t memberId, juce::Point<float> position) {
  if (auto *node = graph.findNode(memberId))
    node->position = position;
  else if (auto *group = graph.findGroup(memberId))
    group->position = position;
}

/**
 * @brief Places Group Input/Output around members, preserving relative layout.
 *
 * Content stays in its original arrangement. The cluster is translated so
 * Group Input sits on the left and Group Output on the right, each vertically
 * centred on the content and separated by @ref groupBoundaryContentGap.
 * @param graph Graph to mutate.
 * @param groupId Group whose interior was just given boundary hubs.
 */
void layoutGroupInteriorAroundBoundaryHubs(
    openyourbox::graph::NodeGraph &graph, std::int32_t groupId) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::groupBoundaryContentGap;
  using openyourbox::graph::groupInteriorOrigin;
  auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return;

  openyourbox::graph::GraphNode *inputHub = nullptr;
  openyourbox::graph::GraphNode *outputHub = nullptr;
  std::vector<std::int32_t> contentIds;
  contentIds.reserve(group->memberIds.size());
  for (const auto memberId : group->memberIds) {
    if (auto *node = graph.findNode(memberId)) {
      if (node->type == NodeType::groupInput)
        inputHub = node;
      else if (node->type == NodeType::groupOutput)
        outputHub = node;
      else
        contentIds.push_back(memberId);
    } else if (graph.findGroup(memberId) != nullptr) {
      contentIds.push_back(memberId);
    }
  }
  if (inputHub == nullptr || outputHub == nullptr)
    return;

  const auto hubExtent = [](const openyourbox::graph::GraphNode &hub) {
    return juce::Point<float>(std::max(8.0f, hub.size.x),
                               std::max(8.0f, hub.size.y));
  };
  const auto inputSize = hubExtent(*inputHub);
  const auto outputSize = hubExtent(*outputHub);

  auto contentMin = juce::Point<float>(std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max());
  auto contentMax = juce::Point<float>(std::numeric_limits<float>::lowest(),
                                        std::numeric_limits<float>::lowest());
  auto anyContent = false;
  for (const auto id : contentIds) {
    const auto bounds = memberBounds(graph, id);
    if (!bounds.has_value())
      continue;
    const auto size = juce::Point<float>(std::max(8.0f, bounds->second.x),
                                        std::max(8.0f, bounds->second.y));
    anyContent = true;
    contentMin.x = std::min(contentMin.x, bounds->first.x);
    contentMin.y = std::min(contentMin.y, bounds->first.y);
    contentMax.x = std::max(contentMax.x, bounds->first.x + size.x);
    contentMax.y = std::max(contentMax.y, bounds->first.y + size.y);
  }

  if (!anyContent) {
    inputHub->position = groupInteriorOrigin;
    outputHub->position = {groupInteriorOrigin.x + inputSize.x +
                               groupBoundaryContentGap * 2.0f,
                           groupInteriorOrigin.y};
    return;
  }

  const auto targetContentMinX =
      groupInteriorOrigin.x + inputSize.x + groupBoundaryContentGap;
  const auto contentCenterY = (contentMin.y + contentMax.y) * 0.5f;
  const auto targetCenterY = groupInteriorOrigin.y + inputSize.y * 0.5f;
  const auto translation = juce::Point<float>(targetContentMinX - contentMin.x,
                                               targetCenterY - contentCenterY);
  for (const auto id : contentIds) {
    const auto bounds = memberBounds(graph, id);
    if (!bounds.has_value())
      continue;
    setItemStoredPosition(graph, id, bounds->first + translation);
  }
  contentMin += translation;
  contentMax += translation;
  const auto centredY = (contentMin.y + contentMax.y) * 0.5f;
  inputHub->position = {groupInteriorOrigin.x, centredY - inputSize.y * 0.5f};
  outputHub->position = {contentMax.x + groupBoundaryContentGap,
                          centredY - outputSize.y * 0.5f};
}

/**
 * @brief Canvas-space origin of a node or nested group.
 * @param graph Graph owning layout.
 * @param memberId Node or group identifier.
 */
juce::Point<float> itemWorldPosition(const openyourbox::graph::NodeGraph &graph,
                                     std::int32_t memberId) {
  if (graph.findNode(memberId) != nullptr)
    return graph.worldPositionOfNode(memberId);
  return graph.worldPositionOfGroup(memberId);
}

/**
 * @brief Writes a canvas-space origin into the item's current parent space.
 * @param graph Graph to mutate.
 * @param memberId Node or group identifier.
 * @param worldPoint Canvas-space origin.
 * @param parent Group that currently owns the item, or empty at the root.
 */
void storeWorldPosition(openyourbox::graph::NodeGraph &graph,
                        std::int32_t memberId, juce::Point<float> worldPoint,
                        std::optional<std::int32_t> parent) {
  const auto stored =
      parent.has_value() ? graph.worldToGroupLocal(*parent, worldPoint)
                         : worldPoint;
  setItemStoredPosition(graph, memberId, stored);
}

/**
 * @brief Ensures Activation and TCN nodes expose a validated Gain property.
 * @param node Loaded or newly created processing node.
 */
void normalizeGainProperty(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::isRecurrentType;
  if (node.type != NodeType::activation && node.type != NodeType::tcn &&
      !isRecurrentType(node.type))
    return;
  for (auto &property : node.properties) {
    if (property.key != "gain")
      continue;
    property.kind = PropertyKind::real;
    property.label = "Gain";
    property.floatMinimum = openyourbox::graph::gainMinimum;
    property.floatMaximum = openyourbox::graph::gainMaximum;
    if (property.floatValue <= 0.0f)
      property.floatValue = openyourbox::graph::gainDefault;
    property.setFloatValue(property.floatValue);
    return;
  }
  openyourbox::graph::NodeProperty gain;
  gain.key = "gain";
  gain.label = "Gain";
  gain.kind = PropertyKind::real;
  gain.floatValue = openyourbox::graph::gainDefault;
  gain.floatMinimum = openyourbox::graph::gainMinimum;
  gain.floatMaximum = openyourbox::graph::gainMaximum;
  node.properties.push_back(std::move(gain));
}

/**
 * @brief Ensures Activation and TCN nodes expose LeakyReLU negative slope.
 * @param node Loaded or newly created processing node.
 */
void normalizeNegativeSlopeProperty(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::isRecurrentType;
  if (node.type != NodeType::activation && node.type != NodeType::tcn &&
      !isRecurrentType(node.type))
    return;
  for (auto &property : node.properties) {
    if (property.key != "negative_slope")
      continue;
    property.kind = PropertyKind::real;
    property.label = "Negative slope";
    property.floatMinimum = openyourbox::graph::leakyReluNegativeSlopeMinimum;
    property.floatMaximum = openyourbox::graph::leakyReluNegativeSlopeMaximum;
    if (property.floatValue < property.floatMinimum ||
        property.floatValue > property.floatMaximum)
      property.floatValue = openyourbox::graph::leakyReluNegativeSlopeDefault;
    return;
  }
  openyourbox::graph::NodeProperty slope;
  slope.key = "negative_slope";
  slope.label = "Negative slope";
  slope.kind = PropertyKind::real;
  slope.floatValue = openyourbox::graph::leakyReluNegativeSlopeDefault;
  slope.floatMinimum = openyourbox::graph::leakyReluNegativeSlopeMinimum;
  slope.floatMaximum = openyourbox::graph::leakyReluNegativeSlopeMaximum;
  node.properties.push_back(std::move(slope));
}

/**
 * @brief Ensures LSTM/RNN leak rate and recurrent-weight scale exist.
 * @param node Loaded or newly created processing node.
 */
void normalizeRecurrentProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::isRecurrentType;
  using openyourbox::graph::leakRateDefault;
  using openyourbox::graph::leakRateMaximum;
  using openyourbox::graph::leakRateMinimum;
  using openyourbox::graph::recurrentWeightScaleDefault;
  using openyourbox::graph::recurrentWeightScaleMaximum;
  using openyourbox::graph::recurrentWeightScaleMinimum;
  if (!isRecurrentType(node.type))
    return;
  auto ensureReal = [&node](const char *key, const char *label, float value,
                              float minimum, float maximum) {
    for (auto &property : node.properties) {
      if (property.key != key)
        continue;
      property.kind = PropertyKind::real;
      property.label = label;
      property.floatMinimum = minimum;
      property.floatMaximum = maximum;
      if (property.floatValue < minimum || property.floatValue > maximum)
        property.floatValue = value;
      property.setFloatValue(property.floatValue);
      return;
    }
    NodeProperty property;
    property.key = key;
    property.label = label;
    property.kind = PropertyKind::real;
    property.floatValue = value;
    property.floatMinimum = minimum;
    property.floatMaximum = maximum;
    node.properties.push_back(std::move(property));
  };
  ensureReal("leak_rate", "Leak Rate", leakRateDefault, leakRateMinimum,
             leakRateMaximum);
  ensureReal("recurrent_weight_scale", "Recurrent Weight Scale",
             recurrentWeightScaleDefault, recurrentWeightScaleMinimum,
             recurrentWeightScaleMaximum);
}

/**
 * @brief Ensures Phase 3 TCN/Activation fields exist on loaded nodes.
 * @param node Loaded or newly created processing node.
 */
void normalizePhase3Node(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::defaultDilationGrowth;
  using openyourbox::graph::isTrainableType;
  using openyourbox::graph::minimumDilationGrowth;
  using openyourbox::graph::unlimitedPropertyMaximum;
  using openyourbox::graph::widenIntegerPropertyBounds;

  auto ensureActivationChoices = [](NodeProperty &property) {
    if (property.key != "activation")
      return;
    property.kind = PropertyKind::choice;
    property.choices = {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"};
    property.minimum = 0;
    property.maximum = 4;
    property.setValue(property.value);
  };

  if (node.type == NodeType::activation) {
    for (auto &property : node.properties)
      ensureActivationChoices(property);
    if (node.hasWeights)
      node.armedForTraining = node.armedForTraining || true;
  }

  if (node.type != NodeType::tcn)
    return;

  auto ensureInt = [&node](const char *key, const char *label, int value,
                           int minimum, int maximum) {
    for (auto &property : node.properties) {
      if (property.key == key) {
        property.minimum = minimum;
        property.maximum = maximum;
        property.setValue(property.value);
        return;
      }
    }
    NodeProperty property;
    property.key = key;
    property.label = label;
    property.value = value;
    property.minimum = minimum;
    property.maximum = maximum;
    node.properties.push_back(property);
  };
  ensureInt("dilation_growth", "Dilation growth", defaultDilationGrowth,
            minimumDilationGrowth, unlimitedPropertyMaximum);
  ensureInt("residual", "Residual", 0, 0, 1);
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
  node.properties.erase(
      std::remove_if(node.properties.begin(), node.properties.end(),
                     [](const NodeProperty &property) {
                       return property.key == "dilation";
                     }),
      node.properties.end());
  for (auto &property : node.properties)
    ensureActivationChoices(property);
  if (node.hasWeights)
    node.armedForTraining = true;
}

/**
 * @brief Restores Knob/XY widths and TCN/BlackBox control pin roles.
 * @param node Loaded source node.
 */
void normalizeConditioningPins(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::controlPinLabel;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  if (openyourbox::graph::isConditioningSourceType(node.type)) {
    for (auto &pin : node.outputs) {
      if (pin.shape.channels <= 0)
        pin.shape.channels = 1;
    }
    return;
  }
  if (node.type != openyourbox::graph::NodeType::tcn &&
      node.type != openyourbox::graph::NodeType::blackBox)
    return;
  for (auto &pin : node.inputs) {
    if (!isControlInputPin(pin))
      continue;
    pin.label = controlPinLabel;
    pin.shape = flexibleTensorShape();
  }
}

/**
 * @brief Reads an integer node property when present.
 * @param node Graph node to inspect.
 * @param key Property key.
 * @param fallback Value returned when the property is absent.
 * @return Stored property value or @p fallback.
 */
int readNodeProperty(const openyourbox::graph::GraphNode &node, const char *key,
                     int fallback = 0) {
  const auto property = std::find_if(
      node.properties.begin(), node.properties.end(),
      [key](const openyourbox::graph::NodeProperty &candidate) {
        return candidate.key == key;
      });
  return property == node.properties.end() ? fallback : property->value;
}

/**
 * @brief Reads a shape-driving integer, honoring `in` bindings and invalid lists.
 * @param node Graph node to inspect.
 * @param key Property key (`features` or `channels`).
 * @param incomingChannels Paired input channel count for `in` resolution.
 * @param fallback Value returned when the property is absent.
 * @return Effective integer, or 0 when unresolved/invalid.
 */
int readBoundShapeProperty(const openyourbox::graph::GraphNode &node,
                           const char *key, int incomingChannels,
                           int fallback = 0) {
  for (const auto &property : node.properties) {
    if (property.key != key)
      continue;
    if (property.repeatListInvalid)
      return 0;
    if (property.preserveInBound)
      return incomingChannels > 0 ? incomingChannels : 0;
    return property.value > 0 ? property.value : fallback;
  }
  return fallback;
}

/**
 * @brief Returns true when @p key on @p node is flagged repeat-list invalid.
 * @param node Graph node to inspect.
 * @param key Property key.
 */
bool propertyRepeatListIsInvalid(const openyourbox::graph::GraphNode &node,
                               const char *key) {
  for (const auto &property : node.properties) {
    if (property.key == key)
      return property.repeatListInvalid;
  }
  return false;
}

/**
 * @brief Returns the configured Utility operating mode for one node.
 * @param node Utility or legacy mixer node.
 * @return Merge mode index.
 */
int mergeModeFor(const openyourbox::graph::GraphNode &node) {
  return std::clamp(readNodeProperty(node, "mode", 0), 0, 2);
}

/**
 * @brief Returns the authored Math Expression formula, or `x1` when missing.
 * @param node Math Expression node.
 */
std::string mathExpressionText(const openyourbox::graph::GraphNode &node) {
  for (const auto &property : node.properties) {
    if (property.key == "expression")
      return property.stringValue.empty() ? "x1" : property.stringValue;
  }
  return "x1";
}

/**
 * @brief Parses the current Math Expression formula for @p node.
 * @param node Math Expression node whose Inputs count bounds `xK`.
 */
openyourbox::graph::ExpressionAst
parsedMathExpression(const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::ExpressionIdentContext;
  using openyourbox::graph::parseExpression;
  const auto n = std::max(1, static_cast<int>(node.inputs.size()));
  const auto parsed =
      parseExpression(mathExpressionText(node), ExpressionIdentContext::mathInputs,
                      n);
  return parsed.accepted ? parsed.ast : openyourbox::graph::ExpressionAst{};
}

/**
 * @brief Returns whether two Utility input widths can share add/multiply.
 * @param left First width, or zero when unknown.
 * @param right Second width, or zero when unknown.
 * @return True when the widths match or either side can broadcast from 1.
 */
bool channelsAreBroadcastCompatible(int left, int right) noexcept {
  return left == 0 || right == 0 || left == right || left == 1 || right == 1;
}

int resolvePinChannels(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t pinId,
                       std::unordered_set<std::int32_t> &visiting);

/**
 * @brief Computes Utility output channels from currently connected inputs.
 * @param graph Graph document used for upstream resolution.
 * @param node Utility node to inspect.
 * @param visiting Active pin-resolution stack for cycle detection.
 * @return Output channel count, or zero when unknown.
 */
int computeMergeOutputChannels(const openyourbox::graph::NodeGraph &graph,
                               const openyourbox::graph::GraphNode &node,
                               std::unordered_set<std::int32_t> &visiting) {
  using openyourbox::graph::MergeMode;
  const auto mode = mergeModeFor(node);
  std::vector<int> connectedChannels;
  connectedChannels.reserve(node.inputs.size());
  for (const auto &inputPin : node.inputs) {
    if (openyourbox::graph::isMathExpressionType(node.type)) {
      int pinIndex = 0;
      for (const auto &candidate : node.inputs) {
        if (candidate.id == inputPin.id)
          break;
        ++pinIndex;
      }
      if (!openyourbox::graph::mathExpressionReferencesInput(
              parsedMathExpression(node), pinIndex + 1))
        continue;
    }
    for (const auto &link : graph.getLinks()) {
      if (link.destinationPinId != inputPin.id)
        continue;
      visiting.clear();
      const auto channels =
          resolvePinChannels(graph, link.sourcePinId, visiting);
      if (channels > 0)
        connectedChannels.push_back(channels);
      break;
    }
  }
  if (connectedChannels.empty())
    return 0;
  if (mode == static_cast<int>(MergeMode::concatenate)) {
    int outputChannels = 0;
    for (const auto channels : connectedChannels)
      outputChannels += channels;
    return outputChannels;
  }
  int width = 0;
  for (const auto channels : connectedChannels) {
    if (!channelsAreBroadcastCompatible(width, channels))
      return 0;
    width = std::max(width, channels);
  }
  return width;
}

int resolvePinChannels(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t pinId,
                       std::unordered_set<std::int32_t> &visiting) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PinKind;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::defaultHiddenSize;
  using openyourbox::graph::defaultNoiseBands;
  if (!visiting.insert(pinId).second)
    return 0;

  const auto *pin = graph.findPin(pinId);
  if (pin == nullptr)
    return 0;
  if (pin->shape.channels > 0)
    return pin->shape.channels;

  const auto ownerId = graph.findNodeForPin(pinId);
  if (!ownerId.has_value())
    return 0;
  const auto *node = graph.findNode(*ownerId);
  if (node == nullptr)
    return 0;

  if (pin->kind == PinKind::input) {
    for (const auto &link : graph.getLinks()) {
      if (link.destinationPinId != pinId)
        continue;
      return resolvePinChannels(graph, link.sourcePinId, visiting);
    }
    return 0;
  }

  switch (node->type) {
  case NodeType::audioInput:
    return openyourbox::graph::hostIoChannelsFromChoice(
        readNodeProperty(*node, "channels",
                         openyourbox::graph::hostIoChoiceFromChannels(2)));
  case NodeType::groupInput:
  case NodeType::groupOutput:
    for (std::size_t index = 0; index < node->outputs.size(); ++index) {
      if (node->outputs[index].id == pinId && index < node->inputs.size())
        return resolvePinChannels(graph, node->inputs[index].id, visiting);
    }
    return 0;
  case NodeType::linear: {
    const auto incoming =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return readBoundShapeProperty(*node, "features", incoming, 0);
  }
  case NodeType::convolution: {
    const auto incoming =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return readBoundShapeProperty(*node, "channels", incoming, 0);
  }
  case NodeType::convTranspose: {
    const auto incoming =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return readBoundShapeProperty(*node, "channels", incoming, 0);
  }
  case NodeType::merge:
  case NodeType::mathExpression:
    visiting.erase(pinId);
    return computeMergeOutputChannels(graph, *node, visiting);
  case NodeType::activation:
  case NodeType::tcn:
  case NodeType::batchNorm:
  case NodeType::audioOutput:
  case NodeType::reverb:
  case NodeType::expDecayReverb:
  case NodeType::filteredNoiseReverb:
  case NodeType::firFilter:
  case NodeType::modDelay:
    if (node->inputs.empty())
      return 0;
    return resolvePinChannels(graph, node->inputs.front().id, visiting);
  case NodeType::lstm:
  case NodeType::rnn: {
    const auto hidden =
        std::max(1, readNodeProperty(*node, "hidden_size", defaultHiddenSize));
    const auto bidirectional = readNodeProperty(*node, "bidirectional", 0) != 0;
    return bidirectional ? hidden * 2 : hidden;
  }
  case NodeType::knobInput:
  case NodeType::xyTrackpad:
    return 1;
  case NodeType::pqmfAnalysis: {
    const auto nBand = readNodeProperty(*node, "n_band", defaultPqmfBands);
    const auto audioChannels =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return nBand * std::max(1, audioChannels);
  }
  case NodeType::pqmfSynthesis: {
    const auto nBand = readNodeProperty(*node, "n_band", defaultPqmfBands);
    const auto bands =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return nBand > 0 ? std::max(1, bands / nBand) : bands;
  }
  case NodeType::rateConv: {
    const auto channels = readNodeProperty(*node, "channels", 0);
    return channels > 0 ? channels : 0;
  }
  case NodeType::variationalBottleneck:
    return readNodeProperty(*node, "latent_size", defaultLatentSize);
  case NodeType::noiseSynthesizer: {
    if (node->inputs.empty())
      return 0;
    const auto incoming =
        resolvePinChannels(graph, node->inputs.front().id, visiting);
    const auto noiseBands =
        std::max(1, readNodeProperty(*node, "noise_bands", defaultNoiseBands));
    if (incoming <= 0)
      return 0;
    if (incoming % noiseBands != 0)
      return 0;
    return incoming / noiseBands;
  }
  default:
    return 0;
  }
}

/**
 * @brief Updates the declared output channels on one Utility node.
 * @param graph Graph document to mutate.
 * @param node Utility node to refresh.
 */
void updateMergeOutputShape(openyourbox::graph::NodeGraph &graph,
                            openyourbox::graph::GraphNode &node) {
  if (node.outputs.empty())
    return;
  std::unordered_set<std::int32_t> visiting;
  node.outputs.front().shape.channels =
      computeMergeOutputChannels(graph, node, visiting);
}

/**
 * @brief Refreshes output channel declarations on every Utility node.
 * @param graph Graph document to mutate.
 */
void refreshAllMergeOutputShapes(openyourbox::graph::NodeGraph &graph) {
  refreshPropagatedPinShapes(graph);
}

/**
 * @brief Returns true when downstream consumers accept the merge output width.
 * @param graph Graph document to inspect.
 * @param merge Utility node whose output width was computed.
 * @return False when a connected destination requires a different channel count.
 */
bool mergeDownstreamIsCompatible(const openyourbox::graph::NodeGraph &graph,
                               const openyourbox::graph::GraphNode &merge) {
  if (merge.outputs.empty())
    return true;
  const auto outputChannels = merge.outputs.front().shape.channels;
  if (outputChannels == 0)
    return true;

  for (const auto &link : graph.getLinks()) {
    if (link.sourcePinId != merge.outputs.front().id)
      continue;
    const auto *destinationPin = graph.findPin(link.destinationPinId);
    if (destinationPin == nullptr)
      continue;
    std::unordered_set<std::int32_t> visiting;
    const auto destinationChannels =
        destinationPin->shape.channels > 0
            ? destinationPin->shape.channels
            : resolvePinChannels(graph, link.destinationPinId, visiting);
    if (destinationChannels > 0 && destinationChannels != outputChannels)
      return false;
  }
  return true;
}

const openyourbox::graph::Pin *
findConnectedSourcePin(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t destinationPinId) {
  for (const auto &link : graph.getLinks()) {
    if (link.destinationPinId == destinationPinId)
      return graph.findPin(link.sourcePinId);
  }
  return nullptr;
}

/**
 * @brief Shape a source pin presents to downstream consumers.
 *
 * Group Output hub exits attach to last-out of the serial stack; every other
 * pin uses its stored @c shape.
 * @param graph Graph owning the pin.
 * @param pin Source endpoint.
 */
openyourbox::graph::ShapeSignature outgoingShapeOf(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::Pin &pin) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PinKind;
  using openyourbox::graph::collapsedGroupAttachShape;
  if (pin.kind != PinKind::output)
    return pin.shape;
  const auto ownerId = graph.findNodeForPin(pin.id);
  if (!ownerId.has_value())
    return pin.shape;
  const auto *owner = graph.findNode(*ownerId);
  if (owner == nullptr || owner->type != NodeType::groupOutput)
    return pin.shape;
  return collapsedGroupAttachShape(pin.repeatShapes, pin.shape,
                                   graph.ancestorRuntimeRepeatCounts(owner->id),
                                   true);
}

/**
 * @brief Resolves the shape a Utility input would consume from @p sourcePinId.
 * @param graph Graph used to walk unspecified channel counts.
 * @param sourcePinId Upstream output pin.
 * @return Stored outgoing shape, with channels filled from graph resolution.
 */
openyourbox::graph::ShapeSignature resolvedUtilitySourceShape(
    const openyourbox::graph::NodeGraph &graph, std::int32_t sourcePinId) {
  const auto *pin = graph.findPin(sourcePinId);
  openyourbox::graph::ShapeSignature shape;
  if (pin != nullptr)
    shape = outgoingShapeOf(graph, *pin);
  if (shape.channels <= 0) {
    std::unordered_set<std::int32_t> visiting;
    const auto channels = resolvePinChannels(graph, sourcePinId, visiting);
    if (channels > 0)
      shape.channels = channels;
  }
  return shape;
}

/**
 * @brief Collects shapes of currently connected Utility inputs.
 * @param graph Graph owning the cables.
 * @param merge Utility node to inspect.
 * @return One shape per connected input, in pin order.
 */
std::vector<openyourbox::graph::ShapeSignature> connectedUtilitySourceShapes(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &merge,
    bool referencedOnly) {
  std::vector<openyourbox::graph::ShapeSignature> sources;
  sources.reserve(merge.inputs.size());
  const auto ast = referencedOnly &&
                           openyourbox::graph::isMathExpressionType(merge.type)
                       ? parsedMathExpression(merge)
                       : openyourbox::graph::ExpressionAst{};
  int pinIndex = 0;
  for (const auto &inputPin : merge.inputs) {
    const auto include =
        !referencedOnly || ast.instructions.empty() ||
        openyourbox::graph::mathExpressionReferencesInput(ast, pinIndex + 1);
    ++pinIndex;
    if (!include)
      continue;
    if (const auto *source = findConnectedSourcePin(graph, inputPin.id))
      sources.push_back(resolvedUtilitySourceShape(graph, source->id));
  }
  return sources;
}

/**
 * @brief Explains why two Utility sources cannot share add, multiply, or concat.
 * @param left First connected or proposed source.
 * @param right Second connected or proposed source.
 * @param concatenate True when channel counts may differ (concat mode).
 * @return Empty when the pair is legal.
 */
std::string utilitySourcePairIncompatibilityMessage(
    const openyourbox::graph::ShapeSignature &left,
    const openyourbox::graph::ShapeSignature &right, bool concatenate) {
  auto rateLeft = left;
  auto rateRight = right;
  rateLeft.channels = 0;
  rateRight.channels = 0;
  if (!rateLeft.isCompatibleWith(rateRight)) {
    auto message = rateLeft.incompatibilityMessage(rateRight);
    if (message.empty())
      message = "Utility inputs must share temporal rate and band count";
    return message;
  }
  if (!concatenate &&
      !channelsAreBroadcastCompatible(left.channels, right.channels)) {
    const auto leftLabel =
        left.displayLabel().empty() ? "unspecified" : left.displayLabel();
    const auto rightLabel =
        right.displayLabel().empty() ? "unspecified" : right.displayLabel();
    return "Utility inputs cannot combine (" + leftLabel + " vs " +
           rightLabel +
           "); channels must match or either side may be 1";
  }
  return {};
}

/**
 * @brief Folds Utility sources left-to-right under add/multiply broadcast rules.
 * @param sources Connected or proposed input shapes.
 * @param concatenate True to skip channel broadcast and only match rate/bands.
 * @return Empty when every fold step is legal.
 */
std::string utilityFoldedSourcesIncompatibilityMessage(
    const std::vector<openyourbox::graph::ShapeSignature> &sources,
    bool concatenate) {
  if (sources.size() < 2)
    return {};
  auto combined = sources.front();
  for (std::size_t index = 1; index < sources.size(); ++index) {
    if (const auto message = utilitySourcePairIncompatibilityMessage(
            combined, sources[index], concatenate);
        !message.empty())
      return message;
    if (concatenate)
      combined.channels += sources[index].channels;
    else
      combined.channels = std::max(combined.channels, sources[index].channels);
  }
  return {};
}

/**
 * @brief Validates every connected Utility input against add/multiply rules.
 * @param graph Graph owning the cables.
 * @param merge Utility node to inspect.
 * @return Empty when connected inputs are legal for the current mode.
 */
std::string utilityInputsIncompatibilityMessage(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &merge) {
  using openyourbox::graph::MergeMode;
  using openyourbox::graph::NodeType;
  if (merge.type != NodeType::merge &&
      !openyourbox::graph::isMathExpressionType(merge.type))
    return {};
  const auto concatenate =
      merge.type == NodeType::merge &&
      mergeModeFor(merge) == static_cast<int>(MergeMode::concatenate);
  const auto referencedOnly =
      openyourbox::graph::isMathExpressionType(merge.type);
  return utilityFoldedSourcesIncompatibilityMessage(
      connectedUtilitySourceShapes(graph, merge, referencedOnly), concatenate);
}

/**
 * @brief Validates a proposed Utility input against already connected sources.
 * @param graph Graph owning the cables.
 * @param merge Destination Utility node.
 * @param newSourcePinId Source pin proposed for one Utility input.
 * @return Empty when the new source is legal for the current mode.
 */
std::string utilityNewInputIncompatibilityMessage(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &merge, std::int32_t newSourcePinId) {
  using openyourbox::graph::MergeMode;
  const auto concatenate =
      merge.type == openyourbox::graph::NodeType::merge &&
      mergeModeFor(merge) == static_cast<int>(MergeMode::concatenate);
  const auto referencedOnly =
      openyourbox::graph::isMathExpressionType(merge.type);
  auto sources = connectedUtilitySourceShapes(graph, merge, referencedOnly);
  sources.push_back(resolvedUtilitySourceShape(graph, newSourcePinId));
  return utilityFoldedSourcesIncompatibilityMessage(sources, concatenate);
}

/**
 * @brief Writes the parent-canvas last-out attach shape onto a Group Output hub.
 * @param graph Graph owning the hub.
 * @param node Group Output hub to update.
 */
void applyGroupOutputExitShapes(openyourbox::graph::NodeGraph &graph,
                                openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::collapsedGroupAttachShape;
  const auto counts = graph.ancestorRuntimeRepeatCounts(node.id);
  const auto laneCount = std::min(node.inputs.size(), node.outputs.size());
  for (std::size_t index = 0; index < laneCount; ++index) {
    auto &output = node.outputs[index];
    output.shape = collapsedGroupAttachShape(output.repeatShapes,
                                             node.inputs[index].shape, counts,
                                             true);
  }
}

/**
 * @brief Returns the first connected tensor source shape, or a wildcard.
 * @param graph Graph document to inspect.
 * @param node Node whose non-control inputs are scanned.
 */
openyourbox::graph::ShapeSignature
firstConnectedTensorSource(const openyourbox::graph::NodeGraph &graph,
                           const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isIrInputPin;
  for (const auto &pin : node.inputs) {
    if (isControlInputPin(pin) || isIrInputPin(pin))
      continue;
    if (const auto *source = findConnectedSourcePin(graph, pin.id))
      return outgoingShapeOf(graph, *source);
  }
  return flexibleTensorShape();
}

/**
 * @brief Copies inherited hop rate and nBand onto one tensor pin.
 * @param pin Pin to update.
 * @param incoming Upstream shape.
 * @param copyChannels When true, also rewrite @c pin.shape.channels.
 */
void inheritTensorFields(openyourbox::graph::Pin &pin,
                         const openyourbox::graph::ShapeSignature &incoming,
                         bool copyChannels = false) {
  pin.shape.temporalRate = incoming.temporalRate;
  pin.shape.nBand = incoming.nBand;
  if (copyChannels)
    pin.shape.channels = incoming.channels;
}

/**
 * @brief Writes Conv1D detail text from stride.
 * @param node Convolution node to update.
 */
void updateConv1dDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::isConvolutionType;
  if (!isConvolutionType(node.type))
    return;
  const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
  if (stride <= 1)
    node.detail = "Temporal convolution";
  else
    node.detail = "Downsample x" + std::to_string(stride);
}

/**
 * @brief Writes ConvTranspose1d detail text from stride.
 * @param node Transposed convolution node to update.
 */
void updateConvTransposeDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  if (node.type != NodeType::convTranspose)
    return;
  const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
  if (stride <= 1)
    node.detail = "Temporal upsampling";
  else
    node.detail = "Upsample x" + std::to_string(stride);
}

/**
 * @brief Returns the declared host I/O mode on an Audio Input/Output node.
 * @param node Host boundary node.
 */
openyourbox::graph::HostIoMode
readHostIoMode(const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::HostIoMode;
  using openyourbox::graph::hostIoModeFromChannels;
  using openyourbox::graph::hostIoModeFromChoice;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return HostIoMode::stereo;
  for (const auto &property : node.properties) {
    if (property.key != "channels")
      continue;
    if (property.kind == openyourbox::graph::PropertyKind::choice) {
      const auto legacyPair =
          property.maximum <= 1 || property.choices.size() == 2;
      return hostIoModeFromChoice(property.value, legacyPair);
    }
    return hostIoModeFromChannels(std::clamp(property.value, 1, 2));
  }
  for (const auto &pin : node.outputs) {
    if (pin.shape.channels == 1 || pin.shape.channels == 2)
      return hostIoModeFromChannels(pin.shape.channels);
  }
  for (const auto &pin : node.inputs) {
    if (pin.shape.channels == 1 || pin.shape.channels == 2)
      return hostIoModeFromChannels(pin.shape.channels);
  }
  return HostIoMode::stereo;
}

/**
 * @brief Returns the mono/stereo width declared on a host I/O node.
 * @param node Audio Input or Audio Output node.
 */
int readHostIoChannels(const openyourbox::graph::GraphNode &node) {
  return openyourbox::graph::hostIoChannelsFromMode(readHostIoMode(node));
}

/**
 * @brief Writes host I/O detail text from the Channels choice.
 * @param node Audio Input or Audio Output node.
 */
void updateHostIoDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::hostIoModeDetail;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  node.detail =
      hostIoModeDetail(readHostIoMode(node), node.type == NodeType::audioInput);
}

/**
 * @brief Applies the Channels property to host I/O pin shapes and detail.
 * @param node Audio Input or Audio Output node.
 */
void applyHostIoChannels(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  const auto channels = readHostIoChannels(node);
  for (auto &pin : node.inputs) {
    pin.shape.channels = channels;
    pin.shape.temporalRate = 1;
    pin.shape.nBand = 0;
  }
  for (auto &pin : node.outputs) {
    pin.shape.channels = channels;
    pin.shape.temporalRate = 1;
    pin.shape.nBand = 0;
  }
  updateHostIoDetail(node);
}

/**
 * @brief Ensures Audio Input/Output expose Mono|Mirrored|Stereo Channels.
 * @param node Host boundary node to normalize.
 */
void normalizeHostIoProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::HostIoMode;
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::hostIoChoiceFromMode;
  using openyourbox::graph::hostIoModeFromChannels;
  using openyourbox::graph::hostIoModeFromChoice;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  const auto inferred = readHostIoMode(node);
  for (auto &property : node.properties) {
    if (property.key != "channels")
      continue;
    const auto wasChoice = property.kind == PropertyKind::choice;
    const auto legacyPair =
        wasChoice && (property.maximum <= 1 || property.choices.size() == 2);
    HostIoMode mode = inferred;
    if (wasChoice)
      mode = hostIoModeFromChoice(property.value, legacyPair);
    else if (property.value == 1 || property.value == 2)
      mode = hostIoModeFromChannels(property.value);
    property.label = "Mode";
    property.kind = PropertyKind::choice;
    property.choices = {"Mono", "Mirrored", "Stereo"};
    property.minimum = 0;
    property.maximum = 2;
    property.setValue(hostIoChoiceFromMode(mode));
    applyHostIoChannels(node);
    return;
  }
  NodeProperty channels;
  channels.key = "channels";
  channels.label = "Mode";
  channels.kind = PropertyKind::choice;
  channels.choices = {"Mono", "Mirrored", "Stereo"};
  channels.minimum = 0;
  channels.maximum = 2;
  channels.setValue(hostIoChoiceFromMode(inferred));
  node.properties.push_back(std::move(channels));
  applyHostIoChannels(node);
}

/**
 * @brief Migrates legacy Rate Conv / Direction upsample nodes and ensures
 *        Conv1D stride properties exist.
 * @param node Loaded or newly created convolution node.
 */
void storeFloatVector(juce::ValueTree &tree, const char *key,
                      const std::vector<float> &values) {
  if (values.empty())
    return;
  juce::MemoryBlock block(values.data(), values.size() * sizeof(float));
  tree.setProperty(key, block.toBase64Encoding(), nullptr);
}

void loadFloatVector(const juce::ValueTree &tree, const char *key,
                     std::vector<float> &values) {
  if (!tree.hasProperty(key))
    return;
  juce::MemoryBlock block;
  if (!block.fromBase64Encoding(tree[key].toString()))
    return;
  const auto count = block.getSize() / sizeof(float);
  values.resize(count);
  if (count > 0)
    std::memcpy(values.data(), block.getData(), count * sizeof(float));
}

void copyTensorToVector(const torch::Tensor &tensor, std::vector<float> &out) {
  if (!tensor.defined() || tensor.numel() < 1) {
    out.clear();
    return;
  }
  auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
  const auto *data = cpu.data_ptr<float>();
  out.assign(data, data + cpu.numel());
}

void copyCompactnessFromArtifact(openyourbox::graph::GraphNode &node,
                                 const std::string &path) {
  using openyourbox::graph::clearCompactness;
  if (path.empty()) {
    clearCompactness(node);
    return;
  }
  try {
    auto module = torch::jit::load(path, torch::kCPU);
    module.eval();
    bool ready = false;
    if (module.hasattr("compactness_ready"))
      ready = module.attr("compactness_ready").toTensor().item<float>() > 0.5f;
    node.compactnessReady = ready;
    if (!ready) {
      node.latentMean.clear();
      node.latentPca.clear();
      node.cumulativeVariance.clear();
      return;
    }
    if (module.hasattr("latent_mean"))
      copyTensorToVector(module.attr("latent_mean").toTensor(), node.latentMean);
    if (module.hasattr("latent_pca"))
      copyTensorToVector(module.attr("latent_pca").toTensor(), node.latentPca);
    if (module.hasattr("cumulative_variance"))
      copyTensorToVector(module.attr("cumulative_variance").toTensor(),
                         node.cumulativeVariance);
  } catch (const std::exception &) {
    clearCompactness(node);
  }
}

void normalizeBottleneckProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultBottleneckKernelSize;
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  if (node.type != NodeType::variationalBottleneck)
    return;
  for (auto &property : node.properties) {
    if (property.key == "kernel_size") {
      property.minimum = minimumPositiveProperty;
      property.maximum = unlimitedPropertyMaximum;
      if (property.value < 1)
        property.value = defaultBottleneckKernelSize;
      return;
    }
  }
  NodeProperty kernel;
  kernel.key = "kernel_size";
  kernel.label = "Kernel Size";
  kernel.value = defaultBottleneckKernelSize;
  kernel.minimum = minimumPositiveProperty;
  kernel.maximum = unlimitedPropertyMaximum;
  node.properties.push_back(std::move(kernel));
}

/**
 * @brief Ensures Noise Synth has no weights and exposes bands plus window size.
 * @param node Loaded or newly created Noise Synth node.
 */
void normalizeNoiseSynthesizerProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultNoiseBands;
  using openyourbox::graph::defaultNoiseWindowSize;
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  if (node.type != NodeType::noiseSynthesizer)
    return;
  node.hasWeights = false;
  node.armedForTraining = false;
  node.detail = "IR × white noise";
  bool hasBands = false;
  bool hasWindow = false;
  for (auto &property : node.properties) {
    if (property.key == "noise_bands") {
      hasBands = true;
      property.minimum = 2;
      property.maximum = unlimitedPropertyMaximum;
      if (property.value < 2)
        property.value = 2;
    } else if (property.key == "window_size") {
      hasWindow = true;
      property.minimum = minimumPositiveProperty;
      property.maximum = unlimitedPropertyMaximum;
      if (property.value < 1)
        property.value = defaultNoiseWindowSize;
    }
  }
  if (!hasBands) {
    NodeProperty bands;
    bands.key = "noise_bands";
    bands.label = "Noise bands";
    bands.value = defaultNoiseBands;
    bands.minimum = 2;
    bands.maximum = unlimitedPropertyMaximum;
    node.properties.push_back(std::move(bands));
  }
  if (!hasWindow) {
    NodeProperty window;
    window.key = "window_size";
    window.label = "Window Size";
    window.value = defaultNoiseWindowSize;
    window.minimum = minimumPositiveProperty;
    window.maximum = unlimitedPropertyMaximum;
    node.properties.push_back(std::move(window));
  }
}

/**
 * @brief Removes persisted upper bounds on numeric node properties.
 * @param node Loaded or newly created graph node.
 */
void normalizePropertyBounds(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::widenIntegerPropertyBounds;
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
}

void normalizeConvolutionProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isShapePassthroughType;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  using openyourbox::graph::widenIntegerPropertyBounds;
  if (node.type == NodeType::rateConv) {
    node.type = NodeType::convolution;
    if (node.label == "Rate Conv")
      node.label = "Conv1D";
  }
  if (node.type == NodeType::convolution) {
    int direction = 0;
    for (const auto &property : node.properties) {
      if (property.key == "direction")
        direction = property.value;
    }
    if (direction != 0) {
      node.type = NodeType::convTranspose;
      node.label = "ConvTranspose1d";
    }
    node.properties.erase(
        std::remove_if(node.properties.begin(), node.properties.end(),
                       [](const NodeProperty &property) {
                         return property.key == "direction";
                       }),
        node.properties.end());
  }
  auto resetFlexiblePin = [&](openyourbox::graph::Pin &pin) {
    if (isControlInputPin(pin))
      return;
    if (pin.shape.temporalRate == 1 && pin.shape.nBand == 0) {
      const auto channels = pin.shape.channels;
      pin.shape = flexibleTensorShape(channels);
    }
  };
  if (isShapePassthroughType(node.type) ||
      node.type == NodeType::variationalBottleneck ||
      node.type == NodeType::batchNorm) {
    for (auto &pin : node.inputs)
      resetFlexiblePin(pin);
    for (auto &pin : node.outputs)
      resetFlexiblePin(pin);
  }
  if (node.type == NodeType::convTranspose) {
    for (auto &pin : node.inputs)
      resetFlexiblePin(pin);
    for (auto &pin : node.outputs)
      resetFlexiblePin(pin);
    auto ensureInt = [&node](const char *key, const char *label, int value,
                             int minimum) {
      for (auto &property : node.properties) {
        if (property.key == key) {
          property.minimum = minimum;
          property.maximum = unlimitedPropertyMaximum;
          return;
        }
      }
      node.properties.push_back(
          NodeProperty{key, label, value, minimum, unlimitedPropertyMaximum});
    };
    ensureInt("channels", "Channels", 2, minimumPositiveProperty);
    ensureInt("kernel_size", "Kernel Size", 3, minimumPositiveProperty);
    ensureInt("dilation", "Dilation", 1, minimumPositiveProperty);
    ensureInt("stride", "Stride", 1, minimumPositiveProperty);
    {
      bool hasWeightNorm = false;
      for (auto &property : node.properties) {
        if (property.key == "weight_norm") {
          property.minimum = 0;
          property.maximum = 1;
          hasWeightNorm = true;
          break;
        }
      }
      if (!hasWeightNorm)
        node.properties.push_back(
            NodeProperty{"weight_norm", "Weight Norm", 0, 0, 1});
    }
    for (auto &property : node.properties)
      widenIntegerPropertyBounds(property);
    updateConvTransposeDetail(node);
    return;
  }
  if (node.type != NodeType::convolution)
    return;

  auto ensureInt = [&node](const char *key, const char *label, int value,
                           int minimum,
                           PropertyKind kind = PropertyKind::integer,
                           std::vector<std::string> choices = {}) {
    for (auto &property : node.properties) {
      if (property.key == key) {
        property.minimum = minimum;
        property.maximum = unlimitedPropertyMaximum;
        return;
      }
    }
    node.properties.push_back(NodeProperty{key, label, value, minimum,
                                           unlimitedPropertyMaximum, kind,
                                           std::move(choices)});
  };
  ensureInt("channels", "Channels", 2, minimumPositiveProperty);
  ensureInt("kernel_size", "Kernel Size", 3, minimumPositiveProperty);
  ensureInt("dilation", "Dilation", 1, minimumPositiveProperty);
  ensureInt("stride", "Stride", 1, minimumPositiveProperty);
  {
    bool hasWeightNorm = false;
    for (auto &property : node.properties) {
      if (property.key == "weight_norm") {
        property.minimum = 0;
        property.maximum = 1;
        hasWeightNorm = true;
        break;
      }
    }
    if (!hasWeightNorm)
      node.properties.push_back(
          NodeProperty{"weight_norm", "Weight Norm", 0, 0, 1});
  }
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
  updateConv1dDetail(node);
}

void applyNodePinShapes(openyourbox::graph::NodeGraph &graph,
                        openyourbox::graph::GraphNode &node);

/**
 * @brief Refreshes inherited hop rate and nBand on every node (first repeat only).
 * @param graph Graph document to mutate.
 */
void refreshPropagatedPinShapesCore(openyourbox::graph::NodeGraph &graph) {
  std::unordered_map<std::int32_t, int> indegree;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  for (const auto &node : graph.getNodes())
    indegree[node.id] = 0;
  for (const auto &link : graph.getLinks()) {
    const auto source = graph.findNodeForPin(link.sourcePinId);
    const auto destination = graph.findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value() ||
        *source == *destination)
      continue;
    outgoing[*source].push_back(*destination);
    ++indegree[*destination];
  }
  std::queue<std::int32_t> ready;
  std::unordered_set<std::int32_t> visited;
  for (const auto &entry : indegree) {
    if (entry.second == 0)
      ready.push(entry.first);
  }
  auto apply = [&graph](std::int32_t nodeId) {
    if (auto *node = graph.findNode(nodeId))
      applyNodePinShapes(graph, *node);
  };
  while (!ready.empty()) {
    const auto nodeId = ready.front();
    ready.pop();
    if (!visited.insert(nodeId).second)
      continue;
    apply(nodeId);
    for (const auto next : outgoing[nodeId]) {
      if (--indegree[next] == 0)
        ready.push(next);
    }
  }
  for (auto &node : graph.getNodes()) {
    if (visited.count(node.id) == 0)
      applyNodePinShapes(graph, node);
  }
}

/**
 * @brief Fills @c Pin::repeatShapes from an unrolled serial stack.
 * @param graph Editable graph whose first-repeat shapes are already current.
 */
void refreshOutputRepeatShapes(openyourbox::graph::NodeGraph &graph) {
  // Materialize restores via ValueTree, which refreshes shapes again. Skip the
  // nested repeat-shape pass so we only unroll once per outer refresh.
  static thread_local int depth = 0;
  if (depth > 0)
    return;
  struct DepthGuard {
    int &value;
    explicit DepthGuard(int &depthValue) : value(depthValue) { ++value; }
    ~DepthGuard() { --value; }
  } guard(depth);

  for (auto &node : graph.getNodes()) {
    for (auto &pin : node.outputs)
      pin.repeatShapes.clear();
    for (auto &pin : node.inputs)
      pin.repeatShapes.clear();
  }
  bool anyMultiRepeat = false;
  for (const auto &group : graph.getGroups()) {
    if (graph.groupRepeatStatus(group.id).effectiveRepeats > 1) {
      anyMultiRepeat = true;
      break;
    }
  }
  if (!anyMultiRepeat)
    return;

  std::unordered_map<std::int32_t, std::pair<std::int32_t, int>> provenance;
  auto expanded = graph.withInvisibleRepeatsMaterialized(&provenance);
  refreshPropagatedPinShapesCore(expanded);

  std::unordered_map<std::int64_t, const openyourbox::graph::GraphNode *>
      nodeAtSlot;
  const auto slotKey = [](std::int32_t originalId, int slot) -> std::int64_t {
    return (static_cast<std::int64_t>(originalId) << 32) |
           static_cast<std::uint32_t>(slot);
  };
  for (const auto &entry : provenance) {
    const auto *expandedNode = expanded.findNode(entry.first);
    if (expandedNode == nullptr)
      continue;
    nodeAtSlot[slotKey(entry.second.first, entry.second.second)] = expandedNode;
  }

  for (auto &node : graph.getNodes()) {
    const auto repeats = graph.effectiveRuntimeRepeatCount(node.id);
    if (repeats <= 1)
      continue;
    for (std::size_t pinIndex = 0; pinIndex < node.outputs.size(); ++pinIndex) {
      auto &pin = node.outputs[pinIndex];
      pin.repeatShapes.assign(static_cast<std::size_t>(repeats), pin.shape);
      for (int slot = 0; slot < repeats; ++slot) {
        const auto found = nodeAtSlot.find(slotKey(node.id, slot));
        if (found == nodeAtSlot.end() || found->second == nullptr)
          continue;
        if (pinIndex >= found->second->outputs.size())
          continue;
        pin.repeatShapes[static_cast<std::size_t>(slot)] =
            found->second->outputs[pinIndex].shape;
      }
    }
    for (std::size_t pinIndex = 0; pinIndex < node.inputs.size(); ++pinIndex) {
      auto &pin = node.inputs[pinIndex];
      pin.repeatShapes.assign(static_cast<std::size_t>(repeats), pin.shape);
      for (int slot = 0; slot < repeats; ++slot) {
        const auto found = nodeAtSlot.find(slotKey(node.id, slot));
        if (found == nodeAtSlot.end() || found->second == nullptr)
          continue;
        if (pinIndex >= found->second->inputs.size())
          continue;
        pin.repeatShapes[static_cast<std::size_t>(slot)] =
            found->second->inputs[pinIndex].shape;
      }
    }
  }
  // Hubs are flattened out of the materialized graph, so repeat their
  // per-slot shapes from the connected member pins instead. Nested group
  // hubs must be filled before a parent hub copies from them, otherwise the
  // parent keeps first-repeat hops (RAVE upsample last-out becomes bottleneck
  // hop and cannot reach PQMF synthesis).
  std::vector<openyourbox::graph::GraphNode *> hubNodes;
  for (auto &node : graph.getNodes()) {
    if (node.type == openyourbox::graph::NodeType::groupOutput ||
        node.type == openyourbox::graph::NodeType::groupInput)
      hubNodes.push_back(&node);
  }
  std::sort(hubNodes.begin(), hubNodes.end(),
            [&graph](const openyourbox::graph::GraphNode *left,
                     const openyourbox::graph::GraphNode *right) {
              const auto depthOf =
                  [&graph](const openyourbox::graph::GraphNode *node) {
                    if (node == nullptr || !node->parentGroupId.has_value())
                      return 0;
                    return groupDepth(graph, *node->parentGroupId);
                  };
              return depthOf(left) > depthOf(right);
            });
  for (auto *node : hubNodes) {
    if (node == nullptr)
      continue;
    if (node->type == openyourbox::graph::NodeType::groupOutput) {
      const auto laneCount = std::min(node->inputs.size(), node->outputs.size());
      for (std::size_t index = 0; index < laneCount; ++index) {
        for (const auto &link : graph.getLinks()) {
          if (link.destinationPinId != node->inputs[index].id)
            continue;
          const auto *source = graph.findPin(link.sourcePinId);
          if (source != nullptr && !source->repeatShapes.empty()) {
            const auto hubRepeats =
                std::max(1, graph.effectiveRuntimeRepeatCount(node->id));
            const auto n = static_cast<int>(source->repeatShapes.size());
            int innerFold = 1;
            if (n >= hubRepeats && n % hubRepeats == 0)
              innerFold = n / hubRepeats;
            node->outputs[index].repeatShapes =
                openyourbox::graph::foldInnerRepeatShapes(source->repeatShapes,
                                                          innerFold, true);
          }
          break;
        }
      }
      applyGroupOutputExitShapes(graph, *node);
    } else if (node->type == openyourbox::graph::NodeType::groupInput) {
      const auto laneCount = std::min(node->inputs.size(), node->outputs.size());
      for (std::size_t index = 0; index < laneCount; ++index) {
        for (const auto &link : graph.getLinks()) {
          if (link.sourcePinId != node->outputs[index].id)
            continue;
          const auto *destination = graph.findPin(link.destinationPinId);
          if (destination != nullptr && !destination->repeatShapes.empty()) {
            const auto hubRepeats =
                std::max(1, graph.effectiveRuntimeRepeatCount(node->id));
            const auto n = static_cast<int>(destination->repeatShapes.size());
            int innerFold = 1;
            if (n >= hubRepeats && n % hubRepeats == 0)
              innerFold = n / hubRepeats;
            const auto folded = openyourbox::graph::foldInnerRepeatShapes(
                destination->repeatShapes, innerFold, false);
            node->inputs[index].repeatShapes = folded;
            node->outputs[index].repeatShapes = folded;
          }
          break;
        }
      }
    }
  }
}

/**
 * @brief Refreshes first-repeat pin shapes and per-repeat shape lists.
 * @param graph Graph document to mutate.
 */
void refreshPropagatedPinShapes(openyourbox::graph::NodeGraph &graph) {
  refreshPropagatedPinShapesCore(graph);
  refreshOutputRepeatShapes(graph);
  // Repeat-shape unroll fills Group Output last-out lists after the first
  // pass, so consumers on the parent canvas inherit again from those exits.
  refreshPropagatedPinShapesCore(graph);
}

/**
 * @brief Splits amplitude channels into IR bins and upsamples by window size.
 * @param node Noise Synth element.
 * @param incoming Connected conditioner shape.
 * @return Output shape, or channels/rate 0 when the split or hop is illegal.
 */
openyourbox::graph::ShapeSignature
noiseSynthOutgoingShape(const openyourbox::graph::GraphNode &node,
                        openyourbox::graph::ShapeSignature incoming) {
  using openyourbox::graph::convolutionOutputTemporalRate;
  using openyourbox::graph::defaultNoiseBands;
  using openyourbox::graph::defaultNoiseWindowSize;
  const auto noiseBands =
      std::max(1, readNodeProperty(node, "noise_bands", defaultNoiseBands));
  const auto windowSize =
      std::max(1, readNodeProperty(node, "window_size", defaultNoiseWindowSize));
  auto outgoing = incoming;
  if (incoming.channels > 0) {
    outgoing.channels = incoming.channels % noiseBands == 0
                            ? incoming.channels / noiseBands
                            : 0;
  }
  if (incoming.temporalRate > 0) {
    const auto rate =
        convolutionOutputTemporalRate(incoming.temporalRate, windowSize, true);
    outgoing.temporalRate = rate < 0 ? 0 : rate;
  }
  return outgoing;
}

/**
 * @brief Applies hop-rate and nBand rules for one node after its producers.
 * @param graph Graph document used to resolve upstream cables.
 * @param node Node whose pins are rewritten.
 */
void applyNodePinShapes(openyourbox::graph::NodeGraph &graph,
                        openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::convolutionOutputTemporalRate;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::defaultHiddenSize;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isConvolutionType;
  using openyourbox::graph::isConvTransposeType;
  using openyourbox::graph::isShapePassthroughType;
  const auto incoming = firstConnectedTensorSource(graph, node);
  auto inheritInputs = [&]() {
    for (auto &pin : node.inputs) {
      if (isControlInputPin(pin) || openyourbox::graph::isIrInputPin(pin))
        continue;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id))
        inheritTensorFields(pin, outgoingShapeOf(graph, *connected), true);
      else
        pin.shape = flexibleTensorShape();
    }
  };

  switch (node.type) {
  case NodeType::groupInput: {
    inheritInputs();
    const auto laneCount = std::min(node.inputs.size(), node.outputs.size());
    for (std::size_t index = 0; index < laneCount; ++index)
      node.outputs[index].shape = node.inputs[index].shape;
    break;
  }
  case NodeType::groupOutput: {
    inheritInputs();
    applyGroupOutputExitShapes(graph, node);
    break;
  }
  case NodeType::pqmfAnalysis: {
    const auto nBand =
        std::max(2, readNodeProperty(node, "n_band", defaultPqmfBands));
    int audioChannels = 0;
    for (auto &pin : node.inputs) {
      pin.shape.temporalRate = 1;
      pin.shape.nBand = 0;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id)) {
        const auto incoming = outgoingShapeOf(graph, *connected);
        pin.shape.channels = incoming.channels;
        if (incoming.channels > 0)
          audioChannels = incoming.channels;
      } else {
        pin.shape.channels = 0;
      }
    }
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = nBand;
      pin.shape.nBand = nBand;
      pin.shape.channels =
          audioChannels > 0 ? audioChannels * nBand : 0;
    }
    break;
  }
  case NodeType::pqmfSynthesis: {
    const auto nBand =
        std::max(2, readNodeProperty(node, "n_band", defaultPqmfBands));
    int bandChannels = 0;
    for (auto &pin : node.inputs) {
      pin.shape.temporalRate = nBand;
      pin.shape.nBand = nBand;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id)) {
        const auto incoming = outgoingShapeOf(graph, *connected);
        pin.shape.channels = incoming.channels;
        if (incoming.channels > 0)
          bandChannels = incoming.channels;
      } else {
        pin.shape.channels = 0;
      }
    }
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = 1;
      pin.shape.nBand = 0;
      pin.shape.channels =
          bandChannels > 0 ? std::max(1, bandChannels / nBand) : 0;
    }
    break;
  }
  case NodeType::variationalBottleneck:
    inheritInputs();
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = incoming.temporalRate;
      pin.shape.nBand = incoming.nBand;
      const auto latent = readNodeProperty(node, "latent_size", defaultLatentSize);
      if (latent > 0)
        pin.shape.channels = latent;
    }
    break;
  case NodeType::noiseSynthesizer:
    inheritInputs();
    {
      const auto outgoing = noiseSynthOutgoingShape(node, incoming);
      for (auto &pin : node.outputs)
        inheritTensorFields(pin, outgoing, true);
    }
    break;
  case NodeType::lstm:
  case NodeType::rnn:
    inheritInputs();
    {
      auto outgoing = incoming;
      const auto hidden =
          std::max(1, readNodeProperty(node, "hidden_size", defaultHiddenSize));
      const auto bidirectional = readNodeProperty(node, "bidirectional", 0) != 0;
      outgoing.channels = bidirectional ? hidden * 2 : hidden;
      for (auto &pin : node.outputs)
        inheritTensorFields(pin, outgoing, true);
    }
    break;
  default: {
    if (!isShapePassthroughType(node.type) && !isConvolutionType(node.type) &&
        !isConvTransposeType(node.type))
      break;
    inheritInputs();
    auto outgoing = incoming;
    for (auto &property : node.properties) {
      if (!property.preserveInBound)
        continue;
      if ((property.key == "features" || property.key == "channels") &&
          incoming.channels > 0)
        property.value = incoming.channels;
    }
    if (isConvolutionType(node.type)) {
      const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
      const auto rate =
          convolutionOutputTemporalRate(incoming.temporalRate, stride, false);
      outgoing.temporalRate = rate < 0 ? 0 : rate;
      const auto channels =
          readBoundShapeProperty(node, "channels", incoming.channels, 0);
      if (channels > 0)
        outgoing.channels = channels;
      else if (propertyRepeatListIsInvalid(node, "channels"))
        outgoing.channels = 0;
      updateConv1dDetail(node);
    } else if (isConvTransposeType(node.type)) {
      const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
      const auto rate =
          convolutionOutputTemporalRate(incoming.temporalRate, stride, true);
      outgoing.temporalRate = rate < 0 ? 0 : rate;
      const auto channels =
          readBoundShapeProperty(node, "channels", incoming.channels, 0);
      if (channels > 0)
        outgoing.channels = channels;
      else if (propertyRepeatListIsInvalid(node, "channels"))
        outgoing.channels = 0;
      updateConvTransposeDetail(node);
    } else if (node.type == NodeType::linear) {
      const auto features =
          readBoundShapeProperty(node, "features", incoming.channels, 0);
      if (features > 0)
        outgoing.channels = features;
      else if (propertyRepeatListIsInvalid(node, "features"))
        outgoing.channels = 0;
    } else if (node.type == NodeType::merge ||
               openyourbox::graph::isMathExpressionType(node.type)) {
      updateMergeOutputShape(graph, node);
      outgoing.channels =
          node.outputs.empty() ? 0 : node.outputs.front().shape.channels;
    }
    for (auto &pin : node.outputs)
      inheritTensorFields(pin, outgoing, true);
    break;
  }
  }
}

/**
 * @brief Returns a tooltip when any committed cable is now illegal.
 * @param graph Graph document to inspect.
 */
std::string firstIncompatibleLinkMessage(const openyourbox::graph::NodeGraph &graph) {
  using openyourbox::graph::convolutionRateIsError;
  using openyourbox::graph::convolutionRateMessage;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::isConvolutionType;
  using openyourbox::graph::isConvTransposeType;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::pqmfSynthesisChannelIsError;
  using openyourbox::graph::pqmfSynthesisChannelMessage;
  using openyourbox::graph::variationalBottleneckChannelIsError;
  using openyourbox::graph::variationalBottleneckChannelMessage;
  for (const auto &node : graph.getNodes()) {
    for (const auto &property : node.properties) {
      if (property.repeatListInvalid && !property.repeatListInvalidMessage.empty())
        return property.repeatListInvalidMessage;
    }
    if (openyourbox::graph::isMathExpressionType(node.type)) {
      if (const auto message = utilityInputsIncompatibilityMessage(graph, node);
          !message.empty())
        return message;
    }
  }
  for (const auto &link : graph.getLinks()) {
    const auto *source = graph.findPin(link.sourcePinId);
    const auto *destination = graph.findPin(link.destinationPinId);
    if (source == nullptr || destination == nullptr)
      continue;
    const auto sourceShape = outgoingShapeOf(graph, *source);
    if (!sourceShape.isCompatibleWith(destination->shape)) {
      auto message = sourceShape.incompatibilityMessage(destination->shape);
      if (message.empty())
        message = "Shape mismatch: channel counts are incompatible";
      return message;
    }
    const auto destNode = graph.findNodeForPin(destination->id);
    if (!destNode.has_value())
      continue;
    const auto *node = graph.findNode(*destNode);
    if (node == nullptr)
      continue;
    if (openyourbox::graph::isExternalLoadNode(*node) &&
        node->externalShapeIncomplete && !openyourbox::graph::isLatentPin(*destination) &&
        !openyourbox::graph::isControlInputPin(*destination))
      return "Enter channel overrides before connecting this checkpoint";
    if (node->type == NodeType::pqmfSynthesis &&
        pqmfSynthesisChannelIsError(sourceShape.channels,
                                    std::max(2, readNodeProperty(
                                                    *node, "n_band",
                                                    defaultPqmfBands))))
      return pqmfSynthesisChannelMessage(
          sourceShape.channels,
          std::max(2, readNodeProperty(*node, "n_band", defaultPqmfBands)));
    if (node->type == NodeType::variationalBottleneck &&
        variationalBottleneckChannelIsError(sourceShape.channels))
      return variationalBottleneckChannelMessage(sourceShape.channels);
    if (node->type == NodeType::noiseSynthesizer) {
      const auto noiseBands = std::max(
          1, readNodeProperty(*node, "noise_bands",
                              openyourbox::graph::defaultNoiseBands));
      const auto windowSize = std::max(
          1, readNodeProperty(*node, "window_size",
                              openyourbox::graph::defaultNoiseWindowSize));
      if (openyourbox::graph::noiseSynthChannelIsError(sourceShape.channels,
                                                       noiseBands))
        return openyourbox::graph::noiseSynthChannelMessage(
            sourceShape.channels, noiseBands);
      if (openyourbox::graph::noiseSynthWindowIsError(windowSize, noiseBands))
        return openyourbox::graph::noiseSynthWindowMessage(windowSize,
                                                           noiseBands);
      if (convolutionRateIsError(windowSize, true, sourceShape.temporalRate))
        return convolutionRateMessage(windowSize, true,
                                      sourceShape.temporalRate);
      continue;
    }
    if (!isConvolutionType(node->type) && !isConvTransposeType(node->type))
      continue;
    const auto stride = std::max(1, readNodeProperty(*node, "stride", 1));
    const auto upsample = isConvTransposeType(node->type);
    if (convolutionRateIsError(stride, upsample, sourceShape.temporalRate))
      return convolutionRateMessage(stride, upsample,
                                    sourceShape.temporalRate);
  }
  return {};
}

juce::Colour colourForType(openyourbox::graph::NodeType type,
                           openyourbox::graph::NodeState state) {
  return openyourbox::graph::chromeColourForType(type, state);
}

const char *nodeTypeName(openyourbox::graph::NodeType type) noexcept {
  using openyourbox::graph::NodeType;
  switch (type) {
  case NodeType::audioInput:
    return "audio_input";
  case NodeType::audioOutput:
    return "audio_output";
  case NodeType::linear:
    return "linear";
  case NodeType::convolution:
    return "conv1d";
  case NodeType::activation:
    return "activation";
  case NodeType::tcn:
    return "tcn";
  case NodeType::merge:
    return "utility";
  case NodeType::blackBox:
    return "blackbox";
  case NodeType::knobInput:
    return "knob_input";
  case NodeType::xyTrackpad:
    return "xy_trackpad";
  case NodeType::pqmfAnalysis:
    return "pqmf_analysis";
  case NodeType::pqmfSynthesis:
    return "pqmf_synthesis";
  case NodeType::rateConv:
    return "rate_conv";
  case NodeType::variationalBottleneck:
    return "variational_bottleneck";
  case NodeType::noiseSynthesizer:
    return "noise_synthesizer";
  case NodeType::convTranspose:
    return "conv_transpose1d";
  case NodeType::batchNorm:
    return "batch_norm";
  case NodeType::mathExpression:
    return "math_expression";
  case NodeType::reverb:
    return "reverb";
  case NodeType::expDecayReverb:
    return "exp_decay_reverb";
  case NodeType::filteredNoiseReverb:
    return "filtered_noise_reverb";
  case NodeType::firFilter:
    return "fir_filter";
  case NodeType::modDelay:
    return "mod_delay";
  case NodeType::lstm:
    return "lstm";
  case NodeType::rnn:
    return "rnn";
  case NodeType::groupInput:
    return "group_input";
  case NodeType::groupOutput:
    return "group_output";
  }
  return "tcn";
}

openyourbox::graph::NodeType nodeTypeFromName(const juce::String &name) {
  using openyourbox::graph::NodeType;
  if (name == "audio_input")
    return NodeType::audioInput;
  if (name == "audio_output")
    return NodeType::audioOutput;
  if (name == "linear")
    return NodeType::linear;
  if (name == "activation")
    return NodeType::activation;
  if (name == "sum" || name == "multiply" || name == "concatenate" ||
      name == "merge" || name == "utility")
    return NodeType::merge;
  if (name == "blackbox")
    return NodeType::blackBox;
  if (name == "knob_input")
    return NodeType::knobInput;
  if (name == "xy_trackpad")
    return NodeType::xyTrackpad;
  if (name == "pqmf_analysis")
    return NodeType::pqmfAnalysis;
  if (name == "pqmf_synthesis")
    return NodeType::pqmfSynthesis;
  if (name == "rate_conv" || name == "conv1d")
    return NodeType::convolution;
  if (name == "conv_transpose1d")
    return NodeType::convTranspose;
  if (name == "batch_norm")
    return NodeType::batchNorm;
  if (name == "variational_bottleneck")
    return NodeType::variationalBottleneck;
  if (name == "noise_synthesizer")
    return NodeType::noiseSynthesizer;
  if (name == "math_expression")
    return NodeType::mathExpression;
  if (name == "reverb")
    return NodeType::reverb;
  if (name == "exp_decay_reverb")
    return NodeType::expDecayReverb;
  if (name == "filtered_noise_reverb")
    return NodeType::filteredNoiseReverb;
  if (name == "fir_filter")
    return NodeType::firFilter;
  if (name == "mod_delay")
    return NodeType::modDelay;
  if (name == "lstm")
    return NodeType::lstm;
  if (name == "rnn")
    return NodeType::rnn;
  if (name == "group_input")
    return NodeType::groupInput;
  if (name == "group_output")
    return NodeType::groupOutput;
  return NodeType::tcn;
}

/**
 * @brief Returns true when @p name is a persisted element type this build knows.
 * @param name ValueTree `type` token.
 */
bool isKnownPersistedNodeType(const juce::String &name) {
  static const std::array<const char *, 31> known{
      "audio_input",
      "audio_output",
      "linear",
      "activation",
      "sum",
      "multiply",
      "concatenate",
      "merge",
      "utility",
      "blackbox",
      "knob_input",
      "xy_trackpad",
      "pqmf_analysis",
      "pqmf_synthesis",
      "rate_conv",
      "conv1d",
      "conv_transpose1d",
      "batch_norm",
      "variational_bottleneck",
      "noise_synthesizer",
      "math_expression",
      "reverb",
      "exp_decay_reverb",
      "filtered_noise_reverb",
      "fir_filter",
      "mod_delay",
      "lstm",
      "rnn",
      "group_input",
      "group_output",
      "tcn"};
  for (const auto *token : known) {
    if (name == token)
      return true;
  }
  return false;
}

/**
 * @brief Reads @p name, falling back to a pre-rename property if needed.
 * @param tree Serialized node, group, or property.
 * @param name Current property identifier.
 * @param legacy Previous identifier used before the copies→repeats rename.
 * @param fallback Value when neither identifier is present.
 */
juce::var propertyOrLegacy(const juce::ValueTree &tree, const char *name,
                           const char *legacy, const juce::var &fallback = {}) {
  if (tree.hasProperty(name))
    return tree.getProperty(name);
  if (tree.hasProperty(legacy))
    return tree.getProperty(legacy);
  return fallback;
}

juce::ValueTree nodeToTree(const openyourbox::graph::GraphNode &node) {
  juce::ValueTree tree{"Node"};
  tree.setProperty("id", node.id, nullptr);
  tree.setProperty("label", juce::String(node.label), nullptr);
  tree.setProperty("detail", juce::String(node.detail), nullptr);
  tree.setProperty("type", nodeTypeName(node.type), nullptr);
  tree.setProperty("state",
                   node.state == openyourbox::graph::NodeState::frozenGold
                       ? "frozen_gold"
                       : "live_blue",
                   nullptr);
  tree.setProperty("x", node.position.x, nullptr);
  tree.setProperty("y", node.position.y, nullptr);
  tree.setProperty("width", node.size.x, nullptr);
  tree.setProperty("height", node.size.y, nullptr);
  tree.setProperty("hasWeights", node.hasWeights, nullptr);
  tree.setProperty("seed", node.seed, nullptr);
  tree.setProperty("explicitSeed", node.explicitSeed, nullptr);
  tree.setProperty("useExplicitSeed", node.useExplicitSeed, nullptr);
  tree.setProperty("artifactPath", juce::String(node.artifactPath), nullptr);
  tree.setProperty("sourceSubgraph", juce::String(node.sourceSubgraph),
                   nullptr);
  tree.setProperty("analysisView", static_cast<int>(node.selectedAnalysisView),
                   nullptr);
  tree.setProperty("conditioningValue", node.conditioningValue, nullptr);
  tree.setProperty("conditioningX", node.conditioningX, nullptr);
  tree.setProperty("conditioningY", node.conditioningY, nullptr);
  tree.setProperty("armedForTraining", node.armedForTraining, nullptr);
  tree.setProperty("residual", node.residual, nullptr);
  tree.setProperty("dilationGrowth", node.dilationGrowth, nullptr);
  tree.setProperty("weightsProvenance",
                   node.weightsProvenance ==
                           openyourbox::graph::WeightsProvenance::file
                       ? "file"
                       : "random",
                   nullptr);
  tree.setProperty("weightsPath", juce::String(node.weightsPath), nullptr);
  tree.setProperty("blackBoxOrigin",
                   openyourbox::graph::blackBoxOriginName(node.blackBoxOrigin),
                   nullptr);
  tree.setProperty("externalLoadStatus",
                   openyourbox::graph::externalLoadStatusName(
                       node.externalLoadStatus),
                   nullptr);
  tree.setProperty("externalLoadErrorMessage",
                   juce::String(node.externalLoadErrorMessage), nullptr);
  tree.setProperty("inferredInputChannels", node.inferredInputChannels,
                   nullptr);
  tree.setProperty("inferredOutputChannels", node.inferredOutputChannels,
                   nullptr);
  tree.setProperty("inferredLatentChannels", node.inferredLatentChannels,
                   nullptr);
  tree.setProperty("overrideInputChannels", node.overrideInputChannels,
                   nullptr);
  tree.setProperty("overrideOutputChannels", node.overrideOutputChannels,
                   nullptr);
  tree.setProperty("overrideLatentChannels", node.overrideLatentChannels,
                   nullptr);
  tree.setProperty("externalHasEncodeDecode", node.externalHasEncodeDecode,
                   nullptr);
  tree.setProperty("externalAcceptsConditioning",
                   node.externalAcceptsConditioning, nullptr);
  tree.setProperty("externalShapeIncomplete", node.externalShapeIncomplete,
                   nullptr);
  tree.setProperty("sampleRateWarning", juce::String(node.sampleRateWarning),
                   nullptr);
  tree.setProperty("fidelityPercent", node.fidelityPercent, nullptr);
  tree.setProperty("compactnessReady", node.compactnessReady, nullptr);
  storeFloatVector(tree, "latentMean", node.latentMean);
  storeFloatVector(tree, "latentPca", node.latentPca);
  storeFloatVector(tree, "cumulativeVariance", node.cumulativeVariance);
  if (node.parentGroupId.has_value())
    tree.setProperty("parentGroupId", *node.parentGroupId, nullptr);

  for (const auto &slot : node.repeatSlots) {
    juce::ValueTree child{"RepeatSlot"};
    child.setProperty("seed", slot.seed, nullptr);
    child.setProperty("weightsProvenance",
                      slot.provenance ==
                              openyourbox::graph::WeightsProvenance::file
                          ? "file"
                          : "random",
                      nullptr);
    child.setProperty("weightsPath", juce::String(slot.weightsPath), nullptr);
    child.setProperty("artifactPath", juce::String(slot.artifactPath), nullptr);
    tree.appendChild(child, nullptr);
  }

  if (node.metrics.has_value()) {
    tree.setProperty("compileMs", node.metrics->compileTimeMilliseconds,
                     nullptr);
    tree.setProperty("inferenceMs", node.metrics->inferenceTimeMilliseconds,
                     nullptr);
  }

  const auto appendPins =
      [&tree](const std::vector<openyourbox::graph::Pin> &pins,
              const char *kind) {
        for (const auto &pin : pins) {
          juce::ValueTree child{"Pin"};
          child.setProperty("id", pin.id, nullptr);
          child.setProperty("label", juce::String(pin.label), nullptr);
          child.setProperty("kind", kind, nullptr);
          child.setProperty("channels", pin.shape.channels, nullptr);
          child.setProperty("temporalRate", pin.shape.temporalRate, nullptr);
          child.setProperty("nBand", pin.shape.nBand, nullptr);
          tree.appendChild(child, nullptr);
        }
      };
  appendPins(node.inputs, "input");
  appendPins(node.outputs, "output");

  for (const auto &property : node.properties) {
    juce::ValueTree child{"Property"};
    child.setProperty("key", juce::String(property.key), nullptr);
    child.setProperty("label", juce::String(property.label), nullptr);
    child.setProperty("value", property.value, nullptr);
    child.setProperty("minimum", property.minimum, nullptr);
    child.setProperty("maximum", property.maximum, nullptr);
    child.setProperty("kind", static_cast<int>(property.kind), nullptr);
    juce::StringArray choices;
    for (const auto &choice : property.choices)
      choices.add(choice);
    child.setProperty("choices", choices.joinIntoString("|"), nullptr);
    child.setProperty("floatValue", property.floatValue, nullptr);
    child.setProperty("floatMinimum", property.floatMinimum, nullptr);
    child.setProperty("floatMaximum", property.floatMaximum, nullptr);
    if (!property.stringValue.empty())
      child.setProperty("stringValue", juce::String(property.stringValue),
                        nullptr);
    if (!property.authoredTokens.empty()) {
      juce::StringArray tokens;
      for (const auto &token : property.authoredTokens)
        tokens.add(token);
      child.setProperty("authoredTokens", tokens.joinIntoString("\n"), nullptr);
    }
    child.setProperty("preserveIn", property.preserveInBound, nullptr);
    child.setProperty("repeatListInvalid", property.repeatListInvalid, nullptr);
    if (!property.repeatListInvalidMessage.empty())
      child.setProperty("repeatListInvalidMessage",
                        juce::String(property.repeatListInvalidMessage), nullptr);
    if (property.preserveInBound) {
      juce::StringArray tokens;
      const auto count = std::max(1, static_cast<int>(property.repeatIntValues.size()));
      for (int index = 0; index < count; ++index)
        tokens.add(openyourbox::graph::preserveInToken);
      child.setProperty("repeatIntValues", tokens.joinIntoString(","), nullptr);
    } else if (!property.repeatIntValues.empty()) {
      juce::StringArray ints;
      for (const auto value : property.repeatIntValues)
        ints.add(juce::String(value));
      child.setProperty("repeatIntValues", ints.joinIntoString(","), nullptr);
    }
    if (!property.repeatFloatValues.empty()) {
      juce::StringArray reals;
      for (const auto value : property.repeatFloatValues)
        reals.add(juce::String(value, 6));
      child.setProperty("repeatFloatValues", reals.joinIntoString(","), nullptr);
    }
    tree.appendChild(child, nullptr);
  }
  return tree;
}

openyourbox::graph::GraphNode nodeFromTree(const juce::ValueTree &tree) {
  using namespace openyourbox::graph;
  GraphNode node;
  node.id = static_cast<std::int32_t>(tree["id"]);
  node.label = tree["label"].toString().toStdString();
  node.detail = tree["detail"].toString().toStdString();
  node.type = nodeTypeFromName(tree["type"].toString());
  node.state = tree["state"].toString() == "frozen_gold" ? NodeState::frozenGold
                                                         : NodeState::liveBlue;
  node.colour = colourForType(node.type, node.state);
  node.position = {static_cast<float>(tree["x"]),
                   static_cast<float>(tree["y"])};
  node.size = {static_cast<float>(tree.getProperty("width", 180.0f)),
               static_cast<float>(tree.getProperty("height", 120.0f))};
  node.hasWeights = static_cast<bool>(tree["hasWeights"]);
  node.seed = openyourbox::graph::clampSeed(
      static_cast<std::int32_t>(tree.getProperty("seed", 42)));
  node.explicitSeed = openyourbox::graph::clampSeed(static_cast<std::int32_t>(
      tree.getProperty("explicitSeed", node.seed)));
  node.useExplicitSeed =
      static_cast<bool>(tree.getProperty("useExplicitSeed", false));
  node.artifactPath = tree["artifactPath"].toString().toStdString();
  node.sourceSubgraph = tree["sourceSubgraph"].toString().toStdString();
  node.selectedAnalysisView = static_cast<AnalysisView>(std::clamp(
      static_cast<int>(tree.getProperty("analysisView", 0)), 0, 3));
  node.conditioningValue = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningValue", 0.0)));
  node.conditioningX = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningX", 0.0)));
  node.conditioningY = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningY", 0.0)));
  node.armedForTraining =
      static_cast<bool>(tree.getProperty("armedForTraining", true));
  node.residual = static_cast<bool>(tree.getProperty("residual", false));
  node.dilationGrowth = std::max(
      static_cast<int>(tree.getProperty("dilationGrowth", defaultDilationGrowth)),
      minimumDilationGrowth);
  node.weightsProvenance =
      tree.getProperty("weightsProvenance", "random").toString() == "file"
          ? WeightsProvenance::file
          : WeightsProvenance::random;
  node.weightsPath = tree["weightsPath"].toString().toStdString();
  node.blackBoxOrigin = openyourbox::graph::blackBoxOriginFromName(
      tree.getProperty("blackBoxOrigin", "manual_freeze").toString());
  node.externalLoadStatus = openyourbox::graph::externalLoadStatusFromName(
      tree.getProperty("externalLoadStatus", "empty").toString());
  node.externalLoadErrorMessage =
      tree.getProperty("externalLoadErrorMessage", "").toString().toStdString();
  node.inferredInputChannels =
      static_cast<int>(tree.getProperty("inferredInputChannels", 0));
  node.inferredOutputChannels =
      static_cast<int>(tree.getProperty("inferredOutputChannels", 0));
  node.inferredLatentChannels =
      static_cast<int>(tree.getProperty("inferredLatentChannels", 0));
  node.overrideInputChannels =
      static_cast<int>(tree.getProperty("overrideInputChannels", -1));
  node.overrideOutputChannels =
      static_cast<int>(tree.getProperty("overrideOutputChannels", -1));
  node.overrideLatentChannels =
      static_cast<int>(tree.getProperty("overrideLatentChannels", -1));
  node.externalHasEncodeDecode =
      static_cast<bool>(tree.getProperty("externalHasEncodeDecode", false));
  node.externalAcceptsConditioning =
      static_cast<bool>(tree.getProperty("externalAcceptsConditioning", false));
  node.externalShapeIncomplete =
      static_cast<bool>(tree.getProperty("externalShapeIncomplete", false));
  node.sampleRateWarning =
      tree.getProperty("sampleRateWarning", "").toString().toStdString();
  node.fidelityPercent = openyourbox::graph::clampFidelity(static_cast<float>(
      tree.getProperty("fidelityPercent", defaultFidelityPercent)));
  node.compactnessReady =
      static_cast<bool>(tree.getProperty("compactnessReady", false));
  loadFloatVector(tree, "latentMean", node.latentMean);
  loadFloatVector(tree, "latentPca", node.latentPca);
  loadFloatVector(tree, "cumulativeVariance", node.cumulativeVariance);
  if (tree.hasProperty("parentGroupId"))
    node.parentGroupId =
        static_cast<std::int32_t>(tree.getProperty("parentGroupId"));

  if (tree.hasProperty("inferenceMs")) {
    node.metrics =
        NodeMetrics{static_cast<double>(tree.getProperty("compileMs", 0.0)),
                    static_cast<double>(tree.getProperty("inferenceMs", 0.0))};
  }

  for (const auto child : tree) {
    if (child.hasType("Pin")) {
      Pin pin;
      pin.id = static_cast<std::int32_t>(child["id"]);
      pin.label = child["label"].toString().toStdString();
      pin.kind = child["kind"].toString() == "output" ? PinKind::output
                                                      : PinKind::input;
      pin.shape.channels = static_cast<int>(child["channels"]);
      pin.shape.temporalRate =
          static_cast<int>(child.getProperty("temporalRate", 1));
      pin.shape.nBand = static_cast<int>(child.getProperty("nBand", 0));
      (pin.kind == PinKind::input ? node.inputs : node.outputs)
          .push_back(std::move(pin));
    } else if (child.hasType("RepeatSlot") || child.hasType("CopySlot")) {
      RepeatWeightSlot slot;
      slot.seed = openyourbox::graph::clampSeed(
          static_cast<std::int32_t>(child.getProperty("seed", 42)));
      slot.provenance =
          child.getProperty("weightsProvenance", "random").toString() == "file"
              ? WeightsProvenance::file
              : WeightsProvenance::random;
      slot.weightsPath = child["weightsPath"].toString().toStdString();
      slot.artifactPath = child["artifactPath"].toString().toStdString();
      node.repeatSlots.push_back(std::move(slot));
    } else if (child.hasType("Property")) {
      NodeProperty property;
      property.key = child["key"].toString().toStdString();
      property.label = child["label"].toString().toStdString();
      property.value = static_cast<int>(child["value"]);
      property.minimum = static_cast<int>(child["minimum"]);
      property.maximum = static_cast<int>(child["maximum"]);
      property.kind =
          static_cast<PropertyKind>(static_cast<int>(child["kind"]));
      const auto choices =
          juce::StringArray::fromTokens(child["choices"].toString(), "|", "");
      for (const auto &choice : choices)
        property.choices.push_back(choice.toStdString());
      property.floatValue =
          static_cast<float>(child.getProperty("floatValue", 0.0));
      property.floatMinimum =
          static_cast<float>(child.getProperty("floatMinimum", 0.0));
      property.floatMaximum =
          static_cast<float>(child.getProperty("floatMaximum", 1.0));
      property.stringValue =
          child.getProperty("stringValue", "").toString().toStdString();
      if (child.hasProperty("authoredTokens")) {
        const auto tokens = juce::StringArray::fromTokens(
            child["authoredTokens"].toString(), "\n", "");
        for (const auto &token : tokens)
          property.authoredTokens.push_back(token.toStdString());
      }
      property.preserveInBound = static_cast<bool>(child.getProperty("preserveIn", false));
      property.repeatListInvalid =
          static_cast<bool>(propertyOrLegacy(child, "repeatListInvalid",
                                             "copyListInvalid", false));
      property.repeatListInvalidMessage =
          propertyOrLegacy(child, "repeatListInvalidMessage",
                           "copyListInvalidMessage", "")
              .toString()
              .toStdString();
      if (child.hasProperty("repeatIntValues") ||
          child.hasProperty("copyIntValues")) {
        const auto tokens = juce::StringArray::fromTokens(
            propertyOrLegacy(child, "repeatIntValues", "copyIntValues")
                .toString(),
            ",", "");
        bool anyIn = false;
        bool allIn = true;
        for (const auto &token : tokens) {
          const auto trimmed = token.trim();
          if (trimmed == openyourbox::graph::preserveInToken) {
            anyIn = true;
            property.repeatIntValues.push_back(0);
          } else {
            allIn = false;
            property.repeatIntValues.push_back(trimmed.getIntValue());
          }
        }
        if (anyIn && allIn)
          property.preserveInBound = true;
      }
      if (child.hasProperty("repeatFloatValues") ||
          child.hasProperty("copyFloatValues")) {
        const auto tokens = juce::StringArray::fromTokens(
            propertyOrLegacy(child, "repeatFloatValues", "copyFloatValues")
                .toString(),
            ",", "");
        for (const auto &token : tokens)
          property.repeatFloatValues.push_back(token.getFloatValue());
      }
      node.properties.push_back(std::move(property));
    }
  }
  migrateLegacyMixerNode(node, tree["type"].toString());
  normalizeMergeNodeProperties(node);
  normalizeGainProperty(node);
  normalizeNegativeSlopeProperty(node);
  normalizeRecurrentProperties(node);
  normalizeConditioningPins(node);
  normalizePhase3Node(node);
  normalizeConvolutionProperties(node);
  normalizeBottleneckProperties(node);
  normalizeNoiseSynthesizerProperties(node);
  normalizePropertyBounds(node);
  normalizeHostIoProperties(node);
  return node;
}

/**
 * @brief Serializes one group container into a value tree.
 * @param group Group to persist.
 */
juce::ValueTree groupToTree(const openyourbox::graph::GraphGroup &group) {
  juce::ValueTree tree{"Group"};
  tree.setProperty("id", group.id, nullptr);
  tree.setProperty("name", juce::String(group.name), nullptr);
  if (group.parentGroupId.has_value())
    tree.setProperty("parentGroupId", *group.parentGroupId, nullptr);
  tree.setProperty("collapsed", group.collapsed, nullptr);
  tree.setProperty("repeats", group.repeats, nullptr);
  tree.setProperty("x", group.position.x, nullptr);
  tree.setProperty("y", group.position.y, nullptr);
  tree.setProperty("width", group.size.x, nullptr);
  tree.setProperty("height", group.size.y, nullptr);
  tree.setProperty("viewPanX", group.viewPan.x, nullptr);
  tree.setProperty("viewPanY", group.viewPan.y, nullptr);
  tree.setProperty("viewZoom", group.viewZoom, nullptr);
  for (const auto member : group.memberIds) {
    juce::ValueTree child{"Member"};
    child.setProperty("id", member, nullptr);
    tree.appendChild(child, nullptr);
  }
  return tree;
}

/**
 * @brief Restores one group container from a value tree.
 * @param tree Serialized group.
 */
openyourbox::graph::GraphGroup groupFromTree(const juce::ValueTree &tree) {
  openyourbox::graph::GraphGroup group;
  group.id = static_cast<std::int32_t>(tree["id"]);
  group.name = tree.getProperty("name", "Group").toString().toStdString();
  if (group.name.empty())
    group.name = "Group";
  if (tree.hasProperty("parentGroupId"))
    group.parentGroupId =
        static_cast<std::int32_t>(tree.getProperty("parentGroupId"));
  group.collapsed = static_cast<bool>(tree.getProperty("collapsed", false));
  group.repeats = std::max(
      1, static_cast<int>(propertyOrLegacy(
             tree, "repeats", "copies",
             openyourbox::graph::defaultGroupRepeats)));
  group.position = {static_cast<float>(tree.getProperty("x", 0.0f)),
                    static_cast<float>(tree.getProperty("y", 0.0f))};
  group.size = {static_cast<float>(tree.getProperty("width", 320.0f)),
                static_cast<float>(tree.getProperty("height", 220.0f))};
  group.viewPan = {static_cast<float>(tree.getProperty("viewPanX", 0.0f)),
                   static_cast<float>(tree.getProperty("viewPanY", 0.0f))};
  group.viewZoom = std::clamp(
      static_cast<float>(tree.getProperty("viewZoom", 1.0f)),
      openyourbox::graph::minimumZoom, openyourbox::graph::maximumZoom);
  for (const auto child : tree) {
    if (!child.hasType("Member"))
      continue;
    group.memberIds.push_back(static_cast<std::int32_t>(child["id"]));
  }
  return group;
}

/**
 * @brief Detaches @p memberId from every group's member list.
 * @param groups Group collection to mutate.
 * @param memberId Node or nested group identifier.
 */
void eraseMemberFromParents(std::vector<openyourbox::graph::GraphGroup> &groups,
                            std::int32_t memberId) {
  for (auto &group : groups) {
    group.memberIds.erase(std::remove(group.memberIds.begin(),
                                      group.memberIds.end(), memberId),
                          group.memberIds.end());
  }
}

/**
 * @brief Writes parentGroupId onto a node or nested group.
 * @param graph Graph to mutate.
 * @param memberId Node or nested group identifier.
 * @param parent New parent, or empty for the canvas root.
 */
void assignParent(openyourbox::graph::NodeGraph &graph, std::int32_t memberId,
                  std::optional<std::int32_t> parent) {
  if (auto *node = graph.findNode(memberId))
    node->parentGroupId = parent;
  else if (auto *group = graph.findGroup(memberId))
    group->parentGroupId = parent;
}

/**
 * @brief Resizes repeat slots for every leaf of @p groupId.
 * @param graph Graph to mutate.
 * @param groupId Group whose descendants need slot updates.
 */
void refreshRepeatSlotsForGroup(openyourbox::graph::NodeGraph &graph,
                              std::int32_t groupId) {
  for (const auto nodeId : graph.collectLeafNodeIds(groupId)) {
    if (auto *node = graph.findNode(nodeId)) {
      if (openyourbox::graph::isGroupBoundaryType(node->type))
        continue;
      const auto repeats = graph.effectiveRepeatCount(nodeId);
      openyourbox::graph::ensureRepeatSlotCount(*node, repeats);
      openyourbox::graph::ensureNodePropertyRepeatCounts(*node, repeats);
      graph.validateAuthoredRepeatLists(nodeId);
    }
  }
}

/**
 * @brief Writes a randomization seed onto a node and its repeat slots.
 * @param node Weighted live element to update.
 * @param primarySeed Seed applied to the visible element (slot 0).
 * @param independentExtraSlots True to derive `primarySeed + i` per repeat slot.
 */
void writeRandomizedWeightSlots(openyourbox::graph::GraphNode &node,
                                std::int32_t primarySeed,
                                bool independentExtraSlots) {
  using openyourbox::graph::WeightsProvenance;
  using openyourbox::graph::clampSeed;
  using openyourbox::graph::repeatSlotFromNode;
  using openyourbox::graph::seedForRepeatSlot;
  node.seed = clampSeed(primarySeed);
  node.weightsProvenance = WeightsProvenance::random;
  node.weightsPath.clear();
  node.artifactPath.clear();
  if (node.repeatSlots.empty())
    node.repeatSlots.push_back(repeatSlotFromNode(node));
  else
    node.repeatSlots.front() = repeatSlotFromNode(node);
  for (std::size_t index = 0; index < node.repeatSlots.size(); ++index) {
    auto &slot = node.repeatSlots[index];
    slot.seed = independentExtraSlots ? seedForRepeatSlot(node.seed, index)
                                      : node.seed;
    slot.provenance = WeightsProvenance::random;
    slot.weightsPath.clear();
    slot.artifactPath.clear();
  }
  if (node.type == openyourbox::graph::NodeType::variationalBottleneck ||
      node.type == openyourbox::graph::NodeType::blackBox)
    openyourbox::graph::clearCompactness(node);
}

/**
 * @brief Upgrades persisted mixer nodes to the unified Utility element.
 * @param node Loaded graph node to normalize in place.
 * @param storedType Serialized type name from the graph document.
 */
void migrateLegacyMixerNode(openyourbox::graph::GraphNode &node,
                            const juce::String &storedType) {
  using openyourbox::graph::MergeMode;
  using openyourbox::graph::NodeType;
  int legacyMode = -1;
  if (storedType == "sum")
    legacyMode = static_cast<int>(MergeMode::add);
  else if (storedType == "multiply")
    legacyMode = static_cast<int>(MergeMode::multiply);
  else if (storedType == "concatenate")
    legacyMode = static_cast<int>(MergeMode::concatenate);
  else
    return;

  node.type = NodeType::merge;
  if (node.label == "Sum" || node.label == "Multiply" ||
      node.label == "Concatenate" || node.label == "Merge")
    node.label = "Utility";

  for (auto &property : node.properties) {
    if (property.key == "mode") {
      property.value = legacyMode;
      return;
    }
  }

  openyourbox::graph::NodeProperty modeProperty;
  modeProperty.key = "mode";
  modeProperty.label = "Mode";
  modeProperty.value = legacyMode;
  modeProperty.minimum = 0;
  modeProperty.maximum = 2;
  modeProperty.kind = openyourbox::graph::PropertyKind::choice;
  modeProperty.choices = {"Add", "Multiply", "Concatenate"};
  node.properties.insert(node.properties.begin(), std::move(modeProperty));
}

/**
 * @brief Ensures persisted Utility nodes expose current labels, modes, and input bounds.
 * @param node Loaded or migrated Utility node.
 */
void normalizeMergeNodeProperties(openyourbox::graph::GraphNode &node) {
  if (node.type != openyourbox::graph::NodeType::merge)
    return;
  if (node.label == "Merge")
    node.label = "Utility";
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  for (auto &property : node.properties) {
    if (property.key == "mode") {
      property.maximum = 2;
      property.choices = {"Add", "Multiply", "Concatenate"};
      property.value =
          std::clamp(property.value, property.minimum, property.maximum);
    } else if (property.key == "inputs") {
      property.minimum = minimumPositiveProperty;
      property.maximum = unlimitedPropertyMaximum;
      property.value =
          std::max(property.value, minimumPositiveProperty);
    }
  }
}

juce::ValueTree linkToTree(const openyourbox::graph::GraphLink &link) {
  juce::ValueTree tree{"Link"};
  tree.setProperty("id", link.id, nullptr);
  tree.setProperty("sourcePin", link.sourcePinId, nullptr);
  tree.setProperty("destinationPin", link.destinationPinId, nullptr);
  return tree;
}

openyourbox::graph::GraphLink linkFromTree(const juce::ValueTree &tree) {
  return {static_cast<std::int32_t>(tree["id"]),
          static_cast<std::int32_t>(tree["sourcePin"]),
          static_cast<std::int32_t>(tree["destinationPin"])};
}

/**
 * @brief Result of simulating whether serial repeats can chain by shape.
 */
struct SerialRepeatShapeResult {
  /** @brief True when every repeat can feed the next. */
  bool ok = true;
  /** @brief User-facing reason when @ref ok is false. */
  std::string message;
  /** @brief Properties that can repair an internal join mismatch. */
  std::vector<openyourbox::graph::GroupRepeatPropertyHint> hints;
};

/**
 * @brief Repeat-slot index for a leaf relative to an enclosing group's slot.
 * @param graph Graph owning membership.
 * @param nodeId Leaf node.
 * @param groupId Group whose serial repeats are being simulated.
 * @param groupSlot Outer repeat index of @p groupId.
 * @return Slot passed to @c integerValueForRepeat.
 */
int leafRepeatSlotForGroup(const openyourbox::graph::NodeGraph &graph,
                         std::int32_t nodeId, std::int32_t groupId,
                         int groupSlot) {
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr)
    return groupSlot;
  int innerProduct = 1;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value() && *parent != groupId) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *ancestor = graph.findGroup(*parent);
    if (ancestor == nullptr)
      break;
    innerProduct *= std::max(1, ancestor->repeats);
    parent = ancestor->parentGroupId;
  }
  return groupSlot * innerProduct;
}

/**
 * @brief Simulated pin shape, falling back to the live first-repeat shape.
 * @param graph Graph owning pins.
 * @param pinId Endpoint to resolve.
 * @param pinShapes Shapes computed for the current repeat slot.
 */
openyourbox::graph::ShapeSignature simulatedPinShape(
    const openyourbox::graph::NodeGraph &graph, std::int32_t pinId,
    const std::unordered_map<std::int32_t, openyourbox::graph::ShapeSignature>
        &pinShapes) {
  using openyourbox::graph::flexibleTensorShape;
  const auto found = pinShapes.find(pinId);
  if (found != pinShapes.end())
    return found->second;
  if (const auto *pin = graph.findPin(pinId))
    return pin->shape;
  return flexibleTensorShape();
}

/**
 * @brief Integer used by repeat @p slot, honoring `in` and invalid lists.
 * @param node Owner of the property.
 * @param key Property key.
 * @param incomingChannels Paired input width for `in`.
 * @param slot Repeat index.
 * @param fallback Value when the property is absent.
 */
int boundIntForRepeatSlot(const openyourbox::graph::GraphNode &node,
                        const char *key, int incomingChannels, int slot,
                        int fallback) {
  using openyourbox::graph::integerValueForRepeat;
  for (const auto &property : node.properties) {
    if (property.key != key)
      continue;
    if (property.repeatListInvalid)
      return 0;
    if (property.preserveInBound)
      return incomingChannels > 0 ? incomingChannels : 0;
    const auto value = integerValueForRepeat(property, slot);
    return value > 0 ? value : fallback;
  }
  return fallback;
}

/**
 * @brief Stride used by repeat @p slot.
 * @param node Convolution or transposed-convolution node.
 * @param slot Repeat index.
 */
int strideForRepeatSlot(const openyourbox::graph::GraphNode &node, int slot) {
  using openyourbox::graph::integerValueForRepeat;
  for (const auto &property : node.properties) {
    if (property.key != "stride")
      continue;
    return std::max(1, integerValueForRepeat(property, slot));
  }
  return 1;
}

/**
 * @brief Records shape-driving properties upstream of @p startNodeId.
 * @param graph Graph to walk.
 * @param startNodeId Node whose producers are searched.
 * @param groupId Walk stops at this group's input hub.
 * @param message Hint text copied onto each candidate.
 * @param hints Destination list.
 */
void appendRepeatShapePropertyHints(
    const openyourbox::graph::NodeGraph &graph, std::int32_t startNodeId,
    std::int32_t groupId, const std::string &message,
    std::vector<openyourbox::graph::GroupRepeatPropertyHint> &hints) {
  using openyourbox::graph::GroupRepeatPropertyHint;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::propertySupportsPreserveIn;
  std::queue<std::int32_t> pending;
  std::unordered_set<std::int32_t> visited;
  pending.push(startNodeId);
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (!visited.insert(current).second)
      continue;
    const auto *node = graph.findNode(current);
    if (node == nullptr || node->type == NodeType::groupInput)
      continue;
    for (const auto &property : node->properties) {
      const bool channels = property.key == "channels" ||
                            property.key == "features" ||
                            property.key == "latent_size";
      if (!channels)
        continue;
      GroupRepeatPropertyHint hint;
      hint.nodeId = node->id;
      hint.propertyKey = property.key;
      hint.message = message;
      if (propertySupportsPreserveIn(property))
        hint.message += "; use 'in' to preserve the incoming width";
      hints.push_back(std::move(hint));
    }
    for (const auto &link : graph.getLinks()) {
      const auto destination = graph.findNodeForPin(link.destinationPinId);
      if (!destination.has_value() || *destination != current)
        continue;
      if (const auto source = graph.findNodeForPin(link.sourcePinId);
          source.has_value() && nodeIsInsideGroup(graph, *source, groupId))
        pending.push(*source);
    }
  }
}

/**
 * @brief Topological node ids owned by @p groupId, including nested leaves.
 * @param graph Graph owning membership and cables.
 * @param groupId Group whose interior is ordered.
 */
std::vector<std::int32_t>
topologicalNodesInsideGroup(const openyourbox::graph::NodeGraph &graph,
                            std::int32_t groupId) {
  std::unordered_set<std::int32_t> inside;
  for (const auto &node : graph.getNodes()) {
    if (nodeIsInsideGroup(graph, node.id, groupId))
      inside.insert(node.id);
  }
  std::unordered_map<std::int32_t, int> indegree;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  for (const auto id : inside)
    indegree[id] = 0;
  for (const auto &link : graph.getLinks()) {
    const auto source = graph.findNodeForPin(link.sourcePinId);
    const auto destination = graph.findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value() ||
        *source == *destination)
      continue;
    if (inside.count(*source) == 0 || inside.count(*destination) == 0)
      continue;
    outgoing[*source].push_back(*destination);
    ++indegree[*destination];
  }
  std::queue<std::int32_t> ready;
  for (const auto &entry : indegree) {
    if (entry.second == 0)
      ready.push(entry.first);
  }
  std::vector<std::int32_t> order;
  order.reserve(inside.size());
  while (!ready.empty()) {
    const auto id = ready.front();
    ready.pop();
    order.push_back(id);
    for (const auto next : outgoing[id]) {
      if (--indegree[next] == 0)
        ready.push(next);
    }
  }
  for (const auto id : inside) {
    if (std::find(order.begin(), order.end(), id) == order.end())
      order.push_back(id);
  }
  return order;
}

/**
 * @brief Writes simulated output shapes for one node at one repeat slot.
 * @param graph Graph owning cables.
 * @param node Node to evaluate.
 * @param leafSlot Repeat index for this node's properties.
 * @param groupId Enclosing group being chained (for merge diagnostics).
 * @param pinShapes In/out map of simulated pin shapes.
 * @param result Failure destination when a residual join cannot combine.
 */
void simulateNodeRepeatShapes(
    const openyourbox::graph::NodeGraph &graph,
    const openyourbox::graph::GraphNode &node, int leafSlot,
    std::int32_t groupId,
    std::unordered_map<std::int32_t, openyourbox::graph::ShapeSignature>
        &pinShapes,
    SerialRepeatShapeResult &result) {
  using openyourbox::graph::MergeMode;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::convolutionOutputTemporalRate;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::defaultHiddenSize;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::integerValueForRepeat;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isConvTransposeType;
  using openyourbox::graph::isConvolutionType;
  using openyourbox::graph::isShapePassthroughType;
  auto incoming = flexibleTensorShape();
  for (const auto &pin : node.inputs) {
    if (isControlInputPin(pin) || openyourbox::graph::isIrInputPin(pin))
      continue;
    if (const auto *source = findConnectedSourcePin(graph, pin.id)) {
      incoming = simulatedPinShape(graph, source->id, pinShapes);
      break;
    }
  }
  auto writeOutputs = [&](const openyourbox::graph::ShapeSignature &outgoing) {
    for (const auto &pin : node.outputs)
      pinShapes[pin.id] = outgoing;
  };
  auto inheritHub = [&]() {
    const auto laneCount = std::min(node.inputs.size(), node.outputs.size());
    for (std::size_t index = 0; index < laneCount; ++index) {
      openyourbox::graph::ShapeSignature shape = flexibleTensorShape();
      if (const auto *source =
              findConnectedSourcePin(graph, node.inputs[index].id))
        shape = simulatedPinShape(graph, source->id, pinShapes);
      pinShapes[node.inputs[index].id] = shape;
      pinShapes[node.outputs[index].id] = shape;
    }
  };

  switch (node.type) {
  case NodeType::groupInput:
  case NodeType::groupOutput:
    inheritHub();
    return;
  case NodeType::pqmfAnalysis: {
    int nBand = defaultPqmfBands;
    for (const auto &property : node.properties) {
      if (property.key == "n_band")
        nBand = std::max(2, integerValueForRepeat(property, leafSlot));
    }
    const auto audioChannels = std::max(0, incoming.channels);
    openyourbox::graph::ShapeSignature outgoing;
    outgoing.temporalRate = nBand;
    outgoing.nBand = nBand;
    outgoing.channels = audioChannels > 0 ? audioChannels * nBand : 0;
    writeOutputs(outgoing);
    return;
  }
  case NodeType::pqmfSynthesis: {
    int nBand = defaultPqmfBands;
    for (const auto &property : node.properties) {
      if (property.key == "n_band")
        nBand = std::max(2, integerValueForRepeat(property, leafSlot));
    }
    openyourbox::graph::ShapeSignature outgoing;
    outgoing.temporalRate = 1;
    outgoing.nBand = 0;
    outgoing.channels =
        incoming.channels > 0 ? std::max(1, incoming.channels / nBand) : 0;
    writeOutputs(outgoing);
    return;
  }
  case NodeType::variationalBottleneck: {
    auto outgoing = incoming;
    outgoing.channels =
        boundIntForRepeatSlot(node, "latent_size", incoming.channels, leafSlot,
                            defaultLatentSize);
    writeOutputs(outgoing);
    return;
  }
  case NodeType::noiseSynthesizer:
    writeOutputs(noiseSynthOutgoingShape(node, incoming));
    return;
  case NodeType::lstm:
  case NodeType::rnn: {
    auto outgoing = incoming;
    int hidden = defaultHiddenSize;
    int bidirectional = 0;
    for (const auto &property : node.properties) {
      if (property.key == "hidden_size")
        hidden = std::max(1, integerValueForRepeat(property, leafSlot));
      if (property.key == "bidirectional")
        bidirectional = integerValueForRepeat(property, leafSlot);
    }
    outgoing.channels = bidirectional != 0 ? hidden * 2 : hidden;
    writeOutputs(outgoing);
    return;
  }
  default:
    break;
  }

  if (!isShapePassthroughType(node.type) && !isConvolutionType(node.type) &&
      !isConvTransposeType(node.type)) {
    writeOutputs(incoming);
    return;
  }

  auto outgoing = incoming;
  if (node.type == NodeType::merge) {
    const auto concatenate =
        mergeModeFor(node) == static_cast<int>(MergeMode::concatenate);
    std::vector<openyourbox::graph::ShapeSignature> sources;
    for (const auto &pin : node.inputs) {
      if (const auto *source = findConnectedSourcePin(graph, pin.id))
        sources.push_back(simulatedPinShape(graph, source->id, pinShapes));
    }
    if (!concatenate) {
      for (std::size_t index = 1; index < sources.size(); ++index) {
        const auto left = sources[0];
        const auto right = sources[index];
        auto rateLeft = left;
        auto rateRight = right;
        rateLeft.channels = 0;
        rateRight.channels = 0;
        if (!channelsAreBroadcastCompatible(left.channels, right.channels) ||
            !rateLeft.isCompatibleWith(rateRight)) {
          result.ok = false;
          const auto leftLabel =
              left.displayLabel().empty() ? "unspecified" : left.displayLabel();
          const auto rightLabel = right.displayLabel().empty()
                                      ? "unspecified"
                                      : right.displayLabel();
          result.message = "Utility inputs cannot combine (" + leftLabel +
                           " vs " + rightLabel + ")";
          appendRepeatShapePropertyHints(graph, node.id, groupId, result.message,
                                       result.hints);
          return;
        }
      }
    }
    if (concatenate) {
      outgoing.channels = 0;
      for (const auto &source : sources)
        outgoing.channels += source.channels;
    } else if (!sources.empty()) {
      outgoing = sources.front();
      for (const auto &source : sources)
        outgoing.channels = std::max(outgoing.channels, source.channels);
    }
    writeOutputs(outgoing);
    return;
  }

  if (isConvolutionType(node.type) || isConvTransposeType(node.type)) {
    const auto stride = strideForRepeatSlot(node, leafSlot);
    const auto upsample = isConvTransposeType(node.type);
    const auto rate =
        convolutionOutputTemporalRate(incoming.temporalRate, stride, upsample);
    outgoing.temporalRate = rate < 0 ? 0 : rate;
    const auto channels =
        boundIntForRepeatSlot(node, "channels", incoming.channels, leafSlot, 0);
    if (channels > 0)
      outgoing.channels = channels;
    writeOutputs(outgoing);
    return;
  }
  if (node.type == NodeType::linear) {
    const auto features =
        boundIntForRepeatSlot(node, "features", incoming.channels, leafSlot, 0);
    if (features > 0)
      outgoing.channels = features;
    writeOutputs(outgoing);
    return;
  }
  writeOutputs(outgoing);
}

/**
 * @brief Simulates N serial repeats using per-repeat properties, not first-repeat I/O.
 * @param graph Graph whose first-repeat pins supply the initial incoming shapes.
 * @param groupId Group whose repeats are requested.
 * @param repeats Requested repeat count N.
 * @param inputs Declared Group Input lanes.
 * @param outputs Declared Group Output lanes.
 * @return Failure when an internal join cannot combine at any repeat.
 */
SerialRepeatShapeResult evaluateSerialRepeatShapeChain(
    const openyourbox::graph::NodeGraph &graph, std::int32_t groupId, int repeats,
    const std::vector<openyourbox::graph::GroupBoundaryPort> &inputs,
    const std::vector<openyourbox::graph::GroupBoundaryPort> &outputs) {
  SerialRepeatShapeResult result;
  if (inputs.empty() || outputs.empty() || repeats <= 1)
    return result;
  const auto *inputHub = graph.findNode(inputs.front().memberNodeId);
  const auto *outputHub = graph.findNode(outputs.front().memberNodeId);
  if (inputHub == nullptr || outputHub == nullptr)
    return result;
  std::vector<openyourbox::graph::ShapeSignature> laneIn;
  laneIn.reserve(inputs.size());
  for (const auto &port : inputs)
    laneIn.push_back(port.shape);
  const auto order = topologicalNodesInsideGroup(graph, groupId);
  for (int slot = 0; slot < repeats; ++slot) {
    std::unordered_map<std::int32_t, openyourbox::graph::ShapeSignature>
        pinShapes;
    for (std::size_t index = 0; index < laneIn.size(); ++index) {
      if (index < inputHub->outputs.size())
        pinShapes[inputHub->outputs[index].id] = laneIn[index];
      if (index < inputHub->inputs.size())
        pinShapes[inputHub->inputs[index].id] = laneIn[index];
    }
    for (const auto nodeId : order) {
      if (nodeId == inputHub->id)
        continue;
      const auto *node = graph.findNode(nodeId);
      if (node == nullptr)
        continue;
      const auto leafSlot =
          leafRepeatSlotForGroup(graph, nodeId, groupId, slot);
      simulateNodeRepeatShapes(graph, *node, leafSlot, groupId, pinShapes,
                             result);
      if (!result.ok) {
        result.message = "Requested " + std::to_string(repeats) +
                         " repeats are inactive: " + result.message;
        return result;
      }
    }
    for (std::size_t index = 0; index < laneIn.size() &&
                                index < outputHub->outputs.size();
         ++index)
      laneIn[index] =
          simulatedPinShape(graph, outputHub->outputs[index].id, pinShapes);
  }
  return result;
}
} // namespace

namespace openyourbox::graph {
void NodeGraph::rebuildFromModel(const dsp::TCNConfiguration &configuration) {
  nodes.clear();
  links.clear();
  nextNodeId = 1;
  nextPinId = 1001;
  nextLinkId = 2001;

  const auto inputId = addNode(NodeType::audioInput, {24.0f, 130.0f});
  const auto tcnId = addNode(NodeType::tcn, {250.0f, 90.0f});
  const auto outputId = addNode(NodeType::audioOutput, {500.0f, 130.0f});
  auto *tcn = findNode(tcnId);
  if (tcn != nullptr) {
    setProperty(tcnId, "depth", configuration.depth);
    setProperty(tcnId, "kernel_size", configuration.kernelSize);
    setProperty(tcnId, "channels", configuration.channels);
    setProperty(tcnId, "activation",
                static_cast<int>(configuration.activation));
    tcn->detail = std::to_string(configuration.channels) + " ch, RF model";
  }

  auto *input = findNode(inputId);
  auto *output = findNode(outputId);
  tcn = findNode(tcnId);
  if (input != nullptr && tcn != nullptr && output != nullptr) {
    setProperty(inputId, "channels",
                hostIoChoiceFromChannels(configuration.inputChannels));
    setProperty(outputId, "channels",
                hostIoChoiceFromChannels(configuration.outputChannels));
    tcn->inputs.front().shape.channels = configuration.inputChannels;
    tcn->outputs.front().shape.channels = configuration.outputChannels;
    connect(input->outputs.front().id, tcn->inputs.front().id);
    connect(tcn->outputs.front().id, output->inputs.front().id);
  }
  ensureFixedHostIo();
}

std::int32_t NodeGraph::addNode(NodeType type, juce::Point<float> position,
                                std::optional<std::int32_t> parentGroupId) {
  return insertConstructedNode(makeNode(type, position), parentGroupId);
}

std::int32_t NodeGraph::insertConstructedNode(
    GraphNode node, std::optional<std::int32_t> parentGroupId) {
  if (isFixedIoType(node.type))
    parentGroupId.reset();
  if (parentGroupId.has_value() && findGroup(*parentGroupId) == nullptr)
    parentGroupId.reset();
  node.parentGroupId = parentGroupId;
  const auto id = node.id;
  nodes.push_back(std::move(node));
  if (parentGroupId.has_value()) {
    if (auto *group = findGroup(*parentGroupId)) {
      if (std::find(group->memberIds.begin(), group->memberIds.end(), id) ==
          group->memberIds.end())
        group->memberIds.push_back(id);
    }
    if (auto *created = findNode(id);
        created != nullptr && !isGroupBoundaryType(created->type)) {
      const auto repeats = effectiveRepeatCount(id);
      ensureRepeatSlotCount(*created, repeats);
      ensureNodePropertyRepeatCounts(*created, repeats);
    }
  }
  return id;
}

std::int32_t NodeGraph::addExternalTorchScriptLoadNode(
    juce::Point<float> position, std::optional<std::int32_t> parentGroupId) {
  return insertConstructedNode(makeExternalTorchScriptLoadNode(position),
                               parentGroupId);
}

void NodeGraph::ensureFixedHostIo() {
  GraphNode *input = nullptr;
  GraphNode *output = nullptr;
  for (auto &node : nodes) {
    if (node.type == NodeType::audioInput && input == nullptr)
      input = &node;
    else if (node.type == NodeType::audioOutput && output == nullptr)
      output = &node;
  }
  if (input == nullptr)
    addNode(NodeType::audioInput, {24.0f, 130.0f});
  if (output == nullptr)
    addNode(NodeType::audioOutput, {500.0f, 130.0f});
  for (auto &node : nodes) {
    if (!isFixedIoType(node.type))
      continue;
    normalizeHostIoProperties(node);
    node.colour = colourForType(node.type, node.state);
  }
}

std::optional<std::int32_t>
NodeGraph::insertNodeOnLink(std::int32_t linkId, NodeType type,
                            juce::Point<float> position) {
  if (isFixedIoType(type) || isGroupBoundaryType(type) ||
      type == NodeType::blackBox ||
      isConditioningSourceType(type))
    return std::nullopt;
  const auto *link = findLink(linkId);
  if (link == nullptr)
    return std::nullopt;
  const auto sourcePinId = link->sourcePinId;
  const auto destinationPinId = link->destinationPinId;
  if (!removeLink(linkId))
    return std::nullopt;

  const auto sourceOwner = findNodeForPin(sourcePinId);
  const auto destOwner = findNodeForPin(destinationPinId);
  std::optional<std::int32_t> parent;
  const auto *sourceNode =
      sourceOwner.has_value() ? findNode(*sourceOwner) : nullptr;
  const auto *destNode =
      destOwner.has_value() ? findNode(*destOwner) : nullptr;
  if (sourceNode != nullptr)
    parent = sourceNode->parentGroupId;
  if (destNode != nullptr && parent != destNode->parentGroupId)
    parent = viewport.focusedGroupId;

  const auto nodeId = addNode(type, position, parent);
  auto *node = findNode(nodeId);
  if (node == nullptr || node->inputs.empty() || node->outputs.empty()) {
    connect(sourcePinId, destinationPinId);
    if (nodeId != 0)
      removeNode(nodeId);
    return std::nullopt;
  }
  const auto inputResult = connect(sourcePinId, node->inputs.front().id);
  const auto outputResult = connect(node->outputs.front().id, destinationPinId);
  if (!inputResult.accepted || !outputResult.accepted) {
    removeNode(nodeId);
    connect(sourcePinId, destinationPinId);
    return std::nullopt;
  }
  return nodeId;
}

std::optional<std::int32_t>
NodeGraph::attachNodeToPin(std::int32_t pinId, NodeType type,
                           juce::Point<float> position) {
  if (isFixedIoType(type) || isGroupBoundaryType(type) ||
      type == NodeType::blackBox)
    return std::nullopt;
  const auto resolvedPinId = resolveCollapsedPin(pinId);
  const auto *pin = findPin(resolvedPinId);
  if (pin == nullptr || isPinConnected(resolvedPinId))
    return std::nullopt;

  std::optional<std::int32_t> parent = viewport.focusedGroupId;
  if (!isCollapsedGroupPin(pinId)) {
    if (const auto ownerId = findNodeForPin(resolvedPinId);
        ownerId.has_value()) {
      if (const auto *owner = findNode(*ownerId))
        parent = owner->parentGroupId;
    }
  }

  const auto nodeId = addNode(type, position, parent);
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return std::nullopt;
  if (pin->kind == PinKind::output) {
    if (node->inputs.empty()) {
      removeNode(nodeId);
      return std::nullopt;
    }
  } else if (node->outputs.empty()) {
    removeNode(nodeId);
    return std::nullopt;
  }

  const auto result =
      pin->kind == PinKind::output
          ? connect(resolvedPinId, node->inputs.front().id)
          : connect(node->outputs.front().id, resolvedPinId);
  if (!result.accepted) {
    removeNode(nodeId);
    return std::nullopt;
  }
  return nodeId;
}

bool NodeGraph::removeNode(std::int32_t nodeId) {
  return removeBoxes({nodeId});
}

bool NodeGraph::removeBoxes(const std::vector<std::int32_t> &boxIds) {
  std::unordered_set<std::int32_t> nodeIds;
  std::unordered_set<std::int32_t> groupIds;
  for (const auto boxId : boxIds) {
    if (const auto *node = findNode(boxId)) {
      if (isFixedIoType(node->type) || isGroupBoundaryType(node->type))
        continue;
      nodeIds.insert(boxId);
    } else if (findGroup(boxId) != nullptr) {
      collectGroupSubtree(*this, boxId, nodeIds, groupIds);
    }
  }
  if (nodeIds.empty() && groupIds.empty())
    return false;

  bool cutFedFromHostInput = false;
  bool cutFedHostOutput = false;
  for (const auto &link : links) {
    const auto sourceId = findNodeForPin(link.sourcePinId);
    const auto destId = findNodeForPin(link.destinationPinId);
    if (!sourceId.has_value() || !destId.has_value())
      continue;
    const auto sourceInCut = nodeIds.count(*sourceId) != 0;
    const auto destInCut = nodeIds.count(*destId) != 0;
    const auto *source = findNode(*sourceId);
    const auto *destination = findNode(*destId);
    if (!sourceInCut && destInCut && source != nullptr &&
        source->type == NodeType::audioInput)
      cutFedFromHostInput = true;
    if (sourceInCut && !destInCut && destination != nullptr &&
        destination->type == NodeType::audioOutput)
      cutFedHostOutput = true;
  }

  const auto bypasses = collectBypassPinPairs(*this, nodeIds);

  std::unordered_set<std::int32_t> pins;
  for (const auto nodeId : nodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs)
      pins.insert(pin.id);
    for (const auto &pin : node->outputs)
      pins.insert(pin.id);
  }
  links.erase(std::remove_if(links.begin(), links.end(),
                             [&pins](const GraphLink &link) {
                               return pins.count(link.sourcePinId) != 0 ||
                                      pins.count(link.destinationPinId) != 0;
                             }),
              links.end());
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [&nodeIds](const GraphNode &candidate) {
                               return nodeIds.count(candidate.id) != 0;
                             }),
              nodes.end());
  for (const auto nodeId : nodeIds)
    eraseMemberFromParents(groups, nodeId);
  for (const auto groupId : groupIds)
    eraseMemberFromParents(groups, groupId);
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [&groupIds](const GraphGroup &candidate) {
                                return groupIds.count(candidate.id) != 0;
                              }),
               groups.end());
  for (const auto id : groupIds)
    pruneStickySpineId(viewport.stickySpine, id);
  if (viewport.focusedGroupId.has_value() &&
      groupIds.count(*viewport.focusedGroupId) != 0)
    viewport.focusedGroupId.reset();

  refreshPropagatedPinShapes(*this);
  for (auto &node : nodes) {
    if (isFixedIoType(node.type))
      applyHostIoChannels(node);
  }
  for (const auto &pair : bypasses)
    connect(pair.first, pair.second);
  if (cutFedFromHostInput || cutFedHostOutput) {
    GraphNode *hostInput = nullptr;
    GraphNode *hostOutput = nullptr;
    for (auto &node : nodes) {
      if (node.type == NodeType::audioInput && hostInput == nullptr)
        hostInput = &node;
      else if (node.type == NodeType::audioOutput && hostOutput == nullptr)
        hostOutput = &node;
    }
    if (hostInput != nullptr && hostOutput != nullptr &&
        !hostInput->outputs.empty() && !hostOutput->inputs.empty() &&
        !isPinConnected(hostInput->outputs.front().id) &&
        !isPinConnected(hostOutput->inputs.front().id))
      connect(hostInput->outputs.front().id, hostOutput->inputs.front().id);
  }
  return true;
}

void NodeGraph::moveNode(std::int32_t nodeId, juce::Point<float> position) {
  if (auto *node = findNode(nodeId))
    node->position = position;
}

const std::vector<GraphGroup> &NodeGraph::getGroups() const noexcept {
  return groups;
}

std::vector<GraphGroup> &NodeGraph::getGroups() noexcept { return groups; }

GraphGroup *NodeGraph::findGroup(std::int32_t groupId) noexcept {
  const auto found = std::find_if(
      groups.begin(), groups.end(),
      [groupId](const GraphGroup &group) { return group.id == groupId; });
  return found != groups.end() ? &*found : nullptr;
}

const GraphGroup *NodeGraph::findGroup(std::int32_t groupId) const noexcept {
  const auto found = std::find_if(
      groups.begin(), groups.end(),
      [groupId](const GraphGroup &group) { return group.id == groupId; });
  return found != groups.end() ? &*found : nullptr;
}

juce::Point<float>
NodeGraph::groupLocalToWorld(std::int32_t groupId,
                             juce::Point<float> localPoint) const {
  auto point = localPoint;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    point = group->position + groupContentOffset() +
            (point - group->viewPan) * group->viewZoom;
    current = group->parentGroupId;
  }
  return point;
}

juce::Point<float>
NodeGraph::worldToGroupLocal(std::int32_t groupId,
                             juce::Point<float> worldPoint) const {
  std::vector<const GraphGroup *> chain;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    chain.push_back(group);
    current = group->parentGroupId;
  }
  auto point = worldPoint;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const auto *group = *it;
    const auto zoom = std::max(0.01f, group->viewZoom);
    const auto origin = group->position + groupContentOffset();
    point = group->viewPan + (point - origin) / zoom;
  }
  return point;
}

juce::Point<float>
NodeGraph::worldPositionOfGroup(std::int32_t groupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr)
    return {};
  if (!group->parentGroupId.has_value())
    return group->position;
  return groupLocalToWorld(*group->parentGroupId, group->position);
}

juce::Point<float> NodeGraph::worldPositionOfNode(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return {};
  if (!node->parentGroupId.has_value())
    return node->position;
  return groupLocalToWorld(*node->parentGroupId, node->position);
}

float NodeGraph::groupContentToCanvasScale(std::int32_t groupId) const {
  float scale = 1.0f;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    scale *= std::max(0.01f, group->viewZoom);
    current = group->parentGroupId;
  }
  return scale;
}

void NodeGraph::setGroupView(std::int32_t groupId, juce::Point<float> pan,
                             float zoom) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->viewPan = pan;
  group->viewZoom =
      std::clamp(zoom, minimumZoom, maximumZoom);
}

std::vector<std::int32_t>
NodeGraph::collectLeafNodeIds(std::int32_t groupId) const {
  std::vector<std::int32_t> leaves;
  collectLeaves(*this, groupId, leaves);
  return leaves;
}

std::vector<std::int32_t> NodeGraph::orderSiblingBoxesByFlow(
    std::optional<std::int32_t> scopeGroupId,
    std::vector<std::int32_t> boxIds) const {
  boxIds.erase(std::remove_if(boxIds.begin(), boxIds.end(),
                              [this](std::int32_t boxId) {
                                return !isStructureBox(*this, boxId);
                              }),
               boxIds.end());
  if (boxIds.empty())
    return boxIds;

  const std::unordered_set<std::int32_t> candidates(boxIds.begin(),
                                                    boxIds.end());
  std::unordered_set<std::int32_t> flowNodes;
  collectScopeFlowNodes(*this, scopeGroupId, candidates, flowNodes);

  std::vector<BoxFlowEdge> edges;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (flowNodes.count(*source) == 0 || flowNodes.count(*destination) == 0)
      continue;
    const auto sourceBox = owningCandidateBox(*this, candidates, *source);
    const auto destinationBox =
        owningCandidateBox(*this, candidates, *destination);
    if (!sourceBox.has_value() || !destinationBox.has_value() ||
        *sourceBox == *destinationBox)
      continue;
    edges.push_back({*sourceBox, *destinationBox});
  }

  std::unordered_map<std::int32_t, juce::String> names;
  names.reserve(boxIds.size());
  for (const auto boxId : boxIds)
    names.emplace(boxId, boxDisplayName(*this, boxId));
  return orderBoxesByFlowRank(std::move(boxIds), names, edges);
}

int NodeGraph::effectiveRepeatCount(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return 1;
  int count = 1;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    count *= std::max(1, group->repeats);
    parent = group->parentGroupId;
  }
  return std::max(1, count);
}

GroupRepeatStatus NodeGraph::groupRepeatStatus(std::int32_t groupId) const {
  GroupRepeatStatus status;
  const auto *group = findGroup(groupId);
  if (group == nullptr) {
    status.active = false;
    status.message = "Group no longer exists";
    return status;
  }
  status.requestedRepeats =
      std::clamp(group->repeats, 1, maximumGroupRepeats);
  if (status.requestedRepeats == 1)
    return status;

  std::vector<GroupBoundaryPort> inputs;
  std::vector<GroupBoundaryPort> outputs;
  appendSerialChainPorts(*this, groupId, inputs, outputs);
  if (inputs.empty() || outputs.empty() || inputs.size() != outputs.size()) {
    status.active = false;
    status.message = "Requested " + std::to_string(status.requestedRepeats) +
                     " repeats are inactive: declare the same number of Group "
                     "Input and Group Output lanes";
    return status;
  }

  const auto *inputHub = findNode(inputs.front().memberNodeId);
  const auto *outputHub = findNode(outputs.front().memberNodeId);
  bool hasThroughPath = inputHub != nullptr && outputHub != nullptr;
  if (hasThroughPath) {
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visited;
    pending.push(inputHub->id);
    hasThroughPath = false;
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      if (current == outputHub->id) {
        hasThroughPath = true;
        break;
      }
      for (const auto &link : links) {
        const auto source = findNodeForPin(link.sourcePinId);
        const auto destination = findNodeForPin(link.destinationPinId);
        if (source.has_value() && destination.has_value() &&
            *source == current)
          pending.push(*destination);
      }
    }
  }
  if (!hasThroughPath) {
    status.active = false;
    status.message = "Requested " + std::to_string(status.requestedRepeats) +
                     " repeats are inactive: connect Group Input through the "
                     "group to Group Output";
    return status;
  }

  const auto chain = evaluateSerialRepeatShapeChain(
      *this, groupId, status.requestedRepeats, inputs, outputs);
  if (!chain.ok) {
    status.active = false;
    status.message = chain.message;
    status.propertyHints = chain.hints;
    return status;
  }

  status.effectiveRepeats = status.requestedRepeats;
  return status;
}

int NodeGraph::effectiveRuntimeRepeatCount(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return 1;
  int count = 1;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto status = groupRepeatStatus(*parent);
    count *= std::max(1, status.effectiveRepeats);
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  return std::max(1, count);
}

std::optional<std::string>
NodeGraph::groupRepeatPropertyHint(std::int32_t nodeId,
                                 const std::string &propertyKey) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return std::nullopt;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto status = groupRepeatStatus(*parent);
    for (const auto &hint : status.propertyHints) {
      if (hint.nodeId == nodeId && hint.propertyKey == propertyKey)
        return hint.message;
    }
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  return std::nullopt;
}

std::vector<std::string> NodeGraph::collectGraphWarnings() const {
  std::vector<std::string> warnings;
  for (const auto &group : groups) {
    const auto status = groupRepeatStatus(group.id);
    if (status.active || status.message.empty() || status.requestedRepeats <= 1)
      continue;
    warnings.push_back(group.name + ": " + status.message);
  }
  for (const auto &node : nodes) {
    if (!isDdspEffectType(node.type))
      continue;
    int reverbLength = 0;
    for (const auto &property : node.properties) {
      if (property.key == "reverb_length")
        reverbLength = property.value;
    }
    if (reverbLength < 1)
      continue;
    const auto milliseconds =
        static_cast<double>(reverbLength) * 1000.0 / 48000.0;
    if (milliseconds > liveSafeIrLengthMilliseconds)
      warnings.push_back(node.label +
                         ": reverb length exceeds the live-safe threshold and "
                         "may be expensive for real-time playback.");
    if (node.type == NodeType::reverb) {
      bool irConnected = false;
      bool irEmpty = true;
      for (const auto &pin : node.inputs) {
        if (!isIrInputPin(pin))
          continue;
        for (const auto &link : links) {
          if (link.destinationPinId != pin.id)
            continue;
          irConnected = true;
          const auto *source = findPin(link.sourcePinId);
          irEmpty = source == nullptr ||
                    (source->shape.channels == 0 && pin.shape.channels == 0);
        }
      }
      if (irConnected && irEmpty)
        warnings.push_back(
            node.label +
            ": external IR is connected but empty; using the internal IR.");
    }
  }
  return warnings;
}

std::string NodeGraph::graphWarningMessage() const {
  const auto warnings = collectGraphWarnings();
  if (warnings.empty())
    return {};
  std::string joined = warnings.front();
  for (std::size_t index = 1; index < warnings.size(); ++index) {
    joined += "\n\n";
    joined += warnings[index];
  }
  return joined;
}

std::vector<int> NodeGraph::ancestorRepeatCounts(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return {};
  std::vector<int> innerToOuter;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    innerToOuter.push_back(std::max(1, group->repeats));
    parent = group->parentGroupId;
  }
  std::reverse(innerToOuter.begin(), innerToOuter.end());
  return innerToOuter;
}

std::vector<int>
NodeGraph::ancestorRuntimeRepeatCounts(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return {};
  std::vector<int> innerToOuter;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto status = groupRepeatStatus(*parent);
    innerToOuter.push_back(std::max(1, status.effectiveRepeats));
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  std::reverse(innerToOuter.begin(), innerToOuter.end());
  return innerToOuter;
}

void NodeGraph::validateAuthoredRepeatLists(std::int32_t nodeId) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return;
  const auto counts = ancestorRepeatCounts(nodeId);
  for (auto &property : node->properties)
    syncRepeatListValidity(property, counts);
}

bool NodeGraph::isNodeHiddenByCollapse(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return false;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    if (group->collapsed)
      return true;
    parent = group->parentGroupId;
  }
  return false;
}

bool NodeGraph::isGroupHiddenByCollapse(std::int32_t groupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr || !group->parentGroupId.has_value())
    return false;
  auto parent = group->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *ancestor = findGroup(*parent);
    if (ancestor == nullptr)
      break;
    if (ancestor->collapsed)
      return true;
    parent = ancestor->parentGroupId;
  }
  return false;
}

bool NodeGraph::isNodeClippedByGroup(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return false;
  const auto *group = findGroup(*node->parentGroupId);
  if (group == nullptr || group->collapsed)
    return false;
  const auto origin = worldPositionOfGroup(group->id);
  const auto parentScale =
      group->parentGroupId.has_value()
          ? groupContentToCanvasScale(*group->parentGroupId)
          : 1.0f;
  const auto contentMinX = origin.x + groupContentPad * parentScale;
  const auto contentMinY = origin.y + groupHeaderHeight * parentScale;
  const auto contentMaxX =
      origin.x + (group->size.x - groupContentPad) * parentScale;
  const auto contentMaxY =
      origin.y + (group->size.y - groupContentPad) * parentScale;
  const auto world = worldPositionOfNode(nodeId);
  const auto width = std::max(8.0f, node->size.x);
  const auto height = std::max(8.0f, node->size.y);
  return world.x + width < contentMinX || world.x > contentMaxX ||
         world.y + height < contentMinY || world.y > contentMaxY;
}

bool NodeGraph::isNodeOnFocusedCanvas(
    std::int32_t nodeId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *node = findNode(nodeId);
  return node != nullptr && node->parentGroupId == focusedGroupId;
}

bool NodeGraph::isGroupOnFocusedCanvas(
    std::int32_t groupId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr || group->id == focusedGroupId)
    return false;
  return group->parentGroupId == focusedGroupId;
}

std::optional<std::int32_t> NodeGraph::focusedCanvasHostGroup(
    std::int32_t nodeId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return std::nullopt;
  if (node->parentGroupId == focusedGroupId)
    return std::nullopt;
  auto current = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    if (group->parentGroupId == focusedGroupId)
      return group->id;
    current = group->parentGroupId;
  }
  return std::nullopt;
}

std::vector<std::int32_t>
NodeGraph::groupAncestorChain(std::int32_t groupId) const {
  std::vector<std::int32_t> chain;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    chain.push_back(*current);
    current = group->parentGroupId;
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

std::optional<std::int32_t>
NodeGraph::findExpandedGroupAt(juce::Point<float> canvasPoint) const {
  std::optional<std::int32_t> best;
  int bestDepth = -1;
  for (const auto &group : groups) {
    if (group.collapsed || isGroupHiddenByCollapse(group.id))
      continue;
    const auto origin = worldPositionOfGroup(group.id);
    const auto scale = group.parentGroupId.has_value()
                           ? groupContentToCanvasScale(*group.parentGroupId)
                           : 1.0f;
    const auto size = juce::Point<float>(group.size.x * scale, group.size.y * scale);
    if (canvasPoint.x < origin.x || canvasPoint.y < origin.y ||
        canvasPoint.x > origin.x + size.x || canvasPoint.y > origin.y + size.y)
      continue;
    const auto depth = groupDepth(*this, group.id);
    if (depth >= bestDepth) {
      bestDepth = depth;
      best = group.id;
    }
  }
  return best;
}

std::vector<GroupBoundaryPort>
NodeGraph::groupBoundaryPorts(std::int32_t groupId) const {
  std::unordered_set<std::int32_t> leaves;
  for (const auto id : collectLeafNodeIds(groupId))
    leaves.insert(id);
  std::vector<GroupBoundaryPort> ports;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto sourceInside = leaves.count(*source) != 0;
    const auto destInside = leaves.count(*destination) != 0;
    if (sourceInside == destInside)
      continue;
    GroupBoundaryPort port;
    if (destInside) {
      const auto *pin = findPin(link.destinationPinId);
      const auto *node = findNode(*destination);
      if (pin == nullptr || node == nullptr)
        continue;
      port.memberPinId = pin->id;
      port.memberNodeId = node->id;
      port.kind = PinKind::input;
      port.shape = pin->shape;
      port.label = pin->label;
    } else {
      const auto *pin = findPin(link.sourcePinId);
      const auto *node = findNode(*source);
      if (pin == nullptr || node == nullptr)
        continue;
      port.memberPinId = pin->id;
      port.memberNodeId = node->id;
      port.kind = PinKind::output;
      port.shape = pin->shape;
      port.label = pin->label;
    }
    ports.push_back(std::move(port));
  }
  std::sort(ports.begin(), ports.end(),
            [](const GroupBoundaryPort &left, const GroupBoundaryPort &right) {
              if (left.kind != right.kind)
                return left.kind == PinKind::input;
              if (left.memberNodeId != right.memberNodeId)
                return left.memberNodeId < right.memberNodeId;
              return left.memberPinId < right.memberPinId;
            });
  return ports;
}

std::vector<GroupBoundaryPort>
NodeGraph::groupInterfacePorts(std::int32_t groupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr)
    return {};
  const GraphNode *inputHub = nullptr;
  const GraphNode *outputHub = nullptr;
  for (const auto memberId : group->memberIds) {
    const auto *member = findNode(memberId);
    if (member == nullptr)
      continue;
    if (member->type == NodeType::groupInput)
      inputHub = member;
    else if (member->type == NodeType::groupOutput)
      outputHub = member;
  }
  if (inputHub != nullptr || outputHub != nullptr) {
    std::vector<GroupBoundaryPort> explicitPorts;
    if (inputHub != nullptr) {
      for (std::size_t index = 0; index < inputHub->inputs.size(); ++index) {
        const auto &pin = inputHub->inputs[index];
        explicitPorts.push_back({pin.id, inputHub->id, PinKind::input, pin.shape,
                                 "Input " + std::to_string(index + 1) + " " +
                                     pin.label});
      }
    }
    if (outputHub != nullptr) {
      for (std::size_t index = 0; index < outputHub->outputs.size(); ++index) {
        const auto &pin = outputHub->outputs[index];
        explicitPorts.push_back(
            {pin.id, outputHub->id, PinKind::output, pin.shape,
             "Output " + std::to_string(index + 1) + " " + pin.label});
      }
    }
    return explicitPorts;
  }

  std::unordered_set<std::int32_t> leaves;
  for (const auto id : collectLeafNodeIds(groupId))
    leaves.insert(id);
  std::unordered_set<std::int32_t> internallyFedInputs;
  std::unordered_set<std::int32_t> internallyUsedOutputs;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (leaves.count(*source) == 0 || leaves.count(*destination) == 0)
      continue;
    internallyUsedOutputs.insert(link.sourcePinId);
    internallyFedInputs.insert(link.destinationPinId);
  }
  std::vector<GroupBoundaryPort> ports;
  const auto appendPin = [&](const GraphNode &node, const Pin &pin) {
    GroupBoundaryPort port;
    port.memberPinId = pin.id;
    port.memberNodeId = node.id;
    port.kind = pin.kind;
    port.shape = pin.shape;
    port.label = node.label.empty() ? pin.label : node.label + " " + pin.label;
    ports.push_back(std::move(port));
  };
  for (const auto id : collectLeafNodeIds(groupId)) {
    const auto *node = findNode(id);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs) {
      if (internallyFedInputs.count(pin.id) == 0)
        appendPin(*node, pin);
    }
    for (const auto &pin : node->outputs) {
      if (internallyUsedOutputs.count(pin.id) == 0)
        appendPin(*node, pin);
    }
  }
  std::sort(ports.begin(), ports.end(),
            [](const GroupBoundaryPort &left, const GroupBoundaryPort &right) {
              if (left.kind != right.kind)
                return left.kind == PinKind::input;
              if (left.memberNodeId != right.memberNodeId)
                return left.memberNodeId < right.memberNodeId;
              return left.memberPinId < right.memberPinId;
            });
  return ports;
}

std::optional<std::int32_t>
NodeGraph::innermostVisibleGroupOf(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return std::nullopt;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    if (!isGroupHiddenByCollapse(*parent))
      return parent;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  return std::nullopt;
}

void NodeGraph::fitGroupToMembers(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr || group->memberIds.empty())
    return;
  const auto oldPosition = group->position;
  const auto oldPan = group->viewPan;
  const auto oldZoom = std::max(0.01f, group->viewZoom);
  auto min = juce::Point<float>(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
  auto max = juce::Point<float>(std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest());
  bool any = false;
  struct ParentRect {
    std::int32_t id = 0;
    juce::Point<float> position;
    juce::Point<float> size;
  };
  std::vector<ParentRect> parentRects;
  parentRects.reserve(group->memberIds.size());
  for (const auto member : group->memberIds) {
    const auto bounds = memberBounds(*this, member);
    if (!bounds.has_value())
      continue;
    const auto parentPosition =
        oldPosition + groupContentOffset() +
        (bounds->first - oldPan) * oldZoom;
    const auto parentSize =
        juce::Point<float>(bounds->second.x * oldZoom, bounds->second.y * oldZoom);
    parentRects.push_back({member, parentPosition, parentSize});
    any = true;
    min.x = std::min(min.x, parentPosition.x);
    min.y = std::min(min.y, parentPosition.y);
    max.x = std::max(max.x, parentPosition.x + parentSize.x);
    max.y = std::max(max.y, parentPosition.y + parentSize.y);
  }
  if (!any)
    return;
  group->position = {min.x - groupFitPadding,
                     min.y - groupFitPadding - groupHeaderHeight};
  group->size = {std::max(160.0f, max.x - min.x + groupFitPadding * 2.0f),
                 std::max(120.0f, max.y - min.y + groupFitPadding * 2.0f +
                                      groupHeaderHeight)};
  group->viewPan = {};
  group->viewZoom = 1.0f;
  const auto origin = group->position + groupContentOffset();
  for (const auto &rect : parentRects)
    setItemStoredPosition(*this, rect.id, rect.position - origin);
}

void NodeGraph::setGroupBounds(std::int32_t groupId, juce::Point<float> position,
                               juce::Point<float> size) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->position = position;
  group->size = {std::max(80.0f, size.x), std::max(60.0f, size.y)};
}

void NodeGraph::moveGroup(std::int32_t groupId, juce::Point<float> position) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->position = position;
}

bool NodeGraph::renameGroup(std::int32_t groupId, const std::string &name) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  auto trimmed = name;
  while (!trimmed.empty() &&
         (trimmed.back() == ' ' || trimmed.back() == '\t'))
    trimmed.pop_back();
  if (trimmed.empty())
    return false;
  group->name = trimmed;
  return true;
}

bool NodeGraph::setGroupCollapsed(std::int32_t groupId, bool collapsed) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  group->collapsed = collapsed;
  return true;
}

bool NodeGraph::toggleGroupCollapsed(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  group->collapsed = !group->collapsed;
  return true;
}

void NodeGraph::ensureGroupBoundaryNodes(std::int32_t groupId,
                                         bool preserveInferredPorts) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  for (const auto memberId : group->memberIds) {
    const auto *member = findNode(memberId);
    if (member != nullptr && isGroupBoundaryType(member->type))
      return;
  }

  auto declared = preserveInferredPorts ? groupInterfacePorts(groupId)
                                        : groupBoundaryPorts(groupId);
  std::vector<GroupBoundaryPort> inputs;
  std::vector<GroupBoundaryPort> outputs;
  for (const auto &port : declared) {
    if (port.kind == PinKind::input)
      inputs.push_back(port);
    else
      outputs.push_back(port);
  }

  std::unordered_set<std::int32_t> leaves;
  for (const auto leaf : collectLeafNodeIds(groupId))
    leaves.insert(leaf);

  const auto inputId =
      addNode(NodeType::groupInput, groupInteriorOrigin, groupId);
  const auto outputId = addNode(
      NodeType::groupOutput,
      {groupInteriorOrigin.x + 220.0f, groupInteriorOrigin.y}, groupId);
  auto *inputHub = findNode(inputId);
  auto *outputHub = findNode(outputId);
  if (inputHub == nullptr || outputHub == nullptr)
    return;
  setGroupBoundaryPortCount(*inputHub,
                            std::max(1, static_cast<int>(inputs.size())));
  setGroupBoundaryPortCount(*outputHub,
                            std::max(1, static_cast<int>(outputs.size())));

  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto memberPin = inputs[index].memberPinId;
    if (const auto *pin = findPin(memberPin); pin != nullptr) {
      inputHub->inputs[index].label = pin->label;
      inputHub->outputs[index].label = pin->label;
    }
    for (auto &link : links) {
      if (link.destinationPinId != memberPin)
        continue;
      const auto source = findNodeForPin(link.sourcePinId);
      if (source.has_value() && leaves.count(*source) == 0)
        link.destinationPinId = inputHub->inputs[index].id;
    }
    links.push_back(
        {nextLinkId++, inputHub->outputs[index].id, memberPin});
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto memberPin = outputs[index].memberPinId;
    if (const auto *pin = findPin(memberPin); pin != nullptr) {
      outputHub->inputs[index].label = pin->label;
      outputHub->outputs[index].label = pin->label;
    }
    for (auto &link : links) {
      if (link.sourcePinId != memberPin)
        continue;
      const auto destination = findNodeForPin(link.destinationPinId);
      if (destination.has_value() && leaves.count(*destination) == 0)
        link.sourcePinId = outputHub->outputs[index].id;
    }
    links.push_back(
        {nextLinkId++, memberPin, outputHub->inputs[index].id});
  }
  refreshPropagatedPinShapes(*this);
  layoutGroupInteriorAroundBoundaryHubs(*this, groupId);
}

GroupActionResult
NodeGraph::createGroup(const std::vector<std::int32_t> &memberIds) {
  std::vector<std::int32_t> unique;
  unique.reserve(memberIds.size());
  std::unordered_set<std::int32_t> seen;
  std::optional<std::int32_t> sharedParent;
  bool parentInitialized = false;
  for (const auto id : memberIds) {
    if (!seen.insert(id).second)
      continue;
    const auto *node = findNode(id);
    const auto *group = findGroup(id);
    if (node == nullptr && group == nullptr)
      return {false, "Selection includes an unknown box", 0};
    if (node != nullptr &&
        (isFixedIoType(node->type) || isGroupBoundaryType(node->type)))
      return {false, "Audio and group boundary boxes cannot join a group", 0};
    const auto parent = parentGroupOf(*this, id);
    if (!parentInitialized) {
      sharedParent = parent;
      parentInitialized = true;
    } else if (parent != sharedParent) {
      return {false, "Group members must share the same parent group", 0};
    }
    unique.push_back(id);
  }
  if (unique.size() < 2)
    return {false, "Select at least two boxes to create a group", 0};

  const auto parentDepth =
      sharedParent.has_value() ? groupDepth(*this, *sharedParent) : 0;
  if (parentDepth + 1 > maximumGroupNestingDepth)
    return {false, "Group nesting is limited to 8 levels", 0};

  std::vector<juce::Point<float>> memberWorlds;
  memberWorlds.reserve(unique.size());
  for (const auto id : unique)
    memberWorlds.push_back(itemWorldPosition(*this, id));

  GraphGroup group;
  group.id = nextNodeId++;
  group.name = "Group";
  group.parentGroupId = sharedParent;
  group.memberIds = unique;
  groups.push_back(group);
  for (const auto id : unique) {
    eraseMemberFromParents(groups, id);
    assignParent(*this, id, group.id);
  }
  if (auto *created = findGroup(group.id))
    created->memberIds = unique;
  if (sharedParent.has_value()) {
    if (auto *parent = findGroup(*sharedParent))
      parent->memberIds.push_back(group.id);
  }

  auto min = juce::Point<float>(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
  auto max = juce::Point<float>(std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest());
  bool any = false;
  for (std::size_t index = 0; index < unique.size(); ++index) {
    const auto bounds = memberBounds(*this, unique[index]);
    auto parentSpace = memberWorlds[index];
    if (sharedParent.has_value())
      parentSpace = worldToGroupLocal(*sharedParent, memberWorlds[index]);
    const auto size = bounds.has_value() ? bounds->second
                                         : juce::Point<float>(8.0f, 8.0f);
    any = true;
    min.x = std::min(min.x, parentSpace.x);
    min.y = std::min(min.y, parentSpace.y);
    max.x = std::max(max.x, parentSpace.x + size.x);
    max.y = std::max(max.y, parentSpace.y + size.y);
  }
  if (any) {
    if (auto *created = findGroup(group.id)) {
      created->position = {min.x - groupFitPadding,
                           min.y - groupFitPadding - groupHeaderHeight};
      created->size = {
          std::max(160.0f, max.x - min.x + groupFitPadding * 2.0f),
          std::max(120.0f, max.y - min.y + groupFitPadding * 2.0f +
                               groupHeaderHeight)};
      created->viewPan = {};
      created->viewZoom = 1.0f;
    }
  }
  for (std::size_t index = 0; index < unique.size(); ++index)
    storeWorldPosition(*this, unique[index], memberWorlds[index], group.id);
  ensureGroupBoundaryNodes(group.id, false);
  refreshRepeatSlotsForGroup(*this, group.id);
  return {true, {}, group.id};
}

GroupActionResult NodeGraph::ungroup(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Group no longer exists", 0};
  const auto parent = group->parentGroupId;
  std::vector<std::int32_t> boundaryIds;
  for (const auto memberId : group->memberIds) {
    const auto *member = findNode(memberId);
    if (member != nullptr && isGroupBoundaryType(member->type))
      boundaryIds.push_back(memberId);
  }
  for (const auto boundaryId : boundaryIds)
    spliceGroupBoundaryNode(boundaryId);
  group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Group no longer exists", 0};
  const auto members = group->memberIds;
  std::vector<juce::Point<float>> memberWorlds;
  memberWorlds.reserve(members.size());
  for (const auto member : members)
    memberWorlds.push_back(itemWorldPosition(*this, member));
  for (std::size_t index = 0; index < members.size(); ++index) {
    assignParent(*this, members[index], parent);
    storeWorldPosition(*this, members[index], memberWorlds[index], parent);
  }
  if (parent.has_value()) {
    if (auto *parentGroup = findGroup(*parent)) {
      parentGroup->memberIds.erase(std::remove(parentGroup->memberIds.begin(),
                                               parentGroup->memberIds.end(),
                                               groupId),
                                   parentGroup->memberIds.end());
      parentGroup->memberIds.insert(parentGroup->memberIds.end(),
                                    members.begin(), members.end());
    }
  }
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [groupId](const GraphGroup &candidate) {
                                return candidate.id == groupId;
                              }),
               groups.end());
  pruneStickySpineId(viewport.stickySpine, groupId);
  if (viewport.focusedGroupId == groupId)
    viewport.focusedGroupId = parent;
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::deleteGroup(std::int32_t groupId) {
  if (findGroup(groupId) == nullptr)
    return {false, "Group no longer exists", 0};
  if (!removeBoxes({groupId}))
    return {false, "Group no longer exists", 0};
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::addToGroup(std::int32_t groupId,
                                        std::int32_t memberId,
                                        bool preserveStoredPosition) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Drop target group no longer exists", 0};
  if (memberId == groupId)
    return {false, "A group cannot contain itself", 0};
  if (const auto *node = findNode(memberId)) {
    if (isFixedIoType(node->type) || isGroupBoundaryType(node->type))
      return {false, "Audio and group boundary boxes cannot join a group", 0};
  } else if (findGroup(memberId) == nullptr) {
    return {false, "Drop source is not a graph box", 0};
  }
  if (findGroup(memberId) != nullptr && groupOwnsId(*this, memberId, groupId))
    return {false, "A group cannot be nested inside one of its descendants", 0};
  const auto newDepth = groupDepth(*this, groupId) +
                        (findGroup(memberId) != nullptr
                             ? groupDepth(*this, memberId)
                             : 1);
  if (newDepth > maximumGroupNestingDepth)
    return {false, "Group nesting is limited to 8 levels", 0};
  if (std::find(group->memberIds.begin(), group->memberIds.end(), memberId) !=
      group->memberIds.end())
    return {true, {}, groupId};

  const auto world = itemWorldPosition(*this, memberId);
  eraseMemberFromParents(groups, memberId);
  assignParent(*this, memberId, groupId);
  group->memberIds.push_back(memberId);
  if (!preserveStoredPosition)
    storeWorldPosition(*this, memberId, world, groupId);
  refreshRepeatSlotsForGroup(*this, groupId);
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::removeFromGroup(std::int32_t memberId) {
  if (const auto *node = findNode(memberId);
      node != nullptr && isGroupBoundaryType(node->type))
    return {false, "Group Input and Group Output cannot leave their group", 0};
  const auto parent = parentGroupOf(*this, memberId);
  if (!parent.has_value())
    return {false, "Box is not inside a group", 0};
  auto *group = findGroup(*parent);
  if (group == nullptr)
    return {false, "Parent group no longer exists", 0};
  const auto grandparent = group->parentGroupId;
  const auto world = itemWorldPosition(*this, memberId);
  eraseMemberFromParents(groups, memberId);
  assignParent(*this, memberId, grandparent);
  storeWorldPosition(*this, memberId, world, grandparent);
  if (grandparent.has_value()) {
    if (auto *ancestor = findGroup(*grandparent))
      ancestor->memberIds.push_back(memberId);
  }
  if (findNode(memberId) != nullptr) {
    auto *node = findNode(memberId);
    const auto repeats = effectiveRepeatCount(memberId);
    ensureRepeatSlotCount(*node, repeats);
    ensureNodePropertyRepeatCounts(*node, repeats);
  } else if (findGroup(memberId) != nullptr)
    refreshRepeatSlotsForGroup(*this, memberId);
  return {true, {}, *parent};
}

GroupActionResult NodeGraph::disconnectAllLinksForBox(std::int32_t boxId) {
  std::unordered_set<std::int32_t> pins;
  bool groupBox = false;
  if (const auto *node = findNode(boxId)) {
    for (const auto &pin : node->inputs)
      pins.insert(pin.id);
    for (const auto &pin : node->outputs)
      pins.insert(pin.id);
  } else if (findGroup(boxId) != nullptr) {
    groupBox = true;
    std::unordered_set<std::int32_t> nodeIds;
    std::unordered_set<std::int32_t> groupIds;
    collectGroupSubtree(*this, boxId, nodeIds, groupIds);
    for (const auto nodeId : nodeIds) {
      const auto *member = findNode(nodeId);
      if (member == nullptr)
        continue;
      for (const auto &pin : member->inputs)
        pins.insert(pin.id);
      for (const auto &pin : member->outputs)
        pins.insert(pin.id);
    }
  } else
    return {false, "Box no longer exists", 0};

  const auto oldSize = links.size();
  if (groupBox) {
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&pins](const GraphLink &link) {
                                 const auto sourceInside =
                                     pins.count(link.sourcePinId) != 0;
                                 const auto destInside =
                                     pins.count(link.destinationPinId) != 0;
                                 return sourceInside != destInside;
                               }),
                links.end());
  } else {
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&pins](const GraphLink &link) {
                                 return pins.count(link.sourcePinId) != 0 ||
                                        pins.count(link.destinationPinId) != 0;
                               }),
                links.end());
  }
  if (links.size() != oldSize) {
    refreshAllMergeOutputShapes(*this);
    refreshPropagatedPinShapes(*this);
  }
  return {true, {}, boxId};
}

GroupActionResult NodeGraph::reparentBoxLikeInsert(
    std::int32_t boxId, std::optional<std::int32_t> targetParent) {
  const auto *node = findNode(boxId);
  const auto *group = findGroup(boxId);
  if (node == nullptr && group == nullptr)
    return {false, "Box no longer exists", 0};
  if (node != nullptr &&
      (isFixedIoType(node->type) || isGroupBoundaryType(node->type)))
    return {false, "Audio and group boundary boxes cannot be reparented", 0};
  if (targetParent.has_value()) {
    if (findGroup(*targetParent) == nullptr)
      return {false, "Drop target group no longer exists", 0};
    if (*targetParent == boxId)
      return {false, "A group cannot contain itself", 0};
    if (group != nullptr && groupOwnsId(*this, boxId, *targetParent))
      return {false, "A group cannot be nested inside one of its descendants",
              0};
    const auto newDepth = groupDepth(*this, *targetParent) +
                          (group != nullptr ? groupDepth(*this, boxId) : 1);
    if (newDepth > maximumGroupNestingDepth)
      return {false, "Group nesting is limited to 8 levels", 0};
  }

  const auto previousParent = parentGroupOf(*this, boxId);
  if (previousParent == targetParent)
    return {true, {}, previousParent.value_or(boxId)};

  disconnectAllLinksForBox(boxId);

  eraseMemberFromParents(groups, boxId);
  assignParent(*this, boxId, std::nullopt);
  if (previousParent.has_value())
    refreshRepeatSlotsForGroup(*this, *previousParent);

  if (auto *movedNode = findNode(boxId))
    movedNode->position = defaultNewBoxPosition;
  else if (auto *movedGroup = findGroup(boxId))
    movedGroup->position = defaultNewBoxPosition;

  if (targetParent.has_value()) {
    const auto result = addToGroup(*targetParent, boxId, true);
    if (!result.accepted) {
      return result;
    }
    return result;
  }
  if (auto *movedNode = findNode(boxId)) {
    const auto repeats = effectiveRepeatCount(boxId);
    ensureRepeatSlotCount(*movedNode, repeats);
    ensureNodePropertyRepeatCounts(*movedNode, repeats);
  } else if (findGroup(boxId) != nullptr)
    refreshRepeatSlotsForGroup(*this, boxId);
  return {true, {}, 0};
}

std::vector<std::int32_t> NodeGraph::expandSelectionToFreezableLeaves(
    const std::vector<std::int32_t> &selectedIds) const {
  std::vector<std::int32_t> leaves;
  std::unordered_set<std::int32_t> seen;
  const auto appendNode = [&](std::int32_t nodeId) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isGroupBoundaryType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return;
    if (seen.insert(nodeId).second)
      leaves.push_back(nodeId);
  };
  for (const auto id : selectedIds) {
    if (findGroup(id) != nullptr) {
      for (const auto leaf : collectLeafNodeIds(id))
        appendNode(leaf);
    } else {
      appendNode(id);
    }
  }
  return leaves;
}

GroupActionResult NodeGraph::setGroupRepeats(std::int32_t groupId, int repeats) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Group no longer exists", 0};
  repeats = std::clamp(repeats, 1, maximumGroupRepeats);
  if (repeats == group->repeats)
    return {true, groupRepeatStatus(groupId).message, groupId};
  group->repeats = repeats;
  refreshRepeatSlotsForGroup(*this, groupId);
  refreshPropagatedPinShapes(*this);
  const auto status = groupRepeatStatus(groupId);
  if (!status.active)
    return {true, status.message, groupId};
  const auto warning = firstIncompatibleLinkMessage(*this);
  return {true, warning, groupId};
}

bool NodeGraph::randomizeGroupWeights(std::int32_t groupId) {
  if (findGroup(groupId) == nullptr)
    return false;
  bool any = false;
  for (const auto nodeId : collectLeafNodeIds(groupId)) {
    auto *node = findNode(nodeId);
    if (node == nullptr || !node->hasWeights ||
        node->state == NodeState::frozenGold)
      continue;
    if (node->type == NodeType::batchNorm)
      writeRandomizedWeightSlots(*node, 0, false);
    else if (node->useExplicitSeed)
      writeRandomizedWeightSlots(*node, node->explicitSeed, true);
    else {
      const auto seed =
          juce::Random::getSystemRandom().nextInt(maximumSeed + 1);
      writeRandomizedWeightSlots(*node, seed, true);
    }
    any = true;
  }
  return any;
}

NodeGraph NodeGraph::withInvisibleRepeatsMaterialized() const {
  return withInvisibleRepeatsMaterialized(nullptr);
}

NodeGraph NodeGraph::withInvisibleRepeatsMaterialized(
    std::unordered_map<std::int32_t, std::pair<std::int32_t, int>> *provenance)
    const {
  NodeGraph expanded;
  expanded.restoreFromValueTree(toValueTree());
  if (provenance != nullptr)
    provenance->clear();
  std::vector<std::int32_t> ordered;
  ordered.reserve(expanded.groups.size());
  for (const auto &group : expanded.groups)
    ordered.push_back(group.id);
  std::sort(ordered.begin(), ordered.end(),
            [&expanded](std::int32_t left, std::int32_t right) {
              return groupDepth(expanded, left) > groupDepth(expanded, right);
            });

  struct WorkingNode {
    std::int32_t originalId = 0;
    int slotIndex = 0;
  };
  std::unordered_map<std::int32_t, WorkingNode> working;
  for (const auto &node : expanded.nodes) {
    working[node.id] = WorkingNode{node.id, 0};
    if (provenance != nullptr)
      (*provenance)[node.id] = {node.id, 0};
  }

  for (const auto groupId : ordered) {
    auto *group = expanded.findGroup(groupId);
    if (group == nullptr || group->repeats <= 1)
      continue;
    const auto repeatStatus = expanded.groupRepeatStatus(groupId);
    if (!repeatStatus.active || repeatStatus.effectiveRepeats <= 1) {
      group->repeats = 1;
      continue;
    }
    const auto repeats = repeatStatus.effectiveRepeats;
    const auto templateNodes = expanded.collectLeafNodeIds(groupId);
    if (templateNodes.empty())
      continue;

    const auto innerProduct = [&](std::int32_t nodeId) {
      const auto found = working.find(nodeId);
      const auto originalId =
          found != working.end() ? found->second.originalId : nodeId;
      const auto *original = findNode(originalId);
      if (original == nullptr)
        return 1;
      int product = 1;
      auto parent = original->parentGroupId;
      std::unordered_set<std::int32_t> visiting;
      while (parent.has_value() && *parent != groupId) {
        if (!visiting.insert(*parent).second)
          break;
        const auto *ancestor = findGroup(*parent);
        if (ancestor == nullptr)
          break;
        product *= std::max(1, ancestor->repeats);
        parent = ancestor->parentGroupId;
      }
      return std::max(1, product);
    };

    std::unordered_set<std::int32_t> templateSet(templateNodes.begin(),
                                                 templateNodes.end());
    std::vector<GraphLink> internalLinks;
    for (const auto &link : expanded.links) {
      const auto sourceNode = expanded.findNodeForPin(link.sourcePinId);
      const auto dest = expanded.findNodeForPin(link.destinationPinId);
      if (sourceNode.has_value() && dest.has_value() &&
          templateSet.count(*sourceNode) != 0 && templateSet.count(*dest) != 0)
        internalLinks.push_back(link);
    }
    std::vector<GroupBoundaryPort> inputs;
    std::vector<GroupBoundaryPort> outputs;
    appendSerialChainPorts(expanded, groupId, inputs, outputs);

    struct RepeatInstance {
      std::unordered_map<std::int32_t, std::int32_t> pinMap;
      std::unordered_map<std::int32_t, std::int32_t> nodeMap;
    };
    std::vector<RepeatInstance> instances(static_cast<std::size_t>(repeats));
    for (const auto nodeId : templateNodes) {
      instances[0].nodeMap[nodeId] = nodeId;
      if (const auto *node = expanded.findNode(nodeId)) {
        for (const auto &pin : node->inputs)
          instances[0].pinMap[pin.id] = pin.id;
        for (const auto &pin : node->outputs)
          instances[0].pinMap[pin.id] = pin.id;
      }
      if (auto *node = expanded.findNode(nodeId)) {
        const auto slot = working[nodeId].slotIndex;
        const auto originalId = working[nodeId].originalId;
        if (const auto *original = findNode(originalId))
          node->properties = original->properties;
        ensureRepeatSlotCount(
            *node, std::max(slot + 1, effectiveRepeatCount(originalId)));
        if (slot < static_cast<int>(node->repeatSlots.size()))
          applyRepeatSlot(*node, node->repeatSlots[static_cast<std::size_t>(slot)]);
        applyRepeatPropertyValues(*node, slot);
        if (provenance != nullptr)
          (*provenance)[nodeId] = {originalId, slot};
      }
    }

    for (int repeat = 1; repeat < repeats; ++repeat) {
      auto &instance = instances[static_cast<std::size_t>(repeat)];
      for (const auto nodeId : templateNodes) {
        const auto *templateNode = expanded.findNode(nodeId);
        if (templateNode == nullptr)
          continue;
        GraphNode clone = *templateNode;
        clone.id = expanded.nextNodeId++;
        clone.parentGroupId = templateNode->parentGroupId;
        for (auto &pin : clone.inputs) {
          const auto originalPin = pin.id;
          pin.id = expanded.nextPinId++;
          instance.pinMap[originalPin] = pin.id;
        }
        for (auto &pin : clone.outputs) {
          const auto originalPin = pin.id;
          pin.id = expanded.nextPinId++;
          instance.pinMap[originalPin] = pin.id;
        }
        const auto slot =
            working[nodeId].slotIndex + repeat * innerProduct(nodeId);
        const auto originalId = working[nodeId].originalId;
        if (const auto *original = findNode(originalId)) {
          clone.repeatSlots = original->repeatSlots;
          clone.properties = original->properties;
          ensureRepeatSlotCount(
              clone, std::max(slot + 1,
                              static_cast<int>(original->repeatSlots.size())));
          if (slot < static_cast<int>(clone.repeatSlots.size()))
            applyRepeatSlot(clone,
                          clone.repeatSlots[static_cast<std::size_t>(slot)]);
          applyRepeatPropertyValues(clone, slot);
        }
        instance.nodeMap[nodeId] = clone.id;
        working[clone.id] = WorkingNode{originalId, slot};
        if (provenance != nullptr)
          (*provenance)[clone.id] = {originalId, slot};
        registerCloneMembership(expanded, clone.parentGroupId, clone.id);
        expanded.nodes.push_back(std::move(clone));
      }
      for (const auto &link : internalLinks) {
        const auto sourcePin = instance.pinMap.find(link.sourcePinId);
        const auto destPin = instance.pinMap.find(link.destinationPinId);
        if (sourcePin == instance.pinMap.end() ||
            destPin == instance.pinMap.end())
          continue;
        GraphLink cloned;
        cloned.id = expanded.nextLinkId++;
        cloned.sourcePinId = sourcePin->second;
        cloned.destinationPinId = destPin->second;
        expanded.links.push_back(cloned);
      }
    }

    // Retarget outs that leave this group onto the last repeat. Destinations
    // still inside the group — including serial edges from a nested unroll —
    // stay put; rewriting those onto the last outer repeat closes a cycle.
    const auto &last = instances.back();
    for (auto &link : expanded.links) {
      const auto sourceNode = expanded.findNodeForPin(link.sourcePinId);
      if (!sourceNode.has_value() || templateSet.count(*sourceNode) == 0)
        continue;
      const auto dest = expanded.findNodeForPin(link.destinationPinId);
      if (dest.has_value() && (templateSet.count(*dest) != 0 ||
                               nodeIsInsideGroup(expanded, *dest, groupId)))
        continue;
      const auto mapped = last.pinMap.find(link.sourcePinId);
      if (mapped != last.pinMap.end())
        link.sourcePinId = mapped->second;
    }

    const auto pairCount = std::min(inputs.size(), outputs.size());
    for (int repeat = 0; repeat + 1 < repeats; ++repeat) {
      const auto &current = instances[static_cast<std::size_t>(repeat)];
      const auto &next = instances[static_cast<std::size_t>(repeat + 1)];
      for (std::size_t index = 0; index < pairCount; ++index) {
        const auto outPin = current.pinMap.find(outputs[index].memberPinId);
        const auto inPin = next.pinMap.find(inputs[index].memberPinId);
        if (outPin == current.pinMap.end() || inPin == next.pinMap.end())
          continue;
        GraphLink chain;
        chain.id = expanded.nextLinkId++;
        chain.sourcePinId = outPin->second;
        chain.destinationPinId = inPin->second;
        expanded.links.push_back(chain);
      }
    }
    group->repeats = 1;
  }
  expanded.flattenGroupBoundaryNodes();
  return expanded;
}

ConnectionResult NodeGraph::connect(std::int32_t firstPinId,
                                    std::int32_t secondPinId) {
  firstPinId = resolveCollapsedPin(firstPinId);
  secondPinId = resolveCollapsedPin(secondPinId);
  const auto *first = findPin(firstPinId);
  const auto *second = findPin(secondPinId);
  if (first == nullptr || second == nullptr)
    return {false, "Connection endpoint no longer exists"};
  if (first->kind == second->kind)
    return {false, "Connect an output port to an input port"};

  const auto *source = first->kind == PinKind::output ? first : second;
  const auto *destination = first->kind == PinKind::input ? first : second;

  const auto sourceNode = findNodeForPin(source->id);
  const auto destinationNode = findNodeForPin(destination->id);
  if (!sourceNode.has_value() || !destinationNode.has_value())
    return {false, "Connection endpoint has no owning element"};
  if (*sourceNode == *destinationNode ||
      wouldCreateCycle(*sourceNode, *destinationNode))
    return {false, "Cycles are not allowed in the audio graph"};

  const auto *destinationNodePtr = findNode(*destinationNode);
  const auto *sourceNodePtr = findNode(*sourceNode);

  if (destinationNodePtr != nullptr &&
      (destinationNodePtr->type == NodeType::merge ||
       isMathExpressionType(destinationNodePtr->type)) &&
      destination->kind == PinKind::input) {
    if (const auto message = utilityNewInputIncompatibilityMessage(
            *this, *destinationNodePtr, source->id);
        !message.empty())
      return {false, message};
  }

  if (sourceNodePtr != nullptr &&
      (sourceNodePtr->type == NodeType::merge ||
       isMathExpressionType(sourceNodePtr->type)))
    updateMergeOutputShape(*this, *const_cast<GraphNode *>(sourceNodePtr));

  const auto sourceShape = outgoingShapeOf(*this, *source);
  if (sourceNodePtr != nullptr && isExternalLoadNode(*sourceNodePtr) &&
      sourceNodePtr->externalShapeIncomplete && !isLatentPin(*source) &&
      !isControlInputPin(*source))
    return {false,
            "Enter channel overrides before connecting this checkpoint"};
  if (destinationNodePtr != nullptr &&
      isExternalLoadNode(*destinationNodePtr) &&
      destinationNodePtr->externalShapeIncomplete &&
      !isLatentPin(*destination) && !isControlInputPin(*destination))
    return {false,
            "Enter channel overrides before connecting this checkpoint"};
  if (!sourceShape.isCompatibleWith(destination->shape)) {
    if (sourceNodePtr != nullptr && sourceNodePtr->type == NodeType::merge) {
      std::unordered_set<std::int32_t> visiting;
      const auto outputChannels =
          computeMergeOutputChannels(*this, *sourceNodePtr, visiting);
      ShapeSignature mergeShape = sourceShape;
      if (outputChannels > 0)
        mergeShape.channels = outputChannels;
      if (!mergeShape.isCompatibleWith(destination->shape)) {
        auto message = mergeShape.incompatibilityMessage(destination->shape);
        if (message.empty())
          message = "Shape mismatch: channel counts are incompatible";
        return {false, message};
      }
    } else {
      auto message = sourceShape.incompatibilityMessage(destination->shape);
      if (message.empty())
        message = "Shape mismatch: channel counts are incompatible";
      return {false, message};
    }
  }

  if (destinationNodePtr != nullptr &&
      (isConvolutionType(destinationNodePtr->type) ||
       isConvTransposeType(destinationNodePtr->type))) {
    const auto stride =
        std::max(1, readNodeProperty(*destinationNodePtr, "stride", 1));
    const auto upsample = isConvTransposeType(destinationNodePtr->type);
    if (convolutionRateIsError(stride, upsample, sourceShape.temporalRate))
      return {false, convolutionRateMessage(stride, upsample,
                                            sourceShape.temporalRate)};
  }

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::pqmfSynthesis &&
      pqmfSynthesisChannelIsError(
          sourceShape.channels,
          std::max(2, readNodeProperty(*destinationNodePtr, "n_band",
                                       defaultPqmfBands))))
    return {false, pqmfSynthesisChannelMessage(
                       sourceShape.channels,
                       std::max(2, readNodeProperty(*destinationNodePtr,
                                                      "n_band",
                                                      defaultPqmfBands)))};

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::variationalBottleneck &&
      variationalBottleneckChannelIsError(sourceShape.channels))
    return {false,
            variationalBottleneckChannelMessage(sourceShape.channels)};

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::noiseSynthesizer) {
    const auto noiseBands =
        std::max(1, readNodeProperty(*destinationNodePtr, "noise_bands",
                                     defaultNoiseBands));
    const auto windowSize =
        std::max(1, readNodeProperty(*destinationNodePtr, "window_size",
                                     defaultNoiseWindowSize));
    if (noiseSynthChannelIsError(sourceShape.channels, noiseBands))
      return {false, noiseSynthChannelMessage(sourceShape.channels, noiseBands)};
    if (noiseSynthWindowIsError(windowSize, noiseBands))
      return {false, noiseSynthWindowMessage(windowSize, noiseBands)};
    if (convolutionRateIsError(windowSize, true, sourceShape.temporalRate))
      return {false, convolutionRateMessage(windowSize, true,
                                            sourceShape.temporalRate)};
  }

  const auto duplicate = std::any_of(
      links.begin(), links.end(), [source, destination](const GraphLink &link) {
        return link.sourcePinId == source->id &&
               link.destinationPinId == destination->id;
      });
  if (duplicate)
    return {false, "These ports are already connected"};

  const auto occupied = std::any_of(
      links.begin(), links.end(), [destination](const GraphLink &link) {
        return link.destinationPinId == destination->id;
      });
  if (occupied)
    return {false, "This input already has a connection"};

  links.push_back({nextLinkId++, source->id, destination->id});
  refreshPropagatedPinShapes(*this);
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    links.pop_back();
    refreshPropagatedPinShapes(*this);
    return {false, incompatible};
  }
  return {true, {}};
}

bool NodeGraph::removeLink(std::int32_t linkId) {
  const auto oldSize = links.size();
  links.erase(std::remove_if(links.begin(), links.end(),
                             [linkId](const GraphLink &link) {
                               return link.id == linkId;
                             }),
              links.end());
  if (links.size() != oldSize)
    refreshAllMergeOutputShapes(*this);
  return links.size() != oldSize;
}

bool NodeGraph::setProperty(std::int32_t nodeId, const std::string &key,
                            int value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end())
    return false;
  const auto previousValue = property->value;
  const auto previousRepeatValues = property->repeatIntValues;
  const auto previousPreserveIn = property->preserveInBound;
  const auto previousInvalid = property->repeatListInvalid;
  if (key == "ports" && isGroupBoundaryType(node->type)) {
    if (!setGroupBoundaryPortCount(*node, value))
      return false;
    refreshPropagatedPinShapes(*this);
    return true;
  }
  if ((key == "n_frames" || key == "n_filter_banks") &&
      (node->type == NodeType::firFilter ||
       node->type == NodeType::filteredNoiseReverb) &&
      value < 1) {
    lastPropertyError = "Magnitude-grid dimensions must be at least 1";
    return false;
  }
  if (key == "window_size" &&
      (node->type == NodeType::firFilter ||
       node->type == NodeType::filteredNoiseReverb) &&
      value < 1) {
    lastPropertyError = "Window size must be at least 1";
    return false;
  }
  property->setValue(value);

  const auto syncRepeatValues = [&]() {
    if (!propertySupportsRepeatValueList(*property))
      return;
    property->preserveInBound = false;
    property->repeatListInvalid = false;
    property->repeatListInvalidMessage.clear();
    property->repeatIntValues = {property->value};
    property->authoredTokens.clear();
  };
  const auto restoreRepeatValues = [&]() {
    property->repeatIntValues = previousRepeatValues;
    property->preserveInBound = previousPreserveIn;
    property->repeatListInvalid = previousInvalid;
  };
  syncRepeatValues();

  if (key == "channels" && isFixedIoType(node->type)) {
    property->setValue(std::clamp(property->value, 0, 2));
    const auto choice = property->value;
    int pairedPrevious = choice;
    GraphNode *paired = nullptr;
    for (auto &candidate : nodes) {
      if (!isFixedIoType(candidate.type) || candidate.id == node->id)
        continue;
      paired = &candidate;
      for (auto &pairedProperty : candidate.properties) {
        if (pairedProperty.key != "channels")
          continue;
        pairedPrevious = pairedProperty.value;
        pairedProperty.setValue(choice);
        break;
      }
      applyHostIoChannels(candidate);
    }
    applyHostIoChannels(*node);
    refreshPropagatedPinShapes(*this);
    const auto incompatible = firstIncompatibleLinkMessage(*this);
    if (!incompatible.empty()) {
      property->setValue(previousValue);
      restoreRepeatValues();
      applyHostIoChannels(*node);
      if (paired != nullptr) {
        for (auto &pairedProperty : paired->properties) {
          if (pairedProperty.key != "channels")
            continue;
          pairedProperty.setValue(pairedPrevious);
          break;
        }
        applyHostIoChannels(*paired);
      }
      refreshPropagatedPinShapes(*this);
      return false;
    }
    syncRepeatValues();
    return true;
  } else if (key == "channels" && isConvolutionType(node->type)) {
    for (auto &pin : node->outputs)
      pin.shape.channels = property->value;
  } else if (key == "features" && node->type == NodeType::linear) {
    for (auto &pin : node->outputs)
      pin.shape.channels = property->value;
  } else if (key == "inputs" && isMixerType(node->type)) {
    setMixerInputCount(*node, property->value);
  } else if (key == "inputs" && isMathExpressionType(node->type)) {
    const auto parsed = parseExpression(mathExpressionText(*node),
                                        ExpressionIdentContext::mathInputs,
                                        property->value);
    if (!parsed.accepted) {
      property->setValue(previousValue);
      restoreRepeatValues();
      lastPropertyError = parsed.message;
      return false;
    }
    const auto maxRef = maxReferencedMathInput(parsed.ast);
    if (maxRef > property->value) {
      property->setValue(previousValue);
      restoreRepeatValues();
      lastPropertyError =
          "Reduce Inputs only after the expression no longer references x" +
          std::to_string(maxRef);
      return false;
    }
    setMixerInputCount(*node, property->value, true);
  } else if (key == "mode" && node->type == NodeType::merge) {
    updateMergeOutputShape(*this, *node);
    if (!mergeDownstreamIsCompatible(*this, *node)) {
      property->setValue(previousValue);
      restoreRepeatValues();
      updateMergeOutputShape(*this, *node);
      return false;
    }
  } else if (key == "activation" && node->type == NodeType::activation &&
             property->kind == PropertyKind::choice && property->value >= 0 &&
             property->value < static_cast<int>(property->choices.size())) {
    node->detail = property->choices[static_cast<std::size_t>(property->value)];
    node->hasWeights = property->value == 4;
    if (node->hasWeights)
      node->armedForTraining = true;
  } else if (key == "residual" && node->type == NodeType::tcn) {
    node->residual = property->value != 0;
  } else if (key == "dilation_growth" && node->type == NodeType::tcn) {
    node->dilationGrowth =
        std::max(minimumDilationGrowth, property->value);
    property->value = node->dilationGrowth;
  } else if (key == "n_band" &&
             (node->type == NodeType::pqmfAnalysis ||
              node->type == NodeType::pqmfSynthesis)) {
    property->value = std::max(minimumPqmfBands, property->value);
  } else if (key == "latent_size" &&
             node->type == NodeType::variationalBottleneck) {
    if (variationalBottleneckLatentIsError(property->value)) {
      property->setValue(previousValue);
      restoreRepeatValues();
      return false;
    }
    for (auto &pin : node->outputs) {
      pin.shape.channels = property->value;
    }
  } else if (key == "stride" &&
             (isConvolutionType(node->type) ||
              isConvTransposeType(node->type))) {
    property->value = std::max(1, property->value);
    if (isConvolutionType(node->type))
      updateConv1dDetail(*node);
    else
      updateConvTransposeDetail(*node);
  } else if (key == "hidden_size" && isRecurrentType(node->type)) {
    property->value = std::max(1, property->value);
    const auto bidirectional = readNodeProperty(*node, "bidirectional", 0) != 0;
    for (auto &pin : node->outputs)
      pin.shape.channels =
          bidirectional ? property->value * 2 : property->value;
  } else if (key == "bidirectional" && isRecurrentType(node->type)) {
    const auto hidden =
        std::max(1, readNodeProperty(*node, "hidden_size", defaultHiddenSize));
    for (auto &pin : node->outputs)
      pin.shape.channels = property->value != 0 ? hidden * 2 : hidden;
  } else if (key == "window_size" &&
             (node->type == NodeType::firFilter ||
              node->type == NodeType::filteredNoiseReverb)) {
    if (property->value < 1) {
      property->setValue(previousValue);
      restoreRepeatValues();
      lastPropertyError = "Window size must be at least 1";
      return false;
    }
    if ((property->value % 2) == 0)
      property->value += 1;
  } else if ((key == "n_frames" || key == "n_filter_banks") &&
             (node->type == NodeType::firFilter ||
              node->type == NodeType::filteredNoiseReverb)) {
    if (property->value < 1) {
      property->setValue(previousValue);
      restoreRepeatValues();
      lastPropertyError = "Magnitude-grid dimensions must be at least 1";
      return false;
    }
  } else if (key == "reverb_length" && isDdspEffectType(node->type)) {
    property->value = std::max(minimumReverbLength, property->value);
  }
  refreshPropagatedPinShapes(*this);
  auto rollbackProperty = [&]() {
    property->setValue(previousValue);
    restoreRepeatValues();
    if (isConvolutionType(node->type))
      updateConv1dDetail(*node);
    else if (isConvTransposeType(node->type))
      updateConvTransposeDetail(*node);
    if (key == "mode" && node->type == NodeType::merge)
      updateMergeOutputShape(*this, *node);
    refreshPropagatedPinShapes(*this);
  };
  for (const auto &candidate : nodes) {
    if (!isMixerType(candidate.type) && !isMathExpressionType(candidate.type))
      continue;
    if (const auto message =
            utilityInputsIncompatibilityMessage(*this, candidate);
        !message.empty()) {
      rollbackProperty();
      lastPropertyError = message;
      return false;
    }
  }
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    rollbackProperty();
    return false;
  }
  syncRepeatValues();
  lastPropertyError.clear();
  return true;
}

bool NodeGraph::setFloatProperty(std::int32_t nodeId, const std::string &key,
                                 float value) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  const bool fidelityOnGold =
      key == "fidelity" && node->state == NodeState::frozenGold;
  if (node->state == NodeState::frozenGold && !fidelityOnGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      property->kind != PropertyKind::real)
    return false;
  if (key == "negative_slope" || key == "leak_rate" ||
      key == "recurrent_weight_scale") {
    if (value < property->floatMinimum || value > property->floatMaximum)
      return false;
  }
  if (node->type == NodeType::modDelay &&
      (key == "center_ms" || key == "depth_ms")) {
    float center = defaultModDelayCenterMs;
    float depth = defaultModDelayDepthMs;
    for (const auto &candidate : node->properties) {
      if (candidate.key == "center_ms")
        center = candidate.floatValue;
      if (candidate.key == "depth_ms")
        depth = candidate.floatValue;
    }
    if (key == "center_ms")
      center = value;
    else
      depth = value;
    if (center - depth < 0.0f) {
      lastPropertyError =
          "ModDelay depth cannot exceed center delay (negative delay)";
      return false;
    }
  }
  property->setFloatValue(value);
  if (key == "fidelity")
    node->fidelityPercent = clampFidelity(property->floatValue);
  if (propertySupportsRepeatValueList(*property)) {
    property->repeatListInvalid = false;
    property->repeatListInvalidMessage.clear();
    property->repeatFloatValues = {property->floatValue};
    property->authoredTokens.clear();
  }
  lastPropertyError.clear();
  return true;
}

bool NodeGraph::setStringProperty(std::int32_t nodeId, const std::string &key,
                                  const std::string &value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      property->kind != PropertyKind::string)
    return false;
  lastPropertyError.clear();
  if (key == "expression" && isMathExpressionType(node->type)) {
    const auto inputCount = std::max(1, static_cast<int>(node->inputs.size()));
    const auto parsed = parseExpression(
        value, ExpressionIdentContext::mathInputs, inputCount);
    if (!parsed.accepted) {
      lastPropertyError = parsed.message;
      return false;
    }
    property->stringValue = value;
    node->detail = value;
    refreshPropagatedPinShapes(*this);
    return true;
  }
  property->stringValue = value;
  return true;
}

bool NodeGraph::setPropertyRepeatValues(std::int32_t nodeId,
                                      const std::string &key,
                                      const std::vector<int> &values,
                                      const std::vector<std::string> &authoredTokens) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      !propertySupportsRepeatValueList(*property) ||
      property->kind != PropertyKind::integer)
    return false;
  const auto counts = ancestorRepeatCounts(nodeId);
  if (!isDividingSetLength(static_cast<int>(values.size()), counts))
    return false;
  std::vector<int> clamped;
  clamped.reserve(values.size());
  for (const auto value : values) {
    if (value < property->minimum || value > property->maximum)
      return false;
    clamped.push_back(std::clamp(value, property->minimum, property->maximum));
  }
  if (!setProperty(nodeId, key, clamped.front()))
    return false;
  property->preserveInBound = false;
  property->repeatListInvalid = false;
  property->repeatListInvalidMessage.clear();
  property->repeatIntValues = std::move(clamped);
  property->value = property->repeatIntValues.front();
  property->authoredTokens = authoredTokens;
  lastPropertyError.clear();
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::setPropertyPreserveIn(std::int32_t nodeId, const std::string &key,
                                      int authoredLength) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      !propertySupportsPreserveIn(*property))
    return false;
  const auto counts = ancestorRepeatCounts(nodeId);
  if (!isDividingSetLength(std::max(1, authoredLength), counts))
    return false;
  const auto previousValue = property->value;
  const auto previousRepeatValues = property->repeatIntValues;
  const auto previousPreserveIn = property->preserveInBound;
  const auto previousInvalid = property->repeatListInvalid;
  property->preserveInBound = true;
  property->repeatListInvalid = false;
  property->repeatListInvalidMessage.clear();
  property->repeatIntValues.assign(static_cast<std::size_t>(std::max(1, authoredLength)),
                                 0);
  refreshPropagatedPinShapes(*this);
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    property->value = previousValue;
    property->repeatIntValues = previousRepeatValues;
    property->preserveInBound = previousPreserveIn;
    property->repeatListInvalid = previousInvalid;
    refreshPropagatedPinShapes(*this);
    return false;
  }
  return true;
}

bool NodeGraph::setFloatPropertyRepeatValues(std::int32_t nodeId,
                                           const std::string &key,
                                           const std::vector<float> &values,
                                           const std::vector<std::string> &authoredTokens) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  const bool fidelityOnGold =
      key == "fidelity" && node->state == NodeState::frozenGold;
  if (node->state == NodeState::frozenGold && !fidelityOnGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      !propertySupportsRepeatValueList(*property) ||
      property->kind != PropertyKind::real)
    return false;
  const auto counts = ancestorRepeatCounts(nodeId);
  if (!isDividingSetLength(static_cast<int>(values.size()), counts))
    return false;
  std::vector<float> clamped;
  clamped.reserve(values.size());
  for (const auto value : values) {
    if (value < property->floatMinimum || value > property->floatMaximum)
      return false;
    clamped.push_back(
        std::clamp(value, property->floatMinimum, property->floatMaximum));
  }
  if (!setFloatProperty(nodeId, key, clamped.front()))
    return false;
  property->repeatListInvalid = false;
  property->repeatListInvalidMessage.clear();
  property->repeatFloatValues = std::move(clamped);
  property->floatValue = property->repeatFloatValues.front();
  property->authoredTokens = authoredTokens;
  lastPropertyError.clear();
  if (key == "fidelity")
    node->fidelityPercent = clampFidelity(property->floatValue);
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::setConditioningValue(std::int32_t nodeId, float value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->type != NodeType::knobInput)
    return false;
  node->conditioningValue = clampConditioning(value);
  return true;
}

bool NodeGraph::setConditioningPad(std::int32_t nodeId, float x, float y) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->type != NodeType::xyTrackpad)
    return false;
  node->conditioningX = clampConditioning(x);
  node->conditioningY = clampConditioning(y);
  return true;
}

bool NodeGraph::setSelectedAnalysisView(std::int32_t nodeId,
                                        AnalysisView view) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  node->selectedAnalysisView = view;
  return true;
}

bool NodeGraph::setSeed(std::int32_t nodeId, std::int32_t seed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights ||
      node->state == NodeState::frozenGold)
    return false;
  node->seed = clampSeed(seed);
  if (node->useExplicitSeed)
    node->explicitSeed = node->seed;
  if (!node->repeatSlots.empty() &&
      node->weightsProvenance == WeightsProvenance::random) {
    for (std::size_t index = 0; index < node->repeatSlots.size(); ++index) {
      auto &slot = node->repeatSlots[index];
      if (slot.provenance != WeightsProvenance::random)
        continue;
      slot.seed = seedForRepeatSlot(node->seed, index);
    }
    node->repeatSlots.front() = repeatSlotFromNode(*node);
  }
  return true;
}

std::optional<std::int32_t>
NodeGraph::freezeSelection(const std::vector<std::int32_t> &selectedNodeIds,
                           const FreezeSelectionResult &result) {
  if (!result.succeeded || selectedNodeIds.empty() ||
      result.inputChannels < 1 || result.outputChannels < 1 ||
      !selectionIsConnected(selectedNodeIds))
    return std::nullopt;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isGroupBoundaryType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return std::nullopt;
  }

  for (const auto nodeId : selectedNodeIds) {
    auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    node->state = NodeState::frozenGold;
    node->colour = colourForType(node->type, node->state);
    node->artifactPath = result.artifactPath;
    node->metrics.reset();
    node->sourceSubgraph.clear();
  }

  std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                            selectedNodeIds.end());
  std::int32_t sinkId = 0;
  for (const auto nodeId : selectedNodeIds) {
    bool hasSuccessorInSelection = false;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (source.has_value() && *source == nodeId &&
          destination.has_value() && selected.count(*destination) != 0) {
        hasSuccessorInSelection = true;
        break;
      }
    }
    if (!hasSuccessorInSelection) {
      sinkId = nodeId;
      break;
    }
  }
  if (auto *sink = findNode(sinkId != 0 ? sinkId : selectedNodeIds.back()))
    sink->metrics = result.baselineMetrics;
  return selectedNodeIds.front();
}

bool NodeGraph::unfreeze(std::int32_t nodeId) {
  const auto *node = findNode(nodeId);
  if (node == nullptr || node->state != NodeState::frozenGold)
    return false;
  if (isExternalLoadNode(*node))
    return false;

  if (node->type == NodeType::blackBox) {
    const auto fragmentXml = juce::XmlDocument::parse(node->sourceSubgraph);
    if (fragmentXml == nullptr)
      return false;
    const auto fragment = juce::ValueTree::fromXml(*fragmentXml);
    if (!fragment.hasType("GraphFragment"))
      return false;
    const auto trainedPath =
        node->weightsPath.empty() ? node->artifactPath : node->weightsPath;
    const auto keepTrainedWeights =
        node->blackBoxOrigin == BlackBoxOrigin::trainAutoload ||
        node->weightsProvenance == WeightsProvenance::file;
    const auto trainOrigin =
        node->blackBoxOrigin == BlackBoxOrigin::trainAutoload;
    const auto fidelity = node->fidelityPercent;

    // Strip Gold without bypassing so restored cables are the only signal path.
    const auto goldId = nodeId;
    std::unordered_set<std::int32_t> goldPins;
    if (const auto *gold = findNode(goldId)) {
      for (const auto &pin : gold->inputs)
        goldPins.insert(pin.id);
      for (const auto &pin : gold->outputs)
        goldPins.insert(pin.id);
    }
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&goldPins](const GraphLink &link) {
                                 return goldPins.count(link.sourcePinId) != 0 ||
                                        goldPins.count(link.destinationPinId) != 0;
                               }),
                links.end());
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                [goldId](const GraphNode &candidate) {
                                  return candidate.id == goldId;
                                }),
                 nodes.end());
    eraseMemberFromParents(groups, goldId);
    for (const auto child : fragment) {
      if (!child.hasType("Group"))
        continue;
      auto restored = groupFromTree(child);
      nextNodeId = std::max(nextNodeId, restored.id + 1);
      if (findGroup(restored.id) == nullptr)
        groups.push_back(std::move(restored));
    }
    for (const auto child : fragment) {
      if (!child.hasType("Node"))
        continue;
      auto restored = nodeFromTree(child);
      nextNodeId = std::max(nextNodeId, restored.id + 1);
      for (const auto &pin : restored.inputs)
        nextPinId = std::max(nextPinId, pin.id + 1);
      for (const auto &pin : restored.outputs)
        nextPinId = std::max(nextPinId, pin.id + 1);
      if (keepTrainedWeights && restored.hasWeights) {
        restored.weightsProvenance = WeightsProvenance::file;
        restored.weightsPath = trainedPath;
        restored.artifactPath = trainedPath;
        if (trainOrigin)
          restored.blackBoxOrigin = BlackBoxOrigin::trainAutoload;
      }
      if (restored.type == NodeType::variationalBottleneck) {
        restored.fidelityPercent = fidelity;
        for (auto &property : restored.properties) {
          if (property.key == "fidelity")
            property.floatValue = fidelity;
        }
        copyCompactnessFromArtifact(restored, trainedPath);
      }
      nodes.push_back(std::move(restored));
    }
    for (const auto child : fragment) {
      if (!child.hasType("Link"))
        continue;
      auto restored = linkFromTree(child);
      if (findPin(restored.sourcePinId) != nullptr &&
          findPin(restored.destinationPinId) != nullptr) {
        nextLinkId = std::max(nextLinkId, restored.id + 1);
        links.push_back(restored);
      }
    }
    return true;
  }

  const auto artifactPath = node->artifactPath;
  for (auto &candidate : nodes) {
    if (candidate.state != NodeState::frozenGold)
      continue;
    const auto sameGroup =
        artifactPath.empty() ? candidate.id == nodeId
                             : candidate.artifactPath == artifactPath;
    if (!sameGroup)
      continue;
    if (!artifactPath.empty()) {
      candidate.weightsProvenance = WeightsProvenance::file;
      candidate.weightsPath = artifactPath;
    }
    candidate.state = NodeState::liveBlue;
    candidate.colour = colourForType(candidate.type, candidate.state);
    candidate.artifactPath.clear();
    candidate.sourceSubgraph.clear();
    candidate.metrics.reset();
  }
  return true;
}

std::optional<FreezeSelectionRequest> NodeGraph::createFreezeRequest(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  const auto hasBoundary = std::any_of(
      nodes.begin(), nodes.end(),
      [](const GraphNode &node) { return isGroupBoundaryType(node.type); });
  if (hasBoundary) {
    NodeGraph prepared;
    prepared.restoreFromValueTree(toValueTree());
    for (auto &group : prepared.groups)
      group.repeats = 1;
    prepared.flattenGroupBoundaryNodes();
    return prepared.createFreezeRequest(selectedNodeIds);
  }
  if (!selectionIsConnected(selectedNodeIds))
    return std::nullopt;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isGroupBoundaryType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return std::nullopt;
  }

  const std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                                  selectedNodeIds.end());
  FreezeSelectionRequest request;
  request.requestId = juce::Uuid().toString().toStdString();
  request.selectedNodeIds = selectedNodeIds;

  auto root = std::make_unique<juce::DynamicObject>();
  root->setProperty("request_id", juce::String(request.requestId));
  root->setProperty("operation", "freeze_selection");

  juce::Array<juce::var> selectedIds;
  juce::Array<juce::var> elements;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    selectedIds.add(nodeId);

    auto element = std::make_unique<juce::DynamicObject>();
    element->setProperty("id", node->id);
    element->setProperty("type", nodeTypeName(node->type));
    element->setProperty("label", juce::String(node->label));
    element->setProperty("seed", node->seed);
    juce::Array<juce::var> properties;
    for (const auto &property : node->properties) {
      auto serialized = std::make_unique<juce::DynamicObject>();
      serialized->setProperty("key", juce::String(property.key));
      serialized->setProperty("value", property.value);
      if (property.kind == PropertyKind::real)
        serialized->setProperty("float_value", property.floatValue);
      else if (property.kind == PropertyKind::string)
        serialized->setProperty("string_value",
                                juce::String(property.stringValue));
      properties.add(juce::var(serialized.release()));
    }
    element->setProperty("properties", properties);
    elements.add(juce::var(element.release()));
  }
  root->setProperty("selected_element_ids", selectedIds);

  auto fragment = std::make_unique<juce::DynamicObject>();
  fragment->setProperty("elements", elements);
  juce::Array<juce::var> connections;
  juce::Array<juce::var> boundaryInputs;
  juce::Array<juce::var> boundaryOutputs;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto sourceSelected = selected.count(*source) != 0;
    const auto destinationSelected = selected.count(*destination) != 0;
    if (!sourceSelected && !destinationSelected)
      continue;

    auto serialized = std::make_unique<juce::DynamicObject>();
    serialized->setProperty("connection_id", link.id);
    serialized->setProperty("source_element_id", *source);
    serialized->setProperty("source_pin_id", link.sourcePinId);
    serialized->setProperty("destination_element_id", *destination);
    serialized->setProperty("destination_pin_id", link.destinationPinId);
    if (const auto *destinationNode = findNode(*destination))
      serialized->setProperty(
          "destination_pin_index",
          inputPinIndexForId(*destinationNode, link.destinationPinId));
    if (sourceSelected && destinationSelected)
      connections.add(juce::var(serialized.release()));
    else if (destinationSelected)
      boundaryInputs.add(juce::var(serialized.release()));
    else
      boundaryOutputs.add(juce::var(serialized.release()));
  }
  fragment->setProperty("connections", connections);
  auto boundary = std::make_unique<juce::DynamicObject>();
  boundary->setProperty("inputs", boundaryInputs);
  boundary->setProperty("outputs", boundaryOutputs);
  fragment->setProperty("io_boundary", juce::var(boundary.release()));
  root->setProperty("graph_fragment", juce::var(fragment.release()));

  auto options = std::make_unique<juce::DynamicObject>();
  auto sourceIds = selected;
  auto sinkIds = selected;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (source.has_value() && destination.has_value() &&
        selected.count(*source) != 0 && selected.count(*destination) != 0) {
      sourceIds.erase(*destination);
      sinkIds.erase(*source);
    }
  }
  if (sourceIds.size() != 1 || sinkIds.size() != 1)
    return std::nullopt;
  const auto *sourceNode = findNode(*sourceIds.begin());
  const auto *sinkNode = findNode(*sinkIds.begin());
  if (sourceNode == nullptr || sinkNode == nullptr)
    return std::nullopt;
  const auto inputChannels =
      !sourceNode->inputs.empty()
          ? sourceNode->inputs.front().shape.channels
          : (!sourceNode->outputs.empty()
                 ? sourceNode->outputs.front().shape.channels
                 : 0);
  const auto outputChannels =
      !sinkNode->outputs.empty()
          ? sinkNode->outputs.front().shape.channels
          : (!sinkNode->inputs.empty() ? sinkNode->inputs.front().shape.channels
                                       : 0);
  options->setProperty("mode", "manual_freeze");
  options->setProperty("host_input_channels", inputChannels);
  options->setProperty("host_output_channels", outputChannels);
  options->setProperty("example_samples", 256);
  int condDim = 0;
  for (const auto &link : links) {
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!destination.has_value() || selected.count(*destination) == 0)
      continue;
    const auto *destinationPin = findPin(link.destinationPinId);
    if (destinationPin == nullptr || !isControlInputPin(*destinationPin))
      continue;
    const auto *sourcePin = findPin(link.sourcePinId);
    if (sourcePin != nullptr && sourcePin->shape.channels > 0)
      condDim = sourcePin->shape.channels;
  }
  options->setProperty("cond_dim", condDim);
  options->setProperty("conditioning", condDim > 0);
  root->setProperty("compile_options", juce::var(options.release()));
  request.graphFragment =
      juce::JSON::toString(juce::var(root.release()), true).toStdString();
  return request;
}

const std::vector<GraphNode> &NodeGraph::getNodes() const noexcept {
  return nodes;
}

std::vector<GraphNode> &NodeGraph::getNodes() noexcept { return nodes; }

const std::vector<GraphLink> &NodeGraph::getLinks() const noexcept {
  return links;
}

std::vector<GraphLink> &NodeGraph::getLinks() noexcept { return links; }

GraphNode *NodeGraph::findNode(std::int32_t nodeId) noexcept {
  const auto found =
      std::find_if(nodes.begin(), nodes.end(), [nodeId](const GraphNode &node) {
        return node.id == nodeId;
      });
  return found != nodes.end() ? &*found : nullptr;
}

const GraphNode *NodeGraph::findNode(std::int32_t nodeId) const noexcept {
  const auto found =
      std::find_if(nodes.begin(), nodes.end(), [nodeId](const GraphNode &node) {
        return node.id == nodeId;
      });
  return found != nodes.end() ? &*found : nullptr;
}

const Pin *NodeGraph::findPin(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  for (const auto &node : nodes) {
    const auto input =
        std::find_if(node.inputs.begin(), node.inputs.end(),
                     [pinId](const Pin &pin) { return pin.id == pinId; });
    if (input != node.inputs.end())
      return &*input;
    const auto output =
        std::find_if(node.outputs.begin(), node.outputs.end(),
                     [pinId](const Pin &pin) { return pin.id == pinId; });
    if (output != node.outputs.end())
      return &*output;
  }
  return nullptr;
}

const GraphLink *NodeGraph::findLink(std::int32_t linkId) const noexcept {
  const auto found =
      std::find_if(links.begin(), links.end(), [linkId](const GraphLink &link) {
        return link.id == linkId;
      });
  return found != links.end() ? &*found : nullptr;
}

bool NodeGraph::isFixedIoNode(std::int32_t nodeId) const noexcept {
  const auto *node = findNode(nodeId);
  return node != nullptr && isFixedIoType(node->type);
}

bool NodeGraph::isGroupBoundaryNode(std::int32_t nodeId) const noexcept {
  const auto *node = findNode(nodeId);
  return node != nullptr && isGroupBoundaryType(node->type);
}

std::optional<std::int32_t>
NodeGraph::findNodeForPin(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  for (const auto &node : nodes) {
    const auto owns = [pinId](const Pin &pin) { return pin.id == pinId; };
    if (std::any_of(node.inputs.begin(), node.inputs.end(), owns) ||
        std::any_of(node.outputs.begin(), node.outputs.end(), owns))
      return node.id;
  }
  return std::nullopt;
}

ViewportState &NodeGraph::getViewport() noexcept { return viewport; }

const ViewportState &NodeGraph::getViewport() const noexcept {
  return viewport;
}

juce::ValueTree NodeGraph::toValueTree() const {
  juce::ValueTree tree{"GraphDocument"};
  tree.setProperty("version", 3, nullptr);
  tree.setProperty("panX", viewport.pan.x, nullptr);
  tree.setProperty("panY", viewport.pan.y, nullptr);
  tree.setProperty("zoom", viewport.zoom, nullptr);
  tree.setProperty("mapVisible", viewport.mapVisible, nullptr);
  if (viewport.focusedGroupId.has_value())
    tree.setProperty("focusedGroupId", *viewport.focusedGroupId, nullptr);
  if (!viewport.stickySpine.empty()) {
    juce::StringArray spine;
    for (const auto id : viewport.stickySpine)
      spine.add(juce::String(id));
    tree.setProperty("stickySpine", spine.joinIntoString(","), nullptr);
  }
  for (const auto &node : nodes)
    tree.appendChild(nodeToTree(node), nullptr);
  for (const auto &group : groups)
    tree.appendChild(groupToTree(group), nullptr);
  for (const auto &link : links)
    tree.appendChild(linkToTree(link), nullptr);
  return tree;
}

bool NodeGraph::documentIsRestorable(const juce::ValueTree &tree,
                                     juce::String &error) {
  error.clear();
  if (!tree.hasType("GraphDocument")) {
    error = "Preset graph document is missing or corrupt";
    return false;
  }
  for (const auto child : tree) {
    if (!child.hasType("Node"))
      continue;
    const auto typeName = child["type"].toString();
    if (!isKnownPersistedNodeType(typeName)) {
      error = "Preset uses unknown element type '" + typeName + "'";
      return false;
    }
  }
  return true;
}

bool NodeGraph::restoreFromValueTree(const juce::ValueTree &tree) {
  juce::String error;
  if (!documentIsRestorable(tree, error))
    return false;

  std::vector<GraphNode> restoredNodes;
  std::vector<GraphGroup> restoredGroups;
  std::vector<GraphLink> restoredLinks;
  std::int32_t restoredNextNode = 1;
  std::int32_t restoredNextPin = 1001;
  std::int32_t restoredNextLink = 2001;
  for (const auto child : tree) {
    if (child.hasType("Node")) {
      auto node = nodeFromTree(child);
      restoredNextNode = std::max(restoredNextNode, node.id + 1);
      for (const auto &pin : node.inputs)
        restoredNextPin = std::max(restoredNextPin, pin.id + 1);
      for (const auto &pin : node.outputs)
        restoredNextPin = std::max(restoredNextPin, pin.id + 1);
      restoredNodes.push_back(std::move(node));
    } else if (child.hasType("Group")) {
      auto group = groupFromTree(child);
      restoredNextNode = std::max(restoredNextNode, group.id + 1);
      restoredGroups.push_back(std::move(group));
    } else if (child.hasType("Link")) {
      auto link = linkFromTree(child);
      restoredNextLink = std::max(restoredNextLink, link.id + 1);
      restoredLinks.push_back(link);
    }
  }
  nodes = std::move(restoredNodes);
  groups = std::move(restoredGroups);
  links = std::move(restoredLinks);
  nextNodeId = restoredNextNode;
  nextPinId = restoredNextPin;
  nextLinkId = restoredNextLink;
  viewport.pan = {static_cast<float>(tree.getProperty("panX", 0.0f)),
                  static_cast<float>(tree.getProperty("panY", 0.0f))};
  viewport.zoom = std::clamp(static_cast<float>(tree.getProperty("zoom", 1.0f)),
                             minimumZoom, maximumZoom);
  viewport.mapVisible = static_cast<bool>(tree.getProperty("mapVisible", true));
  if (tree.hasProperty("focusedGroupId"))
    viewport.focusedGroupId =
        static_cast<std::int32_t>(tree.getProperty("focusedGroupId"));
  else
    viewport.focusedGroupId.reset();
  if (viewport.focusedGroupId.has_value() &&
      findGroup(*viewport.focusedGroupId) == nullptr)
    viewport.focusedGroupId.reset();
  viewport.stickySpine.clear();
  if (tree.hasProperty("stickySpine")) {
    const auto tokens = juce::StringArray::fromTokens(
        tree.getProperty("stickySpine").toString(), ",", "");
    for (const auto &token : tokens) {
      const auto id = token.getIntValue();
      if (findGroup(id) != nullptr)
        viewport.stickySpine.push_back(id);
    }
  }
  links.erase(std::remove_if(links.begin(), links.end(),
                             [this](const GraphLink &link) {
                               return findPin(link.sourcePinId) == nullptr ||
                                      findPin(link.destinationPinId) == nullptr;
                             }),
              links.end());
  std::vector<std::int32_t> groupsByDepth;
  groupsByDepth.reserve(groups.size());
  for (const auto &group : groups)
    groupsByDepth.push_back(group.id);
  std::sort(groupsByDepth.begin(), groupsByDepth.end(),
            [this](std::int32_t left, std::int32_t right) {
              return groupDepth(*this, left) > groupDepth(*this, right);
            });
  for (const auto groupId : groupsByDepth)
    ensureGroupBoundaryNodes(groupId, true);
  for (auto &node : nodes)
    validateAuthoredRepeatLists(node.id);
  ensureFixedHostIo();
  refreshAllMergeOutputShapes(*this);
  for (auto &node : nodes) {
    if (node.type != NodeType::tcn && node.type != NodeType::blackBox)
      continue;
    bool hasControl = false;
    for (auto &pin : node.inputs) {
      if (isControlInputPin(pin)) {
        pin.label = controlPinLabel;
        pin.shape = flexibleTensorShape();
        hasControl = true;
      }
    }
    if (!hasControl && node.blackBoxOrigin != BlackBoxOrigin::externalLoad)
      node.inputs.push_back({nextPinId++, controlPinLabel, PinKind::input,
                             flexibleTensorShape()});
  }
  return true;
}

juce::ValueTree NodeGraph::exportBox(std::int32_t boxId,
                                     juce::String &error) const {
  error.clear();
  juce::ValueTree tree{"BoxSnapshot"};
  tree.setProperty("version", 1, nullptr);
  tree.setProperty("rootId", boxId, nullptr);
  if (const auto *node = findNode(boxId)) {
    if (isFixedIoType(node->type) || isGroupBoundaryType(node->type)) {
      error = "Audio and group boundary boxes cannot be saved independently";
      return {};
    }
    auto cloned = nodeToTree(*node);
    cloned.removeProperty("parentGroupId", nullptr);
    tree.setProperty("kind", "element", nullptr);
    tree.setProperty("rootTypeHint", nodeTypeName(node->type), nullptr);
    tree.appendChild(cloned, nullptr);
    return tree;
  }
  if (findGroup(boxId) != nullptr) {
    std::unordered_set<std::int32_t> nodeIds;
    std::unordered_set<std::int32_t> groupIds;
    collectGroupSubtree(*this, boxId, nodeIds, groupIds);
    for (const auto nodeId : nodeIds) {
      const auto *member = findNode(nodeId);
      if (member != nullptr && isFixedIoType(member->type)) {
        error =
            "Audio Input and Audio Output cannot be saved to the box library";
        return {};
      }
    }
    tree.setProperty("kind", "group", nullptr);
    tree.setProperty("rootTypeHint", "group", nullptr);
    for (const auto nodeId : nodeIds) {
      if (const auto *member = findNode(nodeId))
        tree.appendChild(nodeToTree(*member), nullptr);
    }
    for (const auto groupId : groupIds) {
      if (const auto *nested = findGroup(groupId)) {
        auto cloned = groupToTree(*nested);
        if (groupId == boxId)
          cloned.removeProperty("parentGroupId", nullptr);
        tree.appendChild(cloned, nullptr);
      }
    }
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (nodeIds.count(*source) == 0 || nodeIds.count(*destination) == 0)
        continue;
      tree.appendChild(linkToTree(link), nullptr);
    }
    return tree;
  }
  error = "Save to Box Library applies to one element or one group";
  return {};
}

/**
 * @brief Collects snapshot node/group ids in the subtree of @p rootId.
 * @param snapshot Box snapshot tree.
 * @param rootId Node or group id inside the snapshot.
 * @param nodeIds Destination node id set.
 * @param groupIds Destination group id set.
 */
void collectSnapshotSubtree(const juce::ValueTree &snapshot, std::int32_t rootId,
                            std::unordered_set<std::int32_t> &nodeIds,
                            std::unordered_set<std::int32_t> &groupIds) {
  std::unordered_map<std::int32_t, juce::ValueTree> groupsById;
  std::unordered_set<std::int32_t> nodesById;
  for (const auto child : snapshot) {
    if (child.hasType("Node"))
      nodesById.insert(static_cast<std::int32_t>(child["id"]));
    else if (child.hasType("Group"))
      groupsById.emplace(static_cast<std::int32_t>(child["id"]), child);
  }
  std::function<void(std::int32_t)> walk = [&](std::int32_t id) {
    if (nodesById.count(id) != 0) {
      nodeIds.insert(id);
      return;
    }
    const auto found = groupsById.find(id);
    if (found == groupsById.end() || !groupIds.insert(id).second)
      return;
    for (const auto member : found->second) {
      if (!member.hasType("Member"))
        continue;
      walk(static_cast<std::int32_t>(member["id"]));
    }
  };
  walk(rootId);
}

std::optional<std::int32_t>
NodeGraph::importBox(const juce::ValueTree &snapshot, juce::Point<float> position,
                     bool collapseGroups, juce::String &error,
                     std::int32_t nestedRootId) {
  error.clear();
  if (!snapshot.hasType("BoxSnapshot")) {
    error = "Box library entry is not a valid snapshot";
    return std::nullopt;
  }
  for (const auto child : snapshot) {
    if (!child.hasType("Node"))
      continue;
    const auto typeName = child["type"].toString();
    if (!isKnownPersistedNodeType(typeName)) {
      error = "Box uses unknown element type '" + typeName + "'";
      return std::nullopt;
    }
    if (isFixedIoType(nodeTypeFromName(typeName))) {
      error = "Audio Input and Audio Output cannot be inserted from the library";
      return std::nullopt;
    }
  }

  const auto backup = toValueTree();
  const auto originalRoot = static_cast<std::int32_t>(snapshot["rootId"]);
  const auto requestedRoot = nestedRootId != 0 ? nestedRootId : originalRoot;
  for (const auto child : snapshot) {
    if (!child.hasType("Node") ||
        static_cast<std::int32_t>(child["id"]) != requestedRoot)
      continue;
    if (isGroupBoundaryType(nodeTypeFromName(child["type"].toString()))) {
      error = "Group Input and Group Output cannot be inserted independently";
      return std::nullopt;
    }
  }
  std::unordered_set<std::int32_t> keepNodes;
  std::unordered_set<std::int32_t> keepGroups;
  collectSnapshotSubtree(snapshot, requestedRoot, keepNodes, keepGroups);
  if (keepNodes.empty() && keepGroups.empty()) {
    error = "Nested library path is missing from this box";
    return std::nullopt;
  }
  std::unordered_map<std::int32_t, std::int32_t> idMap;
  std::unordered_map<std::int32_t, std::int32_t> pinMap;
  std::vector<GraphNode> importedNodes;
  std::vector<GraphGroup> importedGroups;
  std::vector<GraphLink> importedLinks;

  const auto remapId = [this, &idMap](std::int32_t oldId) {
    const auto found = idMap.find(oldId);
    if (found != idMap.end())
      return found->second;
    const auto fresh = nextNodeId++;
    idMap.emplace(oldId, fresh);
    return fresh;
  };

  for (const auto child : snapshot) {
    if (child.hasType("Node")) {
      const auto oldId = static_cast<std::int32_t>(child["id"]);
      if (keepNodes.count(oldId) == 0)
        continue;
      auto node = nodeFromTree(child);
      node.id = remapId(node.id);
      if (oldId == requestedRoot)
        node.parentGroupId.reset();
      else if (node.parentGroupId.has_value()) {
        if (keepGroups.count(*node.parentGroupId) != 0)
          node.parentGroupId = remapId(*node.parentGroupId);
        else
          node.parentGroupId.reset();
      }
      for (auto &pin : node.inputs) {
        const auto fresh = nextPinId++;
        pinMap.emplace(pin.id, fresh);
        pin.id = fresh;
      }
      for (auto &pin : node.outputs) {
        const auto fresh = nextPinId++;
        pinMap.emplace(pin.id, fresh);
        pin.id = fresh;
      }
      importedNodes.push_back(std::move(node));
    } else if (child.hasType("Group")) {
      const auto oldId = static_cast<std::int32_t>(child["id"]);
      if (keepGroups.count(oldId) == 0)
        continue;
      auto group = groupFromTree(child);
      group.id = remapId(group.id);
      if (oldId == requestedRoot)
        group.parentGroupId.reset();
      else if (group.parentGroupId.has_value()) {
        if (keepGroups.count(*group.parentGroupId) != 0)
          group.parentGroupId = remapId(*group.parentGroupId);
        else
          group.parentGroupId.reset();
      }
      if (collapseGroups)
        group.collapsed = true;
      importedGroups.push_back(std::move(group));
    }
  }

  for (auto &group : importedGroups) {
    group.memberIds.erase(
        std::remove_if(group.memberIds.begin(), group.memberIds.end(),
                       [&idMap](std::int32_t member) {
                         return idMap.find(member) == idMap.end();
                       }),
        group.memberIds.end());
    for (auto &member : group.memberIds)
      member = idMap[member];
  }

  for (const auto child : snapshot) {
    if (!child.hasType("Link"))
      continue;
    auto link = linkFromTree(child);
    const auto source = pinMap.find(link.sourcePinId);
    const auto destination = pinMap.find(link.destinationPinId);
    if (source == pinMap.end() || destination == pinMap.end()) {
      if (nestedRootId != 0)
        continue;
      error = "Box snapshot has a connection to a missing pin";
      restoreFromValueTree(backup);
      return std::nullopt;
    }
    link.id = nextLinkId++;
    link.sourcePinId = source->second;
    link.destinationPinId = destination->second;
    importedLinks.push_back(link);
  }

  const auto rootFound = idMap.find(requestedRoot);
  if (rootFound == idMap.end()) {
    error = "Box snapshot is missing its root box";
    restoreFromValueTree(backup);
    return std::nullopt;
  }
  const auto newRoot = rootFound->second;

  std::vector<std::int32_t> importedGroupIds;
  importedGroupIds.reserve(importedGroups.size());
  for (const auto &group : importedGroups)
    importedGroupIds.push_back(group.id);
  for (auto &node : importedNodes) {
    if (node.id == newRoot)
      node.position = position;
    nodes.push_back(std::move(node));
  }
  for (auto &group : importedGroups) {
    if (group.id == newRoot)
      group.position = position;
    groups.push_back(std::move(group));
  }
  for (const auto &link : importedLinks)
    links.push_back(link);

  std::sort(importedGroupIds.begin(), importedGroupIds.end(),
            [this](std::int32_t left, std::int32_t right) {
              return groupDepth(*this, left) > groupDepth(*this, right);
            });
  for (const auto groupId : importedGroupIds)
    ensureGroupBoundaryNodes(groupId, true);
  refreshAllMergeOutputShapes(*this);
  return newRoot;
}

std::string NodeGraph::toJson() const {
  auto root = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> nodeArray;
  for (const auto &node : nodes) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", node.id);
    object->setProperty("type", nodeTypeName(node.type));
    object->setProperty("label", juce::String(node.label));
    object->setProperty("x", node.position.x);
    object->setProperty("y", node.position.y);
    object->setProperty("seed", node.seed);
    object->setProperty("state", node.state == NodeState::frozenGold
                                     ? "frozen_gold"
                                     : "live_blue");
    juce::Array<juce::var> inputs;
    for (const auto &pin : node.inputs) {
      auto endpoint = std::make_unique<juce::DynamicObject>();
      endpoint->setProperty("id", pin.id);
      endpoint->setProperty("channels", pin.shape.channels);
      inputs.add(juce::var(endpoint.release()));
    }
    object->setProperty("inputs", inputs);
    juce::Array<juce::var> outputs;
    for (const auto &pin : node.outputs) {
      auto endpoint = std::make_unique<juce::DynamicObject>();
      endpoint->setProperty("id", pin.id);
      endpoint->setProperty("channels", pin.shape.channels);
      outputs.add(juce::var(endpoint.release()));
    }
    object->setProperty("outputs", outputs);
    juce::Array<juce::var> properties;
    for (const auto &property : node.properties) {
      auto value = std::make_unique<juce::DynamicObject>();
      value->setProperty("key", juce::String(property.key));
      value->setProperty("value", property.value);
      if (property.kind == PropertyKind::real)
        value->setProperty("float_value", property.floatValue);
      else if (property.kind == PropertyKind::string)
        value->setProperty("string_value", juce::String(property.stringValue));
      properties.add(juce::var(value.release()));
    }
    object->setProperty("properties", properties);
    nodeArray.add(juce::var(object.release()));
  }
  juce::Array<juce::var> linkArray;
  for (const auto &link : links) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", link.id);
    object->setProperty("source_pin", link.sourcePinId);
    object->setProperty("destination_pin", link.destinationPinId);
    linkArray.add(juce::var(object.release()));
  }
  root->setProperty("elements", nodeArray);
  root->setProperty("connections", linkArray);
  return juce::JSON::toString(juce::var(root.release()), true).toStdString();
}

GraphNode NodeGraph::makeNode(NodeType type, juce::Point<float> position) {
  if (type == NodeType::rateConv)
    type = NodeType::convolution;
  GraphNode node;
  node.id = nextNodeId++;
  node.type = type;
  node.position = position;
  node.state =
      type == NodeType::blackBox ? NodeState::frozenGold : NodeState::liveBlue;
  node.colour = colourForType(type, node.state);

  const auto addInput = [&](const char *label = "in", int channels = 0) {
    node.inputs.push_back(
        {nextPinId++, label, PinKind::input, {channels}});
  };
  const auto addOutput = [&](const char *label = "out", int channels = 0) {
    node.outputs.push_back(
        {nextPinId++, label, PinKind::output, {channels}});
  };
  const auto property = [](std::string key, std::string label, int value,
                           int minimum, int maximum,
                           PropertyKind kind = PropertyKind::integer,
                           std::vector<std::string> choices = {}) {
    return NodeProperty{std::move(key),    std::move(label), value,
                        minimum,           maximum,          kind,
                        std::move(choices)};
  };
  const auto gainProperty = []() {
    NodeProperty gain;
    gain.key = "gain";
    gain.label = "Gain";
    gain.kind = PropertyKind::real;
    gain.floatValue = gainDefault;
    gain.floatMinimum = gainMinimum;
    gain.floatMaximum = gainMaximum;
    return gain;
  };
  const auto fidelityProperty = []() {
    NodeProperty fidelity;
    fidelity.key = "fidelity";
    fidelity.label = "Fidelity";
    fidelity.kind = PropertyKind::real;
    fidelity.floatValue = defaultFidelityPercent;
    fidelity.floatMinimum = fidelityMinimum;
    fidelity.floatMaximum = fidelityMaximum;
    return fidelity;
  };

  switch (type) {
  case NodeType::audioInput:
    node.label = "Audio Input";
    node.detail = hostIoModeDetail(HostIoMode::stereo, true);
    addOutput("out", 2);
    node.properties.push_back(property("channels", "Mode", 2, 0, 2,
                                       PropertyKind::choice,
                                       {"Mono", "Mirrored", "Stereo"}));
    applyHostIoChannels(node);
    break;
  case NodeType::audioOutput:
    node.label = "Audio Output";
    node.detail = hostIoModeDetail(HostIoMode::stereo, false);
    addInput("in", 2);
    node.properties.push_back(property("channels", "Mode", 2, 0, 2,
                                       PropertyKind::choice,
                                       {"Mono", "Mirrored", "Stereo"}));
    applyHostIoChannels(node);
    break;
  case NodeType::groupInput:
    node.label = "Group Input";
    node.detail = "Declared group inputs";
    addInput("in 1");
    addOutput("out 1");
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("ports", "Inputs", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::groupOutput:
    node.label = "Group Output";
    node.detail = "Declared group outputs";
    addInput("in 1");
    addOutput("out 1");
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("ports", "Outputs", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::linear:
    node.label = "Linear";
    node.detail = "Weighted projection";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("features", "Features", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::convolution:
  case NodeType::rateConv:
    node.label = "Conv1D";
    node.detail = "Temporal convolution";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("channels", "Channels", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation", "Dilation", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("stride", "Stride", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("weight_norm", "Weight Norm", 0, 0, 1));
    break;
  case NodeType::convTranspose:
    node.label = "ConvTranspose1d";
    node.detail = "Temporal upsampling";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("channels", "Channels", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation", "Dilation", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("stride", "Stride", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("weight_norm", "Weight Norm", 0, 0, 1));
    break;
  case NodeType::batchNorm:
    node.label = "BatchNorm1d";
    node.detail = "Affine normalization";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    break;
  case NodeType::activation:
    node.label = "Activation";
    node.detail = "ReLU";
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("activation", "Function", 0, 0, 4, PropertyKind::choice,
                 {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}));
    node.properties.push_back(gainProperty());
    {
      NodeProperty slope;
      slope.key = "negative_slope";
      slope.label = "Negative slope";
      slope.kind = PropertyKind::real;
      slope.floatValue = leakyReluNegativeSlopeDefault;
      slope.floatMinimum = leakyReluNegativeSlopeMinimum;
      slope.floatMaximum = leakyReluNegativeSlopeMaximum;
      node.properties.push_back(std::move(slope));
    }
    break;
  case NodeType::tcn:
    node.label = "TCN";
    node.detail = "Live modular network";
    node.hasWeights = true;
    node.armedForTraining = true;
    node.dilationGrowth = defaultDilationGrowth;
    addInput();
    addInput(controlPinLabel);
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.inputs.back().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("depth", "Depth", 4, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("channels", "Channels", 16,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation_growth", "Dilation growth",
                                       defaultDilationGrowth,
                                       minimumDilationGrowth,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("residual", "Residual", 0, 0, 1));
    node.properties.push_back(
        property("activation", "Activation", 0, 0, 4, PropertyKind::choice,
                 {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}));
    node.properties.push_back(gainProperty());
    {
      NodeProperty slope;
      slope.key = "negative_slope";
      slope.label = "Negative slope";
      slope.kind = PropertyKind::real;
      slope.floatValue = leakyReluNegativeSlopeDefault;
      slope.floatMinimum = leakyReluNegativeSlopeMinimum;
      slope.floatMaximum = leakyReluNegativeSlopeMaximum;
      node.properties.push_back(std::move(slope));
    }
    break;
  case NodeType::merge:
    node.label = "Utility";
    node.detail = "Elementwise combine";
    addOutput();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("mode", "Mode", 0, 0, 2, PropertyKind::choice,
                 {"Add", "Multiply", "Concatenate"}));
    node.properties.push_back(property("inputs", "Inputs", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    setMixerInputCount(node, 2);
    break;
  case NodeType::knobInput:
    node.label = "Knob Input";
    node.detail = "1D conditioning";
    addOutput("c", 1);
    break;
  case NodeType::xyTrackpad:
    node.label = "XY Trackpad";
    node.detail = "Independent X and Y outputs";
    addOutput("x", 1);
    addOutput("y", 1);
    break;
  case NodeType::blackBox:
    node.label = "Frozen Selection";
    node.detail = "Locked";
    node.hasWeights = true;
    addInput();
    addInput(controlPinLabel);
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.inputs.back().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(fidelityProperty());
    break;
  case NodeType::pqmfAnalysis:
    node.label = "PQMF Analysis";
    node.detail = "Audio to multiband";
    addInput("in", 0);
    addOutput("bands", 0);
    node.outputs.front().shape.temporalRate = defaultPqmfBands;
    node.outputs.front().shape.nBand = defaultPqmfBands;
    node.properties.push_back(property("n_band", "nBand", defaultPqmfBands,
                                       minimumPqmfBands,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::pqmfSynthesis:
    node.label = "PQMF Synthesis";
    node.detail = "Multiband to audio";
    addInput("bands", 0);
    addOutput("out", 0);
    node.inputs.front().shape.temporalRate = defaultPqmfBands;
    node.inputs.front().shape.nBand = defaultPqmfBands;
    node.properties.push_back(property("n_band", "nBand", defaultPqmfBands,
                                       minimumPqmfBands,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::variationalBottleneck:
    node.label = "Variational Bottleneck";
    node.detail = "Latent sample";
    node.hasWeights = true;
    node.armedForTraining = true;
    node.fidelityPercent = defaultFidelityPercent;
    addInput("features", 0);
    addOutput("z", defaultLatentSize);
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape.temporalRate = 0;
    node.properties.push_back(property("latent_size", "Latent", defaultLatentSize,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size",
                                       defaultBottleneckKernelSize,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(fidelityProperty());
    break;
  case NodeType::noiseSynthesizer:
    node.label = "Noise Synth";
    node.detail = "IR × white noise";
    node.armedForTraining = false;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("noise_bands", "Noise bands", defaultNoiseBands, 2,
                 unlimitedPropertyMaximum));
    node.properties.push_back(
        property("window_size", "Window Size", defaultNoiseWindowSize,
                 minimumPositiveProperty, unlimitedPropertyMaximum));
    break;
  case NodeType::mathExpression:
    node.label = "Math Expression";
    node.detail = "x1";
    addOutput();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("inputs", "Inputs", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    {
      NodeProperty expression;
      expression.key = "expression";
      expression.label = "Expression";
      expression.kind = PropertyKind::string;
      expression.stringValue = "x1";
      node.properties.push_back(std::move(expression));
    }
    setMixerInputCount(node, 1, true);
    break;
  case NodeType::reverb: {
    node.label = "Reverb";
    node.detail = "Convolutional IR";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addInput(irPinLabel);
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.inputs.back().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("reverb_length", "Reverb Length",
                                       defaultReverbLength, minimumReverbLength,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("add_dry", "Add Dry", 1, 0, 1));
    {
      NodeProperty blend;
      blend.key = "ir_blend";
      blend.label = "IR Blend";
      blend.kind = PropertyKind::real;
      blend.floatValue = 0.0f;
      blend.floatMinimum = 0.0f;
      blend.floatMaximum = 1.0f;
      node.properties.push_back(std::move(blend));
    }
    break;
  }
  case NodeType::expDecayReverb: {
    node.label = "ExpDecayReverb";
    node.detail = "Exponential decay IR";
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    {
      NodeProperty gain;
      gain.key = "gain";
      gain.label = "IR Gain";
      gain.kind = PropertyKind::real;
      gain.floatValue = defaultExpDecayGain;
      gain.floatMinimum = -10.0f;
      gain.floatMaximum = 10.0f;
      node.properties.push_back(std::move(gain));
    }
    {
      NodeProperty decay;
      decay.key = "decay";
      decay.label = "Decay";
      decay.kind = PropertyKind::real;
      decay.floatValue = defaultExpDecayDecay;
      decay.floatMinimum = -10.0f;
      decay.floatMaximum = 10.0f;
      node.properties.push_back(std::move(decay));
    }
    node.properties.push_back(property("reverb_length", "Reverb Length",
                                       defaultReverbLength, minimumReverbLength,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("add_dry", "Add Dry", 1, 0, 1));
    break;
  }
  case NodeType::filteredNoiseReverb: {
    node.label = "FilteredNoiseReverb";
    node.detail = "Filtered-noise IR";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("n_frames", "Time Steps",
                                       defaultFilterFrames, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("n_filter_banks", "Filter Banks",
                                       defaultFilterBanks, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("window_size", "Window Size",
                                       defaultFirWindowSize, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("reverb_length", "Reverb Length",
                                       defaultReverbLength, minimumReverbLength,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("add_dry", "Add Dry", 1, 0, 1));
    break;
  }
  case NodeType::firFilter: {
    node.label = "FIRFilter";
    node.detail = "LTV-FIR";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("n_frames", "Time Steps",
                                       defaultFilterFrames, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("n_filter_banks", "Filter Banks",
                                       defaultFilterBanks, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("window_size", "Window Size",
                                       defaultFirWindowSize, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    break;
  }
  case NodeType::modDelay: {
    node.label = "ModDelay";
    node.detail = "Modulated delay";
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    {
      NodeProperty center;
      center.key = "center_ms";
      center.label = "Center (ms)";
      center.kind = PropertyKind::real;
      center.floatValue = defaultModDelayCenterMs;
      center.floatMinimum = 0.0f;
      center.floatMaximum = 1000.0f;
      node.properties.push_back(std::move(center));
    }
    {
      NodeProperty depth;
      depth.key = "depth_ms";
      depth.label = "Depth (ms)";
      depth.kind = PropertyKind::real;
      depth.floatValue = defaultModDelayDepthMs;
      depth.floatMinimum = 0.0f;
      depth.floatMaximum = 1000.0f;
      node.properties.push_back(std::move(depth));
    }
    {
      NodeProperty gain;
      gain.key = "gain";
      gain.label = "Wet Gain";
      gain.kind = PropertyKind::real;
      gain.floatValue = defaultExpDecayGain;
      gain.floatMinimum = -10.0f;
      gain.floatMaximum = 10.0f;
      node.properties.push_back(std::move(gain));
    }
    {
      NodeProperty phase;
      phase.key = "phase";
      phase.label = "Phase";
      phase.kind = PropertyKind::real;
      phase.floatValue = defaultModDelayPhase;
      phase.floatMinimum = 0.0f;
      phase.floatMaximum = 1.0f;
      node.properties.push_back(std::move(phase));
    }
    node.properties.push_back(property("add_dry", "Add Dry", 1, 0, 1));
    break;
  }
  case NodeType::lstm:
  case NodeType::rnn: {
    node.label = type == NodeType::lstm ? "LSTM" : "RNN";
    node.detail = "Single recurrent layer";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape.channels = defaultHiddenSize;
    node.properties.push_back(property("hidden_size", "Hidden Size",
                                       defaultHiddenSize, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("bidirectional", "Bidirectional", 0, 0, 1));
    node.properties.push_back(property("bias", "Bias", 1, 0, 1));
    node.properties.push_back(
        property("activation", "Activation", tanhActivationIndex, 0, 4,
                 PropertyKind::choice,
                 {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}));
    node.properties.push_back(gainProperty());
    {
      NodeProperty slope;
      slope.key = "negative_slope";
      slope.label = "Negative slope";
      slope.kind = PropertyKind::real;
      slope.floatValue = leakyReluNegativeSlopeDefault;
      slope.floatMinimum = leakyReluNegativeSlopeMinimum;
      slope.floatMaximum = leakyReluNegativeSlopeMaximum;
      node.properties.push_back(std::move(slope));
    }
    {
      NodeProperty leak;
      leak.key = "leak_rate";
      leak.label = "Leak Rate";
      leak.kind = PropertyKind::real;
      leak.floatValue = leakRateDefault;
      leak.floatMinimum = leakRateMinimum;
      leak.floatMaximum = leakRateMaximum;
      node.properties.push_back(std::move(leak));
    }
    {
      NodeProperty scale;
      scale.key = "recurrent_weight_scale";
      scale.label = "Recurrent Weight Scale";
      scale.kind = PropertyKind::real;
      scale.floatValue = recurrentWeightScaleDefault;
      scale.floatMinimum = recurrentWeightScaleMinimum;
      scale.floatMaximum = recurrentWeightScaleMaximum;
      node.properties.push_back(std::move(scale));
    }
    break;
  }
  }
  return node;
}

GraphNode NodeGraph::makeExternalTorchScriptLoadNode(juce::Point<float> position) {
  GraphNode node;
  node.id = nextNodeId++;
  node.type = NodeType::blackBox;
  node.position = position;
  node.state = NodeState::frozenGold;
  node.colour = colourForType(node.type, node.state);
  node.label = "TorchScript Load";
  node.detail = "Choose a checkpoint";
  node.hasWeights = false;
  node.armedForTraining = false;
  node.blackBoxOrigin = BlackBoxOrigin::externalLoad;
  node.externalLoadStatus = ExternalLoadStatus::empty;
  node.inputs.push_back(
      {nextPinId++, "in", PinKind::input, flexibleTensorShape()});
  node.outputs.push_back(
      {nextPinId++, "out", PinKind::output, flexibleTensorShape()});
  return node;
}

void NodeGraph::applyExternalLoadSurface(GraphNode &node) {
  if (!isExternalLoadNode(node))
    return;

  std::int32_t audioInId = 0;
  std::int32_t audioOutId = 0;
  std::int32_t controlId = 0;
  std::int32_t latentInId = 0;
  std::int32_t latentOutId = 0;
  for (const auto &pin : node.inputs) {
    if (isControlInputPin(pin))
      controlId = pin.id;
    else if (isLatentPin(pin))
      latentInId = pin.id;
    else if (audioInId == 0)
      audioInId = pin.id;
  }
  for (const auto &pin : node.outputs) {
    if (isLatentPin(pin))
      latentOutId = pin.id;
    else if (audioOutId == 0)
      audioOutId = pin.id;
  }
  if (audioInId == 0) {
    audioInId = nextPinId++;
    node.inputs.insert(node.inputs.begin(),
                       {audioInId, "in", PinKind::input, flexibleTensorShape()});
  }
  if (audioOutId == 0) {
    audioOutId = nextPinId++;
    node.outputs.insert(
        node.outputs.begin(),
        {audioOutId, "out", PinKind::output, flexibleTensorShape()});
  }

  const bool wantControl =
      node.externalLoadStatus == ExternalLoadStatus::ready &&
      node.externalAcceptsConditioning;
  const bool wantLatent = node.externalLoadStatus == ExternalLoadStatus::ready &&
                          node.externalHasEncodeDecode;

  std::vector<std::int32_t> removed;
  if (!wantControl && controlId != 0)
    removed.push_back(controlId);
  if (!wantLatent) {
    if (latentInId != 0)
      removed.push_back(latentInId);
    if (latentOutId != 0)
      removed.push_back(latentOutId);
  }
  if (!removed.empty()) {
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&removed](const GraphLink &link) {
                                 return std::find(removed.begin(), removed.end(),
                                                  link.sourcePinId) !=
                                            removed.end() ||
                                        std::find(removed.begin(), removed.end(),
                                                  link.destinationPinId) !=
                                            removed.end();
                               }),
                links.end());
  }

  std::vector<Pin> inputs;
  std::vector<Pin> outputs;
  for (auto &pin : node.inputs) {
    if (pin.id == audioInId)
      inputs.push_back(pin);
  }
  if (wantControl) {
    if (controlId == 0)
      controlId = nextPinId++;
    Pin control{controlId, controlPinLabel, PinKind::input,
                flexibleTensorShape()};
    inputs.push_back(control);
  }
  if (wantLatent) {
    if (latentInId == 0)
      latentInId = nextPinId++;
    const auto latentWidth = std::max(0, effectiveLatentChannels(node));
    Pin latentIn{latentInId, latentPinLabel, PinKind::input,
                 flexibleTensorShape(latentWidth)};
    inputs.push_back(latentIn);
  }
  for (auto &pin : node.outputs) {
    if (pin.id == audioOutId)
      outputs.push_back(pin);
  }
  if (wantLatent) {
    if (latentOutId == 0)
      latentOutId = nextPinId++;
    const auto latentWidth = std::max(0, effectiveLatentChannels(node));
    Pin latentOut{latentOutId, latentPinLabel, PinKind::output,
                  flexibleTensorShape(latentWidth)};
    outputs.push_back(latentOut);
  }
  node.inputs = std::move(inputs);
  node.outputs = std::move(outputs);

  const bool empty = node.externalLoadStatus == ExternalLoadStatus::empty;
  const int inCh = empty || node.externalShapeIncomplete
                       ? 0
                       : std::max(0, effectiveInputChannels(node));
  const int outCh = empty || node.externalShapeIncomplete
                        ? 0
                        : std::max(0, effectiveOutputChannels(node));
  for (auto &pin : node.inputs) {
    if (isControlInputPin(pin))
      pin.shape = flexibleTensorShape();
    else if (isLatentPin(pin))
      pin.shape.channels =
          empty || node.externalShapeIncomplete
              ? 0
              : std::max(0, effectiveLatentChannels(node));
    else
      pin.shape.channels = inCh;
  }
  for (auto &pin : node.outputs) {
    if (isLatentPin(pin))
      pin.shape.channels =
          empty || node.externalShapeIncomplete
              ? 0
              : std::max(0, effectiveLatentChannels(node));
    else
      pin.shape.channels = outCh;
  }

  node.properties.erase(
      std::remove_if(node.properties.begin(), node.properties.end(),
                     [](const NodeProperty &property) {
                       return property.key == "fidelity";
                     }),
      node.properties.end());
  if (wantLatent) {
    NodeProperty fidelity;
    fidelity.key = "fidelity";
    fidelity.label = "Fidelity";
    fidelity.kind = PropertyKind::real;
    fidelity.floatValue = clampFidelity(node.fidelityPercent);
    fidelity.floatMinimum = fidelityMinimum;
    fidelity.floatMaximum = fidelityMaximum;
    node.properties.push_back(std::move(fidelity));
  }
}

bool NodeGraph::beginExternalCheckpointLoad(std::int32_t nodeId,
                                            const std::string &path) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node))
    return false;
  node->artifactPath = path;
  node->weightsPath = path;
  node->weightsProvenance = WeightsProvenance::file;
  node->externalLoadStatus = ExternalLoadStatus::loading;
  node->externalLoadErrorMessage.clear();
  node->detail = "Loading…";
  return true;
}

bool NodeGraph::applyExternalCheckpointReady(
    std::int32_t nodeId, const std::string &path, int inferredIn,
    int inferredOut, int inferredLatent, bool hasEncodeDecode,
    bool acceptsConditioning, bool compactnessReadyFlag,
    std::vector<float> mean, std::vector<float> pca,
    std::vector<float> cumulative, const std::string &rateWarning) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node))
    return false;
  node->artifactPath = path;
  node->weightsPath = path;
  node->runtimeArtifactPath = path;
  node->weightsProvenance = WeightsProvenance::file;
  node->externalLoadStatus = ExternalLoadStatus::ready;
  node->externalLoadErrorMessage.clear();
  node->inferredInputChannels = std::max(0, inferredIn);
  node->inferredOutputChannels = std::max(0, inferredOut);
  node->inferredLatentChannels = std::max(0, inferredLatent);
  if (node->overrideInputChannels < 1 && node->inferredInputChannels > 0)
    node->overrideInputChannels = node->inferredInputChannels;
  if (node->overrideOutputChannels < 1 && node->inferredOutputChannels > 0)
    node->overrideOutputChannels = node->inferredOutputChannels;
  if (hasEncodeDecode && node->overrideLatentChannels < 1 &&
      node->inferredLatentChannels > 0)
    node->overrideLatentChannels = node->inferredLatentChannels;
  node->externalHasEncodeDecode = hasEncodeDecode;
  node->externalAcceptsConditioning = acceptsConditioning;
  node->compactnessReady = compactnessReadyFlag && hasEncodeDecode;
  node->latentMean = std::move(mean);
  node->latentPca = std::move(pca);
  node->cumulativeVariance = std::move(cumulative);
  node->sampleRateWarning = rateWarning;
  node->externalShapeIncomplete =
      effectiveInputChannels(*node) < 1 ||
      effectiveOutputChannels(*node) < 1 ||
      (hasEncodeDecode && effectiveLatentChannels(*node) < 1);
  const auto stem = [&path]() {
    const auto slash = path.find_last_of("/\\");
    auto name = slash == std::string::npos ? path : path.substr(slash + 1);
    const auto dot = name.find_last_of('.');
    if (dot != std::string::npos && dot > 0)
      name.resize(dot);
    return name;
  }();
  node->label = stem.empty() ? "TorchScript Load" : stem;
  node->detail = node->externalShapeIncomplete ? "Enter channel overrides"
                                               : "Locked";
  applyExternalLoadSurface(*node);
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::applyExternalCheckpointError(std::int32_t nodeId,
                                            const std::string &message) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node))
    return false;
  node->externalLoadStatus = ExternalLoadStatus::error;
  node->externalLoadErrorMessage =
      message.empty() ? "Checkpoint could not be loaded" : message;
  node->detail = "Load failed";
  if (node->runtimeArtifactPath.empty()) {
    node->externalHasEncodeDecode = false;
    node->externalAcceptsConditioning = false;
    node->compactnessReady = false;
    node->latentMean.clear();
    node->latentPca.clear();
    node->cumulativeVariance.clear();
    applyExternalLoadSurface(*node);
  }
  return true;
}

bool NodeGraph::clearExternalCheckpoint(std::int32_t nodeId) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node))
    return false;
  node->artifactPath.clear();
  node->weightsPath.clear();
  node->runtimeArtifactPath.clear();
  node->externalLoadStatus = ExternalLoadStatus::empty;
  node->externalLoadErrorMessage.clear();
  node->inferredInputChannels = 0;
  node->inferredOutputChannels = 0;
  node->inferredLatentChannels = 0;
  node->overrideInputChannels = -1;
  node->overrideOutputChannels = -1;
  node->overrideLatentChannels = -1;
  node->externalHasEncodeDecode = false;
  node->externalAcceptsConditioning = false;
  node->externalShapeIncomplete = false;
  node->sampleRateWarning.clear();
  node->compactnessReady = false;
  node->latentMean.clear();
  node->latentPca.clear();
  node->cumulativeVariance.clear();
  node->fidelityPercent = defaultFidelityPercent;
  node->label = "TorchScript Load";
  node->detail = "Choose a checkpoint";
  node->metrics.reset();
  applyExternalLoadSurface(*node);
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::setExternalChannelOverride(std::int32_t nodeId, const char *which,
                                          int value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node) || which == nullptr)
    return false;
  int *target = nullptr;
  if (std::strcmp(which, "input") == 0)
    target = &node->overrideInputChannels;
  else if (std::strcmp(which, "output") == 0)
    target = &node->overrideOutputChannels;
  else if (std::strcmp(which, "latent") == 0)
    target = &node->overrideLatentChannels;
  if (target == nullptr)
    return false;
  const auto previous = *target;
  *target = value < 1 ? -1 : value;
  const auto previousIncomplete = node->externalShapeIncomplete;
  node->externalShapeIncomplete =
      effectiveInputChannels(*node) < 1 || effectiveOutputChannels(*node) < 1 ||
      (node->externalHasEncodeDecode && effectiveLatentChannels(*node) < 1);
  applyExternalLoadSurface(*node);
  refreshPropagatedPinShapes(*this);
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    *target = previous;
    node->externalShapeIncomplete = previousIncomplete;
    applyExternalLoadSurface(*node);
    refreshPropagatedPinShapes(*this);
    lastPropertyError = incompatible;
    return false;
  }
  lastPropertyError.clear();
  return true;
}

bool NodeGraph::resetExternalChannelOverrides(std::int32_t nodeId) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !isExternalLoadNode(*node))
    return false;
  node->overrideInputChannels = -1;
  node->overrideOutputChannels = -1;
  node->overrideLatentChannels = -1;
  node->externalShapeIncomplete =
      effectiveInputChannels(*node) < 1 || effectiveOutputChannels(*node) < 1 ||
      (node->externalHasEncodeDecode && effectiveLatentChannels(*node) < 1);
  applyExternalLoadSurface(*node);
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::canUnfreeze(std::int32_t nodeId) const noexcept {
  const auto *node = findNode(nodeId);
  if (node == nullptr || node->state != NodeState::frozenGold)
    return false;
  return !isExternalLoadNode(*node);
}

void NodeGraph::setMixerInputCount(GraphNode &node, int inputCount,
                                   bool mathLabels) {
  const auto count = std::max(inputCount, 1);
  while (static_cast<int>(node.inputs.size()) > count) {
    const auto pinId = node.inputs.back().id;
    links.erase(std::remove_if(links.begin(), links.end(),
                               [pinId](const GraphLink &link) {
                                 return link.sourcePinId == pinId ||
                                        link.destinationPinId == pinId;
                               }),
                links.end());
    node.inputs.pop_back();
  }
  while (static_cast<int>(node.inputs.size()) < count) {
    const auto index = static_cast<int>(node.inputs.size()) + 1;
    const auto label = mathLabels ? "x" + std::to_string(index)
                                  : "in " + std::to_string(index);
    node.inputs.push_back(
        {nextPinId++, label, PinKind::input, flexibleTensorShape()});
  }
  for (int index = 0; index < static_cast<int>(node.inputs.size()); ++index)
    node.inputs[static_cast<std::size_t>(index)].label =
        mathLabels ? "x" + std::to_string(index + 1)
                   : "in " + std::to_string(index + 1);
}

bool NodeGraph::setGroupBoundaryPortCount(GraphNode &node, int portCount) {
  if (!isGroupBoundaryType(node.type))
    return false;
  const auto count = std::max(1, portCount);
  const auto oldCount =
      static_cast<int>(std::min(node.inputs.size(), node.outputs.size()));
  if (count < oldCount) {
    for (int index = count; index < oldCount; ++index) {
      const auto inputId = node.inputs[static_cast<std::size_t>(index)].id;
      const auto outputId = node.outputs[static_cast<std::size_t>(index)].id;
      if (isPinConnected(inputId) || isPinConnected(outputId))
        return false;
    }
  }
  while (static_cast<int>(node.inputs.size()) > count)
    node.inputs.pop_back();
  while (static_cast<int>(node.outputs.size()) > count)
    node.outputs.pop_back();
  while (static_cast<int>(node.inputs.size()) < count) {
    const auto index = static_cast<int>(node.inputs.size()) + 1;
    node.inputs.push_back({nextPinId++, "in " + std::to_string(index),
                           PinKind::input, flexibleTensorShape()});
  }
  while (static_cast<int>(node.outputs.size()) < count) {
    const auto index = static_cast<int>(node.outputs.size()) + 1;
    node.outputs.push_back({nextPinId++, "out " + std::to_string(index),
                            PinKind::output, flexibleTensorShape()});
  }
  for (auto &property : node.properties) {
    if (property.key == "ports")
      property.setValue(count);
  }
  return true;
}

void NodeGraph::spliceGroupBoundaryNode(std::int32_t nodeId) {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !isGroupBoundaryType(node->type))
    return;
  const auto parent = node->parentGroupId;
  const auto inputs = node->inputs;
  const auto outputs = node->outputs;
  std::unordered_set<std::int32_t> incidentPins;
  for (const auto &pin : inputs)
    incidentPins.insert(pin.id);
  for (const auto &pin : outputs)
    incidentPins.insert(pin.id);

  std::vector<GraphLink> bypasses;
  const auto laneCount = std::min(inputs.size(), outputs.size());
  for (std::size_t index = 0; index < laneCount; ++index) {
    std::vector<std::int32_t> sources;
    std::vector<std::int32_t> destinations;
    for (const auto &link : links) {
      if (link.destinationPinId == inputs[index].id)
        sources.push_back(link.sourcePinId);
      if (link.sourcePinId == outputs[index].id)
        destinations.push_back(link.destinationPinId);
    }
    for (const auto sourcePin : sources) {
      for (const auto destinationPin : destinations) {
        if (sourcePin == destinationPin)
          continue;
        const auto duplicate = std::any_of(
            links.begin(), links.end(),
            [sourcePin, destinationPin](const GraphLink &link) {
              return link.sourcePinId == sourcePin &&
                     link.destinationPinId == destinationPin;
            });
        if (!duplicate)
          bypasses.push_back({nextLinkId++, sourcePin, destinationPin});
      }
    }
  }
  links.erase(std::remove_if(links.begin(), links.end(),
                             [&incidentPins](const GraphLink &link) {
                               return incidentPins.count(link.sourcePinId) != 0 ||
                                      incidentPins.count(link.destinationPinId) !=
                                          0;
                             }),
              links.end());
  links.insert(links.end(), bypasses.begin(), bypasses.end());
  if (parent.has_value()) {
    if (auto *group = findGroup(*parent)) {
      group->memberIds.erase(
          std::remove(group->memberIds.begin(), group->memberIds.end(), nodeId),
          group->memberIds.end());
    }
  }
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [nodeId](const GraphNode &candidate) {
                               return candidate.id == nodeId;
                             }),
              nodes.end());
}

void NodeGraph::flattenGroupBoundaryNodes() {
  std::vector<std::int32_t> boundaryIds;
  for (const auto &node : nodes) {
    if (isGroupBoundaryType(node.type))
      boundaryIds.push_back(node.id);
  }
  for (const auto nodeId : boundaryIds)
    spliceGroupBoundaryNode(nodeId);
  refreshPropagatedPinShapesCore(*this);
}

bool NodeGraph::wouldCreateCycle(std::int32_t sourceNodeId,
                                 std::int32_t destinationNodeId) const {
  std::queue<std::int32_t> pending;
  std::unordered_set<std::int32_t> visited;
  pending.push(destinationNodeId);
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (current == sourceNodeId)
      return true;
    if (!visited.insert(current).second)
      continue;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (source.has_value() && destination.has_value() && *source == current)
        pending.push(*destination);
    }
  }
  return false;
}

bool NodeGraph::selectionIsConnected(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  if (selectedNodeIds.empty())
    return false;
  const std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                                  selectedNodeIds.end());
  for (const auto id : selected) {
    const auto *node = findNode(id);
    if (node == nullptr || node->state != NodeState::liveBlue)
      return false;
  }
  if (selected.size() == 1)
    return true;

  std::queue<std::int32_t> pending;
  std::unordered_set<std::int32_t> visited;
  pending.push(*selected.begin());
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (!visited.insert(current).second)
      continue;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (*source == current && selected.count(*destination) != 0)
        pending.push(*destination);
      if (*destination == current && selected.count(*source) != 0)
        pending.push(*source);
    }
  }
  return visited.size() == selected.size();
}

bool NodeGraph::isPinConnected(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  return std::any_of(links.begin(), links.end(),
                     [pinId](const GraphLink &link) {
                       return link.sourcePinId == pinId ||
                              link.destinationPinId == pinId;
                     });
}

std::vector<std::vector<std::int32_t>> NodeGraph::partitionFreezeChains(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  std::unordered_set<std::int32_t> selected;
  for (const auto id : selectedNodeIds) {
    const auto *node = findNode(id);
    if (node == nullptr || isFixedIoType(node->type) ||
        isGroupBoundaryType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return {};
    selected.insert(id);
  }
  if (selected.empty())
    return {};

  std::unordered_map<std::int32_t, std::vector<std::int32_t>> undirected;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (selected.count(*source) != 0 && selected.count(*destination) != 0) {
      undirected[*source].push_back(*destination);
      undirected[*destination].push_back(*source);
    }
  }

  std::unordered_set<std::int32_t> visited;
  std::vector<std::vector<std::int32_t>> chains;
  for (const auto id : selectedNodeIds) {
    if (selected.count(id) == 0 || visited.count(id) != 0)
      continue;
    std::vector<std::int32_t> component;
    std::queue<std::int32_t> pending;
    pending.push(id);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      component.push_back(current);
      for (const auto neighbor : undirected[current])
        pending.push(neighbor);
    }

    const std::unordered_set<std::int32_t> members(component.begin(),
                                                   component.end());
    auto sources = members;
    auto sinks = members;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (members.count(*source) != 0 && members.count(*destination) != 0) {
        sources.erase(*destination);
        sinks.erase(*source);
      }
    }
    if (sources.size() != 1 || sinks.size() != 1)
      return {};
    chains.push_back(std::move(component));
  }
  return chains;
}

std::vector<std::int32_t> NodeGraph::getArmedTrainableNodeIds() const {
  std::vector<std::int32_t> armed;
  for (const auto &node : nodes) {
    if (node.state != NodeState::liveBlue)
      continue;
    if (isControlSourceType(node.type) || !node.hasWeights)
      continue;
    if (node.armedForTraining)
      armed.push_back(node.id);
  }
  return armed;
}

bool NodeGraph::setArmedForTraining(std::int32_t nodeId, bool armed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights || isControlSourceType(node->type) ||
      node->state == NodeState::frozenGold)
    return false;
  node->armedForTraining = armed;
  return true;
}

bool NodeGraph::setWeightsPath(std::int32_t nodeId, const std::string &path) {
  auto *node = findNode(nodeId);
  if (node == nullptr || (!node->hasWeights && node->type != NodeType::blackBox))
    return false;
  node->weightsProvenance = WeightsProvenance::file;
  node->weightsPath = path;
  if (node->type == NodeType::blackBox)
    node->artifactPath = path;
  return true;
}

bool NodeGraph::clearWeightsToSeed(std::int32_t nodeId, std::int32_t seed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights ||
      node->state == NodeState::frozenGold)
    return false;
  ensureRepeatSlotCount(*node, effectiveRepeatCount(nodeId));
  // BatchNorm keeps a shared identity seed; every other weighted member derives
  // `seed + i` so materialized repeats do not share live audition weights.
  const bool independentExtraSlots = node->type != NodeType::batchNorm;
  writeRandomizedWeightSlots(*node, seed, independentExtraSlots);
  node->blackBoxOrigin = BlackBoxOrigin::manualFreeze;
  return true;
}

std::optional<TrainJobRequest> NodeGraph::createTrainRequest() const {
  const auto hasBoundary = std::any_of(
      nodes.begin(), nodes.end(),
      [](const GraphNode &node) { return isGroupBoundaryType(node.type); });
  for (const auto &group : groups) {
    if (group.repeats > 1) {
      auto expanded = withInvisibleRepeatsMaterialized();
      return expanded.createTrainRequest();
    }
  }
  if (hasBoundary) {
    auto prepared = withInvisibleRepeatsMaterialized();
    return prepared.createTrainRequest();
  }
  const auto armed = getArmedTrainableNodeIds();
  if (armed.empty())
    return std::nullopt;

  TrainJobRequest request;
  request.requestId = juce::Uuid().toString().toStdString();
  request.armedNodeIds = armed;

  auto root = std::make_unique<juce::DynamicObject>();
  root->setProperty("request_id", juce::String(request.requestId));
  root->setProperty("operation", "train_steerable");
  root->setProperty("command", "start");

  juce::Array<juce::var> armedIds;
  juce::Array<juce::var> elements;
  const auto snapshot = collectTrainSnapshotNodeIds(*this);
  for (const auto nodeId : armed)
    armedIds.add(nodeId);
  for (const auto nodeId : snapshot) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      continue;
    auto element = std::make_unique<juce::DynamicObject>();
    element->setProperty("id", node->id);
    element->setProperty("type", nodeTypeName(node->type));
    element->setProperty("label", juce::String(node->label));
    element->setProperty("seed", node->seed);
    juce::Array<juce::var> properties;
    for (const auto &property : node->properties) {
      auto serialized = std::make_unique<juce::DynamicObject>();
      serialized->setProperty("key", juce::String(property.key));
      serialized->setProperty("value", property.value);
      if (property.kind == PropertyKind::real)
        serialized->setProperty("float_value", property.floatValue);
      else if (property.kind == PropertyKind::string)
        serialized->setProperty("string_value",
                                juce::String(property.stringValue));
      properties.add(juce::var(serialized.release()));
    }
    element->setProperty("properties", properties);
    elements.add(juce::var(element.release()));
  }
  root->setProperty("armed_element_ids", armedIds);

  auto fragment = std::make_unique<juce::DynamicObject>();
  fragment->setProperty("elements", elements);
  juce::Array<juce::var> connections;
  juce::Array<juce::var> audioInputs;
  juce::Array<juce::var> audioOutputs;
  juce::Array<juce::var> conditioningInputs;
  for (const auto &link : links) {
    const auto sourceNode = findNodeForPin(link.sourcePinId);
    const auto destinationNode = findNodeForPin(link.destinationPinId);
    if (!sourceNode.has_value() || !destinationNode.has_value())
      continue;
    const auto sourceArmed = snapshot.count(*sourceNode) != 0;
    const auto destinationArmed = snapshot.count(*destinationNode) != 0;
    if (sourceArmed && destinationArmed) {
      auto connection = std::make_unique<juce::DynamicObject>();
      connection->setProperty("source_element_id", *sourceNode);
      connection->setProperty("destination_element_id", *destinationNode);
      if (const auto *dest = findNode(*destinationNode))
        connection->setProperty(
            "destination_pin_index",
            inputPinIndexForId(*dest, link.destinationPinId));
      connections.add(juce::var(connection.release()));
    } else if (!sourceArmed && destinationArmed)
      audioInputs.add(*destinationNode);
    else if (sourceArmed && !destinationArmed)
      audioOutputs.add(*sourceNode);
  }
  fragment->setProperty("connections", connections);
  auto io = std::make_unique<juce::DynamicObject>();
  io->setProperty("audio_inputs", audioInputs);
  io->setProperty("audio_outputs", audioOutputs);
  io->setProperty("conditioning_inputs", conditioningInputs);
  fragment->setProperty("io_boundary", juce::var(io.release()));
  root->setProperty("graph_fragment", juce::var(fragment.release()));
  request.graphFragment =
      juce::JSON::toString(juce::var(root.release()), true).toStdString();
  return request;
}

std::optional<std::int32_t>
NodeGraph::absorbArmedChain(const TrainJobResult &result) {
  if (result.artifactPath.empty())
    return std::nullopt;
  if (getArmedTrainableNodeIds().empty())
    return std::nullopt;
  const auto processing = collectTrainSnapshotNodeIds(*this);
  if (processing.empty())
    return std::nullopt;

  std::unordered_map<std::int32_t, bool> absorbMemo;
  std::unordered_set<std::int32_t> absorbedGroups;
  for (const auto &group : groups) {
    if (groupIsFullyAbsorbed(*this, group.id, processing, absorbMemo))
      absorbedGroups.insert(group.id);
  }

  std::unordered_set<std::int32_t> selected(processing.begin(),
                                            processing.end());
  for (const auto groupId : absorbedGroups) {
    std::unordered_set<std::int32_t> subtreeNodes;
    std::unordered_set<std::int32_t> subtreeGroups;
    collectGroupSubtree(*this, groupId, subtreeNodes, subtreeGroups);
    selected.insert(subtreeNodes.begin(), subtreeNodes.end());
  }

  juce::ValueTree fragment{"GraphFragment"};
  juce::Point<float> centroid;
  int centroidCount = 0;
  for (const auto nodeId : selected) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    fragment.appendChild(nodeToTree(*node), nullptr);
    centroid += itemWorldPosition(*this, nodeId);
    ++centroidCount;
  }
  for (const auto groupId : absorbedGroups) {
    const auto *group = findGroup(groupId);
    if (group == nullptr)
      return std::nullopt;
    fragment.appendChild(groupToTree(*group), nullptr);
  }
  if (centroidCount > 0)
    centroid /= static_cast<float>(centroidCount);
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (selected.count(*source) != 0 || selected.count(*destination) != 0)
      fragment.appendChild(linkToTree(link), nullptr);
  }

  struct Boundary {
    std::int32_t sourcePin = 0;
    std::int32_t destinationPin = 0;
    bool control = false;
  };
  std::vector<Boundary> incoming;
  std::vector<Boundary> outgoing;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto *destinationPin = findPin(link.destinationPinId);
    const bool destControl =
        destinationPin != nullptr && isControlInputPin(*destinationPin);
    if (selected.count(*source) == 0 && selected.count(*destination) != 0)
      incoming.push_back({link.sourcePinId, 0, destControl});
    else if (selected.count(*source) != 0 && selected.count(*destination) == 0)
      outgoing.push_back({0, link.destinationPinId, false});
  }

  auto box = makeNode(NodeType::blackBox, centroid);
  box.label = result.hasEncodeDecode ? "Trained RAVE" : "Trained Steerable";
  box.detail = "Locked";
  box.state = NodeState::frozenGold;
  box.colour = colourForType(box.type, box.state);
  box.artifactPath = result.artifactPath;
  box.weightsPath = result.artifactPath;
  box.weightsProvenance = WeightsProvenance::file;
  box.blackBoxOrigin = BlackBoxOrigin::trainAutoload;
  box.hasWeights = true;
  box.compactnessReady = result.compactnessReady;
  if (result.compactnessReady)
    copyCompactnessFromArtifact(box, result.artifactPath);
  box.sourceSubgraph = fragment.toXmlString().toStdString();
  if (result.hasEncodeDecode) {
    box.outputs.push_back({nextPinId++, latentPinLabel, PinKind::output,
                           flexibleTensorShape(defaultLatentSize)});
    box.inputs.push_back({nextPinId++, latentPinLabel, PinKind::input,
                          flexibleTensorShape(defaultLatentSize)});
    box.fidelityPercent = defaultFidelityPercent;
  }
  const auto boxId = box.id;
  const auto audioIn = box.inputs.front().id;
  std::int32_t controlIn = 0;
  for (const auto &pin : box.inputs) {
    if (isControlInputPin(pin))
      controlIn = pin.id;
  }
  const auto audioOut = box.outputs.front().id;
  nodes.push_back(std::move(box));

  bool wiredAudioIn = false;
  bool wiredControl = false;
  for (const auto &edge : incoming) {
    if (edge.control && controlIn != 0 && !wiredControl) {
      links.push_back({nextLinkId++, edge.sourcePin, controlIn});
      wiredControl = true;
    } else if (!edge.control && !wiredAudioIn) {
      links.push_back({nextLinkId++, edge.sourcePin, audioIn});
      wiredAudioIn = true;
    }
  }
  bool wiredAudioOut = false;
  for (const auto &edge : outgoing) {
    if (!wiredAudioOut) {
      links.push_back({nextLinkId++, audioOut, edge.destinationPin});
      wiredAudioOut = true;
    }
  }

  std::vector<std::int32_t> toRemove(absorbedGroups.begin(),
                                      absorbedGroups.end());
  for (const auto nodeId : processing) {
    if (!nodeCoveredByAbsorbedGroup(*this, nodeId, absorbedGroups))
      toRemove.push_back(nodeId);
  }
  if (!toRemove.empty())
    removeBoxes(toRemove);
  return boxId;
}

bool NodeGraph::hasReconstructionTrainPath() const {
  return reconstructionGateMessage().empty();
}

std::string NodeGraph::reconstructionGateMessage() const {
  const GraphNode *bottleneck = nullptr;
  for (const auto &node : nodes) {
    if (node.type == NodeType::variationalBottleneck &&
        node.state == NodeState::liveBlue && node.armedForTraining) {
      bottleneck = &node;
      break;
    }
  }
  if (bottleneck == nullptr)
    return "Reconstruction requires an armed variational bottleneck";

  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> incoming;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    outgoing[*source].push_back(*destination);
    incoming[*destination].push_back(*source);
  }

  const auto reaches = [&](std::int32_t from, NodeType targetType) {
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visited;
    pending.push(from);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      const auto *node = findNode(current);
      if (node != nullptr && node->type == targetType)
        return true;
      for (const auto next : outgoing[current])
        pending.push(next);
    }
    return false;
  };
  const auto reachedFrom = [&](NodeType sourceType, std::int32_t to) {
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visited;
    pending.push(to);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      const auto *node = findNode(current);
      if (node != nullptr && node->type == sourceType)
        return true;
      for (const auto prev : incoming[current])
        pending.push(prev);
    }
    return false;
  };

  if (!reachedFrom(NodeType::audioInput, bottleneck->id))
    return "Reconstruction requires a path from Audio Input to the bottleneck";
  if (!reaches(bottleneck->id, NodeType::audioOutput))
    return "Reconstruction requires a decode path from the bottleneck to Audio Output";
  return {};
}
} // namespace openyourbox::graph
