#include "graph/BoxFlowOrder.h"

#include <iostream>
#include <unordered_map>
#include <vector>

namespace {
/**
 * @brief Reports a failed flow-order invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Returns true when @p ordered equals @p expected.
 * @param ordered Actual order.
 * @param expected Expected order.
 */
bool sameOrder(const std::vector<std::int32_t> &ordered,
               const std::vector<std::int32_t> &expected) {
  return ordered == expected;
}
} // namespace

/**
 * @brief Runs information-flow rank ordering checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::BoxFlowEdge;
  using openyourbox::graph::orderBoxesByFlowRank;

  bool passed = true;

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {3, "Zeta"}, {1, "alpha"}, {2, "Beta"}};
    const auto ordered =
        orderBoxesByFlowRank({3, 1, 2}, names, {});
    passed &= expect(sameOrder(ordered, {1, 2, 3}),
                     "no edges must sort by case-insensitive name, then id");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "Delay"}, {2, "Delay"}};
    const auto ordered = orderBoxesByFlowRank({2, 1}, names, {});
    passed &= expect(sameOrder(ordered, {1, 2}),
                     "equal names must sort by id");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "encoder"},
        {2, "variational encoder"},
        {3, "decoder"}};
    const std::vector<BoxFlowEdge> edges{{1, 2}, {2, 3}};
    const auto ordered = orderBoxesByFlowRank({3, 1, 2}, names, edges);
    passed &= expect(sameOrder(ordered, {1, 2, 3}),
                     "chain must follow longest-path rank");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "encoder"}, {2, "decoder"}, {3, "Zeta"}};
    const std::vector<BoxFlowEdge> edges{{1, 2}};
    const auto ordered = orderBoxesByFlowRank({2, 3, 1}, names, edges);
    passed &= expect(sameOrder(ordered, {1, 3, 2}),
                     "unconnected boxes must sit with sources, then by name");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "A"}, {2, "B"}, {3, "C"}, {4, "D"}};
    const std::vector<BoxFlowEdge> edges{{1, 2}, {1, 4}, {2, 4}};
    const auto ordered = orderBoxesByFlowRank({4, 3, 2, 1}, names, edges);
    passed &= expect(sameOrder(ordered, {1, 3, 2, 4}),
                     "longest path must rank D after B, with orphan C at rank 0");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "encoder"}, {2, "decoder"}};
    const std::vector<BoxFlowEdge> edges{{1, 2}, {2, 1}};
    const auto ordered = orderBoxesByFlowRank({1, 2}, names, edges);
    passed &= expect(sameOrder(ordered, {2, 1}),
                     "feedback loop must share a rank and sort by name");
  }

  {
    std::unordered_map<std::int32_t, juce::String> names{
        {1, "encoder"}, {2, "decoder"}, {3, "Foo"}};
    const std::vector<BoxFlowEdge> edges{{3, 1}, {1, 2}, {2, 1}};
    const auto ordered = orderBoxesByFlowRank({2, 1, 3}, names, edges);
    passed &= expect(sameOrder(ordered, {3, 2, 1}),
                     "loop must rank after its upstream source");
  }

  if (passed)
    std::cout << "OpenYourBox flow order tests passed\n";
  return passed ? 0 : 1;
}
