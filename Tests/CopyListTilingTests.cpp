#include "graph/GraphTypes.h"
#include "graph/NodeGraph.h"

#include <JuceHeader.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
/**
 * @brief Reports a failed tiling or binding invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}
} // namespace

/**
 * @brief Runs dividing-set, tiling, `in` binding, and sticky-trail checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::dividingSetLengths;
  using openyourbox::graph::formatExpandedPropertyCopyList;
  using openyourbox::graph::formatHierarchicalCopyList;
  using openyourbox::graph::formatShapeCopyList;
  using openyourbox::graph::isDividingSetLength;
  using openyourbox::graph::parsePropertyCopyList;
  using openyourbox::graph::updateHierarchyStickySpine;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  const std::vector<int> nest{2, 2, 2};
  const auto allowed = dividingSetLengths(nest);
  passed &= expect(allowed.size() == 4 && allowed[0] == 1 && allowed[1] == 2 &&
                       allowed[2] == 4 && allowed[3] == 8,
                   "O=2,M=2,N=2 dividing set is {1,2,4,8}");
  passed &= expect(isDividingSetLength(2, nest) && !isDividingSetLength(3, nest),
                   "illegal length 3 is outside the dividing set");

  const std::vector<std::string> eight{"1", "2", "3", "4", "5", "6", "7", "8"};
  passed &= expect(formatHierarchicalCopyList(eight, nest) ==
                       "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
                   "three-level nest formats as a list of lists of lists");
  passed &= expect(formatHierarchicalCopyList({"a", "b", "c"}, {3}) ==
                       "[a, b, c]",
                   "one copy axis wraps a single bracketed list");
  passed &= expect(formatHierarchicalCopyList({"a", "b", "c", "d"}, {2, 2}) ==
                       "[[a, b], [c, d]]",
                   "two copy axes format as a list of lists");
  passed &= expect(formatHierarchicalCopyList({"x"}, {}) == "x",
                   "a single ungrouped value is unbracketed");
  passed &= expect(formatHierarchicalCopyList({"a", "b", "c"}, {1, 3}) ==
                       "[a, b, c]",
                   "copy axes of 1 are skipped when nesting");

  NodeProperty preview;
  preview.key = "features";
  preview.label = "Features";
  preview.kind = PropertyKind::integer;
  preview.copyIntValues = {1, 2, 3, 4, 5, 6, 7, 8};
  passed &= expect(formatExpandedPropertyCopyList(preview, 8, nest) ==
                       "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
                   "expanded property preview nests along C");
  preview.copyIntValues = {1, 2};
  passed &= expect(formatExpandedPropertyCopyList(preview, 8, nest) ==
                       "[[[1, 2], [1, 2]], [[1, 2], [1, 2]]]",
                   "tiled L=2 preview keeps inner-pair grouping");

  openyourbox::graph::ShapeSignature twoCh;
  twoCh.channels = 2;
  openyourbox::graph::ShapeSignature fourCh;
  fourCh.channels = 4;
  openyourbox::graph::ShapeSignature eightCh;
  eightCh.channels = 8;
  openyourbox::graph::ShapeSignature sixteenCh;
  sixteenCh.channels = 16;
  passed &= expect(formatShapeCopyList({twoCh, fourCh, eightCh, sixteenCh},
                                       twoCh, {2, 2}) ==
                       "[[2ch, 4ch], [8ch, 16ch]]",
                   "pin shapes nest as a list of lists");

  NodeProperty channels;
  channels.key = "channels";
  channels.label = "Channels";
  channels.kind = PropertyKind::integer;
  channels.minimum = 1;
  channels.maximum = 1024;
  channels.value = 16;
  const auto tiledTwo = parsePropertyCopyList(channels, nest, "16, 32", false);
  passed &= expect(tiledTwo.accepted && tiledTwo.intValues.size() == 2,
                   "L=2 is accepted for a three-level nest");
  channels.copyIntValues = tiledTwo.intValues;
  passed &= expect(integerValueForCopy(channels, 0) == 16 &&
                       integerValueForCopy(channels, 1) == 32 &&
                       integerValueForCopy(channels, 2) == 16 &&
                       integerValueForCopy(channels, 7) == 32,
                   "L=2 tiles as s % L across P=8");
  const auto tiledFour =
      parsePropertyCopyList(channels, nest, "1, 2, 3, 4", false);
  passed &= expect(tiledFour.accepted && tiledFour.intValues.size() == 4,
                   "L=4 is accepted for a three-level nest");
  channels.copyIntValues = tiledFour.intValues;
  passed &= expect(integerValueForCopy(channels, 0) == 1 &&
                       integerValueForCopy(channels, 4) == 1 &&
                       integerValueForCopy(channels, 5) == 2,
                   "L=4 tiles across the outermost axis");
  const auto refused = parsePropertyCopyList(channels, nest, "1, 2, 3", false);
  passed &= expect(!refused.accepted, "L=3 is refused for O=M=N=2");

  const auto bound = parsePropertyCopyList(channels, nest, "in", true);
  passed &= expect(bound.accepted && bound.preserveIn,
                   "single in token binds a channels field");
  const auto mixed = parsePropertyCopyList(channels, nest, "in, 32", true);
  passed &= expect(!mixed.accepted, "mixed in and numbers are refused");
  NodeProperty gain;
  gain.key = "gain";
  gain.label = "Gain";
  gain.kind = PropertyKind::real;
  gain.floatMinimum = 0.1f;
  gain.floatMaximum = 10.0f;
  const auto gainIn = parsePropertyCopyList(gain, nest, "in", true);
  passed &= expect(!gainIn.accepted, "in is refused on non-bindable fields");

  NodeGraph graph;
  const auto linear = graph.addNode(NodeType::linear, {0.0f, 0.0f});
  const auto partner = graph.addNode(NodeType::activation, {80.0f, 0.0f});
  const auto extraMiddle = graph.addNode(NodeType::activation, {160.0f, 0.0f});
  const auto extraOuter = graph.addNode(NodeType::activation, {240.0f, 0.0f});
  const auto inner = graph.createGroup({linear, partner});
  passed &= expect(inner.accepted, "inner group");
  passed &= expect(graph.setGroupCopies(inner.groupId, 2).accepted, "inner N=2");
  const auto middle = graph.createGroup({inner.groupId, extraMiddle});
  passed &= expect(middle.accepted, "middle group");
  passed &= expect(graph.setGroupCopies(middle.groupId, 2).accepted, "middle M=2");
  const auto outer = graph.createGroup({middle.groupId, extraOuter});
  passed &= expect(outer.accepted, "outer group");
  passed &= expect(graph.setGroupCopies(outer.groupId, 2).accepted, "outer O=2");
  passed &= expect(graph.effectiveCopyCount(linear) == 8, "P=8 for O=M=N=2");
  const auto counts = graph.ancestorCopyCounts(linear);
  passed &= expect(counts.size() == 3 && counts[0] == 2 && counts[2] == 2,
                   "ancestor copy vector is outer to inner");
  passed &= expect(graph.setPropertyCopyValues(linear, "features", {8, 16}),
                   "authored L=2 commits");
  const auto *node = graph.findNode(linear);
  const openyourbox::graph::NodeProperty *features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->copyIntValues.size() == 2,
                   "authored length is stored, not expanded to P");
  passed &= expect(graph.setPropertyPreserveIn(linear, "features", 1),
                   "in binding commits on features");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->preserveInBound,
                   "preserveInBound persists on the property");

  passed &= expect(graph.setPropertyCopyValues(linear, "features", {4, 5, 6, 7}),
                   "L=4 remains valid");
  passed &= expect(graph.setGroupCopies(outer.groupId, 3).accepted,
                   "outer copies can change");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && !features->copyListInvalid &&
                       features->copyIntValues.size() == 4,
                   "L=4 still valid when O becomes 3 (P=12)");
  passed &= expect(graph.setGroupCopies(inner.groupId, 3).accepted,
                   "inner copies can change");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->copyListInvalid &&
                       features->copyIntValues.size() == 4,
                   "L=4 is flagged invalid when N becomes 3");

  std::vector<std::int32_t> spine;
  updateHierarchyStickySpine(spine, {10, 20, 30}, {10});
  passed &= expect(spine.size() == 2 && spine[0] == 20 && spine[1] == 30,
                   "navigating up keeps descendant spine");
  updateHierarchyStickySpine(spine, {10}, {10, 20});
  passed &= expect(spine.size() == 1 && spine[0] == 30,
                   "clicking a sticky child keeps deeper descendants");
  updateHierarchyStickySpine(spine, {10}, {10, 40});
  passed &= expect(spine.empty(), "opening a sibling branch clears the spine");

  return passed ? 0 : 1;
}
