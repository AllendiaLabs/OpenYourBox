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
  using openyourbox::graph::formatAuthoredPropertyRepeatList;
  using openyourbox::graph::formatExpandedPropertyRepeatList;
  using openyourbox::graph::formatHierarchicalRepeatList;
  using openyourbox::graph::formatCollapsedGroupPinShapes;
  using openyourbox::graph::collapsedGroupAttachShape;
  using openyourbox::graph::formatShapeRepeatList;
  using openyourbox::graph::isDividingSetLength;
  using openyourbox::graph::parsePropertyRepeatList;
  using openyourbox::graph::integerValueForRepeat;
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
  passed &= expect(formatHierarchicalRepeatList(eight, nest) ==
                       "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
                   "three-level nest formats as a list of lists of lists");
  passed &= expect(formatHierarchicalRepeatList({"a", "b", "c"}, {3}) ==
                       "[a, b, c]",
                   "one repeat axis wraps a single bracketed list");
  passed &= expect(formatHierarchicalRepeatList({"a", "b", "c", "d"}, {2, 2}) ==
                       "[[a, b], [c, d]]",
                   "two repeat axes format as a list of lists");
  passed &= expect(formatHierarchicalRepeatList({"x"}, {}) == "x",
                   "a single ungrouped value is unbracketed");
  passed &= expect(formatHierarchicalRepeatList({"a", "b", "c"}, {1, 3}) ==
                       "[a, b, c]",
                   "repeat axes of 1 are skipped when nesting");

  NodeProperty preview;
  preview.key = "features";
  preview.label = "Features";
  preview.kind = PropertyKind::integer;
  preview.repeatIntValues = {1, 2, 3, 4, 5, 6, 7, 8};
  passed &= expect(formatExpandedPropertyRepeatList(preview, 8, nest) ==
                       "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
                   "expanded property preview nests along C");
  preview.repeatIntValues = {1, 2};
  passed &= expect(formatExpandedPropertyRepeatList(preview, 8, nest) ==
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
  passed &= expect(formatShapeRepeatList({twoCh, fourCh, eightCh, sixteenCh},
                                       twoCh, {2, 2}) ==
                       "[[2ch, 4ch], [8ch, 16ch]]",
                   "pin shapes nest as a list of lists");
  passed &= expect(formatShapeRepeatList({}, twoCh, {}) == "2ch",
                   "a pin outside a group still formats its shape");
  passed &= expect(formatShapeRepeatList({twoCh}, twoCh, {1}) == "2ch",
                   "repeats=1 still formats a shape without brackets");
  passed &= expect(formatCollapsedGroupPinShapes({twoCh, fourCh, eightCh},
                                                 twoCh, {3}, true) == "8ch",
                   "collapsed group output shows last repeat, not the inner list");
  passed &= expect(collapsedGroupAttachShape({twoCh, fourCh, eightCh}, twoCh,
                                             {3}, true)
                           .channels == 8,
                   "parent-canvas group output attaches at last-out");
  passed &= expect(collapsedGroupAttachShape(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, sixteenCh},
                       twoCh, {2, 3}, true)
                           .channels == 8,
                   "nested attach shape is first outer slot's last-out, not 16");
  passed &= expect(formatCollapsedGroupPinShapes({twoCh, fourCh, eightCh},
                                                 twoCh, {3}, false) == "2ch",
                   "collapsed group input shows first repeat, not the inner list");
  passed &= expect(formatCollapsedGroupPinShapes(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, eightCh}, twoCh,
                       {2, 3}, true) == "[8ch, 8ch]",
                   "collapsed group output lists ancestor repeats of last-out");
  passed &= expect(formatCollapsedGroupPinShapes(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, eightCh}, twoCh,
                       {2, 3}, false) == "[2ch, 2ch]",
                   "collapsed group input lists ancestor repeats of first-in");
  passed &=
      expect(formatCollapsedGroupPinShapes(
                 {twoCh, fourCh, eightCh, twoCh, fourCh, sixteenCh}, twoCh,
                 {2, 3}, true) == "[8ch, 16ch]",
             "nested ancestor last-outs can differ per outer repeat");
  passed &= expect(
      formatCollapsedGroupPinShapes(
          {twoCh, fourCh, eightCh, twoCh, fourCh, sixteenCh, twoCh, fourCh,
           eightCh, twoCh, fourCh, sixteenCh},
          twoCh, {2, 2, 3}, true) == "[[8ch, 16ch], [8ch, 16ch]]",
      "collapsed pins nest ancestor repeat axes outside the folded group");
  passed &= expect(formatCollapsedGroupPinShapes({twoCh, eightCh}, twoCh,
                                                 {2, 1}, true) == "[2ch, 8ch]",
                   "own repeats=1 still lists ancestor repeats on the group box");
  passed &= expect(formatCollapsedGroupPinShapes({twoCh, fourCh, eightCh},
                                                 twoCh, {1}, true) == "8ch",
                   "parent repeats=1 folds nested inner repeats to last-out");
  passed &= expect(formatCollapsedGroupPinShapes(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, eightCh}, twoCh,
                       {1}, true) == "8ch",
                   "parent repeats=1 folds a deeper nested repeat product to last-out");
  passed &= expect(formatCollapsedGroupPinShapes(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, sixteenCh}, twoCh,
                       {2}, true) == "16ch",
                   "collapsed parent folds own repeats and nested repeats together");
  passed &= expect(formatCollapsedGroupPinShapes(
                       {twoCh, fourCh, eightCh, twoCh, fourCh, sixteenCh}, twoCh,
                       {2, 1}, true) == "[8ch, 16ch]",
                   "repeats=1 group still lists ancestor repeats of nested last-out");

  NodeProperty channels;
  channels.key = "channels";
  channels.label = "Channels";
  channels.kind = PropertyKind::integer;
  channels.minimum = 1;
  channels.maximum = 1024;
  channels.value = 16;
  const auto tiledTwo = parsePropertyRepeatList(channels, nest, "16, 32", false);
  passed &= expect(tiledTwo.accepted && tiledTwo.intValues.size() == 2,
                   "L=2 is accepted for a three-level nest");
  channels.repeatIntValues = tiledTwo.intValues;
  passed &= expect(integerValueForRepeat(channels, 0) == 16 &&
                       integerValueForRepeat(channels, 1) == 32 &&
                       integerValueForRepeat(channels, 2) == 16 &&
                       integerValueForRepeat(channels, 7) == 32,
                   "L=2 tiles as s % L across P=8");
  const auto tiledFour =
      parsePropertyRepeatList(channels, nest, "1, 2, 3, 4", false);
  passed &= expect(tiledFour.accepted && tiledFour.intValues.size() == 4,
                   "L=4 is accepted for a three-level nest");
  channels.repeatIntValues = tiledFour.intValues;
  passed &= expect(integerValueForRepeat(channels, 0) == 1 &&
                       integerValueForRepeat(channels, 4) == 1 &&
                       integerValueForRepeat(channels, 5) == 2,
                   "L=4 tiles across the outermost axis");
  const auto refused = parsePropertyRepeatList(channels, nest, "1, 2, 3", false);
  passed &= expect(!refused.accepted, "L=3 is refused for O=M=N=2");

  const auto bound = parsePropertyRepeatList(channels, nest, "in", true);
  passed &= expect(bound.accepted && bound.preserveIn,
                   "single in token binds a channels field");
  const auto mixed = parsePropertyRepeatList(channels, nest, "in, 32", true);
  passed &= expect(!mixed.accepted, "mixed in and numbers are refused");
  NodeProperty gain;
  gain.key = "gain";
  gain.label = "Gain";
  gain.kind = PropertyKind::real;
  gain.floatMinimum = 0.1f;
  gain.floatMaximum = 10.0f;
  const auto gainIn = parsePropertyRepeatList(gain, nest, "in", true);
  passed &= expect(!gainIn.accepted, "in is refused on non-bindable fields");

  const auto exprEight = parsePropertyRepeatList(channels, nest, "2*i+1", false);
  passed &= expect(exprEight.accepted && exprEight.authoredTokens.size() == 1,
                   "single i-expression is a legal dividing-set length");
  channels.authoredTokens = exprEight.authoredTokens;
  channels.repeatIntValues = exprEight.intValues;
  passed &= expect(integerValueForRepeat(channels, 0) == 1 &&
                       integerValueForRepeat(channels, 3) == 7 &&
                       integerValueForRepeat(channels, 7) == 15,
                   "2*i+1 fills P=8 from expanded slot i");
  passed &= expect(formatAuthoredPropertyRepeatList(channels) == "2*i+1",
                   "authored field keeps the expression text");
  passed &= expect(formatExpandedPropertyRepeatList(channels, 8, nest) ==
                       "[[[1, 3], [5, 7]], [[9, 11], [13, 15]]]",
                   "expanded preview shows resolved integers");

  NodeProperty ungroupedProp = channels;
  const auto ungrouped = parsePropertyRepeatList(channels, {}, "i+3", false);
  passed &= expect(ungrouped.accepted, "ungrouped i+3 is a legal length-1 list");
  ungroupedProp.authoredTokens = ungrouped.authoredTokens;
  ungroupedProp.repeatIntValues = ungrouped.intValues;
  passed &= expect(integerValueForRepeat(ungroupedProp, 0) == 3,
                   "ungrouped i is 0 so i+3 is 3");

  const auto nonInteger =
      parsePropertyRepeatList(channels, nest, "(i+1)^0.5", false);
  passed &= expect(!nonInteger.accepted,
                   "integer fields refuse non-integer (i+1)^0.5");
  passed &= expect(nonInteger.message.find("integer") != std::string::npos,
                   "integer refuse message mentions integer");

  const auto constantPow = parsePropertyRepeatList(channels, nest, "2^3", false);
  passed &= expect(constantPow.accepted, "constant 2^3 commits");
  channels.authoredTokens = constantPow.authoredTokens;
  channels.repeatIntValues = constantPow.intValues;
  passed &= expect(integerValueForRepeat(channels, 0) == 8 &&
                       integerValueForRepeat(channels, 7) == 8,
                   "2^3 is 8 in every slot");

  const auto mixedExpr =
      parsePropertyRepeatList(channels, nest, "2*i+1, 4", false);
  passed &= expect(mixedExpr.accepted && mixedExpr.authoredTokens.size() == 2,
                   "mixed literal/expression list of length 2 is legal");
  channels.authoredTokens = mixedExpr.authoredTokens;
  channels.repeatIntValues = mixedExpr.intValues;
  passed &= expect(integerValueForRepeat(channels, 0) == 1 &&
                       integerValueForRepeat(channels, 1) == 4 &&
                       integerValueForRepeat(channels, 2) == 5 &&
                       integerValueForRepeat(channels, 3) == 4,
                   "tiled 2*i+1 uses expanded slot i, not the short-list index");
  passed &= expect(formatAuthoredPropertyRepeatList(channels) == "2*i+1, 4",
                   "authored mixed list keeps both tokens");

  NodeGraph graph;
  const auto linear = graph.addNode(NodeType::linear, {0.0f, 0.0f});
  const auto partner = graph.addNode(NodeType::activation, {80.0f, 0.0f});
  const auto extraMiddle = graph.addNode(NodeType::activation, {160.0f, 0.0f});
  const auto extraOuter = graph.addNode(NodeType::activation, {240.0f, 0.0f});
  const auto inner = graph.createGroup({linear, partner});
  passed &= expect(inner.accepted, "inner group");
  passed &= expect(graph.setGroupRepeats(inner.groupId, 2).accepted, "inner N=2");
  const auto middle = graph.createGroup({inner.groupId, extraMiddle});
  passed &= expect(middle.accepted, "middle group");
  passed &= expect(graph.setGroupRepeats(middle.groupId, 2).accepted, "middle M=2");
  const auto outer = graph.createGroup({middle.groupId, extraOuter});
  passed &= expect(outer.accepted, "outer group");
  passed &= expect(graph.setGroupRepeats(outer.groupId, 2).accepted, "outer O=2");
  passed &= expect(graph.effectiveRepeatCount(linear) == 8, "P=8 for O=M=N=2");
  const auto counts = graph.ancestorRepeatCounts(linear);
  passed &= expect(counts.size() == 3 && counts[0] == 2 && counts[2] == 2,
                   "ancestor repeat vector is outer to inner");
  passed &= expect(graph.setPropertyRepeatValues(linear, "features", {8, 16}),
                   "authored L=2 commits");
  const auto *node = graph.findNode(linear);
  const openyourbox::graph::NodeProperty *features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->repeatIntValues.size() == 2,
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

  passed &= expect(graph.setPropertyRepeatValues(linear, "features", {4, 5, 6, 7}),
                   "L=4 remains valid");
  passed &= expect(graph.setGroupRepeats(outer.groupId, 3).accepted,
                   "outer repeats can change");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && !features->repeatListInvalid &&
                       features->repeatIntValues.size() == 4,
                   "L=4 still valid when O becomes 3 (P=12)");
  passed &= expect(graph.setGroupRepeats(inner.groupId, 3).accepted,
                   "inner repeats can change");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->repeatListInvalid &&
                       features->repeatIntValues.size() == 4,
                   "L=4 is flagged invalid when N becomes 3");

  passed &= expect(
      graph.setPropertyRepeatValues(linear, "features", {1}, {"2*i+1"}),
      "i-expression of length 1 is always a legal dividing-set length");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && !features->repeatListInvalid &&
                       integerValueForRepeat(*features, 0) == 1 &&
                       integerValueForRepeat(*features, 4) == 9,
                   "2*i+1 re-evaluates at expanded slots after nest change");
  passed &= expect(graph.setGroupRepeats(outer.groupId, 2).accepted,
                   "outer repeats can change again without retyping");
  node = graph.findNode(linear);
  features = nullptr;
  if (node != nullptr) {
    for (const auto &property : node->properties) {
      if (property.key == "features")
        features = &property;
    }
  }
  passed &= expect(features != nullptr && features->authoredTokens.size() == 1 &&
                       integerValueForRepeat(*features, 11) == 23,
                   "authored 2*i+1 re-evaluates for the new P without retyping");

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
