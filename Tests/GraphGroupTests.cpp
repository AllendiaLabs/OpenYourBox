#include "graph/NodeGraph.h"

#include <JuceHeader.h>

#include <cmath>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
/**
 * @brief Reports a failed group invariant.
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
 * @brief Runs group membership, copies, and ValueTree round-trip checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto convA = graph.addNode(NodeType::convolution, {200.0f, 0.0f});
  const auto convB = graph.addNode(NodeType::convolution, {400.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {600.0f, 0.0f});
  const auto *inputNode = graph.findNode(input);
  const auto *nodeA = graph.findNode(convA);
  const auto *nodeB = graph.findNode(convB);
  const auto *outputNode = graph.findNode(output);
  passed &= expect(inputNode != nullptr && nodeA != nullptr && nodeB != nullptr &&
                       outputNode != nullptr,
                   "palette elements must exist");
  if (inputNode == nullptr || nodeA == nullptr || nodeB == nullptr ||
      outputNode == nullptr)
    return 1;
  passed &= expect(graph
                       .connect(inputNode->outputs.front().id,
                                nodeA->inputs.front().id)
                       .accepted,
                   "input to conv A");
  passed &= expect(graph
                       .connect(nodeA->outputs.front().id,
                                nodeB->inputs.front().id)
                       .accepted,
                   "conv A to conv B");
  passed &= expect(graph
                       .connect(nodeB->outputs.front().id,
                                outputNode->inputs.front().id)
                       .accepted,
                   "conv B to output");

  const auto ioGroup = graph.createGroup({input, convA});
  passed &= expect(!ioGroup.accepted, "audio I/O must refuse grouping");

  const auto created = graph.createGroup({convA, convB});
  passed &= expect(created.accepted, "two processing nodes must group");
  passed &= expect(graph.findGroup(created.groupId) != nullptr,
                   "created group must be findable");
  passed &= expect(graph.findNode(convA)->parentGroupId == created.groupId &&
                       graph.findNode(convB)->parentGroupId == created.groupId,
                   "members store parentGroupId");
  const auto *group = graph.findGroup(created.groupId);
  passed &= expect(group != nullptr && !group->collapsed && group->copies == 1,
                   "new groups start expanded with N=1");
  passed &= expect(group != nullptr && group->size.x >= 160.0f &&
                       group->size.y >= 120.0f,
                   "new groups fit their members");
  passed &= expect(std::abs(graph.worldPositionOfNode(convA).x - 200.0f) < 0.5f &&
                       std::abs(graph.worldPositionOfNode(convA).y - 0.0f) < 0.5f,
                   "grouping keeps member canvas positions");

  const auto cycle = graph.addToGroup(created.groupId, created.groupId);
  passed &= expect(!cycle.accepted, "group cannot contain itself");

  passed &= expect(graph.setGroupCollapsed(created.groupId, true),
                   "collapse must persist on the group");
  passed &= expect(graph.isNodeHiddenByCollapse(convA) &&
                       graph.isNodeHiddenByCollapse(convB),
                   "collapsed groups hide members");
  passed &= expect(graph.toggleGroupCollapsed(created.groupId) &&
                       !graph.findGroup(created.groupId)->collapsed,
                   "toggle expand restores members");

  const auto copies = graph.setGroupCopies(created.groupId, 3);
  passed &= expect(copies.accepted, "chainable group accepts N=3");
  passed &= expect(graph.findGroup(created.groupId)->copies == 3,
                   "copies persist on the group");
  passed &= expect(graph.effectiveCopyCount(convA) == 3,
                   "members report product of ancestor copies");
  passed &= expect(graph.findNode(convA)->copySlots.size() == 3 &&
                       graph.findNode(convB)->copySlots.size() == 3,
                   "members hold independent weight slots per copy");
  graph.findNode(convA)->copySlots[1].seed = 99;
  graph.findNode(convA)->copySlots[2].seed = 100;

  {
    using openyourbox::graph::NodeProperty;
    using openyourbox::graph::PropertyKind;
    using openyourbox::graph::parsePropertyCopyList;
    NodeProperty gain;
    gain.key = "gain";
    gain.label = "Gain";
    gain.kind = PropertyKind::real;
    gain.floatValue = 1.0f;
    gain.floatMinimum = openyourbox::graph::gainMinimum;
    gain.floatMaximum = openyourbox::graph::gainMaximum;
    const auto parsed = parsePropertyCopyList(gain, 3, "0.50, 1.00, 1.50");
    passed &= expect(parsed.accepted && parsed.floatValues.size() == 3 &&
                         std::abs(parsed.floatValues[1] - 1.0f) < 1.0e-5f,
                     "comma-separated reals parse into per-copy values");
    const auto broadcast = parsePropertyCopyList(gain, 3, "0.25");
    passed &= expect(broadcast.accepted && broadcast.floatValues.size() == 3 &&
                         std::abs(broadcast.floatValues[2] - 0.25f) < 1.0e-5f,
                     "a single value broadcasts to every copy");
    const auto commaDecimal = parsePropertyCopyList(gain, 3, "0,5");
    passed &= expect(!commaDecimal.accepted,
                     "comma is a list separator, not a decimal mark");
    const auto wrongCount = parsePropertyCopyList(gain, 3, "0.5, 1.0");
    passed &= expect(!wrongCount.accepted,
                     "lists that are not 1 or N values are refused");
  }
  passed &= expect(graph.setPropertyCopyValues(convA, "kernel_size", {3, 5, 7}),
                   "grouped integer properties accept N values");
  passed &= expect(!graph.setPropertyCopyValues(convA, "kernel_size", {3, 5}),
                   "wrong-sized copy lists are refused");
  {
    const auto *node = graph.findNode(convA);
    const openyourbox::graph::NodeProperty *kernel = nullptr;
    if (node != nullptr) {
      for (const auto &property : node->properties) {
        if (property.key == "kernel_size")
          kernel = &property;
      }
    }
    passed &= expect(kernel != nullptr && kernel->copyIntValues.size() == 3 &&
                         kernel->copyIntValues[0] == 3 &&
                         kernel->copyIntValues[1] == 5 &&
                         kernel->copyIntValues[2] == 7,
                     "per-copy integer values are stored on the property");
  }

  const auto expanded = graph.withInvisibleCopiesMaterialized();
  int convolutionCount = 0;
  for (const auto &node : expanded.getNodes()) {
    if (node.type == NodeType::convolution)
      ++convolutionCount;
  }
  passed &= expect(convolutionCount == 6,
                   "DSP unroll materializes two extra copies of each member");
  {
    int kernel3 = 0;
    int kernel5 = 0;
    int kernel7 = 0;
    for (const auto &node : expanded.getNodes()) {
      if (node.type != NodeType::convolution)
        continue;
      for (const auto &property : node.properties) {
        if (property.key != "kernel_size")
          continue;
        if (property.value == 3)
          ++kernel3;
        else if (property.value == 5)
          ++kernel5;
        else if (property.value == 7)
          ++kernel7;
      }
    }
    passed &= expect(kernel3 == 4 && kernel5 == 1 && kernel7 == 1,
                     "DSP unroll applies per-copy integer properties");
  }
  {
    std::unordered_map<std::int32_t, int> indegree;
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
    for (const auto &node : expanded.getNodes())
      indegree[node.id] = 0;
    for (const auto &link : expanded.getLinks()) {
      const auto source = expanded.findNodeForPin(link.sourcePinId);
      const auto dest = expanded.findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !dest.has_value())
        continue;
      outgoing[*source].push_back(*dest);
      ++indegree[*dest];
    }
    std::queue<std::int32_t> ready;
    for (const auto &[id, degree] : indegree) {
      if (degree == 0)
        ready.push(id);
    }
    int visited = 0;
    while (!ready.empty()) {
      const auto id = ready.front();
      ready.pop();
      ++visited;
      for (const auto destination : outgoing[id]) {
        if (--indegree[destination] == 0)
          ready.push(destination);
      }
    }
    passed &= expect(visited == static_cast<int>(expanded.getNodes().size()),
                     "unrolled copies must remain acyclic");
  }
  passed &= expect(graph.getNodes().size() == 4,
                   "UI graph must not show copied elements");

  const auto illegal = graph.createGroup({convA});
  passed &= expect(!illegal.accepted, "grouping requires two members");

  const auto act = graph.addNode(NodeType::activation, {300.0f, 80.0f});
  const auto added = graph.addToGroup(created.groupId, act);
  passed &= expect(added.accepted, "dragging an element onto a group adds it");
  passed &= expect(std::abs(graph.worldPositionOfNode(act).x - 300.0f) < 0.5f &&
                       std::abs(graph.worldPositionOfNode(act).y - 80.0f) < 0.5f,
                   "adding to a group keeps the element's canvas position");
  const auto removed = graph.removeFromGroup(act);
  passed &= expect(removed.accepted && !graph.findNode(act)->parentGroupId.has_value(),
                   "remove from group lifts the member");
  passed &= expect(std::abs(graph.findNode(act)->position.x - 300.0f) < 0.5f &&
                       std::abs(graph.findNode(act)->position.y - 80.0f) < 0.5f,
                   "lifting a member restores canvas coordinates");

  const auto memberLocalX = graph.findNode(convA)->position.x;
  const auto memberWorldX = graph.worldPositionOfNode(convA).x;
  const auto origin = graph.findGroup(created.groupId)->position;
  graph.moveGroup(created.groupId, {origin.x + 40.0f, origin.y + 10.0f});
  passed &= expect(std::abs(graph.findNode(convA)->position.x - memberLocalX) < 0.01f,
                   "moving a group keeps member-local coordinates");
  passed &= expect(std::abs(graph.worldPositionOfNode(convA).x - memberWorldX - 40.0f) <
                       0.01f,
                   "moving a group moves member canvas positions");

  graph.setGroupView(created.groupId, {12.0f, 8.0f}, 1.5f);
  passed &= expect(std::abs(graph.findGroup(created.groupId)->viewPan.x - 12.0f) <
                           0.01f &&
                       std::abs(graph.findGroup(created.groupId)->viewZoom - 1.5f) <
                           0.01f,
                   "group inner camera can be set independently");

  const auto frozen = graph.expandSelectionToFreezableLeaves({created.groupId});
  passed &= expect(frozen.size() == 2, "freeze expands a group to freezable leaves");

  const auto *groupedA = graph.findNode(convA);
  const auto *groupedB = graph.findNode(convB);
  passed &= expect(groupedA != nullptr && groupedB != nullptr,
                   "grouped members remain after freeze expansion");
  if (groupedA != nullptr && groupedB != nullptr) {
    const auto ports = graph.groupInterfacePorts(created.groupId);
    bool hasInput = false;
    bool hasOutput = false;
    bool hasInternalOutput = false;
    for (const auto &port : ports) {
      if (port.kind == openyourbox::graph::PinKind::input &&
          port.memberNodeId == convA)
        hasInput = true;
      if (port.kind == openyourbox::graph::PinKind::output &&
          port.memberNodeId == convB)
        hasOutput = true;
      if (port.kind == openyourbox::graph::PinKind::output &&
          port.memberNodeId == convA)
        hasInternalOutput = true;
    }
    passed &= expect(hasInput && hasOutput,
                     "group I/O pins mediate external member ports");
    passed &= expect(!hasInternalOutput,
                     "internally wired member outputs are not group I/O");
    passed &= expect(graph.innermostVisibleGroupOf(convA) == created.groupId,
                     "expanded group is the visible host of its members");
    passed &= expect(graph.isNodeOnFocusedCanvas(convA, created.groupId) &&
                         !graph.isNodeOnFocusedCanvas(convA, std::nullopt),
                     "members belong to the group canvas, not the root");
    passed &= expect(graph.isGroupOnFocusedCanvas(created.groupId, std::nullopt) &&
                         !graph.isGroupOnFocusedCanvas(created.groupId,
                                                      created.groupId),
                     "a group box is shown on its parent canvas");
    passed &= expect(!graph.focusedCanvasHostGroup(convA, created.groupId).has_value(),
                     "direct members have no nested host on the group canvas");
  }

  const auto doomedA = graph.addNode(NodeType::activation, {800.0f, 0.0f});
  const auto doomedB = graph.addNode(NodeType::activation, {900.0f, 0.0f});
  const auto doomed = graph.createGroup({doomedA, doomedB});
  passed &= expect(doomed.accepted, "temporary group for delete");
  passed &= expect(graph.deleteGroup(doomed.groupId).accepted,
                   "delete group removes the container");
  passed &= expect(graph.findGroup(doomed.groupId) == nullptr,
                   "deleted group is gone");
  passed &= expect(graph.findNode(doomedA) == nullptr &&
                       graph.findNode(doomedB) == nullptr,
                   "delete group removes members");
  passed &= expect(graph.findNode(convA) != nullptr &&
                       graph.findNode(convB) != nullptr,
                   "delete group does not remove other members");

  const auto tree = graph.toValueTree();
  NodeGraph restored;
  passed &= expect(restored.restoreFromValueTree(tree),
                   "group document must restore");
  passed &= expect(restored.getGroups().size() == 1,
                   "restored graph keeps the group");
  passed &= expect(restored.findGroup(created.groupId) != nullptr &&
                       restored.findGroup(created.groupId)->copies == 3,
                   "copies survive round-trip");
  passed &= expect(restored.findNode(convA) != nullptr &&
                       restored.findNode(convA)->parentGroupId == created.groupId,
                   "parentGroupId survives round-trip");
  passed &= expect(restored.findNode(convA)->copySlots.size() == 3 &&
                       restored.findNode(convA)->copySlots[1].seed == 99,
                   "per-copy weights survive round-trip");
  {
    const auto *node = restored.findNode(convA);
    const openyourbox::graph::NodeProperty *kernel = nullptr;
    if (node != nullptr) {
      for (const auto &property : node->properties) {
        if (property.key == "kernel_size")
          kernel = &property;
      }
    }
    passed &= expect(kernel != nullptr && kernel->copyIntValues.size() == 3 &&
                         kernel->copyIntValues[1] == 5 &&
                         kernel->copyIntValues[2] == 7,
                     "per-copy integer properties survive round-trip");
  }
  passed &= expect(restored.findGroup(created.groupId) != nullptr &&
                       std::abs(restored.findGroup(created.groupId)->viewPan.x -
                                12.0f) < 0.01f &&
                       std::abs(restored.findGroup(created.groupId)->viewZoom -
                                1.5f) < 0.01f,
                   "group inner camera survives round-trip");

  graph.getViewport().focusedGroupId = created.groupId;
  const auto focusedTree = graph.toValueTree();
  NodeGraph focusedRestore;
  passed &= expect(focusedRestore.restoreFromValueTree(focusedTree) &&
                       focusedRestore.getViewport().focusedGroupId ==
                           created.groupId,
                   "focused canvas survives round-trip");

  const auto insertedOnFocus =
      graph.addNode(NodeType::activation, {40.0f, 50.0f}, created.groupId);
  const auto *focusedMember = graph.findNode(insertedOnFocus);
  passed &= expect(focusedMember != nullptr &&
                       focusedMember->parentGroupId == created.groupId &&
                       std::abs(focusedMember->position.x - 40.0f) < 0.01f &&
                       std::abs(focusedMember->position.y - 50.0f) < 0.01f,
                   "addNode places on the focused group canvas");
  const auto *host = graph.findGroup(created.groupId);
  passed &= expect(host != nullptr &&
                       std::find(host->memberIds.begin(), host->memberIds.end(),
                                 insertedOnFocus) != host->memberIds.end(),
                   "focused insert updates group memberIds");

  {
    NodeGraph tcnGraph;
    const auto first = tcnGraph.addNode(NodeType::tcn, {200.0f, 0.0f});
    const auto second = tcnGraph.addNode(NodeType::tcn, {400.0f, 0.0f});
    const auto *tcnA = tcnGraph.findNode(first);
    const auto *tcnB = tcnGraph.findNode(second);
    passed &= expect(tcnA != nullptr && tcnB != nullptr &&
                         tcnA->inputs.size() == 2 && tcnB->outputs.size() == 1,
                     "TCN elements expose audio plus control inputs");
    if (tcnA != nullptr && tcnB != nullptr) {
      passed &= expect(tcnGraph
                           .connect(tcnA->outputs.front().id,
                                    tcnB->inputs.front().id)
                           .accepted,
                       "TCN audio out to TCN audio in");
      const auto grouped = tcnGraph.createGroup({first, second});
      passed &= expect(grouped.accepted, "TCN chain must group");
      const auto tcnCopies = tcnGraph.setGroupCopies(grouped.groupId, 3);
      passed &= expect(tcnCopies.accepted,
                       "TCN chain must accept copies without host I/O cables");
      const auto unrolled = tcnGraph.withInvisibleCopiesMaterialized();
      int tcnCount = 0;
      int controlFeeds = 0;
      int serialAudioFeeds = 0;
      for (const auto &node : unrolled.getNodes()) {
        if (node.type == NodeType::tcn)
          ++tcnCount;
      }
      for (const auto &link : unrolled.getLinks()) {
        const auto *destination = unrolled.findPin(link.destinationPinId);
        const auto *source = unrolled.findPin(link.sourcePinId);
        if (destination == nullptr || source == nullptr)
          continue;
        if (openyourbox::graph::isControlInputPin(*destination))
          ++controlFeeds;
        if (source->kind == openyourbox::graph::PinKind::output &&
            destination->kind == openyourbox::graph::PinKind::input &&
            !openyourbox::graph::isControlInputPin(*destination) &&
            unrolled.findNodeForPin(link.sourcePinId) !=
                unrolled.findNodeForPin(link.destinationPinId))
          ++serialAudioFeeds;
      }
      passed &= expect(tcnCount == 6,
                       "TCN copies unroll to six independent nodes");
      passed &= expect(controlFeeds == 0,
                       "serial copies must not wire into TCN control pins");
      passed &= expect(serialAudioFeeds == 5,
                       "three internal TCN cables plus two inter-copy audio cables");
    }
  }

  {
    NodeGraph knobs;
    const auto left = knobs.addNode(NodeType::knobInput, {0.0f, 0.0f});
    const auto right = knobs.addNode(NodeType::knobInput, {80.0f, 0.0f});
    const auto grouped = knobs.createGroup({left, right});
    passed &= expect(grouped.accepted, "two knobs may group");
    const auto refused = knobs.setGroupCopies(grouped.groupId, 2);
    passed &= expect(!refused.accepted,
                     "knobs without a through-path must refuse copies");
  }

  {
    using openyourbox::graph::WeightsProvenance;
    using openyourbox::graph::seedForCopySlot;
    NodeGraph weights;
    const auto conv = weights.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto norm = weights.addNode(NodeType::batchNorm, {400.0f, 0.0f});
    const auto *convNode = weights.findNode(conv);
    const auto *normNode = weights.findNode(norm);
    passed &= expect(convNode != nullptr && normNode != nullptr,
                     "weighted group members must exist");
    if (convNode != nullptr && normNode != nullptr) {
      passed &= expect(weights
                           .connect(convNode->outputs.front().id,
                                    normNode->inputs.front().id)
                           .accepted,
                       "conv to batch-norm");
      const auto grouped = weights.createGroup({conv, norm});
      passed &= expect(grouped.accepted, "conv and batch-norm must group");
      const auto unlockedId =
          weights.addNode(NodeType::linear, {300.0f, 80.0f});
      passed &= expect(weights.addToGroup(grouped.groupId, unlockedId).accepted,
                       "unlocked linear can join the group");
      weights.setGroupCopies(grouped.groupId, 2);
      auto *locked = weights.findNode(conv);
      auto *reset = weights.findNode(norm);
      auto *unlocked = weights.findNode(unlockedId);
      passed &= expect(locked != nullptr && reset != nullptr &&
                           unlocked != nullptr,
                       "grouped weighted members remain");
      if (locked != nullptr && reset != nullptr && unlocked != nullptr) {
        locked->useExplicitSeed = true;
        locked->explicitSeed = 777;
        weights.setSeed(conv, 777);
        locked->weightsPath = "/tmp/stale.pt";
        locked->weightsProvenance = WeightsProvenance::file;
        reset->seed = 9;
        unlocked->useExplicitSeed = false;
        unlocked->seed = 1;
        passed &= expect(weights.randomizeGroupWeights(grouped.groupId),
                         "group randomize updates live weighted members");
        passed &= expect(locked->seed == 777 && locked->useExplicitSeed &&
                             locked->weightsProvenance ==
                                 WeightsProvenance::random &&
                             locked->weightsPath.empty(),
                         "seed-locked members keep their seed");
        passed &= expect(reset->seed == 0 &&
                             reset->weightsProvenance ==
                                 WeightsProvenance::random,
                         "group randomize resets batch-norm");
        passed &= expect(unlocked->weightsProvenance ==
                                 WeightsProvenance::random &&
                             unlocked->seed >= 0 &&
                             unlocked->seed <= 999999,
                         "unlocked members draw a new seed");
        passed &= expect(locked->copySlots.size() == 2 &&
                             locked->copySlots.front().seed == 777 &&
                             locked->copySlots.back().seed == 778,
                         "seed-locked copy slots use base + i");
        passed &= expect(reset->copySlots.size() == 2 &&
                             reset->copySlots.front().seed == 0 &&
                             reset->copySlots.back().seed == 0,
                         "batch-norm copy slots reset to identity");
        passed &= expect(unlocked->copySlots.size() == 2 &&
                             unlocked->copySlots.front().seed == unlocked->seed &&
                             unlocked->copySlots.back().seed ==
                                 seedForCopySlot(unlocked->seed, 1),
                         "unlocked copy slots use base + i");
      }
    }
  }

  {
    using openyourbox::graph::WeightsProvenance;
    using openyourbox::graph::ensureCopySlotCount;
    NodeGraph elementRandomize;
    const auto linearId =
        elementRandomize.addNode(NodeType::linear, {0.0f, 0.0f});
    const auto actId =
        elementRandomize.addNode(NodeType::activation, {200.0f, 0.0f});
    const auto *linearNode = elementRandomize.findNode(linearId);
    const auto *actNode = elementRandomize.findNode(actId);
    passed &= expect(linearNode != nullptr && actNode != nullptr,
                     "element-randomize fixture nodes exist");
    if (linearNode != nullptr && actNode != nullptr)
      passed &= expect(elementRandomize
                           .connect(linearNode->outputs.front().id,
                                    actNode->inputs.front().id)
                           .accepted,
                       "element-randomize linear to activation");
    const auto grouped = elementRandomize.createGroup({linearId, actId});
    passed &= expect(grouped.accepted, "element-randomize fixture groups");
    auto *linear = elementRandomize.findNode(linearId);
    passed &= expect(linear != nullptr, "element-randomize linear exists");
    if (linear != nullptr) {
      linear->seed = 100;
      linear->weightsProvenance = WeightsProvenance::random;
      ensureCopySlotCount(*linear, 1);
    }
    passed &=
        expect(elementRandomize.setGroupCopies(grouped.groupId, 3).accepted,
               "element-randomize fixture sets N=3");
    linear = elementRandomize.findNode(linearId);
    passed &= expect(linear != nullptr && linear->copySlots.size() == 3,
                     "N=3 creates three copy slots");
    if (linear != nullptr) {
      passed &= expect(linear->copySlots[0].seed == 100 &&
                           linear->copySlots[1].seed == 101 &&
                           linear->copySlots[2].seed == 102,
                       "raising N derives seed + i for random slots");
      passed &= expect(elementRandomize.clearWeightsToSeed(linearId, 12345),
                       "element clearWeightsToSeed succeeds");
      linear = elementRandomize.findNode(linearId);
      passed &= expect(linear != nullptr && linear->seed == 12345 &&
                           linear->copySlots.size() == 3 &&
                           linear->copySlots[0].seed == 12345 &&
                           linear->copySlots[1].seed == 12346 &&
                           linear->copySlots[2].seed == 12347,
                       "element randomize writes seed + i across copies");
      passed &= expect(elementRandomize.setFloatPropertyCopyValues(
                           actId, "gain", {0.5f, 1.0f, 1.5f}),
                       "grouped real properties accept N values");
      const auto expandedGains =
          elementRandomize.withInvisibleCopiesMaterialized();
      std::vector<float> gains;
      for (const auto &node : expandedGains.getNodes()) {
        if (node.type != NodeType::activation)
          continue;
        for (const auto &property : node.properties) {
          if (property.key == "gain")
            gains.push_back(property.floatValue);
        }
      }
      passed &= expect(gains.size() == 3 &&
                           std::abs(gains[0] - 0.5f) < 1.0e-5f &&
                           std::abs(gains[1] - 1.0f) < 1.0e-5f &&
                           std::abs(gains[2] - 1.5f) < 1.0e-5f,
                       "DSP unroll applies per-copy real properties");
    }
  }

  const auto worldBeforeUngroup = restored.worldPositionOfNode(convA);
  const auto lifted = restored.ungroup(created.groupId);
  passed &= expect(lifted.accepted && restored.getGroups().empty(),
                   "ungroup removes the container");
  passed &= expect(restored.findNode(convA) != nullptr &&
                       !restored.findNode(convA)->parentGroupId.has_value(),
                   "ungroup preserves members");
  passed &= expect(std::abs(restored.findNode(convA)->position.x -
                            worldBeforeUngroup.x) < 0.5f &&
                       std::abs(restored.findNode(convA)->position.y -
                                worldBeforeUngroup.y) < 0.5f,
                   "ungroup restores canvas coordinates");

  return passed ? 0 : 1;
}
