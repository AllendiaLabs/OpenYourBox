#include "graph/NodeGraph.h"
#include "library/UserBoxLibrary.h"

#include <JuceHeader.h>

#include <cmath>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

/**
 * @brief Wires a named user-library RAVE group between mono Audio In and Out.
 * @param name Catalog display name (absent boxes are skipped).
 * @return True when the box is missing or both host cables are accepted.
 */
bool testUserLibraryRaveConnectsToHostIo(const char *name) {
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  using openyourbox::library::UserBoxLibrary;
  UserBoxLibrary library;
  const auto *entry = library.findEntryByName(name);
  if (entry == nullptr)
    return true;
  const auto label = std::string(name);
  juce::String snapshotError;
  const auto snapshot = library.loadEntrySnapshot(entry->id, snapshotError);
  const auto snapshotMessage = snapshotError.isEmpty()
                                   ? label + " snapshot must load"
                                   : snapshotError.toStdString();
  if (!expect(snapshot.isValid(), snapshotMessage.c_str()))
    return false;
  NodeGraph graph;
  graph.ensureFixedHostIo();
  std::int32_t inId = 0;
  std::int32_t outId = 0;
  for (const auto &node : graph.getNodes()) {
    if (node.type == NodeType::audioInput)
      inId = node.id;
    if (node.type == NodeType::audioOutput)
      outId = node.id;
  }
  const auto monoMessage = label + " host I/O must be mono";
  if (!expect(inId != 0 && outId != 0 && graph.setProperty(inId, "channels", 0) &&
                  graph.setProperty(outId, "channels", 0),
              monoMessage.c_str()))
    return false;
  juce::String importError;
  const auto rootId = graph.importBox(snapshot, {200.0f, 0.0f}, true, importError);
  const auto importMessage = importError.isEmpty()
                                 ? label + " library box must import"
                                 : importError.toStdString();
  if (!expect(rootId.has_value(), importMessage.c_str()))
    return false;
  const auto *group = graph.findGroup(*rootId);
  std::int32_t inputHubId = 0;
  std::int32_t outputHubId = 0;
  if (group != nullptr) {
    for (const auto memberId : group->memberIds) {
      const auto *node = graph.findNode(memberId);
      if (node == nullptr)
        continue;
      if (node->type == NodeType::groupInput)
        inputHubId = node->id;
      else if (node->type == NodeType::groupOutput)
        outputHubId = node->id;
    }
  }
  const auto *inNode = graph.findNode(inId);
  const auto *outNode = graph.findNode(outId);
  const auto *inputHub = graph.findNode(inputHubId);
  const auto *outputHub = graph.findNode(outputHubId);
  const auto hubMessage = label + " library box must expose I/O hubs";
  if (!expect(inNode != nullptr && outNode != nullptr && inputHub != nullptr &&
                  outputHub != nullptr,
              hubMessage.c_str()))
    return false;
  const auto inLink =
      graph.connect(inNode->outputs.front().id, inputHub->inputs.front().id);
  const auto inLinkMessage = "Audio In to " + label;
  if (!expect(inLink.accepted,
              inLink.accepted ? inLinkMessage.c_str() : inLink.message.c_str()))
    return false;
  const auto outLink =
      graph.connect(outputHub->outputs.front().id, outNode->inputs.front().id);
  const auto outLinkMessage = label + " to Audio Out";
  return expect(outLink.accepted,
                outLink.accepted ? outLinkMessage.c_str()
                                 : outLink.message.c_str());
}

/**
 * @brief Finds one direct fixed interface hub in a group.
 * @param graph Graph containing the group.
 * @param groupId Group whose direct members are inspected.
 * @param type Boundary type to find.
 * @return Hub id, or zero when absent.
 */
std::int32_t findBoundary(const openyourbox::graph::NodeGraph &graph,
                          std::int32_t groupId,
                          openyourbox::graph::NodeType type) {
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return 0;
  for (const auto memberId : group->memberIds) {
    const auto *node = graph.findNode(memberId);
    if (node != nullptr && node->type == type)
      return node->id;
  }
  return 0;
}

/**
 * @brief Connects the first declared lane around an existing member chain.
 * @param graph Graph to mutate.
 * @param groupId Group owning its mandatory hubs.
 * @param firstNodeId First processing node in the chain.
 * @param lastNodeId Last processing node in the chain.
 * @return True when both internal boundary links were accepted.
 */
bool connectDeclaredThrough(openyourbox::graph::NodeGraph &graph,
                            std::int32_t groupId, std::int32_t firstNodeId,
                            std::int32_t lastNodeId) {
  using openyourbox::graph::NodeType;
  const auto inputId = findBoundary(graph, groupId, NodeType::groupInput);
  const auto outputId = findBoundary(graph, groupId, NodeType::groupOutput);
  const auto *input = graph.findNode(inputId);
  const auto *output = graph.findNode(outputId);
  const auto *first = graph.findNode(firstNodeId);
  const auto *last = graph.findNode(lastNodeId);
  if (input == nullptr || output == nullptr || first == nullptr ||
      last == nullptr || input->outputs.empty() || output->inputs.empty() ||
      first->inputs.empty() || last->outputs.empty())
    return false;
  return graph.connect(input->outputs.front().id, first->inputs.front().id)
             .accepted &&
         graph.connect(last->outputs.front().id, output->inputs.front().id)
             .accepted;
}
} // namespace

/**
 * @brief Runs group membership, repeats, and ValueTree round-trip checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultNewBoxPosition;
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

  passed &= expect(graph.setProperty(convA, "kernel_size", 5),
                   "scalar kernel size can be written once");
  passed &= expect(graph.setProperty(convA, "kernel_size", 9),
                   "a later scalar write is accepted");
  {
    const auto *node = graph.findNode(convA);
    const openyourbox::graph::NodeProperty *kernel = nullptr;
    if (node != nullptr) {
      for (const auto &property : node->properties) {
        if (property.key == "kernel_size")
          kernel = &property;
      }
    }
    passed &= expect(kernel != nullptr && kernel->value == 9,
                     "repeated scalar property writes keep the latest value");
  }

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
  passed &= expect(group != nullptr && !group->collapsed && group->repeats == 1,
                   "new groups start expanded with N=1");
  const auto groupInput =
      findBoundary(graph, created.groupId, NodeType::groupInput);
  const auto groupOutput =
      findBoundary(graph, created.groupId, NodeType::groupOutput);
  passed &= expect(groupInput != 0 && groupOutput != 0,
                   "new groups receive explicit input and output hubs");
  passed &= expect(!graph.removeNode(groupInput) &&
                       !graph.removeFromGroup(groupOutput).accepted,
                   "group boundary hubs cannot be removed");
  juce::String boundarySaveError;
  passed &= expect(!graph.exportBox(groupInput, boundarySaveError).isValid(),
                   "group boundary hubs cannot be saved independently");
  passed &= expect(group != nullptr && group->size.x >= 160.0f &&
                       group->size.y >= 120.0f,
                   "new groups fit their members");
  {
    using openyourbox::graph::groupBoundaryContentGap;
    using openyourbox::graph::groupBoxDisplayLabel;
    const auto *inputHub = graph.findNode(groupInput);
    const auto *outputHub = graph.findNode(groupOutput);
    const auto *left = graph.findNode(convA);
    const auto *right = graph.findNode(convB);
    passed &= expect(inputHub != nullptr && outputHub != nullptr &&
                         left != nullptr && right != nullptr,
                     "group interior boxes exist for layout");
    if (inputHub != nullptr && outputHub != nullptr && left != nullptr &&
        right != nullptr) {
      passed &= expect(
          std::abs((right->position.x - left->position.x) - 200.0f) < 0.5f &&
              std::abs(right->position.y - left->position.y) < 0.5f,
          "grouping preserves relative member layout");
      const auto contentMinX = std::min(left->position.x, right->position.x);
      const auto contentMaxX =
          std::max(left->position.x + std::max(8.0f, left->size.x),
                    right->position.x + std::max(8.0f, right->size.x));
      const auto contentMinY = std::min(left->position.y, right->position.y);
      const auto contentMaxY =
          std::max(left->position.y + std::max(8.0f, left->size.y),
                    right->position.y + std::max(8.0f, right->size.y));
      passed &= expect(inputHub->position.x +
                               std::max(8.0f, inputHub->size.x) +
                               groupBoundaryContentGap <=
                           contentMinX + 0.5f,
                       "Group Input sits left of content with a gap");
      passed &= expect(outputHub->position.x + 0.5f >=
                           contentMaxX + groupBoundaryContentGap,
                       "Group Output sits right of content with a gap");
      const auto contentCenterY = (contentMinY + contentMaxY) * 0.5f;
      passed &= expect(std::abs(inputHub->position.y +
                                    std::max(8.0f, inputHub->size.y) * 0.5f -
                                    contentCenterY) < 0.5f,
                       "Group Input is vertically centred on content");
      passed &= expect(std::abs(outputHub->position.y +
                                    std::max(8.0f, outputHub->size.y) * 0.5f -
                                    contentCenterY) < 0.5f,
                       "Group Output is vertically centred on content");
    }
    openyourbox::graph::GraphGroup named;
    named.name = "Encoder";
    passed &= expect(groupBoxDisplayLabel(named) == "Encoder (Block)",
                     "group boxes display Name (Block)");
    passed &= expect(group != nullptr &&
                         groupBoxDisplayLabel(*group) == "Group (Block)",
                     "default group boxes display Group (Block)");
  }

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

  const auto repeats = graph.setGroupRepeats(created.groupId, 3);
  passed &= expect(repeats.accepted, "chainable group accepts N=3");
  passed &= expect(graph.findGroup(created.groupId)->repeats == 3,
                   "repeats persist on the group");
  {
    const auto *repeatedGroup = graph.findGroup(created.groupId);
    const auto repeatStatus = graph.groupRepeatStatus(created.groupId);
    passed &= expect(repeatedGroup != nullptr && repeatStatus.active &&
                         groupBoxDisplayLabel(*repeatedGroup, repeatStatus) ==
                             u8"Group (Block) \xC3\x97" "3",
                     "active repeats append ×N to the group box title");
  }
  passed &= expect(graph.effectiveRepeatCount(convA) == 3,
                   "members report product of ancestor repeats");
  passed &= expect(graph.findNode(convA)->repeatSlots.size() == 3 &&
                       graph.findNode(convB)->repeatSlots.size() == 3,
                   "members hold independent weight slots per repeat");
  graph.findNode(convA)->repeatSlots[1].seed = 99;
  graph.findNode(convA)->repeatSlots[2].seed = 100;

  {
    using openyourbox::graph::NodeProperty;
    using openyourbox::graph::PropertyKind;
    using openyourbox::graph::parsePropertyRepeatList;
    NodeProperty gain;
    gain.key = "gain";
    gain.label = "Gain";
    gain.kind = PropertyKind::real;
    gain.floatValue = 1.0f;
    gain.floatMinimum = openyourbox::graph::gainMinimum;
    gain.floatMaximum = openyourbox::graph::gainMaximum;
    const auto parsed = parsePropertyRepeatList(gain, 3, "0.50, 1.00, 1.50");
    passed &= expect(parsed.accepted && parsed.floatValues.size() == 3 &&
                         std::abs(parsed.floatValues[1] - 1.0f) < 1.0e-5f,
                     "comma-separated reals parse into per-repeat values");
    const auto broadcast = parsePropertyRepeatList(gain, 3, "0.25");
    passed &= expect(broadcast.accepted && broadcast.floatValues.size() == 1 &&
                         std::abs(broadcast.floatValues.front() - 0.25f) < 1.0e-5f,
                     "a single value is stored as authored length 1");
    {
      auto tiled = gain;
      tiled.repeatFloatValues = broadcast.floatValues;
      passed &= expect(std::abs(openyourbox::graph::floatValueForRepeat(tiled, 2) -
                                0.25f) < 1.0e-5f,
                       "authored length 1 tiles across expanded repeat slots");
    }
    const auto commaDecimal = parsePropertyRepeatList(gain, 3, "0,5");
    passed &= expect(!commaDecimal.accepted,
                     "comma is a list separator, not a decimal mark");
    const auto wrongCount = parsePropertyRepeatList(gain, 3, "0.5, 1.0");
    passed &= expect(!wrongCount.accepted,
                     "lists that are not 1 or N values are refused");
  }
  passed &= expect(graph.setPropertyRepeatValues(convA, "kernel_size", {3, 5, 7}),
                   "grouped integer properties accept N values");
  passed &= expect(!graph.setPropertyRepeatValues(convA, "kernel_size", {3, 5}),
                   "wrong-sized repeat lists are refused");
  {
    const auto *node = graph.findNode(convA);
    const openyourbox::graph::NodeProperty *kernel = nullptr;
    if (node != nullptr) {
      for (const auto &property : node->properties) {
        if (property.key == "kernel_size")
          kernel = &property;
      }
    }
    passed &= expect(kernel != nullptr && kernel->repeatIntValues.size() == 3 &&
                         kernel->repeatIntValues[0] == 3 &&
                         kernel->repeatIntValues[1] == 5 &&
                         kernel->repeatIntValues[2] == 7,
                     "per-repeat integer values are stored on the property");
  }

  const auto expanded = graph.withInvisibleRepeatsMaterialized();
  int convolutionCount = 0;
  for (const auto &node : expanded.getNodes()) {
    if (node.type == NodeType::convolution)
      ++convolutionCount;
  }
  passed &= expect(convolutionCount == 6,
                   "DSP unroll materializes two extra repeats of each member");
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
                     "DSP unroll applies per-repeat integer properties");
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
                     "unrolled repeats must remain acyclic");
  }
  passed &= expect(graph.getNodes().size() == 6,
                   "UI graph shows one template plus its two boundary hubs");

  const auto nestedOne = graph.createGroup({convA});
  passed &= expect(nestedOne.accepted,
                   "a single already-grouped box can form a nested group");
  passed &= expect(graph.ungroup(nestedOne.groupId).accepted,
                   "ungroup the nested group-of-one before later checks");

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
          port.memberNodeId == groupInput)
        hasInput = true;
      if (port.kind == openyourbox::graph::PinKind::output &&
          port.memberNodeId == groupOutput)
        hasOutput = true;
      if (port.kind == openyourbox::graph::PinKind::output &&
          port.memberNodeId == convA)
        hasInternalOutput = true;
    }
    passed &= expect(hasInput && hasOutput,
                     "group I/O pins come from explicit boundary hubs");
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
                       restored.findGroup(created.groupId)->repeats == 3,
                   "repeats survive round-trip");
  passed &= expect(restored.findNode(convA) != nullptr &&
                       restored.findNode(convA)->parentGroupId == created.groupId,
                   "parentGroupId survives round-trip");
  passed &= expect(restored.findNode(convA)->repeatSlots.size() == 3 &&
                       restored.findNode(convA)->repeatSlots[1].seed == 99,
                   "per-repeat weights survive round-trip");
  {
    const auto *node = restored.findNode(convA);
    const openyourbox::graph::NodeProperty *kernel = nullptr;
    if (node != nullptr) {
      for (const auto &property : node->properties) {
        if (property.key == "kernel_size")
          kernel = &property;
      }
    }
    passed &= expect(kernel != nullptr && kernel->repeatIntValues.size() == 3 &&
                         kernel->repeatIntValues[1] == 5 &&
                         kernel->repeatIntValues[2] == 7,
                     "per-repeat integer properties survive round-trip");
  }
  passed &= expect(restored.findGroup(created.groupId) != nullptr &&
                       std::abs(restored.findGroup(created.groupId)->viewPan.x -
                                12.0f) < 0.01f &&
                       std::abs(restored.findGroup(created.groupId)->viewZoom -
                                1.5f) < 0.01f,
                   "group inner camera survives round-trip");

  {
    auto legacyTree = tree.createCopy();
    std::unordered_set<std::int32_t> boundaryPins;
    boundaryPins.reserve(4);
    for (const auto boundaryId : {groupInput, groupOutput}) {
      const auto *boundary = graph.findNode(boundaryId);
      if (boundary == nullptr)
        continue;
      for (const auto &pin : boundary->inputs)
        boundaryPins.insert(pin.id);
      for (const auto &pin : boundary->outputs)
        boundaryPins.insert(pin.id);
    }
    for (int index = legacyTree.getNumChildren() - 1; index >= 0; --index) {
      auto child = legacyTree.getChild(index);
      if (child.hasType("Node")) {
        const auto type = child["type"].toString();
        if (type == "group_input" || type == "group_output")
          legacyTree.removeChild(index, nullptr);
      } else if (child.hasType("Link")) {
        const auto source = static_cast<std::int32_t>(child["sourcePin"]);
        const auto destination =
            static_cast<std::int32_t>(child["destinationPin"]);
        if (boundaryPins.count(source) != 0 ||
            boundaryPins.count(destination) != 0)
          legacyTree.removeChild(index, nullptr);
      } else if (child.hasType("Group") &&
                 static_cast<std::int32_t>(child["id"]) == created.groupId) {
        for (int memberIndex = child.getNumChildren() - 1; memberIndex >= 0;
             --memberIndex) {
          const auto member = child.getChild(memberIndex);
          const auto memberId = static_cast<std::int32_t>(member["id"]);
          if (memberId == groupInput || memberId == groupOutput)
            child.removeChild(memberIndex, nullptr);
        }
      }
    }
    juce::ValueTree legacyInputLink{"Link"};
    legacyInputLink.setProperty("id", 90001, nullptr);
    legacyInputLink.setProperty(
        "sourcePin", graph.findNode(input)->outputs.front().id, nullptr);
    legacyInputLink.setProperty(
        "destinationPin", graph.findNode(convA)->inputs.front().id, nullptr);
    legacyTree.appendChild(legacyInputLink, nullptr);
    juce::ValueTree legacyOutputLink{"Link"};
    legacyOutputLink.setProperty("id", 90002, nullptr);
    legacyOutputLink.setProperty(
        "sourcePin", graph.findNode(convB)->outputs.front().id, nullptr);
    legacyOutputLink.setProperty(
        "destinationPin", graph.findNode(output)->inputs.front().id, nullptr);
    legacyTree.appendChild(legacyOutputLink, nullptr);

    NodeGraph migrated;
    passed &= expect(migrated.restoreFromValueTree(legacyTree),
                     "legacy inferred-interface document migrates");
    passed &= expect(
        findBoundary(migrated, created.groupId, NodeType::groupInput) != 0 &&
            findBoundary(migrated, created.groupId, NodeType::groupOutput) != 0 &&
            migrated.groupInterfacePorts(created.groupId).size() == 2,
        "legacy migration creates explicit hubs and preserves two lanes");
  }

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
      passed &= expect(connectDeclaredThrough(tcnGraph, grouped.groupId, first,
                                              second),
                       "TCN chain connects to its declared boundary");
      const auto tcnRepeats = tcnGraph.setGroupRepeats(grouped.groupId, 3);
      passed &= expect(tcnRepeats.accepted,
                       "TCN chain stores requested repeats without host cables");
      const auto unrolled = tcnGraph.withInvisibleRepeatsMaterialized();
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
                       "TCN repeats unroll to six independent nodes");
      passed &= expect(controlFeeds == 0,
                       "serial repeats must not wire into TCN control pins");
      passed &= expect(serialAudioFeeds == 5,
                       "three internal TCN cables plus two inter-repeat audio cables");
    }
  }

  {
    NodeGraph knobs;
    const auto left = knobs.addNode(NodeType::knobInput, {0.0f, 0.0f});
    const auto right = knobs.addNode(NodeType::knobInput, {80.0f, 0.0f});
    const auto grouped = knobs.createGroup({left, right});
    passed &= expect(grouped.accepted, "two knobs may group");
    const auto requested = knobs.setGroupRepeats(grouped.groupId, 2);
    const auto status = knobs.groupRepeatStatus(grouped.groupId);
    passed &= expect(requested.accepted && !status.active &&
                         status.requestedRepeats == 2 &&
                         status.effectiveRepeats == 1,
                     "invalid requested N persists while one repeat runs");
    if (const auto *knobGroup = knobs.findGroup(grouped.groupId)) {
      passed &= expect(groupBoxDisplayLabel(*knobGroup, status) ==
                           u8"Group (Block) \xC3\x97" "2\xE2\x86\x92" "1",
                       "inactive repeats append ×N→1 to the group box title");
    }
    const auto unrolled = knobs.withInvisibleRepeatsMaterialized();
    int knobCount = 0;
    for (const auto &node : unrolled.getNodes()) {
      if (node.type == NodeType::knobInput)
        ++knobCount;
    }
    passed &= expect(knobCount == 2,
                     "inactive repeats do not materialize in execution");
  }

  {
    NodeGraph residual;
    const auto body =
        residual.addNode(NodeType::activation, {160.0f, 0.0f});
    const auto add = residual.addNode(NodeType::merge, {340.0f, 0.0f});
    const auto grouped = residual.createGroup({body, add});
    passed &= expect(grouped.accepted, "residual fixture groups");
    const auto inputId =
        findBoundary(residual, grouped.groupId, NodeType::groupInput);
    const auto outputId =
        findBoundary(residual, grouped.groupId, NodeType::groupOutput);
    const auto *inputHub = residual.findNode(inputId);
    const auto *outputHub = residual.findNode(outputId);
    const auto *bodyNode = residual.findNode(body);
    const auto *addNode = residual.findNode(add);
    if (inputHub != nullptr && outputHub != nullptr && bodyNode != nullptr &&
        addNode != nullptr && addNode->inputs.size() >= 2) {
      passed &= expect(
          residual
                  .connect(inputHub->outputs.front().id,
                           bodyNode->inputs.front().id)
                  .accepted &&
              residual
                  .connect(bodyNode->outputs.front().id,
                           addNode->inputs.front().id)
                  .accepted &&
              residual
                  .connect(inputHub->outputs.front().id, addNode->inputs[1].id)
                  .accepted &&
              residual
                  .connect(addNode->outputs.front().id,
                           outputHub->inputs.front().id)
                  .accepted,
          "declared input fans out into an acyclic residual join");
    }
    passed &= expect(residual.setGroupRepeats(grouped.groupId, 2).accepted &&
                         residual.groupRepeatStatus(grouped.groupId).active,
                     "residual group activates serial repeats");
    const auto expanded = residual.withInvisibleRepeatsMaterialized();
    bool hasBoundary = false;
    for (const auto &node : expanded.getNodes())
      hasBoundary =
          hasBoundary || openyourbox::graph::isGroupBoundaryType(node.type);
    passed &= expect(!hasBoundary,
                     "execution graph flattens editor-only boundary hubs");
  }

  {
    NodeGraph nested;
    const auto bodyAct =
        nested.addNode(NodeType::activation, {160.0f, 40.0f});
    const auto bodyConv =
        nested.addNode(NodeType::convolution, {300.0f, 40.0f});
    const auto join = nested.addNode(NodeType::merge, {460.0f, 0.0f});
    nested.setProperty(join, "inputs", 2);
    nested.connect(nested.findNode(bodyAct)->outputs.front().id,
                   nested.findNode(bodyConv)->inputs.front().id);
    const auto layer = nested.createGroup({bodyAct, bodyConv});
    passed &= expect(layer.accepted &&
                         connectDeclaredThrough(nested, layer.groupId, bodyAct,
                                                bodyConv),
                     "nested residual layer groups and wires through");
    const auto layerOut =
        findBoundary(nested, layer.groupId, NodeType::groupOutput);
    const auto *joinNode = nested.findNode(join);
    const auto *layerOutNode = nested.findNode(layerOut);
    passed &= expect(joinNode != nullptr && layerOutNode != nullptr &&
                         nested
                             .connect(layerOutNode->outputs.front().id,
                                      joinNode->inputs.front().id)
                             .accepted,
                     "layer output feeds the residual join");
    const auto stack = nested.createGroup({layer.groupId, join});
    const auto stackIn =
        findBoundary(nested, stack.groupId, NodeType::groupInput);
    const auto stackOut =
        findBoundary(nested, stack.groupId, NodeType::groupOutput);
    const auto layerIn =
        findBoundary(nested, layer.groupId, NodeType::groupInput);
    const auto *stackInNode = nested.findNode(stackIn);
    const auto *stackOutNode = nested.findNode(stackOut);
    const auto *layerInNode = nested.findNode(layerIn);
    const auto *stackJoin = nested.findNode(join);
    passed &= expect(
        stack.accepted && stackInNode != nullptr && stackOutNode != nullptr &&
            layerInNode != nullptr && stackJoin != nullptr &&
            stackJoin->inputs.size() >= 2 &&
            nested
                .connect(stackInNode->outputs.front().id,
                         layerInNode->inputs.front().id)
                .accepted &&
            nested
                .connect(stackInNode->outputs.front().id,
                         stackJoin->inputs[1].id)
                .accepted &&
            nested
                .connect(stackJoin->outputs.front().id,
                         stackOutNode->inputs.front().id)
                .accepted,
        "stack Group Input fans out into layer and skip join");
    passed &= expect(nested.setGroupRepeats(layer.groupId, 2).accepted &&
                         nested.groupRepeatStatus(layer.groupId).active &&
                         nested.setGroupRepeats(stack.groupId, 3).accepted &&
                         nested.groupRepeatStatus(stack.groupId).active,
                     "nested residual repeats N=2 inside N=3 activate");
    const auto expanded = nested.withInvisibleRepeatsMaterialized();
    int activations = 0;
    int convolutions = 0;
    int joins = 0;
    bool hasBoundary = false;
    for (const auto &node : expanded.getNodes()) {
      if (node.type == NodeType::activation)
        ++activations;
      else if (node.type == NodeType::convolution)
        ++convolutions;
      else if (node.type == NodeType::merge)
        ++joins;
      hasBoundary =
          hasBoundary || openyourbox::graph::isGroupBoundaryType(node.type);
    }
    passed &= expect(!hasBoundary && activations == 6 && convolutions == 6 &&
                         joins == 3,
                     "nested residual unroll keeps a full inner chain per outer repeat");
  }

  {
    NodeGraph interfaceGraph;
    const auto first =
        interfaceGraph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto second =
        interfaceGraph.addNode(NodeType::activation, {260.0f, 0.0f});
    const auto grouped = interfaceGraph.createGroup({first, second});
    const auto inputId =
        findBoundary(interfaceGraph, grouped.groupId, NodeType::groupInput);
    passed &= expect(interfaceGraph.setProperty(inputId, "ports", 2),
                     "Group Input can add a declared lane");
    const auto *inputHub = interfaceGraph.findNode(inputId);
    passed &= expect(inputHub != nullptr && inputHub->inputs.size() == 2 &&
                         inputHub->outputs.size() == 2,
                     "boundary lane count resizes paired pins");
    if (inputHub != nullptr) {
      passed &= expect(
          interfaceGraph
              .connect(inputHub->outputs[1].id,
                       interfaceGraph.findNode(first)->inputs.front().id)
              .accepted,
          "new boundary lane can feed group processing");
      passed &= expect(!interfaceGraph.setProperty(inputId, "ports", 1),
                       "connected boundary lanes cannot be removed");
      std::int32_t laneLink = 0;
      for (const auto &link : interfaceGraph.getLinks()) {
        if (link.sourcePinId == inputHub->outputs[1].id)
          laneLink = link.id;
      }
      passed &= expect(laneLink != 0 && interfaceGraph.removeLink(laneLink) &&
                           interfaceGraph.setProperty(inputId, "ports", 1),
                       "disconnected trailing boundary lanes can be removed");
    }
  }

  {
    NodeGraph mismatch;
    const auto input =
        mismatch.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto linear = mismatch.addNode(NodeType::linear, {120.0f, 0.0f});
    const auto inside =
        mismatch.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto outside =
        mismatch.addNode(NodeType::activation, {420.0f, 0.0f});
    mismatch.setProperty(linear, "features", 4);
    mismatch.connect(mismatch.findNode(input)->outputs.front().id,
                     mismatch.findNode(linear)->inputs.front().id);
    mismatch.connect(mismatch.findNode(linear)->outputs.front().id,
                     mismatch.findNode(inside)->inputs.front().id);
    mismatch.connect(mismatch.findNode(inside)->outputs.front().id,
                     mismatch.findNode(outside)->inputs.front().id);
    const auto grouped = mismatch.createGroup({linear, inside});
    mismatch.setGroupRepeats(grouped.groupId, 3);
    const auto status = mismatch.groupRepeatStatus(grouped.groupId);
    passed &= expect(status.active && status.effectiveRepeats == 3,
                     "width-changing feedforward repeats chain without matching I/O");
    const auto *groupedLinear = mismatch.findNode(linear);
    passed &= expect(
        groupedLinear != nullptr && !groupedLinear->outputs.empty() &&
            groupedLinear->outputs.front().repeatShapes.size() == 3 &&
            groupedLinear->outputs.front().repeatShapes[0].channels == 4 &&
            groupedLinear->outputs.front().repeatShapes[1].channels == 4 &&
            groupedLinear->outputs.front().repeatShapes[2].channels == 4,
        "feedforward Linear lists 4ch on every repeat");
  }

  {
    NodeGraph residualMismatch;
    const auto input =
        residualMismatch.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto linear =
        residualMismatch.addNode(NodeType::linear, {120.0f, 0.0f});
    const auto add =
        residualMismatch.addNode(NodeType::merge, {300.0f, 0.0f});
    residualMismatch.setProperty(linear, "features", 4);
    residualMismatch.setProperty(add, "inputs", 2);
    const auto grouped = residualMismatch.createGroup({linear, add});
    passed &= expect(grouped.accepted, "residual mismatch fixture groups");
    const auto inputId =
        findBoundary(residualMismatch, grouped.groupId, NodeType::groupInput);
    const auto outputId =
        findBoundary(residualMismatch, grouped.groupId, NodeType::groupOutput);
    const auto *audio = residualMismatch.findNode(input);
    const auto *inputHub = residualMismatch.findNode(inputId);
    const auto *outputHub = residualMismatch.findNode(outputId);
    const auto *linearNode = residualMismatch.findNode(linear);
    const auto *addNode = residualMismatch.findNode(add);
    passed &= expect(audio != nullptr && inputHub != nullptr &&
                         outputHub != nullptr && linearNode != nullptr &&
                         addNode != nullptr && addNode->inputs.size() >= 2,
                     "residual mismatch nodes exist");
    if (audio != nullptr && inputHub != nullptr && outputHub != nullptr &&
        linearNode != nullptr && addNode != nullptr &&
        addNode->inputs.size() >= 2) {
      passed &= expect(
          residualMismatch
                  .connect(inputHub->outputs.front().id,
                           linearNode->inputs.front().id)
                  .accepted &&
              residualMismatch
                  .connect(linearNode->outputs.front().id,
                           addNode->inputs.front().id)
                  .accepted &&
              residualMismatch
                  .connect(inputHub->outputs.front().id, addNode->inputs[1].id)
                  .accepted &&
              residualMismatch
                  .connect(addNode->outputs.front().id,
                           outputHub->inputs.front().id)
                  .accepted,
          "residual skip joins group input with Linear before host audio");
      residualMismatch.setGroupRepeats(grouped.groupId, 3);
      passed &= expect(
          residualMismatch
              .connect(audio->outputs.front().id, inputHub->inputs.front().id)
              .accepted,
          "host audio can enter the residual group");
    }
    const auto status = residualMismatch.groupRepeatStatus(grouped.groupId);
    const auto hint = residualMismatch.groupRepeatPropertyHint(linear, "features");
    passed &= expect(!status.active && status.effectiveRepeats == 1 &&
                         hint.has_value() &&
                         hint->find("use 'in'") != std::string::npos,
                     "residual skip mismatch flags the property that can preserve input");
    const auto warnings = residualMismatch.collectGraphWarnings();
    passed &=
        expect(!warnings.empty() &&
                   warnings.front().find("repeats are inactive") !=
                       std::string::npos,
               "inactive residual repeats surface as graph warnings");
    passed &= expect(
        residualMismatch.setPropertyPreserveIn(linear, "features", 1) &&
            residualMismatch.groupRepeatStatus(grouped.groupId).active &&
            residualMismatch.collectGraphWarnings().empty(),
        "fixing the residual skip with 'in' activates N");
  }

  {
    using openyourbox::graph::WeightsProvenance;
    using openyourbox::graph::seedForRepeatSlot;
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
      weights.setGroupRepeats(grouped.groupId, 2);
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
        passed &= expect(locked->repeatSlots.size() == 2 &&
                             locked->repeatSlots.front().seed == 777 &&
                             locked->repeatSlots.back().seed == 778,
                         "seed-locked repeat slots use base + i");
        passed &= expect(reset->repeatSlots.size() == 2 &&
                             reset->repeatSlots.front().seed == 0 &&
                             reset->repeatSlots.back().seed == 0,
                         "batch-norm repeat slots reset to identity");
        passed &= expect(unlocked->repeatSlots.size() == 2 &&
                             unlocked->repeatSlots.front().seed == unlocked->seed &&
                             unlocked->repeatSlots.back().seed ==
                                 seedForRepeatSlot(unlocked->seed, 1),
                         "unlocked repeat slots use base + i");
      }
    }
  }

  {
    using openyourbox::graph::WeightsProvenance;
    using openyourbox::graph::ensureRepeatSlotCount;
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
    passed &= expect(connectDeclaredThrough(elementRandomize, grouped.groupId,
                                            linearId, actId),
                     "element-randomize chain connects to its boundary");
    auto *linear = elementRandomize.findNode(linearId);
    passed &= expect(linear != nullptr, "element-randomize linear exists");
    if (linear != nullptr) {
      linear->seed = 100;
      linear->weightsProvenance = WeightsProvenance::random;
      ensureRepeatSlotCount(*linear, 1);
    }
    passed &=
        expect(elementRandomize.setGroupRepeats(grouped.groupId, 3).accepted,
               "element-randomize fixture sets N=3");
    linear = elementRandomize.findNode(linearId);
    passed &= expect(linear != nullptr && linear->repeatSlots.size() == 3,
                     "N=3 creates three repeat slots");
    if (linear != nullptr) {
      passed &= expect(linear->repeatSlots[0].seed == 100 &&
                           linear->repeatSlots[1].seed == 101 &&
                           linear->repeatSlots[2].seed == 102,
                       "raising N derives seed + i for random slots");
      passed &= expect(elementRandomize.clearWeightsToSeed(linearId, 12345),
                       "element clearWeightsToSeed succeeds");
      linear = elementRandomize.findNode(linearId);
      passed &= expect(linear != nullptr && linear->seed == 12345 &&
                           linear->repeatSlots.size() == 3 &&
                           linear->repeatSlots[0].seed == 12345 &&
                           linear->repeatSlots[1].seed == 12346 &&
                           linear->repeatSlots[2].seed == 12347,
                       "element randomize writes seed + i across repeats");
      passed &= expect(elementRandomize.setFloatPropertyRepeatValues(
                           actId, "gain", {0.5f, 1.0f, 1.5f}),
                       "grouped real properties accept N values");
      const auto expandedGains =
          elementRandomize.withInvisibleRepeatsMaterialized();
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
                       "DSP unroll applies per-repeat real properties");
    }
  }

  {
    NodeGraph shapeGraph;
    const auto inId =
        shapeGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto linearId =
        shapeGraph.addNode(NodeType::linear, {200.0f, 0.0f});
    const auto actId =
        shapeGraph.addNode(NodeType::activation, {320.0f, 0.0f});
    const auto outId =
        shapeGraph.addNode(NodeType::audioOutput, {480.0f, 0.0f});
    const auto *inNode = shapeGraph.findNode(inId);
    const auto *linearNode = shapeGraph.findNode(linearId);
    const auto *actNode = shapeGraph.findNode(actId);
    const auto *outNode = shapeGraph.findNode(outId);
    passed &= expect(inNode != nullptr && linearNode != nullptr &&
                         actNode != nullptr && outNode != nullptr,
                     "shape-list graph nodes exist");
    if (inNode != nullptr && linearNode != nullptr && actNode != nullptr &&
        outNode != nullptr) {
      passed &= expect(shapeGraph.setProperty(linearId, "features", 2),
                       "linear features match stereo width for chaining");
      passed &= expect(shapeGraph
                           .connect(inNode->outputs.front().id,
                                    linearNode->inputs.front().id)
                           .accepted &&
                           shapeGraph
                               .connect(linearNode->outputs.front().id,
                                        actNode->inputs.front().id)
                               .accepted &&
                           shapeGraph
                               .connect(actNode->outputs.front().id,
                                        outNode->inputs.front().id)
                               .accepted,
                       "shape-list graph cables");
      const auto grouped = shapeGraph.createGroup({linearId, actId});
      passed &= expect(grouped.accepted, "linear group for shape lists");
      passed &= expect(shapeGraph.setGroupRepeats(grouped.groupId, 3).accepted,
                       "linear with matching I/O accepts N=3");
      passed &= expect(
          shapeGraph.setPropertyRepeatValues(linearId, "features", {2, 4, 8}),
          "per-repeat features 2,4,8");
      const auto *linear = shapeGraph.findNode(linearId);
      passed &= expect(
          linear != nullptr && !linear->outputs.empty() &&
              linear->outputs.front().repeatShapes.size() == 3 &&
              linear->outputs.front().repeatShapes[0].channels == 2 &&
              linear->outputs.front().repeatShapes[1].channels == 4 &&
              linear->outputs.front().repeatShapes[2].channels == 8,
          "element output lists a shape per repeat");
      const auto listLabel = openyourbox::graph::formatShapeRepeatList(
          linear->outputs.front().repeatShapes, linear->outputs.front().shape,
          {3});
      passed &= expect(listLabel == "[2ch, 4ch, 8ch]",
                       "element display label groups per-repeat shapes");
      openyourbox::graph::ShapeSignature groupOutShape;
      for (const auto &port : shapeGraph.groupInterfacePorts(grouped.groupId)) {
        if (port.kind != openyourbox::graph::PinKind::output)
          continue;
        if (const auto *memberPin = shapeGraph.findPin(port.memberPinId);
            memberPin != nullptr && !memberPin->repeatShapes.empty())
          groupOutShape = memberPin->repeatShapes.back();
        else
          groupOutShape = port.shape;
      }
      passed &= expect(groupOutShape.channels == 8,
                       "group output displays shape after all repeats");
      const auto *innerOutHub = shapeGraph.findNode(
          findBoundary(shapeGraph, grouped.groupId, NodeType::groupOutput));
      passed &= expect(
          innerOutHub != nullptr && !innerOutHub->outputs.empty() &&
              openyourbox::graph::formatCollapsedGroupPinShapes(
                  innerOutHub->outputs.front().repeatShapes,
                  innerOutHub->outputs.front().shape, {3}, true) == "8ch",
          "collapsed group output label is last repeat, not [2ch, 4ch, 8ch]");
      passed &= expect(innerOutHub != nullptr && !innerOutHub->outputs.empty() &&
                           innerOutHub->outputs.front().shape.channels == 8,
                       "group output hub shape is last-out for parent-canvas cables");
      passed &= expect(linear->outputs.front().shape.channels == 2,
                       "visible pin.shape stays first-repeat for chaining");
      outNode = shapeGraph.findNode(outId);
      if (outNode != nullptr && !outNode->inputs.empty()) {
        const auto outPin = outNode->inputs.front().id;
        std::vector<std::int32_t> stale;
        for (const auto &link : shapeGraph.getLinks()) {
          if (link.destinationPinId == outPin)
            stale.push_back(link.id);
        }
        for (const auto linkId : stale)
          shapeGraph.removeLink(linkId);
      }
      const auto downstreamId =
          shapeGraph.addNode(NodeType::convolution, {400.0f, 80.0f});
      const auto *downstream = shapeGraph.findNode(downstreamId);
      passed &= expect(
          innerOutHub != nullptr && !innerOutHub->outputs.empty() &&
              downstream != nullptr && !downstream->inputs.empty() &&
              shapeGraph
                  .connect(innerOutHub->outputs.front().id,
                           downstream->inputs.front().id)
                  .accepted,
          "group output connects to a downstream conv");
      downstream = shapeGraph.findNode(downstreamId);
      passed &= expect(downstream != nullptr && !downstream->inputs.empty() &&
                           downstream->inputs.front().shape.channels == 8,
                       "downstream conv input inherits last-out, not first-repeat");
    }
  }

  {
    NodeGraph convExit;
    const auto inId = convExit.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto convId =
        convExit.addNode(NodeType::convolution, {160.0f, 0.0f});
    const auto actId = convExit.addNode(NodeType::activation, {280.0f, 0.0f});
    const auto downstreamId =
        convExit.addNode(NodeType::convolution, {400.0f, 0.0f});
    const auto *inNode = convExit.findNode(inId);
    const auto *convNode = convExit.findNode(convId);
    const auto *actNode = convExit.findNode(actId);
    const auto *downstreamNode = convExit.findNode(downstreamId);
    passed &= expect(inNode != nullptr && convNode != nullptr &&
                         actNode != nullptr && downstreamNode != nullptr,
                     "conv last-out graph nodes exist");
    if (inNode != nullptr && convNode != nullptr && actNode != nullptr &&
        downstreamNode != nullptr) {
      passed &= expect(convExit
                           .connect(inNode->outputs.front().id,
                                    convNode->inputs.front().id)
                           .accepted &&
                           convExit
                               .connect(convNode->outputs.front().id,
                                        actNode->inputs.front().id)
                               .accepted &&
                           convExit
                               .connect(actNode->outputs.front().id,
                                        downstreamNode->inputs.front().id)
                               .accepted,
                       "audio to conv to act to downstream conv");
      const auto grouped = convExit.createGroup({convId, actId});
      passed &= expect(grouped.accepted, "group around conv [512, 128]");
      passed &= expect(convExit.setGroupRepeats(grouped.groupId, 2).accepted,
                       "conv group N=2");
      passed &= expect(
          convExit.setPropertyRepeatValues(convId, "channels", {512, 128}),
          "per-repeat conv channels 512,128");
      const auto convStatus = convExit.groupRepeatStatus(grouped.groupId);
      passed &= expect(convStatus.active && convStatus.effectiveRepeats == 2,
                       convStatus.message.empty()
                           ? "conv group N=2 is active"
                           : convStatus.message.c_str());
      convNode = convExit.findNode(convId);
      passed &= expect(
          convNode != nullptr && !convNode->outputs.empty() &&
              convNode->outputs.front().repeatShapes.size() == 2 &&
              convNode->outputs.front().repeatShapes[0].channels == 512 &&
              convNode->outputs.front().repeatShapes[1].channels == 128,
          "grouped conv lists per-repeat output shapes 512,128");
      const auto *outHub = convExit.findNode(
          findBoundary(convExit, grouped.groupId, NodeType::groupOutput));
      downstreamNode = convExit.findNode(downstreamId);
      passed &= expect(convNode != nullptr && !convNode->outputs.empty() &&
                           convNode->outputs.front().shape.channels == 512,
                       "grouped conv visible output stays first-repeat 512");
      passed &= expect(outHub != nullptr && !outHub->outputs.empty() &&
                           outHub->outputs.front().shape.channels == 128,
                       "group output pin is last-out 128");
      passed &= expect(
          downstreamNode != nullptr && !downstreamNode->inputs.empty() &&
              downstreamNode->inputs.front().shape.channels == 128,
          "downstream conv input inherits last-out 128, not first-repeat 512");
    }
  }

  {
    NodeGraph nestedShapes;
    const auto inId =
        nestedShapes.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto linearId =
        nestedShapes.addNode(NodeType::linear, {160.0f, 0.0f});
    const auto actId =
        nestedShapes.addNode(NodeType::activation, {320.0f, 0.0f});
    const auto extraId =
        nestedShapes.addNode(NodeType::activation, {400.0f, 0.0f});
    const auto outId =
        nestedShapes.addNode(NodeType::audioOutput, {560.0f, 0.0f});
    const auto *inNode = nestedShapes.findNode(inId);
    const auto *linearNode = nestedShapes.findNode(linearId);
    const auto *actNode = nestedShapes.findNode(actId);
    const auto *extraNode = nestedShapes.findNode(extraId);
    const auto *outNode = nestedShapes.findNode(outId);
    passed &= expect(inNode != nullptr && linearNode != nullptr &&
                         actNode != nullptr && extraNode != nullptr &&
                         outNode != nullptr,
                     "nested shape-list graph nodes exist");
    if (inNode != nullptr && linearNode != nullptr && actNode != nullptr &&
        extraNode != nullptr && outNode != nullptr) {
      passed &= expect(nestedShapes.setProperty(linearId, "features", 2),
                       "nested linear features match stereo width");
      passed &= expect(
          nestedShapes
              .connect(inNode->outputs.front().id,
                       linearNode->inputs.front().id)
              .accepted &&
              nestedShapes
                  .connect(linearNode->outputs.front().id,
                           actNode->inputs.front().id)
                  .accepted &&
              nestedShapes
                  .connect(actNode->outputs.front().id,
                           extraNode->inputs.front().id)
                  .accepted &&
              nestedShapes
                  .connect(extraNode->outputs.front().id,
                           outNode->inputs.front().id)
                  .accepted,
          "nested shape-list graph cables");
      const auto inner = nestedShapes.createGroup({linearId, actId});
      passed &= expect(inner.accepted, "inner group for nested shape lists");
      passed &= expect(
          nestedShapes.setGroupRepeats(inner.groupId, 3).accepted,
          "inner N=3 accepts matching I/O");
      passed &= expect(
          nestedShapes.setPropertyRepeatValues(linearId, "features", {2, 4, 8}),
          "inner per-repeat features 2,4,8");
      extraNode = nestedShapes.findNode(extraId);
      passed &= expect(extraNode != nullptr && !extraNode->inputs.empty() &&
                           extraNode->inputs.front().shape.channels == 8,
                       "node after inner group inherits last-out, not first-repeat");
      const auto outer =
          nestedShapes.createGroup({inner.groupId, extraId});
      passed &= expect(outer.accepted, "outer group wraps inner group and tail");
      passed &= expect(nestedShapes.setGroupRepeats(outer.groupId, 2).accepted &&
                           nestedShapes.groupRepeatStatus(outer.groupId).active,
                       "outer M=2 activates around inner N=3");
      const auto *innerOutHub = nestedShapes.findNode(
          findBoundary(nestedShapes, inner.groupId, NodeType::groupOutput));
      const auto *innerInHub = nestedShapes.findNode(
          findBoundary(nestedShapes, inner.groupId, NodeType::groupInput));
      passed &= expect(innerOutHub != nullptr && innerInHub != nullptr &&
                           !innerOutHub->outputs.empty() &&
                           !innerInHub->inputs.empty(),
                       "inner hubs exist after nesting");
      if (innerOutHub != nullptr && innerInHub != nullptr &&
          !innerOutHub->outputs.empty() && !innerInHub->inputs.empty()) {
        const auto &outPin = innerOutHub->outputs.front();
        const auto &inPin = innerInHub->inputs.front();
        passed &= expect(
            outPin.repeatShapes.size() == 6,
            "inner hub repeatShapes include outer×inner runtime slots");
        passed &= expect(
            openyourbox::graph::formatCollapsedGroupPinShapes(
                outPin.repeatShapes, outPin.shape, {2, 3}, true) == "[8ch, 8ch]",
            "collapsed inner output lists outer repeats of last-out");
        passed &= expect(
            openyourbox::graph::formatCollapsedGroupPinShapes(
                inPin.repeatShapes, inPin.shape, {2, 3}, false) == "[2ch, 8ch]",
            "collapsed inner input lists outer repeats of first-in");
      }
      const auto *outerOutHub = nestedShapes.findNode(
          findBoundary(nestedShapes, outer.groupId, NodeType::groupOutput));
      passed &= expect(outerOutHub != nullptr && !outerOutHub->outputs.empty(),
                       "outer hub exists after nesting");
      if (outerOutHub != nullptr && !outerOutHub->outputs.empty()) {
        const auto &outerOut = outerOutHub->outputs.front();
        passed &= expect(
            openyourbox::graph::formatCollapsedGroupPinShapes(
                outerOut.repeatShapes, outerOut.shape, {2}, true) == "8ch",
            "collapsed outer output folds nested inner repeats to last-out");
      }
    }
  }

  {
    NodeGraph parentOfNested;
    const auto inId =
        parentOfNested.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto extraId =
        parentOfNested.addNode(NodeType::activation, {80.0f, 0.0f});
    const auto linearId =
        parentOfNested.addNode(NodeType::linear, {240.0f, 0.0f});
    const auto actId =
        parentOfNested.addNode(NodeType::activation, {400.0f, 0.0f});
    const auto outId =
        parentOfNested.addNode(NodeType::audioOutput, {560.0f, 0.0f});
    const auto *inNode = parentOfNested.findNode(inId);
    const auto *extraNode = parentOfNested.findNode(extraId);
    const auto *linearNode = parentOfNested.findNode(linearId);
    const auto *actNode = parentOfNested.findNode(actId);
    const auto *outNode = parentOfNested.findNode(outId);
    passed &= expect(inNode != nullptr && extraNode != nullptr &&
                         linearNode != nullptr && actNode != nullptr &&
                         outNode != nullptr,
                     "parent-of-nested shape graph nodes exist");
    if (inNode != nullptr && extraNode != nullptr && linearNode != nullptr &&
        actNode != nullptr && outNode != nullptr) {
      passed &= expect(parentOfNested.setProperty(linearId, "features", 2),
                       "parent-of-nested linear features match stereo width");
      passed &= expect(
          parentOfNested
              .connect(inNode->outputs.front().id,
                       extraNode->inputs.front().id)
              .accepted &&
              parentOfNested
                  .connect(extraNode->outputs.front().id,
                           linearNode->inputs.front().id)
                  .accepted &&
              parentOfNested
                  .connect(linearNode->outputs.front().id,
                           actNode->inputs.front().id)
                  .accepted &&
              parentOfNested
                  .connect(actNode->outputs.front().id,
                           outNode->inputs.front().id)
                  .accepted,
          "parent-of-nested graph cables");
      const auto inner = parentOfNested.createGroup({linearId, actId});
      passed &= expect(inner.accepted &&
                           parentOfNested.setGroupRepeats(inner.groupId, 3)
                               .accepted,
                       "inner N=3 under a repeats=1 parent");
      passed &= expect(
          parentOfNested.setPropertyRepeatValues(linearId, "features",
                                               {2, 4, 8}),
          "inner per-repeat features 2,4,8 under repeats=1 parent");
      const auto parent =
          parentOfNested.createGroup({extraId, inner.groupId});
      passed &= expect(parent.accepted, "repeats=1 parent wraps nested repeats");
      const auto *parentOutHub = parentOfNested.findNode(
          findBoundary(parentOfNested, parent.groupId, NodeType::groupOutput));
      passed &= expect(
          parentOutHub != nullptr && !parentOutHub->outputs.empty(),
          "repeats=1 parent output hub exists");
      if (parentOutHub != nullptr && !parentOutHub->outputs.empty()) {
        const auto &outPin = parentOutHub->outputs.front();
        passed &= expect(
            openyourbox::graph::formatCollapsedGroupPinShapes(
                outPin.repeatShapes, outPin.shape, {1}, true) == "8ch",
            "repeats=1 parent output is nested last-out, not the inner list");
      }
    }
  }

  {
    NodeGraph decoder;
    const auto inId = decoder.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto pqmfId =
        decoder.addNode(NodeType::pqmfAnalysis, {160.0f, 0.0f});
    const auto actId = decoder.addNode(NodeType::activation, {320.0f, 0.0f});
    const auto convTId =
        decoder.addNode(NodeType::convTranspose, {480.0f, 0.0f});
    const auto *inNode = decoder.findNode(inId);
    const auto *pqmfNode = decoder.findNode(pqmfId);
    const auto *actNode = decoder.findNode(actId);
    const auto *convTNode = decoder.findNode(convTId);
    passed &= expect(inNode != nullptr && pqmfNode != nullptr &&
                         actNode != nullptr && convTNode != nullptr,
                     "decoder repeat-list nodes exist");
    if (inNode != nullptr && pqmfNode != nullptr && actNode != nullptr &&
        convTNode != nullptr) {
      passed &= expect(decoder
                           .connect(inNode->outputs.front().id,
                                    pqmfNode->inputs.front().id)
                           .accepted &&
                           decoder
                               .connect(actNode->outputs.front().id,
                                        convTNode->inputs.front().id)
                               .accepted,
                       "audio to PQMF and Activation to ConvTranspose");
      const auto grouped = decoder.createGroup({actId, convTId});
      passed &= expect(grouped.accepted &&
                           connectDeclaredThrough(decoder, grouped.groupId,
                                                  actId, convTId),
                       "ConvTranspose group wires through");
      const auto inputId =
          findBoundary(decoder, grouped.groupId, NodeType::groupInput);
      const auto *inputHub = decoder.findNode(inputId);
      const auto *pqmfAfterGroup = decoder.findNode(pqmfId);
      passed &= expect(
          inputHub != nullptr && pqmfAfterGroup != nullptr &&
              decoder
                  .connect(pqmfAfterGroup->outputs.front().id,
                           inputHub->inputs.front().id)
                  .accepted,
          "PQMF feeds the ConvTranspose group");
      passed &= expect(decoder.setGroupRepeats(grouped.groupId, 4).accepted,
                       "ConvTranspose group stores N=4");
      passed &= expect(decoder.setPropertyRepeatValues(
                           convTId, "channels", {512, 256, 128, 64}),
                       "ConvTranspose channels 512,256,128,64");
      const auto status = decoder.groupRepeatStatus(grouped.groupId);
      passed &= expect(status.active && status.effectiveRepeats == 4,
                       "per-repeat channel list keeps serial repeats active");
      const auto *convT = decoder.findNode(convTId);
      passed &= expect(
          convT != nullptr && !convT->outputs.empty() &&
              convT->outputs.front().repeatShapes.size() == 4 &&
              convT->outputs.front().repeatShapes[0].channels == 512 &&
              convT->outputs.front().repeatShapes[1].channels == 256 &&
              convT->outputs.front().repeatShapes[2].channels == 128 &&
              convT->outputs.front().repeatShapes[3].channels == 64,
          "ConvTranspose output lists 512,256,128,64");
    }
  }

  {
    NodeGraph reparentGraph;
    reparentGraph.ensureFixedHostIo();
    const auto src = reparentGraph.addNode(NodeType::linear, {80.0f, 40.0f});
    const auto mid = reparentGraph.addNode(NodeType::activation, {200.0f, 40.0f});
    const auto dst = reparentGraph.addNode(NodeType::convolution, {320.0f, 40.0f});
    const auto *srcNode = reparentGraph.findNode(src);
    const auto *midNode = reparentGraph.findNode(mid);
    const auto *dstNode = reparentGraph.findNode(dst);
    passed &= expect(srcNode != nullptr && midNode != nullptr && dstNode != nullptr,
                     "reparent fixture nodes exist");
    if (srcNode != nullptr && midNode != nullptr && dstNode != nullptr) {
      passed &= expect(reparentGraph
                           .connect(srcNode->outputs.front().id,
                                    midNode->inputs.front().id)
                           .accepted &&
                           reparentGraph
                               .connect(midNode->outputs.front().id,
                                        dstNode->inputs.front().id)
                               .accepted,
                       "reparent fixture is wired through the middle box");
      const auto *midBeforeNoop = reparentGraph.findNode(mid);
      const auto noopLinks = reparentGraph.getLinks().size();
      const auto noopPos =
          midBeforeNoop != nullptr ? midBeforeNoop->position
                                   : juce::Point<float>{};
      const auto sameParent =
          reparentGraph.reparentBoxLikeInsert(mid, std::nullopt);
      passed &= expect(sameParent.accepted,
                       "dropping a box onto its current parent is accepted");
      const auto *midAfterNoop = reparentGraph.findNode(mid);
      passed &= expect(midAfterNoop != nullptr &&
                           !midAfterNoop->parentGroupId.has_value() &&
                           std::abs(midAfterNoop->position.x - noopPos.x) <
                               0.5f &&
                           std::abs(midAfterNoop->position.y - noopPos.y) <
                               0.5f,
                       "same-parent drop keeps membership and position");
      passed &= expect(reparentGraph.getLinks().size() == noopLinks,
                       "same-parent drop does not disconnect cables");
      const auto beforeDisconnect = reparentGraph.toValueTree();
      const auto disconnected = reparentGraph.disconnectAllLinksForBox(mid);
      passed &= expect(disconnected.accepted,
                       "disconnect-all accepts an existing element");
      auto incident = 0;
      for (const auto &link : reparentGraph.getLinks()) {
        if (link.sourcePinId == midNode->outputs.front().id ||
            link.destinationPinId == midNode->inputs.front().id)
          ++incident;
      }
      passed &= expect(incident == 0,
                       "disconnect-all removes every cable on the box");
      passed &= expect(reparentGraph.findNode(mid) != nullptr,
                       "disconnect-all does not delete the box");

      const auto partner = reparentGraph.addNode(NodeType::linear, {80.0f, 120.0f});
      const auto partnerB =
          reparentGraph.addNode(NodeType::convolution, {200.0f, 120.0f});
      const auto grouped = reparentGraph.createGroup({partner, partnerB});
      passed &= expect(grouped.accepted, "reparent destination group is created");
      if (grouped.accepted) {
        const auto intoGroup =
            reparentGraph.reparentBoxLikeInsert(mid, grouped.groupId);
        passed &= expect(intoGroup.accepted, "reparent into a group is accepted");
        const auto *moved = reparentGraph.findNode(mid);
        passed &= expect(moved != nullptr && moved->parentGroupId.has_value() &&
                             *moved->parentGroupId == grouped.groupId,
                         "reparent attaches the box as a group member");
        passed &=
            expect(std::abs(moved->position.x - defaultNewBoxPosition.x) < 0.5f &&
                       std::abs(moved->position.y - defaultNewBoxPosition.y) <
                           0.5f,
                   "reparent uses new-item placement in the destination");
        incident = 0;
        for (const auto &link : reparentGraph.getLinks()) {
          if (link.sourcePinId == midNode->outputs.front().id ||
              link.destinationPinId == midNode->inputs.front().id)
            ++incident;
        }
        passed &= expect(incident == 0,
                         "reparent into a group leaves the box disconnected");
        const auto stayInGroup =
            reparentGraph.reparentBoxLikeInsert(mid, grouped.groupId);
        passed &= expect(stayInGroup.accepted,
                         "dropping onto the current group is accepted");
        const auto *stayed = reparentGraph.findNode(mid);
        passed &= expect(
            stayed != nullptr && stayed->parentGroupId.has_value() &&
                *stayed->parentGroupId == grouped.groupId &&
                std::abs(stayed->position.x - defaultNewBoxPosition.x) < 0.5f &&
                std::abs(stayed->position.y - defaultNewBoxPosition.y) < 0.5f,
            "same-group drop does not move or reparent the box");

        const auto toRoot =
            reparentGraph.reparentBoxLikeInsert(mid, std::nullopt);
        passed &= expect(toRoot.accepted, "reparent to the project root is accepted");
        const auto *atRoot = reparentGraph.findNode(mid);
        passed &= expect(atRoot != nullptr && !atRoot->parentGroupId.has_value(),
                         "reparent to root clears membership");
      }

      const auto innerA =
          reparentGraph.addNode(NodeType::activation, {40.0f, 200.0f});
      const auto innerB =
          reparentGraph.addNode(NodeType::convolution, {160.0f, 200.0f});
      const auto inner = reparentGraph.createGroup({innerA, innerB});
      const auto outerLeaf =
          reparentGraph.addNode(NodeType::linear, {280.0f, 200.0f});
      passed &= expect(inner.accepted, "cycle-reject inner group is created");
      const auto outer =
          reparentGraph.createGroup({inner.groupId, outerLeaf});
      passed &= expect(outer.accepted, "cycle-reject outer group is created");
      if (inner.accepted && outer.accepted) {
        const auto linkCountBefore = reparentGraph.getLinks().size();
        const auto *innerBefore = reparentGraph.findGroup(inner.groupId);
        const auto parentBefore =
            innerBefore != nullptr ? innerBefore->parentGroupId : std::nullopt;
        const auto cycle = reparentGraph.reparentBoxLikeInsert(
            outer.groupId, inner.groupId);
        passed &= expect(!cycle.accepted,
                         "reparenting a group into its descendant is rejected");
        const auto *innerAfter = reparentGraph.findGroup(inner.groupId);
        passed &= expect(innerAfter != nullptr &&
                             innerAfter->parentGroupId == parentBefore,
                         "cycle reject leaves hierarchy unchanged");
        passed &= expect(reparentGraph.getLinks().size() == linkCountBefore,
                         "cycle reject leaves cables unchanged");
        const auto self = reparentGraph.reparentBoxLikeInsert(outer.groupId,
                                                              outer.groupId);
        passed &= expect(!self.accepted, "dropping a group onto itself is rejected");
      }

      passed &= expect(reparentGraph.restoreFromValueTree(beforeDisconnect),
                       "graph snapshot before disconnect restores");
      const auto *restoredMid = reparentGraph.findNode(mid);
      passed &= expect(restoredMid != nullptr && !reparentGraph.getLinks().empty(),
                       "restoring the snapshot brings cables back");
    }
  }

  {
    NodeGraph offsetLayout;
    const auto upper =
        offsetLayout.addNode(NodeType::convolution, {120.0f, 40.0f});
    const auto lower =
        offsetLayout.addNode(NodeType::activation, {280.0f, 180.0f});
    const auto grouped = offsetLayout.createGroup({upper, lower});
    passed &= expect(grouped.accepted, "offset boxes must group");
    const auto *left = offsetLayout.findNode(upper);
    const auto *right = offsetLayout.findNode(lower);
    const auto inputId =
        findBoundary(offsetLayout, grouped.groupId, NodeType::groupInput);
    const auto outputId =
        findBoundary(offsetLayout, grouped.groupId, NodeType::groupOutput);
    const auto *inputHub = offsetLayout.findNode(inputId);
    const auto *outputHub = offsetLayout.findNode(outputId);
    passed &= expect(left != nullptr && right != nullptr && inputHub != nullptr &&
                         outputHub != nullptr,
                     "offset group interior boxes exist");
    if (left != nullptr && right != nullptr && inputHub != nullptr &&
        outputHub != nullptr) {
      passed &= expect(std::abs((right->position.x - left->position.x) - 160.0f) <
                               0.5f &&
                           std::abs((right->position.y - left->position.y) -
                                    140.0f) < 0.5f,
                       "offset grouping preserves original relative layout");
      const auto contentMinX = std::min(left->position.x, right->position.x);
      const auto contentMaxX =
          std::max(left->position.x + std::max(8.0f, left->size.x),
                    right->position.x + std::max(8.0f, right->size.x));
      const auto gap = openyourbox::graph::groupBoundaryContentGap;
      passed &= expect(inputHub->position.x +
                               std::max(8.0f, inputHub->size.x) + gap <=
                           contentMinX + 0.5f &&
                           outputHub->position.x + 0.5f >= contentMaxX + gap,
                       "offset grouping still bookends content with I/O gaps");
    }
  }

  {
    NodeGraph hops;
    hops.ensureFixedHostIo();
    std::int32_t inId = 0;
    std::int32_t outId = 0;
    for (const auto &node : hops.getNodes()) {
      if (node.type == NodeType::audioInput)
        inId = node.id;
      if (node.type == NodeType::audioOutput)
        outId = node.id;
    }
    passed &= expect(inId != 0 && outId != 0 &&
                         hops.setProperty(inId, "channels", 0) &&
                         hops.setProperty(outId, "channels", 0),
                     "hop-matching RAVE uses mono host I/O");
    const auto analysis =
        hops.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto downAct =
        hops.addNode(NodeType::activation, {260.0f, 0.0f});
    const auto down =
        hops.addNode(NodeType::convolution, {400.0f, 0.0f});
    const auto upAct =
        hops.addNode(NodeType::activation, {540.0f, 0.0f});
    const auto up =
        hops.addNode(NodeType::convTranspose, {680.0f, 0.0f});
    const auto synth =
        hops.addNode(NodeType::pqmfSynthesis, {820.0f, 0.0f});
    passed &= expect(hops.setProperty(analysis, "n_band", 8) &&
                         hops.setProperty(synth, "n_band", 8) &&
                         hops.setProperty(down, "stride", 4) &&
                         hops.setProperty(down, "channels", 8) &&
                         hops.setProperty(up, "stride", 4) &&
                         hops.setProperty(up, "channels", 8),
                     "PQMF 8 with matching stride-4 encode/decode");
    const auto *inNode = hops.findNode(inId);
    const auto *analysisNode = hops.findNode(analysis);
    const auto *downActNode = hops.findNode(downAct);
    const auto *downNode = hops.findNode(down);
    const auto *upActNode = hops.findNode(upAct);
    const auto *upNode = hops.findNode(up);
    const auto *synthNode = hops.findNode(synth);
    const auto *outNode = hops.findNode(outId);
    passed &= expect(inNode != nullptr && analysisNode != nullptr &&
                         downActNode != nullptr && downNode != nullptr &&
                         upActNode != nullptr && upNode != nullptr &&
                         synthNode != nullptr && outNode != nullptr,
                     "hop-matching RAVE nodes exist");
    if (inNode != nullptr && analysisNode != nullptr && downActNode != nullptr &&
        downNode != nullptr && upActNode != nullptr && upNode != nullptr &&
        synthNode != nullptr && outNode != nullptr) {
      passed &= expect(
          hops.connect(downActNode->outputs.front().id,
                       downNode->inputs.front().id)
              .accepted &&
              hops.connect(upActNode->outputs.front().id,
                           upNode->inputs.front().id)
                  .accepted,
          "encoder and decoder bodies wire");
      const auto encoder = hops.createGroup({downAct, down});
      const auto decoder = hops.createGroup({upAct, up});
      passed &= expect(encoder.accepted && decoder.accepted &&
                           connectDeclaredThrough(hops, encoder.groupId,
                                                  downAct, down) &&
                           connectDeclaredThrough(hops, decoder.groupId, upAct,
                                                  up) &&
                           hops.setGroupRepeats(encoder.groupId, 2).accepted &&
                           hops.setGroupRepeats(decoder.groupId, 2).accepted,
                       "stride-4 encoder/decoder groups N=2");
      const auto encoderIn =
          findBoundary(hops, encoder.groupId, NodeType::groupInput);
      const auto encoderOut =
          findBoundary(hops, encoder.groupId, NodeType::groupOutput);
      const auto decoderIn =
          findBoundary(hops, decoder.groupId, NodeType::groupInput);
      const auto decoderOut =
          findBoundary(hops, decoder.groupId, NodeType::groupOutput);
      const auto *encoderInHub = hops.findNode(encoderIn);
      const auto *encoderOutHub = hops.findNode(encoderOut);
      const auto *decoderInHub = hops.findNode(decoderIn);
      const auto *decoderOutHub = hops.findNode(decoderOut);
      analysisNode = hops.findNode(analysis);
      synthNode = hops.findNode(synth);
      inNode = hops.findNode(inId);
      outNode = hops.findNode(outId);
      passed &= expect(encoderInHub != nullptr && encoderOutHub != nullptr &&
                           decoderInHub != nullptr && decoderOutHub != nullptr,
                       "RAVE group hubs exist");
      if (encoderInHub != nullptr && encoderOutHub != nullptr &&
          decoderInHub != nullptr && decoderOutHub != nullptr &&
          analysisNode != nullptr && synthNode != nullptr && inNode != nullptr &&
          outNode != nullptr) {
        const auto wiredIn = hops.connect(inNode->outputs.front().id,
                                          analysisNode->inputs.front().id);
        const auto wiredAnalysis = hops.connect(
            analysisNode->outputs.front().id, encoderInHub->inputs.front().id);
        const auto wiredLatent = hops.connect(
            encoderOutHub->outputs.front().id, decoderInHub->inputs.front().id);
        const auto wiredSynth = hops.connect(
            decoderOutHub->outputs.front().id, synthNode->inputs.front().id);
        const auto wiredOut = hops.connect(synthNode->outputs.front().id,
                                           outNode->inputs.front().id);
        passed &= expect(wiredIn.accepted, wiredIn.message.c_str());
        passed &= expect(wiredAnalysis.accepted, wiredAnalysis.message.c_str());
        passed &= expect(wiredLatent.accepted, wiredLatent.message.c_str());
        passed &= expect(wiredSynth.accepted, wiredSynth.message.c_str());
        passed &= expect(wiredOut.accepted, wiredOut.message.c_str());
        passed &= expect(wiredIn.accepted && wiredAnalysis.accepted &&
                             wiredLatent.accepted && wiredSynth.accepted &&
                             wiredOut.accepted,
                         "matching hop-product RAVE must connect between "
                         "audio I/O");
      }
    }
  }

  {
    NodeGraph nestedExit;
    nestedExit.ensureFixedHostIo();
    std::int32_t inId = 0;
    std::int32_t outId = 0;
    for (const auto &node : nestedExit.getNodes()) {
      if (node.type == NodeType::audioInput)
        inId = node.id;
      if (node.type == NodeType::audioOutput)
        outId = node.id;
    }
    passed &= expect(inId != 0 && outId != 0 &&
                         nestedExit.setProperty(inId, "channels", 0) &&
                         nestedExit.setProperty(outId, "channels", 0),
                     "nested-exit RAVE uses mono host I/O");
    const auto analysis =
        nestedExit.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto downAct =
        nestedExit.addNode(NodeType::activation, {260.0f, 0.0f});
    const auto down =
        nestedExit.addNode(NodeType::convolution, {400.0f, 0.0f});
    const auto upAct =
        nestedExit.addNode(NodeType::activation, {540.0f, 0.0f});
    const auto up =
        nestedExit.addNode(NodeType::convTranspose, {680.0f, 0.0f});
    const auto residualAct =
        nestedExit.addNode(NodeType::activation, {820.0f, 0.0f});
    const auto residualAct2 =
        nestedExit.addNode(NodeType::activation, {900.0f, 0.0f});
    const auto synth =
        nestedExit.addNode(NodeType::pqmfSynthesis, {1040.0f, 0.0f});
    passed &= expect(nestedExit.setProperty(analysis, "n_band", 8) &&
                         nestedExit.setProperty(synth, "n_band", 8) &&
                         nestedExit.setProperty(down, "stride", 4) &&
                         nestedExit.setProperty(down, "channels", 8) &&
                         nestedExit.setProperty(up, "stride", 4) &&
                         nestedExit.setProperty(up, "channels", 8),
                     "nested-exit PQMF 8 with matching stride-4 encode/decode");
    const auto *downActNode = nestedExit.findNode(downAct);
    const auto *downNode = nestedExit.findNode(down);
    const auto *upActNode = nestedExit.findNode(upAct);
    const auto *upNode = nestedExit.findNode(up);
    passed &= expect(downActNode != nullptr && downNode != nullptr &&
                         upActNode != nullptr && upNode != nullptr,
                     "nested-exit encoder/decoder bodies exist");
    if (downActNode != nullptr && downNode != nullptr && upActNode != nullptr &&
        upNode != nullptr) {
      passed &= expect(
          nestedExit
              .connect(downActNode->outputs.front().id,
                       downNode->inputs.front().id)
              .accepted &&
              nestedExit
                  .connect(upActNode->outputs.front().id,
                           upNode->inputs.front().id)
                  .accepted &&
              nestedExit
                  .connect(nestedExit.findNode(residualAct)->outputs.front().id,
                           nestedExit.findNode(residualAct2)->inputs.front().id)
                  .accepted,
          "nested-exit encoder and decoder bodies wire");
      const auto residual = nestedExit.createGroup({residualAct, residualAct2});
      const auto encoder = nestedExit.createGroup({downAct, down});
      const auto decoder =
          nestedExit.createGroup({upAct, up, residual.groupId});
      passed &= expect(
          residual.accepted && encoder.accepted && decoder.accepted &&
              connectDeclaredThrough(nestedExit, residual.groupId, residualAct,
                                     residualAct2) &&
              connectDeclaredThrough(nestedExit, encoder.groupId, downAct,
                                     down) &&
              nestedExit.setGroupRepeats(residual.groupId, 2).accepted &&
              nestedExit.setGroupRepeats(encoder.groupId, 2).accepted,
          "nested residual group sits on the upsample exit");
      const auto decoderIn =
          findBoundary(nestedExit, decoder.groupId, NodeType::groupInput);
      const auto decoderOut =
          findBoundary(nestedExit, decoder.groupId, NodeType::groupOutput);
      const auto residualIn =
          findBoundary(nestedExit, residual.groupId, NodeType::groupInput);
      const auto residualOut =
          findBoundary(nestedExit, residual.groupId, NodeType::groupOutput);
      const auto encoderIn =
          findBoundary(nestedExit, encoder.groupId, NodeType::groupInput);
      const auto encoderOut =
          findBoundary(nestedExit, encoder.groupId, NodeType::groupOutput);
      const auto *decoderInHub = nestedExit.findNode(decoderIn);
      const auto *decoderOutHub = nestedExit.findNode(decoderOut);
      const auto *residualInHub = nestedExit.findNode(residualIn);
      const auto *residualOutHub = nestedExit.findNode(residualOut);
      const auto *encoderInHub = nestedExit.findNode(encoderIn);
      const auto *encoderOutHub = nestedExit.findNode(encoderOut);
      upNode = nestedExit.findNode(up);
      const auto *upActHub = nestedExit.findNode(upAct);
      const auto *analysisNode = nestedExit.findNode(analysis);
      const auto *synthNode = nestedExit.findNode(synth);
      const auto *inNode = nestedExit.findNode(inId);
      const auto *outNode = nestedExit.findNode(outId);
      passed &= expect(
          decoderInHub != nullptr && decoderOutHub != nullptr &&
              residualInHub != nullptr && residualOutHub != nullptr &&
              encoderInHub != nullptr && encoderOutHub != nullptr &&
              upNode != nullptr && upActHub != nullptr &&
              analysisNode != nullptr && synthNode != nullptr &&
              inNode != nullptr && outNode != nullptr,
          "nested-exit hubs exist");
      if (decoderInHub != nullptr && decoderOutHub != nullptr &&
          residualInHub != nullptr && residualOutHub != nullptr &&
          encoderInHub != nullptr && encoderOutHub != nullptr &&
          upNode != nullptr && upActHub != nullptr && analysisNode != nullptr &&
          synthNode != nullptr && inNode != nullptr && outNode != nullptr) {
        passed &= expect(
            nestedExit
                .connect(decoderInHub->outputs.front().id,
                         upActHub->inputs.front().id)
                .accepted &&
                nestedExit
                    .connect(upNode->outputs.front().id,
                             residualInHub->inputs.front().id)
                    .accepted &&
                nestedExit
                    .connect(residualOutHub->outputs.front().id,
                             decoderOutHub->inputs.front().id)
                    .accepted &&
                nestedExit.setGroupRepeats(decoder.groupId, 2).accepted,
            "ConvTranspose feeds nested residual then decoder output");
        const auto wiredIn = nestedExit.connect(
            inNode->outputs.front().id, analysisNode->inputs.front().id);
        const auto wiredAnalysis = nestedExit.connect(
            analysisNode->outputs.front().id, encoderInHub->inputs.front().id);
        const auto wiredLatent = nestedExit.connect(
            encoderOutHub->outputs.front().id, decoderInHub->inputs.front().id);
        const auto wiredSynth = nestedExit.connect(
            decoderOutHub->outputs.front().id, synthNode->inputs.front().id);
        const auto wiredOut = nestedExit.connect(synthNode->outputs.front().id,
                                                 outNode->inputs.front().id);
        passed &= expect(wiredIn.accepted, wiredIn.message.c_str());
        passed &= expect(wiredAnalysis.accepted, wiredAnalysis.message.c_str());
        passed &= expect(wiredLatent.accepted, wiredLatent.message.c_str());
        passed &= expect(wiredSynth.accepted, wiredSynth.message.c_str());
        passed &= expect(wiredOut.accepted, wiredOut.message.c_str());
        decoderOutHub = nestedExit.findNode(decoderOut);
        passed &= expect(
            decoderOutHub != nullptr && !decoderOutHub->outputs.empty() &&
                decoderOutHub->outputs.front().shape.temporalRate == 8,
            "decoder exit is last upsample hop, not bottleneck hop");
      }
    }
  }

  passed &= testUserLibraryRaveConnectsToHostIo("RAVE");
  passed &= testUserLibraryRaveConnectsToHostIo("Small-RAVE");
  passed &= testUserLibraryRaveConnectsToHostIo("Tiny-RAVE");

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
