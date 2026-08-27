#include "graph/NodeGraph.h"
#include "library/UserBoxLibrary.h"

#include <JuceHeader.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {
/**
 * @brief Reports a failed library invariant.
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
 * @brief Creates an isolated temporary boxes directory.
 */
juce::File makeTempBoxesRoot() {
  auto root = juce::File::getSpecialLocation(juce::File::tempDirectory)
                  .getChildFile("OpenYourBoxBoxLibraryTests")
                  .getChildFile(juce::Uuid().toDashedString());
  root.deleteRecursively();
  root.createDirectory();
  return root;
}
} // namespace

/**
 * @brief Runs box-library save, folder, insert, and refusal checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  using openyourbox::library::UserBoxKind;
  using openyourbox::library::UserBoxLibrary;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  const auto root = makeTempBoxesRoot();
  UserBoxLibrary library(root);

  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto convA = graph.addNode(NodeType::convolution, {200.0f, 0.0f});
  const auto convB = graph.addNode(NodeType::convolution, {400.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {600.0f, 0.0f});
  const auto *nodeA = graph.findNode(convA);
  const auto *nodeB = graph.findNode(convB);
  const auto *inputNode = graph.findNode(input);
  const auto *outputNode = graph.findNode(output);
  passed &= expect(nodeA != nullptr && nodeB != nullptr && inputNode != nullptr &&
                       outputNode != nullptr,
                   "palette elements must exist");
  if (nodeA == nullptr || nodeB == nullptr || inputNode == nullptr ||
      outputNode == nullptr)
    return 1;
  graph.connect(inputNode->outputs.front().id, nodeA->inputs.front().id);
  graph.connect(nodeA->outputs.front().id, nodeB->inputs.front().id);
  graph.connect(nodeB->outputs.front().id, outputNode->inputs.front().id);
  graph.setProperty(convA, "kernel_size", 7);

  juce::String error;
  passed &= expect(!library.saveBox(graph, input, "IO", {}, false, error).has_value(),
                   "audio I/O cannot be saved");
  error.clear();

  juce::String folderError;
  passed &= expect(!library.removeFolder({}, false, folderError),
                   "library root cannot be deleted");
  folderError.clear();
  passed &= expect(!library.renameFolder({}, "Other", folderError),
                   "library root cannot be renamed");
  folderError.clear();
  passed &= expect(library.createFolder("FX/Delay", folderError),
                   "nested folders can be created");
  const auto saved = library.saveBox(graph, convA, "E1", "FX/Delay", false, error);
  passed &= expect(saved.has_value() && saved->kind == UserBoxKind::element,
                   "single element saves to the catalog");
  passed &= expect(saved.has_value() && saved->folder == "FX/Delay",
                   "saved boxes keep their folder");

  const auto grouped = graph.createGroup({convA, convB});
  passed &= expect(grouped.accepted, "two processing nodes must group");
  std::int32_t boundaryId = 0;
  if (const auto *group = graph.findGroup(grouped.groupId)) {
    for (const auto memberId : group->memberIds) {
      const auto *member = graph.findNode(memberId);
      if (member != nullptr &&
          openyourbox::graph::isGroupBoundaryType(member->type)) {
        boundaryId = memberId;
        break;
      }
    }
  }
  error.clear();
  passed &= expect(
      boundaryId != 0 &&
          !library
               .saveBox(graph, boundaryId, "Boundary", {}, false, error)
               .has_value(),
      "group boundary hubs cannot be saved as standalone boxes");
  graph.setGroupCopies(grouped.groupId, 3);
  error.clear();
  const auto savedGroup =
      library.saveBox(graph, grouped.groupId, "G1", {}, false, error);
  passed &= expect(savedGroup.has_value() && savedGroup->kind == UserBoxKind::group,
                   "groups save to the catalog");

  error.clear();
  passed &= expect(!library.saveBox(graph, convA, "E1", {}, false, error).has_value(),
                   "duplicate names require overwrite");
  error.clear();
  passed &= expect(
      library.saveBox(graph, convA, "E1", "FX/Delay", true, error).has_value(),
      "overwrite replaces an existing name");

  UserBoxLibrary reloaded(root);
  passed &= expect(reloaded.getEntries().size() == 2,
                   "index survives reload");
  passed &= expect(reloaded.findEntryByName("E1") != nullptr &&
                       reloaded.findEntryByName("G1") != nullptr,
                   "element and group names reload");
  passed &= expect(std::find(reloaded.getFolders().begin(),
                             reloaded.getFolders().end(),
                             juce::String("FX/Delay")) !=
                       reloaded.getFolders().end(),
                   "empty-capable folder paths reload");

  NodeGraph placed;
  error.clear();
  const auto *e1 = reloaded.findEntryByName("E1");
  passed &= expect(e1 != nullptr, "E1 must be findable after reload");
  if (e1 != nullptr) {
    const auto placedId =
        reloaded.insertBox(placed, e1->id, {80.0f, 40.0f}, error);
    passed &= expect(placedId.has_value(), "element insert must succeed");
    const auto *placedNode =
        placedId.has_value() ? placed.findNode(*placedId) : nullptr;
    passed &= expect(placedNode != nullptr &&
                         placedNode->type == NodeType::convolution,
                     "inserted element keeps its type");
    int kernel = 0;
    if (placedNode != nullptr) {
      for (const auto &property : placedNode->properties) {
        if (property.key == "kernel_size")
          kernel = property.value;
      }
    }
    passed &= expect(kernel == 7, "inserted element keeps parameters");
    passed &= expect(placedNode != nullptr &&
                         std::abs(placedNode->position.x - 80.0f) < 0.5f,
                     "inserted element lands at the drop position");
  }

  error.clear();
  const auto *g1 = reloaded.findEntryByName("G1");
  passed &= expect(g1 != nullptr, "G1 must be findable after reload");
  if (g1 != nullptr) {
    const auto placedGroupId =
        reloaded.insertBox(placed, g1->id, {120.0f, 90.0f}, error);
    passed &= expect(placedGroupId.has_value(), "group insert must succeed");
    const auto *placedGroup =
        placedGroupId.has_value() ? placed.findGroup(*placedGroupId) : nullptr;
    passed &= expect(placedGroup != nullptr && placedGroup->collapsed,
                     "inserted groups start collapsed");
    passed &= expect(placedGroup != nullptr && placedGroup->copies == 3,
                     "inserted groups restore copies N");
    passed &= expect(placed.getGroups().size() == 1,
                     "library original is not mutated into extra groups");
    int boundaryCount = 0;
    if (placedGroup != nullptr) {
      for (const auto memberId : placedGroup->memberIds) {
        const auto *member = placed.findNode(memberId);
        if (member != nullptr &&
            openyourbox::graph::isGroupBoundaryType(member->type))
          ++boundaryCount;
      }
    }
    passed &= expect(boundaryCount == 2,
                     "inserted groups restore explicit input/output hubs");

    juce::String snapError;
    auto snapshot = reloaded.loadEntrySnapshot(g1->id, snapError);
    std::int32_t nestedId = 0;
    for (const auto child : snapshot) {
      if (!child.hasType("Node"))
        continue;
      const auto type = child["type"].toString();
      if (type == "group_input" || type == "group_output")
        continue;
      nestedId = static_cast<std::int32_t>(child["id"]);
      break;
    }
    passed &= expect(nestedId != 0, "group snapshot contains a member node");
    error.clear();
    const auto nestedInsert =
        reloaded.insertBox(placed, g1->id, {200.0f, 40.0f}, error, nestedId);
    passed &= expect(nestedInsert.has_value(), "nested subpart insert succeeds");
    passed &= expect(nestedInsert.has_value() &&
                         placed.findNode(*nestedInsert) != nullptr &&
                         placed.findGroup(*nestedInsert) == nullptr,
                     "nested insert places only the selected element");
  }

  error.clear();
  auto corrupt = graph.exportBox(convB, error);
  passed &= expect(corrupt.isValid(), "export of a legal element succeeds");
  for (int index = 0; index < corrupt.getNumChildren(); ++index) {
    auto child = corrupt.getChild(index);
    if (child.hasType("Node"))
      child.setProperty("type", "future_unknown_op", nullptr);
  }
  NodeGraph rejected;
  error.clear();
  const auto beforeCount = rejected.getNodes().size();
  passed &= expect(
      !rejected.importBox(corrupt, {0.0f, 0.0f}, true, error).has_value(),
      "unknown element types refuse insert");
  passed &= expect(rejected.getNodes().size() == beforeCount,
                   "failed insert leaves the graph unchanged");

  passed &= expect(reloaded.rename(reloaded.findEntryByName("E1")->id, "E1b",
                                   error),
                   "catalog rows can be renamed");
  passed &= expect(reloaded.removeEntry(reloaded.findEntryByName("E1b")->id),
                   "catalog rows can be deleted");
  passed &= expect(reloaded.findEntryByName("E1b") == nullptr,
                   "deleted entries are not insertable");

  root.deleteRecursively();
  return passed ? 0 : 1;
}
