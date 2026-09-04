#include "graph/FactoryPalette.h"
#include "graph/GraphTypes.h"
#include "graph/NodeGraph.h"

#include <BinaryData.h>
#include <JuceHeader.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using openyourbox::graph::DataLoaderBindingKind;
using openyourbox::graph::GraphLinkKind;
using openyourbox::graph::GraphNode;
using openyourbox::graph::NodeGraph;
using openyourbox::graph::NodeType;
using openyourbox::graph::TrainingMaterialBinding;
using openyourbox::graph::forEachPaletteItem;

/**
 * @brief Reports a failed invariant.
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
 * @brief Builds an audio-list binding with @p count dummy paths.
 * @param count Example cardinality.
 */
TrainingMaterialBinding audioBinding(int count) {
  TrainingMaterialBinding binding;
  binding.kind = DataLoaderBindingKind::audioList;
  for (int index = 0; index < count; ++index) {
    binding.entries.push_back(
        {"clip-" + std::to_string(index) + ".wav", "id-" + std::to_string(index)});
  }
  return binding;
}

/**
 * @brief First signal input pin of a node, or 0.
 */
std::int32_t firstIn(const NodeGraph &graph, std::int32_t nodeId) {
  const auto *node = graph.findNode(nodeId);
  return node == nullptr || node->inputs.empty() ? 0 : node->inputs.front().id;
}

/**
 * @brief First output pin of a node, or 0.
 */
std::int32_t firstOut(const NodeGraph &graph, std::int32_t nodeId) {
  const auto *node = graph.findNode(nodeId);
  return node == nullptr || node->outputs.empty() ? 0 : node->outputs.front().id;
}

/**
 * @brief Output pin at @p index, or 0.
 */
std::int32_t outAt(const NodeGraph &graph, std::int32_t nodeId, int index) {
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr || index < 0 ||
      index >= static_cast<int>(node->outputs.size()))
    return 0;
  return node->outputs[static_cast<std::size_t>(index)].id;
}

/**
 * @brief Input pin at @p index, or 0.
 */
std::int32_t inAt(const NodeGraph &graph, std::int32_t nodeId, int index) {
  const auto *node = graph.findNode(nodeId);
  if (node == nullptr || index < 0 ||
      index >= static_cast<int>(node->inputs.size()))
    return 0;
  return node->inputs[static_cast<std::size_t>(index)].id;
}

/**
 * @brief True when @p ids contains @p value.
 */
bool containsId(const std::vector<std::int32_t> &ids, std::int32_t value) {
  return std::find(ids.begin(), ids.end(), value) != ids.end();
}

/**
 * @brief Restores a shipped GraphDocument XML blob.
 * @param xml UTF-8 document bytes.
 * @param size Byte count.
 */
bool restoreExampleXml(NodeGraph &graph, const char *xml, int size) {
  auto parsed = juce::parseXML(juce::String::fromUTF8(xml, size));
  if (parsed == nullptr)
    return false;
  return graph.restoreFromValueTree(juce::ValueTree::fromXml(*parsed));
}
} // namespace

/**
 * @brief Equal-count, active-loader, connect, path∩arm, group-of-one, palette.
 * @return Zero when every invariant passes.
 */
int main() {
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  {
    bool hasTcn = false;
    bool hasLinear = false;
    bool hasDataLoader = false;
    bool hasLoss = false;
    forEachPaletteItem([&](const auto &item) {
      hasTcn = hasTcn || item.type == NodeType::tcn;
      hasLinear = hasLinear || item.type == NodeType::linear;
      hasDataLoader = hasDataLoader || item.type == NodeType::dataLoader;
      hasLoss = hasLoss || item.type == NodeType::loss;
    });
    passed &= expect(!hasTcn && !hasLinear,
                     "palette insertion list excludes tcn and linear");
    passed &= expect(hasDataLoader && hasLoss,
                     "palette lists Data Loader and Loss");
  }

  {
    NodeGraph naming;
    const auto input = naming.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto output = naming.addNode(NodeType::audioOutput, {400.0f, 0.0f});
    const auto firstConv =
        naming.addNode(NodeType::convolution, {100.0f, 0.0f});
    const auto secondConv =
        naming.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto firstAct =
        naming.addNode(NodeType::activation, {100.0f, 100.0f});
    const auto firstLoader =
        naming.addNode(NodeType::dataLoader, {0.0f, 200.0f});
    const auto secondLoader =
        naming.addNode(NodeType::dataLoader, {100.0f, 200.0f});
    const auto firstLoss = naming.addNode(NodeType::loss, {0.0f, 300.0f});
    passed &= expect(naming.findNode(input)->label == "Audio Input" &&
                         naming.findNode(output)->label == "Audio Output",
                     "Audio I/O keeps fixed unnumbered labels");
    passed &= expect(naming.findNode(firstConv)->label == "Conv1D 1" &&
                         naming.findNode(secondConv)->label == "Conv1D 2",
                     "new Conv1Ds receive the smallest free numbers");
    passed &= expect(naming.findNode(firstAct)->label == "Activation 1",
                     "other insertable boxes are numbered the same way");
    passed &= expect(naming.findNode(firstLoader)->label == "Data Loader 1" &&
                         naming.findNode(secondLoader)->label ==
                             "Data Loader 2" &&
                         naming.findNode(firstLoss)->label == "Loss 1",
                     "Data Loader and Loss use the same numbering path");
    naming.removeNode(firstConv);
    const auto reused =
        naming.addNode(NodeType::convolution, {150.0f, 0.0f});
    passed &= expect(naming.findNode(reused)->label == "Conv1D 1",
                     "removing Conv1D 1 frees that number for the next insert");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto convA = graph.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto convB = graph.addNode(NodeType::convolution, {400.0f, 0.0f});
    const auto convC = graph.addNode(NodeType::convolution, {600.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {800.0f, 0.0f});
    const auto loader = graph.addNode(NodeType::dataLoader, {0.0f, 180.0f});
    const auto loss = graph.addNode(NodeType::loss, {600.0f, 180.0f});
    passed &= expect(graph.connect(firstOut(graph, input), firstIn(graph, convA))
                         .accepted,
                     "live Audio In to A");
    passed &= expect(
        graph.connect(outAt(graph, loader, 0), firstIn(graph, convB)).accepted,
        "Data Loader may attach to an empty pin");
    passed &= expect(graph.connect(firstOut(graph, convA), firstIn(graph, convB))
                         .accepted,
                     "A to B may coexist with an existing Data Loader feed");
    passed &= expect(graph.connect(firstOut(graph, convB), firstIn(graph, convC))
                         .accepted,
                     "B to C");
    passed &= expect(
        graph.connect(firstOut(graph, convC), firstIn(graph, output)).accepted,
        "C to Audio Out");
    passed &= expect(graph
                         .connect(outAt(graph, loader, 0), firstIn(graph, convA))
                         .accepted,
                     "Data Loader may coexist with live Audio In on A");
    const auto refuseInternal =
        graph.connect(outAt(graph, loader, 0), firstIn(graph, convC));
    passed &= expect(!refuseInternal.accepted,
                     "Data Loader cannot attach to a processing-driven pin");
    passed &= expect(
        graph.connect(firstOut(graph, convC), inAt(graph, loss, 0)).accepted,
        "C fans out to Loss prediction");
    const auto refuseDlPrediction =
        graph.connect(outAt(graph, loader, 1), inAt(graph, loss, 0));
    passed &= expect(!refuseDlPrediction.accepted,
                     "Data Loader cannot connect to Loss prediction");
    const auto refuseLiveTarget =
        graph.connect(firstOut(graph, convC), inAt(graph, loss, 1));
    passed &= expect(!refuseLiveTarget.accepted,
                     "live graph cannot connect to Loss target");
    passed &= expect(
        graph.connect(outAt(graph, loader, 1), inAt(graph, loss, 1)).accepted,
        "Data Loader connects to Loss target");

    passed &= expect(graph.setDataLoaderBinding(loader, 0, audioBinding(2)) &&
                         graph.setDataLoaderBinding(loader, 1, audioBinding(1)),
                     "bindings can be written with unequal counts");
    const auto mismatch = graph.validateTrainStart(loader);
    passed &= expect(mismatch.find("equal") != std::string::npos,
                     "Start refuses unequal connected counts");
    passed &= expect(graph.equalizeDataLoaderOutput(loader, 0, 1),
                     "copy/repeat utility equalizes counts");
    passed &= expect(graph.validateTrainStart(loader).empty(),
                     "equal connected counts pass Start");

    graph.setProperty(loader, "output_count", 3);
    passed &= expect(graph.setDataLoaderBinding(loader, 2, audioBinding(9)),
                     "unconnected output can hold a different count");
    passed &= expect(graph.validateTrainStart(loader).empty(),
                     "unconnected outputs are ignored for equal-count");

    const auto extraLoader =
        graph.addNode(NodeType::dataLoader, {200.0f, 180.0f});
    passed &= expect(graph.validateTrainStart(0).find("Choose") !=
                         std::string::npos,
                     "multiple loaders without a picker refuse Start");
    passed &= expect(graph.validateTrainStart(loader).empty(),
                     "an explicit active loader is accepted");
    graph.removeNode(extraLoader);

    passed &= expect(graph.setArmedForTraining(convB, false), "disarm B");
    const auto onPath = graph.collectDataLoaderPathNodeIds(loader);
    const auto armedOnPath = graph.collectArmedOnPathNodeIds(loader);
    passed &= expect(containsId(onPath, convA) && containsId(onPath, convB) &&
                         containsId(onPath, convC),
                     "path includes A, B, and C");
    passed &= expect(containsId(armedOnPath, convA) &&
                         containsId(armedOnPath, convC) &&
                         !containsId(armedOnPath, convB),
                     "armed∩path is A and C, not disarmed B");
    const auto request = graph.createTrainRequest();
    passed &= expect(request.has_value(), "train_graph request is produced");
    if (request.has_value()) {
      const auto parsed = juce::JSON::parse(request->graphFragment);
      auto *root = parsed.getDynamicObject();
      passed &= expect(root != nullptr && root->getProperty("operation")
                                                  .toString() == "train_graph",
                       "createTrainRequest uses train_graph");
      const auto *armedIds = root->getProperty("armed_element_ids").getArray();
      std::unordered_set<int> armed;
      if (armedIds != nullptr) {
        for (const auto &id : *armedIds)
          armed.insert(static_cast<int>(id));
      }
      passed &= expect(armed.count(convA) == 1 && armed.count(convC) == 1 &&
                           armed.count(convB) == 0,
                       "armed_element_ids is path intersection");
    }
  }

  {
    NodeGraph empty;
    passed &= expect(empty.validateTrainStart(0).find("Data Loader") !=
                         std::string::npos,
                     "Start without a Data Loader is refused");
  }

  {
    NodeGraph grouped;
    const auto conv = grouped.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto created = grouped.createGroup({conv});
    passed &= expect(created.accepted, "a single Conv1D can form a group");
    const auto *group = grouped.findGroup(created.groupId);
    passed &= expect(group != nullptr && group->memberIds.size() >= 3,
                     "group-of-one includes the member plus I/O hubs");
    passed &= expect(grouped.findNode(conv) != nullptr &&
                         grouped.findNode(conv)->parentGroupId == created.groupId,
                     "the single member stores parentGroupId");
    NodeGraph restored;
    passed &= expect(restored.restoreFromValueTree(grouped.toValueTree()),
                     "group-of-one persists");
    passed &= expect(restored.findGroup(created.groupId) != nullptr,
                     "reloaded group is present");
    const auto lifted = restored.ungroup(created.groupId);
    passed &= expect(lifted.accepted && restored.getGroups().empty(),
                     "ungroup removes the container");
    passed &= expect(restored.findNode(conv) != nullptr &&
                         !restored.findNode(conv)->parentGroupId.has_value(),
                     "ungroup preserves the member");
  }

  {
    NodeGraph sharedHub;
    const auto input = sharedHub.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto conv = sharedHub.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto loader =
        sharedHub.addNode(NodeType::dataLoader, {0.0f, 180.0f});
    passed &= expect(
        sharedHub.connect(firstOut(sharedHub, input), firstIn(sharedHub, conv))
            .accepted,
        "Audio In feeds the member before grouping");
    passed &= expect(sharedHub
                         .connect(outAt(sharedHub, loader, 0),
                                  firstIn(sharedHub, conv))
                         .accepted,
                     "Data Loader shares the same member pin");
    const auto created = sharedHub.createGroup({conv});
    passed &= expect(created.accepted, "member with dual feed can group");
    const GraphNode *inputHub = nullptr;
    for (const auto &node : sharedHub.getNodes()) {
      if (node.type == NodeType::groupInput &&
          node.parentGroupId == created.groupId)
        inputHub = &node;
    }
    passed &= expect(inputHub != nullptr && inputHub->inputs.size() == 1,
                     "Audio In + Data Loader share one group input pin");
    int externalToHub = 0;
    int hubToMember = 0;
    for (const auto &link : sharedHub.getLinks()) {
      if (inputHub == nullptr)
        continue;
      if (link.destinationPinId == inputHub->inputs.front().id)
        ++externalToHub;
      if (link.sourcePinId == inputHub->outputs.front().id)
        ++hubToMember;
    }
    passed &= expect(externalToHub == 2 && hubToMember == 1,
                     "both external cables hit one hub; one internal cable");
  }

  {
    NodeGraph mapping;
    passed &= expect(restoreExampleXml(mapping, BinaryData::mappingstylegraph_xml,
                                       BinaryData::mappingstylegraph_xmlSize),
                     "mapping-style example graph restores");
    bool hasLoader = false;
    bool hasLoss = false;
    bool hasTcn = false;
    for (const auto &node : mapping.getNodes()) {
      hasLoader = hasLoader || node.type == NodeType::dataLoader;
      hasLoss = hasLoss || node.type == NodeType::loss;
      hasTcn = hasTcn || node.type == NodeType::tcn;
    }
    passed &= expect(hasLoader && hasLoss && !hasTcn,
                     "mapping-style example has Data Loader and Loss, not TCN");
    NodeGraph reconstruction;
    passed &= expect(
        restoreExampleXml(reconstruction, BinaryData::reconstructionstylegraph_xml,
                          BinaryData::reconstructionstylegraph_xmlSize),
        "reconstruction-style example graph restores");
  }

  return passed ? 0 : 1;
}
