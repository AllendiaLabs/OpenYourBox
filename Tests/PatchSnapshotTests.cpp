#include "PluginProcessor.h"
#include "graph/NodeGraph.h"
#include "state/PatchSnapshot.h"

#include <JuceHeader.h>

#include <iostream>

namespace {
/**
 * @brief Reports a failed snapshot invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Waits for asynchronous graph compilation to publish its runtime.
 * @param processor Processor owning the background graph publisher.
 * @param expectedReceptiveField Receptive field identifying the test graph.
 */
bool waitForGraphRuntime(OpenYourBoxAudioProcessor &processor,
                         std::uint64_t expectedReceptiveField) {
  constexpr int attempts = 200;
  for (int attempt = 0; attempt < attempts; ++attempt) {
    if (processor.getReceptiveFieldSamples() == expectedReceptiveField)
      return true;
    juce::Thread::sleep(10);
  }
  return false;
}

/**
 * @brief Builds a small connected convolution graph for snapshot tests.
 */
openyourbox::graph::NodeGraph makeTestGraph() {
  openyourbox::graph::NodeGraph graph;
  const auto input =
      graph.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto conv =
      graph.addNode(openyourbox::graph::NodeType::convolution, {200.0f, 0.0f});
  const auto output =
      graph.addNode(openyourbox::graph::NodeType::audioOutput, {400.0f, 0.0f});
  const auto *inputNode = graph.findNode(input);
  const auto *convNode = graph.findNode(conv);
  const auto *outputNode = graph.findNode(output);
  if (inputNode != nullptr && convNode != nullptr && outputNode != nullptr) {
    graph.connect(inputNode->outputs.front().id, convNode->inputs.front().id);
    graph.connect(convNode->outputs.front().id, outputNode->inputs.front().id);
  }
  graph.setSeed(conv, 4242);
  return graph;
}
} // namespace

/**
 * @brief Runs PatchSnapshot capture/apply and XML round-trip checks.
 * @return Zero when every invariant passes.
 */
int main() {
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  juce::XmlElement future("OpenYourBoxState");
  future.setAttribute("schemaVersion", 99);
  passed &= expect(!openyourbox::state::PatchSnapshot::fromXml(future).has_value(),
                   "unknown future schema versions must be refused");

  auto graph = makeTestGraph();
  OpenYourBoxAudioProcessor processor;
  processor.setGraphState(graph.toValueTree());
  processor.prepareToPlay(48000.0, 256);
  passed &= expect(waitForGraphRuntime(processor, 3),
                   "test graph must compile before snapshot capture");

  const auto captured = processor.capturePatchSnapshot();
  passed &= expect(captured.isValid(), "captured snapshot must be valid");
  passed &= expect(captured.graphDocument.hasType("GraphDocument"),
                   "captured snapshot must include the graph document");
  const auto xml = captured.toXml();
  passed &= expect(xml != nullptr, "snapshot XML must serialize");
  const auto parsed = openyourbox::state::PatchSnapshot::fromXml(*xml);
  passed &= expect(parsed.has_value() && parsed->isValid(),
                   "snapshot XML must round-trip");

  auto fingerprintA = captured.sonicFingerprint();
  auto moved = captured;
  moved.graphDocument.setProperty("panX", 123.0f, nullptr);
  moved.graphDocument.setProperty("zoom", 2.0f, nullptr);
  moved.graphDocument.setProperty("stickySpine", "1,2", nullptr);
  passed &= expect(fingerprintA == moved.sonicFingerprint(),
                   "sonic fingerprint must ignore pan, zoom, and sticky spine");

  juce::ValueTree groupView{"Group"};
  groupView.setProperty("id", 7, nullptr);
  groupView.setProperty("viewPanX", 10.0f, nullptr);
  groupView.setProperty("viewPanY", 20.0f, nullptr);
  groupView.setProperty("viewZoom", 1.0f, nullptr);
  moved.graphDocument.appendChild(groupView, nullptr);
  const auto groupedFingerprint = moved.sonicFingerprint();
  moved.graphDocument.getChildWithName("Group").setProperty("viewZoom", 2.5f,
                                                            nullptr);
  passed &= expect(groupedFingerprint == moved.sonicFingerprint(),
                   "sonic fingerprint must ignore group cameras");

  openyourbox::state::PatchSnapshot viewportDest;
  viewportDest.graphDocument = juce::ValueTree{"GraphDocument"};
  viewportDest.graphDocument.setProperty("panX", 0.0f, nullptr);
  juce::ValueTree destGroup{"Group"};
  destGroup.setProperty("id", 7, nullptr);
  destGroup.setProperty("viewZoom", 1.0f, nullptr);
  viewportDest.graphDocument.appendChild(destGroup, nullptr);
  juce::ValueTree viewportSource{"GraphDocument"};
  viewportSource.setProperty("panX", 50.0f, nullptr);
  viewportSource.setProperty("stickySpine", "7", nullptr);
  juce::ValueTree sourceGroup{"Group"};
  sourceGroup.setProperty("id", 7, nullptr);
  sourceGroup.setProperty("viewZoom", 1.75f, nullptr);
  viewportSource.appendChild(sourceGroup, nullptr);
  viewportDest.copyViewportFrom(viewportSource);
  passed &=
      expect(static_cast<float>(viewportDest.graphDocument["panX"]) == 50.0f &&
                 viewportDest.graphDocument["stickySpine"].toString() == "7" &&
                 static_cast<float>(viewportDest.graphDocument
                                        .getChildWithName("Group")["viewZoom"]) ==
                     1.75f,
             "copyViewportFrom must keep live root and group cameras");

  OpenYourBoxAudioProcessor restored;
  juce::String error;
  openyourbox::state::ApplyOptions options;
  passed &= expect(restored.applyPatchSnapshot(captured, options, error),
                   "apply must restore a captured snapshot");
  restored.prepareToPlay(48000.0, 256);
  passed &= expect(waitForGraphRuntime(restored, 3),
                   "applied snapshot must compile");
  openyourbox::graph::NodeGraph restoredGraph;
  passed &= expect(restoredGraph.restoreFromValueTree(restored.getGraphState()),
                   "applied graph must restore");
  bool foundSeed = false;
  for (const auto &node : restoredGraph.getNodes()) {
    if (node.type == openyourbox::graph::NodeType::convolution &&
        node.seed == 4242)
      foundSeed = true;
  }
  passed &= expect(foundSeed, "applied snapshot must restore graph fields");

  juce::ValueTree unknown{"GraphDocument"};
  juce::ValueTree badNode{"Node"};
  badNode.setProperty("type", "not_a_real_type", nullptr);
  unknown.appendChild(badNode, nullptr);
  juce::String restoreError;
  passed &= expect(
      !openyourbox::graph::NodeGraph::documentIsRestorable(unknown, restoreError),
      "unknown node types must fail closed");

  return passed ? 0 : 1;
}
