#include "PluginProcessor.h"
#include "graph/NodeGraph.h"

#include <JuceHeader.h>

#include <vector>

namespace {
/**
 * @brief Fills a stereo buffer with a deterministic sine wave.
 * @param buffer Destination buffer.
 * @param sampleRate Test sample rate.
 */
void fillSine(juce::AudioBuffer<float> &buffer, double sampleRate) {
  for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
    const auto value =
        static_cast<float>(0.2 * std::sin(juce::MathConstants<double>::twoPi *
                                          440.0 * sample / sampleRate));
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
      buffer.setSample(channel, sample, value);
  }
}

/**
 * @brief Reports a failed processor invariant.
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
 * @brief Waits for asynchronous graph compilation to publish its runtime.
 * @param processor Processor owning the background graph publisher.
 * @param expectedReceptiveField Receptive field identifying the test graph.
 * @return True when the graph runtime becomes observable before timeout.
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
} // namespace

/**
 * @brief Runs processor-level audio, silence, and state-recall checks.
 * @return Zero when every invariant passes.
 */
int main() {
  constexpr double sampleRate = 48000.0;
  constexpr int blockSize = 256;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  juce::MidiBuffer midi;
  bool passed = true;

  openyourbox::graph::NodeGraph graph;
  const auto inputNode =
      graph.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto convolutionNode =
      graph.addNode(openyourbox::graph::NodeType::convolution, {200.0f, 0.0f});
  const auto outputNode =
      graph.addNode(openyourbox::graph::NodeType::audioOutput, {400.0f, 0.0f});
  const auto *inputElement = graph.findNode(inputNode);
  const auto *convolutionElement = graph.findNode(convolutionNode);
  const auto *outputElement = graph.findNode(outputNode);
  passed &= expect(inputElement != nullptr && convolutionElement != nullptr &&
                       outputElement != nullptr,
                   "graph palette elements must receive stable identifiers");
  passed &= expect(!graph.removeNode(inputNode) && !graph.removeNode(outputNode),
                   "fixed stereo audio I/O must refuse deletion");
  if (inputElement != nullptr && convolutionElement != nullptr &&
      outputElement != nullptr) {
    passed &= expect(graph
                         .connect(inputElement->outputs.front().id,
                                  convolutionElement->inputs.front().id)
                         .accepted,
                     "compatible output-to-input link must commit");
    passed &= expect(graph
                         .connect(convolutionElement->outputs.front().id,
                                  outputElement->inputs.front().id)
                         .accepted,
                     "compatible processing-to-output link must commit");
    passed &=
        expect(!graph
                    .connect(convolutionElement->outputs.front().id,
                             convolutionElement->inputs.front().id)
                    .accepted,
               "self-cycle must be rejected before it reaches runtime state");
  }
  graph.setSeed(convolutionNode, 123456);
  const auto extraConvolution =
      graph.addNode(openyourbox::graph::NodeType::convolution, {200.0f, 80.0f});
  const auto freezeChains =
      graph.partitionFreezeChains({convolutionNode, extraConvolution});
  passed &=
      expect(freezeChains.size() == 2,
             "disconnected freeze selections must form independent chains");
  passed &= expect(graph.removeNode(extraConvolution),
                   "temporary extra convolution must be removable");
  const auto freezeRequest = graph.createFreezeRequest({convolutionNode});
  passed &= expect(freezeRequest.has_value(),
                   "connected live selection must serialize for freezing");
  if (freezeRequest.has_value()) {
    const auto payload =
        juce::JSON::parse(juce::String(freezeRequest->graphFragment));
    const auto fragment = payload.getProperty("graph_fragment", {});
    const auto elements = fragment.getProperty("elements", {});
    passed &= expect(payload.getProperty("operation", {}).toString() ==
                             "freeze_selection" &&
                         elements.isArray() && elements.getArray()->size() == 1,
                     "freeze JSON must contain only the selected graph");

    openyourbox::graph::FreezeSelectionResult freezeResult;
    freezeResult.requestId = freezeRequest->requestId;
    freezeResult.succeeded = true;
    freezeResult.artifactPath = "/tmp/openyourbox-test-blackbox.pt";
    freezeResult.inputChannels = 2;
    freezeResult.outputChannels = 2;
    const auto frozen =
        graph.freezeSelection({convolutionNode}, freezeResult);
    passed &= expect(frozen.has_value() && graph.getNodes().size() == 3,
                     "successful freeze must keep elements in place");
    passed &= expect(
        frozen.has_value() && graph.findNode(*frozen) != nullptr &&
            graph.findNode(*frozen)->state ==
                openyourbox::graph::NodeState::frozenGold,
        "frozen elements must use the Gold colour state");
    passed &= expect(
        frozen.has_value() && graph.findNode(*frozen) != nullptr &&
            graph.findNode(*frozen)->metrics.has_value(),
        "compile and inference times must be written on the frozen chain sink");
    passed &=
        expect(frozen.has_value() && graph.unfreeze(*frozen) &&
                   graph.getNodes().size() == 3 && graph.getLinks().size() == 2,
               "unfreeze must restore the prior live elements in place");
  }

  OpenYourBoxAudioProcessor original;
  original.setGraphState(graph.toValueTree());
  original.prepareToPlay(sampleRate, blockSize);
  passed &= expect(waitForGraphRuntime(original, 3),
                   "live graph must compile before audio assertions");

  juce::AudioBuffer<float> input(2, blockSize);
  fillSine(input, sampleRate);
  auto processed = input;
  original.processBlock(processed, midi);
  passed &= expect(processed.getMagnitude(0, 0, blockSize) > 0.0f,
                   "default model must produce non-silent output");

  float difference = 0.0f;
  for (int sample = 0; sample < blockSize; ++sample)
    difference = std::max(difference, std::abs(processed.getSample(0, sample) -
                                               input.getSample(0, sample)));
  passed &= expect(difference > 1.0e-5f,
                   "default wet model must audibly differ from the dry input");

  juce::AudioBuffer<float> silence(2, blockSize);
  silence.clear();
  original.processBlock(silence, midi);
  passed &= expect(silence.getMagnitude(0, 0, blockSize) == 0.0f,
                   "digital silence must remain digital silence");

  juce::MemoryBlock savedState;
  original.getStateInformation(savedState);
  original.releaseResources();
  original.prepareToPlay(sampleRate, blockSize);

  auto expectedRecall = input;
  original.processBlock(expectedRecall, midi);

  OpenYourBoxAudioProcessor restored;
  restored.setStateInformation(savedState.getData(),
                               static_cast<int>(savedState.getSize()));
  restored.prepareToPlay(sampleRate, blockSize);
  passed &= expect(waitForGraphRuntime(restored, 3),
                   "restored graph must compile before recall assertions");
  auto actualRecall = input;
  restored.processBlock(actualRecall, midi);

  float recallDifference = 0.0f;
  for (int channel = 0; channel < actualRecall.getNumChannels(); ++channel) {
    for (int sample = 0; sample < blockSize; ++sample) {
      recallDifference =
          std::max(recallDifference,
                   std::abs(actualRecall.getSample(channel, sample) -
                            expectedRecall.getSample(channel, sample)));
    }
  }
  passed &= expect(
      recallDifference < 1.0e-6f,
      "serialized parameters and weights must restore exact sonic state");

  openyourbox::graph::NodeGraph restoredGraph;
  passed &=
      expect(restoredGraph.restoreFromValueTree(restored.getGraphState()),
             "serialized processor state must restore the graph document");
  const auto *restoredConvolution = restoredGraph.findNode(convolutionNode);
  passed &= expect(restoredConvolution != nullptr &&
                       restoredConvolution->seed == 123456,
                   "signed per-element randomization seed must survive recall");

  if (restoredConvolution != nullptr) {
    restoredGraph.setProperty(restoredConvolution->id, "dilation", 2);
    restored.setGraphState(restoredGraph.toValueTree());
    passed &= expect(waitForGraphRuntime(restored, 5),
                     "edited graph must publish a replacement runtime");
    auto transition = input;
    restored.processBlock(transition, midi);
    auto transitionIsFinite = true;
    for (int channel = 0; channel < transition.getNumChannels(); ++channel) {
      for (int sample = 0; sample < transition.getNumSamples(); ++sample)
        transitionIsFinite &=
            std::isfinite(transition.getSample(channel, sample));
    }
    passed &= expect(
        transitionIsFinite &&
            transition.getMagnitude(0, 0, transition.getNumSamples()) > 0.0f,
        "atomic graph replacement must crossfade to finite non-silent audio");
  }

  juce::AudioBuffer<float> dcInput(2, blockSize);
  for (int iteration = 0; iteration < 64; ++iteration) {
    for (int channel = 0; channel < dcInput.getNumChannels(); ++channel)
      dcInput.clear(channel, 0, blockSize);
    for (int channel = 0; channel < dcInput.getNumChannels(); ++channel)
      juce::FloatVectorOperations::fill(dcInput.getWritePointer(channel), 0.2f,
                                        blockSize);
    restored.processBlock(dcInput, midi);
  }
  double dcMean = 0.0;
  for (int sample = 0; sample < blockSize; ++sample)
    dcMean += dcInput.getSample(0, sample);
  dcMean = std::abs(dcMean / static_cast<double>(blockSize));
  passed &= expect(dcMean < 0.001,
                   "post-graph DC blocker must reject sustained offset");

  openyourbox::graph::NodeGraph analysisGraph;
  const auto analysisInput = analysisGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto analysisActivation = analysisGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, 0.0f});
  const auto analysisOutput = analysisGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  analysisGraph.connect(
      analysisGraph.findNode(analysisInput)->outputs.front().id,
      analysisGraph.findNode(analysisActivation)->inputs.front().id);
  analysisGraph.connect(
      analysisGraph.findNode(analysisActivation)->outputs.front().id,
      analysisGraph.findNode(analysisOutput)->inputs.front().id);
  analysisGraph.setFloatProperty(analysisActivation, "gain", 2.5f);
  const auto xy = analysisGraph.addNode(
      openyourbox::graph::NodeType::xyTrackpad, {0.0f, 120.0f});
  analysisGraph.setConditioningPad(xy, 0.25f, -0.5f);
  const auto savedAnalysis = analysisGraph.toValueTree();
  openyourbox::graph::NodeGraph recalledAnalysis;
  passed &= expect(recalledAnalysis.restoreFromValueTree(savedAnalysis),
                   "analysis graph with Gain and XY must restore");
  const auto *recalledActivation =
      recalledAnalysis.findNode(analysisActivation);
  const auto *recalledXy = recalledAnalysis.findNode(xy);
  float recalledGain = 0.0f;
  if (recalledActivation != nullptr) {
    for (const auto &property : recalledActivation->properties) {
      if (property.key == "gain")
        recalledGain = property.floatValue;
    }
  }
  passed &= expect(std::abs(recalledGain - 2.5f) < 1.0e-4f,
                   "Gain must survive processor-style graph recall");
  passed &= expect(recalledXy != nullptr &&
                       std::abs(recalledXy->conditioningX - 0.25f) < 1.0e-4f &&
                       std::abs(recalledXy->conditioningY + 0.5f) < 1.0e-4f,
                   "XY pad values must survive graph recall");

  restored.setGraphState(analysisGraph.toValueTree());
  passed &= expect(waitForGraphRuntime(restored, 1),
                   "Gain Activation graph must publish a runtime");
  auto gained = input;
  restored.processBlock(gained, midi);
  passed &= expect(gained.getMagnitude(0, 0, blockSize) > 0.0f,
                   "Gain on Activation must remain audible");

  openyourbox::graph::NodeGraph disconnected;
  passed &=
      expect(disconnected.restoreFromValueTree(restored.getGraphState()),
             "connected graph state must restore for the silence check");
  std::vector<std::int32_t> outputLinks;
  for (const auto &link : disconnected.getLinks()) {
    const auto destination = disconnected.findNodeForPin(link.destinationPinId);
    const auto *destinationNode =
        destination.has_value() ? disconnected.findNode(*destination) : nullptr;
    if (destinationNode != nullptr &&
        destinationNode->type == openyourbox::graph::NodeType::audioOutput)
      outputLinks.push_back(link.id);
  }
  for (const auto linkId : outputLinks)
    disconnected.removeLink(linkId);
  restored.setGraphState(disconnected.toValueTree());
  bool graphWentSilent = false;
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (restored.getReceptiveFieldSamples() == 0)
      graphWentSilent = true;
    if (graphWentSilent)
      break;
    juce::Thread::sleep(10);
  }
  auto muted = input;
  restored.processBlock(muted, midi);
  passed &= expect(graphWentSilent &&
                       muted.getMagnitude(0, 0, muted.getNumSamples()) == 0.0f,
                   "audio output with no input must produce silence");

  {
    OpenYourBoxAudioProcessor activationProcessor;
    auto configuration = activationProcessor.getRequestedConfiguration();
    configuration.activation = openyourbox::dsp::ActivationType::prelu;
    activationProcessor.applyGraphConfiguration(configuration);
    passed &= expect(activationProcessor.getRequestedConfiguration().activation ==
                         openyourbox::dsp::ActivationType::prelu,
                     "PReLU must persist through the host activation parameter");
  }

  if (passed)
    std::cout << "OpenYourBox processor integration tests passed\n";
  return passed ? 0 : 1;
}
