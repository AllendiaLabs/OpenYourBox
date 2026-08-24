#include "dsp/LiveGraphEngine.h"
#include "library/TrainingLibrary.h"
#include "library/UserDataPaths.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>

namespace {
/**
 * @brief Reports a failed live-graph invariant.
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
 * @class TestFrozenKernel
 * @brief Stateless two-tap causal kernel used to verify runtime history.
 */
class TestFrozenKernel final : public openyourbox::dsp::FrozenBlackBoxKernel {
public:
  /** @brief Applies a causal current-plus-previous-sample operation. */
  torch::Tensor forward(const torch::Tensor &input) override {
    const auto delayed =
        torch::nn::functional::pad(
            input, torch::nn::functional::PadFuncOptions({1, 0}))
            .narrow(2, 0, input.size(2));
    return input + delayed;
  }
};

/**
 * @class TestFrozenFactory
 * @brief Supplies deterministic metadata and kernels for frozen graph tests.
 */
class TestFrozenFactory final : public openyourbox::dsp::FrozenBlackBoxFactory {
public:
  /** @brief Returns the stereo test input width. */
  int getInputChannels() const noexcept override { return 2; }
  /** @brief Returns the stereo test output width. */
  int getOutputChannels() const noexcept override { return 2; }
  /** @brief Returns the two-sample causal receptive field. */
  std::uint64_t getReceptiveField() const noexcept override { return 2; }
  /** @brief Reports that the test kernel owns no trainable parameters. */
  std::uint64_t getParameterCount() const noexcept override { return 0; }
  /** @brief Reports exact digital-silence preservation. */
  bool preservesSilence() const noexcept override { return true; }
  /** @brief Creates one runtime-local deterministic test kernel. */
  std::unique_ptr<openyourbox::dsp::FrozenBlackBoxKernel>
  createKernel() const override {
    return std::make_unique<TestFrozenKernel>();
  }
};

/**
 * @class TestConditioningKernel
 * @brief Frozen executor that scales audio by the live Control value.
 */
class TestConditioningKernel final : public openyourbox::dsp::FrozenBlackBoxKernel {
public:
  /** @brief Returns the audio unchanged when Control is absent. */
  torch::Tensor forward(const torch::Tensor &input) override { return input; }

  /**
   * @brief Scales audio by the last Control sample.
   * @param input Current audio block, possibly history-extended.
   * @param conditioning Live Knob/XY trajectory, or undefined.
   */
  torch::Tensor forwardWithConditioning(
      const torch::Tensor &input, const torch::Tensor &conditioning) override {
    if (!conditioning.defined() || conditioning.dim() != 3 ||
        conditioning.size(2) < 1)
      return input;
    return input * conditioning[0][0][conditioning.size(2) - 1];
  }
};

/**
 * @class TestConditioningFactory
 * @brief Frozen factory used to verify Control still reaches a Gold node.
 */
class TestConditioningFactory final : public openyourbox::dsp::FrozenBlackBoxFactory {
public:
  /** @brief Returns the stereo test input width. */
  int getInputChannels() const noexcept override { return 2; }
  /** @brief Returns the stereo test output width. */
  int getOutputChannels() const noexcept override { return 2; }
  /** @brief Returns a single-sample receptive field. */
  std::uint64_t getReceptiveField() const noexcept override { return 1; }
  /** @brief Reports that the test kernel owns no trainable parameters. */
  std::uint64_t getParameterCount() const noexcept override { return 0; }
  /** @brief FiLM Control injects bias, so silence is not preserved. */
  bool preservesSilence() const noexcept override { return false; }
  /** @brief Creates one runtime-local Control-aware test kernel. */
  std::unique_ptr<openyourbox::dsp::FrozenBlackBoxKernel>
  createKernel() const override {
    return std::make_unique<TestConditioningKernel>();
  }
};

/**
 * @brief Builds a valid stereo Conv1D graph for runtime tests.
 * @param firstConvolution Receives the first weighted node identifier.
 * @param secondConvolution Receives the second weighted node identifier.
 * @return Editable graph document.
 */
openyourbox::graph::NodeGraph makeGraph(std::int32_t &firstConvolution,
                                       std::int32_t &secondConvolution) {
  using namespace openyourbox::graph;
  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  firstConvolution = graph.addNode(NodeType::convolution, {180.0f, 0.0f});
  secondConvolution = graph.addNode(NodeType::convolution, {360.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {540.0f, 0.0f});
  graph.setProperty(firstConvolution, "channels", 2);
  graph.setProperty(secondConvolution, "channels", 2);
  graph.setSeed(firstConvolution, 42);
  graph.setSeed(secondConvolution, 91);

  const auto *inputNode = graph.findNode(input);
  const auto *firstNode = graph.findNode(firstConvolution);
  const auto *secondNode = graph.findNode(secondConvolution);
  const auto *outputNode = graph.findNode(output);
  if (inputNode != nullptr && firstNode != nullptr && secondNode != nullptr &&
      outputNode != nullptr) {
    graph.connect(inputNode->outputs.front().id, firstNode->inputs.front().id);
    graph.connect(firstNode->outputs.front().id, secondNode->inputs.front().id);
    graph.connect(secondNode->outputs.front().id,
                  outputNode->inputs.front().id);
  }
  return graph;
}

/**
 * @brief Checks that an unused XY axis does not change TCN control.
 * @param yAxis When true only Y is cabled; otherwise only X is cabled.
 * @param options Host compile options used by the surrounding tests.
 * @return True when the unused axis is ignored and the used axis still modulates.
 */
bool unusedXyAxisDoesNotModulateTcn(
    bool yAxis, const openyourbox::dsp::LiveGraphCompileOptions &options) {
  using namespace openyourbox::dsp;
  using namespace openyourbox::graph;
  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto pad = graph.addNode(NodeType::xyTrackpad, {0.0f, 80.0f});
  const auto tcn = graph.addNode(NodeType::tcn, {180.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
  graph.setSeed(tcn, 23);
  const auto *tcnNode = graph.findNode(tcn);
  const auto *padNode = graph.findNode(pad);
  if (tcnNode == nullptr || padNode == nullptr || padNode->outputs.size() < 2)
    return false;
  if (!graph
           .connect(graph.findNode(input)->outputs.front().id,
                    tcnNode->inputs.front().id)
           .accepted)
    return false;
  const auto sourcePin =
      yAxis ? padNode->outputs[1].id : padNode->outputs[0].id;
  if (!graph.connect(sourcePin, tcnNode->inputs[1].id).accepted)
    return false;
  if (!graph
           .connect(tcnNode->outputs.front().id,
                    graph.findNode(output)->inputs.front().id)
           .accepted)
    return false;

  const auto compiled = LiveGraphEngine::compile(graph, options);
  if (!compiled.succeeded())
    return false;

  auto processAt = [&](float x, float y) -> torch::Tensor {
    graph.setConditioningPad(pad, x, y);
    LiveGraphCompileError prepareError;
    const auto runtime = LiveGraphEngine::prepare(compiled.snapshot, prepareError);
    if (runtime == nullptr)
      return {};
    runtime->bindControls(std::make_shared<const RuntimeControlState>(
        collectRuntimeControlState(graph)));
    return runtime->processTensor(torch::ones({1, 2, 32}, torch::kFloat32));
  };

  const auto baseline = processAt(0.35f, 0.15f);
  const auto unusedChanged =
      yAxis ? processAt(0.95f, 0.15f) : processAt(0.35f, 0.95f);
  const auto usedChanged =
      yAxis ? processAt(0.35f, 0.85f) : processAt(0.85f, 0.15f);
  return baseline.defined() && unusedChanged.defined() &&
         usedChanged.defined() &&
         torch::allclose(baseline, unusedChanged, 1.0e-6, 1.0e-6) &&
         !torch::allclose(baseline, usedChanged, 1.0e-5, 1.0e-5);
}

/**
 * @brief Checks that concatenated XY controls TCN as `[x, y]`, not `x+y`.
 * @param options Host compile options used by the surrounding tests.
 * @return True when `(1,0)` and `(0,1)` produce different TCN outputs.
 */
bool concatenatedXyControlsTcnAsVector(
    const openyourbox::dsp::LiveGraphCompileOptions &options) {
  using namespace openyourbox::dsp;
  using namespace openyourbox::graph;
  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto pad = graph.addNode(NodeType::xyTrackpad, {0.0f, 80.0f});
  const auto merge = graph.addNode(NodeType::merge, {180.0f, 80.0f});
  const auto tcn = graph.addNode(NodeType::tcn, {360.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {540.0f, 0.0f});
  graph.setSeed(tcn, 23);
  graph.setProperty(merge, "mode", static_cast<int>(MergeMode::concatenate));
  const auto *tcnNode = graph.findNode(tcn);
  const auto *padNode = graph.findNode(pad);
  const auto *mergeNode = graph.findNode(merge);
  if (tcnNode == nullptr || padNode == nullptr || mergeNode == nullptr ||
      padNode->outputs.size() < 2 || mergeNode->inputs.size() < 2 ||
      tcnNode->inputs.size() < 2)
    return false;
  if (!graph
           .connect(graph.findNode(input)->outputs.front().id,
                    tcnNode->inputs.front().id)
           .accepted)
    return false;
  if (!graph.connect(padNode->outputs[0].id, mergeNode->inputs[0].id).accepted)
    return false;
  if (!graph.connect(padNode->outputs[1].id, mergeNode->inputs[1].id).accepted)
    return false;
  if (!graph.connect(mergeNode->outputs.front().id, tcnNode->inputs[1].id)
           .accepted)
    return false;
  if (!graph
           .connect(tcnNode->outputs.front().id,
                    graph.findNode(output)->inputs.front().id)
           .accepted)
    return false;

  LiveGraphCompileOptions local = options;
  local.controlRampSeconds = 0.0;
  const auto compiled = LiveGraphEngine::compile(graph, local);
  if (!compiled.succeeded())
    return false;

  auto processAt = [&](float x, float y) -> torch::Tensor {
    graph.setConditioningPad(pad, x, y);
    LiveGraphCompileError prepareError;
    const auto runtime = LiveGraphEngine::prepare(compiled.snapshot, prepareError);
    if (runtime == nullptr)
      return {};
    runtime->bindControls(std::make_shared<const RuntimeControlState>(
        collectRuntimeControlState(graph)));
    return runtime->processTensor(torch::ones({1, 2, 32}, torch::kFloat32));
  };

  const auto xOnly = processAt(1.0f, 0.0f);
  const auto yOnly = processAt(0.0f, 1.0f);
  return xOnly.defined() && yOnly.defined() &&
         !torch::allclose(xOnly, yOnly, 1.0e-5, 1.0e-5);
}

/**
 * @brief Checks that FiLM control history is causal across audio blocks.
 *
 * Splitting a ramping Knob across two buffers must match one long buffer.
 * Repeating the first sample of each block over the TCN RF would diverge.
 *
 * @param options Host compile options used by the surrounding tests.
 * @return True when streamed blocks match the long-block reference.
 */
bool filmControlHistoryIsCausalAcrossBlocks(
    const openyourbox::dsp::LiveGraphCompileOptions &options) {
  using namespace openyourbox::dsp;
  using namespace openyourbox::graph;
  NodeGraph graph;
  const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto knob = graph.addNode(NodeType::knobInput, {0.0f, 80.0f});
  const auto tcn = graph.addNode(NodeType::tcn, {180.0f, 0.0f});
  const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
  graph.setSeed(tcn, 23);
  graph.setProperty(tcn, "depth", 2);
  graph.setProperty(tcn, "kernel_size", 3);
  graph.setProperty(tcn, "channels", 8);
  const auto *tcnNode = graph.findNode(tcn);
  if (tcnNode == nullptr || tcnNode->inputs.size() < 2)
    return false;
  if (!graph
           .connect(graph.findNode(input)->outputs.front().id,
                    tcnNode->inputs.front().id)
           .accepted)
    return false;
  if (!graph.connect(graph.findNode(knob)->outputs.front().id,
                     tcnNode->inputs[1].id)
           .accepted)
    return false;
  if (!graph
           .connect(tcnNode->outputs.front().id,
                    graph.findNode(output)->inputs.front().id)
           .accepted)
    return false;

  LiveGraphCompileOptions local = options;
  local.sampleRate = 48000.0;
  local.controlRampSeconds = 0.001;
  const auto compiled = LiveGraphEngine::compile(graph, local);
  if (!compiled.succeeded())
    return false;

  RuntimeControlState controls;
  controls.conditioningByNodeId[knob] = {1.0f, 0.0f};

  LiveGraphCompileError streamedError;
  const auto streamed = LiveGraphEngine::prepare(compiled.snapshot, streamedError);
  if (streamed == nullptr)
    return false;
  streamed->bindControls(
      std::make_shared<const RuntimeControlState>(controls));
  auto first = streamed->processTensor(torch::ones({1, 2, 16}, torch::kFloat32));
  auto second =
      streamed->processTensor(torch::ones({1, 2, 16}, torch::kFloat32));

  LiveGraphCompileError longError;
  const auto combined = LiveGraphEngine::prepare(compiled.snapshot, longError);
  if (combined == nullptr)
    return false;
  combined->bindControls(
      std::make_shared<const RuntimeControlState>(controls));
  auto reference =
      combined->processTensor(torch::ones({1, 2, 32}, torch::kFloat32));
  if (!first.defined() || !second.defined() || !reference.defined())
    return false;
  const auto streamedOut = torch::cat({first, second}, 2);
  return torch::allclose(streamedOut, reference, 1.0e-4, 1.0e-4);
}
} // namespace

/**
 * @brief Runs immutable live-graph compilation and randomization checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using namespace openyourbox::dsp;
  std::int32_t firstConvolution = 0;
  std::int32_t secondConvolution = 0;
  const auto graph = makeGraph(firstConvolution, secondConvolution);

  LiveGraphCompileOptions options;
  options.hostInputChannels = 2;
  options.hostOutputChannels = 2;
  options.maximumBlockSize = 256;
  const auto compiled = LiveGraphEngine::compile(graph, options);
  bool passed = true;
  passed &= expect(compiled.succeeded(),
                   "valid stereo graph must compile to an immutable snapshot");
  if (!compiled.succeeded())
    return 1;
  passed &= expect(compiled.snapshot->getElementStatistics().size() == 4,
                   "compiled graph must retain every topological element");

  LiveGraphCompileError error;
  const auto runtime = LiveGraphEngine::prepare(compiled.snapshot, error);
  passed &= expect(runtime != nullptr && !error.hasError(),
                   "compiled graph must prepare outside the audio callback");
  if (runtime == nullptr)
    return 1;

  const auto silence = torch::zeros({1, 2, 128}, torch::kFloat32);
  const auto silentOutput = runtime->processTensor(silence);
  passed &= expect(silentOutput.abs().max().item<float>() == 0.0f,
                   "bias-free live graph must preserve digital silence");
  const auto excited = runtime->processTensor(torch::ones({1, 2, 128}, torch::kFloat32));
  const auto tail = runtime->processTensor(silence);
  passed &= expect(excited.defined() && tail.defined() &&
                       tail.abs().max().item<float>() > 1.0e-6f,
                   "causal history must keep sounding into following silence");

  openyourbox::graph::NodeGraph linearGraph;
  const auto linearInput = linearGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto linearNode =
      linearGraph.addNode(openyourbox::graph::NodeType::linear, {180.0f, 0.0f});
  const auto linearOutput = linearGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  linearGraph.setSeed(linearNode, 42);
  linearGraph.connect(linearGraph.findNode(linearInput)->outputs.front().id,
                      linearGraph.findNode(linearNode)->inputs.front().id);
  linearGraph.connect(linearGraph.findNode(linearNode)->outputs.front().id,
                      linearGraph.findNode(linearOutput)->inputs.front().id);
  const auto linearCompiled = LiveGraphEngine::compile(linearGraph, options);
  const auto linearRuntime =
      LiveGraphEngine::prepare(linearCompiled.snapshot, error);
  passed &= expect(linearRuntime != nullptr,
                   "linear fixture must prepare");
  if (linearRuntime == nullptr)
    return 1;
  auto basis = torch::zeros({1, 2, 1}, torch::kFloat32);
  basis[0][0][0] = 1.0f;
  const auto linearResult = linearRuntime->processTensor(basis);
  passed &= expect(linearResult.defined() && linearResult.abs().max().item<float>() > 0.0f,
                   "linear live weights must produce a non-silent output");

  const auto repeated =
      compiled.snapshot->withRandomizedElement(firstConvolution, 42, error);
  passed &= expect(repeated != nullptr && !error.hasError(),
                   "same signed seed must rebuild one weighted element");
  const auto repeatedRuntime = LiveGraphEngine::prepare(repeated, error);
  passed &= expect(repeatedRuntime != nullptr && !error.hasError(),
                   "same-seed snapshot must prepare successfully");
  if (repeatedRuntime == nullptr)
    return 1;
  const auto input = torch::randn({1, 2, 128}, torch::kFloat32);
  const auto originalOutput = runtime->processTensor(input);
  const auto repeatedOutput = repeatedRuntime->processTensor(input);
  passed &= expect(torch::equal(originalOutput, repeatedOutput),
                   "reapplying an element's seed must reproduce graph output");

  const auto changed =
      compiled.snapshot->withRandomizedElement(firstConvolution, -7, error);
  const auto changedRuntime = LiveGraphEngine::prepare(changed, error);
  passed &= expect(changedRuntime != nullptr && !error.hasError(),
                   "different-seed snapshot must prepare successfully");
  if (changedRuntime == nullptr)
    return 1;
  const auto changedOutput = changedRuntime->processTensor(input);
  passed &= expect(!torch::equal(originalOutput, changedOutput),
                   "a different signed seed must change the target element");

  openyourbox::graph::NodeGraph frozenGraph;
  const auto frozenInput = frozenGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto frozenNode = frozenGraph.addNode(
      openyourbox::graph::NodeType::blackBox, {180.0f, 0.0f});
  const auto frozenOutput = frozenGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  frozenGraph.findNode(frozenNode)->artifactPath = "test-frozen-artifact";
  frozenGraph.connect(frozenGraph.findNode(frozenInput)->outputs.front().id,
                      frozenGraph.findNode(frozenNode)->inputs.front().id);
  frozenGraph.connect(frozenGraph.findNode(frozenNode)->outputs.front().id,
                      frozenGraph.findNode(frozenOutput)->inputs.front().id);
  const auto factory = std::make_shared<TestFrozenFactory>();
  const auto frozenCompiled = LiveGraphEngine::compile(
      frozenGraph, options,
      [factory](const openyourbox::graph::GraphNode &) { return factory; });
  passed &= expect(frozenCompiled.succeeded(),
                   "valid frozen graph must compile with prepared metadata");
  const auto wholeRuntime =
      LiveGraphEngine::prepare(frozenCompiled.snapshot, error);
  const auto splitRuntime =
      LiveGraphEngine::prepare(frozenCompiled.snapshot, error);
  passed &= expect(wholeRuntime != nullptr && splitRuntime != nullptr,
                   "frozen graph must prepare independent runtime histories");
  if (wholeRuntime == nullptr || splitRuntime == nullptr)
    return 1;
  const auto frozenInputTensor = torch::randn({1, 2, 128}, torch::kFloat32);
  const auto wholeFrozenOutput = wholeRuntime->processTensor(frozenInputTensor);
  const auto firstHalf =
      splitRuntime->processTensor(frozenInputTensor.narrow(2, 0, 64));
  const auto secondHalf =
      splitRuntime->processTensor(frozenInputTensor.narrow(2, 64, 64));
  const auto splitFrozenOutput = torch::cat({firstHalf, secondHalf}, 2);
  passed &= expect(
      torch::equal(wholeFrozenOutput, splitFrozenOutput),
      "frozen causal history must produce block-size-independent output");
  passed &= expect(
      splitRuntime->getFrozenInferenceTimeMilliseconds(frozenNode) > 0.0,
      "frozen runtime must publish a live per-buffer inference duration");

  openyourbox::graph::NodeGraph frozenControlGraph;
  const auto frozenControlInput = frozenControlGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto frozenControlKnob = frozenControlGraph.addNode(
      openyourbox::graph::NodeType::knobInput, {0.0f, 80.0f});
  const auto frozenControlBox = frozenControlGraph.addNode(
      openyourbox::graph::NodeType::blackBox, {180.0f, 0.0f});
  const auto frozenControlOutput = frozenControlGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  frozenControlGraph.findNode(frozenControlBox)->artifactPath =
      "test-conditioned-frozen-artifact";
  frozenControlGraph.setConditioningValue(frozenControlKnob, 2.0f);
  passed &= expect(
      frozenControlGraph
          .connect(frozenControlGraph.findNode(frozenControlInput)
                       ->outputs.front()
                       .id,
                   frozenControlGraph.findNode(frozenControlBox)
                       ->inputs.front()
                       .id)
          .accepted &&
          frozenControlGraph
              .connect(frozenControlGraph.findNode(frozenControlKnob)
                           ->outputs.front()
                           .id,
                       frozenControlGraph.findNode(frozenControlBox)->inputs[1].id)
              .accepted &&
          frozenControlGraph
              .connect(frozenControlGraph.findNode(frozenControlBox)
                           ->outputs.front()
                           .id,
                       frozenControlGraph.findNode(frozenControlOutput)
                           ->inputs.front()
                           .id)
              .accepted,
      "frozen BlackBox Control pin must accept Knob");
  const auto conditionedFactory = std::make_shared<TestConditioningFactory>();
  LiveGraphCompileOptions freezeControlOptions = options;
  freezeControlOptions.controlRampSeconds = 0.0;
  const auto frozenControlCompiled = LiveGraphEngine::compile(
      frozenControlGraph, freezeControlOptions,
      [conditionedFactory](const openyourbox::graph::GraphNode &) {
        return conditionedFactory;
      });
  LiveGraphCompileError freezeControlError;
  const auto frozenControlRuntime = LiveGraphEngine::prepare(
      frozenControlCompiled.snapshot, freezeControlError);
  passed &= expect(frozenControlCompiled.succeeded() &&
                       frozenControlRuntime != nullptr,
                   "frozen BlackBox with Control must compile and prepare");
  if (frozenControlRuntime != nullptr) {
    frozenControlRuntime->bindControls(
        std::make_shared<const RuntimeControlState>(
            collectRuntimeControlState(frozenControlGraph)));
    const auto scaled = frozenControlRuntime->processTensor(
        torch::ones({1, 2, 8}, torch::kFloat32));
    passed &= expect(
        scaled.defined() &&
            std::abs(scaled[0][0][0].item<float>() - 2.0f) < 1.0e-4f,
        "frozen Gold node must still be steered by live Knob Control");
  }

  const auto invalidRandomization =
      compiled.snapshot->withRandomizedElement(999999, 1, error);
  passed &= expect(invalidRandomization == nullptr &&
                       error.code == LiveGraphErrorCode::invalidRandomization,
                   "unknown element randomization must fail without mutation");

  openyourbox::graph::NodeGraph mixerGraph;
  const auto mixerInput = mixerGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto leftActivation = mixerGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, -40.0f});
  const auto rightActivation = mixerGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, 80.0f});
  const auto mergeNode =
      mixerGraph.addNode(openyourbox::graph::NodeType::merge, {360.0f, 20.0f});
  const auto mixerOutput = mixerGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {540.0f, 20.0f});
  mixerGraph.connect(mixerGraph.findNode(mixerInput)->outputs.front().id,
                     mixerGraph.findNode(leftActivation)->inputs.front().id);
  mixerGraph.connect(mixerGraph.findNode(mixerInput)->outputs.front().id,
                     mixerGraph.findNode(rightActivation)->inputs.front().id);
  mixerGraph.connect(mixerGraph.findNode(leftActivation)->outputs.front().id,
                     mixerGraph.findNode(mergeNode)->inputs[0].id);
  mixerGraph.connect(mixerGraph.findNode(rightActivation)->outputs.front().id,
                     mixerGraph.findNode(mergeNode)->inputs[1].id);
  mixerGraph.connect(mixerGraph.findNode(mergeNode)->outputs.front().id,
                     mixerGraph.findNode(mixerOutput)->inputs.front().id);
  const auto mixerCompiled = LiveGraphEngine::compile(mixerGraph, options);
  passed &= expect(mixerCompiled.succeeded(),
                   "elementwise merge add of two stereo paths must compile");
  const auto mixerRuntime =
      LiveGraphEngine::prepare(mixerCompiled.snapshot, error);
  passed &= expect(mixerRuntime != nullptr,
                   "elementwise merge add graph must prepare");
  if (mixerRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto merged = mixerRuntime->processTensor(ones);
    passed &= expect(std::abs(merged[0][0][0].item<float>() - 2.0f) < 1.0e-6f,
                     "ReLU merge add of two unit paths must double the input");
  }

  openyourbox::graph::NodeGraph multiplyMergeGraph;
  const auto multiplyMergeInput = multiplyMergeGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto leftPath = multiplyMergeGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, -40.0f});
  const auto rightPath = multiplyMergeGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, 80.0f});
  const auto multiplyMergeNode = multiplyMergeGraph.addNode(
      openyourbox::graph::NodeType::merge, {360.0f, 20.0f});
  const auto multiplyMergeOutput = multiplyMergeGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {540.0f, 20.0f});
  multiplyMergeGraph.setProperty(multiplyMergeNode, "mode", 1);
  multiplyMergeGraph.connect(
      multiplyMergeGraph.findNode(multiplyMergeInput)->outputs.front().id,
      multiplyMergeGraph.findNode(leftPath)->inputs.front().id);
  multiplyMergeGraph.connect(
      multiplyMergeGraph.findNode(multiplyMergeInput)->outputs.front().id,
      multiplyMergeGraph.findNode(rightPath)->inputs.front().id);
  multiplyMergeGraph.connect(
      multiplyMergeGraph.findNode(leftPath)->outputs.front().id,
      multiplyMergeGraph.findNode(multiplyMergeNode)->inputs[0].id);
  multiplyMergeGraph.connect(
      multiplyMergeGraph.findNode(rightPath)->outputs.front().id,
      multiplyMergeGraph.findNode(multiplyMergeNode)->inputs[1].id);
  multiplyMergeGraph.connect(
      multiplyMergeGraph.findNode(multiplyMergeNode)->outputs.front().id,
      multiplyMergeGraph.findNode(multiplyMergeOutput)->inputs.front().id);
  const auto multiplyMergeCompiled =
      LiveGraphEngine::compile(multiplyMergeGraph, options);
  passed &= expect(multiplyMergeCompiled.succeeded(),
                   "elementwise merge multiply of two stereo paths must compile");
  const auto multiplyMergeRuntime =
      LiveGraphEngine::prepare(multiplyMergeCompiled.snapshot, error);
  passed &= expect(multiplyMergeRuntime != nullptr,
                   "elementwise merge multiply graph must prepare");
  if (multiplyMergeRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto product = multiplyMergeRuntime->processTensor(ones);
    passed &= expect(std::abs(product[0][0][0].item<float>() - 1.0f) < 1.0e-6f,
                     "ReLU merge multiply of two unit paths must stay unity");
  }

  openyourbox::graph::NodeGraph concatGraph;
  const auto concatInput = concatGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto concatNode = concatGraph.addNode(
      openyourbox::graph::NodeType::merge, {180.0f, 0.0f});
  const auto concatOutput = concatGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  concatGraph.setProperty(concatNode, "mode", 2);
  concatGraph.setProperty(concatNode, "inputs", 2);
  concatGraph.connect(concatGraph.findNode(concatInput)->outputs.front().id,
                      concatGraph.findNode(concatNode)->inputs[0].id);
  concatGraph.connect(concatGraph.findNode(concatNode)->outputs.front().id,
                      concatGraph.findNode(concatOutput)->inputs.front().id);
  const auto concatCompiled = LiveGraphEngine::compile(concatGraph, options);
  passed &= expect(concatCompiled.succeeded(),
                   "a mixer with unused inputs must still compile");
  const auto concatRuntime =
      LiveGraphEngine::prepare(concatCompiled.snapshot, error);
  passed &= expect(concatRuntime != nullptr,
                   "a mixer with unused inputs must prepare");
  if (concatRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto concatenated = concatRuntime->processTensor(ones);
    passed &= expect(concatenated.size(1) == 2 &&
                         std::abs(concatenated[0][0][0].item<float>() - 1.0f) <
                             1.0e-6f,
                     "merge concatenate must omit unused inputs without extra channels");
  }

  openyourbox::graph::NodeGraph addMergeGraph;
  const auto unusedAddInput = addMergeGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto unusedAddNode =
      addMergeGraph.addNode(openyourbox::graph::NodeType::merge, {180.0f, 0.0f});
  const auto unusedAddOutput = addMergeGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  addMergeGraph.connect(addMergeGraph.findNode(unusedAddInput)->outputs.front().id,
                        addMergeGraph.findNode(unusedAddNode)->inputs[0].id);
  addMergeGraph.connect(addMergeGraph.findNode(unusedAddNode)->outputs.front().id,
                        addMergeGraph.findNode(unusedAddOutput)->inputs.front().id);
  const auto addCompiled = LiveGraphEngine::compile(addMergeGraph, options);
  const auto addRuntime = LiveGraphEngine::prepare(addCompiled.snapshot, error);
  passed &= expect(addCompiled.succeeded() && addRuntime != nullptr,
                   "merge add with one connected input must compile");
  if (addRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto merged = addRuntime->processTensor(ones);
    passed &= expect(std::abs(merged[0][0][0].item<float>() - 1.0f) < 1.0e-6f,
                     "an unused merge add input must contribute zeros");
  }

  openyourbox::graph::NodeGraph multiplyGraph;
  const auto multiplyInput = multiplyGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto multiplyNode = multiplyGraph.addNode(
      openyourbox::graph::NodeType::merge, {180.0f, 0.0f});
  const auto multiplyOutput = multiplyGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  multiplyGraph.setProperty(multiplyNode, "mode", 1);
  multiplyGraph.connect(multiplyGraph.findNode(multiplyInput)->outputs.front().id,
                        multiplyGraph.findNode(multiplyNode)->inputs[0].id);
  multiplyGraph.connect(multiplyGraph.findNode(multiplyNode)->outputs.front().id,
                        multiplyGraph.findNode(multiplyOutput)->inputs.front().id);
  const auto multiplyCompiled = LiveGraphEngine::compile(multiplyGraph, options);
  const auto multiplyRuntime =
      LiveGraphEngine::prepare(multiplyCompiled.snapshot, error);
  passed &= expect(multiplyCompiled.succeeded() && multiplyRuntime != nullptr,
                   "merge multiply with one connected input must compile");
  if (multiplyRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto product = multiplyRuntime->processTensor(ones);
    passed &= expect(std::abs(product[0][0][0].item<float>() - 1.0f) < 1.0e-6f,
                     "an unused merge multiply input must contribute ones");
  }

  openyourbox::graph::NodeGraph orphanGraph;
  const auto orphanInput = orphanGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto orphanOutput = orphanGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {180.0f, 0.0f});
  orphanGraph.addNode(openyourbox::graph::NodeType::convolution, {90.0f, 80.0f});
  orphanGraph.addNode(openyourbox::graph::NodeType::tcn, {90.0f, -80.0f});
  orphanGraph.connect(orphanGraph.findNode(orphanInput)->outputs.front().id,
                      orphanGraph.findNode(orphanOutput)->inputs.front().id);
  const auto orphanCompiled = LiveGraphEngine::compile(orphanGraph, options);
  passed &= expect(orphanCompiled.succeeded(),
                   "unwired Conv1D and TCN nodes must not fail compilation");
  passed &= expect(orphanCompiled.snapshot->getElementStatistics().size() == 2,
                   "unwired processing nodes must be omitted from the live path");
  const auto orphanRuntime =
      LiveGraphEngine::prepare(orphanCompiled.snapshot, error);
  passed &= expect(orphanRuntime != nullptr,
                   "a graph with unwired Conv1D and TCN must prepare");
  if (orphanRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto passedThrough = orphanRuntime->processTensor(ones);
    passed &= expect(torch::equal(passedThrough, ones),
                     "unwired Conv1D and TCN must leave the I/O path unchanged");
  }

  openyourbox::graph::NodeGraph openGraph;
  openGraph.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  openGraph.addNode(openyourbox::graph::NodeType::audioOutput, {180.0f, 0.0f});
  const auto openCompiled = LiveGraphEngine::compile(openGraph, options);
  passed &= expect(!openCompiled.succeeded() &&
                       openCompiled.error.code == LiveGraphErrorCode::incompletePath,
                   "disconnected stereo I/O must stay idle without a live runtime");

  openyourbox::graph::NodeGraph activationGraph;
  const auto activationInput = activationGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto activationNode = activationGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, 0.0f});
  const auto activationOutput = activationGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  activationGraph.connect(
      activationGraph.findNode(activationInput)->outputs.front().id,
      activationGraph.findNode(activationNode)->inputs.front().id);
  activationGraph.connect(
      activationGraph.findNode(activationNode)->outputs.front().id,
      activationGraph.findNode(activationOutput)->inputs.front().id);
  passed &= expect(activationGraph.setFloatProperty(activationNode, "gain", 4.0f),
                   "Activation Gain must persist as a real property");
  const auto activationCompiled =
      LiveGraphEngine::compile(activationGraph, options);
  passed &= expect(activationCompiled.succeeded(),
                   "Activation graph with Gain must compile");

  openyourbox::dsp::AnalysisRequest analysisRequest;
  analysisRequest.nodeId = activationNode;
  analysisRequest.view = openyourbox::graph::AnalysisView::transfer;
  analysisRequest.revision = 1;
  const auto transfer = LiveGraphEngine::analyse(activationGraph,
                                                 analysisRequest, options);
  passed &= expect(transfer.channelCount >= 2 &&
                       transfer.chainSeries.size() ==
                           static_cast<std::size_t>(transfer.channelCount) &&
                       transfer.elementOnlySeries.size() ==
                           static_cast<std::size_t>(transfer.channelCount),
                   "analysis must return dual N-channel curve families");
  analysisRequest.view = openyourbox::graph::AnalysisView::frequency;
  const auto frequency = LiveGraphEngine::analyse(activationGraph,
                                                  analysisRequest, options);
  passed &= expect(frequency.sourceMode == AnalysisSourceMode::probe &&
                       !frequency.chainSeries.empty() &&
                       !frequency.chainSeries.front().x.empty(),
                   "frequency analysis must fall back to a probe when silent");
  analysisRequest.view = openyourbox::graph::AnalysisView::oscilloscope;
  analysisRequest.sampleRate = 44100.0;
  const auto oscilloscope = LiveGraphEngine::analyse(activationGraph,
                                                     analysisRequest, options);
  passed &= expect(oscilloscope.view ==
                           openyourbox::graph::AnalysisView::oscilloscope &&
                       oscilloscope.sourceMode == AnalysisSourceMode::probe &&
                       oscilloscope.elementOnlySeries.size() >= 2 &&
                       oscilloscope.elementOnlySeries.front().x.size() >= 2 &&
                       oscilloscope.elementOnlySeries.front().y.size() >= 2,
                   "oscilloscope analysis must return element waveforms");

  openyourbox::graph::NodeGraph knobGraph;
  const auto knobInput = knobGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto knob = knobGraph.addNode(openyourbox::graph::NodeType::knobInput,
                                      {0.0f, 80.0f});
  const auto knobMerge =
      knobGraph.addNode(openyourbox::graph::NodeType::merge, {180.0f, 20.0f});
  const auto knobTcn =
      knobGraph.addNode(openyourbox::graph::NodeType::tcn, {360.0f, 20.0f});
  const auto knobOutput = knobGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {540.0f, 20.0f});
  knobGraph.setProperty(knobTcn, "channels", 2);
  knobGraph.setConditioningValue(knob, 1.5f);
  passed &= expect(
      knobGraph
          .connect(knobGraph.findNode(knobInput)->outputs.front().id,
                   knobGraph.findNode(knobMerge)->inputs[0].id)
          .accepted,
      "Audio In must connect to Merge");
  passed &= expect(
      knobGraph
          .connect(knobGraph.findNode(knob)->outputs.front().id,
                   knobGraph.findNode(knobMerge)->inputs[1].id)
          .accepted,
      "Knob Input must connect to Merge");
  passed &= expect(
      knobGraph
          .connect(knobGraph.findNode(knobMerge)->outputs.front().id,
                   knobGraph.findNode(knobTcn)->inputs.front().id)
          .accepted,
      "Merge must connect to TCN");
  passed &= expect(
      knobGraph
          .connect(knobGraph.findNode(knobTcn)->outputs.front().id,
                   knobGraph.findNode(knobOutput)->inputs.front().id)
          .accepted,
      "TCN must connect to Audio Out");
  const auto knobCompiled = LiveGraphEngine::compile(knobGraph, options);
  passed &= expect(knobCompiled.succeeded(),
                   "Knob-through-Merge graph must compile");
  LiveGraphCompileError knobError;
  const auto knobRuntime =
      LiveGraphEngine::prepare(knobCompiled.snapshot, knobError);
  passed &= expect(knobRuntime != nullptr, "Knob graph must prepare");
  if (knobRuntime != nullptr) {
    auto controls = collectRuntimeControlState(knobGraph);
    knobRuntime->bindControls(
        std::make_shared<const RuntimeControlState>(controls));
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto withKnob = knobRuntime->processTensor(ones);
    passed &= expect(withKnob.defined() && withKnob.size(1) == 2,
                     "Knob-conditioned graph must produce stereo output");
    auto zeros = torch::zeros({1, 2, 64}, torch::kFloat32);
    auto blockOnes = torch::ones({1, 2, 64}, torch::kFloat32);
    knobRuntime->reset();
    const auto isolatedZeros =
        knobRuntime->processIsolated(knobMerge, zeros);
    knobRuntime->reset();
    const auto isolatedOnes = knobRuntime->processIsolated(knobMerge, blockOnes);
    passed &= expect(isolatedZeros.defined() && isolatedOnes.defined(),
                     "Merge isolation must produce output");
    passed &= expect(
        std::abs(isolatedZeros[0][0][0].item<float>() - 1.5f) < 1.0e-4f &&
            std::abs(isolatedOnes[0][0][0].item<float>() - 2.5f) < 1.0e-4f,
        "isolated Merge must keep conditioning inputs instead of the probe");
  }
  analysisRequest.nodeId = knob;
  analysisRequest.liveInputSuitable = true;
  analysisRequest.liveInputChannels = 2;
  analysisRequest.liveInputSamples = 64;
  std::array<float, 128> fakeLive{};
  for (std::size_t index = 0; index < fakeLive.size(); ++index)
    fakeLive[index] = std::sin(static_cast<float>(index) * 0.11f);
  analysisRequest.liveInput = fakeLive.data();
  analysisRequest.view = openyourbox::graph::AnalysisView::oscilloscope;
  const auto knobOscilloscope = LiveGraphEngine::analyse(knobGraph, analysisRequest,
                                                         options);
  passed &= expect(
      knobOscilloscope.sourceMode == AnalysisSourceMode::probe &&
          knobOscilloscope.elementOnlySeries.size() >= 1 &&
          !knobOscilloscope.elementOnlySeries.front().y.empty() &&
          std::abs(knobOscilloscope.elementOnlySeries.front().y.front() - 1.5f) <
              1.0e-3f,
      "conditioning-source oscilloscope must ignore live audio capture");

  const auto restoredTree = knobGraph.toValueTree();
  openyourbox::graph::NodeGraph restoredKnob;
  passed &= expect(restoredKnob.restoreFromValueTree(restoredTree),
                   "Knob graph must serialize");
  const auto *restoredKnobNode = restoredKnob.findNode(knob);
  passed &= expect(restoredKnobNode != nullptr &&
                       std::abs(restoredKnobNode->conditioningValue - 1.5f) <
                           1.0e-4f,
                   "Knob conditioning value must survive ValueTree recall");

  openyourbox::graph::NodeGraph xyVolumeGraph;
  const auto xyVolumeInput = xyVolumeGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto xyPad = xyVolumeGraph.addNode(
      openyourbox::graph::NodeType::xyTrackpad, {0.0f, 80.0f});
  const auto xyConcat =
      xyVolumeGraph.addNode(openyourbox::graph::NodeType::merge, {180.0f, 80.0f});
  const auto xyMultiply =
      xyVolumeGraph.addNode(openyourbox::graph::NodeType::merge, {360.0f, 20.0f});
  const auto xyVolumeOutput = xyVolumeGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {540.0f, 20.0f});
  xyVolumeGraph.setProperty(xyConcat, "mode", 2);
  xyVolumeGraph.setProperty(xyMultiply, "mode", 1);
  xyVolumeGraph.setConditioningPad(xyPad, 0.5f, 2.0f);
  const auto *xyPadNode = xyVolumeGraph.findNode(xyPad);
  const auto *xyConcatNode = xyVolumeGraph.findNode(xyConcat);
  const auto *xyMultiplyNode = xyVolumeGraph.findNode(xyMultiply);
  passed &= expect(
      xyPadNode != nullptr && xyConcatNode != nullptr &&
          xyMultiplyNode != nullptr &&
          xyVolumeGraph
              .connect(xyPadNode->outputs[0].id, xyConcatNode->inputs[0].id)
              .accepted &&
          xyVolumeGraph
              .connect(xyPadNode->outputs[1].id, xyConcatNode->inputs[1].id)
              .accepted,
      "XY X and Y must concatenate");
  passed &= expect(
      xyVolumeGraph
          .connect(xyVolumeGraph.findNode(xyVolumeInput)->outputs.front().id,
                   xyMultiplyNode->inputs[0].id)
          .accepted &&
          xyVolumeGraph
              .connect(xyConcatNode->outputs.front().id,
                       xyMultiplyNode->inputs[1].id)
              .accepted &&
          xyVolumeGraph
              .connect(xyMultiplyNode->outputs.front().id,
                       xyVolumeGraph.findNode(xyVolumeOutput)->inputs.front().id)
              .accepted,
      "concatenated XY must multiply with audio");
  const auto xyVolumeCompiled = LiveGraphEngine::compile(xyVolumeGraph, options);
  passed &= expect(xyVolumeCompiled.succeeded(),
                   "XY concatenate-then-multiply graph must compile");
  LiveGraphCompileError xyVolumeError;
  const auto xyVolumeRuntime =
      LiveGraphEngine::prepare(xyVolumeCompiled.snapshot, xyVolumeError);
  passed &= expect(xyVolumeRuntime != nullptr,
                   "XY concatenate-then-multiply graph must prepare");
  if (xyVolumeRuntime != nullptr) {
    xyVolumeRuntime->bindControls(
        std::make_shared<const RuntimeControlState>(
            collectRuntimeControlState(xyVolumeGraph)));
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto scaled = xyVolumeRuntime->processTensor(ones);
    passed &= expect(scaled.defined() && scaled.size(1) == 2 &&
                         std::abs(scaled[0][0][0].item<float>() - 0.5f) <
                             1.0e-5f &&
                         std::abs(scaled[0][1][0].item<float>() - 2.0f) <
                             1.0e-5f,
                     "concatenated XY must scale stereo audio per channel");
  }

  passed &= expect(unusedXyAxisDoesNotModulateTcn(false, options),
                   "unused XY Y must not change TCN control when only X is cabled");
  passed &= expect(unusedXyAxisDoesNotModulateTcn(true, options),
                   "unused XY X must not change TCN control when only Y is cabled");
  passed &= expect(
      concatenatedXyControlsTcnAsVector(options),
      "XY concatenate into TCN control must keep independent X and Y");
  passed &= expect(
      filmControlHistoryIsCausalAcrossBlocks(options),
      "FiLM control history must match a long block when Knob ramps across buffers");

  {
    openyourbox::graph::NodeGraph freezeGraph;
    const auto freezeIn = freezeGraph.addNode(
        openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto freezeKnob = freezeGraph.addNode(
        openyourbox::graph::NodeType::knobInput, {0.0f, 80.0f});
    const auto freezeTcn =
        freezeGraph.addNode(openyourbox::graph::NodeType::tcn, {180.0f, 0.0f});
    const auto freezeOut = freezeGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
    freezeGraph.connect(freezeGraph.findNode(freezeIn)->outputs.front().id,
                        freezeGraph.findNode(freezeTcn)->inputs.front().id);
    freezeGraph.connect(freezeGraph.findNode(freezeKnob)->outputs.front().id,
                        freezeGraph.findNode(freezeTcn)->inputs[1].id);
    freezeGraph.connect(freezeGraph.findNode(freezeTcn)->outputs.front().id,
                        freezeGraph.findNode(freezeOut)->inputs.front().id);
    const auto freezeRequest = freezeGraph.createFreezeRequest({freezeTcn});
    passed &= expect(freezeRequest.has_value(),
                     "TCN with Knob Control must be freezable");
    if (freezeRequest.has_value()) {
      const auto payload =
          juce::JSON::parse(juce::String(freezeRequest->graphFragment));
      const auto *root = payload.getDynamicObject();
      const auto *compileOptions =
          root != nullptr
              ? root->getProperty("compile_options").getDynamicObject()
              : nullptr;
      passed &= expect(
          compileOptions != nullptr &&
              static_cast<bool>(compileOptions->getProperty("conditioning")) &&
              static_cast<int>(compileOptions->getProperty("cond_dim")) == 1,
          "freeze request must keep Control width so Knob/XY still steer");
    }
  }

  openyourbox::graph::NodeGraph rampGraph;
  const auto rampInput =
      rampGraph.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto rampKnob =
      rampGraph.addNode(openyourbox::graph::NodeType::knobInput, {0.0f, 80.0f});
  const auto rampActivation = rampGraph.addNode(
      openyourbox::graph::NodeType::activation, {180.0f, -40.0f});
  const auto rampXy =
      rampGraph.addNode(openyourbox::graph::NodeType::xyTrackpad, {0.0f, 160.0f});
  const auto rampMerge =
      rampGraph.addNode(openyourbox::graph::NodeType::merge, {360.0f, 20.0f});
  const auto rampOutput =
      rampGraph.addNode(openyourbox::graph::NodeType::audioOutput, {540.0f, 20.0f});
  rampGraph.setProperty(rampMerge, "mode", 1);
  rampGraph.setProperty(rampMerge, "inputs", 3);
  passed &= expect(
      rampGraph
          .connect(rampGraph.findNode(rampInput)->outputs.front().id,
                   rampGraph.findNode(rampActivation)->inputs.front().id)
          .accepted &&
          rampGraph
              .connect(rampGraph.findNode(rampActivation)->outputs.front().id,
                       rampGraph.findNode(rampMerge)->inputs[0].id)
              .accepted &&
          rampGraph
              .connect(rampGraph.findNode(rampKnob)->outputs.front().id,
                       rampGraph.findNode(rampMerge)->inputs[1].id)
              .accepted &&
          rampGraph
              .connect(rampGraph.findNode(rampXy)->outputs[0].id,
                       rampGraph.findNode(rampMerge)->inputs[2].id)
              .accepted &&
          rampGraph
              .connect(rampGraph.findNode(rampMerge)->outputs.front().id,
                       rampGraph.findNode(rampOutput)->inputs.front().id)
              .accepted,
      "ramp graph must connect Gain, Knob, and XY into multiply");
  LiveGraphCompileOptions rampOptions = options;
  rampOptions.sampleRate = 48000.0;
  rampOptions.controlRampSeconds = 0.001;
  const auto rampCompiled = LiveGraphEngine::compile(rampGraph, rampOptions);
  LiveGraphCompileError rampError;
  const auto rampRuntime =
      LiveGraphEngine::prepare(rampCompiled.snapshot, rampError);
  passed &= expect(rampCompiled.succeeded() && rampRuntime != nullptr,
                   "control-ramp graph must compile and prepare");
  if (rampRuntime != nullptr) {
    RuntimeControlState rampControls;
    rampControls.gainByNodeId[rampActivation] = 2.0f;
    rampControls.conditioningByNodeId[rampKnob] = {1.0f, 0.0f};
    rampControls.conditioningByNodeId[rampXy] = {1.0f, 0.0f};
    rampRuntime->bindControls(
        std::make_shared<const RuntimeControlState>(rampControls));
    auto ones = torch::ones({1, 2, 64}, torch::kFloat32);
    const auto ramped = rampRuntime->processTensor(ones);
    passed &= expect(
        ramped.defined() && ramped.size(2) == 64 &&
            ramped[0][0][0].item<float>() < ramped[0][0][32].item<float>() &&
            ramped[0][0][32].item<float>() < ramped[0][0][63].item<float>() &&
            ramped[0][0][63].item<float>() > 1.5f,
        "Gain, Knob, and XY must chase the new target across the block");
  }

  openyourbox::graph::NodeGraph growthGraph;
  const auto growthInput = growthGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto growthTcn = growthGraph.addNode(
      openyourbox::graph::NodeType::tcn, {180.0f, 0.0f});
  const auto growthOutput = growthGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  const auto *growthTcnNode = growthGraph.findNode(growthTcn);
  passed &= expect(
      growthTcnNode != nullptr && growthTcnNode->inputs.size() >= 2 &&
          growthTcnNode->inputs[1].signalKind ==
              openyourbox::graph::SignalKind::conditioning &&
          growthTcnNode->inputs[1].label ==
              openyourbox::graph::controlPinLabel,
      "TCN must expose a control conditioning pin");
  passed &= expect(openyourbox::graph::tcnLayerDilation(8, 0) == 1 &&
                       openyourbox::graph::tcnLayerDilation(8, 1) == 8 &&
                       openyourbox::graph::tcnLayerDilation(8, 2) == 64,
                   "dilation growth 8 must produce 1, 8, 64");
  growthGraph.setProperty(growthTcn, "dilation_growth", 8);
  growthGraph.setProperty(growthTcn, "residual", 1);
  growthGraph.setProperty(growthTcn, "activation", 4);
  passed &=
      expect(growthGraph
                 .connect(growthGraph.findNode(growthInput)->outputs.front().id,
                          growthTcnNode->inputs.front().id)
                 .accepted,
             "TCN audio pin must accept host input");
  passed &= expect(growthGraph.setProperty(growthTcn, "activation", 4),
                   "PReLU must be a selectable activation index");
  {
    int activation = -1;
    for (const auto &property : growthTcnNode->properties) {
      if (property.key == "activation")
        activation = property.value;
    }
    passed &= expect(activation == 4, "selecting PReLU must store index 4");
  }
  passed &= expect(
      growthGraph
          .connect(growthGraph.findNode(growthInput)->outputs.front().id,
                   growthTcnNode->inputs[1].id)
          .accepted,
      "TCN control pin must accept an audio signal of any width");
  {
    std::int32_t controlLink = 0;
    for (const auto &link : growthGraph.getLinks()) {
      if (link.destinationPinId == growthTcnNode->inputs[1].id)
        controlLink = link.id;
    }
    passed &= expect(controlLink != 0 && growthGraph.removeLink(controlLink),
                     "temporary audio-to-control cable must be removable");
  }
  const auto growthXy = growthGraph.addNode(
      openyourbox::graph::NodeType::xyTrackpad, {180.0f, 120.0f});
  passed &= expect(
      growthGraph
          .connect(growthGraph.findNode(growthXy)->outputs.front().id,
                   growthTcnNode->inputs[1].id)
          .accepted,
      "TCN control pin must accept XY conditioning");
  {
    std::int32_t xyControlLink = 0;
    for (const auto &link : growthGraph.getLinks()) {
      if (link.destinationPinId == growthTcnNode->inputs[1].id)
        xyControlLink = link.id;
    }
    const auto inserted = growthGraph.insertNodeOnLink(
        xyControlLink, openyourbox::graph::NodeType::merge, {180.0f, 80.0f});
    passed &= expect(inserted.has_value(),
                     "Merge must insert between XY and the control pin");
  }
  passed &= expect(
      !growthGraph
           .connect(growthGraph.findNode(growthXy)->outputs.front().id,
                    growthGraph.findNode(growthOutput)->inputs.front().id)
           .accepted,
      "audio input must reject a conditioning cable");
  passed &= expect(
      growthGraph
          .connect(growthGraph.findNode(growthTcn)->outputs.front().id,
                   growthGraph.findNode(growthOutput)->inputs.front().id)
          .accepted,
      "TCN output must connect to host output");
  const auto growthCompiled = LiveGraphEngine::compile(growthGraph, options);
  passed &= expect(growthCompiled.succeeded(),
                   "Control TCN with growth 8, residual, and PReLU must compile");

  openyourbox::graph::NodeGraph trainGraph;
  const auto trainIn =
      trainGraph.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto trainTcn =
      trainGraph.addNode(openyourbox::graph::NodeType::tcn, {180.0f, 0.0f});
  const auto trainOut = trainGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  const auto *trainTcnNode = trainGraph.findNode(trainTcn);
  trainGraph.connect(trainGraph.findNode(trainIn)->outputs.front().id,
                     trainTcnNode->inputs.front().id);
  trainGraph.connect(trainTcnNode->outputs.front().id,
                     trainGraph.findNode(trainOut)->inputs.front().id);
  const auto linksBefore = static_cast<int>(trainGraph.getLinks().size());
  const auto nodesBefore = static_cast<int>(trainGraph.getNodes().size());
  openyourbox::graph::TrainJobResult trainResult;
  trainResult.artifactPath = "/tmp/openyourbox-test-trained.pt";
  trainResult.inputChannels = 2;
  trainResult.outputChannels = 2;
  trainResult.acceptsConditioning = true;
  const auto absorbed = trainGraph.absorbArmedChain(trainResult);
  passed &= expect(absorbed.has_value(),
                   "train auto-load must replace the armed chain");
  passed &= expect(trainGraph.unfreeze(*absorbed) &&
                       static_cast<int>(trainGraph.getNodes().size()) ==
                           nodesBefore &&
                       static_cast<int>(trainGraph.getLinks().size()) ==
                           linksBefore,
                   "unfreeze must restore trained-chain nodes and cables");

  const auto weightsRoot = openyourbox::library::weightsDirectory();
  const auto samplesRoot = openyourbox::library::samplesDirectory();
  passed &= expect(weightsRoot.getFileName() == "Weights" &&
                       weightsRoot.getParentDirectory().getFileName() ==
                           "OpenYourBox",
                   "trained weights must live under OpenYourBox/Weights");
  passed &= expect(samplesRoot.getFileName() == "Samples" ||
                       samplesRoot.getFileName() == "TrainingLibrary",
                   "sample library must live under Samples (or the legacy folder)");

  const auto tempRoot =
      juce::File::getSpecialLocation(juce::File::tempDirectory)
          .getChildFile("OpenYourBoxLibraryTest")
          .getChildFile(juce::Uuid().toDashedString());
  tempRoot.createDirectory();
  openyourbox::library::TrainingLibrary library(tempRoot);
  juce::AudioBuffer<float> tone(1, 512);
  for (int i = 0; i < 512; ++i)
    tone.setSample(0, i, 0.1f);
  juce::WavAudioFormat wav;
  const auto xFile = tempRoot.getChildFile("x.wav");
  const auto yFile = tempRoot.getChildFile("y.wav");
  auto writeTone = [&](const juce::File &file) {
    auto *stream = file.createOutputStream().release();
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wav.createWriterFor(stream, 44100.0, 1, 32, {}, 0));
    return writer != nullptr && writer->writeFromAudioSampleBuffer(tone, 0, 512);
  };
  passed &= expect(writeTone(xFile) && writeTone(yFile),
                   "library test fixtures must write");
  juce::String importError;
  passed &= expect(library.importPair(xFile, yFile, importError).has_value(),
                   "library import must succeed for aligned files");
  passed &= expect(library.getSelectedCount() == 1,
                   "imported pairs are selected for train by default");
  juce::String mixed;
  passed &= expect(library.selectedSampleRatesMatch(mixed),
                   "single selected pair must pass sample-rate gate");
  library.selectNone();
  passed &= expect(!library.selectedSampleRatesMatch(mixed),
                   "empty selection must block Train");
  tempRoot.deleteRecursively();

  if (passed)
    std::cout << "OpenYourBox live graph engine tests passed\n";
  return passed ? 0 : 1;
}
