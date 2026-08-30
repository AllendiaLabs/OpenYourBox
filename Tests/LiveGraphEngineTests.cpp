#include "dsp/LiveGraphEngine.h"
#include "dsp/PqmfBank.h"
#include "dsp/RateConv.h"
#include "dsp/TorchScriptBlackBox.h"
#include "library/TrainingLibrary.h"
#include "library/UserBoxLibrary.h"
#include "library/UserDataPaths.h"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

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
 * @brief Imports a mono user-library RAVE group and processes live audio.
 * @param name Catalog display name.
 * @param options Live compile options (host width is forced to mono).
 * @return True when the named box is absent, or when import/compile/process pass.
 */
bool testUserLibraryRaveBox(
    const char *name, const openyourbox::dsp::LiveGraphCompileOptions &options) {
  using openyourbox::dsp::LiveGraphCompileError;
  using openyourbox::dsp::LiveGraphEngine;
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
  const auto inId = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto outId = graph.addNode(NodeType::audioOutput, {800.0f, 0.0f});
  const auto monoMessage = label + " host I/O must be mono";
  if (!expect(graph.setProperty(inId, "channels", 0) &&
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
  if (!expect(inNode != nullptr && outNode != nullptr && inputHub != nullptr &&
                  outputHub != nullptr,
              (label + " library box must expose I/O hubs").c_str()))
    return false;
  const auto inLink =
      graph.connect(inNode->outputs.front().id, inputHub->inputs.front().id);
  const auto outLink =
      graph.connect(outputHub->outputs.front().id, outNode->inputs.front().id);
  const auto inLinkMessage = "Audio In to " + label;
  if (!expect(inLink.accepted,
              inLink.accepted ? inLinkMessage.c_str() : inLink.message.c_str()))
    return false;
  const auto outLinkMessage = label + " to Audio Out";
  if (!expect(outLink.accepted,
              outLink.accepted ? outLinkMessage.c_str() : outLink.message.c_str()))
    return false;
  const auto expanded = graph.withInvisibleRepeatsMaterialized();
  const auto compiled = LiveGraphEngine::compile(expanded, options);
  const auto compileMessage = compiled.succeeded()
                                  ? label + " library graph must compile"
                                  : compiled.error.message;
  if (!expect(compiled.succeeded(), compileMessage.c_str()))
    return false;
  LiveGraphCompileError prepareError;
  const auto runtime = LiveGraphEngine::prepare(compiled.snapshot, prepareError);
  const auto prepareMessage = label + " library graph must prepare";
  if (!expect(runtime != nullptr, prepareMessage.c_str()))
    return false;
  bool processed = false;
  std::string processError;
  try {
    for (const auto block : {32, 64, 128, 256, 512}) {
      if (block > options.maximumBlockSize)
        continue;
      const auto output =
          runtime->processTensor(torch::randn({1, 2, block}, torch::kFloat32));
      if (!output.defined() || output.size(2) != block) {
        processError = "block " + std::to_string(block) + " returned time " +
                        std::to_string(output.defined() ? output.size(2) : -1);
        processed = false;
        break;
      }
      processed = true;
    }
  } catch (const std::exception &exception) {
    processError = exception.what();
    processed = false;
  }
  if (!processed && processError.empty() &&
      runtime->getLastProcessingFailureNodeId() != 0)
    processError = "muted at node " +
                   std::to_string(runtime->getLastProcessingFailureNodeId());
  const auto processMessage = processError.empty()
                                  ? label + " from library must process"
                                  : processError;
  return expect(processed, processMessage.c_str());
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
 * @class TestRaveDecodeKernel
 * @brief Distinguishes decode-from-latent (wide z) from encode-of-audio.
 */
class TestRaveDecodeKernel final : public openyourbox::dsp::FrozenBlackBoxKernel {
public:
  /** @brief Passes stereo audio through unchanged. */
  torch::Tensor forward(const torch::Tensor &input) override { return input; }

  /** @brief Returns the stereo audio as a 2-channel latent. */
  torch::Tensor encode(const torch::Tensor &input) override { return input; }

  /**
   * @brief Emits stereo audio whose value flags the incoming latent width.
   * @param latent Full-width or compact latent trajectory.
   */
  torch::Tensor decode(const torch::Tensor &latent) override {
    auto audio = torch::zeros({1, 2, latent.size(2)}, latent.options());
    audio.fill_(latent.size(1) >= 8 ? 0.25f : -0.5f);
    return audio;
  }

  /** @brief Advertises encode/decode so Gold latent pins compile. */
  bool hasEncodeDecode() const noexcept override { return true; }
};

/**
 * @class TestRaveDecodeFactory
 * @brief Stereo RAVE-style factory with an 8-channel latent (not host width).
 */
class TestRaveDecodeFactory final : public openyourbox::dsp::FrozenBlackBoxFactory {
public:
  /** @brief Returns the stereo test input width. */
  int getInputChannels() const noexcept override { return 2; }
  /** @brief Returns the stereo test output width. */
  int getOutputChannels() const noexcept override { return 2; }
  /** @brief Returns a single-sample receptive field. */
  std::uint64_t getReceptiveField() const noexcept override { return 1; }
  /** @brief Reports that the test kernel owns no trainable parameters. */
  std::uint64_t getParameterCount() const noexcept override { return 0; }
  /** @brief Reports exact digital-silence preservation. */
  bool preservesSilence() const noexcept override { return true; }
  /** @brief Advertises encode/decode beside forward. */
  bool hasEncodeDecode() const noexcept override { return true; }
  /** @brief Width that must not leak into Audio Output shape checks. */
  int getLatentChannels() const noexcept override { return 8; }
  /** @brief Creates one runtime-local encode/decode test kernel. */
  std::unique_ptr<openyourbox::dsp::FrozenBlackBoxKernel>
  createKernel() const override {
    return std::make_unique<TestRaveDecodeKernel>();
  }
};

/**
 * @struct PriorMixProbe
 * @brief Shared encode-call counter and last decode input for prior-mix tests.
 */
struct PriorMixProbe {
  /** @brief Number of encodeDistribution calls on prepared kernels. */
  std::atomic<int> encodeCalls{0};
  /** @brief Clone of the latent tensor passed to decode. */
  torch::Tensor lastDecodeInput;
};

/**
 * @class TestPriorMixKernel
 * @brief Deterministic RAVE stub: μ = audio mean, σ = 0, decode = z mean.
 */
class TestPriorMixKernel final : public openyourbox::dsp::FrozenBlackBoxKernel {
public:
  /**
   * @brief Binds a shared probe used by the factory after prepare.
   * @param probeToShare Encode counter and last decode input.
   */
  explicit TestPriorMixKernel(std::shared_ptr<PriorMixProbe> probeToShare)
      : probe(std::move(probeToShare)) {}

  torch::Tensor forward(const torch::Tensor &input) override { return input; }

  torch::Tensor encode(const torch::Tensor &input) override {
    return encodeDistribution(input).mean;
  }

  LatentDistribution encodeDistribution(const torch::Tensor &input) override {
    if (probe)
      probe->encodeCalls.fetch_add(1, std::memory_order_relaxed);
    LatentDistribution distribution;
    distribution.mean =
        torch::zeros({1, 8, input.size(2)}, input.options());
    if (input.defined() && input.numel() > 0)
      distribution.mean.fill_(input.mean().item<float>());
    distribution.std = torch::zeros_like(distribution.mean);
    return distribution;
  }

  torch::Tensor decode(const torch::Tensor &latent) override {
    if (probe)
      probe->lastDecodeInput = latent.clone();
    auto audio = torch::zeros({1, 2, latent.size(2)}, latent.options());
    if (latent.defined() && latent.numel() > 0)
      audio.fill_(latent.mean().item<float>());
    return audio;
  }

  bool hasEncodeDecode() const noexcept override { return true; }

private:
  /** @brief Shared observability for tests. */
  std::shared_ptr<PriorMixProbe> probe;
};

/**
 * @class TestPriorMixFactory
 * @brief Stereo RAVE factory with an 8-channel latent for prior-mix tests.
 */
class TestPriorMixFactory final : public openyourbox::dsp::FrozenBlackBoxFactory {
public:
  /**
   * @brief Binds the probe every created kernel will share.
   * @param probeToShare Encode counter and last decode input.
   */
  explicit TestPriorMixFactory(std::shared_ptr<PriorMixProbe> probeToShare)
      : probe(std::move(probeToShare)) {}

  int getInputChannels() const noexcept override { return 2; }
  int getOutputChannels() const noexcept override { return 2; }
  std::uint64_t getReceptiveField() const noexcept override { return 1; }
  std::uint64_t getParameterCount() const noexcept override { return 0; }
  bool preservesSilence() const noexcept override { return true; }
  bool hasEncodeDecode() const noexcept override { return true; }
  int getLatentChannels() const noexcept override { return 8; }
  std::unique_ptr<openyourbox::dsp::FrozenBlackBoxKernel>
  createKernel() const override {
    return std::make_unique<TestPriorMixKernel>(probe);
  }

private:
  /** @brief Shared observability for tests. */
  std::shared_ptr<PriorMixProbe> probe;
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

  {
    openyourbox::graph::NodeGraph meterGraph;
    const auto meterInput = meterGraph.addNode(
        openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto meterOutput = meterGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {180.0f, 0.0f});
    passed &= expect(
        meterGraph
            .connect(meterGraph.findNode(meterInput)->outputs.front().id,
                     meterGraph.findNode(meterOutput)->inputs.front().id)
            .accepted,
        "passthrough meter graph must connect Audio In to Audio Out");
    const auto meterCompiled = LiveGraphEngine::compile(meterGraph, options);
    const auto meterRuntime =
        LiveGraphEngine::prepare(meterCompiled.snapshot, error);
    passed &= expect(meterRuntime != nullptr,
                     "passthrough meter graph must prepare");
    if (meterRuntime != nullptr) {
      float rms = 1.0f;
      meterRuntime->processTensor(torch::zeros({1, 2, 64}, torch::kFloat32));
      passed &= expect(meterRuntime->getTapRms(meterInput, rms) && rms == 0.0f,
                       "silence must publish zero output RMS");
      auto ones = torch::ones({1, 2, 64}, torch::kFloat32);
      meterRuntime->processTensor(ones);
      passed &= expect(meterRuntime->getTapRms(meterInput, rms) &&
                           std::abs(rms - 1.0f) < 1.0e-5f,
                       "unity stereo must publish RMS 1 after collapsing dims");
      auto leftOnly = torch::zeros({1, 2, 64}, torch::kFloat32);
      leftOnly[0][0] = 1.0f;
      meterRuntime->processTensor(leftOnly);
      const auto expectedCollapsed = std::sqrt(0.5f);
      passed &= expect(
          meterRuntime->getTapRms(meterInput, rms) &&
              std::abs(rms - expectedCollapsed) < 1.0e-5f,
          "multi-channel tensors must collapse to a single RMS level");
      passed &= expect(meterRuntime->getTapRms(meterOutput, rms) &&
                           std::abs(rms - expectedCollapsed) < 1.0e-5f,
                       "Audio Out RMS must match the collapsed passthrough");
    }
  }

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

  openyourbox::graph::NodeGraph utilityGraph;
  const auto utilityInput = utilityGraph.addNode(
      openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
  const auto utilityNodeId = utilityGraph.addNode(
      openyourbox::graph::NodeType::merge, {180.0f, 0.0f});
  const auto utilityOutput = utilityGraph.addNode(
      openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
  const auto *utilityNode = utilityGraph.findNode(utilityNodeId);
  passed &= expect(utilityNode != nullptr && utilityNode->label == "Utility" &&
                       utilityNode->inputs.size() == 2,
                   "a new Utility element must default to two inputs");
  passed &= expect(utilityGraph.setProperty(utilityNodeId, "inputs", 1),
                   "Utility must accept Inputs >= 1");
  utilityNode = utilityGraph.findNode(utilityNodeId);
  passed &= expect(utilityNode != nullptr && utilityNode->inputs.size() == 1,
                   "Utility Inputs = 1 must leave a single input pin");
  utilityGraph.setProperty(utilityNodeId, "inputs", 0);
  utilityNode = utilityGraph.findNode(utilityNodeId);
  passed &= expect(utilityNode != nullptr && utilityNode->inputs.size() == 1,
                   "Utility Inputs below 1 must clamp to one pin");
  passed &= expect(
      utilityGraph
          .connect(utilityGraph.findNode(utilityInput)->outputs.front().id,
                   utilityGraph.findNode(utilityNodeId)->inputs.front().id)
          .accepted,
      "a one-input Utility must accept a connection");
  passed &= expect(
      utilityGraph
          .connect(utilityGraph.findNode(utilityNodeId)->outputs.front().id,
                   utilityGraph.findNode(utilityOutput)->inputs.front().id)
          .accepted,
      "a one-input Utility must connect to Audio Output");
  const auto utilityCompiled = LiveGraphEngine::compile(utilityGraph, options);
  const auto utilityRuntime =
      LiveGraphEngine::prepare(utilityCompiled.snapshot, error);
  passed &= expect(utilityCompiled.succeeded() && utilityRuntime != nullptr,
                   "a one-input Utility must compile as a passthrough");
  if (utilityRuntime != nullptr) {
    auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto passedThrough = utilityRuntime->processTensor(ones);
    passed &= expect(torch::equal(passedThrough, ones),
                     "a one-input Utility must pass its connected input through");
  }

  {
    using openyourbox::graph::MergeMode;
    using openyourbox::graph::NodeType;
    auto propertyValue = [](const openyourbox::graph::NodeGraph &document,
                            std::int32_t nodeId, const char *key, int fallback) {
      const auto *node = document.findNode(nodeId);
      if (node == nullptr)
        return fallback;
      for (const auto &property : node->properties) {
        if (property.key == key)
          return property.value;
      }
      return fallback;
    };
    auto wireTwoLinears = [](openyourbox::graph::NodeGraph &document, int leftFeatures,
                             int rightFeatures, int mode, int inputs = 2) {
      const auto audioIn = document.addNode(NodeType::audioInput, {0.0f, 0.0f});
      const auto left = document.addNode(NodeType::linear, {180.0f, -40.0f});
      const auto right = document.addNode(NodeType::linear, {180.0f, 80.0f});
      const auto merge = document.addNode(NodeType::merge, {360.0f, 20.0f});
      document.setProperty(left, "features", leftFeatures);
      document.setProperty(right, "features", rightFeatures);
      document.setProperty(merge, "mode", mode);
      document.setProperty(merge, "inputs", inputs);
      const auto *inNode = document.findNode(audioIn);
      const auto *leftNode = document.findNode(left);
      const auto *rightNode = document.findNode(right);
      document.connect(inNode->outputs.front().id, leftNode->inputs.front().id);
      document.connect(inNode->outputs.front().id, rightNode->inputs.front().id);
      return std::array<std::int32_t, 4>{audioIn, left, right, merge};
    };

    openyourbox::graph::NodeGraph addMismatch;
    const auto addIds = wireTwoLinears(addMismatch, 2, 3,
                                       static_cast<int>(MergeMode::add));
    const auto *addLeft = addMismatch.findNode(addIds[1]);
    const auto *addRight = addMismatch.findNode(addIds[2]);
    const auto *addMerge = addMismatch.findNode(addIds[3]);
    passed &= expect(
        addLeft != nullptr && addRight != nullptr && addMerge != nullptr &&
            addMismatch
                .connect(addLeft->outputs.front().id, addMerge->inputs[0].id)
                .accepted,
        "Utility add must accept the first 2ch input");
    addMerge = addMismatch.findNode(addIds[3]);
    const auto addSecond = addMismatch.connect(addRight->outputs.front().id,
                                               addMerge->inputs[1].id);
    passed &= expect(!addSecond.accepted &&
                         addSecond.message.find("cannot combine") !=
                             std::string::npos,
                     "Utility add must refuse 2ch vs 3ch");

    openyourbox::graph::NodeGraph multiplyMismatch;
    const auto mulIds = wireTwoLinears(multiplyMismatch, 2, 3,
                                       static_cast<int>(MergeMode::multiply));
    const auto *mulLeft = multiplyMismatch.findNode(mulIds[1]);
    const auto *mulRight = multiplyMismatch.findNode(mulIds[2]);
    const auto *mulMerge = multiplyMismatch.findNode(mulIds[3]);
    multiplyMismatch.connect(mulLeft->outputs.front().id,
                             mulMerge->inputs[0].id);
    mulMerge = multiplyMismatch.findNode(mulIds[3]);
    passed &= expect(
        !multiplyMismatch
             .connect(mulRight->outputs.front().id, mulMerge->inputs[1].id)
             .accepted,
        "Utility multiply must refuse 2ch vs 3ch");

    openyourbox::graph::NodeGraph broadcastGraph;
    const auto bcIds = wireTwoLinears(broadcastGraph, 2, 1,
                                      static_cast<int>(MergeMode::add));
    const auto *bcLeft = broadcastGraph.findNode(bcIds[1]);
    const auto *bcRight = broadcastGraph.findNode(bcIds[2]);
    const auto *bcMerge = broadcastGraph.findNode(bcIds[3]);
    passed &= expect(
        broadcastGraph.connect(bcLeft->outputs.front().id, bcMerge->inputs[0].id)
                .accepted &&
            broadcastGraph
                .connect(bcRight->outputs.front().id, bcMerge->inputs[1].id)
                .accepted,
        "Utility add must accept 2ch with a 1ch broadcast");

    openyourbox::graph::NodeGraph reshapeGraph;
    const auto rsIds = wireTwoLinears(reshapeGraph, 2, 2,
                                      static_cast<int>(MergeMode::add));
    const auto *rsLeft = reshapeGraph.findNode(rsIds[1]);
    const auto *rsRight = reshapeGraph.findNode(rsIds[2]);
    const auto *rsMerge = reshapeGraph.findNode(rsIds[3]);
    reshapeGraph.connect(rsLeft->outputs.front().id, rsMerge->inputs[0].id);
    reshapeGraph.connect(rsRight->outputs.front().id, rsMerge->inputs[1].id);
    passed &= expect(!reshapeGraph.setProperty(rsIds[2], "features", 3) &&
                         propertyValue(reshapeGraph, rsIds[2], "features", 0) ==
                             2,
                     "changing a Utility add input from 2ch to 3ch must roll back");

    openyourbox::graph::NodeGraph concatThenAdd;
    const auto catIds = wireTwoLinears(concatThenAdd, 2, 3,
                                       static_cast<int>(MergeMode::concatenate));
    const auto *catLeft = concatThenAdd.findNode(catIds[1]);
    const auto *catRight = concatThenAdd.findNode(catIds[2]);
    const auto *catMerge = concatThenAdd.findNode(catIds[3]);
    passed &= expect(
        concatThenAdd.connect(catLeft->outputs.front().id, catMerge->inputs[0].id)
                .accepted &&
            concatThenAdd
                .connect(catRight->outputs.front().id, catMerge->inputs[1].id)
                .accepted,
        "Utility concatenate must accept 2ch and 3ch");
    passed &= expect(
        !concatThenAdd.setProperty(catIds[3], "mode",
                                   static_cast<int>(MergeMode::add)) &&
            propertyValue(concatThenAdd, catIds[3], "mode", -1) ==
                static_cast<int>(MergeMode::concatenate),
        "switching concatenate 2ch+3ch to add must roll back");
    passed &= expect(
        !concatThenAdd.setProperty(catIds[3], "mode",
                                   static_cast<int>(MergeMode::multiply)) &&
            propertyValue(concatThenAdd, catIds[3], "mode", -1) ==
                static_cast<int>(MergeMode::concatenate),
        "switching concatenate 2ch+3ch to multiply must roll back");
  }

  auto utilityTree = utilityGraph.toValueTree();
  bool persistedUtilityType = false;
  for (const auto child : utilityTree) {
    if (child.hasType("Node") && child["type"].toString() == "utility")
      persistedUtilityType = true;
  }
  passed &= expect(persistedUtilityType,
                   "Utility must persist as type utility");
  for (int index = 0; index < utilityTree.getNumChildren(); ++index) {
    auto child = utilityTree.getChild(index);
    if (!child.hasType("Node") || child["type"].toString() != "utility")
      continue;
    child.setProperty("type", "merge", nullptr);
    child.setProperty("label", "Merge", nullptr);
    for (int propertyIndex = 0; propertyIndex < child.getNumChildren();
         ++propertyIndex) {
      auto property = child.getChild(propertyIndex);
      if (!property.hasType("Property") ||
          property["key"].toString() != "inputs")
        continue;
      property.setProperty("minimum", 2, nullptr);
      property.setProperty("value", 1, nullptr);
    }
  }
  openyourbox::graph::NodeGraph restoredUtility;
  passed &= expect(restoredUtility.restoreFromValueTree(utilityTree),
                   "legacy merge documents must load as Utility");
  const auto *restoredUtilityNode = restoredUtility.findNode(utilityNodeId);
  passed &= expect(
      restoredUtilityNode != nullptr && restoredUtilityNode->label == "Utility" &&
          restoredUtility.setProperty(utilityNodeId, "inputs", 1) &&
          restoredUtility.findNode(utilityNodeId)->inputs.size() == 1,
      "loaded Merge nodes must rename to Utility and allow Inputs >= 1");

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
      growthGraph
          .connect(growthGraph.findNode(growthXy)->outputs.front().id,
                   growthGraph.findNode(growthOutput)->inputs.front().id)
          .accepted == false,
      "1-channel XY must not connect to a 2-channel host output");
  passed &= expect(
      growthGraph
          .connect(growthGraph.findNode(growthTcn)->outputs.front().id,
                   growthGraph.findNode(growthOutput)->inputs.front().id)
          .accepted,
      "TCN output must connect to host output");
  const auto growthCompiled = LiveGraphEngine::compile(growthGraph, options);
  passed &= expect(growthCompiled.succeeded(),
                   "Control TCN with growth 8, residual, and PReLU must compile");

  {
    openyourbox::graph::NodeGraph groupedGraph;
    const auto groupedInput = groupedGraph.addNode(
        openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto first = groupedGraph.addNode(
        openyourbox::graph::NodeType::convolution, {140.0f, 0.0f});
    const auto second = groupedGraph.addNode(
        openyourbox::graph::NodeType::convolution, {280.0f, 0.0f});
    const auto groupedOutput = groupedGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {420.0f, 0.0f});
    groupedGraph.connect(groupedGraph.findNode(groupedInput)->outputs.front().id,
                         groupedGraph.findNode(first)->inputs.front().id);
    groupedGraph.connect(groupedGraph.findNode(first)->outputs.front().id,
                         groupedGraph.findNode(second)->inputs.front().id);
    groupedGraph.connect(groupedGraph.findNode(second)->outputs.front().id,
                         groupedGraph.findNode(groupedOutput)->inputs.front().id);
    const auto grouped = groupedGraph.createGroup({first, second});
    passed &= expect(grouped.accepted &&
                         groupedGraph.setGroupRepeats(grouped.groupId, 2).accepted &&
                         groupedGraph.groupRepeatStatus(grouped.groupId).active,
                     "explicit group interface activates compatible repeats");
    const auto prepared = groupedGraph.withInvisibleRepeatsMaterialized();
    int convolutionCount = 0;
    bool hasBoundaryHub = false;
    for (const auto &node : prepared.getNodes()) {
      if (node.type == openyourbox::graph::NodeType::convolution)
        ++convolutionCount;
      hasBoundaryHub =
          hasBoundaryHub || openyourbox::graph::isGroupBoundaryType(node.type);
    }
    passed &= expect(convolutionCount == 4 && !hasBoundaryHub,
                     "runtime graph unrolls repeats and removes boundary hubs");
    passed &= expect(LiveGraphEngine::compile(prepared, options).succeeded(),
                     "flattened explicit group compiles in the live engine");
    const auto trainRequest = groupedGraph.createTrainRequest();
    passed &= expect(
        trainRequest.has_value() && trainRequest->armedNodeIds.size() == 4 &&
            trainRequest->graphFragment.find("group_input") ==
                std::string::npos &&
            trainRequest->graphFragment.find("group_output") ==
                std::string::npos,
        "training materializes active repeats without editor-only hubs");
    const auto freezeRequest =
        groupedGraph.createFreezeRequest({first, second});
    passed &= expect(
        freezeRequest.has_value() &&
            freezeRequest->graphFragment.find("group_input") ==
                std::string::npos &&
            freezeRequest->graphFragment.find("group_output") ==
                std::string::npos,
        "freeze requests flatten editor-only boundary hubs");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph modesGraph;
    const auto audioIn =
        modesGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto utility1 =
        modesGraph.addNode(NodeType::merge, {120.0f, 0.0f});
    const auto bodyAct =
        modesGraph.addNode(NodeType::activation, {240.0f, 40.0f});
    const auto bodyConv =
        modesGraph.addNode(NodeType::convolution, {360.0f, 40.0f});
    const auto utility2 =
        modesGraph.addNode(NodeType::merge, {480.0f, 0.0f});
    const auto audioOut =
        modesGraph.addNode(NodeType::audioOutput, {620.0f, 0.0f});
    modesGraph.setProperty(utility1, "inputs", 1);
    modesGraph.setProperty(utility2, "inputs", 2);
    modesGraph.connect(modesGraph.findNode(bodyAct)->outputs.front().id,
                       modesGraph.findNode(bodyConv)->inputs.front().id);
    const auto body = modesGraph.createGroup({bodyAct, bodyConv});
    std::int32_t bodyIn = 0;
    std::int32_t bodyOut = 0;
    if (const auto *group = modesGraph.findGroup(body.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = modesGraph.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          bodyIn = memberId;
        if (member->type == NodeType::groupOutput)
          bodyOut = memberId;
      }
    }
    // Group the residual first, then wire through stack hubs (UI order).
    modesGraph.connect(modesGraph.findNode(audioIn)->outputs.front().id,
                       modesGraph.findNode(utility1)->inputs.front().id);
    modesGraph.connect(modesGraph.findNode(utility1)->outputs.front().id,
                       modesGraph.findNode(bodyIn)->inputs.front().id);
    modesGraph.connect(modesGraph.findNode(bodyOut)->outputs.front().id,
                       modesGraph.findNode(utility2)->inputs.front().id);
    modesGraph.connect(modesGraph.findNode(utility1)->outputs.front().id,
                       modesGraph.findNode(utility2)->inputs[1].id);
    modesGraph.connect(modesGraph.findNode(utility2)->outputs.front().id,
                       modesGraph.findNode(audioOut)->inputs.front().id);
    const auto stack =
        modesGraph.createGroup({utility1, body.groupId, utility2});
    passed &= expect(stack.accepted, "mode residual stack groups");
    for (int mode = 0; mode <= 2; ++mode) {
      passed &= expect(modesGraph.setProperty(utility1, "mode", mode),
                       "fork utility mode can be set");
      const auto prepared = modesGraph.withInvisibleRepeatsMaterialized();
      const auto compiled = LiveGraphEngine::compile(prepared, options);
      if (!compiled.succeeded())
        std::cerr << "fork mode " << mode
                  << " failed: " << compiled.error.message << '\n';
      passed &= expect(compiled.succeeded(),
                       "fork utility residual compiles in every mode");
    }

    // Prefer forking from Group Input (no Utility fork at all).
    openyourbox::graph::NodeGraph hubFork;
    const auto audioIn2 =
        hubFork.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto bodyAct2 =
        hubFork.addNode(NodeType::activation, {200.0f, 40.0f});
    const auto bodyConv2 =
        hubFork.addNode(NodeType::convolution, {320.0f, 40.0f});
    const auto join2 = hubFork.addNode(NodeType::merge, {460.0f, 0.0f});
    const auto audioOut2 =
        hubFork.addNode(NodeType::audioOutput, {600.0f, 0.0f});
    hubFork.setProperty(join2, "inputs", 2);
    hubFork.connect(hubFork.findNode(bodyAct2)->outputs.front().id,
                    hubFork.findNode(bodyConv2)->inputs.front().id);
    const auto body2 = hubFork.createGroup({bodyAct2, bodyConv2});
    std::int32_t nestedIn = 0;
    std::int32_t nestedOut = 0;
    if (const auto *group = hubFork.findGroup(body2.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = hubFork.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          nestedIn = memberId;
        if (member->type == NodeType::groupOutput)
          nestedOut = memberId;
      }
    }
    hubFork.connect(hubFork.findNode(audioIn2)->outputs.front().id,
                    hubFork.findNode(nestedIn)->inputs.front().id);
    hubFork.connect(hubFork.findNode(nestedOut)->outputs.front().id,
                    hubFork.findNode(join2)->inputs.front().id);
    hubFork.connect(hubFork.findNode(audioIn2)->outputs.front().id,
                    hubFork.findNode(join2)->inputs[1].id);
    hubFork.connect(hubFork.findNode(join2)->outputs.front().id,
                    hubFork.findNode(audioOut2)->inputs.front().id);
    const auto stack2 = hubFork.createGroup({body2.groupId, join2});
    passed &= expect(stack2.accepted, "hub-fork residual stack groups");
    const auto hubPrepared = hubFork.withInvisibleRepeatsMaterialized();
    const auto hubCompiled = LiveGraphEngine::compile(hubPrepared, options);
    if (!hubCompiled.succeeded())
      std::cerr << "hub-fork residual failed: " << hubCompiled.error.message
                << '\n';
    passed &= expect(hubCompiled.succeeded(),
                     "Group Input fan-out residual compiles");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph residualGraph;
    const auto audioIn =
        residualGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto utility1 =
        residualGraph.addNode(NodeType::merge, {120.0f, 0.0f});
    const auto bodyAct =
        residualGraph.addNode(NodeType::activation, {240.0f, 40.0f});
    const auto bodyConv =
        residualGraph.addNode(NodeType::convolution, {360.0f, 40.0f});
    const auto utility2 =
        residualGraph.addNode(NodeType::merge, {480.0f, 0.0f});
    const auto audioOut =
        residualGraph.addNode(NodeType::audioOutput, {620.0f, 0.0f});
    residualGraph.setProperty(utility1, "inputs", 1);
    residualGraph.setProperty(utility2, "inputs", 2);
    residualGraph.connect(residualGraph.findNode(bodyAct)->outputs.front().id,
                          residualGraph.findNode(bodyConv)->inputs.front().id);
    const auto body = residualGraph.createGroup({bodyAct, bodyConv});
    passed &= expect(body.accepted, "nested residual body groups");
    std::int32_t bodyIn = 0;
    std::int32_t bodyOut = 0;
    if (const auto *group = residualGraph.findGroup(body.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = residualGraph.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          bodyIn = memberId;
        if (member->type == NodeType::groupOutput)
          bodyOut = memberId;
      }
    }
    residualGraph.connect(residualGraph.findNode(audioIn)->outputs.front().id,
                          residualGraph.findNode(utility1)->inputs.front().id);
    const auto bodyFeed =
        residualGraph.connect(residualGraph.findNode(utility1)->outputs.front().id,
                              residualGraph.findNode(bodyIn)->inputs.front().id);
    const auto bodyJoin =
        residualGraph.connect(residualGraph.findNode(bodyOut)->outputs.front().id,
                              residualGraph.findNode(utility2)->inputs.front().id);
    const auto skip =
        residualGraph.connect(residualGraph.findNode(utility1)->outputs.front().id,
                              residualGraph.findNode(utility2)->inputs[1].id);
    passed &= expect(bodyFeed.accepted && bodyJoin.accepted && skip.accepted,
                     "utility residual fork/join connects");
    if (!skip.accepted)
      std::cerr << "skip refused: " << skip.message << '\n';
    residualGraph.connect(residualGraph.findNode(utility2)->outputs.front().id,
                          residualGraph.findNode(audioOut)->inputs.front().id);
    const auto stack =
        residualGraph.createGroup({utility1, body.groupId, utility2});
    passed &= expect(stack.accepted, "utility residual stack groups");
    passed &= expect(residualGraph.setGroupRepeats(stack.groupId, 2).accepted &&
                         residualGraph.groupRepeatStatus(stack.groupId).active,
                     "utility residual stack activates N=2");
    const auto prepared = residualGraph.withInvisibleRepeatsMaterialized();
    bool hasBoundary = false;
    for (const auto &node : prepared.getNodes())
      hasBoundary =
          hasBoundary || openyourbox::graph::isGroupBoundaryType(node.type);
    const auto residualCompiled = LiveGraphEngine::compile(prepared, options);
    passed &= expect(!hasBoundary && residualCompiled.succeeded(),
                     "nested residual stack compiles without a directed cycle");

    openyourbox::graph::NodeGraph nestedRepeats;
    const auto nestedIn =
        nestedRepeats.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto nestedAct =
        nestedRepeats.addNode(NodeType::activation, {200.0f, 40.0f});
    const auto nestedConv =
        nestedRepeats.addNode(NodeType::convolution, {320.0f, 40.0f});
    const auto nestedJoin =
        nestedRepeats.addNode(NodeType::merge, {460.0f, 0.0f});
    const auto nestedOut =
        nestedRepeats.addNode(NodeType::audioOutput, {620.0f, 0.0f});
    nestedRepeats.setProperty(nestedJoin, "inputs", 2);
    nestedRepeats.connect(nestedRepeats.findNode(nestedAct)->outputs.front().id,
                         nestedRepeats.findNode(nestedConv)->inputs.front().id);
    const auto nestedBody = nestedRepeats.createGroup({nestedAct, nestedConv});
    passed &= expect(nestedBody.accepted, "nested-repeat residual body groups");
    std::int32_t nestedBodyIn = 0;
    std::int32_t nestedBodyOut = 0;
    if (const auto *group = nestedRepeats.findGroup(nestedBody.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = nestedRepeats.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          nestedBodyIn = memberId;
        if (member->type == NodeType::groupOutput)
          nestedBodyOut = memberId;
      }
    }
    const auto *nestedBodyInNode = nestedRepeats.findNode(nestedBodyIn);
    const auto *nestedBodyOutNode = nestedRepeats.findNode(nestedBodyOut);
    const auto *nestedActNode = nestedRepeats.findNode(nestedAct);
    const auto *nestedConvNode = nestedRepeats.findNode(nestedConv);
    const auto *nestedJoinNode = nestedRepeats.findNode(nestedJoin);
    passed &= expect(
        nestedBodyInNode != nullptr && nestedBodyOutNode != nullptr &&
            nestedActNode != nullptr && nestedConvNode != nullptr &&
            nestedJoinNode != nullptr &&
            nestedRepeats
                .connect(nestedBodyInNode->outputs.front().id,
                         nestedActNode->inputs.front().id)
                .accepted &&
            nestedRepeats
                .connect(nestedConvNode->outputs.front().id,
                         nestedBodyOutNode->inputs.front().id)
                .accepted &&
            nestedRepeats
                .connect(nestedBodyOutNode->outputs.front().id,
                         nestedJoinNode->inputs.front().id)
                .accepted,
        "nested-repeat layer through-path and join");
    const auto nestedStack =
        nestedRepeats.createGroup({nestedBody.groupId, nestedJoin});
    std::int32_t nestedStackIn = 0;
    std::int32_t nestedStackOut = 0;
    if (const auto *group = nestedRepeats.findGroup(nestedStack.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = nestedRepeats.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          nestedStackIn = memberId;
        if (member->type == NodeType::groupOutput)
          nestedStackOut = memberId;
      }
    }
    const auto *nestedStackInNode = nestedRepeats.findNode(nestedStackIn);
    const auto *nestedStackOutNode = nestedRepeats.findNode(nestedStackOut);
    const auto *stackBodyInNode = nestedRepeats.findNode(nestedBodyIn);
    const auto *stackJoinNode = nestedRepeats.findNode(nestedJoin);
    passed &= expect(
        nestedStack.accepted && nestedStackInNode != nullptr &&
            nestedStackOutNode != nullptr && stackBodyInNode != nullptr &&
            stackJoinNode != nullptr && stackJoinNode->inputs.size() >= 2 &&
            nestedRepeats
                .connect(nestedStackInNode->outputs.front().id,
                         stackBodyInNode->inputs.front().id)
                .accepted &&
            nestedRepeats
                .connect(nestedStackInNode->outputs.front().id,
                         stackJoinNode->inputs[1].id)
                .accepted &&
            nestedRepeats
                .connect(stackJoinNode->outputs.front().id,
                         nestedStackOutNode->inputs.front().id)
                .accepted &&
            nestedRepeats
                .connect(nestedRepeats.findNode(nestedIn)->outputs.front().id,
                         nestedStackInNode->inputs.front().id)
                .accepted &&
            nestedRepeats
                .connect(nestedStackOutNode->outputs.front().id,
                         nestedRepeats.findNode(nestedOut)->inputs.front().id)
                .accepted,
        "nested-repeat stack Group Input fans out into layer and skip");
    passed &= expect(
        nestedRepeats.setGroupRepeats(nestedBody.groupId, 2).accepted &&
            nestedRepeats.groupRepeatStatus(nestedBody.groupId).active &&
            nestedRepeats.setGroupRepeats(nestedStack.groupId, 3).accepted &&
            nestedRepeats.groupRepeatStatus(nestedStack.groupId).active,
        "ResidualLayer N=2 inside ResidualStack N=3 activates");
    const auto nestedPrepared = nestedRepeats.withInvisibleRepeatsMaterialized();
    int nestedActs = 0;
    int nestedConvs = 0;
    int nestedJoins = 0;
    bool nestedHasBoundary = false;
    for (const auto &node : nestedPrepared.getNodes()) {
      if (node.type == NodeType::activation)
        ++nestedActs;
      else if (node.type == NodeType::convolution)
        ++nestedConvs;
      else if (node.type == NodeType::merge)
        ++nestedJoins;
      nestedHasBoundary =
          nestedHasBoundary ||
          openyourbox::graph::isGroupBoundaryType(node.type);
    }
    const auto nestedCompiled =
        LiveGraphEngine::compile(nestedPrepared, options);
    if (!nestedCompiled.succeeded())
      std::cerr << "nested residual repeats failed: "
                << nestedCompiled.error.message << '\n';
    passed &= expect(!nestedHasBoundary && nestedActs == 6 && nestedConvs == 6 &&
                         nestedJoins == 3 && nestedCompiled.succeeded(),
                     "nested residual repeats compile without a directed cycle");

    openyourbox::graph::NodeGraph sameUtility;
    const auto sameIn =
        sameUtility.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto sameUtil =
        sameUtility.addNode(NodeType::merge, {120.0f, 0.0f});
    const auto sameAct =
        sameUtility.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto sameConv =
        sameUtility.addNode(NodeType::convolution, {360.0f, 0.0f});
    const auto sameOut =
        sameUtility.addNode(NodeType::audioOutput, {520.0f, 0.0f});
    sameUtility.setProperty(sameUtil, "inputs", 2);
    sameUtility.connect(sameUtility.findNode(sameAct)->outputs.front().id,
                        sameUtility.findNode(sameConv)->inputs.front().id);
    const auto bodyGroup = sameUtility.createGroup({sameAct, sameConv});
    passed &= expect(bodyGroup.accepted, "same-utility body groups");
    std::int32_t sameBodyIn = 0;
    std::int32_t sameBodyOut = 0;
    if (const auto *group = sameUtility.findGroup(bodyGroup.groupId)) {
      for (const auto memberId : group->memberIds) {
        const auto *member = sameUtility.findNode(memberId);
        if (member == nullptr)
          continue;
        if (member->type == NodeType::groupInput)
          sameBodyIn = memberId;
        if (member->type == NodeType::groupOutput)
          sameBodyOut = memberId;
      }
    }
    sameUtility.connect(sameUtility.findNode(sameIn)->outputs.front().id,
                        sameUtility.findNode(sameUtil)->inputs.front().id);
    sameUtility.connect(sameUtility.findNode(sameUtil)->outputs.front().id,
                        sameUtility.findNode(sameBodyIn)->inputs.front().id);
    sameUtility.connect(sameUtility.findNode(sameBodyIn)->outputs.front().id,
                        sameUtility.findNode(sameAct)->inputs.front().id);
    sameUtility.connect(sameUtility.findNode(sameConv)->outputs.front().id,
                        sameUtility.findNode(sameBodyOut)->inputs.front().id);
    const auto cyclic = sameUtility.connect(
        sameUtility.findNode(sameBodyOut)->outputs.front().id,
        sameUtility.findNode(sameUtil)->inputs[1].id);
    passed &= expect(!cyclic.accepted,
                     "same Utility cannot be both residual fork and join");
    sameUtility.connect(sameUtility.findNode(sameUtil)->outputs.front().id,
                        sameUtility.findNode(sameOut)->inputs.front().id);
  }

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

  {
    using openyourbox::graph::NodeGraph;
    using openyourbox::graph::NodeType;
    using openyourbox::graph::NodeState;
    NodeGraph raveAbsorb;
    const auto inId =
        raveAbsorb.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        raveAbsorb.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto conv =
        raveAbsorb.addNode(NodeType::convolution, {240.0f, 0.0f});
    const auto math =
        raveAbsorb.addNode(NodeType::mathExpression, {360.0f, 0.0f});
    const auto merge =
        raveAbsorb.addNode(NodeType::merge, {480.0f, 0.0f});
    const auto synth =
        raveAbsorb.addNode(NodeType::pqmfSynthesis, {600.0f, 0.0f});
    const auto outId =
        raveAbsorb.addNode(NodeType::audioOutput, {720.0f, 0.0f});
    raveAbsorb.setProperty(analysis, "n_band", 2);
    raveAbsorb.setProperty(synth, "n_band", 2);
    raveAbsorb.setProperty(conv, "channels", 4);
    const auto *inNode = raveAbsorb.findNode(inId);
    const auto *analysisNode = raveAbsorb.findNode(analysis);
    const auto *convNode = raveAbsorb.findNode(conv);
    const auto *mathNode = raveAbsorb.findNode(math);
    const auto *mergeNode = raveAbsorb.findNode(merge);
    const auto *synthNode = raveAbsorb.findNode(synth);
    const auto *outNode = raveAbsorb.findNode(outId);
    passed &= expect(
        inNode != nullptr && analysisNode != nullptr && convNode != nullptr &&
            mathNode != nullptr && mergeNode != nullptr &&
            synthNode != nullptr && outNode != nullptr &&
            raveAbsorb
                .connect(inNode->outputs.front().id,
                         analysisNode->inputs.front().id)
                .accepted &&
            raveAbsorb
                .connect(analysisNode->outputs.front().id,
                         convNode->inputs.front().id)
                .accepted &&
            raveAbsorb
                .connect(convNode->outputs.front().id,
                         mathNode->inputs.front().id)
                .accepted &&
            raveAbsorb
                .connect(convNode->outputs.front().id, mergeNode->inputs[0].id)
                .accepted &&
            raveAbsorb
                .connect(mathNode->outputs.front().id, mergeNode->inputs[1].id)
                .accepted &&
            raveAbsorb
                .connect(mergeNode->outputs.front().id,
                         synthNode->inputs.front().id)
                .accepted &&
            raveAbsorb
                .connect(synthNode->outputs.front().id,
                         outNode->inputs.front().id)
                .accepted,
        "RAVE absorb fixture must wire");
    const auto raveNodesBefore =
        static_cast<int>(raveAbsorb.getNodes().size());
    const auto raveLinksBefore =
        static_cast<int>(raveAbsorb.getLinks().size());
    openyourbox::graph::TrainJobResult raveResult;
    raveResult.artifactPath = "/tmp/openyourbox-test-trained-rave.pt";
    raveResult.hasEncodeDecode = true;
    const auto raveGold = raveAbsorb.absorbArmedChain(raveResult);
    int leftoverPqmf = 0;
    int leftoverMath = 0;
    int leftoverMerge = 0;
    int goldCount = 0;
    bool goldWiredIn = false;
    bool goldWiredOut = false;
    if (raveGold.has_value()) {
      const auto *gold = raveAbsorb.findNode(*raveGold);
      if (gold != nullptr && gold->state == NodeState::frozenGold) {
        ++goldCount;
        const auto *hostIn = raveAbsorb.findNode(inId);
        const auto *hostOut = raveAbsorb.findNode(outId);
        for (const auto &link : raveAbsorb.getLinks()) {
          if (hostIn != nullptr &&
              link.sourcePinId == hostIn->outputs.front().id &&
              link.destinationPinId == gold->inputs.front().id)
            goldWiredIn = true;
          if (hostOut != nullptr &&
              link.sourcePinId == gold->outputs.front().id &&
              link.destinationPinId == hostOut->inputs.front().id)
            goldWiredOut = true;
        }
      }
    }
    for (const auto &node : raveAbsorb.getNodes()) {
      if (node.type == NodeType::pqmfAnalysis ||
          node.type == NodeType::pqmfSynthesis)
        ++leftoverPqmf;
      else if (node.type == NodeType::mathExpression)
        ++leftoverMath;
      else if (node.type == NodeType::merge)
        ++leftoverMerge;
    }
    passed &= expect(raveGold.has_value() && goldCount == 1 && goldWiredIn &&
                         goldWiredOut && leftoverPqmf == 0 && leftoverMath == 0 &&
                         leftoverMerge == 0,
                     "RAVE absorb must replace the full chain with one wired Gold");
    passed &= expect(
        raveGold.has_value() && raveAbsorb.unfreeze(*raveGold) &&
            static_cast<int>(raveAbsorb.getNodes().size()) == raveNodesBefore &&
            static_cast<int>(raveAbsorb.getLinks().size()) == raveLinksBefore,
        "RAVE unfreeze must restore PQMF, math, and utility with cables");
  }

  {
    using openyourbox::graph::NodeGraph;
    using openyourbox::graph::NodeType;
    NodeGraph grouped;
    const auto inId = grouped.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        grouped.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto conv =
        grouped.addNode(NodeType::convolution, {240.0f, 0.0f});
    const auto synth =
        grouped.addNode(NodeType::pqmfSynthesis, {360.0f, 0.0f});
    const auto outId =
        grouped.addNode(NodeType::audioOutput, {480.0f, 0.0f});
    grouped.setProperty(analysis, "n_band", 2);
    grouped.setProperty(synth, "n_band", 2);
    grouped.setProperty(conv, "channels", 4);
    passed &= expect(
        grouped
            .connect(grouped.findNode(inId)->outputs.front().id,
                     grouped.findNode(analysis)->inputs.front().id)
            .accepted &&
            grouped
                .connect(grouped.findNode(analysis)->outputs.front().id,
                         grouped.findNode(conv)->inputs.front().id)
                .accepted &&
            grouped
                .connect(grouped.findNode(conv)->outputs.front().id,
                         grouped.findNode(synth)->inputs.front().id)
                .accepted &&
            grouped
                .connect(grouped.findNode(synth)->outputs.front().id,
                         grouped.findNode(outId)->inputs.front().id)
                .accepted,
        "grouped RAVE absorb fixture must wire");
    const auto groupedBox =
        grouped.createGroup({analysis, conv, synth});
    passed &= expect(groupedBox.accepted, "RAVE processing must group");
    openyourbox::graph::TrainJobResult groupedResult;
    groupedResult.artifactPath = "/tmp/openyourbox-test-trained-rave-group.pt";
    groupedResult.hasEncodeDecode = true;
    const auto groupedGold = grouped.absorbArmedChain(groupedResult);
    int leftoverPqmf = 0;
    bool goldWiredIn = false;
    bool goldWiredOut = false;
    if (groupedGold.has_value()) {
      const auto *gold = grouped.findNode(*groupedGold);
      const auto *hostIn = grouped.findNode(inId);
      const auto *hostOut = grouped.findNode(outId);
      if (gold != nullptr &&
          gold->state == openyourbox::graph::NodeState::frozenGold &&
          hostIn != nullptr && hostOut != nullptr) {
        for (const auto &link : grouped.getLinks()) {
          if (link.sourcePinId == hostIn->outputs.front().id &&
              link.destinationPinId == gold->inputs.front().id)
            goldWiredIn = true;
          if (link.sourcePinId == gold->outputs.front().id &&
              link.destinationPinId == hostOut->inputs.front().id)
            goldWiredOut = true;
        }
      }
    }
    for (const auto &node : grouped.getNodes()) {
      if (node.type == NodeType::pqmfAnalysis ||
          node.type == NodeType::pqmfSynthesis)
        ++leftoverPqmf;
    }
    passed &= expect(groupedGold.has_value() && goldWiredIn && goldWiredOut &&
                         leftoverPqmf == 0 && grouped.getGroups().empty(),
                     "grouped RAVE absorb must replace the group with wired Gold");
  }

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
  juce::String mixedRateMessage;
  passed &= expect(library.selectedSampleRatesMatch(mixedRateMessage),
                   "single selected pair must pass sample-rate gate");

  constexpr int clipSamples = 100000;
  juce::AudioBuffer<float> longTone(1, clipSamples);
  for (int i = 0; i < clipSamples; ++i)
    longTone.setSample(0, i, 0.05f);
  const auto clipFile = tempRoot.getChildFile("clip.wav");
  auto *clipStream = clipFile.createOutputStream().release();
  std::unique_ptr<juce::AudioFormatWriter> clipWriter(
      wav.createWriterFor(clipStream, 44100.0, 1, 32, {}, 0));
  passed &= expect(clipWriter != nullptr && clipWriter->writeFromAudioSampleBuffer(
                                                longTone, 0, clipSamples),
                   "library clip fixture must write");
  clipWriter.reset();
  juce::String clipError;
  const auto importedClip = library.importClip(clipFile, clipError);
  passed &= expect(importedClip.has_value(), "library clip import must succeed");
  passed &= expect(importedClip.has_value() &&
                       std::abs(importedClip->durationSeconds -
                                static_cast<double>(clipSamples) / 44100.0) <
                           1.0e-6,
                   "imported clip duration must match the source file length");
  if (importedClip.has_value()) {
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> stored(
        formats.createReaderFor(juce::File(importedClip->xPath)));
    passed &= expect(stored != nullptr && stored->lengthInSamples == clipSamples,
                     "stored clip WAV must keep every source sample");
  }

  library.selectNone();
  passed &= expect(!library.selectedSampleRatesMatch(mixedRateMessage),
                   "empty selection must block Train");
  tempRoot.deleteRecursively();

  {
    openyourbox::dsp::PqmfBank bank(4);
    auto audio = torch::randn({1, 1, 2048});
    const auto bands = bank.analyse(audio);
    const auto reconstructed = bank.synthesise(bands, 1);
    passed &= expect(bands.size(1) == 4, "PQMF analysis must emit nBand channels");
    passed &= expect(reconstructed.size(1) == 1 && reconstructed.size(2) >= 1,
                     "PQMF synthesis must return host-rate audio");
    passed &= expect(bank.getCausalDelaySamples() > 0,
                     "PQMF must report a positive causal delay");
    float bestRelative = 1.0e9f;
    const auto energy = audio.square().mean().item<float>();
    const auto maxDelay = std::min<std::int64_t>(
        reconstructed.size(2) / 2, static_cast<std::int64_t>(512));
    for (std::int64_t delay = 0; delay < maxDelay; delay += 1) {
      const auto length =
          std::min(audio.size(2), reconstructed.size(2) - delay);
      if (length < 128)
        break;
      const auto error = (audio.narrow(2, 0, length) -
                          reconstructed.narrow(2, delay, length))
                             .square()
                             .mean()
                             .item<float>();
      if (energy > 0.0f)
        bestRelative = std::min(bestRelative, error / energy);
    }
    passed &= expect(bestRelative < 1.5f,
                     "PQMF analysis then synthesis must be approximately invertible");
  }

  {
    openyourbox::dsp::PqmfBank bank(16);
    auto analysisLeftover = bank.makeLeftover(1);
    auto synthesisLeftover = bank.makeLeftover(16);
    const auto blockSize = static_cast<std::int64_t>(512);
    const auto totalSamples = static_cast<std::int64_t>(8192);
    auto source = torch::randn({1, 1, totalSamples});
    auto reconstructed = torch::zeros_like(source);
    std::int64_t writeOffset = 0;
    for (std::int64_t offset = 0; offset + blockSize <= totalSamples;
         offset += blockSize) {
      const auto block = source.narrow(2, offset, blockSize);
      const auto bands =
          bank.analyseStreaming(block, analysisLeftover);
      if (bands.size(2) < 1)
        continue;
      const auto audio =
          bank.synthesiseStreaming(bands, synthesisLeftover, 1);
      const auto emit = std::min(audio.size(2), blockSize);
      reconstructed.narrow(2, writeOffset, emit).copy_(audio.narrow(2, 0, emit));
      writeOffset += emit;
    }
    float bestRelative = 1.0e9f;
    const auto energy = source.narrow(2, 0, writeOffset).square().mean().item<float>();
    const auto maxDelay = std::min<std::int64_t>(writeOffset / 2, 512);
    for (std::int64_t delay = 0; delay < maxDelay; delay += 1) {
      const auto length = writeOffset - delay;
      if (length < 256)
        break;
      const auto error =
          (source.narrow(2, 0, length) - reconstructed.narrow(2, delay, length))
              .square()
              .mean()
              .item<float>();
      if (energy > 0.0f)
        bestRelative = std::min(bestRelative, error / energy);
    }
    passed &= expect(writeOffset >= totalSamples - blockSize,
                     "PQMF streaming must emit nearly all host samples");
    passed &= expect(bestRelative < 1.5f,
                     "PQMF streaming analysis then synthesis must be invertible");
  }

  {
    openyourbox::dsp::RateConv down(2, 3, 1,
                                    openyourbox::dsp::RateConvMode::downsample);
    auto input = torch::randn({1, 2, 64});
    auto weight = torch::randn({4, 2, 3});
    const auto downsampled = down.process(input, weight);
    passed &= expect(downsampled.size(2) == 32,
                     "causal downsample rateConv must emit T/stride samples");
    openyourbox::dsp::RateConv up(2, 3, 1,
                                  openyourbox::dsp::RateConvMode::upsample);
    auto upWeight = torch::randn({2, 4, 3});
    const auto upsampled = up.process(downsampled, upWeight);
    passed &= expect(upsampled.size(2) == 64,
                     "causal upsample rateConv must restore T * stride samples");
  }

  {
    using openyourbox::graph::NodeGraph;
    using openyourbox::graph::NodeType;
    NodeGraph raveGraph;
    const auto raveInputId =
        raveGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        raveGraph.addNode(NodeType::pqmfAnalysis, {100.0f, 0.0f});
    const auto downsample =
        raveGraph.addNode(NodeType::convolution, {200.0f, 0.0f});
    const auto bottleneck =
        raveGraph.addNode(NodeType::variationalBottleneck, {300.0f, 0.0f});
    const auto upsample =
        raveGraph.addNode(NodeType::convTranspose, {400.0f, 0.0f});
    const auto left =
        raveGraph.addNode(NodeType::activation, {500.0f, -50.0f});
    const auto right =
        raveGraph.addNode(NodeType::activation, {500.0f, 50.0f});
    const auto envelopeMath =
        raveGraph.addNode(NodeType::mathExpression, {600.0f, 50.0f});
    const auto merge =
        raveGraph.addNode(NodeType::merge, {700.0f, 0.0f});
    const auto synthesis =
        raveGraph.addNode(NodeType::pqmfSynthesis, {800.0f, 0.0f});
    const auto raveOutputId =
        raveGraph.addNode(NodeType::audioOutput, {900.0f, 0.0f});
    raveGraph.setProperty(analysis, "n_band", 2);
    raveGraph.setProperty(downsample, "channels", 4);
    raveGraph.setProperty(downsample, "stride", 2);
    raveGraph.setProperty(bottleneck, "latent_size", 2);
    raveGraph.setProperty(upsample, "channels", 4);
    raveGraph.setProperty(upsample, "stride", 2);
    raveGraph.setProperty(synthesis, "n_band", 2);

    const auto *raveInputNode = raveGraph.findNode(raveInputId);
    const auto *analysisNode = raveGraph.findNode(analysis);
    const auto *downsampleNode = raveGraph.findNode(downsample);
    const auto *bottleneckNode = raveGraph.findNode(bottleneck);
    const auto *upsampleNode = raveGraph.findNode(upsample);
    const auto *leftNode = raveGraph.findNode(left);
    const auto *rightNode = raveGraph.findNode(right);
    const auto *envelopeMathNode = raveGraph.findNode(envelopeMath);
    const auto *raveMergeNode = raveGraph.findNode(merge);
    const auto *synthesisNode = raveGraph.findNode(synthesis);
    const auto *raveOutputNode = raveGraph.findNode(raveOutputId);
    const auto wired =
        raveInputNode != nullptr && analysisNode != nullptr &&
        downsampleNode != nullptr && bottleneckNode != nullptr &&
        upsampleNode != nullptr && leftNode != nullptr && rightNode != nullptr &&
        envelopeMathNode != nullptr && raveMergeNode != nullptr &&
        synthesisNode != nullptr && raveOutputNode != nullptr &&
        raveGraph
            .connect(raveInputNode->outputs.front().id,
                     analysisNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(analysisNode->outputs.front().id,
                     downsampleNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(downsampleNode->outputs.front().id,
                     bottleneckNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(bottleneckNode->outputs.front().id,
                     upsampleNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(upsampleNode->outputs.front().id,
                     leftNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(upsampleNode->outputs.front().id,
                     rightNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(rightNode->outputs.front().id,
                     envelopeMathNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(leftNode->outputs.front().id,
                     raveMergeNode->inputs[0].id)
            .accepted &&
        raveGraph
            .connect(envelopeMathNode->outputs.front().id,
                     raveMergeNode->inputs[1].id)
            .accepted &&
        raveGraph
            .connect(raveMergeNode->outputs.front().id,
                     synthesisNode->inputs.front().id)
            .accepted &&
        raveGraph
            .connect(synthesisNode->outputs.front().id,
                     raveOutputNode->inputs.front().id)
            .accepted;
    passed &= expect(wired, "end-to-end RAVE fixture must wire");

    const auto raveCompiled = LiveGraphEngine::compile(raveGraph, options);
    const auto raveRuntime =
        LiveGraphEngine::prepare(raveCompiled.snapshot, error);
    passed &= expect(raveCompiled.succeeded() && raveRuntime != nullptr,
                     "end-to-end RAVE graph must compile and prepare");
    if (raveRuntime != nullptr) {
      torch::Tensor raveOutput;
      bool processed = false;
      try {
        raveOutput =
            raveRuntime->processTensor(torch::randn({1, 2, 256}, torch::kFloat32));
        processed = true;
      } catch (...) {
      }
      passed &= expect(processed && raveOutput.defined() &&
                           raveOutput.size(2) == 256 &&
                           raveOutput.abs().max().item<float>() > 0.0f,
                       "untrained end-to-end RAVE must emit non-silent audio");

      std::array<std::vector<float>, 2> inputPlanes{
          std::vector<float>(256, 0.25f), std::vector<float>(256, -0.25f)};
      std::array<std::vector<float>, 2> outputPlanes{
          std::vector<float>(256), std::vector<float>(256)};
      const float *inputPointers[] = {inputPlanes[0].data(),
                                      inputPlanes[1].data()};
      float *outputPointers[] = {outputPlanes[0].data(),
                                 outputPlanes[1].data()};
      passed &= expect(
          !raveRuntime->processHost(inputPointers, 1, outputPointers, 2, 256) &&
              raveRuntime->getLastProcessingFailureNodeId() == -1,
          "host processing failures must latch a UI-readable warning");
    }
  }

  {
    openyourbox::graph::NodeGraph domainGraph;
    const auto analysis = domainGraph.addNode(
        openyourbox::graph::NodeType::pqmfAnalysis, {180.0f, 0.0f});
    const auto tcn = domainGraph.addNode(openyourbox::graph::NodeType::tcn,
                                         {360.0f, 0.0f});
    const auto conv = domainGraph.addNode(
        openyourbox::graph::NodeType::convolution, {450.0f, 0.0f});
    const auto merge = domainGraph.addNode(
        openyourbox::graph::NodeType::merge, {620.0f, 0.0f});
    const auto synthesis = domainGraph.addNode(
        openyourbox::graph::NodeType::pqmfSynthesis, {800.0f, 0.0f});
    const auto bottleneck = domainGraph.addNode(
        openyourbox::graph::NodeType::variationalBottleneck, {980.0f, 0.0f});
    const auto output = domainGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {1160.0f, 0.0f});
    const auto *analysisNode = domainGraph.findNode(analysis);
    const auto *tcnNode = domainGraph.findNode(tcn);
    const auto *convNode = domainGraph.findNode(conv);
    const auto *mergeNode = domainGraph.findNode(merge);
    const auto *synthesisNode = domainGraph.findNode(synthesis);
    const auto *bottleneckNode = domainGraph.findNode(bottleneck);
    const auto *outputNode = domainGraph.findNode(output);
    passed &= expect(analysisNode != nullptr && tcnNode != nullptr &&
                         domainGraph
                              .connect(analysisNode->outputs.front().id,
                                       tcnNode->inputs.front().id)
                              .accepted,
                     "PQMF analysis must connect to a passthrough TCN");
    passed &= expect(tcnNode != nullptr &&
                         tcnNode->inputs.front().shape.nBand ==
                             openyourbox::graph::defaultPqmfBands &&
                         tcnNode->inputs.front().shape.temporalRate ==
                             openyourbox::graph::defaultPqmfBands,
                     "TCN input must inherit PQMF hop rate and nBand");
    passed &= expect(convNode != nullptr && tcnNode != nullptr &&
                         domainGraph
                             .connect(tcnNode->outputs.front().id,
                                      convNode->inputs.front().id)
                             .accepted,
                     "multiband TCN must connect to Conv1D");
    passed &= expect(domainGraph.setProperty(conv, "channels", 16),
                     "multiband Conv1D must preserve nBand width before synthesis");
    passed &= expect(
        mergeNode != nullptr && synthesisNode != nullptr && convNode != nullptr &&
            domainGraph
                .connect(convNode->outputs.front().id, mergeNode->inputs.front().id)
                .accepted &&
            domainGraph
                .connect(mergeNode->outputs.front().id,
                         synthesisNode->inputs.front().id)
                .accepted,
        "multiband Merge must connect to PQMF synthesis");
    passed &=
        expect(bottleneckNode != nullptr && outputNode != nullptr &&
                   !domainGraph
                        .connect(bottleneckNode->outputs.front().id,
                                 outputNode->inputs.front().id)
                        .accepted,
               "host audio output must refuse a 128-channel bottleneck cable");
    const auto knob = domainGraph.addNode(
        openyourbox::graph::NodeType::knobInput, {360.0f, 180.0f});
    const auto *tcnAfterKnob = domainGraph.findNode(tcn);
    passed &= expect(
        tcnAfterKnob != nullptr && tcnAfterKnob->inputs.size() >= 2 &&
            domainGraph.findNode(knob) != nullptr &&
            domainGraph
                .connect(domainGraph.findNode(knob)->outputs.front().id,
                         tcnAfterKnob->inputs[1].id)
                .accepted,
        "TCN control pin must accept a knob tensor");
  }

  {
    openyourbox::graph::NodeGraph rateGraph;
    const auto analysis = rateGraph.addNode(
        openyourbox::graph::NodeType::pqmfAnalysis, {0.0f, 0.0f});
    const auto conv = rateGraph.addNode(
        openyourbox::graph::NodeType::convTranspose, {180.0f, 0.0f});
    passed &= expect(rateGraph.setProperty(conv, "stride", 3),
                     "unwired ConvTranspose1d may set stride 3");
    const auto *analysisNode = rateGraph.findNode(analysis);
    const auto *convNode = rateGraph.findNode(conv);
    passed &= expect(
        analysisNode != nullptr && convNode != nullptr &&
            !rateGraph
                 .connect(analysisNode->outputs.front().id,
                          convNode->inputs.front().id)
                 .accepted,
        "ConvTranspose1d must refuse a temporal rate that stride does not divide");
  }

  {
    openyourbox::graph::NodeGraph bandGraph;
    const auto analysis = bandGraph.addNode(
        openyourbox::graph::NodeType::pqmfAnalysis, {0.0f, 0.0f});
    const auto linear = bandGraph.addNode(
        openyourbox::graph::NodeType::linear, {180.0f, 0.0f});
    const auto synthesis = bandGraph.addNode(
        openyourbox::graph::NodeType::pqmfSynthesis, {360.0f, 0.0f});
    const auto *analysisNode = bandGraph.findNode(analysis);
    const auto *linearNode = bandGraph.findNode(linear);
    const auto *synthesisNode = bandGraph.findNode(synthesis);
    passed &= expect(
        analysisNode != nullptr && linearNode != nullptr &&
            synthesisNode != nullptr &&
            bandGraph
                .connect(analysisNode->outputs.front().id,
                         linearNode->inputs.front().id)
                .accepted,
        "PQMF analysis must connect to Linear");
    passed &= expect(
        linearNode != nullptr && synthesisNode != nullptr &&
            !bandGraph
                 .connect(linearNode->outputs.front().id,
                          synthesisNode->inputs.front().id)
                 .accepted,
        "PQMF synthesis must refuse a band count that is not a multiple of nBand");
    passed &= expect(bandGraph.setProperty(linear, "features", 16),
                     "Linear features may be set to preserve nBand width");
    passed &= expect(
        linearNode != nullptr && synthesisNode != nullptr &&
            bandGraph
                .connect(linearNode->outputs.front().id,
                         synthesisNode->inputs.front().id)
                .accepted,
        "PQMF synthesis must accept a channel count that is a multiple of nBand");
  }

  {
    openyourbox::graph::NodeGraph invalid;
    const auto inId =
        invalid.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto tcnId =
        invalid.addNode(openyourbox::graph::NodeType::tcn, {180.0f, 0.0f});
    const auto outId =
        invalid.addNode(openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
    invalid.connect(invalid.findNode(inId)->outputs.front().id,
                    invalid.findNode(tcnId)->inputs.front().id);
    invalid.connect(invalid.findNode(tcnId)->outputs.front().id,
                    invalid.findNode(outId)->inputs.front().id);
    passed &= expect(!invalid.hasReconstructionTrainPath(),
                     "TCN-only graphs must fail the reconstruction path gate");

    openyourbox::graph::NodeGraph valid;
    const auto vIn =
        valid.addNode(openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto vBn = valid.addNode(
        openyourbox::graph::NodeType::variationalBottleneck, {180.0f, 0.0f});
    const auto vDecode =
        valid.addNode(openyourbox::graph::NodeType::linear, {270.0f, 0.0f});
    const auto vOut =
        valid.addNode(openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
    auto *bn = const_cast<openyourbox::graph::GraphNode *>(valid.findNode(vBn));
    if (bn != nullptr)
      bn->armedForTraining = true;
    const auto *inNode = valid.findNode(vIn);
    const auto *bnNode = valid.findNode(vBn);
    const auto *decodeNode = valid.findNode(vDecode);
    const auto *outNode = valid.findNode(vOut);
    passed &= expect(
        inNode != nullptr && bnNode != nullptr && decodeNode != nullptr &&
            outNode != nullptr &&
            valid.connect(inNode->outputs.front().id, bnNode->inputs.front().id)
                .accepted &&
            valid
                .connect(bnNode->outputs.front().id,
                         decodeNode->inputs.front().id)
                .accepted &&
            valid
                .connect(decodeNode->outputs.front().id,
                         outNode->inputs.front().id)
                .accepted,
        "test reconstruction path cables must connect");
    passed &= expect(valid.hasReconstructionTrainPath(),
                     "armed bottleneck with decode-to-output must pass the gate");
  }

  {
    openyourbox::graph::NodeGraph ioGraph;
    const auto inId = ioGraph.addNode(
        openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto actId = ioGraph.addNode(
        openyourbox::graph::NodeType::activation, {180.0f, 0.0f});
    const auto outId = ioGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(ioGraph.setProperty(inId, "channels", 0),
                     "Audio Input must accept Mono");
    passed &= expect(ioGraph.setProperty(outId, "channels", 0),
                     "Audio Output must accept Mono");
    const auto *inNode = ioGraph.findNode(inId);
    const auto *outNode = ioGraph.findNode(outId);
    passed &= expect(
        inNode != nullptr && outNode != nullptr &&
            inNode->outputs.front().shape.channels == 1 &&
            outNode->inputs.front().shape.channels == 1 &&
            inNode->outputs.front().shape.displayLabel().find("1ch") !=
                std::string::npos &&
            inNode->detail.find("1ch") != std::string::npos,
        "Mono host I/O must update pin shapes, labels, and detail");
    passed &= expect(
        ioGraph
            .connect(ioGraph.findNode(inId)->outputs.front().id,
                     ioGraph.findNode(actId)->inputs.front().id)
            .accepted &&
            ioGraph
                .connect(ioGraph.findNode(actId)->outputs.front().id,
                         ioGraph.findNode(outId)->inputs.front().id)
                .accepted,
        "mono I/O must connect through Activation");
    openyourbox::dsp::LiveGraphCompileOptions monoOptions;
    monoOptions.hostInputChannels = 1;
    monoOptions.hostOutputChannels = 1;
    monoOptions.maximumBlockSize = 32;
    const auto monoCompiled = LiveGraphEngine::compile(ioGraph, monoOptions);
    passed &= expect(monoCompiled.succeeded(),
                     "mono graph must compile against a mono host");
    openyourbox::dsp::LiveGraphCompileOptions stereoHost;
    stereoHost.hostInputChannels = 2;
    stereoHost.hostOutputChannels = 2;
    stereoHost.maximumBlockSize = 32;
    const auto summed = LiveGraphEngine::compile(ioGraph, stereoHost);
    passed &= expect(summed.succeeded(),
                     "mono mode must compile on a stereo host via (L+R)/2");
    if (summed.succeeded()) {
      openyourbox::dsp::LiveGraphCompileError error;
      const auto runtime = LiveGraphEngine::prepare(summed.snapshot, error);
      passed &= expect(runtime != nullptr, "summed mono runtime must prepare");
      if (runtime != nullptr) {
        auto stereo = torch::zeros({1, 2, 8}, torch::kFloat32);
        stereo[0][0].fill_(1.0f);
        stereo[0][1].fill_(3.0f);
        const auto out = runtime->processTensor(stereo);
        passed &= expect(out.size(1) == 2, "stereo host still receives 2 outs");
        const auto left = out[0][0].mean().item<float>();
        const auto right = out[0][1].mean().item<float>();
        passed &= expect(std::abs(left - 2.0f) < 1.0e-4f &&
                             std::abs(right - 2.0f) < 1.0e-4f,
                         "mono mode must fold stereo host with (L+R)/2");
      }
    }
    passed &= expect(ioGraph.setProperty(inId, "channels", 1),
                     "Audio Input must accept Mirrored");
    passed &= expect(ioGraph.findNode(inId)->outputs.front().shape.channels == 2 &&
                         ioGraph.findNode(inId)->detail.find("L=R") !=
                             std::string::npos,
                     "Mirrored mode must declare 2ch L=R");
    const auto *actAfterMirror = ioGraph.findNode(actId);
    passed &= expect(
        actAfterMirror != nullptr &&
            actAfterMirror->inputs.front().shape.channels == 2 &&
            actAfterMirror->outputs.front().shape.channels == 2,
        "Mirrored host width must propagate through Activation pins");
    const auto mirrored = LiveGraphEngine::compile(ioGraph, stereoHost);
    passed &= expect(mirrored.succeeded(),
                     "mirrored mode must compile on a stereo host");
    passed &= expect(ioGraph.setProperty(inId, "channels", 0),
                     "host I/O must switch back to Mono");
    const auto *actAfterMono = ioGraph.findNode(actId);
    passed &= expect(
        actAfterMono != nullptr &&
            actAfterMono->inputs.front().shape.channels == 1 &&
            actAfterMono->outputs.front().shape.channels == 1 &&
            actAfterMono->outputs.front().shape.displayLabel().find("1ch") !=
                std::string::npos,
        "Mono host width must re-propagate through Activation pins");
    passed &= expect(ioGraph.setProperty(inId, "channels", 2) &&
                         ioGraph.setProperty(outId, "channels", 2),
                     "host I/O must switch to Stereo");
    passed &= expect(ioGraph.findNode(inId)->outputs.front().shape.channels == 2,
                     "Stereo Audio Input must declare 2 channels");
  }

  {
    openyourbox::graph::NodeGraph xyMixGraph;
    const auto xyMixIn = xyMixGraph.addNode(
        openyourbox::graph::NodeType::audioInput, {0.0f, 0.0f});
    const auto xyMixPad = xyMixGraph.addNode(
        openyourbox::graph::NodeType::xyTrackpad, {0.0f, 140.0f});
    const auto xyMixTcn = xyMixGraph.addNode(
        openyourbox::graph::NodeType::tcn, {180.0f, 140.0f});
    const auto xyMixMerge = xyMixGraph.addNode(
        openyourbox::graph::NodeType::merge, {360.0f, 0.0f});
    const auto xyMixOut = xyMixGraph.addNode(
        openyourbox::graph::NodeType::audioOutput, {540.0f, 0.0f});
    passed &= expect(xyMixGraph.setProperty(xyMixIn, "channels", 0),
                     "XY mix graph must switch host I/O to Mono");
    const auto *xyMixInNode = xyMixGraph.findNode(xyMixIn);
    const auto *xyMixPadNode = xyMixGraph.findNode(xyMixPad);
    const auto *xyMixTcnNode = xyMixGraph.findNode(xyMixTcn);
    const auto *xyMixMergeNode = xyMixGraph.findNode(xyMixMerge);
    const auto *xyMixOutNode = xyMixGraph.findNode(xyMixOut);
    passed &= expect(
        xyMixInNode != nullptr && xyMixPadNode != nullptr &&
            xyMixTcnNode != nullptr && xyMixMergeNode != nullptr &&
            xyMixOutNode != nullptr && xyMixMergeNode->inputs.size() >= 2 &&
            xyMixGraph
                .connect(xyMixPadNode->outputs.front().id,
                         xyMixTcnNode->inputs.front().id)
                .accepted &&
            xyMixGraph
                .connect(xyMixTcnNode->outputs.front().id,
                         xyMixMergeNode->inputs.front().id)
                .accepted &&
            xyMixGraph
                .connect(xyMixInNode->outputs.front().id,
                         xyMixMergeNode->inputs[1].id)
                .accepted &&
            xyMixGraph
                .connect(xyMixMergeNode->outputs.front().id,
                         xyMixOutNode->inputs.front().id)
                .accepted,
        "XY through TCN must merge with mono host audio");
    openyourbox::dsp::LiveGraphCompileOptions xyMixOptions;
    xyMixOptions.hostInputChannels = 1;
    xyMixOptions.hostOutputChannels = 1;
    xyMixOptions.maximumBlockSize = 32;
    const auto xyMixCompiled =
        LiveGraphEngine::compile(xyMixGraph, xyMixOptions);
    passed &= expect(
        xyMixCompiled.succeeded(),
        xyMixCompiled.succeeded()
            ? "XY through TCN must compile into a mono Mix"
            : xyMixCompiled.error.message.c_str());
  }

  {
    openyourbox::graph::NodeGraph oddGraph;
    oddGraph.ensureFixedHostIo();
    const auto linear = oddGraph.addNode(
        openyourbox::graph::NodeType::linear, {180.0f, 0.0f});
    const auto bottleneck = oddGraph.addNode(
        openyourbox::graph::NodeType::variationalBottleneck, {360.0f, 0.0f});
    passed &= expect(oddGraph.setProperty(linear, "features", 3),
                     "linear features can be set to an odd width");
    const auto *linearNode = oddGraph.findNode(linear);
    const auto *bottleneckNode = oddGraph.findNode(bottleneck);
    passed &= expect(
        linearNode != nullptr && bottleneckNode != nullptr &&
            !oddGraph
                 .connect(linearNode->outputs.front().id,
                          bottleneckNode->inputs.front().id)
                 .accepted,
        "variational bottleneck must refuse odd upstream channels");
  }

  {
    openyourbox::graph::NodeGraph bottleneckGraph;
    bottleneckGraph.ensureFixedHostIo();
    std::int32_t inputId = 0;
    std::int32_t outputId = 0;
    for (const auto &node : bottleneckGraph.getNodes()) {
      if (node.type == openyourbox::graph::NodeType::audioInput)
        inputId = node.id;
      if (node.type == openyourbox::graph::NodeType::audioOutput)
        outputId = node.id;
    }
    const auto bottleneckId = bottleneckGraph.addNode(
        openyourbox::graph::NodeType::variationalBottleneck, {220.0f, 0.0f});
    const auto linearId = bottleneckGraph.addNode(
        openyourbox::graph::NodeType::linear, {400.0f, 0.0f});
    const auto *input = bottleneckGraph.findNode(inputId);
    const auto *output = bottleneckGraph.findNode(outputId);
    const auto *bottleneck = bottleneckGraph.findNode(bottleneckId);
    const auto *linear = bottleneckGraph.findNode(linearId);
    passed &= expect(
        input != nullptr && output != nullptr && bottleneck != nullptr &&
            linear != nullptr &&
            bottleneckGraph.setProperty(bottleneckId, "latent_size", 4) &&
            bottleneckGraph.setProperty(linearId, "features", 2) &&
            bottleneckGraph
                .connect(input->outputs.front().id, bottleneck->inputs.front().id)
                .accepted &&
            bottleneckGraph
                .connect(bottleneck->outputs.front().id, linear->inputs.front().id)
                .accepted &&
            bottleneckGraph
                .connect(linear->outputs.front().id, output->inputs.front().id)
                .accepted,
        "2ch host audio must wire through a 4-wide bottleneck");
    openyourbox::dsp::LiveGraphCompileOptions options;
    options.hostInputChannels = 2;
    options.hostOutputChannels = 2;
    options.maximumBlockSize = 32;
    const auto compiled =
        openyourbox::dsp::LiveGraphEngine::compile(bottleneckGraph, options);
    passed &= expect(compiled.succeeded(),
                     compiled.succeeded()
                         ? "bottleneck graph must compile"
                         : compiled.error.message.c_str());
    if (compiled.succeeded()) {
      openyourbox::dsp::LiveGraphCompileError prepareError;
      const auto runtimeA = openyourbox::dsp::LiveGraphEngine::prepare(
          compiled.snapshot, prepareError);
      const auto runtimeB = openyourbox::dsp::LiveGraphEngine::prepare(
          compiled.snapshot, prepareError);
      auto inputTensor = torch::randn({1, 2, 32});
      const auto outA =
          runtimeA != nullptr ? runtimeA->processTensor(inputTensor)
                              : torch::Tensor{};
      const auto outB =
          runtimeB != nullptr ? runtimeB->processTensor(inputTensor)
                              : torch::Tensor{};
      passed &= expect(outA.defined() && outB.defined() && torch::equal(outA, outB),
                       "live bottleneck encode must be deterministic μ-only");
    }
  }

  {
    openyourbox::dsp::ConvTranspose1d conv(4, 8, 1);
    auto weight = torch::randn({8, 16, 8}, torch::kFloat32);
    auto input = torch::randn({1, 16, 4}, torch::kFloat32);
    torch::Tensor leftover;
    bool threw = false;
    std::string message;
    torch::Tensor streamed;
    try {
      streamed = conv.processStreaming(input, weight, leftover);
      streamed = conv.processStreaming(input, weight, leftover);
    } catch (const std::exception &exception) {
      threw = true;
      message = exception.what();
    }
    passed &= expect(!threw && streamed.defined() && streamed.size(1) == 8,
                     threw ? message.c_str()
                           : "stacked Tiny-RAVE ConvTranspose streaming must run");
  }

  {
    using openyourbox::graph::NodeGraph;
    using openyourbox::graph::NodeType;
    NodeGraph tiny;
    const auto inId = tiny.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        tiny.addNode(NodeType::pqmfAnalysis, {100.0f, 0.0f});
    const auto down1 =
        tiny.addNode(NodeType::convolution, {180.0f, 0.0f});
    const auto down2 =
        tiny.addNode(NodeType::convolution, {240.0f, 0.0f});
    const auto act = tiny.addNode(NodeType::activation, {300.0f, 0.0f});
    const auto convT =
        tiny.addNode(NodeType::convTranspose, {400.0f, 0.0f});
    const auto synth =
        tiny.addNode(NodeType::pqmfSynthesis, {500.0f, 0.0f});
    const auto outId = tiny.addNode(NodeType::audioOutput, {600.0f, 0.0f});
    tiny.setProperty(analysis, "n_band", 4);
    tiny.setProperty(synth, "n_band", 4);
    tiny.setProperty(down1, "channels", 16);
    tiny.setProperty(down1, "stride", 4);
    tiny.setProperty(down1, "kernel_size", 9);
    tiny.setProperty(down2, "channels", 32);
    tiny.setProperty(down2, "stride", 4);
    tiny.setProperty(down2, "kernel_size", 9);
    tiny.setProperty(convT, "channels", 16);
    tiny.setProperty(convT, "stride", 4);
    tiny.setProperty(convT, "kernel_size", 8);
    auto requireConnect = [&](std::int32_t sourcePin, std::int32_t destPin,
                              const char *label) {
      const auto result = tiny.connect(sourcePin, destPin);
      passed &= expect(result.accepted,
                       result.accepted ? label : result.message.c_str());
    };
    requireConnect(tiny.findNode(inId)->outputs.front().id,
                   tiny.findNode(analysis)->inputs.front().id,
                   "Audio In to PQMF");
    requireConnect(tiny.findNode(analysis)->outputs.front().id,
                   tiny.findNode(down1)->inputs.front().id, "PQMF to down 1");
    requireConnect(tiny.findNode(down1)->outputs.front().id,
                   tiny.findNode(down2)->inputs.front().id, "down 1 to down 2");
    requireConnect(tiny.findNode(act)->outputs.front().id,
                   tiny.findNode(convT)->inputs.front().id,
                   "Activation to ConvTranspose");
    const auto grouped = tiny.createGroup({act, convT});
    passed &= expect(grouped.accepted, "upsample group must be created");
    auto findBoundary = [](const NodeGraph &graph, std::int32_t groupId,
                           NodeType type) -> std::int32_t {
      const auto *group = graph.findGroup(groupId);
      if (group == nullptr)
        return 0;
      for (const auto memberId : group->memberIds) {
        const auto *node = graph.findNode(memberId);
        if (node != nullptr && node->type == type)
          return node->id;
      }
      return 0;
    };
    const auto inputHub = tiny.findNode(
        findBoundary(tiny, grouped.groupId, NodeType::groupInput));
    const auto outputHub = tiny.findNode(
        findBoundary(tiny, grouped.groupId, NodeType::groupOutput));
    passed &= expect(inputHub != nullptr && outputHub != nullptr,
                     "upsample group must have hubs");
    if (inputHub != nullptr && outputHub != nullptr) {
      requireConnect(inputHub->outputs.front().id,
                     tiny.findNode(act)->inputs.front().id, "hub to Activation");
      requireConnect(tiny.findNode(convT)->outputs.front().id,
                     outputHub->inputs.front().id, "ConvTranspose to hub");
      requireConnect(tiny.findNode(down2)->outputs.front().id,
                     inputHub->inputs.front().id, "down 2 to upsample group");
    }
    const auto repeats = tiny.setGroupRepeats(grouped.groupId, 2);
    passed &= expect(repeats.accepted,
                     repeats.accepted ? "upsample group repeats=2"
                                      : repeats.message.c_str());
    passed &= expect(tiny.setPropertyRepeatValues(convT, "channels", {16, 8}),
                     "ConvTranspose channels list 16, 8");
    if (outputHub != nullptr) {
      requireConnect(outputHub->outputs.front().id,
                     tiny.findNode(synth)->inputs.front().id,
                     "upsample group to PQMF synth");
      requireConnect(tiny.findNode(synth)->outputs.front().id,
                     tiny.findNode(outId)->inputs.front().id,
                     "PQMF synth to Audio Out");
    }
    const auto expanded = tiny.withInvisibleRepeatsMaterialized();
    const auto tinyCompiled = LiveGraphEngine::compile(expanded, options);
    passed &= expect(tinyCompiled.succeeded(),
                     tinyCompiled.succeeded()
                         ? "Tiny-RAVE upsample graph must compile"
                         : tinyCompiled.error.message.c_str());
    if (tinyCompiled.succeeded()) {
      for (const auto &stats : tinyCompiled.snapshot->getElementStatistics()) {
        if (stats.type != openyourbox::graph::NodeType::convTranspose)
          continue;
        std::cerr << "ConvTranspose node " << stats.nodeId << " in="
                  << stats.inputChannels << " out=" << stats.outputChannels
                  << "\n";
      }
      LiveGraphCompileError prepareError;
      const auto tinyRuntime =
          LiveGraphEngine::prepare(tinyCompiled.snapshot, prepareError);
      passed &= expect(tinyRuntime != nullptr, "Tiny-RAVE upsample must prepare");
      if (tinyRuntime != nullptr) {
        bool processed = false;
        std::string processError;
        try {
          const auto output = tinyRuntime->processTensor(
              torch::randn({1, 2, 256}, torch::kFloat32));
          processed = output.defined() && output.size(2) == 256;
        } catch (const std::exception &exception) {
          processError = exception.what();
        }
        if (!processed && processError.empty() &&
            tinyRuntime->getLastProcessingFailureNodeId() != 0)
          processError =
              "muted at node " +
              std::to_string(tinyRuntime->getLastProcessingFailureNodeId());
        passed &= expect(processed, processError.empty()
                                        ? "Tiny-RAVE upsample must process live"
                                        : processError.c_str());
      }
    }
  }

  {
    const auto boxFile = juce::File(
        "/Users/hugo/Library/Audio/Presets/Allendia/OpenYourBox/Boxes/entries/"
        "c6b7b94b-52cc-497f-b5ee-b24f213ec1b9/box.xml");
    if (boxFile.existsAsFile()) {
      const auto xml = juce::XmlDocument::parse(boxFile);
      passed &= expect(xml != nullptr, "Tiny-RAVE box.xml must parse");
      if (xml != nullptr) {
        using openyourbox::graph::NodeGraph;
        using openyourbox::graph::NodeType;
        NodeGraph libraryGraph;
        const auto inId =
            libraryGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
        const auto outId =
            libraryGraph.addNode(NodeType::audioOutput, {800.0f, 0.0f});
        passed &= expect(libraryGraph.setProperty(inId, "channels", 0) &&
                             libraryGraph.setProperty(outId, "channels", 0),
                         "Tiny-RAVE host I/O must be mono");
        juce::String boxImportError;
        const auto snapshot = juce::ValueTree::fromXml(*xml);
        const auto rootId = libraryGraph.importBox(snapshot, {200.0f, 0.0f},
                                                   true, boxImportError);
        passed &= expect(rootId.has_value(),
                         boxImportError.isEmpty()
                             ? "Tiny-RAVE library box must import"
                             : boxImportError.toStdString().c_str());
        if (rootId.has_value()) {
          const auto *group = libraryGraph.findGroup(*rootId);
          std::int32_t inputHubId = 0;
          std::int32_t outputHubId = 0;
          if (group != nullptr) {
            for (const auto memberId : group->memberIds) {
              const auto *node = libraryGraph.findNode(memberId);
              if (node == nullptr)
                continue;
              if (node->type == NodeType::groupInput)
                inputHubId = node->id;
              else if (node->type == NodeType::groupOutput)
                outputHubId = node->id;
            }
          }
          const auto *inNode = libraryGraph.findNode(inId);
          const auto *outNode = libraryGraph.findNode(outId);
          const auto *inputHub = libraryGraph.findNode(inputHubId);
          const auto *outputHub = libraryGraph.findNode(outputHubId);
          passed &= expect(inNode != nullptr && outNode != nullptr &&
                               inputHub != nullptr && outputHub != nullptr,
                           "Tiny-RAVE library box must expose I/O hubs");
          if (inNode != nullptr && outNode != nullptr && inputHub != nullptr &&
              outputHub != nullptr) {
            const auto inLink = libraryGraph.connect(inNode->outputs.front().id,
                                                     inputHub->inputs.front().id);
            const auto outLink = libraryGraph.connect(
                outputHub->outputs.front().id, outNode->inputs.front().id);
            passed &= expect(inLink.accepted,
                             inLink.accepted ? "Audio In to Tiny-RAVE"
                                             : inLink.message.c_str());
            passed &= expect(outLink.accepted,
                             outLink.accepted ? "Tiny-RAVE to Audio Out"
                                              : outLink.message.c_str());
          }
          const auto expanded = libraryGraph.withInvisibleRepeatsMaterialized();
          int expandedIn = 0;
          int expandedOut = 0;
          int expandedLinks = 0;
          for (const auto &node : expanded.getNodes()) {
            if (node.type == NodeType::audioInput)
              expandedIn = node.id;
            else if (node.type == NodeType::audioOutput)
              expandedOut = node.id;
          }
          for (const auto &link : expanded.getLinks()) {
            const auto source = expanded.findNodeForPin(link.sourcePinId);
            if (source.has_value() && *source == expandedIn)
              ++expandedLinks;
          }
          const auto compiled = LiveGraphEngine::compile(expanded, options);
          std::string compileMessage = compiled.error.message;
          if (!compiled.succeeded())
            compileMessage +=
                " (in=" + std::to_string(expandedIn) +
                " out=" + std::to_string(expandedOut) +
                " inLinks=" + std::to_string(expandedLinks) +
                " errNode=" + std::to_string(compiled.error.nodeId) + ")";
          passed &= expect(compiled.succeeded(),
                           compiled.succeeded()
                               ? "Tiny-RAVE library graph must compile"
                               : compileMessage.c_str());
          if (compiled.succeeded()) {
            LiveGraphCompileError prepareError;
            const auto runtime =
                LiveGraphEngine::prepare(compiled.snapshot, prepareError);
            passed &= expect(runtime != nullptr,
                             "Tiny-RAVE library graph must prepare");
            if (runtime != nullptr) {
              bool processed = false;
              std::string processError;
              try {
                for (const auto block : {32, 64, 128, 256, 512}) {
                  if (block > options.maximumBlockSize)
                    continue;
                  const auto output = runtime->processTensor(
                      torch::randn({1, 2, block}, torch::kFloat32));
                  if (!output.defined() || output.size(2) != block) {
                    processError = "block " + std::to_string(block) +
                                   " returned time " +
                                   std::to_string(output.defined()
                                                      ? output.size(2)
                                                      : -1);
                    processed = false;
                    break;
                  }
                  processed = true;
                }
              } catch (const std::exception &exception) {
                processError = exception.what();
                processed = false;
              }
              if (!processed && processError.empty() &&
                  runtime->getLastProcessingFailureNodeId() != 0)
                processError =
                    "muted at node " +
                    std::to_string(runtime->getLastProcessingFailureNodeId());
              passed &=
                  expect(processed, processError.empty()
                                        ? "Tiny-RAVE from library must process"
                                        : processError.c_str());
            }
          }
        }
      }
    }
  }

  passed &= testUserLibraryRaveBox("Small-RAVE", options);

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph decayGraph;
    const auto audioIn = decayGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto effect =
        decayGraph.addNode(NodeType::expDecayReverb, {180.0f, 0.0f});
    const auto audioOut =
        decayGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(decayGraph.setProperty(effect, "reverb_length", 64),
                     "ExpDecayReverb length is editable");
    passed &= expect(decayGraph.setProperty(effect, "add_dry", 0),
                     "ExpDecayReverb add_dry is editable");
    const auto *inNode = decayGraph.findNode(audioIn);
    const auto *fxNode = decayGraph.findNode(effect);
    const auto *outNode = decayGraph.findNode(audioOut);
    passed &= expect(inNode != nullptr && fxNode != nullptr && outNode != nullptr &&
                         decayGraph
                             .connect(inNode->outputs.front().id,
                                      fxNode->inputs.front().id)
                             .accepted &&
                         decayGraph
                             .connect(fxNode->outputs.front().id,
                                      outNode->inputs.front().id)
                             .accepted,
                     "ExpDecayReverb chain connects");
    const auto compiled = LiveGraphEngine::compile(decayGraph, options);
    if (!compiled.succeeded())
      std::cerr << "ExpDecayReverb compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "ExpDecayReverb compiles");
    if (compiled.succeeded()) {
      LiveGraphCompileError prepareError;
      const auto runtime =
          LiveGraphEngine::prepare(compiled.snapshot, prepareError);
      passed &= expect(runtime != nullptr, "ExpDecayReverb prepares");
      if (runtime != nullptr) {
        const auto wet = runtime->processTensor(
            torch::ones({1, 2, 32}, torch::kFloat32));
        passed &= expect(wet.defined() && wet.size(1) == 2 && wet.size(2) == 32,
                         "ExpDecayReverb emits stereo audio");
        passed &= expect(wet.abs().sum().item<float>() > 0.0f,
                         "ExpDecayReverb is audible");
      }
    }
    decayGraph.setFloatProperty(effect, "decay", 6.0f);
    const auto longer = LiveGraphEngine::compile(decayGraph, options);
    passed &= expect(longer.succeeded(), "ExpDecayReverb recompiles after decay");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph firGraph;
    const auto audioIn = firGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto fir = firGraph.addNode(NodeType::firFilter, {180.0f, 0.0f});
    const auto audioOut = firGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(firGraph.setProperty(fir, "n_frames", 4),
                     "FIRFilter frames editable");
    passed &= expect(!firGraph.setProperty(fir, "n_filter_banks", 0),
                     "FIRFilter refuses zero filter banks");
    const auto *inNode = firGraph.findNode(audioIn);
    const auto *firNode = firGraph.findNode(fir);
    const auto *outNode = firGraph.findNode(audioOut);
    passed &= expect(inNode != nullptr && firNode != nullptr && outNode != nullptr &&
                         firGraph
                             .connect(inNode->outputs.front().id,
                                      firNode->inputs.front().id)
                             .accepted &&
                         firGraph
                             .connect(firNode->outputs.front().id,
                                      outNode->inputs.front().id)
                             .accepted,
                     "FIRFilter chain connects");
    const auto compiled = LiveGraphEngine::compile(firGraph, options);
    if (!compiled.succeeded())
      std::cerr << "FIRFilter compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "FIRFilter compiles");
    if (compiled.succeeded()) {
      LiveGraphCompileError prepareError;
      const auto runtime =
          LiveGraphEngine::prepare(compiled.snapshot, prepareError);
      if (runtime != nullptr) {
        const auto out = runtime->processTensor(
            torch::randn({1, 2, 32}, torch::kFloat32));
        passed &= expect(out.defined() && out.size(2) == 32,
                         "FIRFilter preserves time");
      }
    }
    const auto noise =
        firGraph.addNode(NodeType::filteredNoiseReverb, {260.0f, 40.0f});
    passed &= expect(firGraph.setProperty(noise, "reverb_length", 128),
                     "FilteredNoiseReverb length editable");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph delayGraph;
    const auto audioIn = delayGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto delay = delayGraph.addNode(NodeType::modDelay, {180.0f, 0.0f});
    const auto audioOut =
        delayGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(!delayGraph.setFloatProperty(delay, "depth_ms", 40.0f),
                     "ModDelay refuses depth greater than center");
    passed &= expect(delayGraph.setFloatProperty(delay, "depth_ms", 5.0f),
                     "ModDelay accepts legal depth");
    const auto *inNode = delayGraph.findNode(audioIn);
    const auto *delayNode = delayGraph.findNode(delay);
    const auto *outNode = delayGraph.findNode(audioOut);
    passed &= expect(inNode != nullptr && delayNode != nullptr &&
                         outNode != nullptr &&
                         delayGraph
                             .connect(inNode->outputs.front().id,
                                      delayNode->inputs.front().id)
                             .accepted &&
                         delayGraph
                             .connect(delayNode->outputs.front().id,
                                      outNode->inputs.front().id)
                             .accepted,
                     "ModDelay chain connects");
    const auto compiled = LiveGraphEngine::compile(delayGraph, options);
    if (!compiled.succeeded())
      std::cerr << "ModDelay compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "ModDelay compiles");
    if (compiled.succeeded()) {
      LiveGraphCompileError prepareError;
      const auto runtime =
          LiveGraphEngine::prepare(compiled.snapshot, prepareError);
      if (runtime != nullptr) {
        const auto impulse = torch::zeros({1, 2, 64}, torch::kFloat32);
        impulse[0][0][0] = 1.0f;
        const auto out = runtime->processTensor(impulse);
        passed &= expect(out.abs().sum().item<float>() > 0.0f,
                         "ModDelay produces output");
      }
    }
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph reverbGraph;
    const auto audioIn = reverbGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto reverb = reverbGraph.addNode(NodeType::reverb, {180.0f, 0.0f});
    const auto audioOut =
        reverbGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(reverbGraph.setProperty(reverb, "reverb_length", 48),
                     "Reverb length editable");
    const auto *inNode = reverbGraph.findNode(audioIn);
    const auto *revNode = reverbGraph.findNode(reverb);
    const auto *outNode = reverbGraph.findNode(audioOut);
    passed &= expect(revNode != nullptr && revNode->inputs.size() == 2,
                     "Reverb has optional IR pin");
    passed &= expect(inNode != nullptr && revNode != nullptr && outNode != nullptr &&
                         reverbGraph
                             .connect(inNode->outputs.front().id,
                                      revNode->inputs.front().id)
                             .accepted &&
                         reverbGraph
                             .connect(revNode->outputs.front().id,
                                      outNode->inputs.front().id)
                             .accepted,
                     "Reverb chain connects");
    const auto compiled = LiveGraphEngine::compile(reverbGraph, options);
    if (!compiled.succeeded())
      std::cerr << "Reverb compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "Reverb compiles");
    reverbGraph.setProperty(reverb, "reverb_length", 96000);
    const auto warning = reverbGraph.graphWarningMessage();
    passed &= expect(warning.find("live-safe") != std::string::npos,
                     "Long reverb length warns without clamping");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph rnnGraph;
    const auto audioIn = rnnGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto rnn = rnnGraph.addNode(NodeType::rnn, {180.0f, 0.0f});
    const auto lstm = rnnGraph.addNode(NodeType::lstm, {300.0f, 0.0f});
    const auto linear = rnnGraph.addNode(NodeType::linear, {390.0f, 0.0f});
    const auto audioOut = rnnGraph.addNode(NodeType::audioOutput, {480.0f, 0.0f});
    passed &= expect(rnnGraph.setProperty(rnn, "hidden_size", 4),
                     "RNN hidden size editable");
    passed &= expect(rnnGraph.setProperty(lstm, "hidden_size", 4),
                     "LSTM hidden size editable");
    passed &= expect(rnnGraph.setProperty(lstm, "bidirectional", 1),
                     "LSTM bidirectional editable");
    passed &= expect(rnnGraph.setProperty(lstm, "bias", 0),
                     "LSTM bias is a 0/1 flag");
    passed &= expect(rnnGraph.setFloatProperty(rnn, "leak_rate", 0.5f),
                     "RNN leak rate editable");
    passed &= expect(!rnnGraph.setFloatProperty(rnn, "leak_rate", 1.5f),
                     "RNN leak rate refuses out of range");
    passed &= expect(rnnGraph.setFloatProperty(lstm, "recurrent_weight_scale",
                                              2.0f),
                     "LSTM recurrent weight scale editable");
    passed &= expect(!rnnGraph.setFloatProperty(lstm, "recurrent_weight_scale",
                                               11.0f),
                     "LSTM recurrent weight scale refuses out of range");
    passed &= expect(rnnGraph.setProperty(linear, "features", 2),
                     "Linear maps recurrent channels to host stereo");
    const auto *inNode = rnnGraph.findNode(audioIn);
    const auto *rnnNode = rnnGraph.findNode(rnn);
    const auto *lstmNode = rnnGraph.findNode(lstm);
    const auto *linearNode = rnnGraph.findNode(linear);
    const auto *outNode = rnnGraph.findNode(audioOut);
    passed &= expect(inNode != nullptr && rnnNode != nullptr &&
                         lstmNode != nullptr && linearNode != nullptr &&
                         outNode != nullptr &&
                         rnnGraph
                             .connect(inNode->outputs.front().id,
                                      rnnNode->inputs.front().id)
                             .accepted &&
                         rnnGraph
                             .connect(rnnNode->outputs.front().id,
                                      lstmNode->inputs.front().id)
                             .accepted &&
                         rnnGraph
                             .connect(lstmNode->outputs.front().id,
                                      linearNode->inputs.front().id)
                             .accepted &&
                         rnnGraph
                             .connect(linearNode->outputs.front().id,
                                      outNode->inputs.front().id)
                             .accepted,
                     "RNN then LSTM chain connects");
    passed &= expect(lstmNode->outputs.front().shape.channels == 8,
                     "Bidirectional LSTM doubles hidden channels");
    const auto compiled = LiveGraphEngine::compile(rnnGraph, options);
    if (!compiled.succeeded())
      std::cerr << "RNN/LSTM compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "RNN/LSTM compiles");
    if (compiled.succeeded()) {
      LiveGraphCompileError prepareError;
      const auto runtime =
          LiveGraphEngine::prepare(compiled.snapshot, prepareError);
      passed &= expect(runtime != nullptr, "RNN/LSTM prepares");
      if (runtime != nullptr) {
        const auto first = runtime->processTensor(
            torch::ones({1, 2, 16}, torch::kFloat32));
        const auto second = runtime->processTensor(
            torch::ones({1, 2, 16}, torch::kFloat32));
        passed &= expect(first.defined() && first.size(1) == 2,
                         "Stacked recurrent output matches host");
        runtime->reset();
        const auto afterReset = runtime->processTensor(
            torch::ones({1, 2, 16}, torch::kFloat32));
        passed &= expect(afterReset.defined(),
                         "Recurrent state resets without stopping");
        (void)second;
      }
    }
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph original;
    const auto reverb = original.addNode(NodeType::reverb, {0.0f, 0.0f});
    const auto decay = original.addNode(NodeType::expDecayReverb, {80.0f, 0.0f});
    const auto noise =
        original.addNode(NodeType::filteredNoiseReverb, {160.0f, 0.0f});
    const auto fir = original.addNode(NodeType::firFilter, {240.0f, 0.0f});
    const auto delay = original.addNode(NodeType::modDelay, {320.0f, 0.0f});
    const auto lstm = original.addNode(NodeType::lstm, {400.0f, 0.0f});
    const auto rnn = original.addNode(NodeType::rnn, {480.0f, 0.0f});
    passed &= expect(original.setProperty(reverb, "reverb_length", 64),
                     "persist Reverb length");
    passed &= expect(original.setProperty(lstm, "hidden_size", 5),
                     "persist LSTM hidden size");
    passed &= expect(original.setProperty(lstm, "bidirectional", 1),
                     "persist LSTM bidirectional");
    passed &= expect(original.setFloatProperty(lstm, "leak_rate", 0.25f),
                     "persist LSTM leak rate");
    passed &= expect(original.setFloatProperty(rnn, "recurrent_weight_scale",
                                               3.0f),
                     "persist RNN recurrent weight scale");
    const auto tree = original.toValueTree();
    openyourbox::graph::NodeGraph restored;
    passed &= expect(restored.restoreFromValueTree(tree),
                     "DDSP and recurrent types restore from ValueTree");
    const auto *restoredReverb = restored.findNode(reverb);
    const auto *restoredDecay = restored.findNode(decay);
    const auto *restoredNoise = restored.findNode(noise);
    const auto *restoredFir = restored.findNode(fir);
    const auto *restoredDelay = restored.findNode(delay);
    const auto *restoredLstm = restored.findNode(lstm);
    const auto *restoredRnn = restored.findNode(rnn);
    passed &= expect(restoredReverb != nullptr &&
                         restoredReverb->type == NodeType::reverb &&
                         restoredDecay != nullptr &&
                         restoredDecay->type == NodeType::expDecayReverb &&
                         restoredNoise != nullptr &&
                         restoredNoise->type == NodeType::filteredNoiseReverb &&
                         restoredFir != nullptr &&
                         restoredFir->type == NodeType::firFilter &&
                         restoredDelay != nullptr &&
                         restoredDelay->type == NodeType::modDelay &&
                         restoredLstm != nullptr &&
                         restoredLstm->type == NodeType::lstm &&
                         restoredRnn != nullptr &&
                         restoredRnn->type == NodeType::rnn,
                     "save/reload restores all seven DDSP and recurrent types");
    passed &= expect(restoredLstm != nullptr &&
                         restoredLstm->outputs.front().shape.channels == 10,
                     "restored bidirectional LSTM keeps doubled channels");
    auto hasFloat = [](const openyourbox::graph::GraphNode *node,
                       const char *key, float expected) {
      if (node == nullptr)
        return false;
      for (const auto &property : node->properties) {
        if (property.key == key)
          return std::abs(property.floatValue - expected) < 1.0e-5f;
      }
      return false;
    };
    passed &= expect(hasFloat(restoredLstm, "leak_rate", 0.25f),
                     "restored LSTM keeps leak rate");
    passed &= expect(hasFloat(restoredRnn, "recurrent_weight_scale", 3.0f),
                     "restored RNN keeps recurrent weight scale");
  }

  {
    using openyourbox::dsp::LiveGraphCompileError;
    using openyourbox::dsp::LiveGraphEngine;
    using openyourbox::graph::BlackBoxOrigin;
    using openyourbox::graph::ExternalLoadStatus;
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto load = graph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *loadNode = graph.findNode(load);
    passed &= expect(loadNode != nullptr &&
                         loadNode->blackBoxOrigin ==
                             BlackBoxOrigin::externalLoad &&
                         loadNode->inputs.size() == 1 &&
                         loadNode->outputs.size() == 1,
                     "TorchScript Load places audio-only Gold pins");
    passed &= expect(!graph.canUnfreeze(load),
                     "external load cannot Unfreeze");
    passed &= expect(
        graph
            .connect(graph.findNode(input)->outputs.front().id,
                     graph.findNode(load)->inputs.front().id)
            .accepted &&
            graph
                .connect(graph.findNode(load)->outputs.front().id,
                         graph.findNode(output)->inputs.front().id)
                .accepted,
        "empty TorchScript Load accepts dry-passthrough cables");
    const auto compiled = LiveGraphEngine::compile(
        graph, options,
        [](const openyourbox::graph::GraphNode &) { return nullptr; });
    if (!compiled.succeeded())
      std::cerr << "empty external load: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(),
                     "empty external load must compile as passthrough");
    LiveGraphCompileError prepareError;
    const auto runtime = LiveGraphEngine::prepare(compiled.snapshot, prepareError);
    passed &= expect(runtime != nullptr, "empty external load must prepare");
    if (runtime != nullptr) {
      const auto inputTensor = torch::randn({1, 2, 32}, torch::kFloat32);
      const auto outputTensor = runtime->processTensor(inputTensor);
      passed &= expect(outputTensor.defined() &&
                           torch::allclose(outputTensor, inputTensor),
                       "empty external load dry-passthroughs audio");
    }

    graph.applyExternalCheckpointError(load, "invalid checkpoint");
    const auto silenced = LiveGraphEngine::compile(
        graph, options,
        [](const openyourbox::graph::GraphNode &) { return nullptr; });
    passed &= expect(silenced.succeeded(),
                     "errored external load with no factory must compile");
    const auto silentRuntime =
        LiveGraphEngine::prepare(silenced.snapshot, prepareError);
    passed &= expect(silentRuntime != nullptr, "error silence must prepare");
    if (silentRuntime != nullptr) {
      const auto inputTensor = torch::ones({1, 2, 32}, torch::kFloat32);
      const auto outputTensor = silentRuntime->processTensor(inputTensor);
      passed &= expect(outputTensor.defined() &&
                           torch::count_nonzero(outputTensor).item<int64_t>() == 0,
                       "error with no prior factory outputs silence");
    }

    const auto factory = std::make_shared<TestFrozenFactory>();
    graph.applyExternalCheckpointReady(load, "test-external.pt", 2, 2, 0, false,
                                       false, false, {}, {}, {}, "");
    const auto readyCompiled = LiveGraphEngine::compile(
        graph, options,
        [factory](const openyourbox::graph::GraphNode &) { return factory; });
    passed &= expect(readyCompiled.succeeded(),
                     "ready external load compiles with a fixture factory");
    const auto readyRuntime =
        LiveGraphEngine::prepare(readyCompiled.snapshot, prepareError);
    passed &= expect(readyRuntime != nullptr, "ready external load prepares");
    if (readyRuntime != nullptr) {
      const auto inputTensor = torch::randn({1, 2, 32}, torch::kFloat32);
      const auto outputTensor = readyRuntime->processTensor(inputTensor);
      passed &= expect(outputTensor.defined() &&
                           !torch::equal(outputTensor, inputTensor),
                       "ready external load forwards through the factory");
    }

    graph.applyExternalCheckpointError(load, "reload failed");
    passed &= expect(graph.findNode(load)->runtimeArtifactPath ==
                         "test-external.pt",
                     "failed reload retains the prior runtime path");
    const auto retained = LiveGraphEngine::compile(
        graph, options, [factory](const openyourbox::graph::GraphNode &node) {
          return node.runtimeArtifactPath.empty() ? nullptr : factory;
        });
    passed &= expect(retained.succeeded(),
                     "failed reload with prior factory must keep compiling");

    passed &= expect(!graph.setExternalChannelOverride(load, "input", 8),
                     "override that breaks a stereo cable is refused");
    passed &= expect(graph.resetExternalChannelOverrides(load),
                     "reset restores inferred channel overrides");

    const auto tree = graph.toValueTree();
    openyourbox::graph::NodeGraph restored;
    passed &= expect(restored.restoreFromValueTree(tree),
                     "external_load origin round-trips through ValueTree");
    const auto *restoredNode = restored.findNode(load);
    passed &= expect(restoredNode != nullptr &&
                         restoredNode->blackBoxOrigin ==
                             BlackBoxOrigin::externalLoad &&
                         restoredNode->artifactPath == "test-external.pt",
                     "restored node keeps external_load path");

    class TestEncodeDecodeFactory final
        : public openyourbox::dsp::FrozenBlackBoxFactory {
    public:
      int getInputChannels() const noexcept override { return 2; }
      int getOutputChannels() const noexcept override { return 2; }
      std::uint64_t getReceptiveField() const noexcept override { return 1; }
      std::uint64_t getParameterCount() const noexcept override { return 0; }
      bool preservesSilence() const noexcept override { return true; }
      bool hasEncodeDecode() const noexcept override { return true; }
      int getLatentChannels() const noexcept override { return 2; }
      std::unique_ptr<openyourbox::dsp::FrozenBlackBoxKernel>
      createKernel() const override {
        return std::make_unique<TestFrozenKernel>();
      }
    };
    graph.applyExternalCheckpointReady(load, "test-rave.pt", 2, 2, 2, true,
                                       true, false, {}, {}, {}, "");
    const auto *rave = graph.findNode(load);
    passed &= expect(rave != nullptr && rave->inputs.size() >= 2 &&
                         rave->outputs.size() >= 2,
                     "encode/decode load morphs bias/scale and latent-out pins");
    bool hasControl = false;
    bool hasLatentIn = false;
    bool hasBias = false;
    bool hasScale = false;
    bool hasLatentOut = false;
    for (const auto &pin : rave->inputs) {
      if (openyourbox::graph::isControlInputPin(pin))
        hasControl = true;
      if (openyourbox::graph::isLatentPin(pin))
        hasLatentIn = true;
      if (openyourbox::graph::isBiasPin(pin))
        hasBias = true;
      if (openyourbox::graph::isScalePin(pin))
        hasScale = true;
    }
    for (const auto &pin : rave->outputs) {
      if (openyourbox::graph::isLatentPin(pin))
        hasLatentOut = true;
    }
    passed &= expect(hasControl && hasBias && hasScale && hasLatentOut &&
                         !hasLatentIn,
                     "conditioned encode/decode load exposes Control, bias, scale, latent out");
    const auto encodeFactory = std::make_shared<TestEncodeDecodeFactory>();
    const auto encodeCompiled = LiveGraphEngine::compile(
        graph, options,
        [encodeFactory](const openyourbox::graph::GraphNode &) {
          return encodeFactory;
        });
    if (!encodeCompiled.succeeded())
      std::cerr << "encode fixture: " << encodeCompiled.error.message << '\n';
    passed &= expect(encodeCompiled.succeeded(),
                     "encode/decode fixture compiles");

    graph.clearExternalCheckpoint(load);
    const auto *cleared = graph.findNode(load);
    passed &= expect(cleared != nullptr &&
                         cleared->externalLoadStatus ==
                             ExternalLoadStatus::empty &&
                         cleared->inputs.size() == 1 &&
                         cleared->outputs.size() == 1,
                     "Clear returns to audio-only empty passthrough");
  }

  {
    using openyourbox::dsp::LiveGraphCompileError;
    using openyourbox::dsp::LiveGraphEngine;
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph latentGraph;
    const auto input = latentGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto load = latentGraph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto output = latentGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto knob = latentGraph.addNode(NodeType::knobInput, {180.0f, 80.0f});
    const auto linear = latentGraph.addNode(NodeType::linear, {270.0f, 80.0f});
    passed &= expect(latentGraph.setProperty(linear, "features", 8),
                     "Linear features match latent width");
    passed &= expect(
        latentGraph.applyExternalCheckpointReady(load, "test-rave-latent.pt", 2,
                                                2, 8, true, false, false, {}, {},
                                                {}, ""),
        "encode/decode load with 8-ch latent must succeed");
    const auto *loadNode = latentGraph.findNode(load);
    const auto *linearNode = latentGraph.findNode(linear);
    const auto *knobNode = latentGraph.findNode(knob);
    std::int32_t biasIn = 0;
    if (loadNode != nullptr) {
      for (const auto &pin : loadNode->inputs) {
        if (openyourbox::graph::isBiasPin(pin))
          biasIn = pin.id;
      }
    }
    passed &= expect(
        loadNode != nullptr && linearNode != nullptr && knobNode != nullptr &&
            biasIn != 0 &&
            latentGraph
                .connect(latentGraph.findNode(input)->outputs.front().id,
                         loadNode->inputs.front().id)
                .accepted &&
            latentGraph
                .connect(loadNode->outputs.front().id,
                         latentGraph.findNode(output)->inputs.front().id)
                .accepted &&
            latentGraph.connect(knobNode->outputs.front().id,
                                 linearNode->inputs.front().id)
                .accepted &&
            latentGraph.connect(linearNode->outputs.front().id, biasIn)
                .accepted,
        "Knob → Linear → bias must wire beside audio I/O");
    const auto raveFactory = std::make_shared<TestRaveDecodeFactory>();
    const auto latentCompiled = LiveGraphEngine::compile(
        latentGraph, options,
        [raveFactory](const openyourbox::graph::GraphNode &) {
          return raveFactory;
        });
    if (!latentCompiled.succeeded())
      std::cerr << "knob-linear-bias: " << latentCompiled.error.message << '\n';
    passed &= expect(latentCompiled.succeeded(),
                     "Knob → Linear → bias must not steal Audio Output width");
    LiveGraphCompileError prepareError;
    const auto latentRuntime =
        LiveGraphEngine::prepare(latentCompiled.snapshot, prepareError);
    passed &= expect(latentRuntime != nullptr,
                     "bias-steered RAVE graph must prepare");
    if (latentRuntime != nullptr) {
      const auto inputTensor = torch::randn({1, 2, 32}, torch::kFloat32);
      const auto outputTensor = latentRuntime->processTensor(inputTensor);
      passed &= expect(outputTensor.defined() && outputTensor.size(1) == 2,
                       "bias-steered decode still emits stereo audio");
    }

    openyourbox::graph::NodeGraph xyGraph;
    const auto xyInput = xyGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto xyLoad = xyGraph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto xyOutput = xyGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto pad = xyGraph.addNode(NodeType::xyTrackpad, {180.0f, 80.0f});
    const auto xyLinear = xyGraph.addNode(NodeType::linear, {270.0f, 80.0f});
    passed &= expect(xyGraph.setProperty(xyLinear, "features", 8),
                     "XY Linear features match latent width");
    passed &= expect(
        xyGraph.applyExternalCheckpointReady(xyLoad, "test-rave-xy-latent.pt",
                                             2, 2, 8, true, false, false, {}, {},
                                             {}, ""),
        "XY encode/decode load with 8-ch latent must succeed");
    const auto *xyLoadNode = xyGraph.findNode(xyLoad);
    const auto *xyLinearNode = xyGraph.findNode(xyLinear);
    const auto *padNode = xyGraph.findNode(pad);
    std::int32_t xyBiasIn = 0;
    if (xyLoadNode != nullptr) {
      for (const auto &pin : xyLoadNode->inputs) {
        if (openyourbox::graph::isBiasPin(pin))
          xyBiasIn = pin.id;
      }
    }
    passed &= expect(
        xyLoadNode != nullptr && xyLinearNode != nullptr && padNode != nullptr &&
            xyBiasIn != 0 &&
            xyGraph
                .connect(xyGraph.findNode(xyInput)->outputs.front().id,
                         xyLoadNode->inputs.front().id)
                .accepted &&
            xyGraph
                .connect(xyLoadNode->outputs.front().id,
                         xyGraph.findNode(xyOutput)->inputs.front().id)
                .accepted &&
            xyGraph.connect(padNode->outputs.front().id,
                             xyLinearNode->inputs.front().id)
                .accepted &&
            xyGraph.connect(xyLinearNode->outputs.front().id, xyBiasIn)
                .accepted,
        "XY → Linear → bias must wire beside audio I/O");
    const auto xyCompiled = LiveGraphEngine::compile(
        xyGraph, options,
        [raveFactory](const openyourbox::graph::GraphNode &) {
          return raveFactory;
        });
    if (!xyCompiled.succeeded())
      std::cerr << "xy-linear-bias: " << xyCompiled.error.message << '\n';
    passed &= expect(xyCompiled.succeeded(),
                     "XY → Linear → bias must not steal Audio Output width");
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph shapeGraph;
    const auto input = shapeGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto load = shapeGraph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto output = shapeGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto linear = shapeGraph.addNode(NodeType::linear, {180.0f, 120.0f});
    const auto conv = shapeGraph.addNode(NodeType::convolution, {270.0f, 120.0f});
    passed &= expect(
        shapeGraph
            .connect(shapeGraph.findNode(input)->outputs.front().id,
                       shapeGraph.findNode(load)->inputs.front().id)
            .accepted &&
            shapeGraph
                .connect(shapeGraph.findNode(load)->outputs.front().id,
                         shapeGraph.findNode(output)->inputs.front().id)
                .accepted,
        "empty TS load must accept stereo Audio In/Out cables");
    passed &= expect(
        shapeGraph.applyExternalCheckpointReady(load, "mono-rave.pt", 1, 1, 16,
                                                true, false, false, {}, {}, {},
                                                ""),
        "mono encode/decode load must succeed after stereo cables exist");
    passed &= expect(shapeGraph.setProperty(linear, "features", 32),
                     shapeGraph.lastPropertyMessage().empty()
                         ? "disconnected Linear must edit Features beside a TS load"
                         : shapeGraph.lastPropertyMessage().c_str());
    passed &= expect(shapeGraph.setProperty(conv, "channels", 8),
                     shapeGraph.lastPropertyMessage().empty()
                         ? "disconnected Conv1D must edit Channels beside a TS load"
                         : shapeGraph.lastPropertyMessage().c_str());
    auto intProperty = [](const openyourbox::graph::GraphNode *node,
                            const char *key) {
      if (node == nullptr)
        return -1;
      for (const auto &property : node->properties) {
        if (property.key == key)
          return property.value;
      }
      return -1;
    };
    passed &= expect(intProperty(shapeGraph.findNode(linear), "features") == 32,
                     "disconnected Linear Features must stick");
    passed &= expect(intProperty(shapeGraph.findNode(conv), "channels") == 8,
                     "disconnected Conv1D Channels must stick");

    passed &= expect(shapeGraph.setProperty(conv, "channels", 16),
                     shapeGraph.lastPropertyMessage().empty()
                         ? "Conv1D Channels must match the mono RAVE latent width"
                         : shapeGraph.lastPropertyMessage().c_str());
    const auto *loaded = shapeGraph.findNode(load);
    const auto *convNode = shapeGraph.findNode(conv);
    std::int32_t biasIn = 0;
    int latentChannels = 0;
    if (loaded != nullptr) {
      for (const auto &pin : loaded->inputs) {
        if (openyourbox::graph::isBiasPin(pin)) {
          biasIn = pin.id;
          latentChannels = pin.shape.channels;
        }
      }
    }
    const auto latentConnect =
        (convNode != nullptr && biasIn != 0)
            ? shapeGraph.connect(convNode->outputs.front().id, biasIn)
            : openyourbox::graph::ConnectionResult{
                  false, "missing Conv1D or bias pin"};
    passed &= expect(loaded != nullptr && convNode != nullptr &&
                         convNode->outputs.front().shape.channels == 16 &&
                         latentChannels == 16,
                     "Conv1D out and RAVE bias must both declare 16 channels");
    passed &= expect(
        latentConnect.accepted,
        latentConnect.message.empty()
            ? "Conv1D 16ch must connect to bias beside a stereo/mono mismatch"
            : latentConnect.message.c_str());
  }

  {
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph audioPinGraph;
    const auto load =
        audioPinGraph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto conv =
        audioPinGraph.addNode(NodeType::convolution, {0.0f, 120.0f});
    passed &= expect(
        audioPinGraph.applyExternalCheckpointReady(load, "mono-rave-audio.pt", 1,
                                                   1, 16, true, false, false, {},
                                                   {}, {}, ""),
        "mono encode/decode load must succeed for audio-pin refusal");
    passed &= expect(audioPinGraph.setProperty(conv, "channels", 16),
                     "Conv1D Channels must be 16 before the audio-pin check");
    const auto *loadNode = audioPinGraph.findNode(load);
    const auto *convNode = audioPinGraph.findNode(conv);
    const auto audioConnect =
        (loadNode != nullptr && convNode != nullptr)
            ? audioPinGraph.connect(convNode->outputs.front().id,
                                    loadNode->inputs.front().id)
            : openyourbox::graph::ConnectionResult{false, "missing nodes"};
    passed &= expect(!audioConnect.accepted,
                     "Conv1D 16ch must be refused by unused mono RAVE audio in");
  }

  {
    using openyourbox::dsp::LiveGraphCompileError;
    using openyourbox::dsp::LiveGraphEngine;
    using openyourbox::dsp::collectRuntimeControlState;
    using openyourbox::graph::NodeType;
    openyourbox::graph::NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto load = graph.addExternalTorchScriptLoadNode({180.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    passed &= expect(
        graph.applyExternalCheckpointReady(load, "test-prior-mix.pt", 2, 2, 8,
                                           true, false, false, {}, {}, {}, ""),
        "prior-mix RAVE load must succeed");
    const auto *loadNode = graph.findNode(load);
    std::int32_t biasId = 0;
    std::int32_t scaleId = 0;
    bool hasLatentIn = false;
    bool hasPriorMix = false;
    if (loadNode != nullptr) {
      for (const auto &pin : loadNode->inputs) {
        if (openyourbox::graph::isBiasPin(pin))
          biasId = pin.id;
        if (openyourbox::graph::isScalePin(pin))
          scaleId = pin.id;
        if (openyourbox::graph::isLatentPin(pin))
          hasLatentIn = true;
      }
      for (const auto &property : loadNode->properties) {
        if (property.key == "priorMix")
          hasPriorMix = true;
      }
    }
    passed &= expect(biasId != 0 && scaleId != 0 && !hasLatentIn && hasPriorMix,
                     "RAVE surface has priorMix, bias, scale, and no latent in");
    passed &= expect(
        graph
            .connect(graph.findNode(input)->outputs.front().id,
                     loadNode->inputs.front().id)
            .accepted &&
            graph
                .connect(loadNode->outputs.front().id,
                         graph.findNode(output)->inputs.front().id)
            .accepted,
        "prior-mix graph must wire audio I/O");
    const auto conv = graph.addNode(NodeType::convolution, {180.0f, 120.0f});
    passed &= expect(graph.setProperty(conv, "channels", 4),
                     "mismatched Conv1D width for bias refuse");
    const auto *convNode = graph.findNode(conv);
    const auto badBias =
        (convNode != nullptr && biasId != 0)
            ? graph.connect(convNode->outputs.front().id, biasId)
            : openyourbox::graph::ConnectionResult{false, "missing"};
    passed &= expect(!badBias.accepted,
                     "illegal bias channel width must be refused");

    auto probe = std::make_shared<PriorMixProbe>();
    const auto factory = std::make_shared<TestPriorMixFactory>(probe);
    auto compiled = LiveGraphEngine::compile(
        graph, options, [factory](const openyourbox::graph::GraphNode &) {
          return factory;
        });
    if (!compiled.succeeded())
      std::cerr << "prior-mix compile: " << compiled.error.message << '\n';
    passed &= expect(compiled.succeeded(), "prior-mix graph must compile");
    LiveGraphCompileError prepareError;
    auto runtime = LiveGraphEngine::prepare(compiled.snapshot, prepareError);
    passed &= expect(runtime != nullptr, "prior-mix graph must prepare");
    if (runtime != nullptr) {
      const auto loud = torch::full({1, 2, 32}, 0.8f, torch::kFloat32);
      const auto quiet = torch::full({1, 2, 32}, 0.1f, torch::kFloat32);
      probe->encodeCalls.store(0);
      auto controls = std::make_shared<openyourbox::dsp::RuntimeControlState>(
          collectRuntimeControlState(graph));
      runtime->bindControls(controls);
      const auto forwardOut = runtime->processTensor(loud);
      const auto forwardCalls = probe->encodeCalls.load();
      passed &= expect(forwardCalls >= 1, "full forward must encode");
      passed &= expect(forwardOut.defined() &&
                           torch::allclose(forwardOut,
                                           torch::full_like(forwardOut, 0.8f),
                                           1e-4, 1e-4),
                       "priorMix 0 decodes encoder mean");
      passed &= expect(probe->lastDecodeInput.defined() &&
                           torch::allclose(probe->lastDecodeInput.mean(),
                                           torch::tensor(0.8f), 1e-4, 1e-4),
                       "latent used for decode matches encoder mean at mix 0");

      passed &= expect(graph.setFloatProperty(load, "priorMix", 0.5f),
                       "Gold priorMix must be editable");
      controls = std::make_shared<openyourbox::dsp::RuntimeControlState>(
          collectRuntimeControlState(graph));
      runtime->bindControls(controls);
      const auto midOut = runtime->processTensor(loud);
      passed &= expect(
          midOut.defined() &&
              !torch::allclose(midOut, torch::full_like(midOut, 0.8f), 1e-3,
                               1e-3),
          "intermediate priorMix must differ from full forward");

      passed &= expect(graph.setFloatProperty(load, "priorMix", 1.0f),
                       "priorMix 1 must set");
      controls = std::make_shared<openyourbox::dsp::RuntimeControlState>(
          collectRuntimeControlState(graph));
      runtime->bindControls(controls);
      probe->encodeCalls.store(0);
      const auto priorA = runtime->processTensor(loud);
      const auto callsAfterLoud = probe->encodeCalls.load();
      const auto priorB = runtime->processTensor(quiet);
      const auto callsAfterQuiet = probe->encodeCalls.load();
      passed &= expect(callsAfterLoud == 0 && callsAfterQuiet == 0,
                       "full prior must skip encode regardless of audio");
      passed &= expect(priorA.defined() && priorB.defined(),
                       "full prior still decodes audio");
      passed &= expect(probe->lastDecodeInput.defined(),
                       "full prior still publishes decode latents");
    }

    passed &= expect(!graph.findNode(load)->externalHasEncodeDecode ||
                         graph.setFloatProperty(load, "priorMix", 0.0f),
                     "restore priorMix 0");
  }

  {
    using openyourbox::dsp::TorchScriptBlackBoxFactory;
    auto saveModule = [](torch::jit::Module &module, const juce::String &name) {
      const auto file = juce::File::getSpecialLocation(juce::File::tempDirectory)
                            .getChildFile(name);
      file.deleteFile();
      module.save(file.getFullPathName().toStdString());
      return file;
    };

    torch::jit::Module identity("TraceModel");
    identity.define(R"(
def forward(self, x):
    return x
)");
    const auto identityFile = saveModule(identity, "oyb-identity-forward.ts");
    std::string identityError;
    const auto identityFactory = TorchScriptBlackBoxFactory::load(
        identityFile.getFullPathName().toStdString(), 1, 1, identityError,
        false, false, 0);
    passed &= expect(identityFactory != nullptr,
                     identityError.empty()
                         ? "1-arg identity TraceModel must load"
                         : identityError.c_str());
    if (identityFactory != nullptr) {
      passed &= expect(!identityFactory->acceptsConditioning(),
                       "1-arg TraceModel must not grow a Control pin");
      auto kernel = identityFactory->createKernel();
      passed &= expect(kernel != nullptr, "identity kernel must construct");
      if (kernel != nullptr) {
        const auto ones = torch::ones({1, 1, 32}, torch::kFloat32);
        const auto out = kernel->forward(ones);
        passed &= expect(out.defined() && torch::allclose(out, ones),
                         "flexible 1-arg models must not add hop latency");
      }
    }
    identityFile.deleteFile();

    torch::jit::Module hop("TraceModel");
    hop.define(R"(
def forward(self, x):
    return x.view(x.size(0), x.size(1), 2048)
)");
    const auto hopFile =
        saveModule(hop, "guitar_iil_b2048_r48000_z16.ts");
    std::string hopError;
    const auto hopFactory = TorchScriptBlackBoxFactory::load(
        hopFile.getFullPathName().toStdString(), 1, 1, hopError, false, false,
        0);
    if (hopFactory == nullptr)
      std::cerr << "2048-hop TraceModel load: " << hopError << '\n';
    passed &= expect(hopFactory != nullptr,
                     hopError.empty()
                         ? "RAVE-style 2048 TraceModel must load"
                         : hopError.c_str());
    passed &= expect(hopError.find("at most 2 argument") == std::string::npos,
                     "1-arg TraceModel must not be probed with cond");
    if (hopFactory != nullptr) {
      passed &= expect(!hopFactory->acceptsConditioning(),
                       "2048-hop TraceModel is unconditioned");
      auto kernel = hopFactory->createKernel();
      passed &= expect(kernel != nullptr, "2048-hop kernel must construct");
      if (kernel != nullptr) {
        bool blocksOk = true;
        for (int i = 0; i < 8; ++i) {
          const auto block = kernel->forward(
              torch::zeros({1, 1, 256}, torch::kFloat32));
          if (!block.defined() || block.size(2) != 256) {
            blocksOk = false;
            break;
          }
        }
        passed &= expect(blocksOk,
                         "2048-hop kernel must emit host-sized blocks");
      }
    }
    hopFile.deleteFile();

    torch::jit::Module cond("CondModel");
    cond.define(R"(
def forward(self, x, c):
    return x
)");
    const auto condFile = saveModule(cond, "oyb-conditioned-forward.ts");
    std::string condError;
    const auto condFactory = TorchScriptBlackBoxFactory::load(
        condFile.getFullPathName().toStdString(), 1, 1, condError, false, false,
        0);
    passed &= expect(condFactory != nullptr &&
                         condFactory->acceptsConditioning(),
                     condError.empty()
                         ? "2-arg forward must still detect conditioning"
                         : condError.c_str());
    condFile.deleteFile();
  }

  if (passed)
    std::cout << "OpenYourBox live graph engine tests passed\n";
  return passed ? 0 : 1;
}
