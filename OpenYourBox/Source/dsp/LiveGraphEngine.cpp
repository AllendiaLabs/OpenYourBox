#include "LiveGraphEngine.h"

#include "NoiseSynthesizer.h"
#include "PqmfBank.h"
#include "RateConv.h"
#include "TCNModel.h"
#include "VariationalBottleneck.h"

#include <torch/nn/functional.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {
/** @brief Immutable execution information for one topologically ordered node.
 */
struct CompiledElement {
  /** @brief Stable source node identifier. */
  std::int32_t nodeId = 0;
  /** @brief Source graph node type. */
  openyourbox::graph::NodeType type = openyourbox::graph::NodeType::activation;
  /** @brief Indices of upstream elements in topological order. */
  std::vector<std::int64_t> inputIndices;
  /** @brief Index of the sole upstream element, or -1 for Audio Input. */
  std::int64_t inputIndex = -1;
  /** @brief Exact input channels, or zero for Audio Input. */
  int inputChannels = 0;
  /** @brief Exact output channels, or zero for Audio Output. */
  int outputChannels = 0;
  /** @brief Kernel size for Conv1D or TCN elements. */
  int kernelSize = 1;
  /** @brief Dilation for Conv1D elements. */
  int dilation = 1;
  /** @brief Number of internal temporal layers in a TCN element. */
  int depth = 0;
  /** @brief TCN hidden channel count. */
  int hiddenChannels = 0;
  /** @brief Element activation selection. */
  openyourbox::dsp::ActivationType activation =
      openyourbox::dsp::ActivationType::relu;
  /** @brief Immutable bias-free parameter tensors in execution order. */
  std::vector<torch::Tensor> weights;
  /** @brief Frozen artifact metadata and off-thread kernel constructor. */
  std::shared_ptr<const openyourbox::dsp::FrozenBlackBoxFactory> blackBoxFactory;
  /** @brief Element-local receptive field in samples. */
  std::uint64_t receptiveField = 1;
  /** @brief Mutable scalar parameter count owned by this element. */
  std::uint64_t parameterCount = 0;
  /** @brief True for live weighted elements that may be randomized. */
  bool randomizable = false;
  /** @brief Utility operating mode: add, multiply, or concatenate. */
  int mergeMode = 0;
  /** @brief Nonlinearity slope Gain for Activation and TCN elements. */
  float gain = 1.0f;
  /** @brief LeakyReLU negative slope for Activation and TCN elements. */
  float negativeSlope = openyourbox::graph::leakyReluNegativeSlopeDefault;
  /** @brief Compiled Knob Input scalar, overridden by live controls. */
  float conditioningValue = 0.0f;
  /** @brief Compiled XY Trackpad X scalar. */
  float conditioningX = 0.0f;
  /** @brief Compiled XY Trackpad Y scalar. */
  float conditioningY = 0.0f;
  /** @brief True when this element's output is scalar conditioning. */
  bool outputIsConditioning = false;
  /**
   * @brief Per-input extraction: -1 keeps the full tensor, otherwise one channel.
   */
  std::vector<int> inputExtractChannels;
  /** @brief True when the corresponding compiled input is the control/FiLM pin. */
  std::vector<char> inputIsConditioning;
  /** @brief True when the corresponding compiled input is a latent encode/decode tap. */
  std::vector<char> inputUseLatentTap;
  /** @brief TCN dilation growth G (layer n uses base * G^n). */
  int dilationGrowth = 2;
  /** @brief Whether TCN blocks add a residual path. */
  bool residual = false;
  /** @brief Compiled index of the FiLM source, or -1 when disconnected. */
  std::int64_t filmInputIndex = -1;
  /**
   * @brief Channel extracted from the FiLM source, or -1 for the full tensor.
   */
  int filmExtractChannel = -1;
  /** @brief Control width for per-layer FiLM adaptors, or 0 when unwired. */
  int condDim = 0;
  /** @brief Per-layer FiLM Linear weights shaped `[2 * hidden, condDim]`. */
  std::vector<torch::Tensor> filmWeights;
  /** @brief Per-layer FiLM Linear biases shaped `[2 * hidden]`. */
  std::vector<torch::Tensor> filmBiases;
  /** @brief PQMF band count when this element is analysis or synthesis. */
  int nBand = 0;
  /** @brief Integer stride for rateConv. */
  int stride = 1;
  /** @brief 0 = downsample, 1 = upsample. */
  int rateDirection = 0;
  /** @brief Shared PQMF coefficient bank, prepared off the audio thread. */
  std::shared_ptr<openyourbox::dsp::PqmfBank> pqmf;
  /** @brief Bottleneck/Gold fidelity percent. */
  float fidelityPercent = 99.0f;
  /** @brief True when compactness PCA tensors are present. */
  bool compactnessReady = false;
  /** @brief Compactness mean `[latent]`. */
  torch::Tensor latentMean;
  /** @brief Compactness PCA `[latent, latent]`. */
  torch::Tensor latentPca;
  /** @brief Linear singular-value cumulative ratios `[latent]`. */
  torch::Tensor cumulativeVariance;
  /** @brief Host Audio Input/Output channel mode when type is fixed I/O. */
  openyourbox::graph::HostIoMode hostIoMode =
      openyourbox::graph::HostIoMode::stereo;
  /** @brief Prepared Math Expression program (GUI-thread parse). */
  openyourbox::graph::ExpressionAst mathAst;
  /**
   * @brief Compiled source index for each Math Expression pin, or -1.
   *
   * Index 0 is `x1`. Unconnected unused pins stay -1.
   */
  std::vector<std::int64_t> mathPinSources;
  /** @brief Per-pin extract channel for Math Expression inputs, or -1. */
  std::vector<int> mathPinExtract;
};

/**
 * @brief Folds or expands host audio into the declared graph I/O mode.
 * @param host Host tensor shaped `[1, hostChannels, time]`.
 * @param mode Declared Audio Input mode.
 * @return Tensor with the graph pin width for @p mode.
 */
torch::Tensor adaptHostInputToMode(const torch::Tensor &host,
                                   openyourbox::graph::HostIoMode mode) {
  using openyourbox::graph::HostIoMode;
  if (!host.defined() || host.dim() != 3)
    return host;
  if (mode == HostIoMode::stereo) {
    if (host.size(1) == 2)
      return host;
    if (host.size(1) == 1)
      return host.repeat({1, 2, 1});
    return host.narrow(1, 0, 2);
  }
  auto mono = host.size(1) <= 1 ? host : host.mean(1, /*keepdim=*/true);
  if (mode == HostIoMode::mono)
    return mono;
  return mono.repeat({1, 2, 1});
}

/**
 * @brief Expands or folds graph audio to the host output bus width.
 * @param graph Graph tensor shaped `[1, graphChannels, time]`.
 * @param hostChannels Host output channel count (1 or 2).
 * @return Tensor shaped `[1, hostChannels, time]`.
 */
torch::Tensor adaptGraphOutputToHost(const torch::Tensor &graph,
                                     int hostChannels) {
  if (!graph.defined() || graph.dim() != 3 || hostChannels < 1)
    return graph;
  if (graph.size(1) == hostChannels)
    return graph;
  if (hostChannels == 1)
    return graph.size(1) <= 1 ? graph : graph.mean(1, /*keepdim=*/true);
  if (graph.size(1) == 1)
    return graph.repeat({1, hostChannels, 1});
  return graph.narrow(1, 0, hostChannels);
}

/** @brief Pin ownership and direction used during graph validation. */
struct PinOwner {
  /** @brief Stable owner node identifier. */
  std::int32_t nodeId = 0;
  /** @brief Declared pin direction. */
  openyourbox::graph::PinKind kind = openyourbox::graph::PinKind::input;
};

/** @brief Advances a deterministic SplitMix64 generator. */
std::uint64_t splitMix64(std::uint64_t &state) noexcept {
  auto value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

/** @brief Returns one deterministic float in the closed signed unit range. */
float uniformSigned(std::uint64_t &state) noexcept {
  constexpr auto inverse = 1.0 / static_cast<double>(std::uint64_t{1} << 53U);
  const auto unit = static_cast<double>(splitMix64(state) >> 11U) * inverse;
  return static_cast<float>(unit * 2.0 - 1.0);
}

/** @brief Converts a signed user seed to a stable unsigned generator state. */
std::uint64_t seedState(std::int32_t seed) noexcept {
  return static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) ^
         0xa0761d6478bd642fULL;
}

/** @brief Saturating addition used by receptive-field accumulation. */
std::uint64_t saturatedAdd(std::uint64_t left, std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

/** @brief Saturating multiplication used by metadata calculations. */
std::uint64_t saturatedMultiply(std::uint64_t left,
                                std::uint64_t right) noexcept {
  return left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left
             ? std::numeric_limits<std::uint64_t>::max()
             : left * right;
}

/** @brief Finds an integer node property by canonical key. */
bool readProperty(const openyourbox::graph::GraphNode &node, const char *key,
                  int &value) noexcept {
  const auto found =
      std::find_if(node.properties.begin(), node.properties.end(),
                   [key](const openyourbox::graph::NodeProperty &property) {
                     return property.key == key;
                   });
  if (found == node.properties.end())
    return false;
  value = found->value;
  return true;
}

/**
 * @brief Finds a real node property by canonical key.
 * @param node Graph node to inspect.
 * @param key Property key.
 * @param value Receives the stored real value.
 * @return True when the property exists.
 */
bool readFloatProperty(const openyourbox::graph::GraphNode &node, const char *key,
                       float &value) noexcept {
  const auto found =
      std::find_if(node.properties.begin(), node.properties.end(),
                   [key](const openyourbox::graph::NodeProperty &property) {
                     return property.key == key;
                   });
  if (found == node.properties.end())
    return false;
  value = found->kind == openyourbox::graph::PropertyKind::real
              ? found->floatValue
              : static_cast<float>(found->value);
  return true;
}

/**
 * @brief Fills a tensor with signed uniform values scaled by fan-in.
 * @param tensor Writable CPU float tensor.
 * @param state SplitMix64 random state.
 * @param fanIn Denominator used for Xavier-style scaling.
 */
void fillRandomized(torch::Tensor tensor, std::uint64_t &state,
                    std::int64_t fanIn) {
  auto *data = tensor.data_ptr<float>();
  const auto count = tensor.numel();
  const auto scale = static_cast<float>(
      std::sqrt(6.0 / static_cast<double>(std::max<std::int64_t>(1, fanIn))));
  for (std::int64_t index = 0; index < count; ++index)
    data[index] = uniformSigned(state) * scale;
}

/** @brief Creates one deterministic bias-free convolution weight tensor. */
torch::Tensor makeWeight(int outputChannels, int inputChannels, int kernelSize,
                         std::uint64_t &state) {
  auto weight = torch::empty(
      {outputChannels, inputChannels, kernelSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  fillRandomized(weight, state,
                 static_cast<std::int64_t>(inputChannels) *
                     static_cast<std::int64_t>(kernelSize));
  return weight;
}

/**
 * @brief Creates one deterministic Linear weight tensor `[out, in]`.
 * @param outputFeatures Output (FiLM adaptor) width.
 * @param inputFeatures Conditioning width.
 * @param state SplitMix64 random state.
 */
torch::Tensor makeLinearWeight(int outputFeatures, int inputFeatures,
                               std::uint64_t &state) {
  auto weight = torch::empty(
      {outputFeatures, inputFeatures},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  fillRandomized(weight, state, static_cast<std::int64_t>(inputFeatures));
  return weight;
}

/**
 * @brief Creates one deterministic Linear bias tensor.
 * @param features Bias width.
 * @param state SplitMix64 random state.
 */
torch::Tensor makeBias(int features, std::uint64_t &state) {
  auto bias = torch::empty(
      {features},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  fillRandomized(bias, state, static_cast<std::int64_t>(features));
  return bias;
}

/** @brief Rebuilds all immutable parameters for one weighted element. */
void randomizeElementWeights(CompiledElement &element, std::int32_t seed) {
  auto state = seedState(seed);
  element.weights.clear();
  element.filmWeights.clear();
  element.filmBiases.clear();

  switch (element.type) {
  case openyourbox::graph::NodeType::linear:
    element.weights.push_back(
        makeWeight(element.outputChannels, element.inputChannels, 1, state));
    break;
  case openyourbox::graph::NodeType::convolution:
    element.weights.push_back(makeWeight(element.outputChannels,
                                         element.inputChannels,
                                         element.kernelSize, state));
    break;
  case openyourbox::graph::NodeType::tcn:
    element.weights.push_back(
        makeWeight(element.hiddenChannels, element.inputChannels, 1, state));
    for (int layer = 0; layer < element.depth; ++layer)
      element.weights.push_back(makeWeight(element.hiddenChannels,
                                           element.hiddenChannels,
                                           element.kernelSize, state));
    element.weights.push_back(
        makeWeight(element.outputChannels, element.hiddenChannels, 1, state));
    if (element.condDim > 0) {
      const auto filmOut = element.hiddenChannels * 2;
      for (int layer = 0; layer < element.depth; ++layer) {
        element.filmWeights.push_back(
            makeLinearWeight(filmOut, element.condDim, state));
        element.filmBiases.push_back(makeBias(filmOut, state));
      }
    }
    break;
  case openyourbox::graph::NodeType::rateConv:
    element.weights.push_back(makeWeight(element.outputChannels,
                                         element.inputChannels,
                                         element.kernelSize, state));
    break;
  case openyourbox::graph::NodeType::convTranspose:
    element.weights.push_back(makeWeight(element.outputChannels,
                                         element.inputChannels,
                                         element.kernelSize, state));
    break;
  case openyourbox::graph::NodeType::batchNorm: {
    const auto channels = std::max(1, element.inputChannels);
    element.weights.push_back(torch::ones({channels}));
    element.weights.push_back(torch::zeros({channels}));
    element.weights.push_back(torch::zeros({channels}));
    element.weights.push_back(torch::ones({channels}));
    break;
  }
  case openyourbox::graph::NodeType::variationalBottleneck: {
    const auto latent = std::max(1, element.outputChannels);
    const auto groupedIn =
        std::max(1, element.inputChannels /
                         openyourbox::graph::variationalBottleneckGroups);
    const auto kernel = std::max(1, element.kernelSize);
    element.weights.push_back(
        makeWeight(2 * latent, groupedIn, kernel, state));
    break;
  }
  case openyourbox::graph::NodeType::noiseSynthesizer:
    element.weights.push_back(makeWeight(
        std::max(1, element.kernelSize), element.inputChannels, 1, state));
    break;
  default:
    throw std::invalid_argument("Element does not own randomizable weights");
  }
}

/**
 * @brief Applies a configured zero-preserving activation.
 * @param value Pre-activation tensor.
 * @param activation Selected nonlinearity.
 * @param gain Pre-nonlinearity slope Gain.
 * @param negativeSlope LeakyReLU negative slope when that function is selected.
 */
torch::Tensor applyActivation(torch::Tensor value,
                              openyourbox::dsp::ActivationType activation,
                              float gain, float negativeSlope = 0.01f) {
  if (std::abs(gain - 1.0f) > 1.0e-6f)
    value = value * gain;
  switch (activation) {
  case openyourbox::dsp::ActivationType::relu:
    return torch::relu(value);
  case openyourbox::dsp::ActivationType::sigmoid:
    return torch::where(value == 0.0, torch::zeros_like(value),
                        torch::sigmoid(value));
  case openyourbox::dsp::ActivationType::tanh:
    return torch::tanh(value);
  case openyourbox::dsp::ActivationType::leakyRelu:
    return torch::leaky_relu(value, negativeSlope);
  case openyourbox::dsp::ActivationType::prelu:
    return torch::prelu(value, torch::full({1}, 0.25f, value.options()));
  }
  return value;
}

/** @brief Executes one bias-free valid Conv1D operation. */
torch::Tensor convolve(const torch::Tensor &input, const torch::Tensor &weight,
                       std::int64_t dilation) {
  const std::array<std::int64_t, 1> stride{1};
  const std::array<std::int64_t, 1> padding{0};
  const std::array<std::int64_t, 1> dilationValues{dilation};
  return torch::conv1d(input, weight, std::optional<torch::Tensor>{}, stride,
                       padding, dilationValues, 1);
}

/** @brief Executes a bias-free channel projection. */
torch::Tensor project(const torch::Tensor &input, const torch::Tensor &weight) {
  return convolve(input, weight, 1);
}

/** @brief Prepends retained causal input and updates it for the next block. */
torch::Tensor extendCausalInput(const torch::Tensor &input,
                                torch::Tensor &history,
                                std::int64_t historyLength) {
  if (historyLength <= 0)
    return input;

  auto extended = torch::cat({history, input}, 2);
  history = extended.narrow(2, extended.size(2) - historyLength, historyLength)
                .clone();
  return extended;
}

/**
 * @brief Prepends retained FiLM control samples so the TCN RF sees the true
 * trajectory.
 *
 * Repeating the first sample of the current block restates the whole receptive
 * field at buffer rate, which is heard as a TV-like buzz when Knob/XY moves
 * quickly. The first call after prepare or reset initialises history by
 * repeating the first frame so a static control has no startup step.
 *
 * @param control Current-block control tensor, typically `[1, condDim, T]`.
 * @param history Mutable causal control prefix of length @p historyLength.
 * @param historyLength Samples to retain, matching the paired audio RF prefix.
 * @return Control tensor whose time dimension is `historyLength + T`.
 */
torch::Tensor extendCausalControl(const torch::Tensor &control,
                                  torch::Tensor &history,
                                  std::int64_t historyLength) {
  if (!control.defined() || historyLength <= 0)
    return control;
  auto timed = control;
  if (timed.dim() == 1)
    timed = timed.unsqueeze(0).unsqueeze(-1);
  else if (timed.dim() == 2)
    timed = timed.unsqueeze(-1);
  if (timed.dim() != 3)
    return control;

  const auto needsInit = !history.defined() || history.dim() != 3 ||
                         history.size(0) != timed.size(0) ||
                         history.size(1) != timed.size(1) ||
                         history.size(2) != historyLength;
  if (needsInit) {
    history = timed.narrow(2, 0, 1)
                  .expand({timed.size(0), timed.size(1), historyLength})
                  .contiguous();
  }
  auto extended = torch::cat({history, timed}, 2);
  history = extended.narrow(2, extended.size(2) - historyLength, historyLength)
                .clone();
  return extended;
}

/** @brief Returns whether a graph node has the required port layout. */
bool hasValidPortLayout(const openyourbox::graph::GraphNode &node) noexcept {
  using openyourbox::graph::NodeType;
  if (node.type == NodeType::audioInput)
    return node.inputs.empty() && node.outputs.size() == 1;
  if (node.type == NodeType::audioOutput)
    return node.inputs.size() == 1 && node.outputs.empty();
  if (node.type == NodeType::knobInput)
    return node.inputs.empty() && node.outputs.size() == 1;
  if (node.type == NodeType::xyTrackpad)
    return node.inputs.empty() && node.outputs.size() == 2;
  if (openyourbox::graph::isMixerType(node.type) ||
      openyourbox::graph::isMathExpressionType(node.type))
    return node.inputs.size() >= 1 && node.outputs.size() == 1;
  if (node.type == NodeType::tcn)
    return node.inputs.size() >= 1 && node.inputs.size() <= 2 &&
           node.outputs.size() == 1;
  if (node.type == NodeType::blackBox)
    return node.inputs.size() >= 1 && node.outputs.size() >= 1;
  if (node.type == NodeType::pqmfAnalysis ||
      node.type == NodeType::pqmfSynthesis ||
      node.type == NodeType::rateConv ||
      node.type == NodeType::convTranspose ||
      node.type == NodeType::variationalBottleneck ||
      node.type == NodeType::noiseSynthesizer ||
      node.type == NodeType::batchNorm)
    return node.inputs.size() == 1 && node.outputs.size() == 1;
  return node.inputs.size() == 1 && node.outputs.size() == 1;
}

/**
 * @brief Collects compiled upstream indices for pins that already have a source.
 * @param pins Input pins of the node or frozen-group source.
 * @param sourceNodeByDestinationPin Destination pin to source node map.
 * @param compiledIndex Compiled node identifier to element index map.
 * @return Topological indices of connected, already-compiled sources.
 */
std::vector<std::int64_t> collectCompiledInputs(
    const std::vector<openyourbox::graph::Pin> &pins,
    const std::unordered_map<std::int32_t, std::int32_t>
        &sourceNodeByDestinationPin,
    const std::unordered_map<std::int32_t, std::size_t> &compiledIndex) {
  std::vector<std::int64_t> indices;
  indices.reserve(pins.size());
  for (const auto &pin : pins) {
    const auto source = sourceNodeByDestinationPin.find(pin.id);
    if (source == sourceNodeByDestinationPin.end())
      continue;
    const auto compiledSource = compiledIndex.find(source->second);
    if (compiledSource == compiledIndex.end())
      continue;
    indices.push_back(static_cast<std::int64_t>(compiledSource->second));
  }
  return indices;
}

/**
 * @brief Returns the output-pin index of a source node, or -1.
 * @param node Source graph node.
 * @param pinId Source pin identifier.
 * @return Zero-based output index, or -1 when not found.
 */
int outputPinIndex(const openyourbox::graph::GraphNode &node,
                   std::int32_t pinId) noexcept {
  for (int index = 0; index < static_cast<int>(node.outputs.size()); ++index) {
    if (node.outputs[static_cast<std::size_t>(index)].id == pinId)
      return index;
  }
  return -1;
}

/**
 * @brief Broadcasts a tensor to a target channel count when it is 1-wide.
 * @param value Source tensor shaped [1, C, T].
 * @param channels Desired channel count.
 * @return Broadcast or original tensor.
 */
torch::Tensor broadcastChannels(const torch::Tensor &value, int channels) {
  if (!value.defined() || value.size(1) == channels)
    return value;
  if (value.size(1) == 1 && channels > 1)
    return value.expand({value.size(0), channels, value.size(2)}).contiguous();
  return value;
}

/**
 * @brief Returns whether two channel counts can share an add/multiply Utility.
 * @param left First connected width, or zero when unknown.
 * @param right Second connected width, or zero when unknown.
 * @return True when the widths match or either side can broadcast from 1.
 */
bool channelsAreBroadcastCompatible(int left, int right) noexcept {
  return left == 0 || right == 0 || left == right || left == 1 || right == 1;
}

/**
 * @brief Channel count contributed by one compiled Utility input.
 * @param source Upstream compiled element.
 * @param extractChannel Extracted source channel, or -1 for the full tensor.
 * @return Positive channel count used at the Utility input.
 */
int compiledSlotChannels(const CompiledElement &source,
                         int extractChannel) noexcept {
  if (extractChannel >= 0)
    return 1;
  return source.outputChannels;
}

/**
 * @brief Extract index for the compiled audio/main input of one element.
 * @param element Compiled node whose `inputIndex` is already set.
 * @return Channel extract, or -1 to keep the full upstream tensor.
 */
int compiledExtractForInput(const CompiledElement &element) noexcept {
  for (std::size_t index = 0; index < element.inputIndices.size(); ++index) {
    if (element.inputIndices[index] != element.inputIndex)
      continue;
    return index < element.inputExtractChannels.size()
               ? element.inputExtractChannels[index]
               : -1;
  }
  return -1;
}

/**
 * @brief Channel count of the compiled audio/main input after XY extraction.
 * @param elements Compiled elements in topological order.
 * @param element Node whose `inputIndex` and extract table are populated.
 * @return Positive width, or the previously stored `inputChannels`.
 */
int compiledMainInputChannels(const std::vector<CompiledElement> &elements,
                              const CompiledElement &element) noexcept {
  if (element.inputIndices.empty())
    return element.inputChannels;
  const auto sourceIndex = static_cast<std::size_t>(element.inputIndex);
  if (sourceIndex >= elements.size())
    return element.inputChannels;
  return compiledSlotChannels(elements[sourceIndex],
                              compiledExtractForInput(element));
}

/**
 * @brief Extracts one channel or returns the full tensor.
 * @param value Source tensor.
 * @param extractChannel Channel to keep, or -1 for the full tensor.
 * @return Tensor shaped [1, 1, T] when extracting, otherwise @p value.
 */
torch::Tensor extractCompiledInput(const torch::Tensor &value,
                                   int extractChannel) {
  if (!value.defined() || extractChannel < 0 || extractChannel >= value.size(1))
    return value;
  return value.narrow(1, extractChannel, 1).contiguous();
}

/**
 * @brief Left-pads or right-crops a `[B, C, T]` control to @p targetLength.
 *
 * FiLM uses stored control history via `extendCausalControl`. This helper
 * remains for Gain envelopes and as a length fallback. Repeating the first
 * frame of the current block for the TCN RF prefix zippers at buffer rate.
 *
 * @param control Conditioning or Gain envelope.
 * @param targetLength Temporal length of the activations being modulated.
 */
torch::Tensor alignControlTime(const torch::Tensor &control,
                               std::int64_t targetLength) {
  if (!control.defined() || control.dim() != 3 || targetLength < 1)
    return control;
  if (control.size(2) == targetLength)
    return control;
  if (control.size(2) > targetLength)
    return control.narrow(2, control.size(2) - targetLength, targetLength);
  const auto missing = targetLength - control.size(2);
  auto head = control.narrow(2, 0, 1).expand(
      {control.size(0), control.size(1), missing});
  return torch::cat({head.contiguous(), control}, 2);
}

/**
 * @brief Applies per-sample FiLM from a multi-dimensional control tensor.
 *
 * Concatenated XY `[x, y]` stays a 2-vector (not `x+y`). Time is not averaged,
 * so a ramped Knob/XY trajectory actually ramps instead of stepping once per
 * audio buffer.
 *
 * @param samples Hidden activations shaped `[batch, channels, time]`.
 * @param cond Control tensor, typically `[batch, condDim, time]`.
 * @param weight Linear adaptor weight `[2 * channels, condDim]`.
 * @param bias Linear adaptor bias `[2 * channels]`.
 * @param condDim Compiled control width.
 * @return Modulated activations with the same shape as @p samples.
 */
torch::Tensor applyFilm(const torch::Tensor &samples, const torch::Tensor &cond,
                        const torch::Tensor &weight, const torch::Tensor &bias,
                        int condDim) {
  if (!samples.defined() || !cond.defined() || !weight.defined() ||
      !bias.defined() || condDim < 1)
    return samples;
  auto timed = cond;
  if (timed.dim() == 1)
    timed = timed.unsqueeze(0).unsqueeze(-1);
  else if (timed.dim() == 2)
    timed = timed.unsqueeze(-1);
  if (timed.dim() != 3)
    return samples;
  timed = alignControlTime(timed, samples.size(2));
  auto layout = timed.permute({0, 2, 1});
  if (layout.size(2) < condDim)
    layout = torch::nn::functional::pad(
        layout, torch::nn::functional::PadFuncOptions(
                    {0, condDim - static_cast<int>(layout.size(2))}));
  else if (layout.size(2) > condDim)
    layout = layout.narrow(2, 0, condDim);
  const auto adapted = torch::linear(layout, weight, bias);
  const auto parts = adapted.chunk(2, /*dim=*/-1);
  return samples * parts[0].permute({0, 2, 1}) +
         parts[1].permute({0, 2, 1});
}

/**
 * @brief Returns the connected control tensor with XY pin extraction applied.
 * @param outputs Runtime element outputs.
 * @param element Compiled consumer with optional FiLM wiring.
 */
torch::Tensor resolveFilmConditioning(const std::vector<torch::Tensor> &outputs,
                                      const CompiledElement &element) {
  if (element.filmInputIndex < 0 ||
      static_cast<std::size_t>(element.filmInputIndex) >= outputs.size())
    return {};
  const auto &cond =
      outputs[static_cast<std::size_t>(element.filmInputIndex)];
  if (!cond.defined())
    return {};
  return extractCompiledInput(cond, element.filmExtractChannel);
}

/** @brief Creates a compile failure value. */
openyourbox::dsp::LiveGraphCompileResult
failure(openyourbox::dsp::LiveGraphErrorCode code, std::int32_t nodeId,
        std::string message) {
  openyourbox::dsp::LiveGraphCompileResult result;
  result.error.code = code;
  result.error.nodeId = nodeId;
  result.error.message = std::move(message);
  return result;
}

/** @brief Clears every writable host output channel. */
void clearHostOutput(float *const *channels, std::size_t channelCount,
                     std::size_t sampleCount) noexcept {
  if (channels == nullptr)
    return;
  for (std::size_t channel = 0; channel < channelCount; ++channel) {
    if (channels[channel] != nullptr)
      std::fill_n(channels[channel], sampleCount, 0.0f);
  }
}

/**
 * @brief One-pole smoother for Knob, XY, and Gain.
 *
 * Linear ramps restart their slope on every UI target update (~60 Hz), which
 * is heard as a TV-like buzz. An exponential chase stays continuous when the
 * target jumps.
 */
struct OnePoleSmoother {
  /**
   * @brief Sets the time constant so the chase settles ~60 dB in @p seconds.
   * @param sampleRate Host sample rate in Hz.
   * @param seconds Time to decay a step to 0.1%, or 0 for an instant jump.
   */
  void reset(double sampleRate, double seconds) noexcept {
    const auto rate = std::max(1.0, sampleRate);
    if (seconds <= 0.0) {
      coeff = 1.0f;
      return;
    }
    coeff = static_cast<float>(
        1.0 - std::exp(-std::log(1000.0) / (seconds * rate)));
  }

  /** @brief Snaps both current and target to @p value. */
  void setCurrentAndTargetValue(float value) noexcept { current = target = value; }

  /** @brief Aims the chase at @p value without a discontinuity in current. */
  void setTargetValue(float value) noexcept { target = value; }

  /** @brief Returns the current smoothed value. */
  [[nodiscard]] float getCurrentValue() const noexcept { return current; }

  /** @brief Returns true while current has not yet reached the target. */
  [[nodiscard]] bool isSmoothing() const noexcept {
    return std::abs(current - target) > 1.0e-7f;
  }

  /** @brief Advances one sample toward the target and returns it. */
  float getNextValue() noexcept {
    current += coeff * (target - current);
    return current;
  }

private:
  /** @brief Per-sample mix of the remaining error, in `(0, 1]`. */
  float coeff = 1.0f;
  /** @brief Latest smoothed value. */
  float current = 0.0f;
  /** @brief Destination of the chase. */
  float target = 0.0f;
};
} // namespace

namespace openyourbox::dsp {
/** @brief Immutable implementation backing LiveGraphSnapshot. */
struct LiveGraphSnapshot::Impl {
  /** @brief Validated host input channel count. */
  int inputChannels = 0;
  /** @brief Validated host output channel count. */
  int outputChannels = 0;
  /** @brief Largest accepted audio block. */
  std::int64_t maximumBlockSize = 0;
  /** @brief Topologically ordered immutable execution plan. */
  std::vector<CompiledElement> elements;
  /** @brief Public per-element metadata in topological order. */
  std::vector<LiveGraphElementStatistics> statistics;
  /** @brief End-to-end causal receptive field. */
  std::uint64_t receptiveField = 1;
  /** @brief Total mutable scalar parameters. */
  std::uint64_t parameterCount = 0;
  /** @brief Host sample rate copied from compile options. */
  double sampleRate = 44100.0;
  /** @brief One-pole control-chase duration in seconds. */
  double controlRampSeconds = controlRampSecondsDefault;
};

/** @brief Mutable implementation backing LiveGraphRuntime. */
struct LiveGraphRuntime::Impl {
  /** @brief Immutable graph program retained for the runtime lifetime. */
  std::shared_ptr<const LiveGraphSnapshot> snapshot;
  /** @brief Intermediate tensors indexed by topological element index. */
  std::vector<torch::Tensor> outputs;
  /** @brief Per-element raw-input causal histories. */
  std::vector<torch::Tensor> histories;
  /** @brief Per-element FiLM control histories, same length as the audio RF. */
  std::vector<torch::Tensor> conditioningHistories;
  /** @brief Per-element frozen kernels created outside the audio callback. */
  std::vector<std::unique_ptr<FrozenBlackBoxKernel>> blackBoxKernels;
  /** @brief Optional Gold encode taps, parallel to `outputs`. */
  std::vector<torch::Tensor> latentOutputs;
  /** @brief Reusable planar-to-tensor host input storage. */
  torch::Tensor hostInput;
  /** @brief Lock-free latest per-element inference durations in milliseconds.
   */
  std::unique_ptr<std::atomic<double>[]> inferenceMilliseconds;
  /** @brief Latest upstream peak per compiled element. */
  std::unique_ptr<std::atomic<float>[]> inputPeaks;
  /** @brief Latest output peak per compiled element. */
  std::unique_ptr<std::atomic<float>[]> outputPeaks;
  /** @brief Published Gain/conditioning table, or null for compiled defaults. */
  std::shared_ptr<const RuntimeControlState> controls;
  /** @brief Per-element Gain ramps for Activation and TCN. */
  std::vector<OnePoleSmoother> gainSmoothers;
  /** @brief Per-element Knob/XY ramps; XY stores X in [0] and Y in [1]. */
  std::vector<std::array<OnePoleSmoother, 2>> conditioningSmoothers;
};

/** @brief Constructs a snapshot from validated immutable storage. */
LiveGraphSnapshot::LiveGraphSnapshot(
    std::shared_ptr<const Impl> implementationToAdopt)
    : implementation(std::move(implementationToAdopt)) {}

/** @brief Destroys immutable snapshot storage. */
LiveGraphSnapshot::~LiveGraphSnapshot() = default;

/** @brief Returns the snapshot input channel count. */
int LiveGraphSnapshot::getInputChannels() const noexcept {
  return implementation->inputChannels;
}

/** @brief Returns the snapshot output channel count. */
int LiveGraphSnapshot::getOutputChannels() const noexcept {
  return implementation->outputChannels;
}

/** @brief Returns the prepared-runtime maximum block size. */
std::int64_t LiveGraphSnapshot::getMaximumBlockSize() const noexcept {
  return implementation->maximumBlockSize;
}

/** @brief Returns the end-to-end graph receptive field. */
std::uint64_t LiveGraphSnapshot::getReceptiveField() const noexcept {
  return implementation->receptiveField;
}

/** @brief Returns the total graph parameter count. */
std::uint64_t LiveGraphSnapshot::getParameterCount() const noexcept {
  return implementation->parameterCount;
}

/** @brief Returns immutable topological element statistics. */
const std::vector<LiveGraphElementStatistics> &
LiveGraphSnapshot::getElementStatistics() const noexcept {
  return implementation->statistics;
}

/** @brief Produces an immutable copy with one element deterministically reset.
 */
std::shared_ptr<const LiveGraphSnapshot>
LiveGraphSnapshot::withRandomizedElement(std::int32_t nodeId, std::int32_t seed,
                                         LiveGraphCompileError &error) const {
  error = {};
  try {
    auto replacement = std::make_shared<Impl>(*implementation);
    const auto target =
        std::find_if(replacement->elements.begin(), replacement->elements.end(),
                     [nodeId](const CompiledElement &element) {
                       return element.nodeId == nodeId;
                     });
    if (target == replacement->elements.end() || !target->randomizable) {
      error.code = LiveGraphErrorCode::invalidRandomization;
      error.nodeId = nodeId;
      error.message =
          "Target element is absent, frozen, or does not own live weights";
      return {};
    }

    randomizeElementWeights(*target, seed);
    return std::shared_ptr<const LiveGraphSnapshot>(
        new LiveGraphSnapshot(std::move(replacement)));
  } catch (const std::exception &exception) {
    error.code = LiveGraphErrorCode::torchFailure;
    error.nodeId = nodeId;
    error.message = exception.what();
    return {};
  }
}

/** @brief Constructs a runtime from fully prepared mutable storage. */
LiveGraphRuntime::LiveGraphRuntime(std::unique_ptr<Impl> implementationToAdopt)
    : implementation(std::move(implementationToAdopt)) {}

/** @brief Destroys runtime state and prepared frozen kernels. */
LiveGraphRuntime::~LiveGraphRuntime() = default;

/** @brief Returns the immutable snapshot paired with this runtime. */
const std::shared_ptr<const LiveGraphSnapshot> &
LiveGraphRuntime::getSnapshot() const noexcept {
  return implementation->snapshot;
}

namespace {
float resolveGain(const CompiledElement &element,
                  const openyourbox::dsp::RuntimeControlState *controls) {
  if (controls != nullptr) {
    const auto found = controls->gainByNodeId.find(element.nodeId);
    if (found != controls->gainByNodeId.end())
      return found->second;
  }
  return element.gain;
}

std::array<float, 2>
resolveConditioning(const CompiledElement &element,
                    const openyourbox::dsp::RuntimeControlState *controls) {
  if (controls != nullptr) {
    const auto found = controls->conditioningByNodeId.find(element.nodeId);
    if (found != controls->conditioningByNodeId.end())
      return found->second;
  }
  return {element.type == openyourbox::graph::NodeType::knobInput
              ? element.conditioningValue
              : element.conditioningX,
          element.conditioningY};
}

float tensorPeak(const torch::Tensor &value) noexcept {
  if (!value.defined() || value.numel() == 0)
    return 0.0f;
  const auto contiguous = value.is_contiguous() ? value : value.contiguous();
  const auto *data = contiguous.data_ptr<float>();
  const auto count = contiguous.numel();
  float peak = 0.0f;
  for (std::int64_t index = 0; index < count; ++index)
    peak = std::max(peak, std::abs(data[index]));
  return peak;
}

torch::Tensor matchTimeLength(const torch::Tensor &value, std::int64_t samples) {
  if (!value.defined() || value.size(2) == samples)
    return value;
  if (value.size(2) > samples)
    return value.narrow(2, value.size(2) - samples, samples);
  return torch::nn::functional::pad(
      value, torch::nn::functional::PadFuncOptions({samples - value.size(2), 0})
                 .mode(torch::kConstant)
                 .value(0.0));
}

torch::Tensor gatherUpstream(const std::vector<torch::Tensor> &outputs,
                             const CompiledElement &element,
                             const std::vector<torch::Tensor> *latentOutputs =
                                 nullptr) {
  if (element.inputIndices.empty())
    return {};
  const auto source = static_cast<std::size_t>(element.inputIndex);
  auto value = outputs[source];
  if (latentOutputs != nullptr && !element.inputUseLatentTap.empty() &&
      element.inputUseLatentTap.front() != 0 &&
      source < latentOutputs->size() && (*latentOutputs)[source].defined())
    value = (*latentOutputs)[source];
  return extractCompiledInput(value, compiledExtractForInput(element));
}

std::vector<torch::Tensor>
gatherAllInputs(const std::vector<torch::Tensor> &outputs,
                const CompiledElement &element) {
  std::vector<torch::Tensor> inputs;
  inputs.reserve(element.inputIndices.size());
  for (std::size_t index = 0; index < element.inputIndices.size(); ++index) {
    auto value = outputs[static_cast<std::size_t>(element.inputIndices[index])];
    const auto extract = index < element.inputExtractChannels.size()
                             ? element.inputExtractChannels[index]
                             : -1;
    inputs.push_back(extractCompiledInput(value, extract));
  }
  return inputs;
}

/**
 * @brief Writes one channel of one-pole-smoothed control samples.
 * @param destination Contiguous destination of length @p samples.
 * @param samples Number of samples to emit.
 * @param smoother Audio-thread chase; target must already be set.
 */
void writeSmoothedChannel(float *destination, std::int64_t samples,
                          OnePoleSmoother &smoother) {
  if (destination == nullptr || samples < 1)
    return;
  if (!smoother.isSmoothing()) {
    std::fill_n(destination, static_cast<std::size_t>(samples),
                smoother.getCurrentValue());
    return;
  }
  for (std::int64_t index = 0; index < samples; ++index)
    destination[index] = smoother.getNextValue();
}

/**
 * @brief Builds a [1, 1, T] Gain envelope, advancing the smoother.
 * @param smoother Audio-thread Gain chase; target must already be set.
 * @param samples Envelope length.
 * @param options Tensor options matching the audio block.
 * @return Broadcastable Gain trajectory.
 */
torch::Tensor makeGainEnvelope(OnePoleSmoother &smoother, std::int64_t samples,
                               torch::TensorOptions options) {
  auto envelope = torch::empty({1, 1, samples}, options);
  writeSmoothedChannel(envelope.data_ptr<float>(), samples, smoother);
  return envelope;
}

torch::Tensor executeMerge(const CompiledElement &element,
                           const std::vector<torch::Tensor> &inputs,
                           std::int64_t samples, torch::TensorOptions options) {
  if (inputs.empty()) {
    const auto fill = element.mergeMode == 1 ? 1.0f : 0.0f;
    return torch::full({1, 1, samples}, fill, options);
  }

  if (element.mergeMode == 2)
    return torch::cat(inputs, 1);

  int width = 1;
  for (const auto &value : inputs) {
    if (value.defined())
      width = std::max(width, static_cast<int>(value.size(1)));
  }

  auto output = element.mergeMode == 1
                    ? torch::ones({1, width, samples}, options)
                    : torch::zeros({1, width, samples}, options);
  for (const auto &value : inputs) {
    const auto aligned = broadcastChannels(value, width);
    output = element.mergeMode == 1 ? output * aligned : output + aligned;
  }
  return output;
}

/**
 * @brief Evaluates a prepared Math Expression over aligned input tensors.
 * @param element Compiled Math Expression with a postfix program.
 * @param pinTensors One tensor per configured pin (`x1` at index 0), undefined if unused.
 * @param samples Output time length.
 * @param options Tensor options matching the audio block.
 */
torch::Tensor executeMathExpression(const CompiledElement &element,
                                    const std::vector<torch::Tensor> &pinTensors,
                                    std::int64_t samples,
                                    torch::TensorOptions options) {
  using openyourbox::graph::ExpressionInstruction;
  int width = 1;
  for (std::size_t index = 0; index < pinTensors.size(); ++index) {
    if (!openyourbox::graph::mathExpressionReferencesInput(
            element.mathAst, static_cast<int>(index) + 1))
      continue;
    if (pinTensors[index].defined())
      width = std::max(width, static_cast<int>(pinTensors[index].size(1)));
  }
  std::vector<torch::Tensor> stack;
  stack.reserve(element.mathAst.instructions.size());
  const auto asTensor = [&](double literal) {
    return torch::full({1, width, samples}, static_cast<float>(literal), options);
  };
  const auto binary = [&](auto op) {
    auto b = stack.back();
    stack.pop_back();
    auto a = stack.back();
    stack.pop_back();
    a = matchTimeLength(a, samples);
    b = matchTimeLength(b, samples);
    const auto opWidth =
        std::max(static_cast<int>(a.size(1)), static_cast<int>(b.size(1)));
    a = broadcastChannels(a, opWidth);
    b = broadcastChannels(b, opWidth);
    stack.push_back(op(a, b));
  };
  for (const auto &instruction : element.mathAst.instructions) {
    switch (instruction.op) {
    case ExpressionInstruction::Op::pushLiteral:
      stack.push_back(asTensor(instruction.literal));
      break;
    case ExpressionInstruction::Op::pushIdent: {
      const auto pin = instruction.identIndex - 1;
      if (pin < 0 || static_cast<std::size_t>(pin) >= pinTensors.size() ||
          !pinTensors[static_cast<std::size_t>(pin)].defined()) {
        stack.push_back(asTensor(0.0));
        break;
      }
      auto value = matchTimeLength(pinTensors[static_cast<std::size_t>(pin)],
                                   samples);
      stack.push_back(broadcastChannels(value, width));
      break;
    }
    case ExpressionInstruction::Op::negate:
      if (!stack.empty())
        stack.back() = -stack.back();
      break;
    case ExpressionInstruction::Op::add:
      binary([](const torch::Tensor &a, const torch::Tensor &b) { return a + b; });
      break;
    case ExpressionInstruction::Op::subtract:
      binary([](const torch::Tensor &a, const torch::Tensor &b) { return a - b; });
      break;
    case ExpressionInstruction::Op::multiply:
      binary([](const torch::Tensor &a, const torch::Tensor &b) { return a * b; });
      break;
    case ExpressionInstruction::Op::divide:
      binary([](const torch::Tensor &a, const torch::Tensor &b) { return a / b; });
      break;
    case ExpressionInstruction::Op::power:
      binary([](const torch::Tensor &a, const torch::Tensor &b) {
        return torch::pow(a, b);
      });
      break;
    case ExpressionInstruction::Op::exp:
      if (!stack.empty())
        stack.back() = torch::exp(stack.back());
      break;
    }
  }
  if (stack.empty())
    return torch::zeros({1, width, samples}, options);
  return matchTimeLength(stack.back(), samples);
}

} // namespace

void LiveGraphRuntime::executeElement(std::size_t index,
                                      const torch::Tensor &blockInput) {
  auto &runtime = *implementation;
  const auto &element =
      runtime.snapshot->implementation->elements[index];
  auto &output = runtime.outputs[index];
  const auto *controls = runtime.controls.get();
  const auto samples = blockInput.size(2);

  if (element.type == openyourbox::graph::NodeType::audioInput) {
    output = adaptHostInputToMode(blockInput, element.hostIoMode);
    return;
  }
  if (element.type == openyourbox::graph::NodeType::knobInput) {
    const auto values = resolveConditioning(element, controls);
    auto &smoother = runtime.conditioningSmoothers[index][0];
    smoother.setTargetValue(values[0]);
    output = torch::empty({1, 1, samples}, blockInput.options());
    writeSmoothedChannel(output.data_ptr<float>(), samples, smoother);
    return;
  }
  if (element.type == openyourbox::graph::NodeType::xyTrackpad) {
    const auto values = resolveConditioning(element, controls);
    auto &xSmoother = runtime.conditioningSmoothers[index][0];
    auto &ySmoother = runtime.conditioningSmoothers[index][1];
    xSmoother.setTargetValue(values[0]);
    ySmoother.setTargetValue(values[1]);
    output = torch::empty({1, 2, samples}, blockInput.options());
    auto *data = output.data_ptr<float>();
    writeSmoothedChannel(data, samples, xSmoother);
    writeSmoothedChannel(data + samples, samples, ySmoother);
    return;
  }
  if (element.type == openyourbox::graph::NodeType::mathExpression) {
    std::vector<torch::Tensor> pinTensors(element.mathPinSources.size());
    for (std::size_t pin = 0; pin < element.mathPinSources.size(); ++pin) {
      const auto source = element.mathPinSources[pin];
      if (source < 0)
        continue;
      auto value =
          runtime.outputs[static_cast<std::size_t>(source)];
      const auto extract =
          pin < element.mathPinExtract.size() ? element.mathPinExtract[pin]
                                              : -1;
      pinTensors[pin] = extractCompiledInput(value, extract);
    }
    output = executeMathExpression(element, pinTensors, samples,
                                   blockInput.options());
    if (runtime.outputPeaks)
      runtime.outputPeaks[index].store(tensorPeak(output),
                                       std::memory_order_relaxed);
    return;
  }
  if (element.inputIndices.empty()) {
    output = torch::zeros(
        {1, runtime.snapshot->implementation->outputChannels, samples},
        blockInput.options());
    return;
  }

  const auto upstream =
      gatherUpstream(runtime.outputs, element, &runtime.latentOutputs);
  if (runtime.inputPeaks)
    runtime.inputPeaks[index].store(tensorPeak(upstream),
                                    std::memory_order_relaxed);

  switch (element.type) {
  case openyourbox::graph::NodeType::audioOutput: {
    torch::Tensor upstreamOut = upstream;
    if (!upstreamOut.defined())
      upstreamOut = torch::zeros(
          {1, element.outputChannels, samples}, blockInput.options());
    else if (upstreamOut.size(2) > samples)
      upstreamOut =
          upstreamOut.narrow(2, upstreamOut.size(2) - samples, samples);
    else if (upstreamOut.size(2) < samples)
      upstreamOut = torch::nn::functional::pad(
          upstreamOut, torch::nn::functional::PadFuncOptions(
                           {samples - upstreamOut.size(2), 0})
                           .mode(torch::kConstant)
                           .value(0.0));
    output = adaptGraphOutputToHost(upstreamOut, element.outputChannels);
    break;
  }
  case openyourbox::graph::NodeType::linear:
    output = project(upstream, element.weights.front());
    break;
  case openyourbox::graph::NodeType::convolution:
  case openyourbox::graph::NodeType::rateConv: {
    if (element.stride > 1) {
      RateConv conv(std::max(1, element.stride), element.kernelSize,
                    element.dilation, RateConvMode::downsample);
      output = conv.processStreaming(upstream, element.weights.front(),
                                     runtime.histories[index]);
    } else {
      const auto historyLength =
          static_cast<std::int64_t>(element.receptiveField - 1);
      auto extended =
          extendCausalInput(upstream, runtime.histories[index], historyLength);
      output = convolve(extended, element.weights.front(), element.dilation);
    }
    break;
  }
  case openyourbox::graph::NodeType::convTranspose: {
    ConvTranspose1d conv(std::max(1, element.stride), element.kernelSize,
                         element.dilation);
    output = conv.processStreaming(upstream, element.weights.front(),
                                   runtime.histories[index]);
    break;
  }
  case openyourbox::graph::NodeType::batchNorm: {
    if (element.weights.size() >= 4) {
      const auto gamma = element.weights[0].view({1, -1, 1});
      const auto beta = element.weights[1].view({1, -1, 1});
      const auto runningMean = element.weights[2].view({1, -1, 1});
      const auto runningVar = element.weights[3].view({1, -1, 1});
      auto normalized =
          (upstream - runningMean) / torch::sqrt(runningVar + 1.0e-5f);
      output = normalized * gamma + beta;
    } else {
      output = upstream;
    }
    break;
  }
  case openyourbox::graph::NodeType::activation: {
    auto &smoother = runtime.gainSmoothers[index];
    smoother.setTargetValue(resolveGain(element, controls));
    if (!smoother.isSmoothing()) {
      output = applyActivation(upstream, element.activation,
                               smoother.getCurrentValue(),
                               element.negativeSlope);
    } else {
      output = applyActivation(upstream * makeGainEnvelope(smoother, samples,
                                                           blockInput.options()),
                               element.activation, 1.0f, element.negativeSlope);
    }
    break;
  }
  case openyourbox::graph::NodeType::tcn: {
    const auto historyLength =
        static_cast<std::int64_t>(element.receptiveField - 1);
    auto value =
        extendCausalInput(upstream, runtime.histories[index], historyLength);
    value = project(value, element.weights.front());
    auto &smoother = runtime.gainSmoothers[index];
    smoother.setTargetValue(resolveGain(element, controls));
    const auto gainEnvelope =
        makeGainEnvelope(smoother, samples, blockInput.options());
    torch::Tensor cond = resolveFilmConditioning(runtime.outputs, element);
    if (cond.defined())
      cond = extendCausalControl(cond, runtime.conditioningHistories[index],
                                 historyLength);
    for (int layer = 0; layer < element.depth; ++layer) {
      const auto layerDilation = static_cast<std::int64_t>(
          openyourbox::graph::tcnLayerDilation(element.dilationGrowth, layer));
      const auto leftPadding =
          static_cast<std::int64_t>(element.kernelSize - 1) * layerDilation;
      auto residual = value;
      value = torch::nn::functional::pad(
          value, torch::nn::functional::PadFuncOptions({leftPadding, 0})
                     .mode(torch::kConstant)
                     .value(0.0));
      value = convolve(value,
                       element.weights[static_cast<std::size_t>(layer) + 1],
                       layerDilation);
      if (cond.defined() &&
          static_cast<std::size_t>(layer) < element.filmWeights.size() &&
          static_cast<std::size_t>(layer) < element.filmBiases.size()) {
        const auto layerIndex = static_cast<std::size_t>(layer);
        value = applyFilm(value, cond, element.filmWeights[layerIndex],
                          element.filmBiases[layerIndex], element.condDim);
      }
      value = applyActivation(
          value * alignControlTime(gainEnvelope, value.size(2)),
          element.activation, 1.0f, element.negativeSlope);
      if (element.residual) {
        value = value + residual.narrow(2, residual.size(2) - value.size(2),
                                        value.size(2));
      }
    }
    value = project(value, element.weights.back());
    output = value.narrow(2, value.size(2) - samples, samples);
    break;
  }
  case openyourbox::graph::NodeType::merge:
    output = executeMerge(element, gatherAllInputs(runtime.outputs, element),
                          samples,
                          blockInput.options());
    break;
  case openyourbox::graph::NodeType::blackBox: {
    const auto started = std::chrono::steady_clock::now();
    const auto historyLength =
        static_cast<std::int64_t>(element.receptiveField - 1);
    auto *kernel = runtime.blackBoxKernels[index].get();
    const bool decodeFromLatent =
        !element.inputUseLatentTap.empty() && element.inputUseLatentTap.front() != 0;
    if (kernel != nullptr && kernel->hasEncodeDecode() && decodeFromLatent) {
      output = kernel->decode(upstream);
    } else if (kernel != nullptr && kernel->hasEncodeDecode()) {
      auto extended =
          extendCausalInput(upstream, runtime.histories[index], historyLength);
      auto latent = kernel->encode(extended);
      float fidelity = element.fidelityPercent;
      if (controls != nullptr) {
        const auto found = controls->fidelityByNodeId.find(element.nodeId);
        if (found != controls->fidelityByNodeId.end())
          fidelity = found->second;
      }
      if (latent.defined() && kernel->compactnessReady())
        latent = VariationalBottleneck::applyFidelity(
            latent, fidelity, kernel->compactnessMean(), kernel->compactnessPca(),
            kernel->compactnessCumulative());
      runtime.latentOutputs[index] = latent;
      if (latent.defined())
        output = kernel->decode(latent);
      else
        output = kernel->forward(extended);
    } else {
      auto extended =
          extendCausalInput(upstream, runtime.histories[index], historyLength);
      torch::Tensor cond = resolveFilmConditioning(runtime.outputs, element);
      if (cond.defined())
        cond = extendCausalControl(cond, runtime.conditioningHistories[index],
                                   historyLength);
      output = kernel->forwardWithConditioning(extended, cond);
    }
    if (!output.defined() || output.device().type() != torch::kCPU ||
        output.scalar_type() != torch::kFloat32 || output.dim() != 3 ||
        output.size(0) != 1 || output.size(1) < 1)
      throw std::runtime_error(
          "Frozen BlackBox returned an invalid tensor shape or type");
    output = matchTimeLength(output, upstream.size(2));
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started);
    runtime.inferenceMilliseconds[index].store(elapsed.count(),
                                               std::memory_order_relaxed);
    break;
  }
  case openyourbox::graph::NodeType::pqmfAnalysis:
    if (element.pqmf)
      output = element.pqmf->analyseStreaming(upstream, runtime.histories[index]);
    else
      output = upstream;
    break;
  case openyourbox::graph::NodeType::pqmfSynthesis:
    if (element.pqmf)
      output = element.pqmf->synthesiseStreaming(
          upstream, runtime.histories[index], element.outputChannels);
    else
      output = upstream;
    break;
  case openyourbox::graph::NodeType::variationalBottleneck: {
    float fidelity = element.fidelityPercent;
    if (controls != nullptr) {
      const auto found = controls->fidelityByNodeId.find(element.nodeId);
      if (found != controls->fidelityByNodeId.end())
        fidelity = found->second;
    }
    const auto historyLength =
        static_cast<std::int64_t>(element.receptiveField - 1);
    auto extended =
        extendCausalInput(upstream, runtime.histories[index], historyLength);
    output = VariationalBottleneck::encodeMean(
        extended, element.weights.front(), fidelity, element.compactnessReady,
        element.latentMean, element.latentPca, element.cumulativeVariance);
    break;
  }
  case openyourbox::graph::NodeType::noiseSynthesizer:
    output = NoiseSynthesizer::process(upstream, element.weights.front());
    break;
  case openyourbox::graph::NodeType::audioInput:
  case openyourbox::graph::NodeType::knobInput:
  case openyourbox::graph::NodeType::xyTrackpad:
  case openyourbox::graph::NodeType::mathExpression:
  case openyourbox::graph::NodeType::groupInput:
  case openyourbox::graph::NodeType::groupOutput:
    break;
  }

  if (runtime.outputPeaks)
    runtime.outputPeaks[index].store(tensorPeak(output),
                                     std::memory_order_relaxed);
}

torch::Tensor LiveGraphRuntime::processTensor(const torch::Tensor &input) {
  const auto &snapshot = *implementation->snapshot->implementation;
  if (!input.defined() || input.device().type() != torch::kCPU ||
      input.scalar_type() != torch::kFloat32 || input.dim() != 3 ||
      input.size(0) != 1 || input.size(1) != snapshot.inputChannels ||
      input.size(2) < 1 || input.size(2) > snapshot.maximumBlockSize)
    throw std::invalid_argument(
        "Live graph input must be CPU float [1, channels, valid samples]");

  torch::InferenceMode inferenceGuard;
  for (std::size_t index = 0; index < snapshot.elements.size(); ++index)
    executeElement(index, input);
  return implementation->outputs.back();
}

void LiveGraphRuntime::bindControls(
    std::shared_ptr<const RuntimeControlState> controls) noexcept {
  implementation->controls = std::move(controls);
}

torch::Tensor
LiveGraphRuntime::processTensorTapped(const torch::Tensor &input,
                                      std::int32_t nodeId) {
  processTensor(input);
  const auto &elements = implementation->snapshot->implementation->elements;
  for (std::size_t index = 0; index < elements.size(); ++index) {
    if (elements[index].nodeId == nodeId)
      return implementation->outputs[index];
  }
  return {};
}

torch::Tensor LiveGraphRuntime::processIsolated(std::int32_t nodeId,
                                                const torch::Tensor &probe) {
  const auto &elements = implementation->snapshot->implementation->elements;
  std::size_t target = elements.size();
  for (std::size_t index = 0; index < elements.size(); ++index) {
    if (elements[index].nodeId == nodeId) {
      target = index;
      break;
    }
  }
  if (target >= elements.size() || !probe.defined())
    return {};

  seedIsolatedUpstreamOutputs(target, probe);
  torch::InferenceMode inferenceGuard;
  executeElement(target, probe);
  return implementation->outputs[target];
}

bool LiveGraphRuntime::getTapPeaks(std::int32_t nodeId, float &inputPeak,
                                   float &outputPeak) const noexcept {
  const auto &elements = implementation->snapshot->implementation->elements;
  for (std::size_t index = 0; index < elements.size(); ++index) {
    if (elements[index].nodeId != nodeId)
      continue;
    inputPeak = implementation->inputPeaks
                    ? implementation->inputPeaks[index].load(
                          std::memory_order_relaxed)
                    : 0.0f;
    outputPeak = implementation->outputPeaks
                     ? implementation->outputPeaks[index].load(
                           std::memory_order_relaxed)
                     : 0.0f;
    return true;
  }
  return false;
}

/** @brief Executes planar host audio and converts failures to silence. */
bool LiveGraphRuntime::processHost(const float *const *inputChannels,
                                   std::size_t inputChannelCount,
                                   float *const *outputChannels,
                                   std::size_t outputChannelCount,
                                   std::size_t sampleCount) noexcept {
  clearHostOutput(outputChannels, outputChannelCount, sampleCount);
  const auto &snapshot = *implementation->snapshot->implementation;
  if (inputChannels == nullptr || outputChannels == nullptr ||
      inputChannelCount != static_cast<std::size_t>(snapshot.inputChannels) ||
      outputChannelCount != static_cast<std::size_t>(snapshot.outputChannels) ||
      sampleCount == 0 ||
      sampleCount > static_cast<std::size_t>(snapshot.maximumBlockSize))
    return false;

  for (std::size_t channel = 0; channel < inputChannelCount; ++channel) {
    if (inputChannels[channel] == nullptr || outputChannels[channel] == nullptr)
      return false;
  }

  try {
    auto input = implementation->hostInput.narrow(
        2, 0, static_cast<std::int64_t>(sampleCount));
    for (std::size_t channel = 0; channel < inputChannelCount; ++channel) {
      auto plane = input[0][static_cast<std::int64_t>(channel)];
      std::memcpy(plane.data_ptr<float>(), inputChannels[channel],
                  sampleCount * sizeof(float));
    }

    auto output = processTensor(input).contiguous();
    for (std::size_t channel = 0; channel < outputChannelCount; ++channel) {
      const auto plane = output[0][static_cast<std::int64_t>(channel)];
      std::memcpy(outputChannels[channel], plane.data_ptr<float>(),
                  sampleCount * sizeof(float));
    }
    return true;
  } catch (...) {
    clearHostOutput(outputChannels, outputChannelCount, sampleCount);
    return false;
  }
}

/** @brief Reads the latest lock-free timing sample for a frozen element. */
double LiveGraphRuntime::getFrozenInferenceTimeMilliseconds(
    std::int32_t nodeId) const noexcept {
  const auto &elements = implementation->snapshot->implementation->elements;
  for (std::size_t index = 0; index < elements.size(); ++index) {
    if (elements[index].nodeId == nodeId &&
        elements[index].type == graph::NodeType::blackBox)
      return implementation->inferenceMilliseconds[index].load(
          std::memory_order_relaxed);
  }
  return 0.0;
}

/** @brief Clears all retained causal samples in place. */
void LiveGraphRuntime::reset() noexcept {
  try {
    for (auto &history : implementation->histories) {
      if (history.defined())
        history.zero_();
    }
    for (auto &history : implementation->conditioningHistories)
      history = torch::Tensor();
    const auto elementCount =
        implementation->snapshot->implementation->elements.size();
    for (std::size_t index = 0; index < elementCount; ++index)
      implementation->inferenceMilliseconds[index].store(
          0.0, std::memory_order_relaxed);
  } catch (...) {
  }
}

/** @brief Compiles a validated immutable program from the editable graph. */
LiveGraphCompileResult
LiveGraphEngine::compile(const graph::NodeGraph &graphDocument,
                         const LiveGraphCompileOptions &options,
                         FrozenBlackBoxResolver blackBoxResolver) {
  using graph::MergeMode;
  using graph::NodeType;

  if ((options.hostInputChannels != 1 && options.hostInputChannels != 2) ||
      (options.hostOutputChannels != 1 && options.hostOutputChannels != 2) ||
      options.maximumBlockSize < 1 || options.maximumHistorySamples < 1 ||
      options.sampleRate <= 0.0 || options.controlRampSeconds < 0.0)
    return failure(LiveGraphErrorCode::invalidCompileOptions, 0,
                   "Live graph supports mono/stereo hosts, positive blocks, "
                   "and a non-negative control ramp");

  const auto &nodes = graphDocument.getNodes();
  const auto &links = graphDocument.getLinks();
  if (nodes.empty())
    return failure(LiveGraphErrorCode::invalidBoundary, 0, "Graph is empty");

  std::unordered_map<std::int32_t, const graph::GraphNode *> nodesById;
  std::unordered_map<std::int32_t, PinOwner> pins;
  std::int32_t inputNodeId = 0;
  std::int32_t outputNodeId = 0;
  int inputNodeCount = 0;
  int outputNodeCount = 0;

  for (const auto &node : nodes) {
    if (node.id == 0 || !nodesById.emplace(node.id, &node).second)
      return failure(LiveGraphErrorCode::invalidGraph, node.id,
                     "Node identifiers must be unique and non-zero");
    if ((node.type == NodeType::blackBox) &&
        (node.state != graph::NodeState::frozenGold))
      return failure(LiveGraphErrorCode::invalidGraph, node.id,
                     "BlackBox elements must use the frozen execution state");
    if (node.state == graph::NodeState::frozenGold &&
        node.artifactPath.empty() && node.type != NodeType::blackBox)
      return failure(LiveGraphErrorCode::invalidGraph, node.id,
                     "Frozen elements require a compiled artifact");
    if (!hasValidPortLayout(node))
      return failure(LiveGraphErrorCode::invalidGraph, node.id,
                     "Element has an unsupported input/output port layout");
    if (node.type == NodeType::audioInput) {
      inputNodeId = node.id;
      ++inputNodeCount;
    } else if (node.type == NodeType::audioOutput) {
      outputNodeId = node.id;
      ++outputNodeCount;
    } else if (graph::isGroupBoundaryType(node.type)) {
      return failure(
          LiveGraphErrorCode::invalidGraph, node.id,
          "Group Input/Output hubs must be removed before live compilation");
    }

    const auto addPin = [&](const graph::Pin &pin, graph::PinKind kind) {
      return pin.id != 0 && pin.kind == kind &&
             pins.emplace(pin.id, PinOwner{node.id, kind}).second;
    };
    for (const auto &pin : node.inputs) {
      if (!addPin(pin, graph::PinKind::input))
        return failure(LiveGraphErrorCode::invalidGraph, node.id,
                       "Input pin is duplicated or malformed");
    }
    for (const auto &pin : node.outputs) {
      if (!addPin(pin, graph::PinKind::output))
        return failure(LiveGraphErrorCode::invalidGraph, node.id,
                       "Output pin is duplicated or malformed");
    }
  }

  if (inputNodeCount != 1 || outputNodeCount != 1)
    return failure(
        LiveGraphErrorCode::invalidBoundary, 0,
        "Graph must contain exactly one Audio Input and Audio Output");

  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> incoming;
  std::unordered_map<std::int32_t, int> indegree;
  std::unordered_map<std::int32_t, std::int32_t> sourceNodeByDestinationPin;
  std::unordered_map<std::int32_t, std::int32_t> sourcePinIdByDestinationPin;
  std::unordered_set<std::int32_t> linkIds;
  for (const auto &node : nodes)
    indegree[node.id] = 0;

  for (const auto &link : links) {
    const auto source = pins.find(link.sourcePinId);
    const auto destination = pins.find(link.destinationPinId);
    if (link.id == 0 || !linkIds.insert(link.id).second ||
        source == pins.end() || destination == pins.end() ||
        source->second.kind != graph::PinKind::output ||
        destination->second.kind != graph::PinKind::input)
      return failure(LiveGraphErrorCode::invalidGraph, 0,
                     "Link identifiers and endpoints must resolve uniquely");
    if (source->second.nodeId == destination->second.nodeId)
      return failure(LiveGraphErrorCode::cycle, source->second.nodeId,
                     "Self-connections are not permitted");
    if (!sourceNodeByDestinationPin.emplace(link.destinationPinId,
                                            source->second.nodeId)
             .second)
      return failure(LiveGraphErrorCode::invalidGraph, destination->second.nodeId,
                     "An input port cannot have more than one connection");
    sourcePinIdByDestinationPin.emplace(link.destinationPinId,
                                        link.sourcePinId);

    outgoing[source->second.nodeId].push_back(destination->second.nodeId);
    incoming[destination->second.nodeId].push_back(source->second.nodeId);
    ++indegree[destination->second.nodeId];
  }

  for (const auto &node : nodes) {
    if (node.type == NodeType::audioInput && !incoming[node.id].empty())
      return failure(LiveGraphErrorCode::invalidBoundary, node.id,
                     "Audio Input cannot have an incoming connection");
    if (node.type == NodeType::audioOutput && !outgoing[node.id].empty())
      return failure(LiveGraphErrorCode::invalidBoundary, node.id,
                     "Audio Output cannot have an outgoing connection");
  }

  std::queue<std::int32_t> ready;
  for (const auto &node : nodes) {
    if (indegree[node.id] == 0)
      ready.push(node.id);
  }
  std::vector<std::int32_t> topologicalIds;
  while (!ready.empty()) {
    const auto nodeId = ready.front();
    ready.pop();
    topologicalIds.push_back(nodeId);
    for (const auto destination : outgoing[nodeId]) {
      if (--indegree[destination] == 0)
        ready.push(destination);
    }
  }
  if (topologicalIds.size() != nodes.size()) {
    std::unordered_set<std::int32_t> ordered(topologicalIds.begin(),
                                             topologicalIds.end());
    std::string detail = "Graph contains a directed cycle involving";
    int named = 0;
    for (const auto &node : nodes) {
      if (ordered.count(node.id) != 0)
        continue;
      detail += named == 0 ? " " : ", ";
      detail += node.label.empty() ? ("#" + std::to_string(node.id))
                                   : node.label;
      if (++named >= 6)
        break;
    }
    if (named == 0)
      detail = "Graph contains a directed cycle";
    return failure(LiveGraphErrorCode::cycle, 0, detail);
  }

  std::unordered_set<std::int32_t> fromInput;
  std::queue<std::int32_t> traversal;
  traversal.push(inputNodeId);
  while (!traversal.empty()) {
    const auto nodeId = traversal.front();
    traversal.pop();
    if (!fromInput.insert(nodeId).second)
      continue;
    for (const auto destination : outgoing[nodeId])
      traversal.push(destination);
  }

  std::unordered_set<std::int32_t> toOutput;
  traversal.push(outputNodeId);
  while (!traversal.empty()) {
    const auto nodeId = traversal.front();
    traversal.pop();
    if (!toOutput.insert(nodeId).second)
      continue;
    for (const auto source : incoming[nodeId])
      traversal.push(source);
  }
  std::unordered_set<std::int32_t> livePath;
  for (const auto &node : nodes) {
    if (toOutput.count(node.id) != 0)
      livePath.insert(node.id);
  }
  if (fromInput.count(outputNodeId) == 0 || livePath.count(outputNodeId) == 0)
    return failure(LiveGraphErrorCode::incompletePath, outputNodeId,
                   "Graph has no complete Audio Input-to-Output path");

  struct FrozenGroup {
    std::unordered_set<std::int32_t> members;
    std::int32_t sourceId = 0;
    std::int32_t sinkId = 0;
  };
  std::vector<FrozenGroup> frozenGroups;
  std::unordered_map<std::int32_t, std::size_t> frozenGroupIndex;
  std::unordered_map<std::string, std::vector<std::int32_t>> frozenByArtifact;
  for (const auto &node : nodes) {
    const auto frozenRuntime =
        node.state == graph::NodeState::frozenGold ||
        (!node.artifactPath.empty() &&
         node.blackBoxOrigin == graph::BlackBoxOrigin::trainAutoload);
    if (!frozenRuntime)
      continue;
    const auto key =
        node.artifactPath.empty()
            ? std::string("node:") + std::to_string(node.id)
            : node.artifactPath;
    frozenByArtifact[key].push_back(node.id);
  }
  for (const auto &entry : frozenByArtifact) {
    FrozenGroup group;
    group.members.insert(entry.second.begin(), entry.second.end());
    std::vector<std::int32_t> sources;
    std::vector<std::int32_t> sinks;
    for (const auto nodeId : entry.second) {
      bool hasPredecessorInGroup = false;
      bool hasSuccessorInGroup = false;
      for (const auto predecessor : incoming[nodeId]) {
        if (group.members.count(predecessor) != 0)
          hasPredecessorInGroup = true;
      }
      for (const auto successor : outgoing[nodeId]) {
        if (group.members.count(successor) != 0)
          hasSuccessorInGroup = true;
      }
      if (!hasPredecessorInGroup)
        sources.push_back(nodeId);
      if (!hasSuccessorInGroup)
        sinks.push_back(nodeId);
    }
    if (sources.size() != 1 || sinks.size() != 1)
      return failure(
          LiveGraphErrorCode::invalidGraph, sources.empty() ? 0 : sources.front(),
          "A frozen selection must keep a single input and output boundary");
    group.sourceId = sources.front();
    group.sinkId = sinks.front();
    const auto index = frozenGroups.size();
    frozenGroups.push_back(std::move(group));
    for (const auto nodeId : entry.second)
      frozenGroupIndex[nodeId] = index;
  }

  try {
    auto compiled = std::make_shared<LiveGraphSnapshot::Impl>();
    compiled->inputChannels = options.hostInputChannels;
    compiled->outputChannels = options.hostOutputChannels;
    compiled->maximumBlockSize = options.maximumBlockSize;
    compiled->sampleRate = options.sampleRate;
    compiled->controlRampSeconds = options.controlRampSeconds;
    compiled->elements.reserve(nodes.size());
    compiled->statistics.reserve(nodes.size());

    std::unordered_map<std::int32_t, std::size_t> compiledIndex;
    std::unordered_map<std::int32_t, std::uint64_t> pathReceptiveField;
    for (const auto nodeId : topologicalIds) {
      const auto &node = *nodesById.at(nodeId);
      if (livePath.count(nodeId) == 0)
        continue;

      const auto frozenGroup = frozenGroupIndex.find(nodeId);
      if (frozenGroup != frozenGroupIndex.end() &&
          frozenGroups[frozenGroup->second].sinkId != nodeId)
        continue;

      CompiledElement element;
      element.nodeId = node.id;
      element.type = node.type;
      const auto compileFrozenSink =
          frozenGroup != frozenGroupIndex.end() &&
          frozenGroups[frozenGroup->second].sinkId == nodeId;
      if (node.type != NodeType::audioInput) {
        const auto *inputNode = &node;
        if (compileFrozenSink)
          inputNode = nodesById.at(frozenGroups[frozenGroup->second].sourceId);
        element.inputIndices = collectCompiledInputs(
            inputNode->inputs, sourceNodeByDestinationPin, compiledIndex);
        if (compileFrozenSink && element.inputIndices.empty())
          return failure(
              LiveGraphErrorCode::invalidGraph, node.id,
              "Frozen subgraph is missing a live input connection");
        if (element.inputIndices.empty()) {
          if (node.type != NodeType::audioOutput &&
              !graph::isConditioningSourceType(node.type) &&
              !graph::isMathExpressionType(node.type))
            continue;
          if (node.type == NodeType::audioOutput)
            element.inputChannels = options.hostOutputChannels;
        } else {
          element.inputIndex = element.inputIndices.front();
          const auto *inputNodeForPins = inputNode;
          element.inputExtractChannels.assign(element.inputIndices.size(), -1);
          element.inputIsConditioning.assign(element.inputIndices.size(), 0);
          element.inputUseLatentTap.assign(element.inputIndices.size(), 0);
          std::size_t compiledInput = 0;
          for (const auto &pin : inputNodeForPins->inputs) {
            const auto source = sourceNodeByDestinationPin.find(pin.id);
            if (source == sourceNodeByDestinationPin.end())
              continue;
            const auto compiledSource = compiledIndex.find(source->second);
            if (compiledSource == compiledIndex.end())
              continue;
            const auto *sourceNode = nodesById.at(source->second);
            const auto sourcePin = sourcePinIdByDestinationPin.find(pin.id);
            if (sourceNode->type == NodeType::xyTrackpad &&
                sourcePin != sourcePinIdByDestinationPin.end())
              element.inputExtractChannels[compiledInput] =
                  outputPinIndex(*sourceNode, sourcePin->second);
            if (sourcePin != sourcePinIdByDestinationPin.end()) {
              for (const auto &outPin : sourceNode->outputs) {
                if (outPin.id == sourcePin->second &&
                    graph::isLatentPin(outPin))
                  element.inputUseLatentTap[compiledInput] = 1;
              }
            }
            if (graph::isLatentPin(pin))
              element.inputUseLatentTap[compiledInput] = 1;
            element.inputIsConditioning[compiledInput] =
                graph::isControlInputPin(pin) ? 1 : 0;
            ++compiledInput;
          }
          element.inputChannels =
              compiledMainInputChannels(compiled->elements, element);
        }
      }
      if (compileFrozenSink)
        element.type = NodeType::blackBox;

      switch (compileFrozenSink ? NodeType::blackBox : node.type) {
      case NodeType::audioInput: {
        element.hostIoMode = graph::HostIoMode::stereo;
        int choice = static_cast<int>(graph::HostIoMode::stereo);
        if (readProperty(node, "channels", choice)) {
          const auto *property = [&node]() -> const graph::NodeProperty * {
            for (const auto &candidate : node.properties) {
              if (candidate.key == "channels")
                return &candidate;
            }
            return nullptr;
          }();
          const auto legacyPair =
              property != nullptr &&
              (property->maximum <= 1 || property->choices.size() == 2);
          element.hostIoMode =
              graph::hostIoModeFromChoice(choice, legacyPair);
        } else if (!node.outputs.empty() &&
                   node.outputs.front().shape.channels > 0) {
          element.hostIoMode = graph::hostIoModeFromChannels(
              node.outputs.front().shape.channels);
        }
        element.outputChannels =
            graph::hostIoChannelsFromMode(element.hostIoMode);
        break;
      }
      case NodeType::audioOutput: {
        element.hostIoMode = graph::HostIoMode::stereo;
        int choice = static_cast<int>(graph::HostIoMode::stereo);
        if (readProperty(node, "channels", choice)) {
          const auto *property = [&node]() -> const graph::NodeProperty * {
            for (const auto &candidate : node.properties) {
              if (candidate.key == "channels")
                return &candidate;
            }
            return nullptr;
          }();
          const auto legacyPair =
              property != nullptr &&
              (property->maximum <= 1 || property->choices.size() == 2);
          element.hostIoMode =
              graph::hostIoModeFromChoice(choice, legacyPair);
        } else if (!node.inputs.empty() &&
                   node.inputs.front().shape.channels > 0) {
          element.hostIoMode = graph::hostIoModeFromChannels(
              node.inputs.front().shape.channels);
        }
        element.outputChannels = options.hostOutputChannels;
        const auto graphWidth =
            graph::hostIoChannelsFromMode(element.hostIoMode);
        if (!element.inputIndices.empty() &&
            element.inputChannels != graphWidth)
          return failure(
              LiveGraphErrorCode::invalidShape, node.id,
              "Graph output channels do not match the Audio Output mode");
        break;
      }
      case NodeType::linear: {
        int features = 0;
        if (!readProperty(node, "features", features) || features < 1)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "Linear requires Features >= 1");
        element.outputChannels = features;
        element.randomizable = true;
        element.parameterCount = saturatedMultiply(
            static_cast<std::uint64_t>(features),
            static_cast<std::uint64_t>(element.inputChannels));
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::convolution:
      case NodeType::rateConv: {
        int channels = 0;
        if (!readProperty(node, "channels", channels) || channels < 1 ||
            !readProperty(node, "kernel_size", element.kernelSize) ||
            element.kernelSize < 1 ||
            !readProperty(node, "dilation", element.dilation) ||
            element.dilation < 1)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "Conv1D requires Channels, Kernel Size, and Dilation "
                         ">= 1");
        element.stride = 1;
        readProperty(node, "stride", element.stride);
        if (element.stride < 1)
          element.stride = 1;
        element.rateDirection = 0;
        element.outputChannels = channels;
        element.receptiveField =
            1 + static_cast<std::uint64_t>(element.kernelSize - 1) *
                    static_cast<std::uint64_t>(element.dilation);
        if (element.stride > 1)
          element.receptiveField = std::max(
              element.receptiveField,
              1 + static_cast<std::uint64_t>(element.stride));
        if (element.receptiveField - 1 > options.maximumHistorySamples)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "Conv1D receptive field exceeds the history limit");
        element.randomizable = true;
        element.parameterCount = saturatedMultiply(
            saturatedMultiply(
                static_cast<std::uint64_t>(channels),
                static_cast<std::uint64_t>(element.inputChannels)),
            static_cast<std::uint64_t>(element.kernelSize));
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::convTranspose: {
        int channels = 0;
        if (!readProperty(node, "channels", channels) || channels < 1 ||
            !readProperty(node, "kernel_size", element.kernelSize) ||
            element.kernelSize < 1 ||
            !readProperty(node, "dilation", element.dilation) ||
            element.dilation < 1)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "ConvTranspose1d requires Channels, Kernel Size, and "
                         "Dilation >= 1");
        element.stride = 1;
        readProperty(node, "stride", element.stride);
        if (element.stride < 1)
          element.stride = 1;
        element.outputChannels = channels;
        element.receptiveField =
            1 + static_cast<std::uint64_t>(element.kernelSize - 1) *
                    static_cast<std::uint64_t>(element.dilation);
        if (element.receptiveField - 1 > options.maximumHistorySamples)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "ConvTranspose1d receptive field exceeds the history "
                         "limit");
        element.randomizable = true;
        element.parameterCount = saturatedMultiply(
            saturatedMultiply(
                static_cast<std::uint64_t>(channels),
                static_cast<std::uint64_t>(element.inputChannels)),
            static_cast<std::uint64_t>(element.kernelSize));
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::batchNorm: {
        element.outputChannels = element.inputChannels;
        if (element.inputChannels < 1)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "BatchNorm1d requires at least one input channel");
        element.receptiveField = 1;
        element.randomizable = true;
        element.parameterCount =
            static_cast<std::uint64_t>(element.inputChannels) * 2;
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::activation: {
        int activation = 0;
        if (!readProperty(node, "activation", activation) || activation < 0 ||
            activation > openyourbox::dsp::maximumActivationIndex)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "Activation function selection is invalid");
        element.outputChannels = element.inputChannels;
        element.activation = static_cast<ActivationType>(activation);
        float gain = graph::gainDefault;
        readFloatProperty(node, "gain", gain);
        element.gain = graph::clampGain(gain);
        float negativeSlope = graph::leakyReluNegativeSlopeDefault;
        readFloatProperty(node, "negative_slope", negativeSlope);
        element.negativeSlope = std::clamp(
            negativeSlope, graph::leakyReluNegativeSlopeMinimum,
            graph::leakyReluNegativeSlopeMaximum);
        break;
      }
      case NodeType::tcn: {
        int activation = 0;
        int growth = graph::defaultDilationGrowth;
        int residual = 0;
        if (!readProperty(node, "depth", element.depth) ||
            !readProperty(node, "kernel_size", element.kernelSize) ||
            !readProperty(node, "channels", element.hiddenChannels) ||
            !readProperty(node, "activation", activation))
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "TCN is missing one or more required properties");
        readProperty(node, "dilation_growth", growth);
        readProperty(node, "residual", residual);
        element.dilationGrowth =
            std::max(growth, graph::minimumDilationGrowth);
        element.residual = residual != 0;

        const auto dilationRepresentable =
            element.depth >= 1 && element.dilationGrowth >= 1;
        if (!dilationRepresentable || element.kernelSize < 1 ||
            element.hiddenChannels < 1 || element.inputChannels < 1 ||
            activation < 0 ||
            activation > openyourbox::dsp::maximumActivationIndex)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "TCN configuration is outside supported bounds");

        element.activation = static_cast<ActivationType>(activation);
        float gain = graph::gainDefault;
        readFloatProperty(node, "gain", gain);
        element.gain = graph::clampGain(gain);
        float negativeSlope = graph::leakyReluNegativeSlopeDefault;
        readFloatProperty(node, "negative_slope", negativeSlope);
        element.negativeSlope = std::clamp(
            negativeSlope, graph::leakyReluNegativeSlopeMinimum,
            graph::leakyReluNegativeSlopeMaximum);
        element.filmInputIndex = -1;
        element.filmExtractChannel = -1;
        element.condDim = 0;
        for (std::size_t index = 0; index < element.inputIsConditioning.size();
             ++index) {
          if (element.inputIsConditioning[index] != 0 &&
              index < element.inputIndices.size()) {
            element.filmInputIndex = element.inputIndices[index];
            element.filmExtractChannel =
                index < element.inputExtractChannels.size()
                    ? element.inputExtractChannels[index]
                    : -1;
            const auto sourceIndex =
                static_cast<std::size_t>(element.filmInputIndex);
            if (sourceIndex < compiled->elements.size())
              element.condDim = compiledSlotChannels(
                  compiled->elements[sourceIndex], element.filmExtractChannel);
          } else if (element.inputIsConditioning[index] == 0 &&
                     index < element.inputIndices.size()) {
            element.inputIndex = element.inputIndices[index];
            element.inputChannels = compiledSlotChannels(
                compiled->elements[static_cast<std::size_t>(element.inputIndex)],
                index < element.inputExtractChannels.size()
                    ? element.inputExtractChannels[index]
                    : -1);
          }
        }
        element.outputChannels = element.inputChannels;
        std::uint64_t receptiveField = 1;
        for (int layer = 0; layer < element.depth; ++layer) {
          const auto layerDilation = static_cast<std::uint64_t>(
              graph::tcnLayerDilation(element.dilationGrowth, layer));
          receptiveField = saturatedAdd(
              receptiveField, saturatedMultiply(static_cast<std::uint64_t>(
                                                    element.kernelSize - 1),
                                                layerDilation));
        }
        element.receptiveField = receptiveField;
        if (element.receptiveField - 1 > options.maximumHistorySamples)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "TCN receptive field exceeds the history limit");
        const auto projectionIn = saturatedMultiply(
            static_cast<std::uint64_t>(element.inputChannels),
            static_cast<std::uint64_t>(element.hiddenChannels));
        const auto temporal = saturatedMultiply(
            saturatedMultiply(
                static_cast<std::uint64_t>(element.depth),
                saturatedMultiply(
                    static_cast<std::uint64_t>(element.hiddenChannels),
                    static_cast<std::uint64_t>(element.hiddenChannels))),
            static_cast<std::uint64_t>(element.kernelSize));
        const auto projectionOut = saturatedMultiply(
            static_cast<std::uint64_t>(element.hiddenChannels),
            static_cast<std::uint64_t>(element.outputChannels));
        const auto filmOut = saturatedMultiply(
            static_cast<std::uint64_t>(element.hiddenChannels), 2);
        const auto filmAdaptor = saturatedAdd(
            saturatedMultiply(filmOut,
                              static_cast<std::uint64_t>(
                                  std::max(0, element.condDim))),
            filmOut);
        const auto filmParams =
            element.condDim > 0
                ? saturatedMultiply(
                      static_cast<std::uint64_t>(element.depth), filmAdaptor)
                : 0;
        element.parameterCount = saturatedAdd(
            saturatedAdd(saturatedAdd(projectionIn, temporal), projectionOut),
            filmParams);
        element.randomizable = true;
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::merge: {
        int mode = static_cast<int>(MergeMode::add);
        if (!readProperty(node, "mode", mode) || mode < 0 || mode > 2)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         "Utility mode must be Add, Multiply, or Concatenate");
        element.mergeMode = mode;
        int width = 0;
        int connectedCount = 0;
        for (std::size_t index = 0; index < element.inputIndices.size();
             ++index) {
          const auto &source =
              compiled->elements[static_cast<std::size_t>(
                                     element.inputIndices[index])];
          const auto extract =
              index < element.inputExtractChannels.size()
                  ? element.inputExtractChannels[index]
                  : -1;
          const auto channels = compiledSlotChannels(source, extract);
          if (channels < 1)
            return failure(LiveGraphErrorCode::invalidShape, node.id,
                           "Utility inputs must have positive channel counts");
          if (mode == static_cast<int>(MergeMode::concatenate)) {
            width += channels;
          } else if (connectedCount == 0) {
            width = channels;
          } else if (!channelsAreBroadcastCompatible(channels, width)) {
            return failure(LiveGraphErrorCode::invalidShape, node.id,
                           "Utility requires connected inputs to share a channel "
                           "count, or broadcast from 1");
          } else {
            width = std::max(width, channels);
          }
          ++connectedCount;
        }
        if (connectedCount == 0)
          width = 1;
        if (width < 1)
          return failure(LiveGraphErrorCode::invalidShape, node.id,
                         "Utility output channels must be positive");
        element.outputChannels = width;
        break;
      }
      case NodeType::mathExpression: {
        std::string expressionText = "x1";
        for (const auto &property : node.properties) {
          if (property.key == "expression" && !property.stringValue.empty())
            expressionText = property.stringValue;
        }
        const auto inputCount = std::max(1, static_cast<int>(node.inputs.size()));
        const auto parsed = graph::parseExpression(
            expressionText, graph::ExpressionIdentContext::mathInputs,
            inputCount);
        if (!parsed.accepted)
          return failure(LiveGraphErrorCode::invalidProperty, node.id,
                         parsed.message);
        element.mathAst = parsed.ast;
        element.mathPinSources.assign(node.inputs.size(), -1);
        element.mathPinExtract.assign(node.inputs.size(), -1);
        int width = 0;
        std::size_t compiledInput = 0;
        for (std::size_t pinIndex = 0; pinIndex < node.inputs.size();
             ++pinIndex) {
          const auto &pin = node.inputs[pinIndex];
          const auto source = sourceNodeByDestinationPin.find(pin.id);
          if (source == sourceNodeByDestinationPin.end()) {
            if (graph::mathExpressionReferencesInput(
                    parsed.ast, static_cast<int>(pinIndex) + 1))
              return failure(LiveGraphErrorCode::invalidGraph, node.id,
                             "Math Expression input x" +
                                 std::to_string(pinIndex + 1) +
                                 " must be connected");
            continue;
          }
          const auto compiledSource = compiledIndex.find(source->second);
          if (compiledSource == compiledIndex.end())
            return failure(LiveGraphErrorCode::invalidGraph, node.id,
                           "Math Expression is missing a live input connection");
          element.mathPinSources[pinIndex] =
              static_cast<std::int64_t>(compiledSource->second);
          if (compiledInput < element.inputExtractChannels.size())
            element.mathPinExtract[pinIndex] =
                element.inputExtractChannels[compiledInput];
          const auto &sourceElement =
              compiled->elements[compiledSource->second];
          const auto channels = compiledSlotChannels(
              sourceElement, element.mathPinExtract[pinIndex]);
          if (graph::mathExpressionReferencesInput(
                  parsed.ast, static_cast<int>(pinIndex) + 1)) {
            if (channels < 1)
              return failure(LiveGraphErrorCode::invalidShape, node.id,
                             "Math Expression inputs must have positive "
                             "channel counts");
            if (width == 0)
              width = channels;
            else if (!channelsAreBroadcastCompatible(channels, width))
              return failure(
                  LiveGraphErrorCode::invalidShape, node.id,
                  "Math Expression requires referenced inputs to share a "
                  "channel count, or broadcast from 1");
            else
              width = std::max(width, channels);
          }
          ++compiledInput;
        }
        if (width < 1)
          width = 1;
        element.outputChannels = width;
        break;
      }
      case NodeType::knobInput:
        element.outputChannels = 1;
        element.outputIsConditioning = true;
        element.conditioningValue = node.conditioningValue;
        break;
      case NodeType::xyTrackpad:
        element.outputChannels = 2;
        element.outputIsConditioning = true;
        element.conditioningX = node.conditioningX;
        element.conditioningY = node.conditioningY;
        break;
      case NodeType::blackBox:
        if (!blackBoxResolver)
          return failure(LiveGraphErrorCode::invalidBlackBox, node.id,
                         "Frozen BlackBox requires an off-thread resolver");
        element.blackBoxFactory = blackBoxResolver(node);
        element.filmInputIndex = -1;
        element.filmExtractChannel = -1;
        for (std::size_t index = 0; index < element.inputIsConditioning.size();
             ++index) {
          if (element.inputIsConditioning[index] != 0 &&
              index < element.inputIndices.size()) {
            element.filmInputIndex = element.inputIndices[index];
            element.filmExtractChannel =
                index < element.inputExtractChannels.size()
                    ? element.inputExtractChannels[index]
                    : -1;
          } else if (element.inputIsConditioning[index] == 0 &&
                     index < element.inputIndices.size()) {
            element.inputIndex = element.inputIndices[index];
            element.inputChannels = compiledSlotChannels(
                compiled->elements[static_cast<std::size_t>(element.inputIndex)],
                index < element.inputExtractChannels.size()
                    ? element.inputExtractChannels[index]
                    : -1);
          }
        }
        if (!element.blackBoxFactory ||
            element.blackBoxFactory->getOutputChannels() < 1)
          return failure(LiveGraphErrorCode::invalidBlackBox, node.id,
                         "Frozen hook metadata is absent or shape-incompatible");
        // FiLM Control injects bias, so hooked freeze cannot keep digital
        // silence. Train autoload already allows that; a live Control pin must
        // as well.
        if (node.blackBoxOrigin != graph::BlackBoxOrigin::trainAutoload &&
            element.filmInputIndex < 0 &&
            !element.blackBoxFactory->hasEncodeDecode() &&
            !element.blackBoxFactory->preservesSilence())
          return failure(LiveGraphErrorCode::invalidBlackBox, node.id,
                         "Frozen hook metadata is absent, shape-incompatible, "
                         "or not silence-preserving");
        element.outputChannels = element.blackBoxFactory->getOutputChannels();
        if (node.blackBoxOrigin == graph::BlackBoxOrigin::trainAutoload &&
            element.inputChannels > 0)
          element.outputChannels = element.inputChannels;
        element.receptiveField = std::max<std::uint64_t>(
            1, element.blackBoxFactory->getReceptiveField());
        if (element.receptiveField - 1 > options.maximumHistorySamples)
          return failure(LiveGraphErrorCode::invalidBlackBox, node.id,
                         "Frozen receptive field exceeds the history limit");
        element.parameterCount = element.blackBoxFactory->getParameterCount();
        readFloatProperty(node, "fidelity", element.fidelityPercent);
        element.fidelityPercent = graph::clampFidelity(element.fidelityPercent);
        break;
      case NodeType::pqmfAnalysis: {
        int nBand = graph::defaultPqmfBands;
        readProperty(node, "n_band", nBand);
        nBand = std::max(nBand, graph::minimumPqmfBands);
        element.nBand = nBand;
        element.pqmf = std::make_shared<PqmfBank>(nBand);
        element.outputChannels = std::max(1, element.inputChannels) * nBand;
        element.receptiveField = element.pqmf->getCausalDelaySamples() + 1;
        break;
      }
      case NodeType::pqmfSynthesis: {
        int nBand = graph::defaultPqmfBands;
        readProperty(node, "n_band", nBand);
        nBand = std::max(nBand, graph::minimumPqmfBands);
        element.nBand = nBand;
        element.pqmf = std::make_shared<PqmfBank>(nBand);
        element.outputChannels = std::max(1, element.inputChannels / std::max(1, nBand));
        element.receptiveField = element.pqmf->getCausalDelaySamples() + 1;
        break;
      }
      case NodeType::variationalBottleneck: {
        int latent = graph::defaultLatentSize;
        readProperty(node, "latent_size", latent);
        if (graph::variationalBottleneckLatentIsError(latent))
          return failure(LiveGraphErrorCode::invalidGraph, node.id,
                         graph::variationalBottleneckLatentMessage(latent));
        if (graph::variationalBottleneckChannelIsError(element.inputChannels))
          return failure(
              LiveGraphErrorCode::invalidGraph, node.id,
              graph::variationalBottleneckChannelMessage(element.inputChannels));
        int kernel = graph::defaultBottleneckKernelSize;
        readProperty(node, "kernel_size", kernel);
        element.kernelSize = std::max(1, kernel);
        element.outputChannels = std::max(1, latent);
        element.receptiveField =
            static_cast<std::uint64_t>(element.kernelSize);
        element.randomizable = true;
        readFloatProperty(node, "fidelity", element.fidelityPercent);
        element.fidelityPercent = graph::clampFidelity(
            node.fidelityPercent > 0.0f ? node.fidelityPercent
                                        : element.fidelityPercent);
        element.compactnessReady = node.compactnessReady;
        const auto copyBuffer = [](const std::vector<float> &values,
                                   const std::vector<std::int64_t> &shape) {
          if (values.empty())
            return torch::Tensor{};
          auto tensor = torch::from_blob(
              const_cast<float *>(values.data()), shape,
              torch::TensorOptions().dtype(torch::kFloat32));
          return tensor.clone();
        };
        element.latentMean = copyBuffer(
            node.latentMean, {static_cast<std::int64_t>(node.latentMean.size())});
        if (!node.latentPca.empty() &&
            node.latentPca.size() ==
                static_cast<std::size_t>(element.outputChannels) *
                    static_cast<std::size_t>(element.outputChannels))
          element.latentPca = copyBuffer(
              node.latentPca,
              {static_cast<std::int64_t>(element.outputChannels),
               static_cast<std::int64_t>(element.outputChannels)});
        element.cumulativeVariance = copyBuffer(
            node.cumulativeVariance,
            {static_cast<std::int64_t>(node.cumulativeVariance.size())});
        const auto groupedIn = static_cast<std::uint64_t>(std::max(
            1, element.inputChannels / graph::variationalBottleneckGroups));
        element.parameterCount = saturatedMultiply(
            static_cast<std::uint64_t>(2 * element.outputChannels),
            saturatedMultiply(groupedIn,
                              static_cast<std::uint64_t>(element.kernelSize)));
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::noiseSynthesizer: {
        int noiseBands = graph::defaultNoiseBands;
        readProperty(node, "noise_bands", noiseBands);
        element.kernelSize = std::max(1, noiseBands);
        element.outputChannels = std::max(1, element.inputChannels);
        element.randomizable = true;
        element.parameterCount = saturatedMultiply(
            static_cast<std::uint64_t>(element.kernelSize),
            static_cast<std::uint64_t>(element.inputChannels));
        randomizeElementWeights(element, node.seed);
        break;
      }
      case NodeType::groupInput:
      case NodeType::groupOutput:
        return failure(
            LiveGraphErrorCode::invalidGraph, node.id,
            "Group Input/Output hubs must be removed before live compilation");
      }

      std::uint64_t upstreamReceptiveField = 1;
      if (node.type != NodeType::audioInput) {
        const auto *rfNode = &node;
        if (compileFrozenSink)
          rfNode = nodesById.at(frozenGroups[frozenGroup->second].sourceId);
        for (const auto &pin : rfNode->inputs) {
          const auto source = sourceNodeByDestinationPin.find(pin.id);
          if (source == sourceNodeByDestinationPin.end())
            continue;
          const auto upstream = pathReceptiveField.find(source->second);
          if (upstream == pathReceptiveField.end())
            continue;
          upstreamReceptiveField =
              std::max(upstreamReceptiveField, upstream->second);
        }
      }
      pathReceptiveField[node.id] =
          saturatedAdd(upstreamReceptiveField, element.receptiveField - 1);
      compiled->parameterCount =
          saturatedAdd(compiled->parameterCount, element.parameterCount);
      compiled->statistics.push_back(
          {element.nodeId, element.type, element.inputChannels,
           element.outputChannels, element.receptiveField,
           element.parameterCount, element.randomizable});
      compiledIndex[node.id] = compiled->elements.size();
      compiled->elements.push_back(std::move(element));
    }

    compiled->receptiveField = pathReceptiveField.at(outputNodeId);
    LiveGraphCompileResult result;
    result.snapshot = std::shared_ptr<const LiveGraphSnapshot>(
        new LiveGraphSnapshot(std::move(compiled)));
    return result;
  } catch (const std::exception &exception) {
    return failure(LiveGraphErrorCode::torchFailure, 0, exception.what());
  }
}

/** @brief Prepares mutable histories, host storage, and frozen kernels. */
std::shared_ptr<LiveGraphRuntime>
LiveGraphEngine::prepare(std::shared_ptr<const LiveGraphSnapshot> snapshot,
                         LiveGraphCompileError &error) {
  error = {};
  if (!snapshot) {
    error.code = LiveGraphErrorCode::invalidGraph;
    error.message = "Cannot prepare a null live graph snapshot";
    return {};
  }

  try {
    auto runtime = std::make_unique<LiveGraphRuntime::Impl>();
    runtime->snapshot = snapshot;
    const auto &compiled = *snapshot->implementation;
    runtime->outputs.resize(compiled.elements.size());
    runtime->latentOutputs.resize(compiled.elements.size());
    runtime->histories.resize(compiled.elements.size());
    runtime->conditioningHistories.resize(compiled.elements.size());
    runtime->blackBoxKernels.resize(compiled.elements.size());
    runtime->inferenceMilliseconds =
        std::make_unique<std::atomic<double>[]>(compiled.elements.size());
    runtime->inputPeaks =
        std::make_unique<std::atomic<float>[]>(compiled.elements.size());
    runtime->outputPeaks =
        std::make_unique<std::atomic<float>[]>(compiled.elements.size());
    runtime->gainSmoothers.resize(compiled.elements.size());
    runtime->conditioningSmoothers.resize(compiled.elements.size());
    for (std::size_t index = 0; index < compiled.elements.size(); ++index) {
      runtime->inferenceMilliseconds[index].store(0.0,
                                                  std::memory_order_relaxed);
      runtime->inputPeaks[index].store(0.0f, std::memory_order_relaxed);
      runtime->outputPeaks[index].store(0.0f, std::memory_order_relaxed);
      const auto &element = compiled.elements[index];
      auto &gain = runtime->gainSmoothers[index];
      gain.reset(compiled.sampleRate, compiled.controlRampSeconds);
      gain.setCurrentAndTargetValue(element.gain);
      auto &x = runtime->conditioningSmoothers[index][0];
      auto &y = runtime->conditioningSmoothers[index][1];
      x.reset(compiled.sampleRate, compiled.controlRampSeconds);
      y.reset(compiled.sampleRate, compiled.controlRampSeconds);
      const auto initialX =
          element.type == graph::NodeType::knobInput ? element.conditioningValue
                                                     : element.conditioningX;
      x.setCurrentAndTargetValue(initialX);
      y.setCurrentAndTargetValue(element.conditioningY);
    }
    runtime->hostInput = torch::empty(
        {1, compiled.inputChannels, compiled.maximumBlockSize},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    for (std::size_t index = 0; index < compiled.elements.size(); ++index) {
      const auto &element = compiled.elements[index];
      if ((element.type == graph::NodeType::convolution ||
           element.type == graph::NodeType::convTranspose ||
           element.type == graph::NodeType::tcn ||
           element.type == graph::NodeType::blackBox ||
           element.type == graph::NodeType::rateConv ||
           element.type == graph::NodeType::pqmfAnalysis ||
           element.type == graph::NodeType::pqmfSynthesis ||
           element.type == graph::NodeType::variationalBottleneck) &&
          element.receptiveField > 1) {
        runtime->histories[index] = torch::zeros(
            {1, element.inputChannels,
             static_cast<std::int64_t>(element.receptiveField - 1)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      }
      if (element.type == graph::NodeType::blackBox) {
        runtime->blackBoxKernels[index] =
            element.blackBoxFactory->createKernel();
        if (!runtime->blackBoxKernels[index]) {
          error.code = LiveGraphErrorCode::invalidBlackBox;
          error.nodeId = element.nodeId;
          error.message = "Frozen BlackBox kernel could not be prepared";
          return {};
        }
      }
    }

    return std::shared_ptr<LiveGraphRuntime>(
        new LiveGraphRuntime(std::move(runtime)));
  } catch (const std::exception &exception) {
    error.code = LiveGraphErrorCode::torchFailure;
    error.message = exception.what();
    return {};
  }
}

RuntimeControlState
collectRuntimeControlState(const graph::NodeGraph &graphDocument) {
  RuntimeControlState controls;
  for (const auto &node : graphDocument.getNodes()) {
    if (node.type == graph::NodeType::activation ||
        node.type == graph::NodeType::tcn) {
      float gain = graph::gainDefault;
      for (const auto &property : node.properties) {
        if (property.key == "gain") {
          gain = property.kind == graph::PropertyKind::real
                     ? property.floatValue
                     : static_cast<float>(property.value);
          break;
        }
      }
      controls.gainByNodeId[node.id] = graph::clampGain(gain);
    }
    if (node.type == graph::NodeType::knobInput)
      controls.conditioningByNodeId[node.id] = {
          graph::clampConditioning(node.conditioningValue), 0.0f};
    else if (node.type == graph::NodeType::xyTrackpad)
      controls.conditioningByNodeId[node.id] = {
          graph::clampConditioning(node.conditioningX),
          graph::clampConditioning(node.conditioningY)};
    if (node.type == graph::NodeType::variationalBottleneck ||
        node.type == graph::NodeType::blackBox)
      controls.fidelityByNodeId[node.id] =
          graph::clampFidelity(node.fidelityPercent);
  }
  return controls;
}

namespace {
std::vector<AnalysisSeries> emptySeries(int channelCount) {
  std::vector<AnalysisSeries> series;
  series.reserve(static_cast<std::size_t>(std::max(channelCount, 0)));
  for (int channel = 0; channel < channelCount; ++channel) {
    AnalysisSeries trace;
    trace.channelIndex = channel;
    trace.channelLabel = analysisChannelLabel(channel, channelCount);
    series.push_back(std::move(trace));
  }
  return series;
}

void appendTransferPoint(std::vector<AnalysisSeries> &series, int channels,
                         float inputLevel, const torch::Tensor &output) {
  if (!output.defined() || output.dim() != 3)
    return;
  const auto count = std::min(channels, static_cast<int>(output.size(1)));
  for (int channel = 0; channel < count; ++channel) {
    const auto plane = output[0][channel];
    series[static_cast<std::size_t>(channel)].x.push_back(inputLevel);
    series[static_cast<std::size_t>(channel)].y.push_back(tensorPeak(plane));
  }
}

/**
 * @brief Builds a multi-channel sine probe for oscilloscope fallback.
 * @param channels Number of output channels.
 * @param samples Number of samples per channel.
 * @param sampleRate Host sample rate in Hz.
 * @param frequency Probe tone frequency in Hz.
 * @return Tensor shaped `[1, channels, samples]`.
 */
torch::Tensor makeSineProbe(int channels, int samples, double sampleRate,
                            float frequency) {
  auto probe = torch::zeros(
      {1, std::max(channels, 1), samples},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const auto twoPi = static_cast<float>(2.0 * 3.14159265358979323846);
  const auto phaseStep =
      sampleRate > 0.0
          ? twoPi * frequency / static_cast<float>(sampleRate)
          : 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    const auto phaseOffset =
        static_cast<float>(channel) * (twoPi / static_cast<float>(channels));
    for (int sample = 0; sample < samples; ++sample) {
      probe[0][channel][sample] =
          0.85f * std::sin(phaseStep * static_cast<float>(sample) + phaseOffset);
    }
  }
  return probe;
}

/**
 * @brief Fills analysis traces with a time-domain waveform.
 * @param series Destination traces, one per channel.
 * @param output Processed audio tensor shaped `[1, channels, samples]`.
 * @param sampleRate Host sample rate used to label the time axis in ms.
 */
void fillWaveform(std::vector<AnalysisSeries> &series,
                  const torch::Tensor &output, double sampleRate) {
  if (!output.defined() || output.dim() != 3)
    return;
  const auto samples = output.size(2);
  if (samples < 2)
    return;
  const auto channels = std::min(static_cast<int>(output.size(1)),
                                 static_cast<int>(series.size()));
  const auto timeScale =
      sampleRate > 0.0
          ? static_cast<float>(1000.0 / sampleRate)
          : 1.0f;
  for (int channel = 0; channel < channels; ++channel) {
    auto &trace = series[static_cast<std::size_t>(channel)];
    trace.x.clear();
    trace.y.clear();
    trace.x.reserve(static_cast<std::size_t>(samples));
    trace.y.reserve(static_cast<std::size_t>(samples));
    const auto plane = output[0][channel];
    for (std::int64_t sample = 0; sample < samples; ++sample) {
      trace.x.push_back(static_cast<float>(sample) * timeScale);
      trace.y.push_back(plane[sample].item<float>());
    }
  }
}

void fillSpectrum(std::vector<AnalysisSeries> &series, const torch::Tensor &input,
                  const torch::Tensor &output, graph::AnalysisView view,
                  double sampleRate) {
  if (!input.defined() || !output.defined() || input.dim() != 3 ||
      output.dim() != 3)
    return;
  const auto samples = std::min(input.size(2), output.size(2));
  if (samples < 8)
    return;
  const auto channels =
      std::min({static_cast<int>(input.size(1)), static_cast<int>(output.size(1)),
                static_cast<int>(series.size())});
  auto window = torch::hann_window(samples, torch::kFloat32);
  window = window.view({1, 1, samples});
  const auto inputSpec =
      torch::fft::rfft(input.narrow(2, 0, samples) * window, samples, 2);
  const auto outputSpec =
      torch::fft::rfft(output.narrow(2, 0, samples) * window, samples, 2);
  const auto bins = inputSpec.size(2);
  const auto frequencyScale =
      sampleRate > 0.0 ? static_cast<float>(sampleRate / static_cast<double>(samples))
                       : 1.0f;
  constexpr float epsilon = 1.0e-8f;
  for (int channel = 0; channel < channels; ++channel) {
    auto &trace = series[static_cast<std::size_t>(channel)];
    trace.x.clear();
    trace.y.clear();
    trace.x.reserve(static_cast<std::size_t>(bins));
    trace.y.reserve(static_cast<std::size_t>(bins));
    for (std::int64_t bin = 1; bin < bins; ++bin) {
      const auto xValue = inputSpec[0][channel][bin];
      const auto yValue = outputSpec[0][channel][bin];
      const auto transfer = yValue / (torch::abs(xValue) + epsilon);
      trace.x.push_back(static_cast<float>(bin) * frequencyScale);
      if (view == graph::AnalysisView::phase) {
        trace.y.push_back(static_cast<float>(
            torch::angle(transfer).item<float>() * (180.0f / 3.14159265f)));
      } else {
        const auto magnitude = torch::abs(transfer).item<float>();
        trace.y.push_back(20.0f * std::log10(magnitude + epsilon));
      }
    }
  }
}

std::optional<TransferMarker>
markerOnChain(const std::vector<AnalysisSeries> &chain, float inputPeak,
              int channelIndex) {
  if (chain.empty())
    return std::nullopt;
  const auto index = std::clamp(channelIndex, 0,
                                static_cast<int>(chain.size()) - 1);
  const auto &trace = chain[static_cast<std::size_t>(index)];
  if (trace.x.size() < 2 || trace.x.size() != trace.y.size())
    return std::nullopt;
  std::size_t best = 0;
  auto bestDistance = std::abs(trace.x.front() - inputPeak);
  for (std::size_t sample = 1; sample < trace.x.size(); ++sample) {
    const auto distance = std::abs(trace.x[sample] - inputPeak);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = sample;
    }
  }
  TransferMarker marker;
  marker.channelIndex = index;
  marker.inputLevel = trace.x[best];
  marker.outputLevel = trace.y[best];
  return marker;
}

/**
 * @brief Returns true when isolated analysis should inject live host audio.
 * @param elements Compiled elements in topological order.
 * @param nodeId Stable graph node being analysed.
 * @return False when every connected input is conditioning-only.
 */
bool elementUsesAudioProbe(const std::vector<CompiledElement> &elements,
                           std::int32_t nodeId) {
  for (const auto &element : elements) {
    if (element.nodeId != nodeId)
      continue;
    for (std::size_t slot = 0; slot < element.inputIndices.size(); ++slot) {
      if (slot < element.inputIsConditioning.size() &&
          element.inputIsConditioning[slot] != 0)
        continue;
      return true;
    }
    return false;
  }
  return false;
}
} // namespace

void LiveGraphRuntime::seedIsolatedUpstreamOutputs(std::size_t target,
                                                   const torch::Tensor &probe) {
  const auto &elements = implementation->snapshot->implementation->elements;
  const auto &element = elements[target];
  for (std::size_t slot = 0; slot < element.inputIndices.size(); ++slot) {
    const auto upstreamIndex =
        static_cast<std::size_t>(element.inputIndices[slot]);
    const auto &upstream = elements[upstreamIndex];
    const auto isConditioning =
        slot < element.inputIsConditioning.size() &&
        element.inputIsConditioning[slot] != 0;

    if (upstream.type == graph::NodeType::knobInput ||
        upstream.type == graph::NodeType::xyTrackpad || isConditioning ||
        upstream.outputIsConditioning) {
      executeElement(upstreamIndex, probe);
      continue;
    }

    if (upstream.type == graph::NodeType::audioInput) {
      implementation->outputs[upstreamIndex] = probe;
      continue;
    }

    const auto upstreamOutput = processIsolated(upstream.nodeId, probe);
    implementation->outputs[upstreamIndex] =
        upstreamOutput.defined() ? upstreamOutput : probe;
  }
}

AnalysisSnapshot
LiveGraphEngine::analyse(const graph::NodeGraph &graphDocument,
                         const AnalysisRequest &request,
                         const LiveGraphCompileOptions &options,
                         FrozenBlackBoxResolver blackBoxResolver) {
  AnalysisSnapshot snapshot;
  snapshot.nodeId = request.nodeId;
  snapshot.view = request.view;
  snapshot.generatedAtRevision = request.revision;
  snapshot.sourceMode = request.liveInputSuitable ? AnalysisSourceMode::live
                                                  : AnalysisSourceMode::probe;
  snapshot.status = request.liveInputSuitable ? AnalysisStatus::live
                                              : AnalysisStatus::probeFallback;

  const auto *node = graphDocument.findNode(request.nodeId);
  if (node == nullptr) {
    snapshot.status = AnalysisStatus::unavailable;
    return snapshot;
  }
  snapshot.runtimeState = node->state;

  auto compiled = compile(graphDocument, options, std::move(blackBoxResolver));
  if (!compiled.succeeded()) {
    snapshot.status = AnalysisStatus::disconnected;
    snapshot.channelCount = 1;
    snapshot.chainSeries = emptySeries(1);
    snapshot.elementOnlySeries = emptySeries(1);
    return snapshot;
  }

  LiveGraphCompileError error;
  auto runtime = prepare(compiled.snapshot, error);
  if (runtime == nullptr) {
    snapshot.status = AnalysisStatus::unavailable;
    return snapshot;
  }
  runtime->bindControls(std::make_shared<const RuntimeControlState>(
      collectRuntimeControlState(graphDocument)));

  int channelCount = compiled.snapshot->getInputChannels();
  int inputChannels = compiled.snapshot->getInputChannels();
  for (const auto &stats : compiled.snapshot->getElementStatistics()) {
    if (stats.nodeId != request.nodeId)
      continue;
    channelCount = std::max(1, stats.outputChannels);
    inputChannels = std::max(1, stats.inputChannels);
    break;
  }
  snapshot.channelCount = channelCount;
  snapshot.chainSeries = emptySeries(channelCount);
  snapshot.elementOnlySeries = emptySeries(channelCount);

  const auto makeInput = [&](float amplitude, int samples) {
    auto input = torch::full(
        {1, compiled.snapshot->getInputChannels(), samples}, amplitude,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    return input;
  };

  try {
    if (request.view == graph::AnalysisView::transfer) {
      constexpr int sweepPoints = 48;
      constexpr int blockSamples = 32;
      for (int point = 0; point < sweepPoints; ++point) {
        const auto amplitude =
            -1.0f + 2.0f * static_cast<float>(point) /
                        static_cast<float>(sweepPoints - 1);
        runtime->reset();
        const auto chain = runtime->processTensorTapped(
            makeInput(amplitude, blockSamples), request.nodeId);
        appendTransferPoint(snapshot.chainSeries, channelCount, amplitude,
                            chain);
        runtime->reset();
        auto probe = torch::full(
            {1, inputChannels, blockSamples}, amplitude,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        const auto isolated = runtime->processIsolated(request.nodeId, probe);
        appendTransferPoint(snapshot.elementOnlySeries, channelCount, amplitude,
                            isolated.defined() ? isolated : probe);
      }
      if (request.playbackActive)
        snapshot.transferMarker =
            markerOnChain(snapshot.chainSeries, request.liveInputPeak,
                          request.liveChannelIndex);
    } else if (request.view == graph::AnalysisView::oscilloscope) {
      constexpr int waveformSamples = 512;
      const auto &compiledElements =
          compiled.snapshot->implementation->elements;
      const auto useLiveCapture =
          elementUsesAudioProbe(compiledElements, request.nodeId) &&
          request.liveInputSuitable && request.liveInput != nullptr &&
          request.liveInputSamples > 8 && request.liveInputChannels > 0;
      torch::Tensor elementProbe;
      if (useLiveCapture) {
        const auto samples =
            std::min(request.liveInputSamples, waveformSamples);
        const auto channels =
            std::min(request.liveInputChannels, inputChannels);
        elementProbe = torch::zeros(
            {1, inputChannels, samples},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        for (int channel = 0; channel < channels; ++channel) {
          std::memcpy(elementProbe[0][channel].data_ptr<float>(),
                      request.liveInput + channel * request.liveInputSamples,
                      static_cast<std::size_t>(samples) * sizeof(float));
        }
        snapshot.sourceMode = AnalysisSourceMode::live;
        snapshot.status = AnalysisStatus::live;
      } else {
        elementProbe = makeSineProbe(inputChannels, waveformSamples,
                                     request.sampleRate, 220.0f);
        snapshot.sourceMode = AnalysisSourceMode::probe;
        snapshot.status = AnalysisStatus::probeFallback;
      }
      runtime->reset();
      const auto isolated =
          runtime->processIsolated(request.nodeId, elementProbe);
      fillWaveform(snapshot.elementOnlySeries,
                   isolated.defined() ? isolated : elementProbe,
                   request.sampleRate);
    } else {
      constexpr int spectrumSamples = 256;
      torch::Tensor probeInput;
      if (request.liveInputSuitable && request.liveInput != nullptr &&
          request.liveInputSamples > 8 && request.liveInputChannels > 0) {
        const auto samples =
            std::min(request.liveInputSamples, spectrumSamples);
        const auto channels = std::min(request.liveInputChannels,
                                       compiled.snapshot->getInputChannels());
        probeInput = torch::zeros(
            {1, compiled.snapshot->getInputChannels(), samples},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        for (int channel = 0; channel < channels; ++channel) {
          std::memcpy(probeInput[0][channel].data_ptr<float>(),
                      request.liveInput + channel * request.liveInputSamples,
                      static_cast<std::size_t>(samples) * sizeof(float));
        }
        snapshot.sourceMode = AnalysisSourceMode::live;
        snapshot.status = AnalysisStatus::live;
      } else {
        probeInput = torch::randn(
            {1, compiled.snapshot->getInputChannels(), spectrumSamples},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        snapshot.sourceMode = AnalysisSourceMode::probe;
        snapshot.status = AnalysisStatus::probeFallback;
      }
      runtime->reset();
      const auto chain =
          runtime->processTensorTapped(probeInput, request.nodeId);
      fillSpectrum(snapshot.chainSeries, probeInput,
                   chain.defined() ? chain : probeInput, request.view,
                   request.sampleRate);

      auto isolatedProbe = torch::randn(
          {1, inputChannels, spectrumSamples},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      runtime->reset();
      const auto isolated =
          runtime->processIsolated(request.nodeId, isolatedProbe);
      fillSpectrum(snapshot.elementOnlySeries, isolatedProbe,
                   isolated.defined() ? isolated : isolatedProbe, request.view,
                   request.sampleRate);
    }
  } catch (const std::exception &) {
    snapshot.status = AnalysisStatus::unavailable;
  }

  if (snapshot.chainSeries.size() !=
          static_cast<std::size_t>(snapshot.channelCount) ||
      snapshot.elementOnlySeries.size() !=
          static_cast<std::size_t>(snapshot.channelCount))
    snapshot.status = AnalysisStatus::unavailable;
  return snapshot;
}
} // namespace openyourbox::dsp
