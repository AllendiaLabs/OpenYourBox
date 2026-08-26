#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace openyourbox::graph {
/** @brief Semantic role of a node in the live processing graph. */
enum class NodeType {
  audioInput,
  audioOutput,
  linear,
  convolution,
  activation,
  tcn,
  merge,
  blackBox,
  knobInput,
  xyTrackpad,
  pqmfAnalysis,
  pqmfSynthesis,
  rateConv,
  variationalBottleneck,
  noiseSynthesizer,
  convTranspose,
  batchNorm
};

/** @brief Train recipe selected in the unified Train panel. */
enum class TrainObjective { mapping, reconstruction };

/** @brief Rate-changing convolution direction. */
enum class RateConvDirection { downsample, upsample };

/** @brief Merge element operating mode. */
enum class MergeMode : int { add = 0, multiply = 1, concatenate = 2 };

/** @brief Provenance shown by the Weights property on weight-bearing nodes. */
enum class WeightsProvenance {
  /** @brief Weights were drawn from a randomization seed. */
  random,
  /** @brief Weights were loaded from a trained or browsed file. */
  file
};

/** @brief How a Gold BlackBox was produced. */
enum class BlackBoxOrigin {
  /** @brief Manual Freeze Selection. */
  manualFreeze,
  /** @brief Successful Train auto-load. */
  trainAutoload
};

/** @brief Per-element analysis plot family requested by the editor. */
enum class AnalysisView {
  transfer = 0,
  frequency = 1,
  phase = 2,
  /** @brief Time-domain amplitude waveform (oscilloscope). */
  oscilloscope = 3
};

/** @brief Live modular processing node colour. */
inline const juce::Colour liveBlueColour{100, 180, 255};
/** @brief Frozen Gold node colour. */
inline const juce::Colour frozenGoldColour{218, 165, 32};
/** @brief Host audio-input node colour. */
inline const juce::Colour audioInputColour{70, 200, 150};
/** @brief Host audio-output node colour. */
inline const juce::Colour audioOutputColour{240, 160, 80};
/** @brief Conditioning source node colour for Knob Input and XY Trackpad. */
inline const juce::Colour conditioningColour{180, 140, 255};

/** @brief Inclusive lower bound for Knob/XY conditioning scalars. */
inline constexpr float conditioningMinimum = -10.0f;
/** @brief Inclusive upper bound for Knob/XY conditioning scalars. */
inline constexpr float conditioningMaximum = 10.0f;
/** @brief Inclusive lower bound for Activation/TCN Gain. */
inline constexpr float gainMinimum = 0.1f;
/** @brief Inclusive upper bound for Activation/TCN Gain. */
inline constexpr float gainMaximum = 10.0f;
/** @brief Neutral Gain value that leaves nonlinearity slope unchanged. */
inline constexpr float gainDefault = 1.0f;
/** @brief Default copies parameter N for a new group. */
inline constexpr int defaultGroupCopies = 1;
/** @brief Inclusive upper bound for a group's copies parameter. */
inline constexpr int maximumGroupCopies = 32;
/** @brief Supported nesting depth for groups and subgroups. */
inline constexpr int maximumGroupNestingDepth = 8;
/** @brief Bit flag applied to member pin ids drawn as group I/O pins. */
inline constexpr std::int32_t collapsedPinFlag = 0x20000000;
/** @brief Group chrome colour for expanded imgui-node-editor frames. */
inline const juce::Colour groupFrameColour{70, 130, 190};
/** @brief Highlight colour when an element will be added to a group. */
inline const juce::Colour groupDropHighlightColour{120, 210, 255};

/**
 * @brief Returns true when @p pinId is a collapsed-group virtual pin.
 * @param pinId Candidate pin identifier.
 */
inline bool isCollapsedGroupPin(std::int32_t pinId) noexcept {
  return (static_cast<std::uint32_t>(pinId) &
          static_cast<std::uint32_t>(collapsedPinFlag)) != 0;
}

/**
 * @brief Maps a collapsed-group virtual pin back to the member pin.
 * @param pinId Virtual or real pin identifier.
 */
inline std::int32_t resolveCollapsedPin(std::int32_t pinId) noexcept {
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(pinId) &
                                   ~static_cast<std::uint32_t>(collapsedPinFlag));
}

/**
 * @brief Builds a collapsed-group virtual pin id for a member pin.
 * @param memberPinId Underlying member pin identifier.
 */
inline std::int32_t collapsedGroupPinId(std::int32_t memberPinId) noexcept {
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(memberPinId) |
                                   static_cast<std::uint32_t>(collapsedPinFlag));
}

/** @brief Returns true for the undeletable host audio boundary nodes. */
inline bool isFixedIoType(NodeType type) noexcept {
  return type == NodeType::audioInput || type == NodeType::audioOutput;
}

/** @brief Returns true for Knob Input and XY Trackpad source elements. */
inline bool isConditioningSourceType(NodeType type) noexcept {
  return type == NodeType::knobInput || type == NodeType::xyTrackpad;
}

/** @brief Returns true for nodes that never participate in train arm/absorb. */
inline bool isControlSourceType(NodeType type) noexcept {
  return isFixedIoType(type) || isConditioningSourceType(type);
}

/** @brief Returns true for live elements that own trainable parameters. */
inline bool isTrainableType(NodeType type) noexcept {
  return type == NodeType::linear || type == NodeType::convolution ||
         type == NodeType::tcn || type == NodeType::activation ||
         type == NodeType::blackBox || type == NodeType::rateConv ||
         type == NodeType::variationalBottleneck ||
         type == NodeType::noiseSynthesizer ||
         type == NodeType::convTranspose || type == NodeType::batchNorm;
}

/** @brief Returns true for RAVE-specific processing elements. */
inline bool isRaveProcessingType(NodeType type) noexcept {
  return type == NodeType::pqmfAnalysis || type == NodeType::pqmfSynthesis ||
         type == NodeType::rateConv || type == NodeType::convTranspose ||
         type == NodeType::variationalBottleneck ||
         type == NodeType::noiseSynthesizer || type == NodeType::batchNorm;
}

/**
 * @brief Returns true for tensor ops that inherit hop rate and nBand.
 *
 * Linear, Conv1D, Activation, TCN, Merge, and Noise Synth process whatever
 * tensor they receive. Their pins start unspecified and copy the connected
 * upstream rate and band count.
 */
inline bool isShapePassthroughType(NodeType type) noexcept {
  return type == NodeType::linear || type == NodeType::convolution ||
         type == NodeType::activation || type == NodeType::tcn ||
         type == NodeType::merge || type == NodeType::noiseSynthesizer ||
         type == NodeType::batchNorm;
}

/**
 * @brief Returns true for the unified Conv1D element, including loaded Rate Conv.
 * @param type Graph node type.
 */
inline bool isConvolutionType(NodeType type) noexcept {
  return type == NodeType::convolution || type == NodeType::rateConv;
}

/** @brief Returns true for causal ConvTranspose1d upsampling elements. */
inline bool isConvTransposeType(NodeType type) noexcept {
  return type == NodeType::convTranspose;
}

/** @brief Returns true for Conv1D or ConvTranspose1d rate-changing elements. */
inline bool isRateChangingConvType(NodeType type) noexcept {
  return isConvolutionType(type) || isConvTransposeType(type);
}

/** @brief Default TCN dilation growth (RONN/WaveNet-like 2^n). */
inline constexpr int defaultDilationGrowth = 2;
/** @brief Inclusive lower bound for TCN dilation growth. */
inline constexpr int minimumDilationGrowth = 1;
/** @brief Inclusive upper bound for TCN dilation growth. */
inline constexpr int maximumDilationGrowth = 16;
/** @brief Default Train optimization steps (steerable NAfx recipe). */
inline constexpr int defaultTrainSteps = 2500;
/** @brief Default RF-aware train crop length in samples. */
inline constexpr int defaultTrainSegmentLength = 228308;
/** @brief Default Adam learning rate before the 80%/95% schedule. */
inline constexpr float defaultTrainLearningRate = 1.0e-3f;
/** @brief Default interval between hear-while-training checkpoint exports. */
inline constexpr int defaultTrainCheckpointInterval = 50;
/** @brief Default MLflow experiment name for Train logging. */
inline constexpr const char *defaultMlflowExperiment = "openyourbox";
/** @brief Default MLflow tracking server origin for Train logging. */
inline constexpr const char *defaultMlflowTrackingUri = "http://127.0.0.1:5000";
/** @brief Default PQMF band count matching acids-rave continuous layouts. */
inline constexpr int defaultPqmfBands = 16;
/** @brief Inclusive minimum PQMF band count (power-of-two layouts). */
inline constexpr int minimumPqmfBands = 2;
/** @brief Inclusive maximum PQMF band count. */
inline constexpr int maximumPqmfBands = 64;
/** @brief Default variational latent width. */
inline constexpr int defaultLatentSize = 128;
/** @brief Default reconstruction stage-1 (representation) steps. */
inline constexpr int defaultReconstructionStage1Steps = 1000000;
/** @brief Default reconstruction stage-2 (quality) steps. */
inline constexpr int defaultReconstructionStage2Steps = 1000000;
/** @brief Default bottleneck/Gold fidelity percent. */
inline constexpr float defaultFidelityPercent = 99.0f;
/** @brief Inclusive lower bound for fidelity. */
inline constexpr float fidelityMinimum = 0.0f;
/** @brief Inclusive upper bound for fidelity. */
inline constexpr float fidelityMaximum = 100.0f;
/** @brief Default filtered-noise band count (RAVE v1 noise). */
inline constexpr int defaultNoiseBands = 5;

/**
 * @brief Computes integer G^n without overflowing `int`.
 * @param growth Dilation growth G ≥ 1.
 * @param layer Zero-based layer index n.
 * @return Saturated `growth^layer`.
 */
inline int dilationGrowthPower(int growth, int layer) noexcept {
  if (layer <= 0)
    return 1;
  const auto g = std::max(1, growth);
  long long value = 1;
  for (int index = 0; index < layer; ++index) {
    if (value > static_cast<long long>(std::numeric_limits<int>::max()) / g)
      return std::numeric_limits<int>::max();
    value *= g;
  }
  return static_cast<int>(value);
}

/**
 * @brief Returns the per-layer dilation G^n for a TCN stack.
 * @param growth Dilation growth G.
 * @param layer Zero-based layer index.
 * @return Saturated layer dilation.
 */
inline int tcnLayerDilation(int growth, int layer) noexcept {
  return dilationGrowthPower(growth, layer);
}

/**
 * @brief Clamps a Gain value into the supported slope range.
 * @param gain Proposed Gain.
 * @return Gain in `[gainMinimum, gainMaximum]`.
 */
inline float clampGain(float gain) noexcept {
  return std::clamp(gain, gainMinimum, gainMaximum);
}

/**
 * @brief Clamps a fidelity percent into the supported range.
 * @param percent Proposed fidelity.
 * @return Value in `[fidelityMinimum, fidelityMaximum]`.
 */
inline float clampFidelity(float percent) noexcept {
  return std::clamp(percent, fidelityMinimum, fidelityMaximum);
}

/**
 * @brief Parses a persisted Train objective token.
 * @param name Objective string.
 * @return Mapping unless the token is reconstruction.
 */
inline TrainObjective trainObjectiveFromName(const std::string &name) noexcept {
  return name == "reconstruction" ? TrainObjective::reconstruction
                                  : TrainObjective::mapping;
}

/**
 * @brief Returns the persisted Train objective token.
 * @param objective Selected recipe.
 * @return `mapping` or `reconstruction`.
 */
inline const char *trainObjectiveName(TrainObjective objective) noexcept {
  return objective == TrainObjective::reconstruction ? "reconstruction"
                                                     : "mapping";
}

/**
 * @brief Clamps a conditioning scalar into the Knob/XY range.
 * @param value Proposed conditioning value.
 * @return Value in `[conditioningMinimum, conditioningMaximum]`.
 */
inline float clampConditioning(float value) noexcept {
  return std::clamp(value, conditioningMinimum, conditioningMaximum);
}

/** @brief Returns true for nodes that combine several input ports. */
inline bool isMixerType(NodeType type) noexcept {
  return type == NodeType::merge;
}

/** @brief Runtime mode represented by a graph node. */
enum class NodeState { liveBlue, frozenGold };

/** @brief Direction of a graph pin. */
enum class PinKind { input, output };

/** @brief Coarse tensor shape used for interactive link validation. */
struct ShapeSignature {
  /** @brief Channel count, or zero when inferred from the adjacent node. */
  int channels = 0;
  /**
   * @brief Hop product along the path (host audio = 1).
   *
   * Zero means unspecified and matches any rate. PQMF analysis multiplies
   * by `nBand`; downsample Conv1D multiplies by stride; upsample divides.
   */
  int temporalRate = 1;
  /** @brief PQMF band count when set, otherwise 0 (unspecified). */
  int nBand = 0;

  /**
   * @brief Returns whether hop rate, band count, and channels agree.
   * @param other Destination shape.
   */
  [[nodiscard]] bool
  isCompatibleWith(const ShapeSignature &other) const noexcept {
    if (temporalRate != other.temporalRate && temporalRate > 0 &&
        other.temporalRate > 0)
      return false;
    if (nBand > 0 && other.nBand > 0 && nBand != other.nBand)
      return false;
    return channels == 0 || other.channels == 0 || channels == other.channels;
  }

  /**
   * @brief Explains the first failing compatibility field.
   * @param other Destination shape.
   * @return Empty when compatible; otherwise a tooltip sentence.
   */
  [[nodiscard]] std::string
  incompatibilityMessage(const ShapeSignature &other) const {
    if (temporalRate != other.temporalRate && temporalRate > 0 &&
        other.temporalRate > 0)
      return "Shape mismatch: temporal rate " + std::to_string(temporalRate) +
             " cannot connect to " + std::to_string(other.temporalRate);
    if (nBand > 0 && other.nBand > 0 && nBand != other.nBand)
      return "Shape mismatch: nBand " + std::to_string(nBand) +
             " cannot connect to " + std::to_string(other.nBand);
    if (channels != 0 && other.channels != 0 && channels != other.channels)
      return "Shape mismatch: channel counts are incompatible";
    return {};
  }

  /**
   * @brief Builds a compact user-facing label for pin headers.
   * @return Empty when no concrete shape fields are known yet.
   */
  [[nodiscard]] std::string displayLabel() const {
    if (channels <= 0 && temporalRate <= 0 && nBand <= 0)
      return {};
    std::string label;
    if (channels > 0)
      label += std::to_string(channels) + "ch";
    if (nBand > 0) {
      if (!label.empty())
        label += " ";
      label += "n" + std::to_string(nBand);
    }
    if (temporalRate > 1) {
      if (!label.empty())
        label += " ";
      label += "×" + std::to_string(temporalRate);
    }
    return label;
  }
};

/**
 * @brief Declared host Audio Input/Output channel mode.
 *
 * Mono and Mirrored both fold stereo host audio with `(L+R)/2`. Mirrored then
 * copies that mono signal onto both channels so the graph stays 2-wide.
 */
enum class HostIoMode : int {
  /** @brief Single-channel path. */
  mono = 0,
  /** @brief Mono content duplicated across two channels (L = R). */
  mirrored = 1,
  /** @brief Independent left and right channels. */
  stereo = 2
};

/**
 * @brief Returns the graph pin width for a host I/O mode.
 * @param mode Declared Audio Input/Output mode.
 * @return 1 for Mono, 2 for Mirrored or Stereo.
 */
inline int hostIoChannelsFromMode(HostIoMode mode) noexcept {
  return mode == HostIoMode::mono ? 1 : 2;
}

/**
 * @brief Parses a persisted Channels choice index into a host I/O mode.
 * @param choiceIndex 0 = Mono, 1 = Mirrored, 2 = Stereo (legacy 1 = Stereo).
 * @param legacyStereoPair True when the property still used Mono|Stereo only.
 */
inline HostIoMode hostIoModeFromChoice(int choiceIndex,
                                       bool legacyStereoPair = false) noexcept {
  if (legacyStereoPair)
    return choiceIndex <= 0 ? HostIoMode::mono : HostIoMode::stereo;
  switch (choiceIndex) {
  case 0:
    return HostIoMode::mono;
  case 1:
    return HostIoMode::mirrored;
  default:
    return HostIoMode::stereo;
  }
}

/**
 * @brief Maps a mono/stereo width to a default host I/O mode.
 * @param channels Declared pin width.
 * @return Mono for 1 channel, otherwise Stereo.
 */
inline HostIoMode hostIoModeFromChannels(int channels) noexcept {
  return channels <= 1 ? HostIoMode::mono : HostIoMode::stereo;
}

/**
 * @brief Returns the Channels choice index for a host I/O mode.
 * @param mode Declared mode.
 */
inline int hostIoChoiceFromMode(HostIoMode mode) noexcept {
  return static_cast<int>(mode);
}

/**
 * @brief Short UI label for a host I/O mode.
 * @param mode Declared mode.
 */
inline const char *hostIoModeLabel(HostIoMode mode) noexcept {
  switch (mode) {
  case HostIoMode::mono:
    return "Mono";
  case HostIoMode::mirrored:
    return "Mirrored";
  case HostIoMode::stereo:
    break;
  }
  return "Stereo";
}

/**
 * @brief Detail sentence for a host I/O mode.
 * @param mode Declared mode.
 * @param isInput True for Audio Input.
 */
inline std::string hostIoModeDetail(HostIoMode mode, bool isInput) {
  const char *suffix = isInput ? " host input" : " host output";
  switch (mode) {
  case HostIoMode::mono:
    return std::string("1ch graph; sum (L+R)/2") + suffix;
  case HostIoMode::mirrored:
    return std::string("2ch L=R; sum (L+R)/2 then copy") + suffix;
  case HostIoMode::stereo:
    break;
  }
  return std::string("Independent L/R") + suffix;
}

/** @brief @deprecated Prefer `hostIoChannelsFromMode`. */
inline int hostIoChannelsFromChoice(int choiceIndex) noexcept {
  return hostIoChannelsFromMode(hostIoModeFromChoice(choiceIndex));
}

/** @brief @deprecated Prefer `hostIoChoiceFromMode`. */
inline int hostIoChoiceFromChannels(int channels) noexcept {
  return hostIoChoiceFromMode(hostIoModeFromChannels(channels));
}

/**
 * @brief Builds a tensor shape that inherits hop rate and nBand from a cable.
 * @param channels Declared channels, or 0 when inferred.
 */
inline ShapeSignature flexibleTensorShape(int channels = 0) noexcept {
  ShapeSignature shape;
  shape.channels = channels;
  shape.temporalRate = 0;
  shape.nBand = 0;
  return shape;
}

/**
 * @brief Computes Conv1D output hop rate from an incoming rate and stride.
 * @param inputRate Upstream temporal rate, or 0 when unknown.
 * @param stride Integer hop ≥ 1. Stride 1 never changes rate.
 * @param upsample True for ConvTranspose1d upsampling.
 * @return Output rate, 0 when unknown, or -1 when upsample stride does not
 *         divide @p inputRate.
 */
inline int convolutionOutputTemporalRate(int inputRate, int stride,
                                         bool upsample) noexcept {
  const auto hop = std::max(1, stride);
  if (hop == 1 || inputRate <= 0)
    return inputRate;
  if (upsample) {
    if (inputRate % hop != 0)
      return -1;
    return inputRate / hop;
  }
  if (inputRate > std::numeric_limits<int>::max() / hop)
    return std::numeric_limits<int>::max();
  return inputRate * hop;
}

/**
 * @brief User-facing Conv1D warning or error for the current stride settings.
 * @param stride Configured stride.
 * @param upsample True for ConvTranspose1d upsampling.
 * @param inputRate Incoming temporal rate, or 0 when unknown.
 * @return Empty when the configuration is valid and needs no notice.
 */
inline std::string convolutionRateMessage(int stride, bool upsample,
                                          int inputRate) {
  const auto hop = std::max(1, stride);
  if (hop == 1)
    return std::string{};
  if (upsample && inputRate > 0 && inputRate % hop != 0)
    return "Upsample stride must divide incoming temporal rate " +
           std::to_string(inputRate);
  return {};
}

/**
 * @brief Returns true when @p convolutionRateMessage is a hard error.
 * @param stride Configured stride.
 * @param upsample True for ConvTranspose1d upsampling.
 * @param inputRate Incoming temporal rate, or 0 when unknown.
 */
inline bool convolutionRateIsError(int stride, bool upsample,
                                   int inputRate) noexcept {
  const auto hop = std::max(1, stride);
  return hop > 1 && upsample && inputRate > 0 && inputRate % hop != 0;
}

/**
 * @brief User-facing error when multiband channels are not grouped by nBand.
 * @param channels Incoming channel count on a PQMF synthesis input.
 * @param nBand Configured synthesis band count.
 * @return Empty when the cable width is valid for synthesis.
 */
inline std::string pqmfSynthesisChannelMessage(int channels, int nBand) {
  if (channels <= 0 || nBand <= 0)
    return {};
  if (channels % nBand == 0)
    return {};
  return "PQMF synthesis requires channel count to be a multiple of nBand (got " +
         std::to_string(channels) + " for nBand " + std::to_string(nBand) +
         ")";
}

/**
 * @brief Returns true when @p pqmfSynthesisChannelMessage is a hard error.
 * @param channels Incoming channel count on a PQMF synthesis input.
 * @param nBand Configured synthesis band count.
 */
inline bool pqmfSynthesisChannelIsError(int channels, int nBand) noexcept {
  return channels > 0 && nBand > 0 && channels % nBand != 0;
}

/** @brief User-visible label of the TCN/BlackBox control (former FiLM) pin. */
inline constexpr const char *controlPinLabel = "control";
/** @brief User-visible label of Gold RAVE encode/decode latent pins. */
inline constexpr const char *latentPinLabel = "latent";

/** @brief A stable endpoint belonging to one graph node. */
struct Pin {
  /** @brief Stable graph-wide endpoint identifier. */
  std::int32_t id = 0;
  /** @brief User-visible endpoint label. */
  std::string label;
  /** @brief Input or output direction. */
  PinKind kind = PinKind::input;
  /** @brief Tensor shape accepted or produced by the endpoint. */
  ShapeSignature shape;
};

/**
 * @brief Returns true for the TCN/BlackBox control input (FiLM).
 *
 * Pin role only: any shape-compatible tensor may be wired here, including
 * host audio.
 * @param pin Endpoint to inspect.
 */
inline bool isControlInputPin(const Pin &pin) noexcept {
  return pin.kind == PinKind::input &&
         (pin.label == "film" || pin.label == controlPinLabel);
}

/**
 * @brief Returns true for Gold RAVE encode/decode latent endpoints.
 * @param pin Endpoint to inspect.
 */
inline bool isLatentPin(const Pin &pin) noexcept {
  return pin.label == latentPinLabel;
}

/** @brief Value type accepted by an inline graph property. */
enum class PropertyKind { integer, choice, readOnly, real };

/** @brief Ordered, validated inline property belonging to a graph node. */
struct NodeProperty {
  /** @brief Stable property key used for persistence and runtime binding. */
  std::string key;
  /** @brief User-visible property label. */
  std::string label;
  /** @brief Current validated integer or choice index. */
  int value = 0;
  /** @brief Inclusive minimum accepted value. */
  int minimum = std::numeric_limits<int>::min();
  /** @brief Inclusive maximum accepted value. */
  int maximum = std::numeric_limits<int>::max();
  /** @brief Editor control and parsing behavior. */
  PropertyKind kind = PropertyKind::integer;
  /** @brief Ordered labels used when the property is a choice. */
  std::vector<std::string> choices;
  /** @brief Current validated real value used by `PropertyKind::real`. */
  float floatValue = 0.0f;
  /** @brief Inclusive minimum accepted real value. */
  float floatMinimum = 0.0f;
  /** @brief Inclusive maximum accepted real value. */
  float floatMaximum = 1.0f;

  /** @brief Clamps and stores a proposed integer value. */
  void setValue(int proposed) noexcept {
    value = std::clamp(proposed, minimum, maximum);
  }

  /** @brief Clamps and stores a proposed real value. */
  void setFloatValue(float proposed) noexcept {
    floatValue = std::clamp(proposed, floatMinimum, floatMaximum);
  }
};

/** @brief Live performance values displayed by a frozen BlackBox node. */
struct NodeMetrics {
  /** @brief Backend compilation duration. */
  double compileTimeMilliseconds = 0.0;
  /** @brief Most recent per-buffer inference duration. */
  double inferenceTimeMilliseconds = 0.0;
};

/**
 * @brief Independent weight/artifact identity for one invisible group copy.
 *
 * Visible graph elements stay unique. DSP unrolls copies using these slots so
 * live/freeze seeds stay independent across copies. Training does not load
 * these seeds; it initializes with PyTorch defaults.
 */
struct CopyWeightSlot {
  /** @brief Randomization seed used when provenance is random. */
  std::int32_t seed = 42;
  /** @brief Whether this copy loads a file or a seed. */
  WeightsProvenance provenance = WeightsProvenance::random;
  /** @brief Filesystem path of trained or browsed weights. */
  std::string weightsPath;
  /** @brief Optional frozen artifact used by this copy. */
  std::string artifactPath;
};

/** @brief Named hierarchical container on the canvas. */
struct GraphGroup {
  /** @brief Stable identifier unique among nodes and groups. */
  std::int32_t id = 0;
  /** @brief User-visible group title. */
  std::string name = "Group";
  /** @brief Parent group, or empty at the canvas root. */
  std::optional<std::int32_t> parentGroupId;
  /** @brief Child node ids and nested group ids. */
  std::vector<std::int32_t> memberIds;
  /** @brief Presentation-only collapse flag (library insert still forces true). */
  bool collapsed = false;
  /** @brief Independent serial copy count (DSP-only; UI stays unique). */
  int copies = defaultGroupCopies;
  /** @brief Group box origin in parent-canvas or parent-group space. */
  juce::Point<float> position;
  /** @brief Group box size in parent-canvas or parent-group space. */
  juce::Point<float> size{220.0f, 140.0f};
  /** @brief Camera origin used when this group is the focused canvas. */
  juce::Point<float> viewPan;
  /** @brief Camera zoom used when this group is the focused canvas. */
  float viewZoom = 1.0f;
};

/** @brief Outcome of a group create/membership/copies mutation. */
struct GroupActionResult {
  /** @brief True when the graph was mutated. */
  bool accepted = false;
  /** @brief User-facing reason when the action was refused. */
  std::string message;
  /** @brief Created or targeted group id when accepted. */
  std::int32_t groupId = 0;
};

/** @brief Boundary-crossing port used by group I/O pins and copy chaining. */
struct GroupBoundaryPort {
  /** @brief Member pin that the group pin mediates. */
  std::int32_t memberPinId = 0;
  /** @brief Node that owns the member pin. */
  std::int32_t memberNodeId = 0;
  /** @brief Direction of the port on the group boundary. */
  PinKind kind = PinKind::input;
  /** @brief Shape copied from the member pin. */
  ShapeSignature shape;
  /** @brief Label shown on the group pin. */
  std::string label;
};

/** @brief Editable visual and processing description of one operation. */
struct GraphNode {
  /** @brief Stable graph-wide node identifier. */
  std::int32_t id = 0;
  /** @brief User-visible node title. */
  std::string label;
  /** @brief Secondary node status or architecture summary. */
  std::string detail;
  /** @brief Processing operation represented by the node. */
  NodeType type = NodeType::tcn;
  /** @brief Live Blue or frozen Gold runtime state. */
  NodeState state = NodeState::liveBlue;
  /** @brief Primary node accent colour. */
  juce::Colour colour{100, 180, 255};
  /** @brief Persisted canvas position. */
  juce::Point<float> position;
  /** @brief Last measured rendered node size. */
  juce::Point<float> size{180.0f, 120.0f};
  /** @brief Ordered input endpoints. */
  std::vector<Pin> inputs;
  /** @brief Ordered output endpoints. */
  std::vector<Pin> outputs;
  /** @brief Ordered inline property rows. */
  std::vector<NodeProperty> properties;
  /** @brief Whether this element owns mutable trainable parameters. */
  bool hasWeights = false;
  /** @brief Last applied randomization seed, including one-shot random draws. */
  std::int32_t seed = 42;
  /** @brief Memorized typed seed restored when the seed checkbox is re-enabled. */
  std::int32_t explicitSeed = 42;
  /** @brief True when Randomize Weights uses `explicitSeed` instead of a new random value. */
  bool useExplicitSeed = false;
  /** @brief Optional frozen-runtime performance values. */
  std::optional<NodeMetrics> metrics;
  /** @brief Compiled TorchScript artifact used by a BlackBox. */
  std::string artifactPath;
  /** @brief Serialized live source fragment used for unfreeze. */
  std::string sourceSubgraph;
  /** @brief Persisted analysis view preference for this element. */
  AnalysisView selectedAnalysisView = AnalysisView::transfer;
  /** @brief Current Knob Input conditioning scalar. */
  float conditioningValue = 0.0f;
  /** @brief Current XY Trackpad X conditioning scalar. */
  float conditioningX = 0.0f;
  /** @brief Current XY Trackpad Y conditioning scalar. */
  float conditioningY = 0.0f;
  /**
   * @brief Whether this trainable node is included in the next train snapshot.
   *
   * Default true on nodes with trainable parameters. Control sources ignore this.
   */
  bool armedForTraining = true;
  /** @brief Whether TCN blocks use a residual path. */
  bool residual = false;
  /** @brief TCN dilation growth G so layer n uses dilation G^n. */
  int dilationGrowth = defaultDilationGrowth;
  /** @brief Whether Weights currently shows a seed or a file path. */
  WeightsProvenance weightsProvenance = WeightsProvenance::random;
  /** @brief Filesystem path of trained or browsed weights when provenance is file. */
  std::string weightsPath;
  /** @brief How a Gold BlackBox was produced. */
  BlackBoxOrigin blackBoxOrigin = BlackBoxOrigin::manualFreeze;
  /** @brief Live fidelity percent applied inside bottleneck/Gold encode. */
  float fidelityPercent = defaultFidelityPercent;
  /** @brief True when compactness PCA buffers are present on the artifact. */
  bool compactnessReady = false;
  /** @brief Owning group, or empty when the node is on the root canvas. */
  std::optional<std::int32_t> parentGroupId;
  /**
   * @brief Per-copy weights for enclosing groups with N&gt;1.
   *
   * Slot 0 mirrors `seed` / `weightsPath` / `artifactPath`. Extra slots hold
   * independent copies that the UI never draws.
   */
  std::vector<CopyWeightSlot> copySlots;
};

/** @brief Directed connection between two graph pins. */
struct GraphLink {
  /** @brief Stable graph-wide link identifier. */
  std::int32_t id = 0;
  /** @brief Stable output pin identifier. */
  std::int32_t sourcePinId = 0;
  /** @brief Stable input pin identifier. */
  std::int32_t destinationPinId = 0;
};

/** @brief Persisted graph canvas navigation state. */
struct ViewportState {
  /** @brief Persisted canvas translation. */
  juce::Point<float> pan;
  /** @brief Persisted editor zoom clamped to supported bounds. */
  float zoom = 1.0f;
  /** @brief Whether the clickable overview map is visible. */
  bool mapVisible = true;
  /**
   * @brief Group whose interior is the current canvas, or empty at the graph root.
   */
  std::optional<std::int32_t> focusedGroupId;
};

/** @brief Outcome returned when an interactive connection is validated. */
struct ConnectionResult {
  /** @brief True when the link was committed. */
  bool accepted = false;
  /** @brief User-facing reason when the link was rejected. */
  std::string message;
};

/** @brief Serializable manual-freeze operation sent to the backend worker. */
struct FreezeSelectionRequest {
  /** @brief Unique request identifier used for response correlation. */
  std::string requestId;
  /** @brief Stable selected node identifiers. */
  std::vector<std::int32_t> selectedNodeIds;
  /** @brief JSON graph payload consumed by the backend worker. */
  std::string graphFragment;
};

/** @brief Serializable response produced by a manual-freeze worker. */
struct FreezeSelectionResult {
  /** @brief Request identifier echoed by the worker. */
  std::string requestId;
  /** @brief Whether compilation and artifact validation succeeded. */
  bool succeeded = false;
  /** @brief Absolute local path to the compiled TorchScript artifact. */
  std::string artifactPath;
  /** @brief User-facing failure reason. */
  std::string errorMessage;
  /** @brief Baseline compile and inference measurements. */
  NodeMetrics baselineMetrics;
  /** @brief Exact input channel count accepted by the artifact. */
  int inputChannels = 0;
  /** @brief Exact output channel count produced by the artifact. */
  int outputChannels = 0;
  /** @brief Causal receptive field required for continuous block processing. */
  std::uint64_t receptiveFieldSamples = 1;
  /** @brief True when the frozen module accepts a live Control tensor. */
  bool acceptsConditioning = false;
  /** @brief FiLM control width the artifact was traced with, or 0 if unused. */
  int condDim = 0;
  /** @brief True when the artifact exports encode/decode in addition to forward. */
  bool hasEncodeDecode = false;
};

/** @brief Serializable train job sent to the Python train worker. */
struct TrainJobRequest {
  /** @brief Unique request identifier used for response correlation. */
  std::string requestId;
  /** @brief JSON envelope consumed by `train_worker.py`. */
  std::string graphFragment;
  /** @brief Armed trainable node identifiers captured at Run. */
  std::vector<std::int32_t> armedNodeIds;
};

/** @brief Progress or completion snapshot produced by a train worker. */
struct TrainJobResult {
  /** @brief Request identifier echoed by the worker. */
  std::string requestId;
  /** @brief Latest status string (`running`, `paused`, `success`, ...). */
  std::string status;
  /** @brief Current optimization step. */
  int step = 0;
  /** @brief Configured total steps. */
  int totalSteps = 2500;
  /** @brief Latest scalar loss. */
  double loss = 0.0;
  /** @brief Lowest loss observed while the job was running. */
  double bestLoss = 0.0;
  /** @brief Current learning rate. */
  double learningRate = 1.0e-3;
  /** @brief Absolute trained TorchScript path on success. */
  std::string artifactPath;
  /** @brief User-facing failure or stop message. */
  std::string errorMessage;
  /** @brief Exact input channel count accepted by the artifact. */
  int inputChannels = 0;
  /** @brief Exact output channel count produced by the artifact. */
  int outputChannels = 0;
  /** @brief Causal receptive field of the trained model. */
  std::uint64_t receptiveFieldSamples = 1;
  /** @brief True when the trained module accepts a conditioning tensor. */
  bool acceptsConditioning = false;
  /** @brief FiLM control width the artifact was traced with, or 0 if unknown. */
  int condDim = 0;
  /** @brief Reconstruction stage token (`representation`|`quality`), else empty. */
  std::string stage;
  /** @brief Selected Train objective echoed by the worker. */
  std::string objective;
  /** @brief True when the artifact exposes encode/decode methods. */
  bool hasEncodeDecode = false;
};

/** @brief Inclusive lower bound for element randomization seeds. */
inline constexpr std::int32_t minimumSeed = 0;
/** @brief Inclusive upper bound for element randomization seeds. */
inline constexpr std::int32_t maximumSeed = 999999;

/**
 * @brief Clamps a seed into the supported UI range.
 * @param seed Proposed seed value.
 * @return Seed in `[minimumSeed, maximumSeed]`.
 */
inline std::int32_t clampSeed(std::int32_t seed) noexcept {
  return std::clamp(seed, minimumSeed, maximumSeed);
}

/**
 * @brief Derives a per-copy randomization seed from a visible base seed.
 * @param baseSeed Seed shown on the editable template element (slot 0).
 * @param copyIndex Zero-based materialized copy index.
 * @return `baseSeed + copyIndex` wrapped into `[minimumSeed, maximumSeed]`.
 *
 * Live audition and freeze use these seeds. Training builds fresh PyTorch
 * modules with framework default initialization and does not consume them.
 */
inline std::int32_t seedForCopySlot(std::int32_t baseSeed,
                                   std::size_t copyIndex) noexcept {
  constexpr auto span =
      static_cast<std::int64_t>(maximumSeed) - minimumSeed + 1;
  const auto mixed =
      (static_cast<std::int64_t>(clampSeed(baseSeed)) -
       minimumSeed + static_cast<std::int64_t>(copyIndex)) %
      span;
  return static_cast<std::int32_t>(minimumSeed +
                                  (mixed < 0 ? mixed + span : mixed));
}

/** @brief Minimum supported node-editor zoom level. */
inline constexpr float minimumZoom = 0.25f;
/** @brief Maximum supported node-editor zoom level. */
inline constexpr float maximumZoom = 2.0f;
/** @brief Default minimap width in Dear ImGui pixels. */
inline constexpr float mapWidth = 190.0f;
/** @brief Default minimap height in Dear ImGui pixels. */
inline constexpr float mapHeight = 125.0f;
/** @brief Padding around members when a group is created or fitted. */
inline constexpr float groupFitPadding = 28.0f;
/** @brief Extra top padding reserved for the group header and copies control. */
inline constexpr float groupHeaderHeight = 52.0f;
/** @brief Inset from a group box origin used when mapping member coordinates. */
inline constexpr float groupContentPad = 8.0f;

/**
 * @brief Offset from a group box origin used when mapping member coordinates.
 */
inline juce::Point<float> groupContentOffset() noexcept {
  return {groupContentPad, groupHeaderHeight};
}

/**
 * @brief Captures a node's primary weight fields into a copy slot.
 * @param node Source element.
 */
inline CopyWeightSlot copySlotFromNode(const GraphNode &node) {
  CopyWeightSlot slot;
  slot.seed = node.seed;
  slot.provenance = node.weightsProvenance;
  slot.weightsPath = node.weightsPath;
  slot.artifactPath = node.artifactPath;
  return slot;
}

/**
 * @brief Writes one copy slot onto a node's primary weight fields.
 * @param node Destination element.
 * @param slot Independent copy identity to apply.
 */
inline void applyCopySlot(GraphNode &node, const CopyWeightSlot &slot) {
  node.seed = slot.seed;
  node.weightsProvenance = slot.provenance;
  node.weightsPath = slot.weightsPath;
  if (!slot.artifactPath.empty())
    node.artifactPath = slot.artifactPath;
}

/**
 * @brief Ensures @p node has @p count copy slots, deriving or cloning on growth.
 * @param node Element whose enclosing groups changed copy count.
 * @param count Required slot count (≥ 1).
 *
 * Random-provenance slots use `seedForCopySlot` so raising N does not leave
 * identical live weights. File-provenance slots still clone the previous last
 * path (FR-017c).
 */
inline void ensureCopySlotCount(GraphNode &node, int count) {
  const auto target = std::max(1, count);
  if (node.copySlots.empty())
    node.copySlots.push_back(copySlotFromNode(node));
  node.copySlots.front() = copySlotFromNode(node);
  while (static_cast<int>(node.copySlots.size()) < target) {
    CopyWeightSlot slot = node.copySlots.back();
    const auto index = node.copySlots.size();
    if (slot.provenance == WeightsProvenance::random)
      slot.seed = seedForCopySlot(node.seed, index);
    node.copySlots.push_back(std::move(slot));
  }
  if (static_cast<int>(node.copySlots.size()) > target)
    node.copySlots.resize(static_cast<std::size_t>(target));
  applyCopySlot(node, node.copySlots.front());
}
} // namespace openyourbox::graph
