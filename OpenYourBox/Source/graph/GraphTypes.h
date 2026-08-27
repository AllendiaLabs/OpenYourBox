#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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
  batchNorm,
  /** Editor-only source hub declaring a group's external input lanes. */
  groupInput,
  /** Editor-only sink hub declaring a group's external output lanes. */
  groupOutput
};

/** @brief Train recipe selected in the unified Train panel. */
enum class TrainObjective { mapping, reconstruction };

/** @brief Rate-changing convolution direction. */
enum class RateConvDirection { downsample, upsample };

/** @brief Utility element operating mode (add, multiply, or concatenate). */
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
/** @brief Default LeakyReLU negative slope (PyTorch default). */
inline constexpr float leakyReluNegativeSlopeDefault = 0.01f;
/** @brief Inclusive lower bound for LeakyReLU negative slope. */
inline constexpr float leakyReluNegativeSlopeMinimum = 0.0f;
/** @brief Inclusive upper bound for LeakyReLU negative slope. */
inline constexpr float leakyReluNegativeSlopeMaximum = 1.0f;
/** @brief Activation choice index for LeakyReLU. */
inline constexpr int leakyReluActivationIndex = 3;
/** @brief Reserved token binding a dim/channels/features field to its input. */
inline constexpr const char *preserveInToken = "in";
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

/** @brief Returns true for editor-only Group Input and Group Output hubs. */
inline bool isGroupBoundaryType(NodeType type) noexcept {
  return type == NodeType::groupInput || type == NodeType::groupOutput;
}

/** @brief Returns true for Knob Input and XY Trackpad source elements. */
inline bool isConditioningSourceType(NodeType type) noexcept {
  return type == NodeType::knobInput || type == NodeType::xyTrackpad;
}

/** @brief Returns true for nodes that never participate in train arm/absorb. */
inline bool isControlSourceType(NodeType type) noexcept {
  return isFixedIoType(type) || isConditioningSourceType(type) ||
         isGroupBoundaryType(type);
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
 * Linear, Conv1D, Activation, TCN, Utility, and Noise Synth process whatever
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
/** @brief No practical upper bound for free-form integer graph properties. */
inline constexpr int unlimitedPropertyMaximum = std::numeric_limits<int>::max();
/** @brief Inclusive minimum for positive integer graph properties. */
inline constexpr int minimumPositiveProperty = 1;
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
/**
 * @brief Default variational-head temporal kernel (acids-rave v1).
 *
 * The grouped mean/variance convolution uses this width unless the user edits
 * `kernel_size`. Input channel count must be even so `groups=2` can split the
 * encoder features across the μ branch and the variance branch.
 */
inline constexpr int defaultBottleneckKernelSize = 5;
/** @brief Fixed grouped-head group count (mean branch + variance branch). */
inline constexpr int variationalBottleneckGroups = 2;
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

/** @brief Returns true for Utility nodes (one or more combine/passthrough inputs). */
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
 * @brief Formats copy-slot labels as nested @c [] lists along the nest axes.
 *
 * @p copyCounts is outer→inner. Axes of 1 are skipped. A single ungrouped
 * item is returned without brackets. When @p items length does not match the
 * product of @p copyCounts, the items are shown as one flat bracketed list.
 * @param items One label per expanded copy slot.
 * @param copyCounts Outer→inner ancestor copy-count vector C.
 */
inline std::string
formatHierarchicalCopyList(const std::vector<std::string> &items,
                           const std::vector<int> &copyCounts) {
  if (items.empty())
    return {};

  const auto joinRange = [&items](int start, int count) {
    std::string text = "[";
    for (int index = 0; index < count; ++index) {
      if (index > 0)
        text += ", ";
      text += items[static_cast<std::size_t>(start + index)];
    }
    text += "]";
    return text;
  };

  int product = 1;
  for (const auto copies : copyCounts)
    product *= std::max(1, copies);
  product = std::max(1, product);
  const auto count = static_cast<int>(items.size());
  if (product != count)
    return count == 1 ? items.front() : joinRange(0, count);

  struct Formatter {
    const std::vector<std::string> &items;
    const std::vector<int> &copyCounts;
    /**
     * @brief Nests @p span items starting at @p start from axis @p dim.
     * @param start First item index.
     * @param span Number of items in this axis slice.
     * @param dim Current outer→inner copy-count index.
     */
    std::string format(int start, int span, int dim) const {
      if (span <= 0)
        return {};
      while (dim < static_cast<int>(copyCounts.size()) &&
             copyCounts[static_cast<std::size_t>(dim)] <= 1)
        ++dim;
      if (dim >= static_cast<int>(copyCounts.size())) {
        if (span == 1)
          return items[static_cast<std::size_t>(start)];
        std::string text = "[";
        for (int index = 0; index < span; ++index) {
          if (index > 0)
            text += ", ";
          text += items[static_cast<std::size_t>(start + index)];
        }
        text += "]";
        return text;
      }
      const int groups =
          std::max(1, copyCounts[static_cast<std::size_t>(dim)]);
      if (span % groups != 0) {
        std::string text = "[";
        for (int index = 0; index < span; ++index) {
          if (index > 0)
            text += ", ";
          text += items[static_cast<std::size_t>(start + index)];
        }
        text += "]";
        return text;
      }
      const int inner = span / groups;
      std::string text = "[";
      for (int group = 0; group < groups; ++group) {
        if (group > 0)
          text += ", ";
        text += format(start + group * inner, inner, dim + 1);
      }
      text += "]";
      return text;
    }
  };
  return Formatter{items, copyCounts}.format(0, count, 0);
}

/**
 * @brief Formats per-copy pin shapes as a nested copy-hierarchy label.
 * @param shapes Ordered shapes (one per copy). Falls back to @p fallback when
 *        empty.
 * @param fallback Shape used when @p shapes is empty (typically the first copy).
 * @param copyCounts Outer→inner ancestor copy counts used to nest @c [] groups.
 * @return Empty when no concrete shape fields are known yet.
 */
inline std::string formatShapeCopyList(const std::vector<ShapeSignature> &shapes,
                                       const ShapeSignature &fallback,
                                       const std::vector<int> &copyCounts = {}) {
  std::vector<std::string> labels;
  if (shapes.size() <= 1) {
    const auto &shape = shapes.empty() ? fallback : shapes.front();
    auto label = shape.displayLabel();
    if (label.empty())
      return {};
    labels.push_back(std::move(label));
  } else {
    labels.reserve(shapes.size());
    for (const auto &shape : shapes) {
      auto part = shape.displayLabel();
      if (part.empty())
        part = "?";
      labels.push_back(std::move(part));
    }
  }
  return formatHierarchicalCopyList(labels, copyCounts);
}

/**
 * @brief Collapses an inner serial-copy axis to first-in or last-out.
 *
 * @p innerFold is the contiguous innermost chunk width. 1 keeps @p shapes.
 * When @p shapes length is not a multiple of @p innerFold, the whole list
 * reduces to a single first or last entry.
 * @param shapes Flat copy-slot shapes, innermost axis contiguous.
 * @param innerFold Width of the inner axis to fold.
 * @param takeLast True to keep the last slot of each chunk (outputs).
 * @return One shape per remaining outer slot.
 */
inline std::vector<ShapeSignature>
foldInnerCopyShapes(const std::vector<ShapeSignature> &shapes, int innerFold,
                    bool takeLast) {
  innerFold = std::max(1, innerFold);
  if (shapes.empty() || innerFold <= 1)
    return shapes;
  if (static_cast<int>(shapes.size()) % innerFold != 0)
    return {takeLast ? shapes.back() : shapes.front()};
  std::vector<ShapeSignature> folded;
  folded.reserve(shapes.size() / static_cast<std::size_t>(innerFold));
  for (std::size_t start = 0; start < shapes.size();
       start += static_cast<std::size_t>(innerFold)) {
    folded.push_back(takeLast ? shapes[start + static_cast<std::size_t>(
                                                  innerFold - 1)]
                              : shapes[start]);
  }
  return folded;
}

/**
 * @brief Formats collapsed group-box pin shapes for the parent canvas.
 *
 * Copies of the group itself — and of any nested groups whose copy axes
 * leaked onto this hub — chain serially, so the parent canvas only attaches
 * to first-copy inputs and last-copy outputs. Those inner shapes are folded
 * away. Strict ancestor copy axes remain as a nested list.
 *
 * @param shapes Per-copy shapes; may include the group's own axis and nested
 *        descendant copy axes inside it.
 * @param fallback Shape used when @p shapes is empty (typically first-copy).
 * @param ancestorCopyCounts Outer→inner runtime copy counts for the hub,
 *        including the collapsed group's own N as the last entry.
 * @param takeLast True for outputs (last copy of each inner chunk); false for
 *        inputs (first copy of each inner chunk).
 * @return Hierarchical label, or empty when no concrete shape is known.
 */
inline std::string formatCollapsedGroupPinShapes(
    const std::vector<ShapeSignature> &shapes, const ShapeSignature &fallback,
    const std::vector<int> &ancestorCopyCounts, bool takeLast) {
  std::vector<int> outerCounts = ancestorCopyCounts;
  if (!outerCounts.empty())
    outerCounts.pop_back();
  int outerProduct = 1;
  for (const auto copies : outerCounts)
    outerProduct *= std::max(1, copies);
  outerProduct = std::max(1, outerProduct);
  const int shapeCount = static_cast<int>(shapes.size());
  int innerFold = 1;
  if (shapeCount > 0 && shapeCount % outerProduct == 0)
    innerFold = std::max(1, shapeCount / outerProduct);
  else if (shapeCount > 0)
    innerFold = shapeCount;
  const auto folded = foldInnerCopyShapes(shapes, innerFold, takeLast);
  const ShapeSignature &effectiveFallback =
      !folded.empty()
          ? folded.front()
          : (takeLast && !shapes.empty() ? shapes.back() : fallback);
  return formatShapeCopyList(folded, effectiveFallback, outerCounts);
}

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

/**
 * @brief User-facing error when a variational bottleneck sees an odd width.
 * @param channels Incoming encoder feature count.
 * @return Empty when @p channels is unknown or even.
 */
inline std::string variationalBottleneckChannelMessage(int channels) {
  if (channels <= 0 || channels % variationalBottleneckGroups == 0)
    return {};
  return "Variational bottleneck requires an even channel count "
         "(grouped mean/variance head).";
}

/**
 * @brief Returns true when @p variationalBottleneckChannelMessage is an error.
 * @param channels Incoming encoder feature count.
 */
inline bool variationalBottleneckChannelIsError(int channels) noexcept {
  return channels > 0 && channels % variationalBottleneckGroups != 0;
}

/**
 * @brief User-facing error when latent width cannot split across two groups.
 * @param latentSize Proposed latent width.
 * @return Empty when @p latentSize is even and positive.
 */
inline std::string variationalBottleneckLatentMessage(int latentSize) {
  if (latentSize > 0 && latentSize % variationalBottleneckGroups == 0)
    return {};
  return "Variational bottleneck latent size must be even "
         "(grouped mean/variance head).";
}

/**
 * @brief Returns true when @p variationalBottleneckLatentMessage is an error.
 * @param latentSize Proposed latent width.
 */
inline bool variationalBottleneckLatentIsError(int latentSize) noexcept {
  return latentSize < 1 || latentSize % variationalBottleneckGroups != 0;
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
  /**
   * @brief Tensor shape for the visible (first) copy.
   *
   * External cables into a multi-copy group still validate against this shape.
   * Group outputs that leave the stack use @ref copyShapes when present.
   */
  ShapeSignature shape;
  /**
   * @brief Per-copy shapes when the owner participates in N&gt;1 group copies.
   *
   * Empty when the runtime copy product is 1. Index 0 matches @ref shape.
   * The innermost axis is the owner's group; ancestor axes nest outside it.
   * Collapsed group boxes fold that inner axis to first-in / last-out.
   */
  std::vector<ShapeSignature> copyShapes;
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
  /**
   * @brief Per-copy integer values when the owner sits in a group with N&gt;1.
   *
   * Empty means every copy uses `value`. Slot 0 mirrors `value`.
   */
  std::vector<int> copyIntValues;
  /**
   * @brief Per-copy real values when the owner sits in a group with N&gt;1.
   *
   * Empty means every copy uses `floatValue`. Slot 0 mirrors `floatValue`.
   * Length is the **authored** list length L (may be less than P).
   */
  std::vector<float> copyFloatValues;
  /**
   * @brief True when authored length L is not in the nest dividing set.
   *
   * Authored values are preserved; runtime and shape inference treat the
   * property as unresolved until the user commits a legal list.
   */
  bool copyListInvalid = false;
  /** @brief User-facing reason when @ref copyListInvalid is set. */
  std::string copyListInvalidMessage;
  /**
   * @brief True when this integer property is bound to the `in` keyword.
   *
   * Authored length is `copyIntValues.size()` (or 1 when empty). Resolved
   * integers are derived from the paired input shape, not persisted as the
   * source of truth.
   */
  bool preserveInBound = false;

  /** @brief Clamps and stores a proposed integer value. */
  void setValue(int proposed) noexcept {
    value = std::clamp(proposed, minimum, maximum);
  }

  /** @brief Clamps and stores a proposed real value. */
  void setFloatValue(float proposed) noexcept {
    floatValue = std::clamp(proposed, floatMinimum, floatMaximum);
  }
};

/**
 * @brief Returns true when a property keeps fixed choice or boolean bounds.
 * @param property Node property to inspect.
 */
inline bool propertyKeepsFixedBounds(const NodeProperty &property) noexcept {
  return property.kind == PropertyKind::choice || property.key == "residual" ||
         property.key == "weight_norm";
}

/**
 * @brief Removes artificial upper bounds on free-form integer properties.
 * @param property Node property to widen.
 */
inline void widenIntegerPropertyBounds(NodeProperty &property) noexcept {
  if (property.kind == PropertyKind::real || propertyKeepsFixedBounds(property))
    return;
  property.maximum = unlimitedPropertyMaximum;
}

/**
 * @brief Outcome of parsing a comma-separated per-copy property list.
 */
struct PropertyCopyListParse {
  /** @brief True when the text is a legal dividing-set length (or all-`in`). */
  bool accepted = false;
  /** @brief User-facing reason when parsing failed. */
  std::string message;
  /** @brief Parsed authored integers (length L, not expanded to P). */
  std::vector<int> intValues;
  /** @brief Parsed authored reals (length L, not expanded to P). */
  std::vector<float> floatValues;
  /** @brief True when every token is the reserved `in` keyword. */
  bool preserveIn = false;
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

/**
 * @brief One shape-driving property implicated by an inactive copy request.
 */
struct GroupCopyPropertyHint {
  /** @brief Element containing the candidate property. */
  std::int32_t nodeId = 0;
  /** @brief Canonical property key such as `channels` or `features`. */
  std::string propertyKey;
  /** @brief User-facing explanation for highlighting this property. */
  std::string message;
};

/**
 * @brief Derived validity and diagnostics for a group's requested copies.
 */
struct GroupCopyStatus {
  /** @brief User-authored copy count stored on the group. */
  int requestedCopies = defaultGroupCopies;
  /** @brief Count safe to materialize; one while the request is inactive. */
  int effectiveCopies = defaultGroupCopies;
  /** @brief True when requested N can chain after per-copy properties. */
  bool active = true;
  /** @brief Persistent user-facing explanation when inactive. */
  std::string message;
  /** @brief Candidate parameters that can make the chain shape-compatible. */
  std::vector<GroupCopyPropertyHint> propertyHints;
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
  /** @brief Validation-PCA mean `[latent]`, empty until compactness is ready. */
  std::vector<float> latentMean;
  /** @brief Validation-PCA basis `[latent × latent]` row-major. */
  std::vector<float> latentPca;
  /** @brief Linear singular-value cumulative ratios `[latent]`. */
  std::vector<float> cumulativeVariance;
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
  /**
   * @brief Last-visited descendant groups under the current focus (parent→deep).
   *
   * Shown in the hierarchy trail until the user opens a different branch.
   */
  std::vector<std::int32_t> stickySpine;
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
  /** @brief True when validation-PCA compactness buffers are ready. */
  bool compactnessReady = false;
  /** @brief Validation segments used for PCA, or 0 when not ready. */
  int compactnessValidationSegments = 0;
  /** @brief Compactness status token (`ready`|`not_ready`). */
  std::string compactnessStatus = "not_ready";
};

/**
 * @brief Clears compactness PCA state after weight randomization.
 * @param node Bottleneck or Gold node to reset.
 */
inline void clearCompactness(GraphNode &node) noexcept {
  node.compactnessReady = false;
  node.latentMean.clear();
  node.latentPca.clear();
  node.cumulativeVariance.clear();
}

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

/**
 * @brief Returns true when a property can store one numeric value per copy.
 * @param property Candidate inline property.
 */
inline bool propertySupportsCopyValueList(const NodeProperty &property) noexcept {
  if (property.kind != PropertyKind::integer &&
      property.kind != PropertyKind::real)
    return false;
  return property.key != "residual" && property.key != "weight_norm" &&
         property.key != "inputs" && property.key != "ports";
}

/**
 * @brief Returns true when @p property may bind to the reserved `in` token.
 * @param property Candidate inline property.
 */
inline bool propertySupportsPreserveIn(const NodeProperty &property) noexcept {
  if (property.kind != PropertyKind::integer)
    return false;
  return property.key == "features" || property.key == "channels";
}

/**
 * @brief Product of ancestor copy counts (P), or 1 when @p copyCounts is empty.
 * @param copyCounts Outer→inner copy-count vector.
 */
inline int copyCountProduct(const std::vector<int> &copyCounts) noexcept {
  int product = 1;
  for (const auto copies : copyCounts)
    product *= std::max(1, copies);
  return std::max(1, product);
}

/**
 * @brief Dividing-set lengths D(C) = {1} ∪ suffix products of @p copyCounts.
 * @param copyCounts Outer→inner copy-count vector C.
 * @return Sorted unique legal authored lengths including 1 and P.
 */
inline std::vector<int>
dividingSetLengths(const std::vector<int> &copyCounts) {
  std::vector<int> lengths;
  lengths.push_back(1);
  int product = 1;
  for (int index = static_cast<int>(copyCounts.size()) - 1; index >= 0;
       --index) {
    product *= std::max(1, copyCounts[static_cast<std::size_t>(index)]);
    lengths.push_back(product);
  }
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  return lengths;
}

/**
 * @brief Returns true when @p length is in D(@p copyCounts).
 * @param length Authored list length L.
 * @param copyCounts Outer→inner copy-count vector.
 */
inline bool isDividingSetLength(int length,
                                const std::vector<int> &copyCounts) noexcept {
  if (length < 1)
    return false;
  for (const auto allowed : dividingSetLengths(copyCounts)) {
    if (allowed == length)
      return true;
  }
  return false;
}

/**
 * @brief User-facing list of allowed authored lengths for @p label.
 * @param label Property label.
 * @param copyCounts Outer→inner copy-count vector.
 */
inline std::string
formatAllowedCopyListLengths(const std::string &label,
                             const std::vector<int> &copyCounts) {
  const auto allowed = dividingSetLengths(copyCounts);
  std::string text = label + " needs a list of length ";
  for (std::size_t index = 0; index < allowed.size(); ++index) {
    if (index > 0 && index + 1 == allowed.size())
      text += ", or ";
    else if (index > 0)
      text += ", ";
    text += std::to_string(allowed[index]);
  }
  return text;
}

/**
 * @brief Authored list length L stored on @p property (at least 1).
 * @param property Source property.
 */
inline int authoredCopyListLength(const NodeProperty &property) noexcept {
  if (property.kind == PropertyKind::real)
    return std::max(1, static_cast<int>(property.copyFloatValues.size()));
  return std::max(1, static_cast<int>(property.copyIntValues.size()));
}

/**
 * @brief Marks @p property valid or invalid for nest @p copyCounts without resizing.
 * @param property List-capable property whose authored length is preserved.
 * @param copyCounts Outer→inner copy-count vector for the owner node.
 */
inline void syncCopyListValidity(NodeProperty &property,
                                 const std::vector<int> &copyCounts) {
  if (!propertySupportsCopyValueList(property))
    return;
  if (property.kind == PropertyKind::real) {
    if (property.copyFloatValues.empty())
      property.copyFloatValues.push_back(property.floatValue);
  } else if (property.copyIntValues.empty()) {
    property.copyIntValues.push_back(
        property.preserveInBound ? 0 : property.value);
  }
  const auto length = authoredCopyListLength(property);
  if (isDividingSetLength(length, copyCounts)) {
    property.copyListInvalid = false;
    property.copyListInvalidMessage.clear();
    return;
  }
  property.copyListInvalid = true;
  property.copyListInvalidMessage =
      formatAllowedCopyListLengths(property.label, copyCounts) +
      " for this nest (authored length " + std::to_string(length) +
      " is no longer valid)";
}

/**
 * @brief Updates sticky descendant ids when canvas focus moves along a nest.
 * @param stickySpine Ordered descendants under the new focus (parent→deep).
 * @param previousChain Ancestor chain of the previous focus (outer→inner).
 * @param nextChain Ancestor chain of the new focus (outer→inner).
 */
inline void updateHierarchyStickySpine(
    std::vector<std::int32_t> &stickySpine,
    const std::vector<std::int32_t> &previousChain,
    const std::vector<std::int32_t> &nextChain) {
  const auto isPrefix =
      [](const std::vector<std::int32_t> &prefix,
         const std::vector<std::int32_t> &chain) {
        if (prefix.size() > chain.size())
          return false;
        return std::equal(prefix.begin(), prefix.end(), chain.begin());
      };
  const bool goingUp =
      !previousChain.empty() &&
      (nextChain.empty() || (nextChain.size() < previousChain.size() &&
                             isPrefix(nextChain, previousChain)));
  const bool goingDown =
      !nextChain.empty() &&
      (previousChain.empty() || (previousChain.size() < nextChain.size() &&
                                 isPrefix(previousChain, nextChain)));
  if (goingUp) {
    std::vector<std::int32_t> retained;
    if (nextChain.empty()) {
      retained = previousChain;
    } else {
      bool afterFocus = false;
      for (const auto id : previousChain) {
        if (!afterFocus) {
          if (id == nextChain.back())
            afterFocus = true;
          continue;
        }
        retained.push_back(id);
      }
    }
    for (const auto id : stickySpine) {
      if (std::find(retained.begin(), retained.end(), id) == retained.end())
        retained.push_back(id);
    }
    stickySpine = std::move(retained);
    return;
  }
  if (goingDown) {
    const auto found =
        std::find(stickySpine.begin(), stickySpine.end(), nextChain.back());
    if (found != stickySpine.end())
      stickySpine.erase(stickySpine.begin(), std::next(found));
    else
      stickySpine.clear();
    return;
  }
  stickySpine.clear();
}

/**
 * @brief Drops @p removedId and deeper sticky descendants from @p stickySpine.
 * @param stickySpine Ordered descendant group ids.
 * @param removedId Group that was deleted or ungrouped.
 */
inline void pruneStickySpineId(std::vector<std::int32_t> &stickySpine,
                               std::int32_t removedId) {
  const auto found =
      std::find(stickySpine.begin(), stickySpine.end(), removedId);
  if (found != stickySpine.end())
    stickySpine.erase(found, stickySpine.end());
}

/**
 * @brief Integer used by copy slot @p slot, falling back to the primary value.
 * @param property Source property.
 * @param slot Copy index (0 is the visible element).
 */
inline int integerValueForCopy(const NodeProperty &property, int slot) noexcept {
  if (property.copyIntValues.empty())
    return property.value;
  const auto length = static_cast<int>(property.copyIntValues.size());
  const auto index = ((slot % length) + length) % length;
  return property.copyIntValues[static_cast<std::size_t>(index)];
}

/**
 * @brief Real used by copy slot @p slot, tiling the authored list when L &lt; P.
 * @param property Source property.
 * @param slot Copy index (0 is the visible element).
 */
inline float floatValueForCopy(const NodeProperty &property, int slot) noexcept {
  if (property.copyFloatValues.empty())
    return property.floatValue;
  const auto length = static_cast<int>(property.copyFloatValues.size());
  const auto index = ((slot % length) + length) % length;
  return property.copyFloatValues[static_cast<std::size_t>(index)];
}

/**
 * @brief Ensures an authored copy list exists without pad/truncating to P.
 *
 * Empty lists are initialized from the primary scalar. Existing authored
 * lengths are left unchanged so nest changes can flag invalid L instead of
 * silently rewriting it.
 * @param property Property to initialize.
 * @param count Ignored (kept for call-site compatibility); tiling uses P at read.
 */
inline void ensurePropertyCopyCount(NodeProperty &property, int count) {
  (void)count;
  if (!propertySupportsCopyValueList(property))
    return;
  if (property.kind == PropertyKind::real) {
    if (property.copyFloatValues.empty())
      property.copyFloatValues.push_back(property.floatValue);
    return;
  }
  if (property.copyIntValues.empty())
    property.copyIntValues.push_back(
        property.preserveInBound ? 0 : property.value);
}

/**
 * @brief Initializes empty authored lists on @p node without forcing size P.
 * @param node Element whose enclosing copy product changed.
 * @param count Ignored; validity is synced separately from the copy-count vector.
 */
inline void ensureNodePropertyCopyCounts(GraphNode &node, int count) {
  for (auto &property : node.properties)
    ensurePropertyCopyCount(property, count);
}

/**
 * @brief Writes one copy slot's numeric property values onto the primary fields.
 * @param node Destination element (typically an unrolled clone).
 * @param slot Copy index to apply.
 */
inline void applyCopyPropertyValues(GraphNode &node, int slot) {
  for (auto &property : node.properties) {
    if (property.preserveInBound || property.copyListInvalid)
      continue;
    if (property.kind == PropertyKind::real)
      property.setFloatValue(floatValueForCopy(property, slot));
    else if (property.kind == PropertyKind::integer)
      property.setValue(integerValueForCopy(property, slot));
  }
}

/**
 * @brief Formats the authored (short) copy-list for the editable field.
 * @param property Source property.
 */
inline std::string formatAuthoredPropertyCopyList(const NodeProperty &property) {
  const auto count = authoredCopyListLength(property);
  std::string text;
  for (int index = 0; index < count; ++index) {
    if (index > 0)
      text += ", ";
    if (property.preserveInBound)
      text += preserveInToken;
    else if (property.kind == PropertyKind::real) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.2f",
                    floatValueForCopy(property, index));
      text += buffer;
    } else {
      text += std::to_string(integerValueForCopy(property, index));
    }
  }
  return text;
}

/**
 * @brief Formats the expanded P-length preview (tiling the authored list).
 *
 * Values are nested with @c [] along @p copyCounts so inner-group copies are
 * grouped inside outer-group copies.
 * @param property Source property.
 * @param copyCount Expanded slot count P.
 * @param copyCounts Outer→inner ancestor copy-count vector C.
 */
inline std::string
formatExpandedPropertyCopyList(const NodeProperty &property, int copyCount,
                               const std::vector<int> &copyCounts = {}) {
  const auto count = std::max(1, copyCount);
  std::vector<std::string> items;
  items.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (property.preserveInBound)
      items.emplace_back(preserveInToken);
    else if (property.kind == PropertyKind::real) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.2f",
                    floatValueForCopy(property, index));
      items.emplace_back(buffer);
    } else {
      items.push_back(std::to_string(integerValueForCopy(property, index)));
    }
  }
  return formatHierarchicalCopyList(items, copyCounts);
}

/**
 * @brief Formats N numeric copy values as a comma-separated list.
 * @param property Source property.
 * @param copyCount Number of copies to include.
 * @deprecated Use formatAuthoredPropertyCopyList / formatExpandedPropertyCopyList.
 */
inline std::string formatPropertyCopyList(const NodeProperty &property,
                                          int copyCount) {
  return formatExpandedPropertyCopyList(property, copyCount);
}

/**
 * @brief Parses a comma-separated authored copy list (dividing-set length).
 *
 * Commas separate values. Periods are the decimal mark. Length L must be in
 * D(C). The reserved token `in` is accepted when @p allowPreserveIn is true
 * and every token is `in`.
 * @param property Property providing kind and legal ranges.
 * @param ancestorCopyCounts Outer→inner copy-count vector C.
 * @param text User-entered list.
 * @param allowPreserveIn True when this field may bind to `in`.
 */
inline PropertyCopyListParse
parsePropertyCopyList(const NodeProperty &property,
                      const std::vector<int> &ancestorCopyCounts,
                      const std::string &text, bool allowPreserveIn = false) {
  PropertyCopyListParse result;
  std::vector<std::string> tokens;
  std::string_view remaining(text);
  while (!remaining.empty() || tokens.empty()) {
    const auto comma = remaining.find(',');
    const auto token =
        comma == std::string_view::npos ? remaining : remaining.substr(0, comma);
    const auto first = token.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
      result.message = property.label + " uses commas to separate copy values";
      return result;
    }
    const auto last = token.find_last_not_of(" \t");
    tokens.emplace_back(token.substr(first, last - first + 1));
    if (comma == std::string_view::npos)
      break;
    remaining = remaining.substr(comma + 1);
    if (remaining.empty()) {
      result.message = property.label + " uses commas to separate copy values";
      return result;
    }
  }

  int inCount = 0;
  int numericCount = 0;
  for (const auto &token : tokens) {
    if (token == preserveInToken)
      ++inCount;
    else
      ++numericCount;
  }
  if (inCount > 0 && numericCount > 0) {
    result.message = property.label + " cannot mix " + preserveInToken +
                     " with numbers";
    return result;
  }
  if (inCount > 0) {
    if (!allowPreserveIn || !propertySupportsPreserveIn(property)) {
      result.message = property.label + " does not support the " +
                       preserveInToken + " keyword";
      return result;
    }
    if (!isDividingSetLength(static_cast<int>(tokens.size()),
                             ancestorCopyCounts)) {
      result.message =
          formatAllowedCopyListLengths(property.label, ancestorCopyCounts);
      return result;
    }
    result.accepted = true;
    result.preserveIn = true;
    result.intValues.assign(tokens.size(), 0);
    return result;
  }

  if (!isDividingSetLength(static_cast<int>(tokens.size()),
                           ancestorCopyCounts)) {
    result.message =
        formatAllowedCopyListLengths(property.label, ancestorCopyCounts);
    return result;
  }

  const auto parseToken = [&](const std::string &token, float &real,
                              int &integer) {
    if (property.kind == PropertyKind::real) {
      if (token.empty())
        return false;
      std::size_t index = 0;
      const auto negative = token[index] == '-';
      if (negative || token[index] == '+')
        ++index;
      if (index >= token.size())
        return false;
      double value = 0.0;
      auto anyDigit = false;
      while (index < token.size() && token[index] >= '0' &&
             token[index] <= '9') {
        anyDigit = true;
        value = value * 10.0 + static_cast<double>(token[index] - '0');
        ++index;
      }
      if (index < token.size() && token[index] == '.') {
        ++index;
        double place = 0.1;
        while (index < token.size() && token[index] >= '0' &&
               token[index] <= '9') {
          anyDigit = true;
          value += static_cast<double>(token[index] - '0') * place;
          place *= 0.1;
          ++index;
        }
      }
      if (!anyDigit || index != token.size())
        return false;
      real = static_cast<float>(negative ? -value : value);
      return std::isfinite(real);
    }
    const auto *begin = token.data();
    const auto *end = begin + token.size();
    const auto parsed = std::from_chars(begin, end, integer);
    return parsed.ec == std::errc{} && parsed.ptr == end;
  };

  std::vector<float> reals;
  std::vector<int> integers;
  for (const auto &token : tokens) {
    float real = 0.0f;
    int integer = 0;
    if (!parseToken(token, real, integer)) {
      result.message = property.label + " values must be numbers";
      return result;
    }
    if (property.kind == PropertyKind::real) {
      if (real < property.floatMinimum || real > property.floatMaximum) {
        result.message = property.label + " must be between " +
                         std::to_string(property.floatMinimum) + " and " +
                         std::to_string(property.floatMaximum);
        return result;
      }
      reals.push_back(real);
    } else {
      if (integer < property.minimum || integer > property.maximum) {
        result.message = property.label + " must be between " +
                         std::to_string(property.minimum) + " and " +
                         std::to_string(property.maximum);
        return result;
      }
      integers.push_back(integer);
    }
  }
  result.accepted = true;
  result.floatValues = std::move(reals);
  result.intValues = std::move(integers);
  return result;
}

/**
 * @brief Parses a copy list when only P is known (D = {1, P}).
 * @param property Property providing kind and legal ranges.
 * @param copyCount Expanded slot count P.
 * @param text User-entered list.
 */
inline PropertyCopyListParse
parsePropertyCopyList(const NodeProperty &property, int copyCount,
                      const std::string &text) {
  const auto target = std::max(1, copyCount);
  return parsePropertyCopyList(property, std::vector<int>{target}, text, false);
}
} // namespace openyourbox::graph
