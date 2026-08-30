#pragma once

#include <JuceHeader.h>

#include "ExpressionParser.h"

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
  /**
   * @brief Elementwise arithmetic over Utility-style inputs `x1`…`xN`.
   *
   * Persisted as `math_expression`. Pins and the Inputs count follow Utility
   * rebuild rules; the authored `expression` string is the source of truth.
   */
  mathExpression,
  /** @brief Convolutional FIR reverb with optional external IR input. */
  reverb,
  /** @brief Exponential-decay noise impulse-response reverb. */
  expDecayReverb,
  /** @brief Filtered-noise synthesizer impulse-response reverb. */
  filteredNoiseReverb,
  /** @brief Linear time-varying FIR frequency-domain filter. */
  firFilter,
  /** @brief Modulated delay (chorus / flanger / vibrato). */
  modDelay,
  /** @brief Single-layer LSTM with in-cell activation and gain. */
  lstm,
  /** @brief Single-layer vanilla RNN with in-cell activation and gain. */
  rnn,
  /** Editor-only source hub declaring a group's external input lanes. */
  groupInput,
  /** Editor-only sink hub declaring a group's external output lanes. */
  groupOutput
};

/** @brief Train recipe selected in the unified Train panel. */
enum class TrainObjective { mapping, reconstruction };

/** @brief Accelerator requested for the Python train worker. */
enum class TrainDevice { automatic, cpu, mps, cuda };

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
  trainAutoload,
  /** @brief Factory palette TorchScript Load of an external checkpoint. */
  externalLoad
};

/** @brief User-visible load lifecycle for `BlackBoxOrigin::externalLoad`. */
enum class ExternalLoadStatus {
  /** @brief Empty path; never successfully loaded (dry passthrough). */
  empty,
  /** @brief Off-thread prepare in progress. */
  loading,
  /** @brief Factory published and ready to process. */
  ready,
  /** @brief Recoverable failure; silence unless a prior factory is retained. */
  error
};

/**
 * @brief Returns the persisted origin token for a Gold BlackBox.
 * @param origin Graph origin enum.
 */
inline const char *blackBoxOriginName(BlackBoxOrigin origin) noexcept {
  switch (origin) {
  case BlackBoxOrigin::trainAutoload:
    return "train_autoload";
  case BlackBoxOrigin::externalLoad:
    return "external_load";
  case BlackBoxOrigin::manualFreeze:
    break;
  }
  return "manual_freeze";
}

/**
 * @brief Parses a persisted origin token.
 * @param name ValueTree `blackBoxOrigin` string.
 */
inline BlackBoxOrigin blackBoxOriginFromName(const juce::String &name) noexcept {
  if (name == "train_autoload")
    return BlackBoxOrigin::trainAutoload;
  if (name == "external_load")
    return BlackBoxOrigin::externalLoad;
  return BlackBoxOrigin::manualFreeze;
}

/**
 * @brief Returns the persisted load-status token.
 * @param status External-load lifecycle.
 */
inline const char *externalLoadStatusName(ExternalLoadStatus status) noexcept {
  switch (status) {
  case ExternalLoadStatus::loading:
    return "loading";
  case ExternalLoadStatus::ready:
    return "ready";
  case ExternalLoadStatus::error:
    return "error";
  case ExternalLoadStatus::empty:
    break;
  }
  return "empty";
}

/**
 * @brief Parses a persisted load-status token.
 * @param name ValueTree `externalLoadStatus` string.
 */
inline ExternalLoadStatus
externalLoadStatusFromName(const juce::String &name) noexcept {
  if (name == "loading")
    return ExternalLoadStatus::loading;
  if (name == "ready")
    return ExternalLoadStatus::ready;
  if (name == "error")
    return ExternalLoadStatus::error;
  return ExternalLoadStatus::empty;
}

/** @brief Per-element analysis plot family requested by the editor. */
enum class AnalysisView {
  transfer = 0,
  frequency = 1,
  phase = 2,
  /** @brief Time-domain amplitude waveform (oscilloscope). */
  oscilloscope = 3
};

/** @brief Learned-layer chrome (Linear, Conv1D, ConvTranspose1d, TCN, Bottleneck). */
inline const juce::Colour liveBlueColour{100, 180, 255};
/** @brief Frozen Gold node colour. */
inline const juce::Colour frozenGoldColour{218, 165, 32};
/**
 * @brief Host and group-hub audio I/O chrome (Audio/Group Input and Output).
 */
inline const juce::Colour audioIoColour{70, 200, 150};
/** @brief Conditioning source node colour for Knob Input and XY Trackpad. */
inline const juce::Colour conditioningColour{180, 140, 255};
/**
 * @brief Pointwise / DSP-helper chrome (Activation, BatchNorm, Utility, Math,
 *        PQMF, Noise Synth).
 */
inline const juce::Colour helperLayerColour{185, 200, 55};

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
/** @brief Activation choice index for Tanh (LSTM/RNN default). */
inline constexpr int tanhActivationIndex = 2;
/** @brief Default IR length in samples for newly placed reverb elements. */
inline constexpr int defaultReverbLength = 4096;
/** @brief Inclusive minimum IR / reverb length in samples. */
inline constexpr int minimumReverbLength = 1;
/** @brief Non-blocking live-safe IR length warning threshold in milliseconds. */
inline constexpr float liveSafeIrLengthMilliseconds = 1000.0f;
/** @brief Default magnitude-grid time steps for FIR / filtered-noise. */
inline constexpr int defaultFilterFrames = 8;
/** @brief Default magnitude-grid filter-bank count. */
inline constexpr int defaultFilterBanks = 16;
/** @brief Default LTV-FIR / filtered-noise window size (odd). */
inline constexpr int defaultFirWindowSize = 257;
/** @brief Default LSTM/RNN hidden size. */
inline constexpr int defaultHiddenSize = 16;
/** @brief Default leaky-integrator mix (`1` is a standard RNN/LSTM step). */
inline constexpr float leakRateDefault = 1.0f;
/** @brief Inclusive lower bound for recurrent leak rate. */
inline constexpr float leakRateMinimum = 0.0f;
/** @brief Inclusive upper bound for recurrent leak rate. */
inline constexpr float leakRateMaximum = 1.0f;
/** @brief Default scale applied to hidden-to-hidden recurrent weights. */
inline constexpr float recurrentWeightScaleDefault = 1.0f;
/** @brief Inclusive lower bound for recurrent weight scale. */
inline constexpr float recurrentWeightScaleMinimum = 0.0f;
/** @brief Inclusive upper bound for recurrent weight scale. */
inline constexpr float recurrentWeightScaleMaximum = 10.0f;
/** @brief Magenta ExpDecayReverb raw gain initializer. */
inline constexpr float defaultExpDecayGain = 2.0f;
/** @brief Magenta ExpDecayReverb raw decay initializer. */
inline constexpr float defaultExpDecayDecay = 4.0f;
/** @brief Magenta ModDelay center delay in milliseconds. */
inline constexpr float defaultModDelayCenterMs = 15.0f;
/** @brief Magenta ModDelay modulation depth in milliseconds. */
inline constexpr float defaultModDelayDepthMs = 10.0f;
/** @brief Default ModDelay phase (mid delay after sigmoid mapping). */
inline constexpr float defaultModDelayPhase = 0.5f;
/** @brief Pin label for Reverb's optional external impulse-response input. */
inline constexpr const char *irPinLabel = "ir";
/** @brief Reserved token binding a dim/channels/features field to its input. */
inline constexpr const char *preserveInToken = "in";
/** @brief Default repeats parameter N for a new group. */
inline constexpr int defaultGroupRepeats = 1;
/** @brief Inclusive upper bound for a group's repeats parameter. */
inline constexpr int maximumGroupRepeats = 32;
/** @brief Supported nesting depth for groups and subgroups. */
inline constexpr int maximumGroupNestingDepth = 8;
/** @brief Bit flag applied to member pin ids drawn as group I/O pins. */
inline constexpr std::int32_t collapsedPinFlag = 0x20000000;
/** @brief Group box chrome, distinct from learned-layer blue. */
inline const juce::Colour groupFrameColour{95, 125, 155};
/** @brief Highlight colour when an element will be added to a group. */
inline const juce::Colour groupDropHighlightColour{140, 200, 230};
/** @brief Default canvas-local origin when inserting or reparenting like a new item. */
inline const juce::Point<float> defaultNewBoxPosition{250.0f, 140.0f};

/**
 * @brief How the Parameters tab is bound for the current session.
 *
 * Live is an editable graph box, LibraryInspect is a read-only catalog snapshot,
 * and Multi is a simplified state until a single box is selected.
 */
enum class SelectionContextKind {
  None,
  Live,
  LibraryInspect,
  Multi
};

/**
 * @brief Session selection/inspect state that drives the Parameters tab.
 */
struct SelectionContext {
  /** @brief Binding kind for the Parameters tab. */
  SelectionContextKind kind = SelectionContextKind::None;
  /** @brief Primary selected node or group when @c kind is Live. */
  std::optional<std::int32_t> liveBoxId;
  /** @brief Catalog entry UUID when @c kind is LibraryInspect. */
  std::string libraryEntryId;
  /** @brief Nested snapshot root id to inspect; 0 uses the saved entry root. */
  std::int32_t libraryNestedRootId = 0;
  /** @brief One-shot request to activate the Parameters tab this frame. */
  bool forceParametersTab = false;
};

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
         type == NodeType::convTranspose || type == NodeType::batchNorm ||
         type == NodeType::reverb || type == NodeType::filteredNoiseReverb ||
         type == NodeType::firFilter || type == NodeType::lstm ||
         type == NodeType::rnn;
}

/** @brief Returns true for Magenta DDSP-style effect elements. */
inline bool isDdspEffectType(NodeType type) noexcept {
  return type == NodeType::reverb || type == NodeType::expDecayReverb ||
         type == NodeType::filteredNoiseReverb || type == NodeType::firFilter ||
         type == NodeType::modDelay;
}

/** @brief Returns true for single-layer LSTM or RNN elements. */
inline bool isRecurrentType(NodeType type) noexcept {
  return type == NodeType::lstm || type == NodeType::rnn;
}

/** @brief Returns true for Activation/TCN/LSTM/RNN nonlinearity Gain. */
inline bool isNonlinearityGainType(NodeType type) noexcept {
  return type == NodeType::activation || type == NodeType::tcn ||
         isRecurrentType(type);
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
 * Linear, Conv1D, Activation, TCN, Utility, and Math Expression process
 * whatever tensor they receive. Their pins start unspecified and copy the
 * connected upstream rate and band count.
 */
inline bool isShapePassthroughType(NodeType type) noexcept {
  return type == NodeType::linear || type == NodeType::convolution ||
         type == NodeType::activation || type == NodeType::tcn ||
         type == NodeType::merge || type == NodeType::batchNorm ||
         type == NodeType::mathExpression || isDdspEffectType(type);
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
/** @brief Default train-worker device token (`auto` prefers CUDA, then MPS). */
inline constexpr const char *defaultTrainDevice = "auto";
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
/** @brief Default acids-rave v1 crop length (`n_signal` = 65536). */
inline constexpr int defaultReconstructionSegmentLength = 65536;
/** @brief Default acids-rave v1 minibatch size. */
inline constexpr int defaultReconstructionBatchSize = 8;
/** @brief Default generator Adam learning rate (acids-rave v1). */
inline constexpr float defaultReconstructionGeneratorLr = 1.0e-3f;
/** @brief Default discriminator Adam learning rate (acids-rave v1). */
inline constexpr float defaultReconstructionDiscriminatorLr = 1.0e-4f;
/** @brief Default Adam β1 for reconstruction (acids-rave v1). */
inline constexpr float defaultReconstructionAdamBeta1 = 0.5f;
/** @brief Default Adam β2 for reconstruction (acids-rave v1). */
inline constexpr float defaultReconstructionAdamBeta2 = 0.9f;
/** @brief LinearLR end factor over stage 1 (acids-rave v1). */
inline constexpr float defaultReconstructionLrDecayEnd = 0.1f;
/** @brief Constant KL β in acids-rave v1.gin (`BetaWarmupCallback` target). */
inline constexpr float defaultReconstructionKlBeta = 0.1f;
/** @brief KL β at the start of warmup (v1 keeps this equal to the target). */
inline constexpr float defaultReconstructionKlBetaStart = 0.1f;
/** @brief KL warmup length; 1 yields a constant β (acids-rave v1.gin). */
inline constexpr int defaultReconstructionKlWarmupSteps = 1;
/** @brief Feature-matching weight (acids-rave v1.gin `weights.feature_matching`). */
inline constexpr float defaultReconstructionFeatureMatchingWeight = 10.0f;
/** @brief Discriminator update period (acids-rave default `update_discriminator_every`). */
inline constexpr int defaultReconstructionDiscUpdateEvery = 2;
/** @brief Random allpass probability (acids-rave dataset `RandomApply` p=0.8). */
inline constexpr float defaultReconstructionPhaseMangleProb = 0.8f;
/** @brief Dequantization bit depth; 0 disables (acids-rave `Dequantize(16)`). */
inline constexpr int defaultReconstructionDequantizeBits = 16;
/** @brief Default bottleneck/Gold fidelity percent. */
inline constexpr float defaultFidelityPercent = 99.0f;
/** @brief Inclusive lower bound for fidelity. */
inline constexpr float fidelityMinimum = 0.0f;
/** @brief Inclusive upper bound for fidelity. */
inline constexpr float fidelityMaximum = 100.0f;
/**
 * @brief Default IR frequency bins per output channel (acids-rave v1).
 *
 * `NoiseGenerator` reshapes the amplitude net to `[..., data_size, noise_bands]`
 * before `amp_to_impulse_response`. The conv/LReLU stack is not inside the
 * Noise Synth element; this is only the IR bin count.
 */
inline constexpr int defaultNoiseBands = 5;
/**
 * @brief Default IR / hop length in samples (acids-rave v1 `prod([4, 4, 4])`).
 *
 * Users set this to the product of the strides they wire in front of Noise
 * Synth. The element upsamples time by this factor, like a hop fold.
 */
inline constexpr int defaultNoiseWindowSize = 64;

/**
 * @brief Onesided IRFFT length implied by @p noiseBands real bins.
 * @param noiseBands IR frequency bins (≥ 2).
 * @return `2 * (noiseBands - 1)`, matching `torch.fft.irfft` without `n`.
 */
inline int noiseSynthIrLength(int noiseBands) noexcept {
  return 2 * (std::max(2, noiseBands) - 1);
}

/**
 * @brief User-facing error when Noise Synth channels are not `k * noise_bands`.
 * @param channels Incoming amplitude width.
 * @param noiseBands Configured IR bins.
 * @return Empty when @p channels is unknown or divisible by @p noiseBands.
 */
inline std::string noiseSynthChannelMessage(int channels, int noiseBands) {
  if (channels <= 0 || noiseBands <= 0 || channels % noiseBands == 0)
    return {};
  return "Noise bands must divide input channels (" + std::to_string(channels) +
         " is not a multiple of " + std::to_string(noiseBands) + ")";
}

/**
 * @brief Returns true when Noise Synth cannot split @p channels into IR bins.
 * @param channels Incoming amplitude width.
 * @param noiseBands Configured IR bins.
 */
inline bool noiseSynthChannelIsError(int channels, int noiseBands) noexcept {
  return channels > 0 && noiseBands > 0 && channels % noiseBands != 0;
}

/**
 * @brief User-facing error when the IR window is shorter than the irfft length.
 * @param windowSize Configured hop / IR length.
 * @param noiseBands Configured IR bins.
 * @return Empty when @p windowSize is unknown or at least the IR length.
 */
inline std::string noiseSynthWindowMessage(int windowSize, int noiseBands) {
  const auto irLength = noiseSynthIrLength(noiseBands);
  if (windowSize <= 0 || windowSize >= irLength)
    return {};
  return "Window size must be at least the IR length " +
         std::to_string(irLength);
}

/**
 * @brief Returns true when @p windowSize cannot pad the irfft IR.
 * @param windowSize Configured hop / IR length.
 * @param noiseBands Configured IR bins.
 */
inline bool noiseSynthWindowIsError(int windowSize, int noiseBands) noexcept {
  return windowSize > 0 && windowSize < noiseSynthIrLength(noiseBands);
}

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
 * @brief Parses a train-worker device token.
 * @param name Device string (`auto`, `cpu`, `mps`, or `cuda`).
 * @return Automatic unless the token names a concrete backend.
 */
inline TrainDevice trainDeviceFromName(const std::string &name) noexcept {
  if (name == "cpu")
    return TrainDevice::cpu;
  if (name == "mps")
    return TrainDevice::mps;
  if (name == "cuda")
    return TrainDevice::cuda;
  return TrainDevice::automatic;
}

/**
 * @brief Returns the train-worker device token.
 * @param device Selected accelerator.
 * @return `auto`, `cpu`, `mps`, or `cuda`.
 */
inline const char *trainDeviceName(TrainDevice device) noexcept {
  switch (device) {
  case TrainDevice::cpu:
    return "cpu";
  case TrainDevice::mps:
    return "mps";
  case TrainDevice::cuda:
    return "cuda";
  case TrainDevice::automatic:
  default:
    return "auto";
  }
}

/**
 * @brief Returns a short Train-panel label for a device choice.
 * @param device Selected accelerator.
 * @return `Auto`, `CPU`, `MPS`, or `CUDA`.
 */
inline const char *trainDeviceLabel(TrainDevice device) noexcept {
  switch (device) {
  case TrainDevice::cpu:
    return "CPU";
  case TrainDevice::mps:
    return "MPS";
  case TrainDevice::cuda:
    return "CUDA";
  case TrainDevice::automatic:
  default:
    return "Auto";
  }
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

/**
 * @brief Returns true for the Mathematical Expression processing element.
 * @param type Graph node type.
 */
inline bool isMathExpressionType(NodeType type) noexcept {
  return type == NodeType::mathExpression;
}

/** @brief Runtime mode represented by a graph node. */
enum class NodeState { liveBlue, frozenGold };

/**
 * @brief Canvas chrome colour for a node type and freeze state.
 *
 * Frozen Gold and Black Box stay gold. Live boxes use family colours: audio
 * and group I/O, conditioning sources, learned layers, and helpers.
 * @param type Semantic node role.
 * @param state Live Blue or Frozen Gold.
 */
inline juce::Colour chromeColourForType(NodeType type,
                                        NodeState state) noexcept {
  if (state == NodeState::frozenGold)
    return frozenGoldColour;
  switch (type) {
  case NodeType::audioInput:
  case NodeType::audioOutput:
  case NodeType::groupInput:
  case NodeType::groupOutput:
    return audioIoColour;
  case NodeType::knobInput:
  case NodeType::xyTrackpad:
    return conditioningColour;
  case NodeType::activation:
  case NodeType::batchNorm:
  case NodeType::merge:
  case NodeType::mathExpression:
  case NodeType::pqmfAnalysis:
  case NodeType::pqmfSynthesis:
  case NodeType::noiseSynthesizer:
  case NodeType::expDecayReverb:
  case NodeType::modDelay:
    return helperLayerColour;
  case NodeType::linear:
  case NodeType::convolution:
  case NodeType::rateConv:
  case NodeType::convTranspose:
  case NodeType::tcn:
  case NodeType::variationalBottleneck:
  case NodeType::reverb:
  case NodeType::filteredNoiseReverb:
  case NodeType::firFilter:
  case NodeType::lstm:
  case NodeType::rnn:
    return liveBlueColour;
  case NodeType::blackBox:
    return frozenGoldColour;
  }
  return liveBlueColour;
}

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
 * @brief Formats repeat-slot labels as nested @c [] lists along the nest axes.
 *
 * @p repeatCounts is outer→inner. Axes of 1 are skipped. A single ungrouped
 * item is returned without brackets. When @p items length does not match the
 * product of @p repeatCounts, the items are shown as one flat bracketed list.
 * @param items One label per expanded repeat slot.
 * @param repeatCounts Outer→inner ancestor repeat-count vector C.
 */
inline std::string
formatHierarchicalRepeatList(const std::vector<std::string> &items,
                           const std::vector<int> &repeatCounts) {
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
  for (const auto repeats : repeatCounts)
    product *= std::max(1, repeats);
  product = std::max(1, product);
  const auto count = static_cast<int>(items.size());
  if (product != count)
    return count == 1 ? items.front() : joinRange(0, count);

  struct Formatter {
    const std::vector<std::string> &items;
    const std::vector<int> &repeatCounts;
    /**
     * @brief Nests @p span items starting at @p start from axis @p dim.
     * @param start First item index.
     * @param span Number of items in this axis slice.
     * @param dim Current outer→inner repeat-count index.
     */
    std::string format(int start, int span, int dim) const {
      if (span <= 0)
        return {};
      while (dim < static_cast<int>(repeatCounts.size()) &&
             repeatCounts[static_cast<std::size_t>(dim)] <= 1)
        ++dim;
      if (dim >= static_cast<int>(repeatCounts.size())) {
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
          std::max(1, repeatCounts[static_cast<std::size_t>(dim)]);
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
  return Formatter{items, repeatCounts}.format(0, count, 0);
}

/**
 * @brief Formats per-repeat pin shapes as a nested repeat-hierarchy label.
 * @param shapes Ordered shapes (one per repeat). Falls back to @p fallback when
 *        empty.
 * @param fallback Shape used when @p shapes is empty (typically the first repeat).
 * @param repeatCounts Outer→inner ancestor repeat counts used to nest @c [] groups.
 * @return Empty when no concrete shape fields are known yet.
 */
inline std::string formatShapeRepeatList(const std::vector<ShapeSignature> &shapes,
                                       const ShapeSignature &fallback,
                                       const std::vector<int> &repeatCounts = {}) {
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
  return formatHierarchicalRepeatList(labels, repeatCounts);
}

/**
 * @brief Collapses an inner serial-repeat axis to first-in or last-out.
 *
 * @p innerFold is the contiguous innermost chunk width. 1 keeps @p shapes.
 * When @p shapes length is not a multiple of @p innerFold, the whole list
 * reduces to a single first or last entry.
 * @param shapes Flat repeat-slot shapes, innermost axis contiguous.
 * @param innerFold Width of the inner axis to fold.
 * @param takeLast True to keep the last slot of each chunk (outputs).
 * @return One shape per remaining outer slot.
 */
inline std::vector<ShapeSignature>
foldInnerRepeatShapes(const std::vector<ShapeSignature> &shapes, int innerFold,
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
 * @brief Folds a collapsed group-box pin to first-in or last-out per outer slot.
 *
 * Repeats of the group itself — and of any nested groups whose repeat axes
 * leaked onto this hub — chain serially, so the parent canvas only attaches
 * to first-repeat inputs and last-repeat outputs. Those inner shapes are folded
 * away when the list length divides the remaining ancestor product. When it
 * does not divide (a nested hub that only leaked parent hops), the list is
 * left unfolded so sibling cables keep first-in of the shared ancestor slot.
 * Strict ancestor repeat axes remain as one entry per outer slot.
 *
 * @param shapes Per-repeat shapes; may include the group's own axis and nested
 *        descendant repeat axes inside it.
 * @param ancestorRepeatCounts Outer→inner runtime repeat counts for the hub,
 *        including the collapsed group's own N as the last entry.
 * @param takeLast True for outputs (last repeat of each inner chunk); false for
 *        inputs (first repeat of each inner chunk).
 * @return One shape per remaining outer slot; empty when @p shapes is empty.
 */
inline std::vector<ShapeSignature> collapsedGroupPinShapes(
    const std::vector<ShapeSignature> &shapes,
    const std::vector<int> &ancestorRepeatCounts, bool takeLast) {
  std::vector<int> outerCounts = ancestorRepeatCounts;
  if (!outerCounts.empty())
    outerCounts.pop_back();
  int outerProduct = 1;
  for (const auto repeats : outerCounts)
    outerProduct *= std::max(1, repeats);
  outerProduct = std::max(1, outerProduct);
  const int shapeCount = static_cast<int>(shapes.size());
  int innerFold = 1;
  if (shapeCount > 0 && shapeCount % outerProduct == 0)
    innerFold = std::max(1, shapeCount / outerProduct);
  // When the list does not divide the remaining ancestor product it is not an
  // inner serial axis (typical nested residual hubs that only leaked parent
  // hops). Leave it unfolded so attach stays first-in of the shared ancestor
  // slot instead of collapsing the whole list to last-out.
  return foldInnerRepeatShapes(shapes, innerFold, takeLast);
}

/**
 * @brief Parent-canvas attach shape for a collapsed group pin.
 *
 * For outputs this is last-out of the first remaining outer slot; for inputs
 * it is first-in of that slot.
 *
 * @param shapes Per-repeat shapes; may include the group's own axis.
 * @param fallback Shape used when @p shapes is empty (typically first-repeat).
 * @param ancestorRepeatCounts Outer→inner runtime repeat counts for the hub,
 *        including the collapsed group's own N as the last entry.
 * @param takeLast True for outputs; false for inputs.
 * @return Attach shape consumed by cables on the parent canvas.
 */
inline ShapeSignature collapsedGroupAttachShape(
    const std::vector<ShapeSignature> &shapes, const ShapeSignature &fallback,
    const std::vector<int> &ancestorRepeatCounts, bool takeLast) {
  const auto folded =
      collapsedGroupPinShapes(shapes, ancestorRepeatCounts, takeLast);
  if (!folded.empty())
    return folded.front();
  if (takeLast && !shapes.empty())
    return shapes.back();
  return fallback;
}

/**
 * @brief Formats collapsed group-box pin shapes for the parent canvas.
 *
 * @param shapes Per-repeat shapes; may include the group's own axis and nested
 *        descendant repeat axes inside it.
 * @param fallback Shape used when @p shapes is empty (typically first-repeat).
 * @param ancestorRepeatCounts Outer→inner runtime repeat counts for the hub,
 *        including the collapsed group's own N as the last entry.
 * @param takeLast True for outputs (last repeat of each inner chunk); false for
 *        inputs (first repeat of each inner chunk).
 * @return Hierarchical label, or empty when no concrete shape is known.
 */
inline std::string formatCollapsedGroupPinShapes(
    const std::vector<ShapeSignature> &shapes, const ShapeSignature &fallback,
    const std::vector<int> &ancestorRepeatCounts, bool takeLast) {
  std::vector<int> outerCounts = ancestorRepeatCounts;
  if (!outerCounts.empty())
    outerCounts.pop_back();
  const auto folded =
      collapsedGroupPinShapes(shapes, ancestorRepeatCounts, takeLast);
  const ShapeSignature &effectiveFallback =
      !folded.empty()
          ? folded.front()
          : (takeLast && !shapes.empty() ? shapes.back() : fallback);
  return formatShapeRepeatList(folded, effectiveFallback, outerCounts);
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
   * @brief Tensor shape presented by this pin on its canvas.
   *
   * Ordinary pins and Group Input hubs keep the visible first repeat. Group
   * Output hub outputs store the parent-canvas attach shape (last-out of this
   * group's serial axis for the first remaining outer slot). External cables
   * into a multi-repeat group still validate against first-repeat input;
   * cables that leave the stack consume last-repeat output.
   */
  ShapeSignature shape;
  /**
   * @brief Per-repeat shapes when the owner participates in N&gt;1 group repeats.
   *
   * Empty when the runtime repeat product is 1. Index 0 is the first repeat.
   * The innermost axis is the owner's group; ancestor axes nest outside it.
   * Collapsed group boxes fold that inner axis to first-in / last-out. Group
   * Output hub @ref shape may therefore differ from index 0.
   */
  std::vector<ShapeSignature> repeatShapes;
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
 * @brief Returns true for Reverb's optional external impulse-response pin.
 * @param pin Endpoint to inspect.
 */
inline bool isIrInputPin(const Pin &pin) noexcept {
  return pin.kind == PinKind::input && pin.label == irPinLabel;
}

/**
 * @brief Returns true for Gold RAVE encode/decode latent endpoints.
 * @param pin Endpoint to inspect.
 */
inline bool isLatentPin(const Pin &pin) noexcept {
  return pin.label == latentPinLabel;
}

/** @brief Value type accepted by an inline graph property. */
enum class PropertyKind { integer, choice, readOnly, real, string };

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
   * @brief Per-repeat integer values when the owner sits in a group with N&gt;1.
   *
   * Empty means every repeat uses `value`. Slot 0 mirrors `value`.
   */
  std::vector<int> repeatIntValues;
  /**
   * @brief Per-repeat real values when the owner sits in a group with N&gt;1.
   *
   * Empty means every repeat uses `floatValue`. Slot 0 mirrors `floatValue`.
   * Length is the **authored** list length L (may be less than P).
   */
  std::vector<float> repeatFloatValues;
  /**
   * @brief True when authored length L is not in the nest dividing set.
   *
   * Authored values are preserved; runtime and shape inference treat the
   * property as unresolved until the user commits a legal list.
   */
  bool repeatListInvalid = false;
  /** @brief User-facing reason when @ref repeatListInvalid is set. */
  std::string repeatListInvalidMessage;
  /**
   * @brief True when this integer property is bound to the `in` keyword.
   *
   * Authored length is `repeatIntValues.size()` (or 1 when empty). Resolved
   * integers are derived from the paired input shape, not persisted as the
   * source of truth.
   */
  bool preserveInBound = false;
  /**
   * @brief Authored string used by `PropertyKind::string` (e.g. Math Expression).
   *
   * This is the source of truth for formula text. Invalid strings are never
   * stored; callers refuse the commit and keep the previous value.
   */
  std::string stringValue;
  /**
   * @brief Authored list tokens for copy-expanded numeric fields.
   *
   * Each entry is a literal or an `i`-expression. Empty means the numeric
   * `repeatIntValues` / `repeatFloatValues` vectors are the authored form
   * (legacy documents). Length matches the authored list L.
   */
  std::vector<std::string> authoredTokens;

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
 * @brief Returns true for 0/1 flag properties drawn as checkboxes.
 * @param key Persisted property key.
 */
inline bool isBooleanPropertyKey(const std::string &key) noexcept {
  return key == "residual" || key == "weight_norm" || key == "add_dry" ||
         key == "bidirectional" || key == "bias";
}

/**
 * @brief Returns true when a property keeps fixed choice or boolean bounds.
 * @param property Node property to inspect.
 */
inline bool propertyKeepsFixedBounds(const NodeProperty &property) noexcept {
  return property.kind == PropertyKind::choice ||
         isBooleanPropertyKey(property.key);
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
 * @brief Outcome of parsing a comma-separated per-repeat property list.
 */
struct PropertyRepeatListParse {
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
  /**
   * @brief Authored token strings (literals or `i`-expressions), length L.
   *
   * Empty when the parse used only legacy numeric tokens with no expression
   * text to persist. Callers copy this onto `NodeProperty::authoredTokens`.
   */
  std::vector<std::string> authoredTokens;
};

/** @brief Live performance values displayed by a frozen BlackBox node. */
struct NodeMetrics {
  /** @brief Backend compilation duration. */
  double compileTimeMilliseconds = 0.0;
  /** @brief Most recent per-buffer inference duration. */
  double inferenceTimeMilliseconds = 0.0;
};

/**
 * @brief Independent weight/artifact identity for one invisible group repeat.
 *
 * Visible graph elements stay unique. DSP unrolls repeats using these slots so
 * live/freeze seeds stay independent across repeats. Training does not load
 * these seeds; it initializes with PyTorch defaults.
 */
struct RepeatWeightSlot {
  /** @brief Randomization seed used when provenance is random. */
  std::int32_t seed = 42;
  /** @brief Whether this repeat loads a file or a seed. */
  WeightsProvenance provenance = WeightsProvenance::random;
  /** @brief Filesystem path of trained or browsed weights. */
  std::string weightsPath;
  /** @brief Optional frozen artifact used by this repeat. */
  std::string artifactPath;
};

/** @brief Named hierarchical container on the canvas. */
struct GraphGroup {
  /** @brief Stable identifier unique among nodes and groups. */
  std::int32_t id = 0;
  /** @brief User-visible group title (canvas boxes append ` (Block)`). */
  std::string name = "Group";
  /** @brief Parent group, or empty at the canvas root. */
  std::optional<std::int32_t> parentGroupId;
  /** @brief Child node ids and nested group ids. */
  std::vector<std::int32_t> memberIds;
  /** @brief Presentation-only collapse flag (library insert still forces true). */
  bool collapsed = false;
  /** @brief Independent serial repeat count (DSP-only; UI stays unique). */
  int repeats = defaultGroupRepeats;
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
 * @brief One shape-driving property implicated by an inactive repeat request.
 */
struct GroupRepeatPropertyHint {
  /** @brief Element containing the candidate property. */
  std::int32_t nodeId = 0;
  /** @brief Canonical property key such as `channels` or `features`. */
  std::string propertyKey;
  /** @brief User-facing explanation for highlighting this property. */
  std::string message;
};

/**
 * @brief Derived validity and diagnostics for a group's requested repeats.
 */
struct GroupRepeatStatus {
  /** @brief User-authored repeat count stored on the group. */
  int requestedRepeats = defaultGroupRepeats;
  /** @brief Count safe to materialize; one while the request is inactive. */
  int effectiveRepeats = defaultGroupRepeats;
  /** @brief True when requested N can chain after per-repeat properties. */
  bool active = true;
  /** @brief Persistent user-facing explanation when inactive. */
  std::string message;
  /** @brief Candidate parameters that can make the chain shape-compatible. */
  std::vector<GroupRepeatPropertyHint> propertyHints;
};

/**
 * @brief Returns the canvas title drawn on a group box.
 * @param group Group whose stored name is shown.
 * @param repeatStatus Derived repeat validity; when omitted, @p group.repeats is
 *        treated as active.
 * @return The stored name with a ` (Block)` suffix, plus `×N` when requested
 *         repeats exceed one and `×N→1` when the request is inactive.
 */
inline std::string groupBoxDisplayLabel(const GraphGroup &group,
                                        const GroupRepeatStatus &repeatStatus) {
  std::string label = group.name + " (Block)";
  const int requested = std::max(1, repeatStatus.requestedRepeats);
  if (requested <= 1)
    return label;
  label += " \xC3\x97";
  label += std::to_string(requested);
  if (!repeatStatus.active)
    label += "\xE2\x86\x92" + std::to_string(std::max(1, repeatStatus.effectiveRepeats));
  return label;
}

/**
 * @brief Returns the canvas title for a group with no repeat diagnostics.
 * @param group Group whose stored name is shown.
 * @return Label from @ref groupBoxDisplayLabel assuming requested repeats are active.
 */
inline std::string groupBoxDisplayLabel(const GraphGroup &group) {
  GroupRepeatStatus status;
  status.requestedRepeats = group.repeats;
  status.effectiveRepeats = group.repeats;
  status.active = true;
  return groupBoxDisplayLabel(group, status);
}

/** @brief Outcome of a group create/membership/repeats mutation. */
struct GroupActionResult {
  /** @brief True when the graph was mutated. */
  bool accepted = false;
  /** @brief User-facing reason when the action was refused. */
  std::string message;
  /** @brief Created or targeted group id when accepted. */
  std::int32_t groupId = 0;
};

/** @brief Boundary-crossing port used by group I/O pins and repeat chaining. */
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
  /**
   * @brief Load lifecycle for `externalLoad` nodes.
   *
   * Empty → dry passthrough. Error with no retained factory → silence.
   */
  ExternalLoadStatus externalLoadStatus = ExternalLoadStatus::empty;
  /** @brief Recoverable user-facing reason when `externalLoadStatus` is error. */
  std::string externalLoadErrorMessage;
  /**
   * @brief Path of the factory currently retained for live audio.
   *
   * Session-only: kept during a failed or in-progress reload so the prior
   * model keeps running until a successful swap or explicit clear.
   */
  std::string runtimeArtifactPath;
  /** @brief Last successful probe input width, or 0 when unknown. */
  int inferredInputChannels = 0;
  /** @brief Last successful probe output width, or 0 when unknown. */
  int inferredOutputChannels = 0;
  /** @brief Last successful encode latent width, or 0 when unused. */
  int inferredLatentChannels = 0;
  /** @brief Shape-check override for input channels; −1 means use inferred. */
  int overrideInputChannels = -1;
  /** @brief Shape-check override for output channels; −1 means use inferred. */
  int overrideOutputChannels = -1;
  /** @brief Shape-check override for latent channels; −1 means use inferred. */
  int overrideLatentChannels = -1;
  /** @brief True when the loaded checkpoint exposes encode and decode. */
  bool externalHasEncodeDecode = false;
  /** @brief True when the loaded checkpoint advertises conditioning. */
  bool externalAcceptsConditioning = false;
  /**
   * @brief True when inference failed and required overrides are missing.
   *
   * Connections are not treated as legal until valid overrides exist.
   */
  bool externalShapeIncomplete = false;
  /** @brief Best-effort sample-rate mismatch notice; empty when unused. */
  std::string sampleRateWarning;
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
   * @brief Per-repeat weights for enclosing groups with N&gt;1.
   *
   * Slot 0 mirrors `seed` / `weightsPath` / `artifactPath`. Extra slots hold
   * independent repeats that the UI never draws.
   */
  std::vector<RepeatWeightSlot> repeatSlots;
};

/**
 * @brief Returns true when @p node is a Factory TorchScript Load Gold box.
 * @param node Graph node to inspect.
 */
inline bool isExternalLoadNode(const GraphNode &node) noexcept {
  return node.type == NodeType::blackBox &&
         node.blackBoxOrigin == BlackBoxOrigin::externalLoad;
}

/**
 * @brief Returns the shape-check input width (`override ?? inferred`).
 * @param node External-load node.
 */
inline int effectiveInputChannels(const GraphNode &node) noexcept {
  return node.overrideInputChannels > 0 ? node.overrideInputChannels
                                        : node.inferredInputChannels;
}

/**
 * @brief Returns the shape-check output width (`override ?? inferred`).
 * @param node External-load node.
 */
inline int effectiveOutputChannels(const GraphNode &node) noexcept {
  return node.overrideOutputChannels > 0 ? node.overrideOutputChannels
                                         : node.inferredOutputChannels;
}

/**
 * @brief Returns the shape-check latent width (`override ?? inferred`).
 * @param node External-load node.
 */
inline int effectiveLatentChannels(const GraphNode &node) noexcept {
  return node.overrideLatentChannels > 0 ? node.overrideLatentChannels
                                         : node.inferredLatentChannels;
}

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
  /** @brief Effective PyTorch device type used by the worker (`cpu`, `mps`, `cuda`). */
  std::string device;
  /** @brief Device token requested in `train_options` (`auto`, `cpu`, `mps`, `cuda`). */
  std::string requestedDevice;
  /** @brief True when the worker fell back from a concrete requested accelerator. */
  bool deviceFallback = false;
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
 * @brief Derives a per-repeat randomization seed from a visible base seed.
 * @param baseSeed Seed shown on the editable template element (slot 0).
 * @param repeatIndex Zero-based materialized repeat index.
 * @return `baseSeed + repeatIndex` wrapped into `[minimumSeed, maximumSeed]`.
 *
 * Live audition and freeze use these seeds. Training builds fresh PyTorch
 * modules with framework default initialization and does not consume them.
 */
inline std::int32_t seedForRepeatSlot(std::int32_t baseSeed,
                                   std::size_t repeatIndex) noexcept {
  constexpr auto span =
      static_cast<std::int64_t>(maximumSeed) - minimumSeed + 1;
  const auto mixed =
      (static_cast<std::int64_t>(clampSeed(baseSeed)) -
       minimumSeed + static_cast<std::int64_t>(repeatIndex)) %
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
/** @brief Extra top padding reserved for the group header and repeats control. */
inline constexpr float groupHeaderHeight = 52.0f;
/** @brief Inset from a group box origin used when mapping member coordinates. */
inline constexpr float groupContentPad = 8.0f;
/** @brief Space between Group Input/Output hubs and grouped content. */
inline constexpr float groupBoundaryContentGap = 48.0f;
/** @brief Group-canvas origin used when placing a new group's Input hub. */
inline const juce::Point<float> groupInteriorOrigin{24.0f, 140.0f};

/**
 * @brief Offset from a group box origin used when mapping member coordinates.
 */
inline juce::Point<float> groupContentOffset() noexcept {
  return {groupContentPad, groupHeaderHeight};
}

/**
 * @brief Captures a node's primary weight fields into a repeat slot.
 * @param node Source element.
 */
inline RepeatWeightSlot repeatSlotFromNode(const GraphNode &node) {
  RepeatWeightSlot slot;
  slot.seed = node.seed;
  slot.provenance = node.weightsProvenance;
  slot.weightsPath = node.weightsPath;
  slot.artifactPath = node.artifactPath;
  return slot;
}

/**
 * @brief Writes one repeat slot onto a node's primary weight fields.
 * @param node Destination element.
 * @param slot Independent repeat identity to apply.
 */
inline void applyRepeatSlot(GraphNode &node, const RepeatWeightSlot &slot) {
  node.seed = slot.seed;
  node.weightsProvenance = slot.provenance;
  node.weightsPath = slot.weightsPath;
  if (!slot.artifactPath.empty())
    node.artifactPath = slot.artifactPath;
}

/**
 * @brief Ensures @p node has @p count repeat slots, deriving or cloning on growth.
 * @param node Element whose enclosing groups changed repeat count.
 * @param count Required slot count (≥ 1).
 *
 * Random-provenance slots use `seedForRepeatSlot` so raising N does not leave
 * identical live weights. File-provenance slots still clone the previous last
 * path (FR-017c).
 */
inline void ensureRepeatSlotCount(GraphNode &node, int count) {
  const auto target = std::max(1, count);
  if (node.repeatSlots.empty())
    node.repeatSlots.push_back(repeatSlotFromNode(node));
  node.repeatSlots.front() = repeatSlotFromNode(node);
  while (static_cast<int>(node.repeatSlots.size()) < target) {
    RepeatWeightSlot slot = node.repeatSlots.back();
    const auto index = node.repeatSlots.size();
    if (slot.provenance == WeightsProvenance::random)
      slot.seed = seedForRepeatSlot(node.seed, index);
    node.repeatSlots.push_back(std::move(slot));
  }
  if (static_cast<int>(node.repeatSlots.size()) > target)
    node.repeatSlots.resize(static_cast<std::size_t>(target));
  applyRepeatSlot(node, node.repeatSlots.front());
}

/**
 * @brief Returns true when a property can store one numeric value per repeat.
 * @param property Candidate inline property.
 */
inline bool propertySupportsRepeatValueList(const NodeProperty &property) noexcept {
  if (property.kind != PropertyKind::integer &&
      property.kind != PropertyKind::real)
    return false;
  return !isBooleanPropertyKey(property.key) && property.key != "inputs" &&
         property.key != "ports" && property.key != "expression";
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
 * @brief Product of ancestor repeat counts (P), or 1 when @p repeatCounts is empty.
 * @param repeatCounts Outer→inner repeat-count vector.
 */
inline int repeatCountProduct(const std::vector<int> &repeatCounts) noexcept {
  int product = 1;
  for (const auto repeats : repeatCounts)
    product *= std::max(1, repeats);
  return std::max(1, product);
}

/**
 * @brief Dividing-set lengths D(C) = {1} ∪ suffix products of @p repeatCounts.
 * @param repeatCounts Outer→inner repeat-count vector C.
 * @return Sorted unique legal authored lengths including 1 and P.
 */
inline std::vector<int>
dividingSetLengths(const std::vector<int> &repeatCounts) {
  std::vector<int> lengths;
  lengths.push_back(1);
  int product = 1;
  for (int index = static_cast<int>(repeatCounts.size()) - 1; index >= 0;
       --index) {
    product *= std::max(1, repeatCounts[static_cast<std::size_t>(index)]);
    lengths.push_back(product);
  }
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  return lengths;
}

/**
 * @brief Returns true when @p length is in D(@p repeatCounts).
 * @param length Authored list length L.
 * @param repeatCounts Outer→inner repeat-count vector.
 */
inline bool isDividingSetLength(int length,
                                const std::vector<int> &repeatCounts) noexcept {
  if (length < 1)
    return false;
  for (const auto allowed : dividingSetLengths(repeatCounts)) {
    if (allowed == length)
      return true;
  }
  return false;
}

/**
 * @brief User-facing list of allowed authored lengths for @p label.
 * @param label Property label.
 * @param repeatCounts Outer→inner repeat-count vector.
 */
inline std::string
formatAllowedRepeatListLengths(const std::string &label,
                             const std::vector<int> &repeatCounts) {
  const auto allowed = dividingSetLengths(repeatCounts);
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
inline int authoredRepeatListLength(const NodeProperty &property) noexcept {
  if (!property.authoredTokens.empty())
    return std::max(1, static_cast<int>(property.authoredTokens.size()));
  if (property.kind == PropertyKind::real)
    return std::max(1, static_cast<int>(property.repeatFloatValues.size()));
  return std::max(1, static_cast<int>(property.repeatIntValues.size()));
}

/**
 * @brief Marks @p property valid or invalid for nest @p repeatCounts without resizing.
 * @param property List-capable property whose authored length is preserved.
 * @param repeatCounts Outer→inner repeat-count vector for the owner node.
 */
inline void syncRepeatListValidity(NodeProperty &property,
                                 const std::vector<int> &repeatCounts) {
  if (!propertySupportsRepeatValueList(property))
    return;
  if (property.kind == PropertyKind::real) {
    if (property.repeatFloatValues.empty())
      property.repeatFloatValues.push_back(property.floatValue);
  } else if (property.repeatIntValues.empty()) {
    property.repeatIntValues.push_back(
        property.preserveInBound ? 0 : property.value);
  }
  const auto length = authoredRepeatListLength(property);
  if (isDividingSetLength(length, repeatCounts)) {
    property.repeatListInvalid = false;
    property.repeatListInvalidMessage.clear();
    return;
  }
  property.repeatListInvalid = true;
  property.repeatListInvalidMessage =
      formatAllowedRepeatListLengths(property.label, repeatCounts) +
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
 * @brief Integer used by repeat slot @p slot, falling back to the primary value.
 * @param property Source property.
 * @param slot Repeat index (0 is the visible element).
 */
inline int integerValueForRepeat(const NodeProperty &property, int slot) {
  if (!property.authoredTokens.empty() && !property.preserveInBound) {
    const auto length = static_cast<int>(property.authoredTokens.size());
    const auto tokenIndex = ((slot % length) + length) % length;
    const auto evaluated = evaluateParameterToken(
        property.authoredTokens[static_cast<std::size_t>(tokenIndex)],
        static_cast<double>(slot));
    if (evaluated.ok)
      return static_cast<int>(std::nearbyint(evaluated.value));
  }
  if (property.repeatIntValues.empty())
    return property.value;
  const auto length = static_cast<int>(property.repeatIntValues.size());
  const auto index = ((slot % length) + length) % length;
  return property.repeatIntValues[static_cast<std::size_t>(index)];
}

/**
 * @brief Real used by repeat slot @p slot, tiling the authored list when L &lt; P.
 * @param property Source property.
 * @param slot Repeat index (0 is the visible element).
 */
inline float floatValueForRepeat(const NodeProperty &property, int slot) {
  if (!property.authoredTokens.empty()) {
    const auto length = static_cast<int>(property.authoredTokens.size());
    const auto tokenIndex = ((slot % length) + length) % length;
    const auto evaluated = evaluateParameterToken(
        property.authoredTokens[static_cast<std::size_t>(tokenIndex)],
        static_cast<double>(slot));
    if (evaluated.ok)
      return static_cast<float>(evaluated.value);
  }
  if (property.repeatFloatValues.empty())
    return property.floatValue;
  const auto length = static_cast<int>(property.repeatFloatValues.size());
  const auto index = ((slot % length) + length) % length;
  return property.repeatFloatValues[static_cast<std::size_t>(index)];
}

/**
 * @brief Ensures an authored repeat list exists without pad/truncating to P.
 *
 * Empty lists are initialized from the primary scalar. Existing authored
 * lengths are left unchanged so nest changes can flag invalid L instead of
 * silently rewriting it.
 * @param property Property to initialize.
 * @param count Ignored (kept for call-site compatibility); tiling uses P at read.
 */
inline void ensurePropertyRepeatCount(NodeProperty &property, int count) {
  (void)count;
  if (!propertySupportsRepeatValueList(property))
    return;
  if (property.kind == PropertyKind::real) {
    if (property.repeatFloatValues.empty())
      property.repeatFloatValues.push_back(property.floatValue);
    return;
  }
  if (property.repeatIntValues.empty())
    property.repeatIntValues.push_back(
        property.preserveInBound ? 0 : property.value);
}

/**
 * @brief Initializes empty authored lists on @p node without forcing size P.
 * @param node Element whose enclosing repeat product changed.
 * @param count Ignored; validity is synced separately from the repeat-count vector.
 */
inline void ensureNodePropertyRepeatCounts(GraphNode &node, int count) {
  for (auto &property : node.properties)
    ensurePropertyRepeatCount(property, count);
}

/**
 * @brief Writes one repeat slot's numeric property values onto the primary fields.
 * @param node Destination element (typically an unrolled clone).
 * @param slot Repeat index to apply.
 */
inline void applyRepeatPropertyValues(GraphNode &node, int slot) {
  for (auto &property : node.properties) {
    if (property.preserveInBound || property.repeatListInvalid)
      continue;
    if (property.kind == PropertyKind::real)
      property.setFloatValue(floatValueForRepeat(property, slot));
    else if (property.kind == PropertyKind::integer)
      property.setValue(integerValueForRepeat(property, slot));
  }
}

/**
 * @brief Formats the authored (short) repeat-list for the editable field.
 * @param property Source property.
 */
inline std::string formatAuthoredPropertyRepeatList(const NodeProperty &property) {
  if (!property.authoredTokens.empty() && !property.preserveInBound) {
    std::string text;
    for (std::size_t index = 0; index < property.authoredTokens.size(); ++index) {
      if (index > 0)
        text += ", ";
      text += property.authoredTokens[index];
    }
    return text;
  }
  const auto count = authoredRepeatListLength(property);
  std::string text;
  for (int index = 0; index < count; ++index) {
    if (index > 0)
      text += ", ";
    if (property.preserveInBound)
      text += preserveInToken;
    else if (property.kind == PropertyKind::real) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.2f",
                    floatValueForRepeat(property, index));
      text += buffer;
    } else {
      text += std::to_string(integerValueForRepeat(property, index));
    }
  }
  return text;
}

/**
 * @brief Formats the expanded P-length preview (tiling the authored list).
 *
 * Values are nested with @c [] along @p repeatCounts so inner-group repeats are
 * grouped inside outer-group repeats.
 * @param property Source property.
 * @param repeatCount Expanded slot count P.
 * @param repeatCounts Outer→inner ancestor repeat-count vector C.
 */
inline std::string
formatExpandedPropertyRepeatList(const NodeProperty &property, int repeatCount,
                               const std::vector<int> &repeatCounts = {}) {
  const auto count = std::max(1, repeatCount);
  std::vector<std::string> items;
  items.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (property.preserveInBound)
      items.emplace_back(preserveInToken);
    else if (property.kind == PropertyKind::real) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.2f",
                    floatValueForRepeat(property, index));
      items.emplace_back(buffer);
    } else {
      items.push_back(std::to_string(integerValueForRepeat(property, index)));
    }
  }
  return formatHierarchicalRepeatList(items, repeatCounts);
}

/**
 * @brief Formats N numeric repeat values as a comma-separated list.
 * @param property Source property.
 * @param repeatCount Number of repeats to include.
 * @deprecated Use formatAuthoredPropertyRepeatList / formatExpandedPropertyRepeatList.
 */
inline std::string formatPropertyRepeatList(const NodeProperty &property,
                                          int repeatCount) {
  return formatExpandedPropertyRepeatList(property, repeatCount);
}

/**
 * @brief Parses a comma-separated authored repeat list (dividing-set length).
 *
 * Commas separate values. Periods are the decimal mark. Length L must be in
 * D(C). The reserved token `in` is accepted when @p allowPreserveIn is true
 * and every token is `in`.
 * @param property Property providing kind and legal ranges.
 * @param ancestorRepeatCounts Outer→inner repeat-count vector C.
 * @param text User-entered list.
 * @param allowPreserveIn True when this field may bind to `in`.
 */
inline PropertyRepeatListParse
parsePropertyRepeatList(const NodeProperty &property,
                      const std::vector<int> &ancestorRepeatCounts,
                      const std::string &text, bool allowPreserveIn = false) {
  PropertyRepeatListParse result;
  std::vector<std::string> tokens;
  std::string_view remaining(text);
  while (!remaining.empty() || tokens.empty()) {
    const auto comma = remaining.find(',');
    const auto token =
        comma == std::string_view::npos ? remaining : remaining.substr(0, comma);
    const auto first = token.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
      result.message = property.label + " uses commas to separate repeat values";
      return result;
    }
    const auto last = token.find_last_not_of(" \t");
    tokens.emplace_back(token.substr(first, last - first + 1));
    if (comma == std::string_view::npos)
      break;
    remaining = remaining.substr(comma + 1);
    if (remaining.empty()) {
      result.message = property.label + " uses commas to separate repeat values";
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
                             ancestorRepeatCounts)) {
      result.message =
          formatAllowedRepeatListLengths(property.label, ancestorRepeatCounts);
      return result;
    }
    result.accepted = true;
    result.preserveIn = true;
    result.intValues.assign(tokens.size(), 0);
    return result;
  }

  if (!isDividingSetLength(static_cast<int>(tokens.size()),
                           ancestorRepeatCounts)) {
    result.message =
        formatAllowedRepeatListLengths(property.label, ancestorRepeatCounts);
    return result;
  }

  const auto parseToken = [&](const std::string &token, float &real,
                              int &integer, int slotIndex,
                              std::string &tokenMessage) {
    const auto evaluated =
        evaluateParameterToken(token, static_cast<double>(slotIndex));
    if (!evaluated.ok) {
      tokenMessage = evaluated.message;
      return false;
    }
    if (property.kind == PropertyKind::real) {
      real = static_cast<float>(evaluated.value);
      return std::isfinite(real);
    }
    if (!expressionValueIsInteger(evaluated.value)) {
      tokenMessage = property.label +
                     " requires an integer result for every copy "
                     "(expressions are not rounded)";
      return false;
    }
    const auto rounded =
        static_cast<long long>(std::nearbyint(evaluated.value));
    if (rounded < static_cast<long long>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<long long>(std::numeric_limits<int>::max())) {
      tokenMessage = property.label + " result is out of integer range";
      return false;
    }
    integer = static_cast<int>(rounded);
    return true;
  };

  const auto expandedCount =
      std::max(1, repeatCountProduct(ancestorRepeatCounts));
  std::vector<float> reals;
  std::vector<int> integers;
  reals.reserve(tokens.size());
  integers.reserve(tokens.size());
  for (int slot = 0; slot < expandedCount; ++slot) {
    const auto &token =
        tokens[static_cast<std::size_t>(slot) % tokens.size()];
    float real = 0.0f;
    int integer = 0;
    std::string tokenMessage;
    if (!parseToken(token, real, integer, slot, tokenMessage)) {
      result.message =
          tokenMessage.empty()
              ? property.label +
                    " values must be numbers or expressions of i"
              : tokenMessage;
      return result;
    }
    if (property.kind == PropertyKind::real) {
      if (real < property.floatMinimum || real > property.floatMaximum) {
        result.message = property.label + " must be between " +
                         std::to_string(property.floatMinimum) + " and " +
                         std::to_string(property.floatMaximum);
        return result;
      }
    } else if (integer < property.minimum || integer > property.maximum) {
      result.message = property.label + " must be between " +
                       std::to_string(property.minimum) + " and " +
                       std::to_string(property.maximum);
      return result;
    }
  }
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    float real = 0.0f;
    int integer = 0;
    std::string tokenMessage;
    parseToken(tokens[index], real, integer, static_cast<int>(index),
               tokenMessage);
    if (property.kind == PropertyKind::real)
      reals.push_back(real);
    else
      integers.push_back(integer);
  }
  result.accepted = true;
  result.floatValues = std::move(reals);
  result.intValues = std::move(integers);
  result.authoredTokens = tokens;
  return result;
}

/**
 * @brief Parses a repeat list when only P is known (D = {1, P}).
 * @param property Property providing kind and legal ranges.
 * @param repeatCount Expanded slot count P.
 * @param text User-entered list.
 */
inline PropertyRepeatListParse
parsePropertyRepeatList(const NodeProperty &property, int repeatCount,
                      const std::string &text) {
  const auto target = std::max(1, repeatCount);
  return parsePropertyRepeatList(property, std::vector<int>{target}, text, false);
}
} // namespace openyourbox::graph
