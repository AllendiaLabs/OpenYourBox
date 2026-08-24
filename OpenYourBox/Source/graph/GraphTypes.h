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
  xyTrackpad
};

/** @brief Merge element operating mode. */
enum class MergeMode : int { add = 0, multiply = 1, concatenate = 2 };

/** @brief Distinguishes tensor audio paths from scalar conditioning paths. */
enum class SignalKind { audio, conditioning };

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
/** @brief Fixed stereo host-input node colour. */
inline const juce::Colour audioInputColour{70, 200, 150};
/** @brief Fixed stereo host-output node colour. */
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
         type == NodeType::blackBox;
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

/** @brief Coarse audio tensor shape used for interactive link validation. */
struct ShapeSignature {
  /** @brief Channel count, or zero when inferred from the adjacent node. */
  int channels = 0;
  /** @brief Human-readable temporal domain name. */
  std::string domain{"audio"};

  /** @brief Returns whether this shape can connect to another endpoint. */
  [[nodiscard]] bool
  isCompatibleWith(const ShapeSignature &other) const noexcept {
    return domain == other.domain &&
           (channels == 0 || other.channels == 0 || channels == other.channels);
  }
};

/** @brief User-visible label of the TCN/BlackBox control (former FiLM) pin. */
inline constexpr const char *controlPinLabel = "control";

/** @brief A stable endpoint belonging to one graph node. */
struct Pin {
  /** @brief Stable graph-wide endpoint identifier. */
  std::int32_t id = 0;
  /** @brief User-visible endpoint label. */
  std::string label;
  /** @brief Input or output direction. */
  PinKind kind = PinKind::input;
  /** @brief Audio shape accepted or produced by the endpoint. */
  ShapeSignature shape;
  /** @brief Whether this pin carries audio or scalar conditioning. */
  SignalKind signalKind = SignalKind::audio;
};

/**
 * @brief Returns true for the TCN/BlackBox control input (accepts any signal).
 * @param pin Endpoint to inspect.
 */
inline bool isControlInputPin(const Pin &pin) noexcept {
  return pin.kind == PinKind::input &&
         (pin.signalKind == SignalKind::conditioning || pin.label == "film" ||
          pin.label == controlPinLabel);
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

/** @brief Minimum supported node-editor zoom level. */
inline constexpr float minimumZoom = 0.25f;
/** @brief Maximum supported node-editor zoom level. */
inline constexpr float maximumZoom = 2.0f;
/** @brief Default minimap width in Dear ImGui pixels. */
inline constexpr float mapWidth = 190.0f;
/** @brief Default minimap height in Dear ImGui pixels. */
inline constexpr float mapHeight = 125.0f;
} // namespace openyourbox::graph
