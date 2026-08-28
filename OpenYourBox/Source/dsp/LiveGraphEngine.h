#pragma once

#include "../graph/NodeGraph.h"

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace openyourbox::dsp {
/** @brief One-pole chase length for Gain, Knob, XY, and Dry/Wet (seconds). */
inline constexpr double controlRampSecondsDefault = 0.15;

/**
 * @struct LiveGraphCompileOptions
 * @brief Host and execution constraints used while compiling a graph snapshot.
 */
struct LiveGraphCompileOptions {
  /** @brief Number of host input channels; only mono and stereo are supported.
   */
  int hostInputChannels = 2;
  /** @brief Number of host output channels; only mono and stereo are supported.
   */
  int hostOutputChannels = 2;
  /** @brief Largest audio block accepted by a prepared runtime. */
  std::int64_t maximumBlockSize = 512;
  /** @brief Largest per-element causal history accepted during compilation. */
  std::uint64_t maximumHistorySamples = 1048576;
  /** @brief Host sample rate used to size control-value ramps. */
  double sampleRate = 44100.0;
  /**
   * @brief One-pole chase duration for Gain, Knob Input, and XY Trackpad.
   *
   * Linear ramps restart their slope on every UI update (~60 Hz) and buzz.
   * Exponential smoothing stays continuous while the target moves. Knob/XY
   * may be 0, so the chase is additive rather than multiplicative.
   */
  double controlRampSeconds = controlRampSecondsDefault;
};

/**
 * @class FrozenBlackBoxKernel
 * @brief Per-runtime mutable executor for one preloaded frozen BlackBox.
 *
 * Instances are created off the audio thread by a FrozenBlackBoxFactory and are
 * then owned by exactly one LiveGraphRuntime.
 */
class FrozenBlackBoxKernel {
public:
  /** @brief Allows destruction through the abstract kernel interface. */
  virtual ~FrozenBlackBoxKernel() = default;

  /**
   * @brief Processes one tensor shaped [1, channels, samples].
   * @param input Contiguous CPU float tensor for the current audio block.
   * @return Tensor shaped [1, outputChannels, samples].
   */
  virtual torch::Tensor forward(const torch::Tensor &input) = 0;

  /**
   * @brief Processes audio with an optional live conditioning tensor.
   * @param input Contiguous CPU float tensor for the current audio block.
   * @param conditioning Conditioning tensor, or undefined to use zeros.
   * @return Tensor shaped [1, outputChannels, samples].
   */
  virtual torch::Tensor forwardWithConditioning(const torch::Tensor &input,
                                                const torch::Tensor &conditioning) {
    (void)conditioning;
    return forward(input);
  }

  /**
   * @brief Encodes audio into a full-width latent trajectory when exported.
   * @param input Contiguous CPU float tensor for the current audio block.
   * @return Latent tensor, or an undefined tensor when the method is absent.
   */
  virtual torch::Tensor encode(const torch::Tensor &input) {
    (void)input;
    return {};
  }

  /**
   * @brief Decodes a full-width latent trajectory when exported.
   * @param latent Contiguous CPU float latent tensor.
   * @return Audio tensor, or an undefined tensor when the method is absent.
   */
  virtual torch::Tensor decode(const torch::Tensor &latent) {
    (void)latent;
    return {};
  }

  /** @brief True when encode and decode methods were validated on the artifact. */
  [[nodiscard]] virtual bool hasEncodeDecode() const noexcept { return false; }

  /** @brief Compactness mean buffer, or undefined. */
  [[nodiscard]] virtual torch::Tensor compactnessMean() const { return {}; }

  /** @brief Compactness PCA basis, or undefined. */
  [[nodiscard]] virtual torch::Tensor compactnessPca() const { return {}; }

  /** @brief Cumulative singular-value ratios, or undefined. */
  [[nodiscard]] virtual torch::Tensor compactnessCumulative() const { return {}; }

  /** @brief True when validation-PCA compactness may be applied. */
  [[nodiscard]] virtual bool compactnessReady() const noexcept { return false; }
};

/**
 * @class FrozenBlackBoxFactory
 * @brief Immutable metadata and off-thread constructor for a frozen hook.
 */
class FrozenBlackBoxFactory {
public:
  /** @brief Allows destruction through the abstract factory interface. */
  virtual ~FrozenBlackBoxFactory() = default;

  /** @brief Returns the exact input channel count accepted by the artifact. */
  [[nodiscard]] virtual int getInputChannels() const noexcept = 0;

  /** @brief Returns the exact output channel count produced by the artifact. */
  [[nodiscard]] virtual int getOutputChannels() const noexcept = 0;

  /** @brief Returns the artifact receptive field in samples. */
  [[nodiscard]] virtual std::uint64_t getReceptiveField() const noexcept = 0;

  /** @brief Returns the artifact trainable parameter count. */
  [[nodiscard]] virtual std::uint64_t getParameterCount() const noexcept = 0;

  /**
   * @brief Reports whether an all-zero input is guaranteed to remain zero.
   * @return True only when the artifact contract guarantees silence
   * preservation.
   */
  [[nodiscard]] virtual bool preservesSilence() const noexcept = 0;

  /**
   * @brief Creates an inference kernel outside the audio callback.
   * @return A prepared kernel, or null when the artifact cannot be loaded.
   */
  [[nodiscard]] virtual std::unique_ptr<FrozenBlackBoxKernel>
  createKernel() const = 0;

  /**
   * @brief Reports whether the artifact exports encode/decode beside forward.
   * @return True when those methods were present at load.
   */
  [[nodiscard]] virtual bool hasEncodeDecode() const noexcept { return false; }
};

/**
 * @using FrozenBlackBoxResolver
 * @brief Resolves a frozen graph node to immutable artifact metadata.
 */
using FrozenBlackBoxResolver =
    std::function<std::shared_ptr<const FrozenBlackBoxFactory>(
        const graph::GraphNode &)>;

/**
 * @struct RuntimeControlState
 * @brief Lock-free published Gain and conditioning values for one graph revision.
 */
struct RuntimeControlState {
  /** @brief Per-node Gain overrides used by Activation and TCN elements. */
  std::unordered_map<std::int32_t, float> gainByNodeId;
  /** @brief Per-node Knob/XY values; XY stores X in [0] and Y in [1]. */
  std::unordered_map<std::int32_t, std::array<float, 2>> conditioningByNodeId;
  /** @brief Per-node fidelity percent for bottleneck and Gold RAVE. */
  std::unordered_map<std::int32_t, float> fidelityByNodeId;
};

/** @brief Signal source used to compute a static analysis snapshot. */
enum class AnalysisSourceMode { live, probe };

/** @brief Degraded analysis status reported to the editor panel. */
enum class AnalysisStatus {
  live,
  probeFallback,
  disconnected,
  unavailable
};

/**
 * @struct AnalysisSeries
 * @brief Sampled plot trace for one channel or feature dimension.
 */
struct AnalysisSeries {
  /** @brief Zero-based channel or feature index. */
  int channelIndex = 0;
  /** @brief Compact legend label such as `L`, `R`, or `ch3`. */
  std::string channelLabel;
  /** @brief Sampled horizontal axis values. */
  std::vector<float> x;
  /** @brief Sampled vertical axis values. */
  std::vector<float> y;
};

/**
 * @struct TransferMarker
 * @brief Playback-only operating point constrained to the chain transfer curve.
 */
struct TransferMarker {
  /** @brief Input amplitude used to locate the marker. */
  float inputLevel = 0.0f;
  /** @brief Output amplitude lying on the chain curve. */
  float outputLevel = 0.0f;
  /** @brief Channel whose chain series the marker follows. */
  int channelIndex = 0;
};

/**
 * @struct AnalysisSnapshot
 * @brief Immutable dual-family analysis result consumed by InfoPanel.
 */
struct AnalysisSnapshot {
  /** @brief Node that was analysed. */
  std::int32_t nodeId = 0;
  /** @brief Live Blue or frozen Gold runtime state of the node. */
  graph::NodeState runtimeState = graph::NodeState::liveBlue;
  /** @brief Whether live audio or an internal probe drove the curves. */
  AnalysisSourceMode sourceMode = AnalysisSourceMode::probe;
  /** @brief Requested plot family. */
  graph::AnalysisView view = graph::AnalysisView::transfer;
  /** @brief Editor-facing degraded-mode status. */
  AnalysisStatus status = AnalysisStatus::unavailable;
  /** @brief Number of traces in each curve family. */
  int channelCount = 0;
  /** @brief Cumulative chain traces, length `channelCount`. */
  std::vector<AnalysisSeries> chainSeries;
  /** @brief Isolated element traces, length `channelCount`. */
  std::vector<AnalysisSeries> elementOnlySeries;
  /** @brief Chain-curve marker present only during playback transfer view. */
  std::optional<TransferMarker> transferMarker;
  /** @brief Graph revision that produced this snapshot. */
  std::uint64_t generatedAtRevision = 0;
  /** @brief True when a newer graph revision exists. */
  bool isStale = false;
};

/**
 * @struct AnalysisRequest
 * @brief Message-thread description of one analysis computation.
 */
struct AnalysisRequest {
  /** @brief Selected graph node. */
  std::int32_t nodeId = 0;
  /** @brief Plot family to compute. */
  graph::AnalysisView view = graph::AnalysisView::transfer;
  /** @brief Current graph/control revision token. */
  std::uint64_t revision = 0;
  /** @brief True when the host transport is playing. */
  bool playbackActive = false;
  /** @brief True when published live audio is loud enough to drive plots. */
  bool liveInputSuitable = false;
  /** @brief Latest published host-input peak used by the transfer marker. */
  float liveInputPeak = 0.0f;
  /** @brief Latest published tap-output peak used by the transfer marker. */
  float liveOutputPeak = 0.0f;
  /** @brief Channel index used when placing the transfer marker. */
  int liveChannelIndex = 0;
  /** @brief Host sample rate used to label frequency axes. */
  double sampleRate = 44100.0;
  /** @brief Optional captured live input, planar [channels * samples]. */
  const float *liveInput = nullptr;
  /** @brief Channel count of `liveInput`. */
  int liveInputChannels = 0;
  /** @brief Sample count of `liveInput`. */
  int liveInputSamples = 0;
};

/**
 * @brief Advances a monotonic analysis/graph revision token.
 * @param current Previous token.
 * @return Next token, wrapping at the unsigned maximum.
 */
inline std::uint64_t nextGraphRevision(std::uint64_t current) noexcept {
  return current + 1U;
}

/**
 * @brief Builds a compact legend label for one analysis trace.
 * @param channelIndex Zero-based channel or feature index.
 * @param channelCount Total traces at the analysis point.
 * @return `L`/`R` for stereo audio; otherwise `chN` for every channel
 *   including each feature dimension plotted on the shared axes.
 */
inline std::string analysisChannelLabel(int channelIndex,
                                        int channelCount) {
  if (channelCount == 2)
    return channelIndex == 0 ? std::string("L") : std::string("R");
  return "ch" + std::to_string(channelIndex + 1);
}

/**
 * @brief Collects Gain and Knob/XY values from an editable graph document.
 * @param graphDocument Message-thread graph.
 * @return Control table suitable for atomic audio-thread publication.
 */
[[nodiscard]] RuntimeControlState
collectRuntimeControlState(const graph::NodeGraph &graphDocument);

/**
 * @enum LiveGraphErrorCode
 * @brief Stable category for a graph compilation failure.
 */
enum class LiveGraphErrorCode {
  /** @brief No failure occurred. */
  none,
  /** @brief Host channel or block constraints are unsupported. */
  invalidCompileOptions,
  /** @brief A node, pin, link, or identifier is malformed. */
  invalidGraph,
  /** @brief The graph does not contain exactly one input and output boundary.
   */
  invalidBoundary,
  /** @brief A directed cycle prevents topological execution. */
  cycle,
  /** @brief Inferred channel dimensions are invalid or incompatible. */
  invalidShape,
  /** @brief An element property is absent or outside its valid range. */
  invalidProperty,
  /** @brief A frozen BlackBox cannot be safely represented or loaded. */
  invalidBlackBox,
  /** @brief A requested element cannot be randomized. */
  invalidRandomization,
  /** @brief The graph has no complete Audio Input-to-Output path. */
  incompletePath,
  /** @brief LibTorch failed while constructing an immutable snapshot. */
  torchFailure
};

/**
 * @struct LiveGraphCompileError
 * @brief Detailed non-throwing result for a failed graph compilation.
 */
struct LiveGraphCompileError {
  /** @brief Machine-readable failure category. */
  LiveGraphErrorCode code = LiveGraphErrorCode::none;
  /** @brief Node associated with the failure, or zero for graph-wide errors. */
  std::int32_t nodeId = 0;
  /** @brief Human-readable failure explanation suitable for the UI. */
  std::string message;

  /** @brief Returns true when this value represents a failure. */
  [[nodiscard]] bool hasError() const noexcept {
    return code != LiveGraphErrorCode::none;
  }
};

/**
 * @struct LiveGraphElementStatistics
 * @brief Immutable shape and complexity metadata for one compiled element.
 */
struct LiveGraphElementStatistics {
  /** @brief Stable source graph node identifier. */
  std::int32_t nodeId = 0;
  /** @brief Source graph element type. */
  graph::NodeType type = graph::NodeType::activation;
  /** @brief Exact inferred input channel count, or zero for Audio Input. */
  int inputChannels = 0;
  /** @brief Exact inferred output channel count, or zero for Audio Output. */
  int outputChannels = 0;
  /** @brief Element-local receptive field in samples. */
  std::uint64_t receptiveField = 1;
  /** @brief Number of mutable scalar parameters owned by the element. */
  std::uint64_t parameterCount = 0;
  /** @brief True when the live element supports deterministic randomization. */
  bool randomizable = false;
};

class LiveGraphSnapshot;
class LiveGraphRuntime;

/**
 * @struct LiveGraphCompileResult
 * @brief Snapshot-or-error result returned by LiveGraphEngine.
 */
struct LiveGraphCompileResult {
  /** @brief Immutable snapshot when compilation succeeded. */
  std::shared_ptr<const LiveGraphSnapshot> snapshot;
  /** @brief Failure details when snapshot is null. */
  LiveGraphCompileError error;

  /** @brief Returns true when an immutable snapshot was produced. */
  [[nodiscard]] bool succeeded() const noexcept {
    return snapshot != nullptr && !error.hasError();
  }
};

/**
 * @class LiveGraphSnapshot
 * @brief Immutable, validated, topologically ordered live graph program.
 *
 * A snapshot contains immutable architecture and weight tensors. Mutable causal
 * history and frozen kernels live in a separately prepared LiveGraphRuntime.
 */
class LiveGraphSnapshot final {
public:
  /** @brief Releases immutable implementation storage. */
  ~LiveGraphSnapshot();

  /** @brief Returns exact host input channels accepted by the snapshot. */
  [[nodiscard]] int getInputChannels() const noexcept;

  /** @brief Returns exact host output channels produced by the snapshot. */
  [[nodiscard]] int getOutputChannels() const noexcept;

  /** @brief Returns the maximum block size accepted by prepared runtimes. */
  [[nodiscard]] std::int64_t getMaximumBlockSize() const noexcept;

  /** @brief Returns the complete graph receptive field in samples. */
  [[nodiscard]] std::uint64_t getReceptiveField() const noexcept;

  /** @brief Returns the total mutable scalar parameter count. */
  [[nodiscard]] std::uint64_t getParameterCount() const noexcept;

  /** @brief Returns topologically ordered per-element statistics. */
  [[nodiscard]] const std::vector<LiveGraphElementStatistics> &
  getElementStatistics() const noexcept;

  /**
   * @brief Creates a new snapshot with exactly one weighted element randomized.
   * @param nodeId Stable weighted graph node identifier.
   * @param seed Signed deterministic element seed.
   * @param error Receives a failure description when no snapshot is returned.
   * @return New immutable snapshot; all non-target parameter tensors are shared
   * unchanged with this snapshot.
   */
  [[nodiscard]] std::shared_ptr<const LiveGraphSnapshot>
  withRandomizedElement(std::int32_t nodeId, std::int32_t seed,
                        LiveGraphCompileError &error) const;

private:
  /** @brief Opaque immutable snapshot representation. */
  struct Impl;

  /**
   * @brief Adopts a fully validated immutable implementation.
   * @param implementationToAdopt Snapshot storage prepared by the compiler.
   */
  explicit LiveGraphSnapshot(std::shared_ptr<const Impl> implementationToAdopt);

  /** @brief Immutable architecture, topology, metadata, and parameter storage.
   */
  std::shared_ptr<const Impl> implementation;

  /** @brief Grants the compiler access to snapshot construction. */
  friend class LiveGraphEngine;
  /** @brief Grants prepared runtimes access to the immutable execution plan. */
  friend class LiveGraphRuntime;
};

/**
 * @class LiveGraphRuntime
 * @brief Prepared mutable execution state paired with one immutable snapshot.
 *
 * Construct this object away from the audio callback, then atomically publish a
 * shared_ptr to it. Exactly one audio thread may execute a runtime at a time.
 */
class LiveGraphRuntime final {
public:
  /** @brief Releases prepared histories and frozen kernels. */
  ~LiveGraphRuntime();

  /** @brief Returns the immutable snapshot executed by this runtime. */
  [[nodiscard]] const std::shared_ptr<const LiveGraphSnapshot> &
  getSnapshot() const noexcept;

  /**
   * @brief Executes one CPU float tensor shaped [1, channels, samples].
   * @param input Current input block.
   * @return Graph output with the same temporal length.
   * @throws c10::Error When eager LibTorch inference fails.
   */
  torch::Tensor processTensor(const torch::Tensor &input);

  /**
   * @brief Processes planar host audio and clears output on any failure.
   * @param inputChannels Array of readable planar channel pointers.
   * @param inputChannelCount Number of readable host channels.
   * @param outputChannels Array of writable planar channel pointers.
   * @param outputChannelCount Number of writable host channels.
   * @param sampleCount Number of samples in each channel.
   * @return True when the graph produced a valid output block.
   *
   * This method performs no explicit C++ heap allocation after preparation,
   * but eager LibTorch operators may allocate internal tensor storage.
   */
  bool processHost(const float *const *inputChannels,
                   std::size_t inputChannelCount, float *const *outputChannels,
                   std::size_t outputChannelCount,
                   std::size_t sampleCount) noexcept;

  /**
   * @brief Publishes the latest message-thread Gain/conditioning table.
   * @param controls Immutable control table, or null to use compiled defaults.
   */
  void bindControls(
      std::shared_ptr<const RuntimeControlState> controls) noexcept;

  /**
   * @brief Processes one input block and returns the tapped node output.
   * @param input Current input block shaped [1, channels, samples].
   * @param nodeId Stable graph node to tap.
   * @return Tensor at the selected node, or an undefined tensor when absent.
   */
  torch::Tensor processTensorTapped(const torch::Tensor &input,
                                    std::int32_t nodeId);

  /**
   * @brief Runs one compiled element in isolation with a probe at its input.
   * @param nodeId Stable graph node to isolate.
   * @param probe Probe tensor shaped [1, inputChannels, samples].
   * @return Isolated element output, or an undefined tensor when absent.
   */
  torch::Tensor processIsolated(std::int32_t nodeId,
                                const torch::Tensor &probe);

  /**
   * @brief Returns the latest audio-thread peak pair for one compiled node.
   * @param nodeId Stable graph node identifier.
   * @param inputPeak Receives the upstream peak.
   * @param outputPeak Receives the node-output peak.
   * @return True when the node is present in this runtime.
   */
  [[nodiscard]] bool getTapPeaks(std::int32_t nodeId, float &inputPeak,
                                 float &outputPeak) const noexcept;

  /**
   * @brief Returns the latest audio-thread output RMS for one compiled node.
   *
   * Multi-channel, feature, and latent tensors collapse to one scalar
   * `sqrt(mean(x^2))` over every element of the output tensor.
   * @param nodeId Stable graph node identifier.
   * @param outputRms Receives the linear RMS level.
   * @return True when the node is present in this runtime.
   */
  [[nodiscard]] bool getTapRms(std::int32_t nodeId, float &outputRms) const
      noexcept;

  /**
   * @brief Returns the latest measured inference time for one frozen node.
   * @param nodeId Stable graph node identifier.
   * @return Per-buffer duration in milliseconds, or zero when unavailable.
   */
  [[nodiscard]] double
  getFrozenInferenceTimeMilliseconds(std::int32_t nodeId) const noexcept;

  /**
   * @brief Returns the node where host processing most recently failed.
   * @return Stable graph node id, zero when no failure is latched, or -1 when
   *   the failure occurred outside an individual element.
   *
   * The audio thread publishes only this integer marker; UI text is assembled
   * later on the message thread.
   */
  [[nodiscard]] std::int32_t
  getLastProcessingFailureNodeId() const noexcept;

  /** @brief Clears causal audio and FiLM control history without changing architecture. */
  void reset() noexcept;

private:
  /** @brief Opaque mutable runtime representation. */
  struct Impl;

  /**
   * @brief Adopts a fully prepared off-thread runtime implementation.
   * @param implementationToAdopt Runtime storage and frozen kernels.
   */
  explicit LiveGraphRuntime(std::unique_ptr<Impl> implementationToAdopt);

  /** @brief Mutable single-audio-thread histories, kernels, and host buffers.
   */
  std::unique_ptr<Impl> implementation;

  /** @brief Grants the compiler access to runtime construction. */
  friend class LiveGraphEngine;

  /**
   * @brief Populates upstream tensors before an isolated element executes.
   * @param target Topological index of the isolated element.
   * @param probe Probe tensor injected at audio input boundaries.
   */
  void seedIsolatedUpstreamOutputs(std::size_t target,
                                   const torch::Tensor &probe);

  /**
   * @brief Executes one compiled element into prepared runtime storage.
   * @param index Topological element index.
   * @param blockInput Current graph input block.
   */
  void executeElement(std::size_t index, const torch::Tensor &blockInput);
};

/**
 * @class LiveGraphEngine
 * @brief Off-audio-thread compiler and runtime preparer for editable graphs.
 */
class LiveGraphEngine final {
public:
  /**
   * @brief Validates and compiles an editable NodeGraph.
   * @param graphDocument Message-thread graph document to snapshot.
   * @param options Host shape and maximum block constraints.
   * @param blackBoxResolver Optional resolver required by frozen graph nodes.
   * @return Immutable snapshot or detailed validation failure.
   *
   * Unwired processing nodes are ignored. Nodes that can reach Audio Output are
   * executed when a complete Audio Input-to-Output path exists, including
   * Knob/XY subgraphs that join that path. A missing complete path returns
   * `incompletePath` rather than a validation error, so incomplete graphs may
   * sit on the canvas.
   */
  [[nodiscard]] static LiveGraphCompileResult
  compile(const graph::NodeGraph &graphDocument,
          const LiveGraphCompileOptions &options,
          FrozenBlackBoxResolver blackBoxResolver = {});

  /**
   * @brief Prepares causal state and frozen kernels off the audio thread.
   * @param snapshot Valid immutable graph snapshot.
   * @param error Receives preparation failure details.
   * @return Runtime suitable for atomic shared_ptr publication.
   */
  [[nodiscard]] static std::shared_ptr<LiveGraphRuntime>
  prepare(std::shared_ptr<const LiveGraphSnapshot> snapshot,
          LiveGraphCompileError &error);

  /**
   * @brief Computes dual chain/element-only analysis off the audio thread.
   * @param graphDocument Message-thread graph document to analyse.
   * @param request Selected node, view, live-capture, and revision metadata.
   * @param options Host shape and maximum block constraints.
   * @param blackBoxResolver Optional resolver required by frozen graph nodes.
   * @return Snapshot tagged with `generatedAtRevision`; never mutates audio state.
   */
  [[nodiscard]] static AnalysisSnapshot
  analyse(const graph::NodeGraph &graphDocument, const AnalysisRequest &request,
          const LiveGraphCompileOptions &options,
          FrozenBlackBoxResolver blackBoxResolver = {});
};
} // namespace openyourbox::dsp
