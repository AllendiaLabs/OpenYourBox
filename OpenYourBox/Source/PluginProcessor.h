#pragma once

#include <JuceHeader.h>

#include "dsp/LiveGraphPublisher.h"
#include "dsp/LookbackBuffer.h"
#include "dsp/TorchScriptBlackBox.h"
#include "dsp/WeightRandomizer.h"
#include "graph/NodeGraph.h"
#include "capture/CapturePairing.h"
#include "capture/CaptureRecorder.h"
#include "library/TrainingLibrary.h"
#include "state/EditHistory.h"
#include "state/PatchSnapshot.h"
#include "train/CloudSettings.h"

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class OpenYourBoxAudioProcessor
 * @brief Host-facing audio effect that owns parameter, model, and runtime
 * state.
 */
class OpenYourBoxAudioProcessor final
    : public juce::AudioProcessor,
      private juce::AudioProcessorValueTreeState::Listener,
      private juce::AsyncUpdater,
      private juce::Timer {
public:
  /** @brief Constructs the plugin and registers all parameter listeners. */
  OpenYourBoxAudioProcessor();
  /** @brief Unregisters listeners and cancels pending asynchronous work. */
  ~OpenYourBoxAudioProcessor() override;

  /** @brief Preallocates inference resources for the host configuration. */
  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  /** @brief Releases inference resources after playback stops. */
  void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
  /** @brief Accepts matching mono or stereo main buses. */
  bool isBusesLayoutSupported(const BusesLayout &layouts) const override;
#endif

  /** @brief Processes one audio block through the current immutable TCN. */
  void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

  /** @brief Creates the Dear ImGui plugin editor. */
  juce::AudioProcessorEditor *createEditor() override;
  /** @brief Reports that this processor supplies an editor. */
  bool hasEditor() const override;

  /** @brief Returns the host-visible plugin name. */
  const juce::String getName() const override;

  /** @brief Reports MIDI input support for randomize CC control. */
  bool acceptsMidi() const override;
  /** @brief Reports that the effect does not emit MIDI. */
  bool producesMidi() const override;
  /** @brief Reports that this is an audio effect, not a MIDI effect. */
  bool isMidiEffect() const override;
  /** @brief Returns the causal tail so the host keeps processing after stop. */
  double getTailLengthSeconds() const override;

  /** @brief Returns the mandatory single host program. */
  int getNumPrograms() override;
  /** @brief Returns the active host program index. */
  int getCurrentProgram() override;
  /** @brief Accepts the only valid program index. */
  void setCurrentProgram(int index) override;
  /** @brief Returns the default program name. */
  const juce::String getProgramName(int index) override;
  /** @brief Ignores host attempts to rename the fixed program. */
  void changeProgramName(int index, const juce::String &newName) override;

  /** @brief Serializes parameters, architecture hash, and exact model weights.
   */
  void getStateInformation(juce::MemoryBlock &destData) override;
  /** @brief Restores parameters and exact model weights when compatible. */
  void setStateInformation(const void *data, int sizeInBytes) override;

  /**
   * @brief Captures the live patch as a shared snapshot (graph + params + weights).
   */
  [[nodiscard]] openyourbox::state::PatchSnapshot capturePatchSnapshot();

  /**
   * @brief Applies a snapshot on the message thread with atomic runtime publish.
   * @param snapshot Patch to restore.
   * @param options Weight policy and viewport handling.
   * @param error Receives a user-facing failure.
   * @return True when the live patch was replaced.
   */
  bool applyPatchSnapshot(const openyourbox::state::PatchSnapshot &snapshot,
                          const openyourbox::state::ApplyOptions &options,
                          juce::String &error);

  /** @brief Returns the session undo/redo controller. */
  [[nodiscard]] openyourbox::state::EditHistory &getEditHistory() noexcept;

  /** @brief Returns the session undo/redo controller. */
  [[nodiscard]] const openyourbox::state::EditHistory &
  getEditHistory() const noexcept;

  /** @brief Returns the current named-preset association. */
  [[nodiscard]] openyourbox::state::CurrentPresetState &
  getCurrentPreset() noexcept;

  /** @brief Returns the current named-preset association. */
  [[nodiscard]] const openyourbox::state::CurrentPresetState &
  getCurrentPreset() const noexcept;

  /**
   * @brief Replaces the current-preset association.
   * @param next Association to store.
   */
  void setCurrentPreset(openyourbox::state::CurrentPresetState next);

  /**
   * @brief Marks the current preset dirty when one is associated.
   */
  void markPresetDirty();

  /**
   * @brief Clears dirty and refreshes the baseline fingerprint.
   * @param fingerprint Sonic fingerprint of the just-saved or loaded patch.
   */
  void clearPresetDirty(const juce::String &fingerprint);

  /**
   * @brief Sets dirty from whether @p fingerprint differs from the baseline.
   * @param fingerprint Live patch fingerprint.
   */
  void refreshPresetDirtyFromFingerprint(const juce::String &fingerprint);

  /** @brief Optional GUI callback invoked after a successful snapshot apply. */
  std::function<void()> onPatchApplied;

  /** @brief Returns mutable APVTS access for editor controls and attachments.
   */
  juce::AudioProcessorValueTreeState &getParameterState() noexcept;

  /** @brief Returns a snapshot of the currently requested architecture. */
  [[nodiscard]] openyourbox::dsp::TCNConfiguration
  getRequestedConfiguration() const noexcept;

  /** @brief Returns the current receptive field in samples. */
  [[nodiscard]] std::uint64_t getReceptiveFieldSamples() const noexcept;

  /** @brief Returns the current model parameter count. */
  [[nodiscard]] std::uint64_t getModelParameterCount() const noexcept;

  /**
   * @brief Returns the latest per-buffer inference time for a frozen node.
   * @param nodeId Stable frozen graph node identifier.
   * @return Duration in milliseconds, or zero before the first processed block.
   */
  [[nodiscard]] double
  getFrozenInferenceTimeMilliseconds(std::int32_t nodeId) const noexcept;

  /** @brief Returns the current host sample rate. */
  [[nodiscard]] double getCurrentSampleRate() const noexcept;

  /** @brief Returns the most recent asynchronous build/runtime error. */
  [[nodiscard]] juce::String getModelError() const;

  /**
   * @brief Applies an inline TCN graph configuration through host parameters.
   * @param configuration Valid live TCN configuration.
   */
  void applyGraphConfiguration(
      const openyourbox::dsp::TCNConfiguration &configuration);

  /**
   * @brief Requests deterministic randomization for one live weighted element.
   * @param nodeId Stable graph element identifier.
   * @param seed Signed per-element seed.
   */
  void randomizeGraphElement(std::int32_t nodeId, std::int32_t seed);

  /** @brief Returns the latest persisted message-thread graph snapshot. */
  [[nodiscard]] juce::ValueTree getGraphState() const;

  /**
   * @brief Publishes a graph snapshot for project-state persistence.
   * @param graphState Complete serialized graph document.
   * @param compileRuntime Whether the audio graph should be recompiled.
   */
  void setGraphState(const juce::ValueTree &graphState,
                     bool compileRuntime = true);

  /**
   * @brief Publishes Gain and Knob/XY values without recompiling the graph.
   * @param controls Immutable control table consumed on the audio thread.
   */
  void setRuntimeControls(
      const openyourbox::dsp::RuntimeControlState &controls);

  /** @brief Returns the monotonic graph/control revision used by analysis. */
  [[nodiscard]] std::uint64_t getGraphRevision() const noexcept;

  /** @brief Returns true when the host transport is currently playing. */
  [[nodiscard]] bool isTransportPlaying() const noexcept;

  /**
   * @brief Copies the latest lock-free live-capture slot for analysis.
   * @param inputPeak Receives the host-input peak.
   * @param outputPeak Receives the host-output peak.
   * @param suitable Receives whether live audio can drive analysis.
   * @param input Destination for planar captured input, or null to skip copy.
   * @param maxSamples Capacity of each `input` channel.
   * @param channels Receives captured channel count.
   * @param samples Receives captured sample count.
   */
  void copyLiveCapture(float &inputPeak, float &outputPeak, bool &suitable,
                       float *const *input, int maxSamples, int &channels,
                       int &samples) const noexcept;

  /**
   * @brief Reads the latest audio-thread tap peaks for one compiled node.
   * @param nodeId Stable graph node identifier.
   * @param inputPeak Receives the upstream peak.
   * @param outputPeak Receives the node-output peak.
   * @return True when a published runtime contains the node.
   */
  [[nodiscard]] bool getAnalysisTapPeaks(std::int32_t nodeId, float &inputPeak,
                                         float &outputPeak) const noexcept;

  /**
   * @brief Reads the latest audio-thread output RMS for one compiled node.
   * @param nodeId Stable graph node identifier.
   * @param outputRms Receives the linear RMS level collapsed over all dimensions.
   * @return True when a published runtime contains the node.
   */
  [[nodiscard]] bool getNodeOutputRms(std::int32_t nodeId,
                                      float &outputRms) const noexcept;

  /**
   * @brief Loads, warms, and publishes a validated frozen artifact handle.
   * @param result Worker result containing artifact path and channel metadata.
   * @param error Receives a human-readable load or validation error.
   * @return True only when the prepared artifact handle has been published.
   *
   * This function performs blocking file and Torch work and must be called from
   * a background thread. Publication alone does not switch processBlock to the
   * artifact; graph-runtime integration owns that separate transition.
   */
  bool
  prepareFrozenArtifact(const openyourbox::graph::FreezeSelectionResult &result,
                        std::string &error);

  /**
   * @brief Loads a trained artifact without requiring silence preservation.
   * @param result Train worker success payload.
   * @param error Receives a human-readable load or validation error.
   * @return True only when the prepared artifact handle has been published.
   */
  bool prepareTrainedArtifact(const openyourbox::graph::TrainJobResult &result,
                              std::string &error);

  /**
   * @brief Loads an external TorchScript checkpoint into the frozen registry.
   * @param artifactPath Absolute path to a `.pt`, `.pth`, or `.ts` file.
   * @param inputChannelsHint Preferred probe width (host channels or override).
   * @param error Receives a human-readable load or validation error.
   * @return True only when the prepared factory has been published.
   *
   * Performs blocking `torch::jit::load` and must not run on the audio thread.
   * Silence preservation is not required. Channel hints retry `{hint, 1, 2, 4,
   * 8, 16}` when the first probe fails.
   */
  bool prepareExternalArtifact(const std::string &artifactPath,
                               int inputChannelsHint, std::string &error);

  /**
   * @brief Returns the last factory published for @p artifactPath.
   * @param artifactPath Canonical checkpoint path.
   */
  [[nodiscard]] std::shared_ptr<const openyourbox::dsp::TorchScriptBlackBoxFactory>
  getPreparedExternalArtifact(const std::string &artifactPath) const;

  /**
   * @brief Drops a registry entry when no persisted node still references it.
   * @param artifactPath Canonical checkpoint path.
   */
  void releaseFrozenArtifactIfUnused(const std::string &artifactPath);

  /**
   * @brief Compiles armed nodes as a frozen preview of a training checkpoint.
   * @param artifactPath Validated TorchScript checkpoint path.
   * @param nodeIds Armed node identifiers captured at Run.
   */
  void setTrainingPreview(const std::string &artifactPath,
                          const std::vector<std::int32_t> &nodeIds);

  /** @brief Clears a hear-while-training preview and recompiles the live graph. */
  void clearTrainingPreview();

  /**
   * @brief Sets the capture bypass flag read by the audio thread.
   * @param enabled True to passthrough input while capturing.
   */
  void setCaptureBypass(bool enabled) noexcept;

  /** @brief Returns the capture bypass flag. */
  [[nodiscard]] bool isCaptureBypassEnabled() const noexcept;

  /**
   * @brief Starts capturing host input into a preallocated ring.
   * @param destination WAV destination.
   * @param sampleRate Host sample rate.
   * @param channels Host input channels.
   * @return False when the WAV writer could not be opened.
   */
  bool startInputCapture(const juce::File &destination, double sampleRate,
                         int channels);

  /** @brief Drains the capture ring on the message thread. */
  void drainInputCapture();

  /**
   * @brief Stops capture and closes the WAV file.
   * @return Destination file when samples were written.
   */
  juce::File stopInputCapture();

  /** @brief Returns true while input capture is armed. */
  [[nodiscard]] bool isInputCaptureActive() const noexcept;

  /** @brief Returns the processor-owned pairing endpoint (editor-independent). */
  [[nodiscard]] openyourbox::capture::CapturePairing &getCapturePairing() noexcept;

  /** @brief Returns the processor-owned pairing endpoint. */
  [[nodiscard]] const openyourbox::capture::CapturePairing &
  getCapturePairing() const noexcept;

  /** @brief Returns the master-owned training library. */
  [[nodiscard]] openyourbox::library::TrainingLibrary &getTrainingLibrary() noexcept;

  /** @brief Returns the master-owned training library. */
  [[nodiscard]] const openyourbox::library::TrainingLibrary &
  getTrainingLibrary() const noexcept;

  /**
   * @brief Sets whether Record should start host playback when stopped.
   * @param enabled True to request transport play (default).
   */
  void setStartTransportOnRecord(bool enabled) noexcept;

  /** @brief Returns whether Record starts host playback when stopped. */
  [[nodiscard]] bool getStartTransportOnRecord() const noexcept;

  /** @brief Starts a synchronized dual-instance capture take. */
  void startPairedRecording();

  /** @brief Starts a single-instance unpaired clip capture. */
  void startSingleRecording();

  /** @brief Stops capture and assembles the library pair when both clips exist. */
  void stopPairedRecording();

  /**
   * @brief Returns the project training-config snapshot JSON.
   */
  [[nodiscard]] const juce::String &getTrainConfigJson() const noexcept {
    return trainConfigJson;
  }

  /**
   * @brief Stores the project training-config snapshot JSON.
   * @param json Serialized Training Configuration object.
   */
  void setTrainConfigJson(juce::String json) { trainConfigJson = std::move(json); }

  /**
   * @brief Returns persisted cloud account settings for this instance.
   */
  [[nodiscard]] openyourbox::train::CloudSettings &getCloudSettings() noexcept;

  /**
   * @brief Returns persisted cloud account settings for this instance.
   */
  [[nodiscard]] const openyourbox::train::CloudSettings &
  getCloudSettings() const noexcept;

  /**
   * @brief Consumes a pending capture status string for the editor.
   * @return User-facing status, or empty when none is waiting.
   */
  juce::String takeCaptureStatusMessage();

  /**
   * @brief Consumes a request to focus the Library side tab.
   * @return True when the editor should switch to Library.
   */
  bool takeLibraryFocusRequest() noexcept;

  /**
   * @brief Starts in-plugin preview of a library WAV on the audio output.
   * @param file Audio file to play.
   * @return False when the file could not be decoded.
   */
  bool startPreviewFile(const juce::File &file);

  /** @brief Stops in-plugin preview playback. */
  void stopPreview();

  /** @brief Returns true while preview audio is being mixed. */
  [[nodiscard]] bool isPreviewPlaying() const noexcept;

  /**
   * @brief Releases the matching published frozen artifact on unfreeze.
   * @param artifactPath Artifact owned by the frozen group being restored.
   */
  void releaseFrozenArtifact(const std::string &artifactPath);

  /**
   * @brief Reports whether a matching validated artifact is published.
   * @param artifactPath Absolute local artifact path.
   * @return True when the artifact is ready for frozen runtime integration.
   */
  [[nodiscard]] bool
  hasPreparedFrozenArtifact(const std::string &artifactPath) const noexcept;

  /**
   * @brief Resolves a frozen graph node for off-thread analysis.
   * @param node Frozen BlackBox graph node.
   * @return Matching immutable factory, or null when unavailable.
   */
  [[nodiscard]] std::shared_ptr<const openyourbox::dsp::FrozenBlackBoxFactory>
  resolveFrozenBlackBoxForAnalysis(
      const openyourbox::graph::GraphNode &node) const;

private:
  /** @brief Immutable artifact map atomically replaced by background work. */
  struct FrozenArtifactRegistry {
    /** @brief Validated factories indexed by canonical artifact path. */
    std::unordered_map<
        std::string,
        std::shared_ptr<const openyourbox::dsp::TorchScriptBlackBoxFactory>>
        artifacts;
  };

  /** @brief Prepared state for the legacy Phase 1 whole-TCN runtime. */
  struct RuntimeState {
    std::shared_ptr<const openyourbox::dsp::ModelSnapshot> snapshot;
    openyourbox::dsp::LookbackBuffer lookback;
    torch::Tensor inputTensor;
    int maximumBlockSize = 0;
    /** @brief Per-channel previous DC-blocker input samples. */
    std::array<float, 2> dcInput{};
    /** @brief Per-channel previous DC-blocker output samples. */
    std::array<float, 2> dcOutput{};
  };

  void parameterChanged(const juce::String &parameterID,
                        float newValue) override;
  void handleAsyncUpdate() override;
  void publishRuntime(
      const std::shared_ptr<const openyourbox::dsp::ModelSnapshot> &snapshot);
  std::shared_ptr<RuntimeState> createRuntime(
      const std::shared_ptr<const openyourbox::dsp::ModelSnapshot> &snapshot);
  torch::Tensor runModel(RuntimeState &runtime,
                         const juce::AudioBuffer<float> &input, int numSamples);
  /**
   * @brief Applies a preallocated first-order high-pass stage in place.
   * @param runtime Active runtime containing per-channel filter memory.
   * @param buffer Host audio buffer.
   * @param channels Number of channels to process.
   * @param samples Number of valid samples.
   */
  void applyDcBlocker(RuntimeState &runtime, juce::AudioBuffer<float> &buffer,
                      int channels, int samples) noexcept;
  /**
   * @brief Applies the DC blocker with explicit per-channel state.
   * @param inputState Previous input samples.
   * @param outputState Previous output samples.
   * @param buffer Host audio buffer.
   * @param channels Number of channels to process.
   * @param samples Number of valid samples.
   */
  void applyDcBlocker(std::array<float, 2> &inputState,
                      std::array<float, 2> &outputState,
                      juce::AudioBuffer<float> &buffer, int channels,
                      int samples) noexcept;
  /**
   * @brief Processes through the latest published modular graph when ready.
   * @param buffer In-place host audio buffer.
   * @param inputChannels Number of readable input channels.
   * @param numSamples Number of valid samples.
   * @return True when a modular runtime produced the block.
   */
  bool processLiveGraph(juce::AudioBuffer<float> &buffer, int inputChannels,
                        int numSamples) noexcept;
  /** @brief Queues the latest persisted graph for off-thread compilation. */
  void requestGraphCompile();
  /**
   * @brief Points the Dry/Wet smoother at the current host parameter.
   */
  void syncDryWetSmoother() noexcept;
  /**
   * @brief Resolves a frozen graph node to a validated artifact factory.
   * @param node Frozen BlackBox graph node.
   * @return Matching immutable factory, or null when unavailable.
   */
  [[nodiscard]] std::shared_ptr<const openyourbox::dsp::FrozenBlackBoxFactory>
  resolveFrozenBlackBox(const openyourbox::graph::GraphNode &node) const;
  void requestCurrentArchitecture(bool randomizeWeights);
  void resetRandomizeParameter();
  /**
   * @brief Mixes in-plugin preview audio into the host output buffer.
   * @param buffer Host output buffer.
   * @param channels Number of output channels.
   * @param samples Block size.
   */
  void mixPreview(juce::AudioBuffer<float> &buffer, int channels,
                  int samples) noexcept;
  /** @brief Drains the capture ring while a take is active. */
  void timerCallback() override;
  /**
   * @brief Handles pairing control JSON on the message thread.
   * @param message Parsed control object.
   */
  void handlePairingMessage(const juce::var &message);
  /**
   * @brief Opens a local WAV capture destination for the current take.
   * @param pairId Shared take identifier.
   * @param suffix Filename suffix such as `_local` or `_slave`.
   * @return False when the WAV writer could not be opened.
   */
  bool startLocalCapture(const juce::String &pairId, const juce::String &suffix);
  /** @brief Requests host playback on the next audio block when stopped. */
  void requestTransportStartIfNeeded() noexcept;
  /** @brief Assembles the library pair once both clips are present. */
  void tryAssembleCapturedPair();
  /**
   * @brief Stores a user-facing capture status for the editor.
   * @param message Status text.
   */
  void setCaptureStatusMessage(const juce::String &message);

  /**
   * @brief Re-prepares every persisted `external_load` checkpoint into the registry.
   *
   * Missing files keep the path string and are marked error. Existing files
   * are loaded off the audio thread (caller must already be off-audio).
   */
  void reprepareExternalArtifactsFromGraph();

  /**
   * @brief Applies a history snapshot and restores current-preset state.
   * @param snapshot Patch to restore.
   * @param association Current-preset to restore.
   */
  bool applyHistorySnapshot(
      const openyourbox::state::PatchSnapshot &snapshot,
      const openyourbox::state::CurrentPresetState &association);

  juce::AudioProcessorValueTreeState parameters;
  openyourbox::dsp::WeightRandomizer modelBuilder;
  /** @brief Atomically published immutable frozen artifact registry. */
  mutable std::shared_ptr<const FrozenArtifactRegistry>
      publishedFrozenArtifacts;
  /** @brief Dedicated off-thread modular graph compiler and publisher. */
  openyourbox::dsp::LiveGraphPublisher graphPublisher;
  mutable std::shared_ptr<RuntimeState> publishedRuntime;
  std::shared_ptr<RuntimeState> activeRuntime;
  std::shared_ptr<RuntimeState> previousRuntime;
  /** @brief Modular runtime currently owned by the audio thread. */
  std::shared_ptr<openyourbox::dsp::LiveGraphRuntime> activeGraphRuntime;
  /** @brief Prior modular runtime retained during click-free replacement. */
  std::shared_ptr<openyourbox::dsp::LiveGraphRuntime> previousGraphRuntime;
  /** @brief Preallocated output workspace for the active modular graph. */
  juce::AudioBuffer<float> graphWetBuffer;
  /** @brief Preallocated output workspace for graph crossfades. */
  juce::AudioBuffer<float> previousGraphWetBuffer;
  /** @brief Per-channel graph DC-blocker input memory. */
  std::array<float, 2> graphDcInput{};
  /** @brief Per-channel graph DC-blocker output memory. */
  std::array<float, 2> graphDcOutput{};
  /** @brief Linear Dry/Wet ramp shared by modular and fallback mix paths. */
  juce::LinearSmoothedValue<float> dryWetSmoother{1.0f};

  std::atomic<double> currentSampleRate{44100.0};
  /** @brief Runtime high-pass feedback coefficient derived from sample rate. */
  std::atomic<float> dcBlockerCoefficient{0.997f};
  std::atomic<int> preparedBlockSize{0};
  std::atomic<bool> prepared{false};
  std::atomic<bool> architectureChangePending{false};
  std::atomic<bool> randomizePending{false};
  std::atomic<bool> midiRandomizePending{false};
  std::atomic<bool> restoringState{false};
  /** @brief True while applyPatchSnapshot is nested (rollback). */
  bool applyingSnapshot = false;
  std::atomic<bool> lastRandomizeValue{false};
  std::atomic<std::uint64_t> randomizationCounter{0};
  int crossfadeSamplesRemaining = 0;
  /** @brief Samples remaining in the current modular-runtime crossfade. */
  int graphCrossfadeSamplesRemaining = 0;

  mutable juce::CriticalSection errorLock;
  juce::String runtimeError;
  /** @brief Protects host-state access to the UI-owned graph snapshot. */
  mutable juce::CriticalSection graphStateLock;
  /** @brief Latest immutable graph document copy used for project recall. */
  juce::ValueTree persistedGraphState{"GraphDocument"};
  /** @brief Monotonic revision bumped on graph or conditioning publication. */
  std::atomic<std::uint64_t> graphRevision{1};
  /** @brief Latest Gain/conditioning table published for the audio thread. */
  mutable std::shared_ptr<const openyourbox::dsp::RuntimeControlState>
      publishedControls;
  /** @brief Double-buffered live input capture used by analysis. */
  struct LiveCaptureSlot {
    /** @brief Host-input peak of the captured block. */
    float inputPeak = 0.0f;
    /** @brief Host-output peak of the captured block. */
    float outputPeak = 0.0f;
    /** @brief True when the captured block is loud enough for live analysis. */
    bool suitable = false;
    /** @brief Captured channel count. */
    int channels = 0;
    /** @brief Captured sample count. */
    int samples = 0;
    /** @brief Planar captured input, stereo maximum of 512 samples. */
    std::array<std::array<float, 512>, 2> input{};
  };
  /** @brief Two capture slots swapped with `liveCaptureIndex`. */
  std::array<LiveCaptureSlot, 2> liveCaptureSlots{};
  /** @brief Index of the slot that message-thread readers should consume. */
  std::atomic<int> liveCaptureIndex{0};
  /** @brief True when the host play-head reports playback. */
  std::atomic<bool> transportPlaying{false};
  /** @brief Capture bypass flag defaulting on during a capture session. */
  std::atomic<bool> captureBypass{false};
  /** @brief True when Record should start host playback if it is stopped. */
  std::atomic<bool> startTransportOnRecord{true};
  /** @brief Audio-thread request to call AudioPlayHead::transportPlay. */
  std::atomic<bool> pendingTransportPlay{false};
  /** @brief Real-time-safe input recorder drained on the message thread. */
  openyourbox::capture::CaptureRecorder inputCapture;
  /** @brief Localhost pairing endpoint; advertises only while searching. */
  openyourbox::capture::CapturePairing capturePairing;
  /** @brief Durable training-library store owned by this instance. */
  openyourbox::library::TrainingLibrary trainingLibrary;
  /** @brief Identifier of the in-flight capture take. */
  juce::String activeCapturePairId;
  /** @brief Local clip written by this instance. */
  juce::File localCaptureClip;
  /** @brief Peer clip path received over IPC. */
  juce::File pendingPeerClip;
  /** @brief True after local stop while waiting for the peer WAV. */
  bool waitingForPeerClip = false;
  /** @brief True when the editor should switch to the Library tab. */
  bool libraryFocusRequested = false;
  /** @brief Latest capture status consumed by the editor. */
  juce::String captureStatusMessage;
  /** @brief Project training-config snapshot JSON. */
  juce::String trainConfigJson;
  /** @brief Linked platform session, URL overrides, and submitter ids. */
  openyourbox::train::CloudSettings cloudSettings;
  /** @brief True when the active take is unpaired Single capture. */
  bool singleCaptureActive = false;
  /** @brief Protects capture assembly paths and status. */
  juce::CriticalSection captureStateLock;
  /** @brief Preview playback buffer mixed on the audio thread. */
  struct PreviewState {
    /** @brief Decoded preview samples. */
    juce::AudioBuffer<float> samples;
    /** @brief Next sample index to mix. */
    std::atomic<int> position{0};
    /** @brief True while preview is audible. */
    std::atomic<bool> active{false};
  };
  /** @brief Atomically published preview playback state. */
  std::shared_ptr<PreviewState> previewPlayback;
  /** @brief Protects hear-while-training preview identifiers. */
  juce::CriticalSection trainingPreviewLock;
  /** @brief Checkpoint path compiled over armed nodes while training. */
  std::string trainingPreviewPath;
  /** @brief Armed node identifiers included in the training preview. */
  std::vector<std::int32_t> trainingPreviewNodeIds;
  /** @brief Session association to a named catalog entry. */
  openyourbox::state::CurrentPresetState currentPreset;
  /** @brief Session undo/redo stack owned by this plugin instance. */
  openyourbox::state::EditHistory editHistory;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OpenYourBoxAudioProcessor)
};
