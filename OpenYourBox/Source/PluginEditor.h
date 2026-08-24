#pragma once

#include "PluginProcessor.h"
#include "freeze/FreezeCoordinator.h"
#include "graph/NodeGraph.h"
#include "graph/NodeRenderer.h"
#include "library/CopyrightAcknowledgment.h"
#include "train/TrainCoordinator.h"
#include "ui/CaptureSamplesPanel.h"
#include "ui/CopyrightModal.h"
#include "ui/ErrorModal.h"
#include "ui/ImGuiHost.h"
#include "ui/InfoPanel.h"
#include "ui/TrainPanel.h"
#include "ui/TrainingLibraryPanel.h"
#include <JuceHeader.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @class OpenYourBoxAudioProcessorEditor
 * @brief Resizable Dear ImGui editor for TCN controls and graph visualization.
 */
class OpenYourBoxAudioProcessorEditor final : public juce::AudioProcessorEditor {
public:
  /** @brief Creates and attaches the OpenGL-backed editor. */
  OpenYourBoxAudioProcessorEditor(OpenYourBoxAudioProcessor &);
  /** @brief Releases editor resources before the processor. */
  ~OpenYourBoxAudioProcessorEditor() override;

  /** @brief Paints the fallback JUCE background behind OpenGL. */
  void paint(juce::Graphics &) override;
  /** @brief Fits the ImGui host to the complete editor area. */
  void resized() override;

private:
  void renderFrame();
  /** @brief Restores or synchronizes the editable graph document. */
  void updateGraphIfNeeded();
  /** @brief Publishes the current graph document to processor state.
   *  @param compileRuntime Whether the live audio graph should be rebuilt.
   */
  void persistGraph(bool compileRuntime = true);
  /**
   * @brief Publishes Knob/XY/Gain values without rebuilding the audio graph.
   */
  void publishRuntimeControls();
  /**
   * @brief Bumps analysis revision and refreshes the selected snapshot.
   */
  void invalidateAnalysis();
  /**
   * @brief Marks the current analysis snapshot as current for the graph revision.
   */
  void syncAnalysisRevision();
  /**
   * @brief Requests or refreshes analysis for the current target node.
   */
  void refreshAnalysisIfNeeded();
  /**
   * @brief Applies an inline node property to the live processor.
   * @param nodeId Stable graph node identifier.
   * @param key Canonical property key.
   * @param value Validated property value.
   */
  void handlePropertyChanged(std::int32_t nodeId, const std::string &key,
                             int value);
  /**
   * @brief Applies a real inline property such as Gain.
   * @param nodeId Stable graph node identifier.
   * @param key Canonical property key.
   * @param value Validated real value.
   */
  void handleFloatPropertyChanged(std::int32_t nodeId, const std::string &key,
                                  float value);
  /**
   * @brief Opens analysis on a Blue or Gold node.
   * @param nodeId Stable graph node identifier.
   */
  void handleAnalysisRequested(std::int32_t nodeId);
  /**
   * @brief Publishes Knob Input conditioning without a graph recompile.
   * @param nodeId Knob Input node identifier.
   * @param value Current conditioning scalar.
   */
  void handleKnobChanged(std::int32_t nodeId, float value);
  /**
   * @brief Publishes XY Trackpad conditioning without a graph recompile.
   * @param nodeId XY Trackpad node identifier.
   * @param x Current X conditioning scalar.
   * @param y Current Y conditioning scalar.
   */
  void handleXyChanged(std::int32_t nodeId, float x, float y);
  /**
   * @brief Requests element-scoped deterministic randomization.
   * @param nodeId Stable graph node identifier.
   * @param seed Signed per-element seed.
   */
  void handleRandomize(std::int32_t nodeId, std::int32_t seed);
  /**
   * @brief Starts a manual freeze for one or more source-to-sink chains.
   * @param selectedNodeIds Stable selected node identifiers.
   */
  void handleFreeze(const std::vector<std::int32_t> &selectedNodeIds);
  /**
   * @brief Starts compilation of the next queued freeze chain.
   * @return False when the next chain could not be submitted.
   */
  bool startNextFreezeChain();
  /**
   * @brief Restores a frozen Gold group to editable Live Blue elements.
   * @param nodeId Any frozen node in the compiled group.
   */
  void handleUnfreeze(std::int32_t nodeId);
  /** @brief Applies a completed background freeze result on the message thread.
   */
  void applyCompletedFreeze();
  /** @brief Applies a completed train job on the message thread. */
  void applyCompletedTrain();
  /**
   * @brief Loads a training checkpoint into the live graph when hear-while-training is on.
   */
  void pollTrainingPreview();
  /** @brief Starts a Train job when gates pass. */
  void handleTrainRun();
  /** @brief Requests the Library side tab on the next frame. */
  void focusLibraryTab();
  /** @brief Imports a file pair chosen via native file choosers. */
  void handleLibraryImport();
  /** @brief Opens the clean-file chooser on the message thread. */
  void promptLibraryImportClean();
  /** @brief Opens the processed-file chooser on the message thread. */
  void promptLibraryImportProcessed();
  /**
   * @brief Decodes and stores one imported pair off the UI/OpenGL threads.
   * @param clean Selected x file.
   * @param processed Selected y file.
   */
  void importLibraryPair(const juce::File &clean, const juce::File &processed);
  /** @brief Opens a native chooser to load compatible weights. */
  void handleBrowseWeights(std::int32_t nodeId);
  /**
   * @brief Displays transient non-error graph feedback.
   * @param message Human-readable status or validation text.
   */
  void showGraphMessage(const std::string &message);
  /**
   * @brief Opens the copyable error dialog for a plugin failure.
   * @param message Human-readable error text.
   * @param forceReopen True to show even when this text was already dismissed.
   */
  void showError(const juce::String &message, bool forceReopen = false);
  /**
   * @brief Opens the error dialog when the live graph reports a new failure.
   */
  void pollModelError();

  OpenYourBoxAudioProcessor &audioProcessor;
  /** @brief Owns detached freeze worker execution. */
  openyourbox::freeze::FreezeCoordinator freezeCoordinator;
  /** @brief Owns detached train worker execution. */
  openyourbox::train::TrainCoordinator trainCoordinator;
  openyourbox::graph::NodeGraph nodeGraph;
  openyourbox::ui::ImGuiHost imguiHost;
  openyourbox::graph::NodeRenderer nodeRenderer;
  openyourbox::library::CopyrightAcknowledgment copyrightAcknowledgment;
  openyourbox::ui::TrainingLibraryPanel libraryPanel;
  openyourbox::ui::CaptureSamplesPanel capturePanel;
  openyourbox::ui::TrainPanel trainPanel;
  openyourbox::ui::CopyrightModal copyrightModal;
  /** @brief Copyable error dialog for plugin, train, freeze, and model failures. */
  openyourbox::ui::ErrorModal errorModal;
  /** @brief Side-panel tab: 0 info, 1 library, 2 capture, 3 train. */
  int sidePanelTab = 0;
  /** @brief Tab index to select on the next frame, or -1. */
  int pendingSidePanelTab = -1;
  /** @brief True while the copyright modal should be shown. */
  bool copyrightModalVisible = false;
  /** @brief Last model/runtime error already presented in the error dialog. */
  juce::String lastPresentedError;
  /** @brief Last hear-while-training checkpoint path already applied. */
  std::string lastTrainPreviewPath;
  /** @brief Armed node identifiers captured when the current job started. */
  std::vector<std::int32_t> trainPreviewNodeIds;
  /** @brief True while an off-thread checkpoint load is in flight. */
  bool trainPreviewLoadInFlight = false;
  /** @brief Last successful train result retained for retry-load. */
  std::optional<openyourbox::graph::TrainJobResult> retryTrainResult;
  /** @brief Bypass state restored when leaving capture. */
  bool restoreBypassOnExit = false;
  openyourbox::dsp::TCNConfiguration displayedConfiguration;
  bool graphInitialized = false;
  /** @brief Current transient graph workflow message. */
  juce::String graphMessage;
  /** @brief Dear ImGui time when graph workflow feedback expires. */
  double graphMessageDeadline = 0.0;
  /** @brief Selection retained unchanged until worker and artifact success. */
  std::vector<std::int32_t> pendingFreezeSelection;
  /** @brief Remaining freeze chains waiting after the active request. */
  std::vector<std::vector<std::int32_t>> pendingFreezeChains;
  /** @brief Exact graph snapshot used to reject stale worker completions. */
  std::string pendingFreezeGraph;
  /** @brief Side panel showing architecture metrics and analysis plots. */
  openyourbox::ui::InfoPanel infoPanel;
  /** @brief Node currently shown in the analysis panel, or zero. */
  std::int32_t analysisNodeId = 0;
  /** @brief Latest computed analysis snapshot. */
  openyourbox::dsp::AnalysisSnapshot analysisSnapshot;
  /** @brief Graph revision consumed by the latest snapshot. */
  std::uint64_t analysisRevision = 0;
  /** @brief Dear ImGui time of the last analysis computation. */
  double lastAnalysisTime = 0.0;
  /** @brief True while the Dry/Wet slider is recording a host gesture. */
  bool dryWetGestureActive = false;
  /** @brief Native file chooser kept alive across async host dialogs. */
  std::shared_ptr<juce::FileChooser> fileChooser;
  /** @brief Clean file chosen before the processed-file dialog. */
  juce::File pendingImportClean;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OpenYourBoxAudioProcessorEditor)
};
