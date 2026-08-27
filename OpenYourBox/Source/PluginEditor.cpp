#include "PluginEditor.h"
#include "dsp/WeightLoader.h"
#include "library/UserDataPaths.h"
#include "params/ParamIDs.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
bool sameArchitecture(const openyourbox::dsp::TCNConfiguration &left,
                      const openyourbox::dsp::TCNConfiguration &right) noexcept {
  return left.depth == right.depth && left.kernelSize == right.kernelSize &&
         left.channels == right.channels &&
         left.inputChannels == right.inputChannels &&
         left.outputChannels == right.outputChannels &&
         left.activation == right.activation;
}
} // namespace

OpenYourBoxAudioProcessorEditor::OpenYourBoxAudioProcessorEditor(
    OpenYourBoxAudioProcessor &processorToUse)
    : AudioProcessorEditor(&processorToUse), audioProcessor(processorToUse),
      freezeCoordinator(
          [this](const openyourbox::graph::FreezeSelectionResult &result,
                 std::string &error) {
            return audioProcessor.prepareFrozenArtifact(result, error);
          }),
      trainCoordinator(
          [this](const openyourbox::graph::TrainJobResult &result,
                 std::string &error) {
            return audioProcessor.prepareTrainedArtifact(result, error);
          }),
      imguiHost([this] { renderFrame(); }) {
  setOpaque(true);
  setResizable(true, true);
  setResizeLimits(760, 480, 1920, 1200);
  addAndMakeVisible(imguiHost);
  setSize(1100, 680);
  trainPanel.objective = audioProcessor.getLastTrainObjective();
  audioProcessor.onPatchApplied = [this] { reloadLiveGraphFromProcessor(); };
}

OpenYourBoxAudioProcessorEditor::~OpenYourBoxAudioProcessorEditor() {
  audioProcessor.onPatchApplied = nullptr;
}

void OpenYourBoxAudioProcessorEditor::paint(juce::Graphics &graphics) {
  graphics.fillAll(juce::Colour(20, 23, 30));
}

void OpenYourBoxAudioProcessorEditor::resized() {
  imguiHost.setBounds(getLocalBounds());
}

void OpenYourBoxAudioProcessorEditor::renderFrame() {
  updateGraphIfNeeded();
  applyCompletedFreeze();
  applyCompletedTrain();
  pollTrainingPreview();
  audioProcessor.drainInputCapture();
  if (const auto captureStatus = audioProcessor.takeCaptureStatusMessage();
      captureStatus.isNotEmpty()) {
    if (captureStatus.containsIgnoreCase("could not") ||
        captureStatus.containsIgnoreCase("failed") ||
        captureStatus.containsIgnoreCase("error") ||
        captureStatus.containsIgnoreCase("discarded") ||
        captureStatus.containsIgnoreCase("disconnected"))
      showError(captureStatus);
    else
      showGraphMessage(captureStatus.toStdString());
  }
  if (audioProcessor.takeLibraryFocusRequest())
    focusLibraryTab();
  for (auto &node : nodeGraph.getNodes()) {
    if (node.state != openyourbox::graph::NodeState::frozenGold ||
        !node.metrics.has_value())
      continue;
    const auto measured =
        audioProcessor.getFrozenInferenceTimeMilliseconds(node.id);
    if (measured > 0.0)
      node.metrics->inferenceTimeMilliseconds = measured;
  }

  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
  constexpr auto flags = ImGuiWindowFlags_NoTitleBar |
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::Begin("OpenYourBox", nullptr, flags);

  ImGui::TextColored(ImVec4(0.39f, 0.70f, 1.0f, 1.0f), "OpenYourBox");
  ImGui::SameLine();
  const auto &currentPreset = audioProcessor.getCurrentPreset();
  if (currentPreset.isAssociated())
    ImGui::TextDisabled("%s%s", currentPreset.name.toRawUTF8(),
                        currentPreset.dirty ? " *" : "");
  else
    ImGui::TextDisabled("No preset");
  ImGui::SameLine();
  {
    auto &history = audioProcessor.getEditHistory();
    ImGui::BeginDisabled(!history.canUndo());
    if (ImGui::SmallButton("Undo"))
      performUndo();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!history.canRedo());
    if (ImGui::SmallButton("Redo"))
      performRedo();
    ImGui::EndDisabled();
  }
  const auto &io = ImGui::GetIO();
  const auto undoMod = io.KeySuper || io.KeyCtrl;
  if (undoMod && !io.WantTextInput) {
    if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
      if (io.KeyShift)
        performRedo();
      else
        performUndo();
    } else if (!io.KeySuper && ImGui::IsKeyPressed(ImGuiKey_Y, false)) {
      performRedo();
    }
  }
  ImGui::SameLine();
  const auto sampleRate = audioProcessor.getCurrentSampleRate();
  const auto receptiveField = audioProcessor.getReceptiveFieldSamples();
  const auto receptiveFieldMs =
      sampleRate > 0.0
          ? static_cast<double>(receptiveField) * 1000.0 / sampleRate
          : 0.0;
  ImGui::TextDisabled(
      "Embedded Builder  |  RF %.2f ms / %llu samples  |  %llu params",
      receptiveFieldMs, static_cast<unsigned long long>(receptiveField),
      static_cast<unsigned long long>(audioProcessor.getModelParameterCount()));
  pollModelError();
  const auto runtimeError = audioProcessor.getModelError();
  if (runtimeError.isNotEmpty()) {
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.12f, 0.12f, 1.0f));
    if (ImGui::SmallButton("Error"))
      showError(runtimeError, true);
    ImGui::PopStyleColor();
  }
  const auto graphWarning = juce::String(nodeGraph.graphWarningMessage());
  if (graphWarning.isNotEmpty()) {
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.38f, 0.08f, 1.0f));
    if (ImGui::SmallButton("Warning"))
      showWarning(graphWarning);
    ImGui::PopStyleColor();
  }
  if (graphMessage.isNotEmpty() && ImGui::GetTime() < graphMessageDeadline) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f), "%s",
                       graphMessage.toRawUTF8());
  }
  const auto freezeStatus = freezeCoordinator.getStatus();
  if (freezeStatus == openyourbox::freeze::FreezeStatus::compiling) {
    ImGui::SameLine();
    const auto status = freezeCoordinator.getStatusMessage();
    const auto progress =
        static_cast<float>(std::fmod(ImGui::GetTime() * 0.35, 1.0));
    ImGui::ProgressBar(progress, ImVec2(110.0f, 0.0f), "##freeze");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f), "%s",
                       status.toRawUTF8());
  }
  ImGui::Separator();

  constexpr float splitterWidth = 6.0f;
  constexpr float minRight = 180.0f;
  constexpr float maxRight = 520.0f;
  constexpr float minCanvas = 280.0f;
  const auto avail = ImGui::GetContentRegionAvail();
  auto right = std::clamp(rightPanelWidth, minRight, maxRight);
  right = std::min(right, std::max(minRight, avail.x - minCanvas - splitterWidth));
  rightPanelWidth = right;

  ImGui::BeginChild("GraphArea", ImVec2(-(right + splitterWidth), 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar);
  openyourbox::graph::NodeRendererCallbacks callbacks;
  callbacks.propertyChanged = [this](std::int32_t nodeId,
                                     const std::string &key, int value) {
    handlePropertyChanged(nodeId, key, value);
  };
  callbacks.floatPropertyChanged = [this](std::int32_t nodeId,
                                          const std::string &key, float value) {
    handleFloatPropertyChanged(nodeId, key, value);
  };
  callbacks.randomizeNode = [this](std::int32_t nodeId, std::int32_t seed) {
    handleRandomize(nodeId, seed);
  };
  callbacks.freezeSelection =
      [this](const std::vector<std::int32_t> &selection) {
        handleFreeze(selection);
      };
  callbacks.unfreezeNode = [this](std::int32_t nodeId) {
    handleUnfreeze(nodeId);
  };
  callbacks.showMessage = [this](const std::string &message) {
    showGraphMessage(message);
  };
  callbacks.documentChanged = [this](bool recompile, bool refreshAnalysis) {
    auto &history = audioProcessor.getEditHistory();
    if (history.isGestureOpen())
      persistGraph(false, false);
    else
      persistGraph(recompile, true);
    if (refreshAnalysis)
      invalidateAnalysis();
    else
      syncAnalysisRevision();
  };
  callbacks.analysisRequested = [this](std::int32_t nodeId) {
    handleAnalysisRequested(nodeId);
  };
  callbacks.knobChanged = [this](std::int32_t nodeId, float value) {
    handleKnobChanged(nodeId, value);
  };
  callbacks.xyChanged = [this](std::int32_t nodeId, float x, float y) {
    handleXyChanged(nodeId, x, y);
  };
  callbacks.armChanged = [this](std::int32_t nodeId, bool armed) {
    nodeGraph.setArmedForTraining(nodeId, armed);
    persistGraph(false);
  };
  callbacks.browseWeights = [this](std::int32_t nodeId) {
    handleBrowseWeights(nodeId);
  };
  callbacks.beginPatchGesture = [this](const char *label) {
    beginHistoryGesture(label);
  };
  callbacks.endPatchGesture = [this] { endHistoryGesture(); };
  nodeRenderer.render(nodeGraph, callbacks, imguiHost.takeMagnification(),
                      &boxLibrary);
  ImGui::EndChild();

  ImGui::SameLine(0.0f, 0.0f);
  ImGui::InvisibleButton("RightPanelSplitter",
                         ImVec2(splitterWidth, avail.y > 0.0f ? avail.y : 1.0f));
  if (ImGui::IsItemActive()) {
    rightPanelWidth = std::clamp(
        rightPanelWidth - ImGui::GetIO().MouseDelta.x, minRight,
        std::min(maxRight, avail.x - minCanvas - splitterWidth));
  }
  if (ImGui::IsItemHovered() || ImGui::IsItemActive())
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

  ImGui::SameLine(0.0f, 0.0f);
  ImGui::BeginChild("InfoArea", ImVec2(right, 0.0f), true);
  auto &pairing = audioProcessor.getCapturePairing();
  auto &library = audioProcessor.getTrainingLibrary();
  if (ImGui::BeginTabBar("SideTabs")) {
    const auto tabFlags = [this](int index) {
      return pendingSidePanelTab == index ? ImGuiTabItemFlags_SetSelected
                                          : ImGuiTabItemFlags_None;
    };
    if (ImGui::BeginTabItem("Info", nullptr, tabFlags(0))) {
      sidePanelTab = 0;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Library", nullptr, tabFlags(1))) {
      sidePanelTab = 1;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Capture", nullptr, tabFlags(2))) {
      sidePanelTab = 2;
      ImGui::EndTabItem();
    }
    if (pairing.getPairingRole() != openyourbox::capture::PairingRole::slave &&
        ImGui::BeginTabItem("Train", nullptr, tabFlags(3))) {
      sidePanelTab = 3;
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Presets", nullptr, tabFlags(4))) {
      sidePanelTab = 4;
      ImGui::EndTabItem();
    }
    pendingSidePanelTab = -1;
    ImGui::EndTabBar();
  }
  if (sidePanelTab == 0) {
  if (auto *dryWetParameter = audioProcessor.getParameterState().getParameter(
          openyourbox::params::dryWet)) {
    auto mixPercent = dryWetParameter->getValue() * 100.0f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Dry/Wet");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1.0f);
    const auto changed =
        ImGui::SliderFloat("##dryWet", &mixPercent, 0.0f, 100.0f, "%.0f%%");
    if (ImGui::IsItemActive()) {
      if (!dryWetGestureActive) {
        beginHistoryGesture("Dry/Wet");
        dryWetParameter->beginChangeGesture();
        dryWetGestureActive = true;
      }
      if (changed)
        dryWetParameter->setValueNotifyingHost(
            std::clamp(mixPercent / 100.0f, 0.0f, 1.0f));
    }
    if (dryWetGestureActive && ImGui::IsItemDeactivated()) {
      dryWetParameter->endChangeGesture();
      dryWetGestureActive = false;
      endHistoryGesture();
    }
  }
  ImGui::Separator();
  refreshAnalysisIfNeeded();
  openyourbox::ui::AnalysisPanelState analysisState;
  analysisState.nodeId = analysisNodeId;
  if (const auto *node = nodeGraph.findNode(analysisNodeId))
    analysisState.view = node->selectedAnalysisView;
  analysisState.snapshot = analysisSnapshot;
  analysisState.snapshot.isStale =
      analysisRevision != audioProcessor.getGraphRevision();
  analysisState.playbackActive = audioProcessor.isTransportPlaying();
  infoPanel.render(
      receptiveField, sampleRate, audioProcessor.getModelParameterCount(),
      runtimeError, graphWarning, analysisState,
      [this](openyourbox::graph::AnalysisView view) {
        if (analysisNodeId != 0)
          nodeGraph.setSelectedAnalysisView(analysisNodeId, view);
        persistGraph(false, false);
        invalidateAnalysis();
      },
      [this, runtimeError] { showError(runtimeError, true); },
      [this, graphWarning] { showWarning(graphWarning); });
  } else if (sidePanelTab == 1) {
    openyourbox::ui::TrainingLibraryPanel::Callbacks libraryCallbacks;
    libraryCallbacks.importPair = [this] { handleLibraryImport(); };
    libraryCallbacks.importClip = [this] { handleLibraryImportClip(); };
    libraryCallbacks.rename = [&library](const juce::String &id,
                                         const juce::String &name) {
      library.rename(id, name);
    };
    libraryCallbacks.removeEntry = [&library](const juce::String &id) {
      library.removeEntry(id);
    };
    libraryCallbacks.preview = [this, &library](const juce::String &id,
                                                bool playX) {
      if (const auto *entry = library.findEntry(id))
        audioProcessor.startPreviewFile(
            juce::File(playX ? entry->xPath : entry->yPath));
    };
    libraryCallbacks.stopPreview = [this] { audioProcessor.stopPreview(); };
    libraryPanel.objective = trainPanel.objective;
    libraryPanel.render(library, libraryCallbacks,
                        audioProcessor.isPreviewPlaying());
  } else if (sidePanelTab == 2) {
    openyourbox::ui::CaptureSamplesPanel::Callbacks captureCallbacks;
    captureCallbacks.beginDiscovery = [&pairing] { pairing.beginDiscovery(); };
    captureCallbacks.stopDiscovery = [&pairing] { pairing.stopDiscovery(); };
    captureCallbacks.pairWith =
        [&pairing](const openyourbox::capture::DiscoveredInstance &peer) {
          pairing.pairWith(peer);
        };
    captureCallbacks.unpair = [this, &pairing] {
      pairing.unpair();
      audioProcessor.setCaptureBypass(false);
    };
    captureCallbacks.setRole = [&pairing](openyourbox::capture::CaptureRole role) {
      pairing.setCaptureRole(role);
    };
    captureCallbacks.setBypass = [this, &pairing](bool enabled) {
      pairing.setCaptureBypass(enabled);
      audioProcessor.setCaptureBypass(enabled);
    };
    captureCallbacks.getStartTransportOnRecord = [this] {
      return audioProcessor.getStartTransportOnRecord();
    };
    captureCallbacks.setStartTransportOnRecord = [this](bool enabled) {
      audioProcessor.setStartTransportOnRecord(enabled);
    };
    captureCallbacks.startRecording = [this] {
      audioProcessor.startPairedRecording();
    };
    captureCallbacks.startSingleRecording = [this] {
      audioProcessor.startSingleRecording();
    };
    captureCallbacks.stopRecording = [this] {
      audioProcessor.stopPairedRecording();
    };
    captureCallbacks.openLibrary = [this] { focusLibraryTab(); };
    capturePanel.render(pairing, pairing.listPeers(), captureCallbacks);
  } else if (sidePanelTab == 3) {
    juce::String mixedReason;
    openyourbox::ui::TrainPanel::Gates gates;
    gates.copyrightAcknowledged = copyrightAcknowledgment.isAcknowledged();
    gates.selectedPairCount = library.getSelectedCount();
    gates.selectedDurationSeconds = library.getSelectedDurationSeconds();
    gates.armedElementCount =
        static_cast<int>(nodeGraph.getArmedTrainableNodeIds().size());
    gates.mixedSampleRates =
        !library.selectedSampleRatesMatch(mixedReason) &&
        gates.selectedPairCount > 0;
    gates.blockReason = mixedReason;
    gates.receptiveFieldMilliseconds = receptiveFieldMs;
    gates.trainWindowSeconds =
        sampleRate > 0.0
            ? static_cast<double>(trainPanel.hyperparameters.segmentLength) /
                  sampleRate
            : 0.0;
    gates.isMaster = pairing.getPairingRole() !=
                     openyourbox::capture::PairingRole::slave;
    gates.retryAvailable = retryTrainResult.has_value();
    gates.unpairedSelected =
        library.selectedContainsUnpaired();
    gates.reconstructionPathInvalid =
        trainPanel.objective ==
            openyourbox::graph::TrainObjective::reconstruction &&
        !nodeGraph.hasReconstructionTrainPath();
    gates.reconstructionReason =
        juce::String(nodeGraph.reconstructionGateMessage());
    openyourbox::ui::TrainPanel::Callbacks trainCallbacks;
    trainCallbacks.run = [this] { handleTrainRun(); };
    trainCallbacks.pause = [this] { trainCoordinator.pause(); };
    trainCallbacks.resume = [this] { trainCoordinator.resume(); };
    trainCallbacks.stop = [this] { trainCoordinator.stop(); };
    trainCallbacks.retryLoad = [this] {
      if (retryTrainResult.has_value())
        trainCoordinator.retryLoad(*retryTrainResult);
    };
    trainCallbacks.openLibrary = [this] { focusLibraryTab(); };
    trainCallbacks.requestCopyright = [this] { copyrightModalVisible = true; };
    trainCallbacks.hearWhileTrainingChanged = [this](bool enabled) {
      if (!enabled) {
        lastTrainPreviewPath.clear();
        audioProcessor.clearTrainingPreview();
      }
    };
    trainCallbacks.viewError = [this](const juce::String &detail) {
      showError(detail, true);
    };
    trainPanel.render(trainCoordinator, gates, trainCallbacks);
  } else if (sidePanelTab == 4) {
    openyourbox::ui::UserPresetPanel::Callbacks presetCallbacks;
    presetCallbacks.save = [this] { handlePresetSave(); };
    presetCallbacks.saveAs = [this](const juce::String &name, bool overwrite) {
      handlePresetSaveAs(name, overwrite);
    };
    presetCallbacks.load = [this](const juce::String &id) {
      handlePresetLoad(id);
    };
    presetCallbacks.rename = [this](const juce::String &id,
                                    const juce::String &name) {
      handlePresetRename(id, name);
    };
    presetCallbacks.remove = [this](const juce::String &id) {
      handlePresetDelete(id);
    };
    presetCallbacks.showMessage = [this](const std::string &message) {
      showGraphMessage(message);
    };
    presetPanel.render(presetLibrary, audioProcessor.getCurrentPreset(),
                       presetCallbacks);
  }
  copyrightModal.render(copyrightAcknowledgment, copyrightModalVisible,
                        [this] { copyrightModalVisible = false; });
  ImGui::EndChild();
  errorModal.render();
  warningModal.render();

  ImGui::End();
}

void OpenYourBoxAudioProcessorEditor::updateGraphIfNeeded() {
  const auto requested = audioProcessor.getRequestedConfiguration();
  if (!graphInitialized) {
    const auto restored = audioProcessor.getGraphState();
    if (restored.getNumChildren() > 0)
      nodeGraph.restoreFromValueTree(restored);
    else
      nodeGraph.rebuildFromModel(requested);
    displayedConfiguration = requested;
    graphInitialized = true;
    persistGraph(true, false);
    captureHistoryBaseline();
    return;
  }

  if (!sameArchitecture(requested, displayedConfiguration)) {
    displayedConfiguration = requested;
    const auto tcn =
        std::find_if(nodeGraph.getNodes().begin(), nodeGraph.getNodes().end(),
                     [](const openyourbox::graph::GraphNode &node) {
                       return node.type == openyourbox::graph::NodeType::tcn;
                     });
    if (tcn != nodeGraph.getNodes().end()) {
      nodeGraph.setProperty(tcn->id, "depth", requested.depth);
      nodeGraph.setProperty(tcn->id, "kernel_size", requested.kernelSize);
      nodeGraph.setProperty(tcn->id, "channels", requested.channels);
      nodeGraph.setProperty(tcn->id, "activation",
                            static_cast<int>(requested.activation));
      persistGraph(true, false);
    }
  }
}

void OpenYourBoxAudioProcessorEditor::persistGraph(bool compileRuntime,
                                                   bool recordHistory) {
  audioProcessor.setGraphState(nodeGraph.toValueTree(), compileRuntime);
  publishRuntimeControls();
  if (!recordHistory)
    return;
  auto &history = audioProcessor.getEditHistory();
  if (history.isSuppressed() || history.isGestureOpen())
    return;
  commitHistoryFromBaseline("Graph edit");
}

void OpenYourBoxAudioProcessorEditor::publishRuntimeControls() {
  audioProcessor.setRuntimeControls(
      openyourbox::dsp::collectRuntimeControlState(nodeGraph));
}

void OpenYourBoxAudioProcessorEditor::invalidateAnalysis() {
  analysisRevision = 0;
}

void OpenYourBoxAudioProcessorEditor::syncAnalysisRevision() {
  analysisRevision = audioProcessor.getGraphRevision();
}

void OpenYourBoxAudioProcessorEditor::refreshAnalysisIfNeeded() {
  if (analysisNodeId == 0)
    return;
  const auto revision = audioProcessor.getGraphRevision();
  const auto now = ImGui::GetTime();
  constexpr double minInterval = 1.0 / 12.0;
  if (analysisRevision == revision &&
      analysisSnapshot.nodeId == analysisNodeId &&
      now - lastAnalysisTime < 0.25)
    return;
  if (now - lastAnalysisTime < minInterval)
    return;

  openyourbox::dsp::AnalysisRequest request;
  request.nodeId = analysisNodeId;
  if (const auto *node = nodeGraph.findNode(analysisNodeId))
    request.view = node->selectedAnalysisView;
  request.revision = revision;
  request.playbackActive = audioProcessor.isTransportPlaying();
  request.sampleRate = audioProcessor.getCurrentSampleRate();
  std::array<std::array<float, 512>, 2> live{};
  float *planes[2] = {live[0].data(), live[1].data()};
  int channels = 0;
  int samples = 0;
  audioProcessor.copyLiveCapture(request.liveInputPeak, request.liveOutputPeak,
                                 request.liveInputSuitable, planes, 512,
                                 channels, samples);
  std::array<float, 1024> packed{};
  for (int channel = 0; channel < channels; ++channel)
    std::memcpy(packed.data() + channel * samples, live[static_cast<std::size_t>(channel)].data(),
                static_cast<std::size_t>(samples) * sizeof(float));
  request.liveInputChannels = channels;
  request.liveInputSamples = samples;
  request.liveInput = samples > 0 ? packed.data() : nullptr;
  float tapIn = 0.0f;
  float tapOut = 0.0f;
  if (audioProcessor.getAnalysisTapPeaks(analysisNodeId, tapIn, tapOut)) {
    request.liveInputPeak = tapIn;
    request.liveOutputPeak = tapOut;
  }

  openyourbox::dsp::LiveGraphCompileOptions options;
  options.hostInputChannels =
      std::max(1, audioProcessor.getTotalNumInputChannels());
  options.hostOutputChannels =
      std::max(1, audioProcessor.getTotalNumOutputChannels());
  options.maximumBlockSize = 512;
  options.sampleRate = std::max(1.0, audioProcessor.getCurrentSampleRate());
  analysisSnapshot = openyourbox::dsp::LiveGraphEngine::analyse(
      nodeGraph, request, options,
      [this](const openyourbox::graph::GraphNode &node) {
        return audioProcessor.resolveFrozenBlackBoxForAnalysis(node);
      });
  analysisSnapshot.isStale = false;
  analysisRevision = revision;
  lastAnalysisTime = now;
}

void OpenYourBoxAudioProcessorEditor::handleFloatPropertyChanged(
    std::int32_t nodeId, const std::string &key, float value) {
  juce::ignoreUnused(nodeId, key, value);
  persistGraph(false);
  invalidateAnalysis();
}

void OpenYourBoxAudioProcessorEditor::handleAnalysisRequested(
    std::int32_t nodeId) {
  analysisNodeId = nodeId;
  invalidateAnalysis();
}

void OpenYourBoxAudioProcessorEditor::handleKnobChanged(std::int32_t nodeId,
                                                       float value) {
  juce::ignoreUnused(nodeId, value);
  persistGraph(false);
  syncAnalysisRevision();
}

void OpenYourBoxAudioProcessorEditor::handleXyChanged(std::int32_t nodeId,
                                                     float x, float y) {
  juce::ignoreUnused(nodeId, x, y);
  persistGraph(false);
  syncAnalysisRevision();
}

void OpenYourBoxAudioProcessorEditor::handlePropertyChanged(
    std::int32_t nodeId, const std::string &key, int value) {
  juce::ignoreUnused(key, value);
  const auto *node = nodeGraph.findNode(nodeId);
  if (node == nullptr || node->type != openyourbox::graph::NodeType::tcn) {
    persistGraph();
    return;
  }

  auto configuration = audioProcessor.getRequestedConfiguration();
  for (const auto &property : node->properties) {
    if (property.key == "depth")
      configuration.depth = property.value;
    else if (property.key == "kernel_size")
      configuration.kernelSize = property.value;
    else if (property.key == "channels")
      configuration.channels = property.value;
    else if (property.key == "activation")
      configuration.activation =
          static_cast<openyourbox::dsp::ActivationType>(property.value);
  }
  displayedConfiguration = configuration;
  audioProcessor.applyGraphConfiguration(configuration);
  persistGraph();
}

void OpenYourBoxAudioProcessorEditor::handleRandomize(std::int32_t nodeId,
                                                     std::int32_t seed) {
  nodeGraph.clearWeightsToSeed(nodeId, seed);
  audioProcessor.randomizeGraphElement(nodeId, seed);
  persistGraph(true, false);
  commitHistoryFromBaseline("Randomize");
}

void OpenYourBoxAudioProcessorEditor::handleFreeze(
    const std::vector<std::int32_t> &selectedNodeIds) {
  const auto expanded =
      nodeGraph.expandSelectionToFreezableLeaves(selectedNodeIds);
  if (expanded.empty()) {
    showError("Freeze requires live Blue elements inside the selection");
    return;
  }
  auto chains = nodeGraph.partitionFreezeChains(expanded);
  if (chains.empty()) {
    showError(
        "Freeze requires live Blue chains with a single input and output");
    return;
  }

  pendingFreezeChains = std::move(chains);
  if (!startNextFreezeChain()) {
    pendingFreezeChains.clear();
    pendingFreezeSelection.clear();
    pendingFreezeGraph.clear();
  }
}

bool OpenYourBoxAudioProcessorEditor::startNextFreezeChain() {
  if (pendingFreezeChains.empty())
    return false;
  auto request = nodeGraph.createFreezeRequest(pendingFreezeChains.front());
  if (!request.has_value()) {
    showError(
        "Freeze requires a valid connected selection of live Blue nodes");
    return false;
  }

  auto payload = juce::JSON::parse(juce::String(request->graphFragment));
  auto *root = payload.getDynamicObject();
  auto *options = root != nullptr
                      ? root->getProperty("compile_options").getDynamicObject()
                      : nullptr;
  if (options == nullptr) {
    showError("Freeze request could not be configured for host audio");
    return false;
  }
  if (static_cast<int>(options->getProperty("host_input_channels")) < 1)
    options->setProperty("host_input_channels",
                         audioProcessor.getTotalNumInputChannels());
  if (static_cast<int>(options->getProperty("host_output_channels")) < 1)
    options->setProperty("host_output_channels",
                         audioProcessor.getTotalNumOutputChannels());
  options->setProperty("example_samples", 256);
  request->graphFragment = juce::JSON::toString(payload, true).toStdString();

  if (!freezeCoordinator.start(*request)) {
    showGraphMessage("A freeze compilation is already running");
    return false;
  }
  pendingFreezeSelection = pendingFreezeChains.front();
  pendingFreezeGraph = nodeGraph.toJson();
  return true;
}

void OpenYourBoxAudioProcessorEditor::handleUnfreeze(std::int32_t nodeId) {
  const auto *blackBox = nodeGraph.findNode(nodeId);
  const auto artifactPath =
      blackBox != nullptr ? blackBox->artifactPath : std::string{};
  const auto keepTrained =
      blackBox != nullptr &&
      (blackBox->blackBoxOrigin ==
           openyourbox::graph::BlackBoxOrigin::trainAutoload ||
       blackBox->weightsProvenance ==
           openyourbox::graph::WeightsProvenance::file);
  if (nodeGraph.unfreeze(nodeId)) {
    if (!keepTrained)
      audioProcessor.releaseFrozenArtifact(artifactPath);
    showGraphMessage("Selection restored to Live Blue");
    persistGraph(true, false);
    commitHistoryFromBaseline("Unfreeze");
  }
}

void OpenYourBoxAudioProcessorEditor::applyCompletedFreeze() {
  auto result = freezeCoordinator.takeResult();
  if (!result.has_value())
    return;

  if (!result->succeeded) {
    showError("Freeze failed: " + juce::String(result->errorMessage));
    pendingFreezeSelection.clear();
    pendingFreezeChains.clear();
    pendingFreezeGraph.clear();
    return;
  }

  if (pendingFreezeGraph != nodeGraph.toJson()) {
    audioProcessor.releaseFrozenArtifact(result->artifactPath);
    pendingFreezeSelection.clear();
    pendingFreezeChains.clear();
    pendingFreezeGraph.clear();
    showError("Freeze result was discarded because the graph changed "
              "while compiling");
    return;
  }

  const auto frozen =
      nodeGraph.freezeSelection(pendingFreezeSelection, *result);
  if (!pendingFreezeChains.empty())
    pendingFreezeChains.erase(pendingFreezeChains.begin());
  pendingFreezeSelection.clear();
  pendingFreezeGraph.clear();
  if (!frozen.has_value()) {
    audioProcessor.releaseFrozenArtifact(result->artifactPath);
    pendingFreezeChains.clear();
    showError(
        "Freeze result was discarded because the graph selection changed");
    return;
  }

  persistGraph(true, false);
  commitHistoryFromBaseline("Freeze");
  if (!pendingFreezeChains.empty() && startNextFreezeChain()) {
    showGraphMessage("Freeze succeeded: compiling the next frozen chain");
    return;
  }
  pendingFreezeChains.clear();
  showGraphMessage("Freeze succeeded: selection is frozen Gold");
}

void OpenYourBoxAudioProcessorEditor::showGraphMessage(
    const std::string &message) {
  graphMessage = message;
  graphMessageDeadline = ImGui::GetTime() + 3.0;
}

void OpenYourBoxAudioProcessorEditor::showError(const juce::String &message,
                                                bool forceReopen) {
  if (message.isEmpty())
    return;
  if (lastPresentedError.contains(message)) {
    if (forceReopen)
      errorModal.show(lastPresentedError);
    return;
  }
  if (errorModal.isVisible() && lastPresentedError.isNotEmpty())
    lastPresentedError << "\n\n" << message;
  else
    lastPresentedError = message;
  errorModal.show(lastPresentedError);
}

void OpenYourBoxAudioProcessorEditor::showWarning(const juce::String &message) {
  if (message.isEmpty()) {
    warningModal.dismiss();
    return;
  }
  warningModal.showWarning(message);
}

void OpenYourBoxAudioProcessorEditor::pollModelError() {
  const auto runtimeError = audioProcessor.getModelError();
  if (runtimeError.isEmpty())
    return;
  showError(runtimeError);
}

void OpenYourBoxAudioProcessorEditor::handleTrainRun() {
  juce::String mixed;
  if (!copyrightAcknowledgment.isAcknowledged()) {
    copyrightModalVisible = true;
    return;
  }
  if (!audioProcessor.getTrainingLibrary().selectedSampleRatesMatch(mixed)) {
    showError(mixed);
    return;
  }
  if (trainPanel.objective == openyourbox::graph::TrainObjective::mapping &&
      audioProcessor.getTrainingLibrary().selectedContainsUnpaired()) {
    showError("Mapping cannot train unpaired clips. Deselect them or switch to reconstruction.");
    return;
  }
  if (trainPanel.objective ==
          openyourbox::graph::TrainObjective::reconstruction &&
      !nodeGraph.hasReconstructionTrainPath()) {
    showError(juce::String(nodeGraph.reconstructionGateMessage()));
    return;
  }
  audioProcessor.setLastTrainObjective(trainPanel.objective);
  auto request = nodeGraph.createTrainRequest();
  if (!request.has_value()) {
    showError("Arm at least one trainable element before Train");
    return;
  }
  auto payload = juce::JSON::parse(juce::String(request->graphFragment));
  auto *root = payload.getDynamicObject();
  if (root == nullptr)
    return;
  juce::Array<juce::var> pairs;
  juce::Array<juce::var> clips;
  for (const auto &entry : audioProcessor.getTrainingLibrary().getEntries()) {
    if (!entry.selectedForTrain)
      continue;
    if (entry.kind == openyourbox::library::LibraryEntryKind::clip) {
      auto clip = std::make_unique<juce::DynamicObject>();
      clip->setProperty("clip_id", entry.id);
      clip->setProperty("path", entry.xPath);
      clip->setProperty("kind", "clip");
      clips.add(juce::var(clip.release()));
      continue;
    }
    auto pair = std::make_unique<juce::DynamicObject>();
    pair->setProperty("pair_id", entry.id);
    pair->setProperty("x_path", entry.xPath);
    pair->setProperty("y_path", entry.yPath);
    pair->setProperty("kind", "pair");
    pair->setProperty("source", entry.source ==
                                        openyourbox::library::PairSource::capture
                                    ? "capture"
                                    : "import");
    pairs.add(juce::var(pair.release()));
  }
  auto captureSet = std::make_unique<juce::DynamicObject>();
  captureSet->setProperty("pairs", pairs);
  captureSet->setProperty("clips", clips);
  root->setProperty("capture_set", juce::var(captureSet.release()));
  auto options = std::make_unique<juce::DynamicObject>();
  options->setProperty("optimizer", "adam");
  options->setProperty(
      "objective",
      juce::String(openyourbox::graph::trainObjectiveName(trainPanel.objective)));
  auto reconstruction = std::make_unique<juce::DynamicObject>();
  reconstruction->setProperty("stage1_steps",
                              openyourbox::graph::defaultReconstructionStage1Steps);
  reconstruction->setProperty("stage2_steps",
                              openyourbox::graph::defaultReconstructionStage2Steps);
  options->setProperty("reconstruction", juce::var(reconstruction.release()));
  auto loss = std::make_unique<juce::DynamicObject>();
  loss->setProperty("type", "multiresolution_stft");
  loss->setProperty("fft_sizes", juce::Array<juce::var>{32, 128, 512, 2048});
  loss->setProperty("win_lengths", juce::Array<juce::var>{32, 128, 512, 2048});
  loss->setProperty("hop_sizes", juce::Array<juce::var>{16, 64, 256, 1024});
  options->setProperty("loss", juce::var(loss.release()));
  options->setProperty("steer_conditioning", 0.0);
  options->setProperty("total_steps", trainPanel.hyperparameters.totalSteps);
  options->setProperty("learning_rate",
                       static_cast<double>(trainPanel.hyperparameters.learningRate));
  options->setProperty("segment_length",
                       trainPanel.hyperparameters.segmentLength);
  options->setProperty("checkpoint_interval",
                       trainPanel.hyperparameters.checkpointInterval);
  options->setProperty("export_checkpoints", trainPanel.hearWhileTraining);
  options->setProperty("rf_aware_crop", true);
  options->setProperty("host_input_channels",
                       std::max(1, audioProcessor.getTotalNumInputChannels()));
  int condDim = 2;
  for (const auto nodeId : nodeGraph.getArmedTrainableNodeIds()) {
    const auto *node = nodeGraph.findNode(nodeId);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs) {
      if (!openyourbox::graph::isControlInputPin(pin))
        continue;
      for (const auto &link : nodeGraph.getLinks()) {
        if (link.destinationPinId != pin.id)
          continue;
        const auto *sourcePin = nodeGraph.findPin(link.sourcePinId);
        if (sourcePin != nullptr && sourcePin->shape.channels > 0)
          condDim = sourcePin->shape.channels;
      }
    }
  }
  options->setProperty("cond_dim", condDim);
  auto mlflow = std::make_unique<juce::DynamicObject>();
  mlflow->setProperty("enabled", trainPanel.logToMlflow);
  mlflow->setProperty("tracking_uri", juce::String(trainPanel.mlflowTrackingUri));
  mlflow->setProperty("experiment", juce::String(trainPanel.mlflowExperiment));
  mlflow->setProperty("name", juce::String(trainPanel.mlflowRunName));
  mlflow->setProperty("tags", juce::Array<juce::var>{"train", "steerable"});
  options->setProperty("mlflow", juce::var(mlflow.release()));
  root->setProperty("train_options", juce::var(options.release()));
  request->graphFragment = juce::JSON::toString(payload, true).toStdString();
  if (!trainCoordinator.start(*request)) {
    showGraphMessage("A training job is already running");
    return;
  }
  trainPreviewNodeIds = request->armedNodeIds;
  lastTrainPreviewPath.clear();
}

void OpenYourBoxAudioProcessorEditor::pollTrainingPreview() {
  const auto status = trainCoordinator.getStatus();
  const auto busy = status == openyourbox::train::TrainStatus::running ||
                    status == openyourbox::train::TrainStatus::paused;
  if (!trainPanel.hearWhileTraining || !busy) {
    trainPreviewLoadInFlight = false;
    return;
  }
  const auto progress = trainCoordinator.getProgress();
  if (progress.artifactPath.empty() ||
      progress.artifactPath == lastTrainPreviewPath || trainPreviewLoadInFlight)
    return;
  lastTrainPreviewPath = progress.artifactPath;
  trainPreviewLoadInFlight = true;
  auto *processor = &audioProcessor;
  const auto result = progress;
  const auto nodeIds = trainPreviewNodeIds.empty()
                           ? nodeGraph.getArmedTrainableNodeIds()
                           : trainPreviewNodeIds;
  juce::Component::SafePointer<OpenYourBoxAudioProcessorEditor> safeThis(this);
  juce::Thread::launch([processor, result, nodeIds, safeThis] {
    std::string error;
    const auto ok = processor->prepareTrainedArtifact(result, error);
    juce::MessageManager::callAsync([safeThis, ok, result, nodeIds, error] {
      if (safeThis == nullptr)
        return;
      safeThis->trainPreviewLoadInFlight = false;
      if (!ok) {
        if (!error.empty())
          safeThis->showError("Training checkpoint: " + juce::String(error));
        return;
      }
      if (!safeThis->trainPanel.hearWhileTraining)
        return;
      safeThis->audioProcessor.setTrainingPreview(result.artifactPath, nodeIds);
    });
  });
}

void OpenYourBoxAudioProcessorEditor::applyCompletedTrain() {
  auto result = trainCoordinator.takeResult();
  if (!result.has_value())
    return;
  audioProcessor.clearTrainingPreview();
  lastTrainPreviewPath.clear();
  if (result->status == "stopped") {
    showGraphMessage("Training stopped; prior model unchanged");
    persistGraph(true, false);
    return;
  }
  if (result->status != "success") {
    if (!result->artifactPath.empty())
      retryTrainResult = result;
    showError("Training failed: " + juce::String(result->errorMessage));
    persistGraph(true, false);
    return;
  }
  retryTrainResult.reset();
  const auto absorbed = nodeGraph.absorbArmedChain(*result);
  if (!absorbed.has_value()) {
    retryTrainResult = result;
    showError("Trained artifact is ready but the graph could not swap");
    persistGraph(true, false);
    return;
  }
    persistGraph(true, false);
    commitHistoryFromBaseline("Train");
    showGraphMessage("Training succeeded: armed chain is Gold");
}

void OpenYourBoxAudioProcessorEditor::focusLibraryTab() {
  pendingSidePanelTab = 1;
}

void OpenYourBoxAudioProcessorEditor::handleLibraryImport() {
  juce::MessageManager::callAsync([this] { promptLibraryImportClean(); });
}

void OpenYourBoxAudioProcessorEditor::handleLibraryImportClip() {
  fileChooser = std::make_shared<juce::FileChooser>(
      "Import unpaired clip", openyourbox::library::samplesDirectory(),
      "*.wav;*.aiff;*.flac");
  fileChooser->launchAsync(juce::FileBrowserComponent::openMode |
                               juce::FileBrowserComponent::canSelectFiles,
                           [this](const juce::FileChooser &chosen) {
                             const auto file = chosen.getResult();
                             fileChooser.reset();
                             if (!file.existsAsFile())
                               return;
                             juce::String error;
                             if (!audioProcessor.getTrainingLibrary().importClip(
                                     file, error))
                               showError(error);
                           });
}

void OpenYourBoxAudioProcessorEditor::promptLibraryImportClean() {
  fileChooser = std::make_shared<juce::FileChooser>(
      "Import clean (x)", openyourbox::library::samplesDirectory(),
      "*.wav;*.aiff;*.flac");
  fileChooser->launchAsync(juce::FileBrowserComponent::openMode |
                               juce::FileBrowserComponent::canSelectFiles,
                           [this](const juce::FileChooser &chosen) {
                             pendingImportClean = chosen.getResult();
                             fileChooser.reset();
                             if (!pendingImportClean.existsAsFile())
                               return;
                             juce::MessageManager::callAsync(
                                 [this] { promptLibraryImportProcessed(); });
                           });
}

void OpenYourBoxAudioProcessorEditor::promptLibraryImportProcessed() {
  fileChooser = std::make_shared<juce::FileChooser>(
      "Import processed (y)", openyourbox::library::samplesDirectory(),
      "*.wav;*.aiff;*.flac");
  fileChooser->launchAsync(
      juce::FileBrowserComponent::openMode |
          juce::FileBrowserComponent::canSelectFiles,
      [this](const juce::FileChooser &chosen) {
        const auto processed = chosen.getResult();
        const auto clean = pendingImportClean;
        fileChooser.reset();
        pendingImportClean = juce::File();
        if (!processed.existsAsFile())
          return;
        importLibraryPair(clean, processed);
      });
}

void OpenYourBoxAudioProcessorEditor::importLibraryPair(
    const juce::File &clean, const juce::File &processed) {
  auto *processor = &audioProcessor;
  juce::Component::SafePointer<OpenYourBoxAudioProcessorEditor> safeThis(this);
  juce::Thread::launch([processor, safeThis, clean, processed] {
    juce::String error;
    const auto imported =
        processor->getTrainingLibrary().importPair(clean, processed, error);
    juce::MessageManager::callAsync([safeThis, imported, error] {
      if (safeThis == nullptr)
        return;
      if (!imported.has_value())
        safeThis->showError(error);
      else
        safeThis->showGraphMessage("Imported pair into Training Library");
    });
  });
}

void OpenYourBoxAudioProcessorEditor::handleBrowseWeights(std::int32_t nodeId) {
  juce::MessageManager::callAsync([this, nodeId] {
    const auto *node = nodeGraph.findNode(nodeId);
    if (node == nullptr)
      return;
    fileChooser = std::make_shared<juce::FileChooser>(
        "Load weights", openyourbox::library::weightsDirectory(),
        "*.pt;*.pth");
    fileChooser->launchAsync(
        juce::FileBrowserComponent::openMode |
            juce::FileBrowserComponent::canSelectFiles,
        [this, nodeId](const juce::FileChooser &chosen) {
          const auto file = chosen.getResult();
          fileChooser.reset();
          if (!file.existsAsFile())
            return;
          const auto *target = nodeGraph.findNode(nodeId);
          if (target == nullptr)
            return;
          openyourbox::dsp::WeightLoader::loadAsync(
              *target, file,
              [this](std::int32_t id, const std::string &path) {
                nodeGraph.setWeightsPath(id, path);
                persistGraph(true, false);
                commitHistoryFromBaseline("Load weights");
                showGraphMessage("Loaded weights from file");
              },
              [this](std::int32_t, const std::string &error) {
                showError("Incompatible weights: " + juce::String(error));
              });
        });
  });
}

openyourbox::state::PatchSnapshot
OpenYourBoxAudioProcessorEditor::captureLiveSnapshot() {
  persistGraph(false, false);
  return audioProcessor.capturePatchSnapshot();
}

void OpenYourBoxAudioProcessorEditor::captureHistoryBaseline() {
  lastCommittedSnapshot = captureLiveSnapshot();
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
}

void OpenYourBoxAudioProcessorEditor::commitHistoryFromBaseline(
    const juce::String &label) {
  auto after = captureLiveSnapshot();
  auto afterCurrent = audioProcessor.getCurrentPreset();
  if (afterCurrent.isAssociated())
    afterCurrent.dirty = true;
  audioProcessor.getEditHistory().pushStep(
      label, lastCommittedSnapshot, after, lastCommittedCurrent, afterCurrent);
  lastCommittedSnapshot = after;
  lastCommittedCurrent = afterCurrent;
  audioProcessor.refreshPresetDirtyFromFingerprint(after.sonicFingerprint());
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
}

void OpenYourBoxAudioProcessorEditor::beginHistoryGesture(
    const juce::String &label) {
  audioProcessor.getEditHistory().beginGesture(label, lastCommittedSnapshot,
                                               lastCommittedCurrent);
}

void OpenYourBoxAudioProcessorEditor::endHistoryGesture() {
  persistGraph(true, false);
  auto after = captureLiveSnapshot();
  auto afterCurrent = audioProcessor.getCurrentPreset();
  if (afterCurrent.isAssociated())
    afterCurrent.dirty = true;
  audioProcessor.getEditHistory().endGesture(after, afterCurrent);
  lastCommittedSnapshot = after;
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
  audioProcessor.refreshPresetDirtyFromFingerprint(after.sonicFingerprint());
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
}

void OpenYourBoxAudioProcessorEditor::reloadLiveGraphFromProcessor() {
  const auto restored = audioProcessor.getGraphState();
  if (restored.isValid())
    nodeGraph.restoreFromValueTree(restored);
  displayedConfiguration = audioProcessor.getRequestedConfiguration();
  graphInitialized = true;
  publishRuntimeControls();
  invalidateAnalysis();
  lastCommittedSnapshot = audioProcessor.capturePatchSnapshot();
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
  updateDirtyFromLivePatch();
}

void OpenYourBoxAudioProcessorEditor::performUndo() {
  const auto view = nodeGraph.getViewport();
  if (!audioProcessor.getEditHistory().undo())
    return;
  nodeGraph.getViewport() = view;
  persistGraph(false, false);
  updateDirtyFromLivePatch();
}

void OpenYourBoxAudioProcessorEditor::performRedo() {
  const auto view = nodeGraph.getViewport();
  if (!audioProcessor.getEditHistory().redo())
    return;
  nodeGraph.getViewport() = view;
  persistGraph(false, false);
  updateDirtyFromLivePatch();
}

void OpenYourBoxAudioProcessorEditor::associateCurrentPreset(
    const openyourbox::library::UserPresetEntry &entry) {
  openyourbox::state::CurrentPresetState current;
  current.entryId = entry.id;
  current.name = entry.name;
  current.dirty = false;
  current.baselineFingerprint = captureLiveSnapshot().sonicFingerprint();
  audioProcessor.setCurrentPreset(std::move(current));
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
}

void OpenYourBoxAudioProcessorEditor::updateDirtyFromLivePatch() {
  audioProcessor.refreshPresetDirtyFromFingerprint(
      audioProcessor.capturePatchSnapshot().sonicFingerprint());
  lastCommittedCurrent = audioProcessor.getCurrentPreset();
}

void OpenYourBoxAudioProcessorEditor::handlePresetSave() {
  const auto current = audioProcessor.getCurrentPreset();
  if (!current.isAssociated()) {
    showGraphMessage("Save As a name first");
    return;
  }
  juce::String error;
  const auto saved = presetLibrary.saveOverwrite(current.entryId,
                                                 captureLiveSnapshot(), error);
  if (!saved.has_value()) {
    showError(error);
    return;
  }
  associateCurrentPreset(*saved);
  showGraphMessage("Saved preset " + saved->name.toStdString());
}

void OpenYourBoxAudioProcessorEditor::handlePresetSaveAs(const juce::String &name,
                                                        bool overwrite) {
  juce::String error;
  const auto saved =
      presetLibrary.saveAs(captureLiveSnapshot(), name, overwrite, error);
  if (!saved.has_value()) {
    if (error.containsIgnoreCase("already exists") && !overwrite) {
      presetPanel.requestSaveAsOverwrite();
      return;
    }
    showError(error.isNotEmpty() ? error : juce::String("Could not save preset"));
    return;
  }
  presetPanel.closeSaveAsPopup();
  associateCurrentPreset(*saved);
  showGraphMessage("Saved preset " + saved->name.toStdString());
}

void OpenYourBoxAudioProcessorEditor::handlePresetLoad(const juce::String &id) {
  juce::String error;
  const auto snapshot = presetLibrary.loadSnapshot(id, error);
  if (!snapshot.has_value()) {
    showError(error);
    return;
  }
  const auto *entry = presetLibrary.findEntry(id);
  if (entry == nullptr) {
    showError("Preset catalog entry no longer exists");
    return;
  }
  const auto before = captureLiveSnapshot();
  const auto beforeCurrent = audioProcessor.getCurrentPreset();
  openyourbox::state::ApplyOptions options;
  options.weightPolicy =
      openyourbox::state::ApplyOptions::WeightPolicy::failClosed;
  if (!audioProcessor.applyPatchSnapshot(*snapshot, options, error)) {
    showError(error);
    return;
  }
  associateCurrentPreset(*entry);
  const auto after = audioProcessor.capturePatchSnapshot();
  const auto afterCurrent = audioProcessor.getCurrentPreset();
  audioProcessor.getEditHistory().pushStep("Load preset " + entry->name, before,
                                           after, beforeCurrent, afterCurrent);
  lastCommittedSnapshot = after;
  lastCommittedCurrent = afterCurrent;
  showGraphMessage("Loaded preset " + entry->name.toStdString());
}

void OpenYourBoxAudioProcessorEditor::handlePresetRename(
    const juce::String &id, const juce::String &name) {
  juce::String error;
  if (!presetLibrary.rename(id, name, error)) {
    showError(error);
    return;
  }
  auto current = audioProcessor.getCurrentPreset();
  if (current.entryId == id) {
    current.name = name.trim();
    audioProcessor.setCurrentPreset(current);
    lastCommittedCurrent = current;
  }
}

void OpenYourBoxAudioProcessorEditor::handlePresetDelete(const juce::String &id) {
  const auto wasCurrent = audioProcessor.getCurrentPreset().entryId == id;
  if (!presetLibrary.removeEntry(id)) {
    showError("Could not delete preset");
    return;
  }
  if (wasCurrent) {
    audioProcessor.setCurrentPreset({});
    lastCommittedCurrent = {};
  }
}
