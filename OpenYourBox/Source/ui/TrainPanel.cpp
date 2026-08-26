#include "TrainPanel.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

namespace openyourbox::ui {
TrainPanel::TrainPanel() {
  std::snprintf(mlflowExperiment, sizeof mlflowExperiment, "%s",
                graph::defaultMlflowExperiment);
  std::snprintf(mlflowTrackingUri, sizeof mlflowTrackingUri, "%s",
                graph::defaultMlflowTrackingUri);
}

void TrainPanel::render(const train::TrainCoordinator &coordinator,
                        const Gates &gates, const Callbacks &callbacks) {
  ImGui::TextUnformatted("Train");
  if (!gates.isMaster) {
    ImGui::TextDisabled("Train is available only on the capture master.");
    return;
  }

  const auto status = coordinator.getStatus();
  const auto progress = coordinator.getProgress();
  const auto busy = status == train::TrainStatus::running ||
                    status == train::TrainStatus::paused;
  const bool mixed = gates.mixedSampleRates;
  const bool unpairedMapping =
      objective == graph::TrainObjective::mapping && gates.unpairedSelected;
  const bool reconstructionBlocked =
      objective == graph::TrainObjective::reconstruction &&
      gates.reconstructionPathInvalid;
  const bool gated =
      !gates.copyrightAcknowledged || gates.selectedPairCount < 1 ||
      gates.armedElementCount < 1 || mixed || unpairedMapping ||
      reconstructionBlocked;

  const char *objectiveLabel =
      objective == graph::TrainObjective::reconstruction ? "Reconstruction"
                                                         : "Mapping";
  if (ImGui::BeginCombo("Objective", objectiveLabel)) {
    if (ImGui::Selectable("Mapping",
                          objective == graph::TrainObjective::mapping))
      objective = graph::TrainObjective::mapping;
    if (ImGui::Selectable("Reconstruction",
                          objective == graph::TrainObjective::reconstruction))
      objective = graph::TrainObjective::reconstruction;
    ImGui::EndCombo();
  }

  ImGui::Text("%d items selected · ~%.1f min", gates.selectedPairCount,
              gates.selectedDurationSeconds / 60.0);
  if (ImGui::SmallButton("Open Library") && callbacks.openLibrary)
    callbacks.openLibrary();

  if (!gates.copyrightAcknowledged) {
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                       "Copyright acknowledgment required.");
    if (ImGui::Button("Acknowledge samples") && callbacks.requestCopyright)
      callbacks.requestCopyright();
  }

  if (gated && !busy)
    ImGui::BeginDisabled();
  if (status != train::TrainStatus::running &&
      status != train::TrainStatus::paused) {
    if (ImGui::Button("Run") && callbacks.run)
      callbacks.run();
  } else if (status == train::TrainStatus::running) {
    if (ImGui::Button("Pause") && callbacks.pause)
      callbacks.pause();
  } else if (ImGui::Button("Resume") && callbacks.resume) {
    callbacks.resume();
  }
  ImGui::SameLine();
  const bool canStop = busy;
  if (!canStop)
    ImGui::BeginDisabled();
  if (ImGui::Button("Stop") && callbacks.stop)
    callbacks.stop();
  if (!canStop)
    ImGui::EndDisabled();
  if (gated && !busy)
    ImGui::EndDisabled();

  if (ImGui::Checkbox("Hear changes while training", &hearWhileTraining) &&
      callbacks.hearWhileTrainingChanged)
    callbacks.hearWhileTrainingChanged(hearWhileTraining);

  if (mixed)
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s",
                       gates.blockReason.toRawUTF8());
  else if (unpairedMapping)
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
                       "Mapping cannot train unpaired clips. Deselect them.");
  else if (reconstructionBlocked)
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s",
                       gates.reconstructionReason.toRawUTF8());
  else if (gates.selectedPairCount < 1)
    ImGui::TextDisabled("Select at least one library pair.");
  else if (gates.armedElementCount < 1)
    ImGui::TextDisabled("Arm at least one trainable element.");

  ImGui::Separator();
  ImGui::TextUnformatted("Hyperparameters");
  if (busy)
    ImGui::BeginDisabled();
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Steps", &hyperparameters.totalSteps, 100, 500);
  hyperparameters.totalSteps =
      std::clamp(hyperparameters.totalSteps, 1, 100000);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Learning rate", &hyperparameters.learningRate, 0.0f, 0.0f,
                    "%.1e");
  hyperparameters.learningRate =
      std::clamp(hyperparameters.learningRate, 1.0e-6f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Segment length", &hyperparameters.segmentLength, 1024,
                  8192);
  hyperparameters.segmentLength =
      std::clamp(hyperparameters.segmentLength, 256, 2000000);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Checkpoint every", &hyperparameters.checkpointInterval, 10,
                  50);
  hyperparameters.checkpointInterval =
      std::clamp(hyperparameters.checkpointInterval, 1, 10000);

  ImGui::Separator();
  ImGui::TextUnformatted("MLflow");
  ImGui::Checkbox("Log to MLflow", &logToMlflow);
  if (!logToMlflow)
    ImGui::BeginDisabled();
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputText("Tracking URI", mlflowTrackingUri, sizeof mlflowTrackingUri);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputText("Experiment", mlflowExperiment, sizeof mlflowExperiment);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputText("Run name", mlflowRunName, sizeof mlflowRunName);
  if (!logToMlflow)
    ImGui::EndDisabled();
  if (busy)
    ImGui::EndDisabled();

  ImGui::Separator();
  const auto statusText = coordinator.getStatusMessage();
  const bool knownShortStatus =
      statusText == "Ready" || statusText == "Training..." ||
      statusText == "Paused" || statusText == "Stopped" ||
      statusText == "Training succeeded";
  const bool showErrorAction =
      status == train::TrainStatus::failed || !knownShortStatus ||
      progress.errorMessage.empty() == false;
  if (showErrorAction && status != train::TrainStatus::running &&
      status != train::TrainStatus::paused) {
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Status: failed");
    ImGui::SameLine();
    if (ImGui::Button("View error") && callbacks.viewError) {
      juce::String detail = statusText;
      const juce::String progressError(progress.errorMessage);
      if (progressError.isNotEmpty() && !detail.contains(progressError)) {
        if (detail.isNotEmpty())
          detail << "\n\n";
        detail << progressError;
      }
      callbacks.viewError(detail);
    }
  } else {
    ImGui::Text("Status: %s", statusText.toRawUTF8());
  }
  ImGui::Text("Step %d / %d", progress.step, progress.totalSteps);
  if (progress.stage.empty() == false)
    ImGui::Text("Stage: %s", progress.stage.c_str());
  ImGui::Text("Loss %.4f   Best %.4f   LR %.2e", progress.loss,
              progress.bestLoss > 0.0 ? progress.bestLoss : progress.loss,
              progress.learningRate);
  const auto losses = coordinator.getLossHistory();
  if (!losses.empty()) {
    ImGui::PlotLines("##trainLoss", losses.data(),
                     static_cast<int>(losses.size()), 0, "Loss", FLT_MAX,
                     FLT_MAX, ImVec2(-1.0f, 72.0f));
  } else {
    ImGui::TextDisabled("Loss plot appears after the first step.");
  }
  ImGui::TextDisabled("RF-aware crops · RF ≈ %.1f ms",
                      gates.receptiveFieldMilliseconds);
  ImGui::TextDisabled("Train window ≈ %.1f s", gates.trainWindowSeconds);

  if (gates.retryAvailable && ImGui::Button("Retry load") && callbacks.retryLoad)
    callbacks.retryLoad();
}
} // namespace openyourbox::ui
