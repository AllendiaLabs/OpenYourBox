#include "TrainPanel.h"
#include "../train/CloudJobPackage.h"
#include "../graph/GraphTypes.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace openyourbox::ui {
namespace {
using FreezeNode = TrainPanel::Gates::FreezeStructureNode;

bool freezeContains(const std::vector<std::int32_t> &ids, std::int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void addFreezeId(std::vector<std::int32_t> &ids, std::int32_t id) {
  if (!freezeContains(ids, id))
    ids.push_back(id);
}

void removeFreezeId(std::vector<std::int32_t> &ids, std::int32_t id) {
  ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
}

enum class FreezeTriState { unchecked, checked, mixed };

FreezeTriState freezeLeafState(const TrainPanel::LossStageDraft &stage,
                               const FreezeNode &node) {
  if (!node.armedForTraining)
    return FreezeTriState::checked;
  return freezeContains(stage.freezeElementIds, node.id)
             ? FreezeTriState::checked
             : FreezeTriState::unchecked;
}

FreezeTriState freezeNodeState(const TrainPanel::LossStageDraft &stage,
                               const FreezeNode &node) {
  if (!node.isGroup)
    return freezeLeafState(stage, node);
  bool anyChecked = false;
  bool anyUnchecked = false;
  for (const auto &child : node.children) {
    switch (freezeNodeState(stage, child)) {
    case FreezeTriState::mixed:
      return FreezeTriState::mixed;
    case FreezeTriState::checked:
      anyChecked = true;
      break;
    case FreezeTriState::unchecked:
      anyUnchecked = true;
      break;
    }
  }
  if (anyChecked && anyUnchecked)
    return FreezeTriState::mixed;
  return anyChecked ? FreezeTriState::checked : FreezeTriState::unchecked;
}

void setFreezeSubtree(TrainPanel::LossStageDraft &stage, const FreezeNode &node,
                      bool frozen) {
  if (!node.isGroup) {
    if (!node.armedForTraining) {
      addFreezeId(stage.freezeElementIds, node.id);
      return;
    }
    if (frozen)
      addFreezeId(stage.freezeElementIds, node.id);
    else
      removeFreezeId(stage.freezeElementIds, node.id);
    return;
  }
  for (const auto &child : node.children)
    setFreezeSubtree(stage, child, frozen);
}

void syncForcedFreezeIds(TrainPanel::LossStageDraft &stage,
                         const std::vector<FreezeNode> &roots) {
  std::function<void(const FreezeNode &)> walk = [&](const FreezeNode &node) {
    if (!node.isGroup) {
      if (!node.armedForTraining)
        addFreezeId(stage.freezeElementIds, node.id);
      return;
    }
    for (const auto &child : node.children)
      walk(child);
  };
  for (const auto &root : roots)
    walk(root);
}

void renderFreezeStructureItem(TrainPanel::LossStageDraft &stage,
                               const FreezeNode &node) {
  ImGui::PushID(node.id);
  const auto state = freezeNodeState(stage, node);
  bool checked = state != FreezeTriState::unchecked;
  const bool forceDisabled = !node.armedForTraining;
  constexpr ImVec4 kDisabledGray{0.55f, 0.55f, 0.55f, 1.0f};
  constexpr ImVec4 kDisabledFrame{0.28f, 0.28f, 0.28f, 1.0f};
  if (state == FreezeTriState::mixed)
    ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
  if (forceDisabled) {
    ImGui::PushStyleColor(ImGuiCol_CheckMark, kDisabledGray);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kDisabledFrame);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, kDisabledFrame);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, kDisabledFrame);
    ImGui::PushStyleColor(ImGuiCol_CheckboxSelectedBg, kDisabledFrame);
    ImGui::PushStyleColor(ImGuiCol_Text, kDisabledGray);
    ImGui::BeginDisabled();
  }
  const bool toggled = ImGui::Checkbox("##freeze", &checked);
  if (forceDisabled) {
    ImGui::EndDisabled();
  }
  if (state == FreezeTriState::mixed)
    ImGui::PopItemFlag();
  if (toggled && !forceDisabled)
    setFreezeSubtree(stage, node, checked);
  ImGui::SameLine();
  if (node.isGroup) {
    const auto open = ImGui::TreeNodeEx(
        node.label.toRawUTF8(),
        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth);
    if (forceDisabled)
      ImGui::PopStyleColor(6);
    if (open) {
      for (const auto &child : node.children)
        renderFreezeStructureItem(stage, child);
      ImGui::TreePop();
    }
  } else {
    ImGui::TextUnformatted(node.label.toRawUTF8());
    if (forceDisabled)
      ImGui::PopStyleColor(6);
  }
  ImGui::PopID();
}
} // namespace

TrainPanel::TrainPanel() {
  std::snprintf(mlflowExperiment, sizeof mlflowExperiment, "%s",
                graph::defaultMlflowExperiment);
  std::snprintf(mlflowTrackingUri, sizeof mlflowTrackingUri, "%s",
                graph::defaultMlflowTrackingUri);
}

juce::var TrainPanel::captureConfig() const {
  auto object = std::make_unique<juce::DynamicObject>();
  object->setProperty("optimizer", "adam");
  object->setProperty("device",
                      juce::String(graph::trainDeviceName(hyperparameters.device)));
  object->setProperty("total_steps", hyperparameters.totalSteps);
  object->setProperty("learning_rate",
                      static_cast<double>(hyperparameters.learningRate));
  object->setProperty("generator_lr",
                      static_cast<double>(hyperparameters.generatorLr));
  object->setProperty("discriminator_lr",
                      static_cast<double>(hyperparameters.discriminatorLr));
  object->setProperty("batch_size", hyperparameters.batchSize);
  object->setProperty("segment_length", hyperparameters.segmentLength);
  object->setProperty("checkpoint_interval",
                      hyperparameters.checkpointInterval);
  object->setProperty("kl_beta", static_cast<double>(hyperparameters.klBeta));
  object->setProperty("adam_beta1",
                      static_cast<double>(hyperparameters.adamBeta1));
  object->setProperty("adam_beta2",
                      static_cast<double>(hyperparameters.adamBeta2));
  object->setProperty("lr_decay_end_factor",
                      static_cast<double>(hyperparameters.lrDecayEndFactor));
  object->setProperty("update_discriminator_every",
                      hyperparameters.updateDiscriminatorEvery);
  object->setProperty("phase_mangle_prob",
                      static_cast<double>(hyperparameters.phaseMangleProb));
  object->setProperty("dequantize_bits", hyperparameters.dequantizeBits);
  juce::Array<juce::var> stages;
  for (const auto &stage : lossStages) {
    auto item = std::make_unique<juce::DynamicObject>();
    item->setProperty("name", juce::String(stage.name));
    item->setProperty("steps", stage.steps);
    juce::Array<juce::var> losses;
    for (const auto &entry : stage.losses) {
      auto loss = std::make_unique<juce::DynamicObject>();
      loss->setProperty("loss_node_id", entry.lossNodeId);
      loss->setProperty("weight", static_cast<double>(entry.weight));
      losses.add(juce::var(loss.release()));
    }
    item->setProperty("losses", losses);
    juce::Array<juce::var> freezeIds;
    for (const auto id : stage.freezeElementIds)
      freezeIds.add(id);
    item->setProperty("freeze_element_ids", freezeIds);
    stages.add(juce::var(item.release()));
  }
  object->setProperty("loss_stage_schedule", stages);
  return juce::var(object.release());
}

bool TrainPanel::applyConfig(const juce::var &config) {
  configLoadWarning.clear();
  if (!config.isObject())
    return true;
  auto *object = config.getDynamicObject();
  if (object == nullptr)
    return true;
  bool missing = false;
  const auto applyInt = [&](const char *key, int &dest) {
    if (!object->hasProperty(key)) {
      missing = true;
      return;
    }
    dest = static_cast<int>(object->getProperty(key));
  };
  const auto applyFloat = [&](const char *key, float &dest) {
    if (!object->hasProperty(key)) {
      missing = true;
      return;
    }
    dest = static_cast<float>(static_cast<double>(object->getProperty(key)));
  };
  applyInt("total_steps", hyperparameters.totalSteps);
  applyFloat("learning_rate", hyperparameters.learningRate);
  applyFloat("generator_lr", hyperparameters.generatorLr);
  applyFloat("discriminator_lr", hyperparameters.discriminatorLr);
  applyInt("batch_size", hyperparameters.batchSize);
  applyInt("segment_length", hyperparameters.segmentLength);
  applyInt("checkpoint_interval", hyperparameters.checkpointInterval);
  applyFloat("kl_beta", hyperparameters.klBeta);
  applyFloat("adam_beta1", hyperparameters.adamBeta1);
  applyFloat("adam_beta2", hyperparameters.adamBeta2);
  applyFloat("lr_decay_end_factor", hyperparameters.lrDecayEndFactor);
  applyInt("update_discriminator_every",
           hyperparameters.updateDiscriminatorEvery);
  applyFloat("phase_mangle_prob", hyperparameters.phaseMangleProb);
  applyInt("dequantize_bits", hyperparameters.dequantizeBits);
  if (object->hasProperty("device"))
    hyperparameters.device = graph::trainDeviceFromName(
        object->getProperty("device").toString().toStdString());
  lossStages.clear();
  const auto stages = object->getProperty("loss_stage_schedule");
  if (auto *array = stages.getArray()) {
    for (const auto &item : *array) {
      if (!item.isObject())
        continue;
      LossStageDraft stage;
      const auto name = item.getProperty("name", "").toString();
      std::snprintf(stage.name, sizeof stage.name, "%s", name.toRawUTF8());
      stage.steps = static_cast<int>(item.getProperty("steps", 1000));
      if (auto *entries = item.getProperty("losses", {}).getArray()) {
        for (const auto &entry : *entries) {
          if (!entry.isObject())
            continue;
          LossStageEntry loss;
          loss.lossNodeId =
              static_cast<std::int32_t>(entry.getProperty("loss_node_id", 0));
          loss.weight = static_cast<float>(static_cast<double>(
              entry.getProperty("weight", graph::defaultLossWeight)));
          if (loss.lossNodeId != 0)
            stage.losses.push_back(loss);
        }
      } else if (auto *ids = item.getProperty("loss_node_ids", {}).getArray()) {
        for (const auto &id : *ids) {
          LossStageEntry loss;
          loss.lossNodeId = static_cast<std::int32_t>(id);
          loss.weight = graph::defaultLossWeight;
          if (loss.lossNodeId != 0)
            stage.losses.push_back(loss);
        }
      }
      if (auto *freeze = item.getProperty("freeze_element_ids", {}).getArray()) {
        for (const auto &id : *freeze) {
          const auto value = static_cast<std::int32_t>(id);
          if (value != 0)
            stage.freezeElementIds.push_back(value);
        }
      }
      lossStages.push_back(stage);
    }
  }
  if (missing)
    configLoadWarning =
        "Some training settings were missing and fell back to defaults.";
  return missing;
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
  const auto busy = status == train::TrainStatus::queued ||
                    status == train::TrainStatus::running;
  const bool mixed = gates.mixedSampleRates;
  // Temporary: Cloud Run does not require Allendia sign-in / credits.
  const bool gated =
      !gates.copyrightAcknowledged || gates.armedElementCount < 1 || mixed;

  if (!gates.dataLoaders.empty()) {
    const char *loaderLabel = "Choose Data Loader";
    for (const auto &loader : gates.dataLoaders) {
      if (loader.first == activeDataLoaderId) {
        loaderLabel = loader.second.toRawUTF8();
        break;
      }
    }
    if (gates.dataLoaders.size() == 1 && activeDataLoaderId == 0)
      activeDataLoaderId = gates.dataLoaders.front().first;
    if (ImGui::BeginCombo("Active Data Loader", loaderLabel)) {
      for (const auto &loader : gates.dataLoaders) {
        const bool selected = activeDataLoaderId == loader.first;
        if (ImGui::Selectable(loader.second.toRawUTF8(), selected)) {
          activeDataLoaderId = loader.first;
          if (callbacks.activeDataLoaderChanged)
            callbacks.activeDataLoaderChanged(activeDataLoaderId);
        }
        if (selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
  }

  const char *destinationLabel =
      destination == graph::TrainDestination::cloud ? "Allendia" : "Local";
  if (ImGui::BeginCombo("Destination", destinationLabel)) {
    if (ImGui::Selectable("Local",
                          destination == graph::TrainDestination::local)) {
      destination = graph::TrainDestination::local;
      if (callbacks.destinationChanged)
        callbacks.destinationChanged(destination);
    }
    if (ImGui::Selectable("Allendia",
                          destination == graph::TrainDestination::cloud)) {
      destination = graph::TrainDestination::cloud;
      if (callbacks.destinationChanged)
        callbacks.destinationChanged(destination);
    }
    ImGui::EndCombo();
  }
  if (destination == graph::TrainDestination::cloud) {
    ImGui::TextDisabled("Allendia sign-in optional for now.");
    if (gates.cloudLinked) {
      if (gates.entitlementUnavailable) {
        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                           "Allendia credits unavailable (ignored for now).");
        if (ImGui::SmallButton("Manage account") && callbacks.openStorefront)
          callbacks.openStorefront();
      } else
        ImGui::TextDisabled("Credits: available");
    }
    if (train::exceedsSoftUploadWarn(gates.selectedUploadBytes,
                                     gates.softUploadWarnBytes))
      ImGui::TextColored(
          ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
          "Upload is larger than 2 GiB. You can still continue.");
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
  if (!busy) {
    if (ImGui::Button("Run") && callbacks.run)
      callbacks.run();
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
  else if (gates.armedElementCount < 1)
    ImGui::TextDisabled("Arm at least one trainable element.");
  else if (gates.dataLoaders.empty())
    ImGui::TextDisabled("Insert a Data Loader to feed training.");
  else if (gates.dataLoaders.size() > 1 && activeDataLoaderId == 0)
    ImGui::TextDisabled("Choose the active Data Loader.");

  ImGui::Separator();
  ImGui::TextUnformatted("Hyperparameters");
  if (busy)
    ImGui::BeginDisabled();
  const char *deviceLabel = graph::trainDeviceLabel(hyperparameters.device);
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::BeginCombo("Device", deviceLabel)) {
    const graph::TrainDevice choices[] = {
        graph::TrainDevice::automatic, graph::TrainDevice::cpu,
        graph::TrainDevice::mps, graph::TrainDevice::cuda};
    for (const auto choice : choices) {
      const bool selected = hyperparameters.device == choice;
      if (ImGui::Selectable(graph::trainDeviceLabel(choice), selected))
        hyperparameters.device = choice;
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  const bool reconstruction = false;
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Steps", &hyperparameters.totalSteps, 100, 500);
  hyperparameters.totalSteps =
      std::clamp(hyperparameters.totalSteps, 1, 10000000);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Learning rate", &hyperparameters.learningRate, 0.0f, 0.0f,
                    "%.1e");
  hyperparameters.learningRate =
      std::clamp(hyperparameters.learningRate, 1.0e-6f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Generator LR", &hyperparameters.generatorLr, 0.0f, 0.0f,
                    "%.1e");
  hyperparameters.generatorLr =
      std::clamp(hyperparameters.generatorLr, 1.0e-8f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Discriminator LR", &hyperparameters.discriminatorLr, 0.0f,
                    0.0f, "%.1e");
  hyperparameters.discriminatorLr =
      std::clamp(hyperparameters.discriminatorLr, 1.0e-8f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Segment length", &hyperparameters.segmentLength, 1024, 8192);
  hyperparameters.segmentLength =
      std::clamp(hyperparameters.segmentLength, 256, 2000000);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Batch size", &hyperparameters.batchSize, 1, 2);
  hyperparameters.batchSize = std::clamp(hyperparameters.batchSize, 1, 64);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Adam beta1", &hyperparameters.adamBeta1, 0.0f, 0.0f,
                    "%.3f");
  hyperparameters.adamBeta1 =
      std::clamp(hyperparameters.adamBeta1, 0.0f, 0.999f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Adam beta2", &hyperparameters.adamBeta2, 0.0f, 0.0f,
                    "%.3f");
  hyperparameters.adamBeta2 =
      std::clamp(hyperparameters.adamBeta2, 0.0f, 0.9999f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("LR decay end", &hyperparameters.lrDecayEndFactor, 0.0f,
                    0.0f, "%.2f");
  hyperparameters.lrDecayEndFactor =
      std::clamp(hyperparameters.lrDecayEndFactor, 0.01f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("KL beta", &hyperparameters.klBeta, 0.0f, 0.0f, "%.4f");
  hyperparameters.klBeta = std::clamp(hyperparameters.klBeta, 0.0f, 10.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("KL beta start", &hyperparameters.klBetaStart, 0.0f, 0.0f,
                    "%.4f");
  hyperparameters.klBetaStart =
      std::clamp(hyperparameters.klBetaStart, 1.0e-12f, 10.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("KL warmup steps", &hyperparameters.klWarmupSteps, 1, 100);
  hyperparameters.klWarmupSteps =
      std::clamp(hyperparameters.klWarmupSteps, 1, 10000000);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Feature matching", &hyperparameters.featureMatchingWeight,
                    0.0f, 0.0f, "%.1f");
  hyperparameters.featureMatchingWeight =
      std::clamp(hyperparameters.featureMatchingWeight, 0.0f, 100.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Disc update every",
                  &hyperparameters.updateDiscriminatorEvery, 1, 2);
  hyperparameters.updateDiscriminatorEvery =
      std::clamp(hyperparameters.updateDiscriminatorEvery, 1, 64);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputFloat("Phase mangle p", &hyperparameters.phaseMangleProb, 0.0f,
                    0.0f, "%.2f");
  hyperparameters.phaseMangleProb =
      std::clamp(hyperparameters.phaseMangleProb, 0.0f, 1.0f);
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Dequant bits", &hyperparameters.dequantizeBits, 1, 2);
  hyperparameters.dequantizeBits =
      std::clamp(hyperparameters.dequantizeBits, 0, 32);
  (void)reconstruction;
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputInt("Checkpoint every", &hyperparameters.checkpointInterval, 10,
                  50);
  hyperparameters.checkpointInterval =
      std::clamp(hyperparameters.checkpointInterval, 1, 10000);

  ImGui::Separator();
  ImGui::TextUnformatted("Loss stages");
  ImGui::TextDisabled("Empty schedule trains all wired losses for Steps.");
  if (ImGui::SmallButton("Add stage")) {
    LossStageDraft stage;
    stage.steps = std::max(1, hyperparameters.totalSteps);
    lossStages.push_back(stage);
  }
  for (std::size_t index = 0; index < lossStages.size(); ++index) {
    auto &stage = lossStages[index];
    ImGui::PushID(static_cast<int>(index));
    ImGui::SetNextItemWidth(140.0f);
    ImGui::InputText("Name", stage.name, sizeof stage.name);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputInt("Steps##stage", &stage.steps, 100, 1000);
    stage.steps = std::clamp(stage.steps, 1, 10000000);
    ImGui::SameLine();
    if (ImGui::SmallButton("Remove")) {
      lossStages.erase(lossStages.begin() +
                       static_cast<std::ptrdiff_t>(index));
      ImGui::PopID();
      break;
    }
    for (const auto &loss : gates.lossNodes) {
      auto entryIt =
          std::find_if(stage.losses.begin(), stage.losses.end(),
                       [&](const LossStageEntry &entry) {
                         return entry.lossNodeId == loss.first;
                       });
      bool included = entryIt != stage.losses.end();
      ImGui::PushID(loss.first);
      if (ImGui::Checkbox(loss.second.toRawUTF8(), &included)) {
        if (included)
          stage.losses.push_back(
              {loss.first, graph::defaultLossWeight});
        else
          stage.losses.erase(std::remove_if(stage.losses.begin(),
                                            stage.losses.end(),
                                            [&](const LossStageEntry &entry) {
                                              return entry.lossNodeId ==
                                                     loss.first;
                                            }),
                             stage.losses.end());
        entryIt =
            std::find_if(stage.losses.begin(), stage.losses.end(),
                         [&](const LossStageEntry &entry) {
                           return entry.lossNodeId == loss.first;
                         });
      }
      if (included && entryIt != stage.losses.end()) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(72.0f);
        ImGui::InputFloat("Weight", &entryIt->weight, 0.0f, 0.0f, "%.3f");
        entryIt->weight = std::clamp(entryIt->weight, 0.0f, 100.0f);
      }
      ImGui::PopID();
    }
    syncForcedFreezeIds(stage, gates.freezeStructure);
    if (ImGui::TreeNode("Freeze##stageFreeze")) {
      ImGui::TextDisabled(
          "Armable boxes only. Unarmed = frozen (gray). Group check cascades.");
      if (gates.freezeStructure.empty())
        ImGui::TextDisabled("Empty graph");
      else {
        for (const auto &root : gates.freezeStructure)
          renderFreezeStructureItem(stage, root);
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Training configs");
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputText("Save as", configName, sizeof configName);
  if (ImGui::SmallButton("Save to library")) {
    juce::String error;
    configLibrary.saveAs(juce::String(configName), captureConfig(), error);
    if (error.isNotEmpty())
      configLoadWarning = error;
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Keep in project") && callbacks.saveProjectConfig)
    callbacks.saveProjectConfig(captureConfig());
  if (ImGui::BeginCombo("Load config", "Choose…")) {
    for (const auto &entry : configLibrary.getEntries()) {
      if (ImGui::Selectable(entry.name.toRawUTF8())) {
        applyConfig(entry.config);
        std::snprintf(configName, sizeof configName, "%s",
                      entry.name.toRawUTF8());
      }
    }
    ImGui::EndCombo();
  }
  if (configLoadWarning.isNotEmpty())
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f), "%s",
                       configLoadWarning.toRawUTF8());
  ImGui::TextDisabled("Examples (templates, not modes)");
  if (ImGui::SmallButton("Example: mapping-style")) {
    if (callbacks.loadExampleTemplate)
      callbacks.loadExampleTemplate("mapping");
    hyperparameters.totalSteps = graph::defaultTrainSteps;
    hyperparameters.learningRate = graph::defaultTrainLearningRate;
    hyperparameters.segmentLength = graph::defaultTrainSegmentLength;
    lossStages.clear();
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Example: reconstruction-style")) {
    if (callbacks.loadExampleTemplate)
      callbacks.loadExampleTemplate("reconstruction");
    hyperparameters.totalSteps = graph::defaultReconstructionStage1Steps +
                                 graph::defaultReconstructionStage2Steps;
    hyperparameters.generatorLr = graph::defaultReconstructionGeneratorLr;
    hyperparameters.discriminatorLr =
        graph::defaultReconstructionDiscriminatorLr;
    hyperparameters.segmentLength = graph::defaultReconstructionSegmentLength;
    hyperparameters.batchSize = graph::defaultReconstructionBatchSize;
    lossStages.clear();
    LossStageDraft representation;
    std::snprintf(representation.name, sizeof representation.name,
                  "representation");
    representation.steps = graph::defaultReconstructionStage1Steps;
    lossStages.push_back(representation);
    LossStageDraft quality;
    std::snprintf(quality.name, sizeof quality.name, "quality");
    quality.steps = graph::defaultReconstructionStage2Steps;
    lossStages.push_back(quality);
  }

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
  if (gates.cloudOffline)
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                       "Offline — reconnecting. The cloud job is still running.");
  const auto submit = coordinator.getSubmitProgress();
  if (submit.phase == train::CloudSubmitProgress::Phase::uploading &&
      submit.bytesTotal > 0)
    ImGui::Text("Upload %.1f / %.1f MiB",
                static_cast<double>(submit.bytesSent) / (1024.0 * 1024.0),
                static_cast<double>(submit.bytesTotal) / (1024.0 * 1024.0));
  const auto statusText = coordinator.getStatusMessage();
  const bool knownShortStatus =
      statusText == "Ready" || statusText == "Training..." ||
      statusText == "Stopped" ||
      statusText == "Training succeeded" || statusText == "Queued..." ||
      statusText == "Packaging..." || statusText == "Uploading..." ||
      statusText == "Attached cloud job" ||
      statusText == "Cloud job succeeded";
  const bool showErrorAction =
      status == train::TrainStatus::failed ||
      (!knownShortStatus && !busy) ||
      (status == train::TrainStatus::failed &&
       progress.errorMessage.empty() == false);
  if (showErrorAction && !busy) {
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
  if (!progress.device.empty()) {
    if (progress.deviceFallback && !progress.requestedDevice.empty() &&
        progress.requestedDevice != "auto")
      ImGui::Text("Device: %s (requested %s)", progress.device.c_str(),
                  progress.requestedDevice.c_str());
    else
      ImGui::Text("Device: %s", progress.device.c_str());
  }
  ImGui::Text("Step %d / %d", progress.step, progress.totalSteps);
  if (progress.stage.empty() == false)
    ImGui::Text("Stage: %s", progress.stage.c_str());
  if (progress.compactnessReady)
    ImGui::Text("Compactness: ready (%d val segments)",
                progress.compactnessValidationSegments);
  else if (progress.hasEncodeDecode)
    ImGui::TextDisabled("Compactness not ready");
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

  if (!gates.cloudCheckpoints.empty()) {
    ImGui::Separator();
    ImGui::TextUnformatted("Allendia checkpoints");
    for (const auto &checkpoint : gates.cloudCheckpoints) {
      ImGui::PushID(checkpoint.checkpointId.toRawUTF8());
      ImGui::Text("step %d %s", checkpoint.step,
                  checkpoint.stage.toRawUTF8());
      ImGui::SameLine();
      if (ImGui::SmallButton("Download") && callbacks.downloadCheckpoint)
        callbacks.downloadCheckpoint(checkpoint.checkpointId);
      ImGui::PopID();
    }
  }
  if (gates.manualCloudLoadAvailable) {
    ImGui::TextUnformatted("Success on another machine: download and load manually.");
    if (ImGui::Button("Download / Load") && callbacks.manualCloudLoad)
      callbacks.manualCloudLoad();
  }

  if (gates.retryAvailable && ImGui::Button("Retry load") && callbacks.retryLoad)
    callbacks.retryLoad();
}
} // namespace openyourbox::ui
