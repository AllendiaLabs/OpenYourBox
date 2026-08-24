#include "TrainingLibraryPanel.h"

#include <cstdio>
#include <cstring>

namespace openyourbox::ui {
void TrainingLibraryPanel::render(library::TrainingLibrary &library,
                                  const Callbacks &callbacks,
                                  bool previewPlaying) {
  ImGui::TextUnformatted("Training Library");
  if (ImGui::Button("Import pair") && callbacks.importPair)
    callbacks.importPair();
  ImGui::SameLine();
  if (ImGui::Button("Select all"))
    library.selectAll();
  ImGui::SameLine();
  if (ImGui::Button("Select none"))
    library.selectNone();
  ImGui::Separator();

  ImGui::BeginChild("LibraryList", ImVec2(0.0f, 180.0f), true);
  for (auto &entry : library.getEntries()) {
    ImGui::PushID(entry.id.toRawUTF8());
    bool selected = entry.selectedForTrain;
    if (ImGui::Checkbox("##sel", &selected))
      library.setSelected(entry.id, selected);
    ImGui::SameLine();
    const auto label =
        entry.displayName + "  [" +
        (entry.source == library::PairSource::capture ? "Capture" : "Import") +
        "]";
    if (ImGui::Selectable(label.toRawUTF8(), focusedId == entry.id)) {
      focusedId = entry.id;
      std::snprintf(nameBuffer.data(), nameBuffer.size(), "%s",
                    entry.displayName.toRawUTF8());
      confirmDelete = false;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();

  auto *focused = library.findEntry(focusedId);
  if (focused == nullptr) {
    ImGui::TextDisabled("Select a pair to inspect, preview, rename, or delete.");
    return;
  }

  ImGui::Separator();
  ImGui::Text("Detail");
  ImGui::InputText("Name", nameBuffer.data(), nameBuffer.size());
  if (ImGui::Button("Rename") && callbacks.rename)
    callbacks.rename(focused->id, juce::String(nameBuffer.data()));
  ImGui::Text("Duration: %.2f s", focused->durationSeconds);
  ImGui::Text("Sample rate: %.0f Hz  |  Channels: %d", focused->sampleRate,
              focused->channels);
  ImGui::TextWrapped("Created: %s", focused->createdAt.toRawUTF8());

  if (ImGui::Button("Preview x") && callbacks.preview)
    callbacks.preview(focused->id, true);
  ImGui::SameLine();
  if (ImGui::Button("Preview y") && callbacks.preview)
    callbacks.preview(focused->id, false);
  ImGui::SameLine();
  if (previewPlaying && ImGui::Button("Stop") && callbacks.stopPreview)
    callbacks.stopPreview();

  if (!confirmDelete) {
    if (ImGui::Button("Delete"))
      confirmDelete = true;
  } else {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
                       "Delete this pair and its audio files?");
    if (ImGui::Button("Confirm delete") && callbacks.removeEntry) {
      callbacks.removeEntry(focused->id);
      focusedId.clear();
      confirmDelete = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
      confirmDelete = false;
  }
}
} // namespace openyourbox::ui
