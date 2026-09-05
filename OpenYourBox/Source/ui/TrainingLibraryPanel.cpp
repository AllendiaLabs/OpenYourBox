#include "TrainingLibraryPanel.h"
#include "InstrumentWidgets.h"

#include <cstdio>
#include <cstring>

namespace openyourbox::ui {
void TrainingLibraryPanel::render(library::TrainingLibrary &library,
                                  const Callbacks &callbacks,
                                  bool previewPlaying) {
  ImGui::TextUnformatted("Training Library");
  if (InstrumentWidgets::button("Import pair") && callbacks.importPair)
    callbacks.importPair();
  ImGui::SameLine();
  if (InstrumentWidgets::button("Import clip") && callbacks.importClip)
    callbacks.importClip();
  ImGui::SameLine();
  if (InstrumentWidgets::button("Select all"))
    library.selectAll();
  ImGui::SameLine();
  if (InstrumentWidgets::button("Select none"))
    library.selectNone();
  ImGui::Separator();

  ImGui::BeginChild("LibraryList", ImVec2(0.0f, 180.0f), true);
  for (auto &entry : library.getEntries()) {
    ImGui::PushID(entry.id.toRawUTF8());
    bool selected = entry.selectedForTrain;
    if (InstrumentWidgets::checkbox("##sel", &selected))
      library.setSelected(entry.id, selected);
    ImGui::SameLine();
    const auto kindLabel =
        entry.kind == library::LibraryEntryKind::clip ? "Clip" : "Pair";
    const auto label =
        entry.displayName + "  [" + kindLabel + " · " +
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
  if (InstrumentWidgets::button("Rename") && callbacks.rename)
    callbacks.rename(focused->id, juce::String(nameBuffer.data()));
  ImGui::Text("Duration: %.2f s", focused->durationSeconds);
  ImGui::Text("Sample rate: %.0f Hz  |  Channels: %d", focused->sampleRate,
              focused->channels);
  ImGui::TextWrapped("Created: %s", focused->createdAt.toRawUTF8());

  if (InstrumentWidgets::button("Preview x") && callbacks.preview)
    callbacks.preview(focused->id, true);
  ImGui::SameLine();
  if (InstrumentWidgets::button("Preview y") && callbacks.preview)
    callbacks.preview(focused->id, false);
  ImGui::SameLine();
  if (previewPlaying && InstrumentWidgets::button("Stop") && callbacks.stopPreview)
    callbacks.stopPreview();

  if (!confirmDelete) {
    if (InstrumentWidgets::button("Delete"))
      confirmDelete = true;
  } else {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
                       "Delete this pair and its audio files?");
    if (InstrumentWidgets::button("Confirm delete") && callbacks.removeEntry) {
      callbacks.removeEntry(focused->id);
      focusedId.clear();
      confirmDelete = false;
    }
    ImGui::SameLine();
    if (InstrumentWidgets::button("Cancel"))
      confirmDelete = false;
  }
}
} // namespace openyourbox::ui
