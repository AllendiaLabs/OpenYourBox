#include "UserPresetPanel.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace openyourbox::ui {
namespace {
/**
 * @brief Copies @p text into a null-terminated ImGui buffer.
 * @param buffer Destination buffer.
 * @param text Source text.
 */
template <std::size_t Size>
void fillBuffer(std::array<char, Size> &buffer, const juce::String &text) {
  const auto utf8 = text.toRawUTF8();
  const auto length = std::min(Size - 1, std::strlen(utf8));
  std::memcpy(buffer.data(), utf8, length);
  buffer[length] = '\0';
}
} // namespace

void UserPresetPanel::render(library::UserPresetLibrary &library,
                             const state::CurrentPresetState &current,
                             const Callbacks &callbacks) {
  ImGui::TextUnformatted("Presets");
  if (current.isAssociated()) {
    ImGui::SameLine();
    ImGui::TextDisabled("— %s%s", current.name.toRawUTF8(),
                        current.dirty ? " *" : "");
  } else {
    ImGui::SameLine();
    ImGui::TextDisabled("— none");
  }

  const auto canSave = current.isAssociated();
  ImGui::BeginDisabled(!canSave);
  if (ImGui::Button("Save") && callbacks.save)
    callbacks.save();
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !canSave)
    ImGui::SetTooltip("Save As a name first");
  ImGui::SameLine();
  if (ImGui::Button("Save As")) {
    fillBuffer(nameBuffer, {});
    saveAsOverwrite = false;
    openSaveAsPopup = true;
  }
  ImGui::SameLine();
  const auto *selected = library.findEntry(selectedId);
  ImGui::BeginDisabled(selected == nullptr);
  if (ImGui::Button("Load") && selected != nullptr && callbacks.load)
    callbacks.load(selected->id);
  ImGui::SameLine();
  if (ImGui::Button("Rename") && selected != nullptr) {
    renamingId = selected->id;
    fillBuffer(nameBuffer, selected->name);
    openRenamePopup = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Delete") && selected != nullptr)
    pendingDeleteId = selected->id;
  ImGui::EndDisabled();

  ImGui::Separator();
  if (library.getEntries().empty()) {
    ImGui::TextWrapped("No presets yet. Save the current patch as a preset.");
  } else {
    std::vector<const library::UserPresetEntry *> rows;
    rows.reserve(library.getEntries().size());
    for (const auto &entry : library.getEntries())
      rows.push_back(&entry);
    std::sort(rows.begin(), rows.end(),
              [](const library::UserPresetEntry *left,
                 const library::UserPresetEntry *right) {
                return left->name.compareIgnoreCase(right->name) < 0;
              });
    if (ImGui::BeginChild("PresetList", ImVec2(0.0f, 0.0f), true)) {
      for (const auto *entry : rows) {
        const auto isCurrent = current.entryId == entry->id;
        const auto label = isCurrent && current.dirty
                               ? entry->name + " *"
                               : entry->name;
        const auto selectedRow = selectedId == entry->id;
        if (ImGui::Selectable(label.toRawUTF8(), selectedRow,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
          selectedId = entry->id;
          if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
              callbacks.load)
            callbacks.load(entry->id);
        }
        if (ImGui::BeginPopupContextItem()) {
          selectedId = entry->id;
          if (ImGui::MenuItem("Load") && callbacks.load)
            callbacks.load(entry->id);
          if (ImGui::MenuItem("Rename")) {
            renamingId = entry->id;
            fillBuffer(nameBuffer, entry->name);
            openRenamePopup = true;
          }
          if (ImGui::MenuItem("Delete"))
            pendingDeleteId = entry->id;
          ImGui::EndPopup();
        }
      }
    }
    ImGui::EndChild();
  }

  renderDialogs(library, callbacks);
}

void UserPresetPanel::renderDialogs(library::UserPresetLibrary &library,
                                    const Callbacks &callbacks) {
  if (openSaveAsPopup) {
    ImGui::OpenPopup("Save Preset As");
    openSaveAsPopup = false;
  }
  if (ImGui::BeginPopupModal("Save Preset As", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Name");
    ImGui::SetNextItemWidth(260.0f);
    ImGui::InputText("##saveAsName", nameBuffer.data(), nameBuffer.size());
    if (saveAsOverwrite)
      ImGui::TextWrapped("A preset with this name exists and will be replaced.");
    if (ImGui::Button("Save") && callbacks.saveAs) {
      const auto name = juce::String(nameBuffer.data());
      callbacks.saveAs(name, saveAsOverwrite);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      saveAsOverwrite = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (openRenamePopup) {
    ImGui::OpenPopup("Rename Preset");
    openRenamePopup = false;
  }
  if (ImGui::BeginPopupModal("Rename Preset", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Name");
    ImGui::SetNextItemWidth(260.0f);
    ImGui::InputText("##renamePreset", nameBuffer.data(), nameBuffer.size());
    if (ImGui::Button("Rename") && callbacks.rename) {
      callbacks.rename(renamingId, juce::String(nameBuffer.data()));
      renamingId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      renamingId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (pendingDeleteId.isNotEmpty())
    ImGui::OpenPopup("Delete Preset");
  if (ImGui::BeginPopupModal("Delete Preset", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    const auto *entry = library.findEntry(pendingDeleteId);
    ImGui::TextWrapped("Delete preset \"%s\"? This cannot be undone.",
                       entry != nullptr ? entry->name.toRawUTF8() : "this preset");
    if (ImGui::Button("Delete") && callbacks.remove) {
      callbacks.remove(pendingDeleteId);
      if (selectedId == pendingDeleteId)
        selectedId.clear();
      pendingDeleteId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      pendingDeleteId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void UserPresetPanel::closeSaveAsPopup() {
  saveAsOverwrite = false;
  if (ImGui::IsPopupOpen("Save Preset As"))
    ImGui::CloseCurrentPopup();
}

void UserPresetPanel::requestSaveAsOverwrite() { saveAsOverwrite = true; }
} // namespace openyourbox::ui
