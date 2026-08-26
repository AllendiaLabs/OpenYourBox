#pragma once

#include "../library/UserPresetLibrary.h"
#include "../state/EditHistory.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <array>
#include <functional>
#include <string>

namespace openyourbox::ui {
/**
 * @class UserPresetPanel
 * @brief ImGui Presets browser with Save / Save As / load / rename / delete.
 */
class UserPresetPanel {
public:
  /** @brief Message-thread actions emitted by the Presets panel. */
  struct Callbacks {
    /** @brief Save overwrites the current catalog entry. */
    std::function<void()> save;
    /**
     * @brief Save As with a user-provided name.
     * @param name Requested catalog name.
     * @param overwrite True after the user confirms a collision.
     */
    std::function<void(const juce::String &name, bool overwrite)> saveAs;
    /**
     * @brief Loads the selected catalog entry.
     * @param id Entry UUID.
     */
    std::function<void(const juce::String &id)> load;
    /**
     * @brief Renames a catalog entry.
     * @param id Entry UUID.
     * @param name New unique display name.
     */
    std::function<void(const juce::String &id, const juce::String &name)> rename;
    /**
     * @brief Deletes a catalog entry after confirmation.
     * @param id Entry UUID.
     */
    std::function<void(const juce::String &id)> remove;
    /** @brief User-facing status or validation text. */
    std::function<void(const std::string &)> showMessage;
  };

  /**
   * @brief Draws the Presets catalog and current/dirty chrome.
   * @param library Mutable catalog.
   * @param current Session association for chrome.
   * @param callbacks Editor-owned actions.
   */
  void render(library::UserPresetLibrary &library,
              const state::CurrentPresetState &current,
              const Callbacks &callbacks);

  /** @brief Closes the Save As modal after a successful write. */
  void closeSaveAsPopup();

  /** @brief Shows overwrite confirmation copy inside the open Save As modal. */
  void requestSaveAsOverwrite();

private:
  /**
   * @brief Draws overwrite, rename, delete, and Save As dialogs.
   * @param library Mutable catalog.
   * @param callbacks Editor-owned actions.
   */
  void renderDialogs(library::UserPresetLibrary &library,
                     const Callbacks &callbacks);

  /** @brief Catalog id currently selected in the list. */
  juce::String selectedId;
  /** @brief Editable Save As / rename buffer. */
  std::array<char, 96> nameBuffer{};
  /** @brief True to open the Save As popup this frame. */
  bool openSaveAsPopup = false;
  /** @brief True when Save As is confirming overwrite of an existing name. */
  bool saveAsOverwrite = false;
  /** @brief Entry waiting for delete confirmation. */
  juce::String pendingDeleteId;
  /** @brief Entry currently being renamed. */
  juce::String renamingId;
  /** @brief True to open the rename popup this frame. */
  bool openRenamePopup = false;
};
} // namespace openyourbox::ui
