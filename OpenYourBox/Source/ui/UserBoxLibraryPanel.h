#pragma once

#include "../library/UserBoxLibrary.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace openyourbox::ui {
/** @brief Drag-drop payload id for user box library rows. */
inline constexpr const char *boxLibraryPayloadId = "OPENYOURBOX_BOX_LIBRARY_ID";

/** @brief Font-size factor for Factory, User Library, and Project structure headers. */
inline constexpr float paletteRootHeaderFontScale = 0.82f;

/**
 * @brief Pushes gray small-caps style for a top-level palette tree header.
 *
 * Tree-node arrows use the text colour, so this mutes both the label and the
 * expand arrow. Header fill is cleared so hover, selection, and drag-over do
 * not highlight these section labels. Pair with @ref popPaletteRootHeaderStyle
 * immediately after TreeNodeEx so nested rows keep the default style.
 */
inline void pushPaletteRootHeaderStyle() {
  const ImVec4 transparent{0.0f, 0.0f, 0.0f, 0.0f};
  ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::PushStyleColor(ImGuiCol_Header, transparent);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, transparent);
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, transparent);
  ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase *
                               paletteRootHeaderFontScale);
}

/**
 * @brief Restores style pushed by @ref pushPaletteRootHeaderStyle.
 */
inline void popPaletteRootHeaderStyle() {
  ImGui::PopFont();
  ImGui::PopStyleColor(4);
}

/**
 * @brief Drag payload identifying a catalog entry and optional nested snapshot id.
 */
struct BoxLibraryDropPayload {
  /** @brief Catalog UUID (UTF-8). */
  char entryId[64]{};
  /** @brief Snapshot node/group id to insert; 0 inserts the saved root. */
  std::int32_t nestedRootId = 0;
};

/**
 * @class UserBoxLibraryPanel
 * @brief User Library tree of saved boxes drawn inside the unified Library palette.
 */
class UserBoxLibraryPanel {
public:
  /** @brief Message-thread actions emitted by the User Library tree. */
  struct Callbacks {
    /** @brief User-facing status or validation text. */
    std::function<void(const std::string &)> showMessage;
    /**
     * @brief Invoked when a catalog entry or nested subpart is single-clicked.
     * @param entryId Catalog UUID.
     * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
     */
    std::function<void(const juce::String &, std::int32_t)> inspectRequested;
    /**
     * @brief Invoked when a catalog row is double-clicked to insert on the canvas.
     * @param entryId Catalog UUID.
     * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
     */
    std::function<void(const juce::String &, std::int32_t)> placeRequested;
  };

  /**
   * @brief Draws the protected User Library root and its folders and boxes.
   * @param library Mutable catalog.
   * @param callbacks Editor-owned actions.
   * @param loadSnapshot Loads a catalog entry snapshot for nested member trees.
   */
  void render(library::UserBoxLibrary &library, const Callbacks &callbacks,
              const std::function<juce::ValueTree(const juce::String &,
                                                  juce::String &)> &loadSnapshot = {});

  /**
   * @brief Marks the protected Factory root as the active library selection.
   */
  void selectFactory();

  /**
   * @brief Returns true when Factory is the active library selection.
   */
  [[nodiscard]] bool isFactorySelected() const noexcept;

  /**
   * @brief Draws an expandable User Library submenu for Pin Add / Link Insert.
   * @param library Catalog to list.
   * @param loadSnapshot Loads a catalog entry snapshot for nested members.
   * @param onInsert Called with `(entryId, nestedRootId)` when a row is chosen.
   */
  static void renderInsertMenu(
      library::UserBoxLibrary &library,
      const std::function<juce::ValueTree(const juce::String &, juce::String &)>
          &loadSnapshot,
      const std::function<void(const juce::String &, std::int32_t)> &onInsert);

  /**
   * @brief Folder currently selected for new saves, or empty for User Library root.
   *
   * Factory selection maps to the User Library root so saves never target Factory.
   */
  [[nodiscard]] juce::String getSelectedFolder() const;

private:
  /**
   * @brief Draws one user-created folder node and its child folders and boxes.
   * @param library Mutable catalog.
   * @param folder Folder path to render.
   * @param callbacks Editor-owned actions.
   */
  void renderFolder(library::UserBoxLibrary &library, const juce::String &folder,
                    const Callbacks &callbacks);

  /**
   * @brief Draws one catalog row as a drag source.
   * @param library Mutable catalog.
   * @param entry Catalog row.
   * @param callbacks Editor-owned actions.
   */
  void renderEntry(library::UserBoxLibrary &library,
                   const library::UserBoxLibraryEntry &entry,
                   const Callbacks &callbacks);

  /**
   * @brief Selects a user folder (or the User Library root when @p folder is empty).
   * @param folder Destination folder path.
   */
  void selectUserFolder(const juce::String &folder);

  /**
   * @brief Queues the New Folder popup under @p parent.
   * @param parent Parent folder path, or empty for the User Library root.
   */
  void beginNewFolder(const juce::String &parent);

  /**
   * @brief Draws delete and new-folder modal dialogs.
   * @param library Mutable catalog.
   * @param callbacks Editor-owned actions.
   */
  void renderDialogs(library::UserBoxLibrary &library,
                     const Callbacks &callbacks);
  /**
   * @brief Starts a library drag for @p entry targeting optional nested id.
   * @param entry Catalog row.
   * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
   */
  void beginBoxDrag(const library::UserBoxLibraryEntry &entry,
                    std::int32_t nestedRootId) const;
  /**
   * @brief Handles inspect, double-click insert, and delayed drag on a row.
   * @param entry Catalog row.
   * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
   * @param callbacks Editor-owned actions.
   */
  void handleLibraryRowAction(const library::UserBoxLibraryEntry &entry,
                              std::int32_t nestedRootId,
                              const Callbacks &callbacks);
  /**
   * @brief Draws nested snapshot members under a group library entry.
   *
   * Members are ordered by direct links between displayed siblings; when none
   * connect them, entries fall back to case-insensitive label. Group Input/Output
   * hubs are omitted.
   * @param entry Catalog row being expanded.
   * @param snapshot Loaded box snapshot.
   * @param rootId Group id whose members to list.
   */
  void renderSnapshotMembers(const library::UserBoxLibraryEntry &entry,
                             const juce::ValueTree &snapshot,
                             std::int32_t rootId, const Callbacks &callbacks);

  /** @brief True when the protected Factory root is selected. */
  bool factorySelected = true;
  /** @brief Folder path selected in the User Library tree. */
  juce::String selectedFolder;
  /** @brief Entry currently being renamed. */
  juce::String renamingEntryId;
  /** @brief Folder currently being renamed. */
  juce::String renamingFolder;
  /** @brief Editable name buffer for rename popups. */
  std::array<char, 96> nameBuffer{};
  /** @brief New-folder name buffer. */
  std::array<char, 96> newFolderBuffer{};
  /** @brief Parent path for the pending New Folder popup. */
  juce::String newFolderParent;
  /** @brief True to open the New Folder popup this frame. */
  bool openNewFolderPopup = false;
  /** @brief Entry waiting for delete confirmation. */
  juce::String pendingDeleteEntryId;
  /** @brief Folder waiting for delete confirmation. */
  juce::String pendingDeleteFolder;
  /** @brief True when folder delete should remove contained boxes. */
  bool pendingDeleteFolderContents = false;
  /** @brief Snapshot loader bound for the current render pass. */
  std::function<juce::ValueTree(const juce::String &, juce::String &)>
      snapshotLoader;
  /** @brief Last catalog UUID used for double-click insert. */
  juce::String lastClickEntryId;
  /** @brief Last nested snapshot id used with @ref lastClickEntryId. */
  std::int32_t lastClickNestedRootId = 0;
  /** @brief Dear ImGui time of the last library row click. */
  double lastClickTime = -1.0;
};
} // namespace openyourbox::ui
