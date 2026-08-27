#include "UserBoxLibraryPanel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace openyourbox::ui {
namespace {
/**
 * @brief Returns immediate child folder names under @p parent.
 * @param folders All persisted folder paths.
 * @param parent Parent folder path, or empty for root.
 */
std::vector<juce::String>
childFolders(const std::vector<juce::String> &folders,
             const juce::String &parent) {
  std::vector<juce::String> children;
  for (const auto &folder : folders) {
    juce::String remainder = folder;
    if (parent.isNotEmpty()) {
      if (folder == parent || !folder.startsWith(parent + "/"))
        continue;
      remainder = folder.substring(parent.length() + 1);
    }
    if (remainder.containsChar('/'))
      remainder = remainder.upToFirstOccurrenceOf("/", false, false);
    if (remainder.isNotEmpty() &&
        std::find(children.begin(), children.end(), remainder) == children.end())
      children.push_back(remainder);
  }
  std::sort(children.begin(), children.end(),
            [](const juce::String &left, const juce::String &right) {
              return left.compareIgnoreCase(right) < 0;
            });
  return children;
}

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

/**
 * @brief Accepts a box-library drag onto the last item and moves it to @p folder.
 * @param library Mutable catalog.
 * @param folder Destination folder path.
 * @param callbacks Editor-owned actions.
 */
void acceptBoxDrop(library::UserBoxLibrary &library, const juce::String &folder,
                   const UserBoxLibraryPanel::Callbacks &callbacks) {
  if (!ImGui::BeginDragDropTarget())
    return;
  if (const auto *payload = ImGui::AcceptDragDropPayload(boxLibraryPayloadId)) {
    const auto *drop =
        static_cast<const BoxLibraryDropPayload *>(payload->Data);
    const auto id = juce::String::fromUTF8(drop->entryId);
    juce::String error;
    if (!library.setFolder(id, folder, error) && callbacks.showMessage &&
        error.isNotEmpty())
      callbacks.showMessage(error.toStdString());
  }
  ImGui::EndDragDropTarget();
}
} // namespace

void UserBoxLibraryPanel::selectFactory() {
  factorySelected = true;
  selectedFolder.clear();
}

bool UserBoxLibraryPanel::isFactorySelected() const noexcept {
  return factorySelected;
}

juce::String UserBoxLibraryPanel::getSelectedFolder() const {
  return factorySelected ? juce::String{} : selectedFolder;
}

void UserBoxLibraryPanel::selectUserFolder(const juce::String &folder) {
  factorySelected = false;
  selectedFolder = folder;
}

void UserBoxLibraryPanel::beginNewFolder(const juce::String &parent) {
  factorySelected = false;
  selectedFolder = parent;
  newFolderParent = parent;
  newFolderBuffer[0] = '\0';
  openNewFolderPopup = true;
}

void UserBoxLibraryPanel::render(
    library::UserBoxLibrary &library, const Callbacks &callbacks,
    const std::function<juce::ValueTree(const juce::String &, juce::String &)>
        &loadSnapshot) {
  snapshotLoader = loadSnapshot;
  const auto selected = !factorySelected && selectedFolder.isEmpty();
  const auto flags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
      ImGuiTreeNodeFlags_SpanAvailWidth |
      (selected ? ImGuiTreeNodeFlags_Selected : 0);
  const auto open = ImGui::TreeNodeEx("User Library", flags);
  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
    selectUserFolder({});
  acceptBoxDrop(library, {}, callbacks);
  if (ImGui::BeginPopupContextItem("UserLibraryRootMenu")) {
    selectUserFolder({});
    if (ImGui::MenuItem("New folder"))
      beginNewFolder({});
    ImGui::BeginDisabled();
    ImGui::MenuItem("Rename");
    ImGui::MenuItem("Delete");
    ImGui::EndDisabled();
    ImGui::EndPopup();
  }
  if (open) {
    if (library.getEntries().empty() && library.getFolders().empty()) {
      ImGui::TextWrapped(
          "Save a box from the graph context menu. Drag boxes onto the canvas "
          "to place them.");
    } else
      renderFolder(library, {}, callbacks);
    ImGui::TreePop();
  }

  if (ImGui::BeginPopupContextWindow(
          "LibraryEmptyMenu", ImGuiPopupFlags_MouseButtonRight |
                                  ImGuiPopupFlags_NoOpenOverItems)) {
    if (ImGui::MenuItem("New folder"))
      beginNewFolder(getSelectedFolder());
    ImGui::EndPopup();
  }

  renderDialogs(library, callbacks);
}

void UserBoxLibraryPanel::renderFolder(library::UserBoxLibrary &library,
                                       const juce::String &folder,
                                       const Callbacks &callbacks) {
  const auto children = childFolders(library.getFolders(), folder);
  std::vector<const library::UserBoxLibraryEntry *> entries;
  for (const auto &entry : library.getEntries()) {
    if (entry.folder == folder)
      entries.push_back(&entry);
  }

  /** @brief One folder or box row, sorted together by display name. */
  struct ChildRow {
    juce::String label;
    juce::String childFolder;
    const library::UserBoxLibraryEntry *entry = nullptr;
  };
  std::vector<ChildRow> rows;
  rows.reserve(children.size() + entries.size());
  for (const auto &child : children) {
    ChildRow row;
    row.label = child;
    row.childFolder = folder.isEmpty() ? child : folder + "/" + child;
    rows.push_back(std::move(row));
  }
  for (const auto *entry : entries) {
    ChildRow row;
    row.label = entry->name;
    row.entry = entry;
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end(),
            [](const ChildRow &left, const ChildRow &right) {
              return left.label.compareIgnoreCase(right.label) < 0;
            });

  const auto renderChildren = [&]() {
    for (const auto &row : rows) {
      if (row.entry != nullptr)
        renderEntry(library, *row.entry, callbacks);
      else
        renderFolder(library, row.childFolder, callbacks);
    }
  };

  if (folder.isEmpty()) {
    renderChildren();
    return;
  }

  ImGui::PushID(folder.toRawUTF8());
  const auto selected = !factorySelected && selectedFolder == folder;
  const auto flags =
      ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
      (selected ? ImGuiTreeNodeFlags_Selected : 0);
  const auto open = ImGui::TreeNodeEx("##folder", flags, "%s",
                                      folder.fromLastOccurrenceOf("/", false, false)
                                          .toRawUTF8());
  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
    selectUserFolder(folder);
  acceptBoxDrop(library, folder, callbacks);
  if (ImGui::BeginPopupContextItem("BoxFolderMenu")) {
    selectUserFolder(folder);
    if (ImGui::MenuItem("New folder"))
      beginNewFolder(folder);
    if (ImGui::MenuItem("Rename")) {
      renamingFolder = folder;
      fillBuffer(nameBuffer,
                 folder.fromLastOccurrenceOf("/", false, false));
    }
    if (ImGui::MenuItem("Delete")) {
      pendingDeleteFolder = folder;
      pendingDeleteFolderContents = false;
      for (const auto &entry : library.getEntries()) {
        if (entry.folder == folder || entry.folder.startsWith(folder + "/")) {
          pendingDeleteFolderContents = true;
          break;
        }
      }
    }
    ImGui::EndPopup();
  }
  if (renamingFolder == folder) {
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputText("##renameFolder", nameBuffer.data(),
                         nameBuffer.size(),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
      juce::String error;
      if (library.renameFolder(folder, juce::String(nameBuffer.data()),
                               error)) {
        auto next = folder.upToLastOccurrenceOf("/", false, false);
        if (next == folder)
          next = juce::String(nameBuffer.data()).trim();
        else
          next = next + "/" + juce::String(nameBuffer.data()).trim();
        selectUserFolder(next);
        renamingFolder.clear();
      } else if (callbacks.showMessage && error.isNotEmpty())
        callbacks.showMessage(error.toStdString());
    }
    if (ImGui::IsItemDeactivated() && !ImGui::IsItemDeactivatedAfterEdit())
      renamingFolder.clear();
  }
  if (open) {
    renderChildren();
    ImGui::TreePop();
  }
  ImGui::PopID();
}

void UserBoxLibraryPanel::renderEntry(
    library::UserBoxLibrary &library, const library::UserBoxLibraryEntry &entry,
    const Callbacks &callbacks) {
  ImGui::PushID(entry.id.toRawUTF8());
  const auto kind =
      entry.kind == library::UserBoxKind::group ? "Group" : "Element";
  char label[160];
  std::snprintf(label, sizeof(label), "%s  (%s)", entry.name.toRawUTF8(), kind);
  const auto groupEntry = entry.kind == library::UserBoxKind::group;
  const auto flags =
      ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
      (groupEntry ? 0
                  : ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
  const auto open = ImGui::TreeNodeEx("##boxEntry", flags, "%s", label);
  beginBoxDrag(entry, 0);
  if (ImGui::BeginPopupContextItem("BoxEntryMenu")) {
    if (ImGui::MenuItem("Rename")) {
      renamingEntryId = entry.id;
      fillBuffer(nameBuffer, entry.name);
    }
    if (ImGui::MenuItem("Delete"))
      pendingDeleteEntryId = entry.id;
    ImGui::EndPopup();
  }
  if (renamingEntryId == entry.id) {
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputText("##renameBox", nameBuffer.data(), nameBuffer.size(),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
      juce::String error;
      if (library.rename(entry.id, juce::String(nameBuffer.data()), error))
        renamingEntryId.clear();
      else if (callbacks.showMessage && error.isNotEmpty())
        callbacks.showMessage(error.toStdString());
    }
    if (ImGui::IsItemDeactivated() && !ImGui::IsItemDeactivatedAfterEdit())
      renamingEntryId.clear();
  }
  if (groupEntry && open) {
    juce::String error;
    juce::ValueTree snapshot;
    if (snapshotLoader)
      snapshot = snapshotLoader(entry.id, error);
    if (snapshot.isValid())
      renderSnapshotMembers(entry, snapshot,
                            static_cast<std::int32_t>(snapshot["rootId"]));
    else if (error.isNotEmpty() && callbacks.showMessage)
      callbacks.showMessage(error.toStdString());
    ImGui::TreePop();
  }
  ImGui::PopID();
}

void UserBoxLibraryPanel::beginBoxDrag(const library::UserBoxLibraryEntry &entry,
                                       std::int32_t nestedRootId) const {
  if (!ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
    return;
  BoxLibraryDropPayload payload;
  const auto utf8 = entry.id.toRawUTF8();
  std::strncpy(payload.entryId, utf8, sizeof(payload.entryId) - 1);
  payload.nestedRootId = nestedRootId;
  ImGui::SetDragDropPayload(boxLibraryPayloadId, &payload, sizeof(payload));
  ImGui::TextUnformatted(entry.name.toRawUTF8());
  ImGui::EndDragDropSource();
}

void UserBoxLibraryPanel::renderSnapshotMembers(
    const library::UserBoxLibraryEntry &entry, const juce::ValueTree &snapshot,
    std::int32_t rootId) {
  struct MemberRow {
    std::int32_t id = 0;
    juce::String label;
    bool isGroup = false;
  };
  std::unordered_map<std::int32_t, juce::ValueTree> groups;
  std::unordered_map<std::int32_t, juce::String> nodeLabels;
  for (const auto child : snapshot) {
    if (child.hasType("Group"))
      groups.emplace(static_cast<std::int32_t>(child["id"]), child);
    else if (child.hasType("Node") &&
             child["type"].toString() != "group_input" &&
             child["type"].toString() != "group_output")
      nodeLabels.emplace(static_cast<std::int32_t>(child["id"]),
                         child["label"].toString());
  }
  const auto found = groups.find(rootId);
  if (found == groups.end())
    return;
  std::vector<MemberRow> rows;
  for (const auto member : found->second) {
    if (!member.hasType("Member"))
      continue;
    MemberRow row;
    row.id = static_cast<std::int32_t>(member["id"]);
    const auto group = groups.find(row.id);
    if (group != groups.end()) {
      row.isGroup = true;
      row.label = group->second.getProperty("name", "Group").toString();
    } else {
      const auto label = nodeLabels.find(row.id);
      if (label == nodeLabels.end())
        continue;
      row.label = label->second;
    }
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end(),
            [](const MemberRow &left, const MemberRow &right) {
              return left.label.compareIgnoreCase(right.label) < 0;
            });
  for (const auto &row : rows) {
    ImGui::PushID(row.id);
    const auto flags =
        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
        (row.isGroup
             ? 0
             : ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
    const auto open =
        ImGui::TreeNodeEx("##member", flags, "%s", row.label.toRawUTF8());
    beginBoxDrag(entry, row.id);
    if (row.isGroup && open) {
      renderSnapshotMembers(entry, snapshot, row.id);
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
}

void UserBoxLibraryPanel::renderDialogs(library::UserBoxLibrary &library,
                                        const Callbacks &callbacks) {
  if (!pendingDeleteEntryId.isEmpty() &&
      ImGui::BeginPopupModal("Delete Box", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Delete this box from the library?");
    if (ImGui::Button("Delete")) {
      library.removeEntry(pendingDeleteEntryId);
      pendingDeleteEntryId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      pendingDeleteEntryId.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else if (!pendingDeleteEntryId.isEmpty())
    ImGui::OpenPopup("Delete Box");

  if (!pendingDeleteFolder.isEmpty() &&
      ImGui::BeginPopupModal("Delete Box Folder", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s",
                       pendingDeleteFolderContents
                           ? "Delete this folder and every box inside it?"
                           : "Delete this empty folder?");
    if (ImGui::Button("Delete")) {
      juce::String error;
      if (!library.removeFolder(pendingDeleteFolder, pendingDeleteFolderContents,
                                error) &&
          callbacks.showMessage && error.isNotEmpty())
        callbacks.showMessage(error.toStdString());
      if (selectedFolder == pendingDeleteFolder)
        selectedFolder.clear();
      pendingDeleteFolder.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      pendingDeleteFolder.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else if (!pendingDeleteFolder.isEmpty())
    ImGui::OpenPopup("Delete Box Folder");

  if (openNewFolderPopup) {
    ImGui::OpenPopup("New Library Folder");
    openNewFolderPopup = false;
  }
  if (ImGui::BeginPopupModal("New Library Folder", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Folder name");
    if (ImGui::IsWindowAppearing())
      ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(220.0f);
    const auto enter =
        ImGui::InputText("##newFolderName", newFolderBuffer.data(),
                         newFolderBuffer.size(),
                         ImGuiInputTextFlags_EnterReturnsTrue);
    const auto submit = ImGui::Button("Create") || enter;
    if (submit) {
      auto leaf = juce::String(newFolderBuffer.data()).trim();
      juce::String error;
      if (leaf.isEmpty() || leaf.containsAnyOf("/\\"))
        error = "Enter a valid folder name";
      else {
        const auto name =
            newFolderParent.isEmpty() ? leaf : newFolderParent + "/" + leaf;
        if (library.createFolder(name, error)) {
          selectUserFolder(name);
          newFolderBuffer[0] = '\0';
          ImGui::CloseCurrentPopup();
        }
      }
      if (error.isNotEmpty() && callbacks.showMessage)
        callbacks.showMessage(error.toStdString());
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}
} // namespace openyourbox::ui
