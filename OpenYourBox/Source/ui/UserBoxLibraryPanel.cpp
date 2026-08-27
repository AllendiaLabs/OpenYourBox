#include "UserBoxLibraryPanel.h"

#include "../graph/BoxFlowOrder.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
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

/**
 * @brief Returns true for editor-only Group Input/Output snapshot nodes.
 * @param type Serialized node type name.
 */
bool isSnapshotBoundaryType(const juce::String &type) {
  return type == "group_input" || type == "group_output";
}

/**
 * @brief Collects node/group ids in the snapshot subtree rooted at @p rootId.
 * @param snapshot Loaded box snapshot.
 * @param rootId Node or group identifier.
 * @param nodeIds Destination node identifiers.
 * @param groupIds Destination group identifiers.
 */
void collectSnapshotSubtree(const juce::ValueTree &snapshot, std::int32_t rootId,
                            std::unordered_set<std::int32_t> &nodeIds,
                            std::unordered_set<std::int32_t> &groupIds);

/**
 * @brief Collects node ids in the snapshot subtree rooted at @p rootId.
 * @param snapshot Loaded box snapshot.
 * @param rootId Node or group identifier.
 * @param nodeIds Destination node identifiers.
 */
void collectSnapshotNodes(const juce::ValueTree &snapshot, std::int32_t rootId,
                          std::unordered_set<std::int32_t> &nodeIds) {
  std::unordered_set<std::int32_t> groupIds;
  collectSnapshotSubtree(snapshot, rootId, nodeIds, groupIds);
}

/**
 * @brief Maps a snapshot node to the sibling box that owns it in @p candidates.
 * @param parentGroupId Parent group keyed by node or nested group id.
 * @param candidates Displayed sibling node and group identifiers.
 * @param nodeId Endpoint node resolved from a link pin.
 */
std::optional<std::int32_t> snapshotOwningCandidateBox(
    const std::unordered_map<std::int32_t, std::optional<std::int32_t>>
        &parentGroupId,
    const std::unordered_set<std::int32_t> &candidates, std::int32_t nodeId) {
  if (candidates.count(nodeId) != 0)
    return nodeId;
  auto found = parentGroupId.find(nodeId);
  if (found == parentGroupId.end())
    return std::nullopt;

  auto groupId = found->second;
  std::unordered_set<std::int32_t> visiting;
  while (groupId.has_value()) {
    if (candidates.count(*groupId) != 0)
      return *groupId;
    const auto parent = parentGroupId.find(*groupId);
    if (parent == parentGroupId.end() || !visiting.insert(*groupId).second)
      break;
    groupId = parent->second;
  }
  return std::nullopt;
}

/**
 * @brief Returns snapshot boxes sorted by information-flow rank, then name.
 *
 * Rank is the longest path from sources in the sibling-link DAG; feedback
 * loops share a rank. Within a rank, order is case-insensitive name, then id.
 * Unconnected boxes sit with sources (rank 0).
 * @param snapshot Loaded box snapshot.
 * @param scopeGroupId Parent group whose members are being listed.
 * @param boxIds Candidate sibling node and group identifiers.
 */
std::vector<std::int32_t>
orderSnapshotBoxesByFlow(const juce::ValueTree &snapshot,
                         std::int32_t scopeGroupId,
                         std::vector<std::int32_t> boxIds) {
  struct SnapshotBox {
    juce::String label;
    bool isBoundary = false;
  };
  std::unordered_map<std::int32_t, SnapshotBox> boxes;
  std::unordered_map<std::int32_t, std::optional<std::int32_t>> parentGroupId;
  for (const auto child : snapshot) {
    if (child.hasType("Node")) {
      const auto id = static_cast<std::int32_t>(child["id"]);
      SnapshotBox box;
      box.label = child["label"].toString();
      box.isBoundary = isSnapshotBoundaryType(child["type"].toString());
      boxes.emplace(id, std::move(box));
      if (child.hasProperty("parentGroupId"))
        parentGroupId.emplace(
            id, static_cast<std::int32_t>(child.getProperty("parentGroupId")));
      else
        parentGroupId.emplace(id, std::nullopt);
    } else if (child.hasType("Group")) {
      const auto id = static_cast<std::int32_t>(child["id"]);
      SnapshotBox box;
      box.label = child.getProperty("name", "Group").toString();
      boxes.emplace(id, std::move(box));
      if (child.hasProperty("parentGroupId"))
        parentGroupId.emplace(
            id, static_cast<std::int32_t>(child.getProperty("parentGroupId")));
      else
        parentGroupId.emplace(id, std::nullopt);
    }
  }

  boxIds.erase(std::remove_if(boxIds.begin(), boxIds.end(),
                              [&boxes](std::int32_t boxId) {
                                const auto found = boxes.find(boxId);
                                return found == boxes.end() ||
                                       found->second.isBoundary;
                              }),
               boxIds.end());
  if (boxIds.empty())
    return boxIds;

  const std::unordered_set<std::int32_t> candidates(boxIds.begin(),
                                                    boxIds.end());
  std::unordered_set<std::int32_t> flowNodes;
  for (const auto boxId : boxIds)
    collectSnapshotNodes(snapshot, boxId, flowNodes);
  for (const auto &[nodeId, parent] : parentGroupId) {
    if (parent.has_value() && *parent == scopeGroupId)
      flowNodes.insert(nodeId);
  }

  std::unordered_map<std::int32_t, std::int32_t> pinOwners;
  for (const auto child : snapshot) {
    if (!child.hasType("Node"))
      continue;
    const auto nodeId = static_cast<std::int32_t>(child["id"]);
    for (const auto pin : child) {
      if (pin.hasType("Pin"))
        pinOwners.emplace(static_cast<std::int32_t>(pin["id"]), nodeId);
    }
  }

  std::vector<openyourbox::graph::BoxFlowEdge> edges;
  for (const auto child : snapshot) {
    if (!child.hasType("Link"))
      continue;
    const auto sourcePin =
        static_cast<std::int32_t>(child.getProperty("sourcePin"));
    const auto destinationPin =
        static_cast<std::int32_t>(child.getProperty("destinationPin"));
    const auto source = pinOwners.find(sourcePin);
    const auto destination = pinOwners.find(destinationPin);
    if (source == pinOwners.end() || destination == pinOwners.end())
      continue;
    if (flowNodes.count(source->second) == 0 ||
        flowNodes.count(destination->second) == 0)
      continue;
    const auto sourceBox =
        snapshotOwningCandidateBox(parentGroupId, candidates, source->second);
    const auto destinationBox = snapshotOwningCandidateBox(
        parentGroupId, candidates, destination->second);
    if (!sourceBox.has_value() || !destinationBox.has_value() ||
        *sourceBox == *destinationBox)
      continue;
    edges.push_back({*sourceBox, *destinationBox});
  }

  std::unordered_map<std::int32_t, juce::String> names;
  names.reserve(boxIds.size());
  for (const auto boxId : boxIds)
    names.emplace(boxId, boxes.at(boxId).label);
  return openyourbox::graph::orderBoxesByFlowRank(std::move(boxIds), names,
                                                  edges);
}

void collectSnapshotSubtree(const juce::ValueTree &snapshot, std::int32_t rootId,
                            std::unordered_set<std::int32_t> &nodeIds,
                            std::unordered_set<std::int32_t> &groupIds) {
  std::unordered_map<std::int32_t, juce::ValueTree> groupsById;
  std::unordered_set<std::int32_t> nodesById;
  for (const auto child : snapshot) {
    if (child.hasType("Node"))
      nodesById.insert(static_cast<std::int32_t>(child["id"]));
    else if (child.hasType("Group"))
      groupsById.emplace(static_cast<std::int32_t>(child["id"]), child);
  }
  std::function<void(std::int32_t)> walk = [&](std::int32_t id) {
    if (nodesById.count(id) != 0) {
      nodeIds.insert(id);
      return;
    }
    const auto found = groupsById.find(id);
    if (found == groupsById.end() || !groupIds.insert(id).second)
      return;
    for (const auto member : found->second) {
      if (!member.hasType("Member"))
        continue;
      walk(static_cast<std::int32_t>(member["id"]));
    }
  };
  walk(rootId);
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
  std::unordered_map<std::int32_t, juce::ValueTree> nodes;
  for (const auto child : snapshot) {
    if (child.hasType("Group"))
      groups.emplace(static_cast<std::int32_t>(child["id"]), child);
    else if (child.hasType("Node") &&
             !isSnapshotBoundaryType(child["type"].toString()))
      nodes.emplace(static_cast<std::int32_t>(child["id"]), child);
  }
  const auto found = groups.find(rootId);
  if (found == groups.end())
    return;

  std::vector<std::int32_t> memberIds;
  for (const auto member : found->second) {
    if (!member.hasType("Member"))
      continue;
    memberIds.push_back(static_cast<std::int32_t>(member["id"]));
  }
  memberIds = orderSnapshotBoxesByFlow(snapshot, rootId, std::move(memberIds));

  std::vector<MemberRow> rows;
  rows.reserve(memberIds.size());
  for (const auto memberId : memberIds) {
    MemberRow row;
    row.id = memberId;
    const auto group = groups.find(memberId);
    if (group != groups.end()) {
      row.isGroup = true;
      row.label = group->second.getProperty("name", "Group").toString();
    } else {
      const auto node = nodes.find(memberId);
      if (node == nodes.end())
        continue;
      row.label = node->second["label"].toString();
    }
    rows.push_back(std::move(row));
  }
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
