#pragma once

#include "../graph/NodeGraph.h"

#include <JuceHeader.h>

#include <optional>
#include <string>
#include <vector>

namespace openyourbox::library {
/** @brief Whether a catalog row stores one element or a group tree. */
enum class UserBoxKind {
  /** @brief Single graph element. */
  element,
  /** @brief Group with nested members and internal links. */
  group
};

/**
 * @struct UserBoxLibraryEntry
 * @brief One named box catalog row persisted under user data.
 */
struct UserBoxLibraryEntry {
  /** @brief Stable UUID for the catalog row and payload folder. */
  juce::String id;
  /** @brief Unique display name within the library. */
  juce::String name;
  /** @brief Element vs group badge. */
  UserBoxKind kind = UserBoxKind::element;
  /** @brief Folder path using `/` separators; empty is the library root. */
  juce::String folder;
  /** @brief ISO-8601 creation timestamp. */
  juce::String createdAt;
  /** @brief ISO-8601 last-write timestamp. */
  juce::String updatedAt;
  /** @brief Payload directory relative to the boxes root. */
  juce::String payloadRelativePath;
  /** @brief Element type token or `group` for list UI. */
  juce::String rootTypeHint;
  /** @brief Snapshot schema version. */
  int schemaVersion = 1;
};

/**
 * @class UserBoxLibrary
 * @brief Persisted catalog of reusable elements and groups with folders.
 */
class UserBoxLibrary {
public:
  /** @brief Creates a library bound to the default boxes user-data folder. */
  UserBoxLibrary();

  /**
   * @brief Creates a library rooted at an explicit directory (tests).
   * @param rootDirectory Directory that will contain `index.json` and entries.
   */
  explicit UserBoxLibrary(juce::File rootDirectory);

  /** @brief Returns the on-disk library root. */
  [[nodiscard]] juce::File getRootDirectory() const;

  /** @brief Reloads `index.json` from disk when present. */
  void load();

  /** @brief Writes `index.json` atomically. */
  bool save() const;

  /** @brief Returns catalog rows in insertion order. */
  [[nodiscard]] const std::vector<UserBoxLibraryEntry> &getEntries() const
      noexcept;

  /** @brief Returns persisted folder paths, including empty folders. */
  [[nodiscard]] const std::vector<juce::String> &getFolders() const noexcept;

  /**
   * @brief Finds a catalog row by identifier.
   * @param id Entry UUID.
   */
  [[nodiscard]] UserBoxLibraryEntry *findEntry(const juce::String &id) noexcept;

  /**
   * @brief Finds a catalog row by identifier.
   * @param id Entry UUID.
   */
  [[nodiscard]] const UserBoxLibraryEntry *
  findEntry(const juce::String &id) const noexcept;

  /**
   * @brief Finds a catalog row by unique display name.
   * @param name User-visible name.
   */
  [[nodiscard]] const UserBoxLibraryEntry *
  findEntryByName(const juce::String &name) const noexcept;

  /**
   * @brief Saves one element or group into the catalog.
   * @param graph Source graph.
   * @param boxId Node or group identifier.
   * @param name Unique display name.
   * @param folder Destination folder path.
   * @param overwrite True to replace an existing name after the caller confirms.
   * @param error Receives a user-facing failure.
   * @return The stored entry, or no value on failure.
   */
  std::optional<UserBoxLibraryEntry> saveBox(const graph::NodeGraph &graph,
                                             std::int32_t boxId,
                                             const juce::String &name,
                                             const juce::String &folder,
                                             bool overwrite, juce::String &error);

  /**
   * @brief Clones a catalog entry into @p graph at @p position.
   * @param graph Destination graph.
   * @param entryId Catalog UUID.
   * @param position Canvas origin for the placed root.
   * @param error Receives a user-facing failure.
   * @param nestedRootId Snapshot node/group id to insert instead of the saved
   *        root; 0 inserts the snapshot root.
   * @return New root node or group id, or no value on failure.
   */
  std::optional<std::int32_t> insertBox(graph::NodeGraph &graph,
                                        const juce::String &entryId,
                                        juce::Point<float> position,
                                        juce::String &error,
                                        std::int32_t nestedRootId = 0);

  /**
   * @brief Loads the box.xml snapshot for a catalog entry.
   * @param entryId Catalog UUID.
   * @param error Receives a user-facing failure.
   * @return Snapshot tree, or invalid on failure.
   */
  juce::ValueTree loadEntrySnapshot(const juce::String &entryId,
                                    juce::String &error) const;

  /**
   * @brief Renames one catalog row.
   * @param id Entry UUID.
   * @param name New unique display name.
   * @param error Receives a user-facing failure.
   */
  bool rename(const juce::String &id, const juce::String &name,
              juce::String &error);

  /**
   * @brief Moves one catalog row into a folder.
   * @param id Entry UUID.
   * @param folder Destination folder path.
   * @param error Receives a user-facing failure.
   */
  bool setFolder(const juce::String &id, const juce::String &folder,
                 juce::String &error);

  /**
   * @brief Deletes one catalog row and its payload folder.
   * @param id Entry UUID.
   */
  bool removeEntry(const juce::String &id);

  /**
   * @brief Creates an empty folder path.
   * @param folder Folder path to add.
   * @param error Receives a user-facing failure.
   */
  bool createFolder(const juce::String &folder, juce::String &error);

  /**
   * @brief Renames a folder and rewrites descendant paths.
   * @param folder Existing folder path.
   * @param newName New last path component.
   * @param error Receives a user-facing failure.
   */
  bool renameFolder(const juce::String &folder, const juce::String &newName,
                    juce::String &error);

  /**
   * @brief Removes a folder and optionally every contained entry.
   * @param folder Folder path to delete.
   * @param deleteContents True to delete contained boxes; false refuses if nonempty.
   * @param error Receives a user-facing failure.
   */
  bool removeFolder(const juce::String &folder, bool deleteContents,
                    juce::String &error);

private:
  /**
   * @brief Copies snapshot artifacts into an entry folder and rewrites paths.
   * @param snapshot Mutable box snapshot.
   * @param artifactsDirectory Destination `artifacts/` folder.
   * @param error Receives a user-facing failure.
   */
  bool copyArtifactsIntoEntry(juce::ValueTree &snapshot,
                              const juce::File &artifactsDirectory,
                              juce::String &error) const;

  /**
   * @brief Copies entry artifacts into the project weights folder.
   * @param snapshot Mutable box snapshot.
   * @param payloadDirectory Entry payload folder.
   * @param error Receives a user-facing failure.
   */
  bool copyArtifactsIntoProject(juce::ValueTree &snapshot,
                                const juce::File &payloadDirectory,
                                juce::String &error) const;

  /** @brief Ensures @p folder is listed even when it contains no entries. */
  void ensureFolderListed(const juce::String &folder);

  /** @brief Root directory containing index.json and entry folders. */
  juce::File root;
  /** @brief In-memory catalog rows. */
  std::vector<UserBoxLibraryEntry> entries;
  /** @brief Persisted folder paths, including empty folders. */
  std::vector<juce::String> folders;
};
} // namespace openyourbox::library
