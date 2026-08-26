#pragma once

#include "../state/PatchSnapshot.h"

#include <JuceHeader.h>

#include <optional>
#include <vector>

namespace openyourbox::library {
/**
 * @struct UserPresetEntry
 * @brief One named full-patch catalog row persisted under UserPresets.
 */
struct UserPresetEntry {
  /** @brief Stable UUID for the catalog row and payload folder. */
  juce::String id;
  /** @brief Unique display name within the catalog. */
  juce::String name;
  /** @brief ISO-8601 creation timestamp. */
  juce::String createdAt;
  /** @brief ISO-8601 last-write timestamp. */
  juce::String updatedAt;
  /** @brief Payload directory relative to the presets root. */
  juce::String payloadRelativePath;
  /** @brief Entry metadata version. */
  int schemaVersion = 1;
};

/**
 * @class UserPresetLibrary
 * @brief Persisted catalog of full sonic presets, distinct from the box library.
 */
class UserPresetLibrary {
public:
  /** @brief Creates a library bound to the default UserPresets folder. */
  UserPresetLibrary();

  /**
   * @brief Creates a library rooted at an explicit directory (tests).
   * @param rootDirectory Directory that will contain `index.json` and entries.
   */
  explicit UserPresetLibrary(juce::File rootDirectory);

  /** @brief Returns the on-disk library root. */
  [[nodiscard]] juce::File getRootDirectory() const;

  /** @brief Reloads `index.json` from disk when present. */
  void load();

  /** @brief Writes `index.json` atomically. */
  bool save() const;

  /** @brief Returns catalog rows in insertion order. */
  [[nodiscard]] const std::vector<UserPresetEntry> &getEntries() const noexcept;

  /**
   * @brief Finds a catalog row by identifier.
   * @param id Entry UUID.
   */
  [[nodiscard]] UserPresetEntry *findEntry(const juce::String &id) noexcept;

  /**
   * @brief Finds a catalog row by identifier.
   * @param id Entry UUID.
   */
  [[nodiscard]] const UserPresetEntry *
  findEntry(const juce::String &id) const noexcept;

  /**
   * @brief Finds a catalog row by unique display name.
   * @param name User-visible name.
   */
  [[nodiscard]] const UserPresetEntry *
  findEntryByName(const juce::String &name) const noexcept;

  /**
   * @brief Saves a snapshot as a new or overwritten named entry (Save As).
   * @param snapshot Live patch to persist.
   * @param name Unique display name.
   * @param overwrite True to replace an existing name after the caller confirms.
   * @param error Receives a user-facing failure.
   */
  std::optional<UserPresetEntry> saveAs(state::PatchSnapshot snapshot,
                                        const juce::String &name, bool overwrite,
                                        juce::String &error);

  /**
   * @brief Overwrites an existing catalog entry's payload (Save).
   * @param id Catalog UUID of the current preset.
   * @param snapshot Live patch to persist.
   * @param error Receives a user-facing failure.
   */
  std::optional<UserPresetEntry> saveOverwrite(const juce::String &id,
                                               state::PatchSnapshot snapshot,
                                               juce::String &error);

  /**
   * @brief Loads a catalog entry into a snapshot for apply.
   * @param id Catalog UUID.
   * @param error Receives a user-facing failure.
   * @return Snapshot with project-local artifact paths, or no value on failure.
   */
  std::optional<state::PatchSnapshot> loadSnapshot(const juce::String &id,
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
   * @brief Deletes one catalog row and its payload folder.
   * @param id Entry UUID.
   */
  bool removeEntry(const juce::String &id);

private:
  /**
   * @brief Writes snapshot XML plus copied artifacts into an entry folder.
   * @param snapshot Mutable snapshot whose graph paths may be rewritten.
   * @param payloadDirectory Entry folder to populate.
   * @param error Receives a user-facing failure.
   */
  bool writePayload(state::PatchSnapshot &snapshot,
                    const juce::File &payloadDirectory,
                    juce::String &error) const;

  /**
   * @brief Reads patch.xml and copies artifacts into the project weights folder.
   * @param payloadDirectory Entry folder.
   * @param error Receives a user-facing failure.
   */
  std::optional<state::PatchSnapshot>
  readPayload(const juce::File &payloadDirectory, juce::String &error) const;

  /**
   * @brief Copies snapshot artifacts into an entry folder and rewrites paths.
   * @param snapshot Mutable snapshot.
   * @param artifactsDirectory Destination `artifacts/` folder.
   * @param error Receives a user-facing failure.
   */
  bool copyArtifactsIntoEntry(state::PatchSnapshot &snapshot,
                              const juce::File &artifactsDirectory,
                              juce::String &error) const;

  /**
   * @brief Copies entry artifacts into the project weights folder.
   * @param snapshot Mutable snapshot.
   * @param payloadDirectory Entry payload folder.
   * @param error Receives a user-facing failure.
   */
  bool copyArtifactsIntoProject(state::PatchSnapshot &snapshot,
                                const juce::File &payloadDirectory,
                                juce::String &error) const;

  /**
   * @brief Inserts or replaces a catalog row and writes the index.
   * @param entry Row to store.
   * @param error Receives a user-facing failure.
   */
  bool upsertEntry(const UserPresetEntry &entry, juce::String &error);

  /** @brief Root directory containing index.json and entry folders. */
  juce::File root;
  /** @brief In-memory catalog rows. */
  std::vector<UserPresetEntry> entries;
};
} // namespace openyourbox::library
