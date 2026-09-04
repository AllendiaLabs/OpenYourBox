#pragma once

#include "UserDataPaths.h"

#include <JuceHeader.h>

#include <optional>
#include <vector>

namespace openyourbox::library {
/**
 * @struct TrainingConfigEntry
 * @brief One named training-configuration catalog row.
 */
struct TrainingConfigEntry {
  /** @brief Stable UUID. */
  juce::String id;
  /** @brief Unique display name. */
  juce::String name;
  /** @brief ISO-8601 last-write timestamp. */
  juce::String updatedAt;
  /** @brief Serialized Training Configuration object. */
  juce::var config;
};

/**
 * @class TrainingConfigLibrary
 * @brief User-level catalog of Train hyperparameters and loss schedules.
 */
class TrainingConfigLibrary {
public:
  /** @brief Creates a library bound to the default TrainingConfigs folder. */
  TrainingConfigLibrary();

  /**
   * @brief Creates a library rooted at an explicit directory (tests).
   * @param rootDirectory Directory that will contain `index.json`.
   */
  explicit TrainingConfigLibrary(juce::File rootDirectory);

  /** @brief Reloads `index.json` from disk when present. */
  void load();

  /** @brief Writes `index.json` atomically. */
  bool save() const;

  /** @brief Catalog rows in insertion order. */
  [[nodiscard]] const std::vector<TrainingConfigEntry> &
  getEntries() const noexcept;

  /**
   * @brief Finds a catalog row by identifier.
   * @param id Entry UUID.
   */
  [[nodiscard]] const TrainingConfigEntry *
  findEntry(const juce::String &id) const noexcept;

  /**
   * @brief Saves @p config under @p name (insert or replace by name).
   * @param name Display name.
   * @param config JSON object of recognized train settings.
   * @param error User-facing failure.
   */
  std::optional<TrainingConfigEntry>
  saveAs(const juce::String &name, const juce::var &config, juce::String &error);

  /**
   * @brief Renames a catalog row.
   * @param id Entry UUID.
   * @param name New unique name.
   * @param error User-facing failure.
   */
  bool rename(const juce::String &id, const juce::String &name,
              juce::String &error);

  /**
   * @brief Deletes a catalog row.
   * @param id Entry UUID.
   */
  bool removeEntry(const juce::String &id);

private:
  juce::File root;
  std::vector<TrainingConfigEntry> entries;
};
} // namespace openyourbox::library
