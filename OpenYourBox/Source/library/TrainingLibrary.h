#pragma once

#include <JuceHeader.h>

#include <optional>
#include <string>
#include <vector>

namespace openyourbox::library {
/** @brief How a training-library pair was ingested. */
enum class PairSource {
  /** @brief Dual-instance Capture Samples recording. */
  capture,
  /** @brief File import of a clean/processed pair. */
  imported
};

/** @brief Library entry shape: aligned pair or unpaired clip. */
enum class LibraryEntryKind {
  /** @brief Clean/processed pair. */
  pair,
  /** @brief Single unpaired clip. */
  clip
};

/**
 * @struct TrainingLibraryEntry
 * @brief One aligned clean/processed pair stored in the training library.
 */
struct TrainingLibraryEntry {
  /** @brief Stable pair identifier used as the on-disk stem. */
  juce::String id;
  /** @brief User-visible name. */
  juce::String displayName;
  /** @brief ISO-8601 creation timestamp. */
  juce::String createdAt;
  /** @brief Capture or import origin. */
  PairSource source = PairSource::imported;
  /** @brief Pair vs unpaired clip. */
  LibraryEntryKind kind = LibraryEntryKind::pair;
  /** @brief Aligned duration in seconds. */
  double durationSeconds = 0.0;
  /** @brief Shared sample rate in Hz. */
  double sampleRate = 0.0;
  /** @brief Channel count after import/capture alignment. */
  int channels = 0;
  /** @brief Absolute path of the clean (x) audio file. */
  juce::String xPath;
  /** @brief Absolute path of the processed (y) audio file. */
  juce::String yPath;
  /** @brief Combined byte size of owned audio files. */
  juce::int64 byteSize = 0;
  /** @brief Optional free-text notes reserved for later UI. */
  juce::String notes;
  /** @brief Optional tags reserved for later UI. */
  juce::StringArray tags;
  /** @brief Whether this pair is included in the next Train run. */
  bool selectedForTrain = false;
};

/**
 * @class TrainingLibrary
 * @brief Persisted master-owned store of sample pairs under plugin user data.
 */
class TrainingLibrary {
public:
  /** @brief Creates an empty library bound to the default user-data folder. */
  TrainingLibrary();

  /**
   * @brief Creates a library rooted at an explicit directory (tests).
   * @param rootDirectory Directory that will contain `index.json` and pair files.
   */
  explicit TrainingLibrary(juce::File rootDirectory);

  /** @brief Returns the on-disk library root. */
  [[nodiscard]] juce::File getRootDirectory() const;

  /** @brief Reloads `index.json` from disk when present. */
  void load();

  /** @brief Writes `index.json` atomically. */
  bool save() const;

  /** @brief Returns all library entries in insertion order. */
  [[nodiscard]] const std::vector<TrainingLibraryEntry> &getEntries() const noexcept;

  /** @brief Returns identifiers currently selected for Train. */
  [[nodiscard]] juce::StringArray getSelectedPairIds() const;

  /**
   * @brief Finds an entry by identifier.
   * @param id Pair identifier.
   * @return Mutable entry, or null when absent.
   */
  [[nodiscard]] TrainingLibraryEntry *findEntry(const juce::String &id) noexcept;

  /**
   * @brief Finds an entry by identifier.
   * @param id Pair identifier.
   * @return Immutable entry, or null when absent.
   */
  [[nodiscard]] const TrainingLibraryEntry *findEntry(const juce::String &id) const
      noexcept;

  /**
   * @brief Sets the Train selection flag for one entry.
   * @param id Pair identifier.
   * @param selected Whether the pair participates in the next Train.
   * @return True when the entry exists.
   */
  bool setSelected(const juce::String &id, bool selected);

  /** @brief Selects every entry for Train. */
  void selectAll();

  /** @brief Clears the Train selection. */
  void selectNone();

  /**
   * @brief Renames one entry.
   * @param id Pair identifier.
   * @param displayName New user-visible name.
   * @return True when the entry exists.
   */
  bool rename(const juce::String &id, const juce::String &displayName);

  /**
   * @brief Deletes an owned pair and its audio files after the caller confirms.
   * @param id Pair identifier.
   * @return True when the entry was removed.
   */
  bool removeEntry(const juce::String &id);

  /**
   * @brief Imports a clean/processed file pair, aligning/cropping to the shorter.
   * @param cleanFile Source x audio.
   * @param processedFile Source y audio.
   * @param error Receives a user-facing failure.
   * @return The new entry, or no value on failure.
   */
  std::optional<TrainingLibraryEntry> importPair(const juce::File &cleanFile,
                                                 const juce::File &processedFile,
                                                 juce::String &error);

  /**
   * @brief Appends a captured pair already written as WAV files.
   * @param displayName User-visible name.
   * @param xFile Clean recording.
   * @param yFile Processed recording.
   * @param sampleRate Shared sample rate.
   * @param channels Shared channel count.
   * @param durationSeconds Aligned duration.
   * @param error Receives a user-facing failure.
   * @return The new entry, or no value on failure.
   */
  std::optional<TrainingLibraryEntry>
  addCapturedPair(const juce::String &displayName, const juce::File &xFile,
                  const juce::File &yFile, double sampleRate, int channels,
                  double durationSeconds, juce::String &error);

  /**
   * @brief Imports one unpaired clip as a `clip` library entry.
   * @param audioFile Source audio.
   * @param error Receives a user-facing failure.
   * @return The new entry, or no value on failure.
   */
  std::optional<TrainingLibraryEntry> importClip(const juce::File &audioFile,
                                                 juce::String &error);

  /**
   * @brief Appends a captured unpaired clip already written as a WAV file.
   */
  std::optional<TrainingLibraryEntry>
  addCapturedClip(const juce::String &displayName, const juce::File &audioFile,
                  double sampleRate, int channels, double durationSeconds,
                  juce::String &error);

  /**
   * @brief Returns true when any selected entry is an unpaired clip.
   */
  [[nodiscard]] bool selectedContainsUnpaired() const noexcept;

  /**
   * @brief Returns true when selected entries share one sample rate.
   * @param mixedRateMessage Receives a user-facing block reason when mixed.
   * @return True when at least one pair is selected and rates match.
   */
  [[nodiscard]] bool selectedSampleRatesMatch(juce::String &mixedRateMessage) const;

  /** @brief Returns the number of selected pairs. */
  [[nodiscard]] int getSelectedCount() const noexcept;

  /** @brief Returns total selected duration in seconds. */
  [[nodiscard]] double getSelectedDurationSeconds() const noexcept;

private:
  /**
   * @brief Copies and length-aligns two audio files into the library folder.
   * @param cleanFile Source x.
   * @param processedFile Source y.
   * @param pairId Destination stem.
   * @param entry Receives metadata for the written files.
   * @param error Receives a user-facing failure.
   * @return True when both WAV files were written.
   */
  bool writeAlignedPair(const juce::File &cleanFile, const juce::File &processedFile,
                        const juce::String &pairId, TrainingLibraryEntry &entry,
                        juce::String &error) const;

  /** @brief Root directory containing index.json and pair audio. */
  juce::File root;
  /** @brief In-memory library entries. */
  std::vector<TrainingLibraryEntry> entries;
};
} // namespace openyourbox::library
