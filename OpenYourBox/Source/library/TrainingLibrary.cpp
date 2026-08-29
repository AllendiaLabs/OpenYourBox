#include "TrainingLibrary.h"
#include "UserDataPaths.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace openyourbox::library {
namespace {

/**
 * @brief Converts a pair source enum to a persisted string.
 * @param source Capture or import origin.
 * @return Stable JSON token.
 */
juce::String sourceName(PairSource source) {
  return source == PairSource::capture ? "capture" : "import";
}

/**
 * @brief Parses a persisted source token.
 * @param name JSON token.
 * @return Capture when the token matches, otherwise import.
 */
PairSource sourceFromName(const juce::String &name) {
  return name == "capture" ? PairSource::capture : PairSource::imported;
}

/**
 * @brief Opens a decoder for an imported audio file.
 * @param file Source audio.
 * @param error Receives a user-facing failure.
 * @return A reader, or null when the file cannot be decoded.
 */
std::unique_ptr<juce::AudioFormatReader>
openAudioReader(const juce::File &file, juce::String &error) {
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr)
    error = "Could not decode " + file.getFileName();
  return reader;
}

/**
 * @brief Streams a sample range from a decoder into a 32-bit float WAV.
 *
 * Copies the full requested range in reader-sized blocks so hour-long files
 * are not truncated by in-memory AudioBuffer limits.
 *
 * @param reader Source decoder.
 * @param destination Output WAV path.
 * @param channels Channel count to write.
 * @param numSamples Number of frames to copy.
 * @param error Receives a user-facing failure.
 * @return True when the WAV was written.
 */
bool streamCopyToWav(juce::AudioFormatReader &reader, const juce::File &destination,
                     int channels, juce::int64 numSamples, juce::String &error) {
  if (channels < 1 || numSamples < 1) {
    error = destination.getFileName() + " contains no audio";
    return false;
  }
  destination.getParentDirectory().createDirectory();
  destination.deleteFile();
  auto stream = destination.createOutputStream();
  if (stream == nullptr) {
    error = "Could not create " + destination.getFileName();
    return false;
  }
  juce::WavAudioFormat wav;
  auto options =
      juce::AudioFormatWriterOptions{}
          .withSampleRate(reader.sampleRate)
          .withNumChannels(channels)
          .withBitsPerSample(32)
          .withSampleFormat(
              juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
  std::unique_ptr<juce::OutputStream> output(std::move(stream));
  auto writer = wav.createWriterFor(output, options);
  if (writer == nullptr) {
    error = "Could not write " + destination.getFileName();
    return false;
  }
  if (!writer->writeFromAudioReader(reader, 0, numSamples)) {
    error = "Failed while writing " + destination.getFileName();
    return false;
  }
  return true;
}
} // namespace

TrainingLibrary::TrainingLibrary()
    : TrainingLibrary(samplesDirectory()) {}

TrainingLibrary::TrainingLibrary(juce::File rootDirectory)
    : root(std::move(rootDirectory)) {
  root.createDirectory();
  load();
}

juce::File TrainingLibrary::getRootDirectory() const { return root; }

void TrainingLibrary::load() {
  entries.clear();
  const auto indexFile = root.getChildFile("index.json");
  if (!indexFile.existsAsFile())
    return;
  const auto parsed = juce::JSON::parse(indexFile.loadFileAsString());
  if (!parsed.isObject())
    return;
  const auto list = parsed.getProperty("entries", juce::var());
  if (auto *array = list.getArray()) {
    for (const auto &item : *array) {
      if (!item.isObject())
        continue;
      TrainingLibraryEntry entry;
      entry.id = item.getProperty("id", {}).toString();
      entry.displayName = item.getProperty("displayName", {}).toString();
      entry.createdAt = item.getProperty("createdAt", {}).toString();
      entry.source = sourceFromName(item.getProperty("source", {}).toString());
      entry.durationSeconds =
          static_cast<double>(item.getProperty("durationSeconds", 0.0));
      entry.sampleRate = static_cast<double>(item.getProperty("sampleRate", 0.0));
      entry.channels = static_cast<int>(item.getProperty("channels", 0));
      entry.xPath = item.getProperty("xPath", {}).toString();
      entry.yPath = item.getProperty("yPath", {}).toString();
      entry.byteSize =
          static_cast<juce::int64>(item.getProperty("byteSize", 0));
      entry.notes = item.getProperty("notes", {}).toString();
      entry.selectedForTrain =
          static_cast<bool>(item.getProperty("selectedForTrain", false));
      const auto kindName = item.getProperty("kind", "pair").toString();
      entry.kind = kindName == "clip" ? LibraryEntryKind::clip
                                      : LibraryEntryKind::pair;
      const auto tags = item.getProperty("tags", juce::var());
      if (auto *tagArray = tags.getArray()) {
        for (const auto &tag : *tagArray)
          entry.tags.add(tag.toString());
      }
      if (entry.id.isNotEmpty()) {
        if (!entry.tags.contains(entry.kind == LibraryEntryKind::clip
                                     ? "unpaired"
                                     : "pair"))
          entry.tags.add(entry.kind == LibraryEntryKind::clip ? "unpaired"
                                                              : "pair");
        entries.push_back(std::move(entry));
      }
    }
  }
  const auto selected = parsed.getProperty("selectedPairIds", juce::var());
  if (auto *ids = selected.getArray()) {
    for (auto &entry : entries)
      entry.selectedForTrain = false;
    for (const auto &id : *ids) {
      if (auto *entry = findEntry(id.toString()))
        entry->selectedForTrain = true;
    }
  }
}

bool TrainingLibrary::save() const {
  root.createDirectory();
  auto rootObject = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> serialized;
  juce::Array<juce::var> selectedIds;
  for (const auto &entry : entries) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", entry.id);
    object->setProperty("displayName", entry.displayName);
    object->setProperty("createdAt", entry.createdAt);
    object->setProperty("source", sourceName(entry.source));
    object->setProperty("durationSeconds", entry.durationSeconds);
    object->setProperty("sampleRate", entry.sampleRate);
    object->setProperty("channels", entry.channels);
    object->setProperty("xPath", entry.xPath);
    object->setProperty("yPath", entry.yPath);
    object->setProperty("byteSize", entry.byteSize);
    object->setProperty("notes", entry.notes);
    object->setProperty("selectedForTrain", entry.selectedForTrain);
    object->setProperty("kind",
                        entry.kind == LibraryEntryKind::clip ? "clip" : "pair");
    juce::Array<juce::var> tags;
    for (const auto &tag : entry.tags)
      tags.add(tag);
    object->setProperty("tags", tags);
    serialized.add(juce::var(object.release()));
    if (entry.selectedForTrain)
      selectedIds.add(entry.id);
  }
  rootObject->setProperty("entries", serialized);
  rootObject->setProperty("selectedPairIds", selectedIds);
  const auto indexFile = root.getChildFile("index.json");
  const auto temporary = indexFile.withFileExtension(".json.tmp");
  if (!temporary.replaceWithText(
          juce::JSON::toString(juce::var(rootObject.release()), true)))
    return false;
  return temporary.moveFileTo(indexFile) ||
         (indexFile.deleteFile() && temporary.moveFileTo(indexFile));
}

const std::vector<TrainingLibraryEntry> &
TrainingLibrary::getEntries() const noexcept {
  return entries;
}

juce::StringArray TrainingLibrary::getSelectedPairIds() const {
  juce::StringArray ids;
  for (const auto &entry : entries) {
    if (entry.selectedForTrain)
      ids.add(entry.id);
  }
  return ids;
}

TrainingLibraryEntry *
TrainingLibrary::findEntry(const juce::String &id) noexcept {
  const auto found =
      std::find_if(entries.begin(), entries.end(),
                   [&id](const TrainingLibraryEntry &entry) {
                     return entry.id == id;
                   });
  return found == entries.end() ? nullptr : &*found;
}

const TrainingLibraryEntry *
TrainingLibrary::findEntry(const juce::String &id) const noexcept {
  const auto found =
      std::find_if(entries.begin(), entries.end(),
                   [&id](const TrainingLibraryEntry &entry) {
                     return entry.id == id;
                   });
  return found == entries.end() ? nullptr : &*found;
}

bool TrainingLibrary::setSelected(const juce::String &id, bool selected) {
  auto *entry = findEntry(id);
  if (entry == nullptr)
    return false;
  entry->selectedForTrain = selected;
  return save();
}

void TrainingLibrary::selectAll() {
  for (auto &entry : entries)
    entry.selectedForTrain = true;
  save();
}

void TrainingLibrary::selectNone() {
  for (auto &entry : entries)
    entry.selectedForTrain = false;
  save();
}

bool TrainingLibrary::rename(const juce::String &id,
                             const juce::String &displayName) {
  auto *entry = findEntry(id);
  if (entry == nullptr || displayName.trim().isEmpty())
    return false;
  entry->displayName = displayName.trim();
  return save();
}

bool TrainingLibrary::removeEntry(const juce::String &id) {
  auto *entry = findEntry(id);
  if (entry == nullptr)
    return false;
  juce::File(entry->xPath).deleteFile();
  if (entry->yPath.isNotEmpty())
    juce::File(entry->yPath).deleteFile();
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&id](const TrainingLibraryEntry &candidate) {
                                 return candidate.id == id;
                               }),
                entries.end());
  return save();
}

bool TrainingLibrary::writeAlignedPair(const juce::File &cleanFile,
                                       const juce::File &processedFile,
                                       const juce::String &pairId,
                                       TrainingLibraryEntry &entry,
                                       juce::String &error) const {
  auto clean = openAudioReader(cleanFile, error);
  if (clean == nullptr)
    return false;
  auto processed = openAudioReader(processedFile, error);
  if (processed == nullptr)
    return false;
  if (std::abs(clean->sampleRate - processed->sampleRate) > 0.5) {
    error = "Imported files must share the same sample rate";
    return false;
  }
  const auto channels = std::min(static_cast<int>(clean->numChannels),
                                 static_cast<int>(processed->numChannels));
  const auto samples =
      std::min(clean->lengthInSamples, processed->lengthInSamples);
  if (channels < 1 || samples < 1) {
    error = "Imported files have no overlapping audio to align";
    return false;
  }
  const auto xFile = root.getChildFile(pairId + "_x.wav");
  const auto yFile = root.getChildFile(pairId + "_y.wav");
  if (!streamCopyToWav(*clean, xFile, channels, samples, error) ||
      !streamCopyToWav(*processed, yFile, channels, samples, error))
    return false;
  entry.xPath = xFile.getFullPathName();
  entry.yPath = yFile.getFullPathName();
  entry.sampleRate = clean->sampleRate;
  entry.channels = channels;
  entry.durationSeconds =
      clean->sampleRate > 0.0
          ? static_cast<double>(samples) / clean->sampleRate
          : 0.0;
  entry.byteSize = xFile.getSize() + yFile.getSize();
  return true;
}

std::optional<TrainingLibraryEntry>
TrainingLibrary::importPair(const juce::File &cleanFile,
                            const juce::File &processedFile,
                            juce::String &error) {
  TrainingLibraryEntry entry;
  entry.id = juce::Uuid().toDashedString();
  entry.source = PairSource::imported;
  entry.createdAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.displayName = cleanFile.getFileNameWithoutExtension() + " / " +
                      processedFile.getFileNameWithoutExtension();
  entry.selectedForTrain = true;
  entry.kind = LibraryEntryKind::pair;
  entry.tags.add("pair");
  if (!writeAlignedPair(cleanFile, processedFile, entry.id, entry, error))
    return std::nullopt;
  entries.push_back(entry);
  if (!save()) {
    error = "Could not persist the training library index";
    return std::nullopt;
  }
  return entry;
}

std::optional<TrainingLibraryEntry> TrainingLibrary::addCapturedPair(
    const juce::String &displayName, const juce::File &xFile,
    const juce::File &yFile, double sampleRate, int channels,
    double durationSeconds, juce::String &error) {
  if (!xFile.existsAsFile() || !yFile.existsAsFile()) {
    error = "Capture files were not written";
    return std::nullopt;
  }
  TrainingLibraryEntry entry;
  entry.id = juce::Uuid().toDashedString();
  entry.source = PairSource::capture;
  entry.createdAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.displayName =
      displayName.isNotEmpty()
          ? displayName
          : juce::String("Capture ") +
                juce::Time::getCurrentTime().formatted("%Y-%m-%d %H:%M");
  entry.sampleRate = sampleRate;
  entry.channels = channels;
  entry.durationSeconds = durationSeconds;
  const auto destX = root.getChildFile(entry.id + "_x.wav");
  const auto destY = root.getChildFile(entry.id + "_y.wav");
  if (!xFile.copyFileTo(destX) || !yFile.copyFileTo(destY)) {
    error = "Could not copy captured audio into the library";
    return std::nullopt;
  }
  entry.xPath = destX.getFullPathName();
  entry.yPath = destY.getFullPathName();
  entry.byteSize = destX.getSize() + destY.getSize();
  entry.selectedForTrain = true;
  entry.kind = LibraryEntryKind::pair;
  entry.tags.add("pair");
  entries.push_back(entry);
  if (!save()) {
    error = "Could not persist the training library index";
    return std::nullopt;
  }
  return entry;
}

bool TrainingLibrary::selectedSampleRatesMatch(
    juce::String &mixedRateMessage) const {
  mixedRateMessage.clear();
  double rate = 0.0;
  int selected = 0;
  for (const auto &entry : entries) {
    if (!entry.selectedForTrain)
      continue;
    ++selected;
    if (rate == 0.0) {
      rate = entry.sampleRate;
      continue;
    }
    if (std::abs(rate - entry.sampleRate) > 0.5) {
      mixedRateMessage =
          "Selected library pairs mix sample rates. Deselect mismatched "
          "pairs or import files that share one sample rate.";
      return false;
    }
  }
  if (selected < 1) {
    mixedRateMessage = "Select at least one library pair before Train.";
    return false;
  }
  return true;
}

int TrainingLibrary::getSelectedCount() const noexcept {
  int count = 0;
  for (const auto &entry : entries) {
    if (entry.selectedForTrain)
      ++count;
  }
  return count;
}

double TrainingLibrary::getSelectedDurationSeconds() const noexcept {
  double total = 0.0;
  for (const auto &entry : entries) {
    if (entry.selectedForTrain)
      total += entry.durationSeconds;
  }
  return total;
}

bool TrainingLibrary::selectedContainsUnpaired() const noexcept {
  for (const auto &entry : entries) {
    if (entry.selectedForTrain && entry.kind == LibraryEntryKind::clip)
      return true;
  }
  return false;
}

std::optional<TrainingLibraryEntry>
TrainingLibrary::importClip(const juce::File &audioFile, juce::String &error) {
  auto reader = openAudioReader(audioFile, error);
  if (reader == nullptr)
    return std::nullopt;
  const auto samples = reader->lengthInSamples;
  const auto channels = static_cast<int>(reader->numChannels);
  if (samples < 1 || channels < 1) {
    error = audioFile.getFileName() + " contains no audio";
    return std::nullopt;
  }
  TrainingLibraryEntry entry;
  entry.id = juce::Uuid().toDashedString();
  entry.source = PairSource::imported;
  entry.kind = LibraryEntryKind::clip;
  entry.createdAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.displayName = audioFile.getFileNameWithoutExtension();
  entry.selectedForTrain = true;
  entry.tags.add("unpaired");
  const auto dest = root.getChildFile(entry.id + "_clip.wav");
  if (!streamCopyToWav(*reader, dest, channels, samples, error))
    return std::nullopt;
  entry.xPath = dest.getFullPathName();
  entry.sampleRate = reader->sampleRate;
  entry.channels = channels;
  entry.durationSeconds =
      reader->sampleRate > 0.0
          ? static_cast<double>(samples) / reader->sampleRate
          : 0.0;
  entry.byteSize = dest.getSize();
  entries.push_back(entry);
  if (!save()) {
    error = "Could not persist the training library index";
    return std::nullopt;
  }
  return entry;
}

std::optional<TrainingLibraryEntry> TrainingLibrary::addCapturedClip(
    const juce::String &displayName, const juce::File &audioFile,
    double sampleRate, int channels, double durationSeconds,
    juce::String &error) {
  if (!audioFile.existsAsFile()) {
    error = "Capture file was not written";
    return std::nullopt;
  }
  TrainingLibraryEntry entry;
  entry.id = juce::Uuid().toDashedString();
  entry.source = PairSource::capture;
  entry.kind = LibraryEntryKind::clip;
  entry.createdAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.displayName =
      displayName.isNotEmpty()
          ? displayName
          : juce::String("Clip ") +
                juce::Time::getCurrentTime().formatted("%Y-%m-%d %H:%M");
  entry.sampleRate = sampleRate;
  entry.channels = channels;
  entry.durationSeconds = durationSeconds;
  entry.tags.add("unpaired");
  const auto dest = root.getChildFile(entry.id + "_clip.wav");
  if (!audioFile.copyFileTo(dest)) {
    error = "Could not copy captured audio into the library";
    return std::nullopt;
  }
  entry.xPath = dest.getFullPathName();
  entry.byteSize = dest.getSize();
  entry.selectedForTrain = true;
  entries.push_back(entry);
  if (!save()) {
    error = "Could not persist the training library index";
    return std::nullopt;
  }
  return entry;
}
} // namespace openyourbox::library
