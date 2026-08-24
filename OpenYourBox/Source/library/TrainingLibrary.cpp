#include "TrainingLibrary.h"
#include "UserDataPaths.h"

#include <algorithm>
#include <cmath>

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
 * @brief Reads an audio file into a buffer using JUCE format managers.
 * @param file Source audio.
 * @param buffer Destination samples.
 * @param sampleRate Receives the file sample rate.
 * @param error Receives a user-facing failure.
 * @return True when the file was decoded.
 */
bool readAudioFile(const juce::File &file, juce::AudioBuffer<float> &buffer,
                   double &sampleRate, juce::String &error) {
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr) {
    error = "Could not decode " + file.getFileName();
    return false;
  }
  sampleRate = reader->sampleRate;
  const auto length = static_cast<int>(
      std::min<juce::int64>(reader->lengthInSamples, 1 << 26));
  if (length < 1) {
    error = file.getFileName() + " contains no audio";
    return false;
  }
  buffer.setSize(static_cast<int>(reader->numChannels), length);
  if (!reader->read(&buffer, 0, length, 0, true, true)) {
    error = "Failed to read " + file.getFileName();
    return false;
  }
  return true;
}

/**
 * @brief Writes a float buffer as a 32-bit WAV file.
 * @param file Destination path.
 * @param buffer Samples to write.
 * @param sampleRate Sample rate in Hz.
 * @param error Receives a user-facing failure.
 * @return True when the file was written.
 */
bool writeWavFile(const juce::File &file, const juce::AudioBuffer<float> &buffer,
                  double sampleRate, juce::String &error) {
  file.getParentDirectory().createDirectory();
  file.deleteFile();
  auto stream = file.createOutputStream();
  if (stream == nullptr) {
    error = "Could not create " + file.getFileName();
    return false;
  }
  juce::WavAudioFormat wav;
  auto options =
      juce::AudioFormatWriterOptions{}
          .withSampleRate(sampleRate)
          .withNumChannels(buffer.getNumChannels())
          .withBitsPerSample(32)
          .withSampleFormat(
              juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
  std::unique_ptr<juce::OutputStream> output(std::move(stream));
  auto writer = wav.createWriterFor(output, options);
  if (writer == nullptr) {
    error = "Could not write " + file.getFileName();
    return false;
  }
  if (!writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples())) {
    error = "Failed while writing " + file.getFileName();
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
      const auto tags = item.getProperty("tags", juce::var());
      if (auto *tagArray = tags.getArray()) {
        for (const auto &tag : *tagArray)
          entry.tags.add(tag.toString());
      }
      if (entry.id.isNotEmpty())
        entries.push_back(std::move(entry));
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
  juce::AudioBuffer<float> clean;
  juce::AudioBuffer<float> processed;
  double cleanRate = 0.0;
  double processedRate = 0.0;
  if (!readAudioFile(cleanFile, clean, cleanRate, error) ||
      !readAudioFile(processedFile, processed, processedRate, error))
    return false;
  if (std::abs(cleanRate - processedRate) > 0.5) {
    error = "Imported files must share the same sample rate";
    return false;
  }
  const auto channels = std::min(clean.getNumChannels(), processed.getNumChannels());
  const auto samples = std::min(clean.getNumSamples(), processed.getNumSamples());
  if (channels < 1 || samples < 1) {
    error = "Imported files have no overlapping audio to align";
    return false;
  }
  juce::AudioBuffer<float> alignedX(channels, samples);
  juce::AudioBuffer<float> alignedY(channels, samples);
  for (int channel = 0; channel < channels; ++channel) {
    alignedX.copyFrom(channel, 0, clean, channel, 0, samples);
    alignedY.copyFrom(channel, 0, processed, channel, 0, samples);
  }
  const auto xFile = root.getChildFile(pairId + "_x.wav");
  const auto yFile = root.getChildFile(pairId + "_y.wav");
  if (!writeWavFile(xFile, alignedX, cleanRate, error) ||
      !writeWavFile(yFile, alignedY, cleanRate, error))
    return false;
  entry.xPath = xFile.getFullPathName();
  entry.yPath = yFile.getFullPathName();
  entry.sampleRate = cleanRate;
  entry.channels = channels;
  entry.durationSeconds = static_cast<double>(samples) / cleanRate;
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
} // namespace openyourbox::library
