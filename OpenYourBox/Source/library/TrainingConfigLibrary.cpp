#include "TrainingConfigLibrary.h"

#include <algorithm>

namespace openyourbox::library {
namespace {
constexpr int trainingConfigSchemaVersion = 1;
} // namespace

TrainingConfigLibrary::TrainingConfigLibrary()
    : TrainingConfigLibrary(trainingConfigsDirectory()) {}

TrainingConfigLibrary::TrainingConfigLibrary(juce::File rootDirectory)
    : root(std::move(rootDirectory)) {
  root.createDirectory();
  load();
}

void TrainingConfigLibrary::load() {
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
      TrainingConfigEntry entry;
      entry.id = item.getProperty("id", {}).toString();
      entry.name = item.getProperty("name", {}).toString();
      entry.updatedAt = item.getProperty("updatedAt", {}).toString();
      entry.config = item.getProperty("config", juce::var());
      if (entry.id.isNotEmpty() && entry.name.isNotEmpty())
        entries.push_back(std::move(entry));
    }
  }
}

bool TrainingConfigLibrary::save() const {
  root.createDirectory();
  auto rootObject = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> serialized;
  for (const auto &entry : entries) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", entry.id);
    object->setProperty("name", entry.name);
    object->setProperty("updatedAt", entry.updatedAt);
    object->setProperty("config", entry.config);
    serialized.add(juce::var(object.release()));
  }
  rootObject->setProperty("schemaVersion", trainingConfigSchemaVersion);
  rootObject->setProperty("entries", serialized);
  const auto indexFile = root.getChildFile("index.json");
  const auto temporary = indexFile.withFileExtension(".json.tmp");
  temporary.deleteFile();
  if (!temporary.replaceWithText(
          juce::JSON::toString(juce::var(rootObject.release()), true)))
    return false;
  return temporary.moveFileTo(indexFile);
}

const std::vector<TrainingConfigEntry> &
TrainingConfigLibrary::getEntries() const noexcept {
  return entries;
}

const TrainingConfigEntry *
TrainingConfigLibrary::findEntry(const juce::String &id) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&id](const TrainingConfigEntry &entry) { return entry.id == id; });
  return found != entries.end() ? &*found : nullptr;
}

std::optional<TrainingConfigEntry>
TrainingConfigLibrary::saveAs(const juce::String &name, const juce::var &config,
                              juce::String &error) {
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the training configuration";
    return std::nullopt;
  }
  TrainingConfigEntry *existing = nullptr;
  for (auto &entry : entries) {
    if (entry.name == trimmed) {
      existing = &entry;
      break;
    }
  }
  const auto now = juce::Time::getCurrentTime().toISO8601(true);
  if (existing != nullptr) {
    existing->config = config;
    existing->updatedAt = now;
    if (!save()) {
      error = "Could not update the training-config catalog";
      return std::nullopt;
    }
    return *existing;
  }
  TrainingConfigEntry entry;
  entry.id = juce::Uuid().toDashedString();
  entry.name = trimmed;
  entry.updatedAt = now;
  entry.config = config;
  entries.push_back(entry);
  if (!save()) {
    entries.pop_back();
    error = "Could not write the training-config catalog";
    return std::nullopt;
  }
  return entries.back();
}

bool TrainingConfigLibrary::rename(const juce::String &id,
                                   const juce::String &name,
                                   juce::String &error) {
  auto *entry = const_cast<TrainingConfigEntry *>(findEntry(id));
  if (entry == nullptr) {
    error = "Training configuration no longer exists";
    return false;
  }
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the training configuration";
    return false;
  }
  for (const auto &candidate : entries) {
    if (candidate.id != id && candidate.name == trimmed) {
      error = "A training configuration named \"" + trimmed + "\" already exists";
      return false;
    }
  }
  entry->name = trimmed;
  entry->updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  return save();
}

bool TrainingConfigLibrary::removeEntry(const juce::String &id) {
  const auto oldSize = entries.size();
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&id](const TrainingConfigEntry &entry) {
                                 return entry.id == id;
                               }),
                entries.end());
  if (entries.size() == oldSize)
    return false;
  return save();
}
} // namespace openyourbox::library
