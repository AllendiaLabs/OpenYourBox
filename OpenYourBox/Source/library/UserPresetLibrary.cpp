#include "UserPresetLibrary.h"
#include "UserDataPaths.h"

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace openyourbox::library {
namespace {
constexpr int presetLibrarySchemaVersion = 1;

/**
 * @brief Recursively rewrites string properties that store artifact paths.
 * @param tree Snapshot node being rewritten.
 * @param rewrite Mapping from stored path to replacement path.
 */
void rewriteArtifactPaths(
    juce::ValueTree tree,
    const std::function<juce::String(const juce::String &)> &rewrite) {
  for (const auto *key : {"weightsPath", "artifactPath"}) {
    if (!tree.hasProperty(key))
      continue;
    const auto current = tree.getProperty(key).toString();
    if (current.isNotEmpty())
      tree.setProperty(key, rewrite(current), nullptr);
  }
  for (int index = 0; index < tree.getNumChildren(); ++index)
    rewriteArtifactPaths(tree.getChild(index), rewrite);
}

/**
 * @brief Collects every non-empty artifact path stored on a snapshot.
 * @param tree Snapshot node to scan.
 * @param paths Destination list.
 */
void collectArtifactPaths(const juce::ValueTree &tree, juce::StringArray &paths) {
  for (const auto *key : {"weightsPath", "artifactPath"}) {
    if (!tree.hasProperty(key))
      continue;
    const auto current = tree.getProperty(key).toString();
    if (current.isNotEmpty())
      paths.addIfNotAlreadyThere(current);
  }
  for (int index = 0; index < tree.getNumChildren(); ++index)
    collectArtifactPaths(tree.getChild(index), paths);
}

/**
 * @brief Copies one file into @p destination with a unique file name.
 * @param source Existing artifact file.
 * @param destinationDirectory Target folder.
 * @param error Receives a user-facing failure.
 * @return Relative file name written under @p destinationDirectory.
 */
juce::String copyUniqueFile(const juce::File &source,
                            const juce::File &destinationDirectory,
                            juce::String &error) {
  destinationDirectory.createDirectory();
  auto name = source.getFileName();
  if (name.isEmpty())
    name = "artifact.bin";
  auto target = destinationDirectory.getChildFile(name);
  int suffix = 1;
  while (target.existsAsFile()) {
    target = destinationDirectory.getChildFile(
        source.getFileNameWithoutExtension() + "-" + juce::String(suffix) +
        source.getFileExtension());
    ++suffix;
  }
  if (!source.copyFileTo(target)) {
    error = "Could not copy artifact " + source.getFileName();
    return {};
  }
  return target.getFileName();
}

} // namespace

UserPresetLibrary::UserPresetLibrary() : UserPresetLibrary(presetsDirectory()) {}

UserPresetLibrary::UserPresetLibrary(juce::File rootDirectory)
    : root(std::move(rootDirectory)) {
  root.createDirectory();
  load();
}

juce::File UserPresetLibrary::getRootDirectory() const { return root; }

void UserPresetLibrary::load() {
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
      UserPresetEntry entry;
      entry.id = item.getProperty("id", {}).toString();
      entry.name = item.getProperty("name", {}).toString();
      entry.createdAt = item.getProperty("createdAt", {}).toString();
      entry.updatedAt = item.getProperty("updatedAt", {}).toString();
      entry.payloadRelativePath =
          item.getProperty("payloadRelativePath", {}).toString();
      entry.schemaVersion =
          static_cast<int>(item.getProperty("schemaVersion", 1));
      if (entry.id.isNotEmpty() && entry.name.isNotEmpty()) {
        if (entry.payloadRelativePath.isEmpty())
          entry.payloadRelativePath = "entries/" + entry.id;
        entries.push_back(std::move(entry));
      }
    }
  }
}

bool UserPresetLibrary::save() const {
  root.createDirectory();
  auto rootObject = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> serialized;
  for (const auto &entry : entries) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", entry.id);
    object->setProperty("name", entry.name);
    object->setProperty("createdAt", entry.createdAt);
    object->setProperty("updatedAt", entry.updatedAt);
    object->setProperty("payloadRelativePath", entry.payloadRelativePath);
    object->setProperty("schemaVersion", entry.schemaVersion);
    serialized.add(juce::var(object.release()));
  }
  rootObject->setProperty("schemaVersion", presetLibrarySchemaVersion);
  rootObject->setProperty("entries", serialized);
  const auto indexFile = root.getChildFile("index.json");
  const auto temporary = indexFile.withFileExtension(".json.tmp");
  temporary.deleteFile();
  if (!temporary.replaceWithText(
          juce::JSON::toString(juce::var(rootObject.release()), true)))
    return false;
  return temporary.moveFileTo(indexFile);
}

const std::vector<UserPresetEntry> &
UserPresetLibrary::getEntries() const noexcept {
  return entries;
}

UserPresetEntry *UserPresetLibrary::findEntry(const juce::String &id) noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&id](const UserPresetEntry &entry) { return entry.id == id; });
  return found != entries.end() ? &*found : nullptr;
}

const UserPresetEntry *
UserPresetLibrary::findEntry(const juce::String &id) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&id](const UserPresetEntry &entry) { return entry.id == id; });
  return found != entries.end() ? &*found : nullptr;
}

const UserPresetEntry *
UserPresetLibrary::findEntryByName(const juce::String &name) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&name](const UserPresetEntry &entry) { return entry.name == name; });
  return found != entries.end() ? &*found : nullptr;
}

std::optional<UserPresetEntry>
UserPresetLibrary::saveAs(state::PatchSnapshot snapshot, const juce::String &name,
                          bool overwrite, juce::String &error) {
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the preset";
    return std::nullopt;
  }
  if (const auto *existing = findEntryByName(trimmed);
      existing != nullptr && !overwrite) {
    error = "A preset named \"" + trimmed + "\" already exists";
    return std::nullopt;
  }

  juce::String id;
  juce::String createdAt;
  if (overwrite) {
    if (auto *existing = findEntryByName(trimmed)) {
      id = existing->id;
      createdAt = existing->createdAt;
    }
  }
  if (id.isEmpty())
    id = juce::Uuid().toDashedString();
  if (createdAt.isEmpty())
    createdAt = juce::Time::getCurrentTime().toISO8601(true);

  const auto payloadRelative = "entries/" + id;
  const auto payloadDirectory = root.getChildFile(payloadRelative);
  payloadDirectory.deleteRecursively();
  payloadDirectory.createDirectory();
  if (!writePayload(snapshot, payloadDirectory, error)) {
    payloadDirectory.deleteRecursively();
    return std::nullopt;
  }

  UserPresetEntry entry;
  entry.id = id;
  entry.name = trimmed;
  entry.createdAt = createdAt;
  entry.updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.payloadRelativePath = payloadRelative;
  entry.schemaVersion = presetLibrarySchemaVersion;
  if (!upsertEntry(entry, error)) {
    payloadDirectory.deleteRecursively();
    return std::nullopt;
  }
  return entry;
}

std::optional<UserPresetEntry>
UserPresetLibrary::saveOverwrite(const juce::String &id,
                                 state::PatchSnapshot snapshot,
                                 juce::String &error) {
  auto *entry = findEntry(id);
  if (entry == nullptr) {
    error = "No current preset to save; use Save As";
    return std::nullopt;
  }
  const auto payloadDirectory = root.getChildFile(entry->payloadRelativePath);
  payloadDirectory.deleteRecursively();
  payloadDirectory.createDirectory();
  if (!writePayload(snapshot, payloadDirectory, error)) {
    payloadDirectory.deleteRecursively();
    return std::nullopt;
  }
  entry->updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  if (!save()) {
    error = "Could not update the preset catalog index";
    return std::nullopt;
  }
  return *entry;
}

std::optional<state::PatchSnapshot>
UserPresetLibrary::loadSnapshot(const juce::String &id,
                                juce::String &error) const {
  const auto *entry = findEntry(id);
  if (entry == nullptr) {
    error = "Preset catalog entry no longer exists";
    return std::nullopt;
  }
  return readPayload(root.getChildFile(entry->payloadRelativePath), error);
}

bool UserPresetLibrary::rename(const juce::String &id, const juce::String &name,
                               juce::String &error) {
  auto *entry = findEntry(id);
  if (entry == nullptr) {
    error = "Preset catalog entry no longer exists";
    return false;
  }
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the preset";
    return false;
  }
  if (const auto *conflict = findEntryByName(trimmed);
      conflict != nullptr && conflict->id != id) {
    error = "A preset named \"" + trimmed + "\" already exists";
    return false;
  }
  entry->name = trimmed;
  entry->updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  return save();
}

bool UserPresetLibrary::removeEntry(const juce::String &id) {
  const auto *entry = findEntry(id);
  if (entry == nullptr)
    return false;
  root.getChildFile(entry->payloadRelativePath).deleteRecursively();
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&id](const UserPresetEntry &candidate) {
                                 return candidate.id == id;
                               }),
                entries.end());
  return save();
}

bool UserPresetLibrary::writePayload(state::PatchSnapshot &snapshot,
                                     const juce::File &payloadDirectory,
                                     juce::String &error) const {
  juce::String restoreError;
  if (!snapshot.isValid() ||
      !state::graphDocumentIsRestorable(snapshot.graphDocument, restoreError)) {
    error = restoreError.isNotEmpty() ? restoreError
                                      : juce::String("Patch snapshot is not restorable");
    return false;
  }
  const auto artifactsDirectory = payloadDirectory.getChildFile("artifacts");
  if (!copyArtifactsIntoEntry(snapshot, artifactsDirectory, error))
    return false;
  const auto xml = snapshot.toXml();
  if (xml == nullptr ||
      !payloadDirectory.getChildFile("patch.xml").replaceWithText(xml->toString())) {
    error = "Could not write the preset snapshot";
    return false;
  }
  return true;
}

std::optional<state::PatchSnapshot>
UserPresetLibrary::readPayload(const juce::File &payloadDirectory,
                               juce::String &error) const {
  const auto xmlFile = payloadDirectory.getChildFile("patch.xml");
  auto xml = juce::XmlDocument::parse(xmlFile);
  if (xml == nullptr) {
    error = "Preset snapshot could not be read";
    return std::nullopt;
  }
  auto snapshot = state::PatchSnapshot::fromXml(*xml);
  juce::String restoreError;
  if (!snapshot.has_value() || !snapshot->isValid() ||
      !state::graphDocumentIsRestorable(snapshot->graphDocument, restoreError)) {
    error = restoreError.isNotEmpty()
                ? restoreError
                : juce::String("Preset snapshot is corrupt");
    return std::nullopt;
  }
  if (!copyArtifactsIntoProject(*snapshot, payloadDirectory, error))
    return std::nullopt;
  return snapshot;
}

bool UserPresetLibrary::copyArtifactsIntoEntry(
    state::PatchSnapshot &snapshot, const juce::File &artifactsDirectory,
    juce::String &error) const {
  juce::StringArray paths;
  collectArtifactPaths(snapshot.graphDocument, paths);
  std::unordered_map<juce::String, juce::String> rewritten;
  for (const auto &path : paths) {
    const juce::File source(path);
    if (!source.existsAsFile()) {
      error = "Missing artifact " + source.getFileName();
      return false;
    }
    const auto fileName = copyUniqueFile(source, artifactsDirectory, error);
    if (fileName.isEmpty())
      return false;
    rewritten[path] = "artifacts/" + fileName;
  }
  rewriteArtifactPaths(snapshot.graphDocument,
                       [&rewritten](const juce::String &path) {
                         const auto found = rewritten.find(path);
                         return found != rewritten.end() ? found->second : path;
                       });
  return true;
}

bool UserPresetLibrary::copyArtifactsIntoProject(
    state::PatchSnapshot &snapshot, const juce::File &payloadDirectory,
    juce::String &error) const {
  juce::StringArray paths;
  collectArtifactPaths(snapshot.graphDocument, paths);
  std::unordered_map<juce::String, juce::String> rewritten;
  const auto weights = weightsDirectory();
  for (const auto &path : paths) {
    auto source = juce::File(path);
    if (!juce::File::isAbsolutePath(path))
      source = payloadDirectory.getChildFile(path);
    if (!source.existsAsFile()) {
      error = "Preset artifact is missing: " + source.getFileName();
      return false;
    }
    const auto fileName = copyUniqueFile(source, weights, error);
    if (fileName.isEmpty())
      return false;
    rewritten[path] = weights.getChildFile(fileName).getFullPathName();
  }
  rewriteArtifactPaths(snapshot.graphDocument,
                       [&rewritten](const juce::String &path) {
                         const auto found = rewritten.find(path);
                         return found != rewritten.end() ? found->second : path;
                       });
  return true;
}

bool UserPresetLibrary::upsertEntry(const UserPresetEntry &entry,
                                    juce::String &error) {
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&entry](const UserPresetEntry &candidate) {
                                 return candidate.id == entry.id;
                               }),
                entries.end());
  entries.push_back(entry);
  if (!save()) {
    error = "Could not update the preset catalog index";
    return false;
  }
  return true;
}
} // namespace openyourbox::library
