#include "UserBoxLibrary.h"
#include "UserDataPaths.h"

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace openyourbox::library {
namespace {
constexpr int boxLibrarySchemaVersion = 1;

/**
 * @brief Normalizes a folder path to `/`-separated components without `..`.
 * @param path User-entered folder path.
 * @param error Receives a reason when the path is illegal.
 * @return Normalized path, or empty with error when invalid. Empty path is root.
 */
juce::String normalizeFolder(const juce::String &path, juce::String &error) {
  error.clear();
  auto parts = juce::StringArray::fromTokens(path.replaceCharacter('\\', '/'),
                                             "/", "");
  juce::StringArray clean;
  for (auto part : parts) {
    part = part.trim();
    if (part.isEmpty() || part == ".")
      continue;
    if (part == ".." || part.containsAnyOf("/\\")) {
      error = "Folder names cannot contain path separators or '..'";
      return {};
    }
    clean.add(part);
  }
  return clean.joinIntoString("/");
}

/**
 * @brief Returns true when @p folder is @p prefix or a descendant of it.
 * @param folder Candidate path.
 * @param prefix Ancestor path.
 */
bool folderIsUnder(const juce::String &folder, const juce::String &prefix) {
  if (prefix.isEmpty())
    return true;
  return folder == prefix || folder.startsWith(prefix + "/");
}

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
void collectArtifactPaths(const juce::ValueTree &tree,
                          juce::StringArray &paths) {
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

UserBoxLibrary::UserBoxLibrary() : UserBoxLibrary(boxesDirectory()) {}

UserBoxLibrary::UserBoxLibrary(juce::File rootDirectory)
    : root(std::move(rootDirectory)) {
  root.createDirectory();
  load();
}

juce::File UserBoxLibrary::getRootDirectory() const { return root; }

void UserBoxLibrary::load() {
  entries.clear();
  folders.clear();
  const auto indexFile = root.getChildFile("index.json");
  if (!indexFile.existsAsFile())
    return;
  const auto parsed = juce::JSON::parse(indexFile.loadFileAsString());
  if (!parsed.isObject())
    return;
  const auto folderList = parsed.getProperty("folders", juce::var());
  if (auto *array = folderList.getArray()) {
    for (const auto &item : *array) {
      juce::String unused;
      const auto folder = normalizeFolder(item.toString(), unused);
      if (folder.isNotEmpty() &&
          std::find(folders.begin(), folders.end(), folder) == folders.end())
        folders.push_back(folder);
    }
  }
  const auto list = parsed.getProperty("entries", juce::var());
  if (auto *array = list.getArray()) {
    for (const auto &item : *array) {
      if (!item.isObject())
        continue;
      UserBoxLibraryEntry entry;
      entry.id = item.getProperty("id", {}).toString();
      entry.name = item.getProperty("name", {}).toString();
      entry.kind = item.getProperty("kind", "element").toString() == "group"
                       ? UserBoxKind::group
                       : UserBoxKind::element;
      juce::String unused;
      entry.folder =
          normalizeFolder(item.getProperty("folder", {}).toString(), unused);
      entry.createdAt = item.getProperty("createdAt", {}).toString();
      entry.updatedAt = item.getProperty("updatedAt", {}).toString();
      entry.payloadRelativePath =
          item.getProperty("payloadRelativePath", {}).toString();
      entry.rootTypeHint = item.getProperty("rootTypeHint", {}).toString();
      entry.schemaVersion =
          static_cast<int>(item.getProperty("schemaVersion", 1));
      if (entry.id.isNotEmpty() && entry.name.isNotEmpty()) {
        if (entry.payloadRelativePath.isEmpty())
          entry.payloadRelativePath = "entries/" + entry.id;
        ensureFolderListed(entry.folder);
        entries.push_back(std::move(entry));
      }
    }
  }
}

bool UserBoxLibrary::save() const {
  root.createDirectory();
  auto rootObject = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> serializedFolders;
  for (const auto &folder : folders)
    serializedFolders.add(folder);
  juce::Array<juce::var> serialized;
  for (const auto &entry : entries) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", entry.id);
    object->setProperty("name", entry.name);
    object->setProperty("kind",
                        entry.kind == UserBoxKind::group ? "group" : "element");
    object->setProperty("folder", entry.folder);
    object->setProperty("createdAt", entry.createdAt);
    object->setProperty("updatedAt", entry.updatedAt);
    object->setProperty("payloadRelativePath", entry.payloadRelativePath);
    object->setProperty("rootTypeHint", entry.rootTypeHint);
    object->setProperty("schemaVersion", entry.schemaVersion);
    serialized.add(juce::var(object.release()));
  }
  rootObject->setProperty("schemaVersion", boxLibrarySchemaVersion);
  rootObject->setProperty("folders", serializedFolders);
  rootObject->setProperty("entries", serialized);
  const auto indexFile = root.getChildFile("index.json");
  const auto temporary = indexFile.withFileExtension(".json.tmp");
  temporary.deleteFile();
  if (!temporary.replaceWithText(juce::JSON::toString(juce::var(rootObject.release()),
                                                      true)))
    return false;
  return temporary.moveFileTo(indexFile);
}

const std::vector<UserBoxLibraryEntry> &
UserBoxLibrary::getEntries() const noexcept {
  return entries;
}

const std::vector<juce::String> &UserBoxLibrary::getFolders() const noexcept {
  return folders;
}

UserBoxLibraryEntry *UserBoxLibrary::findEntry(const juce::String &id) noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&id](const UserBoxLibraryEntry &entry) { return entry.id == id; });
  return found != entries.end() ? &*found : nullptr;
}

const UserBoxLibraryEntry *
UserBoxLibrary::findEntry(const juce::String &id) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&id](const UserBoxLibraryEntry &entry) { return entry.id == id; });
  return found != entries.end() ? &*found : nullptr;
}

const UserBoxLibraryEntry *
UserBoxLibrary::findEntryByName(const juce::String &name) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [&name](const UserBoxLibraryEntry &entry) { return entry.name == name; });
  return found != entries.end() ? &*found : nullptr;
}

std::optional<UserBoxLibraryEntry>
UserBoxLibrary::saveBox(const graph::NodeGraph &graph, std::int32_t boxId,
                        const juce::String &name, const juce::String &folder,
                        bool overwrite, juce::String &error) {
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the box";
    return std::nullopt;
  }
  const auto normalizedFolder = normalizeFolder(folder, error);
  if (error.isNotEmpty())
    return std::nullopt;
  if (const auto *existing = findEntryByName(trimmed);
      existing != nullptr && !overwrite) {
    error = "A box named \"" + trimmed + "\" already exists";
    return std::nullopt;
  }

  auto snapshot = graph.exportBox(boxId, error);
  if (!snapshot.isValid())
    return std::nullopt;

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
  const auto artifactsDirectory = payloadDirectory.getChildFile("artifacts");
  if (!copyArtifactsIntoEntry(snapshot, artifactsDirectory, error)) {
    payloadDirectory.deleteRecursively();
    return std::nullopt;
  }

  const auto xml = snapshot.createXml();
  if (xml == nullptr ||
      !payloadDirectory.getChildFile("box.xml").replaceWithText(xml->toString())) {
    error = "Could not write the box snapshot";
    payloadDirectory.deleteRecursively();
    return std::nullopt;
  }

  UserBoxLibraryEntry entry;
  entry.id = id;
  entry.name = trimmed;
  entry.kind = snapshot.getProperty("kind", "element").toString() == "group"
                   ? UserBoxKind::group
                   : UserBoxKind::element;
  entry.folder = normalizedFolder;
  entry.createdAt = createdAt;
  entry.updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  entry.payloadRelativePath = payloadRelative;
  entry.rootTypeHint = snapshot.getProperty("rootTypeHint", "linear").toString();
  entry.schemaVersion = boxLibrarySchemaVersion;

  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&id](const UserBoxLibraryEntry &candidate) {
                                 return candidate.id == id;
                               }),
                entries.end());
  ensureFolderListed(entry.folder);
  entries.push_back(entry);
  if (!save()) {
    error = "Could not update the box library index";
    return std::nullopt;
  }
  return entry;
}

std::optional<std::int32_t>
UserBoxLibrary::insertBox(graph::NodeGraph &graph, const juce::String &entryId,
                          juce::Point<float> position, juce::String &error,
                          std::int32_t nestedRootId) {
  auto snapshot = loadEntrySnapshot(entryId, error);
  if (!snapshot.isValid())
    return std::nullopt;
  const auto *entry = findEntry(entryId);
  if (entry == nullptr)
    return std::nullopt;
  const auto payloadDirectory = root.getChildFile(entry->payloadRelativePath);
  if (!copyArtifactsIntoProject(snapshot, payloadDirectory, error))
    return std::nullopt;
  return graph.importBox(snapshot, position, true, error, nestedRootId);
}

juce::ValueTree UserBoxLibrary::loadEntrySnapshot(const juce::String &entryId,
                                                  juce::String &error) const {
  error.clear();
  const auto *entry = findEntry(entryId);
  if (entry == nullptr) {
    error = "Box library entry no longer exists";
    return {};
  }
  const auto payloadDirectory = root.getChildFile(entry->payloadRelativePath);
  const auto xmlFile = payloadDirectory.getChildFile("box.xml");
  auto xml = juce::XmlDocument::parse(xmlFile);
  if (xml == nullptr) {
    error = "Box snapshot could not be read";
    return {};
  }
  auto snapshot = juce::ValueTree::fromXml(*xml);
  if (!snapshot.isValid()) {
    error = "Box snapshot is corrupt";
    return {};
  }
  return snapshot;
}

bool UserBoxLibrary::rename(const juce::String &id, const juce::String &name,
                            juce::String &error) {
  auto *entry = findEntry(id);
  if (entry == nullptr) {
    error = "Box library entry no longer exists";
    return false;
  }
  auto trimmed = name.trim();
  if (trimmed.isEmpty()) {
    error = "Enter a name for the box";
    return false;
  }
  if (const auto *conflict = findEntryByName(trimmed);
      conflict != nullptr && conflict->id != id) {
    error = "A box named \"" + trimmed + "\" already exists";
    return false;
  }
  entry->name = trimmed;
  entry->updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  return save();
}

bool UserBoxLibrary::setFolder(const juce::String &id, const juce::String &folder,
                               juce::String &error) {
  auto *entry = findEntry(id);
  if (entry == nullptr) {
    error = "Box library entry no longer exists";
    return false;
  }
  const auto normalized = normalizeFolder(folder, error);
  if (error.isNotEmpty())
    return false;
  entry->folder = normalized;
  entry->updatedAt = juce::Time::getCurrentTime().toISO8601(true);
  ensureFolderListed(normalized);
  return save();
}

bool UserBoxLibrary::removeEntry(const juce::String &id) {
  const auto *entry = findEntry(id);
  if (entry == nullptr)
    return false;
  root.getChildFile(entry->payloadRelativePath).deleteRecursively();
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&id](const UserBoxLibraryEntry &candidate) {
                                 return candidate.id == id;
                               }),
                entries.end());
  return save();
}

bool UserBoxLibrary::createFolder(const juce::String &folder,
                                  juce::String &error) {
  const auto normalized = normalizeFolder(folder, error);
  if (error.isNotEmpty())
    return false;
  if (normalized.isEmpty()) {
    error = "Enter a folder name";
    return false;
  }
  if (std::find(folders.begin(), folders.end(), normalized) != folders.end()) {
    error = "That folder already exists";
    return false;
  }
  auto parent = normalized.upToLastOccurrenceOf("/", false, false);
  if (parent == normalized)
    parent.clear();
  if (parent.isNotEmpty())
    ensureFolderListed(parent);
  folders.push_back(normalized);
  std::sort(folders.begin(), folders.end());
  return save();
}

bool UserBoxLibrary::renameFolder(const juce::String &folder,
                                  const juce::String &newName,
                                  juce::String &error) {
  const auto normalized = normalizeFolder(folder, error);
  if (error.isNotEmpty() || normalized.isEmpty()) {
    if (error.isEmpty())
      error = "Cannot rename the library root";
    return false;
  }
  auto leaf = newName.trim();
  if (leaf.isEmpty() || leaf.containsAnyOf("/\\") || leaf == "..") {
    error = "Enter a valid folder name";
    return false;
  }
  auto parent = normalized.upToLastOccurrenceOf("/", false, false);
  if (parent == normalized)
    parent.clear();
  const auto replacement =
      parent.isEmpty() ? leaf : parent + "/" + leaf;
  if (replacement == normalized)
    return true;
  if (std::find(folders.begin(), folders.end(), replacement) != folders.end()) {
    error = "A folder with that name already exists";
    return false;
  }
  for (auto &path : folders) {
    if (path == normalized)
      path = replacement;
    else if (path.startsWith(normalized + "/"))
      path = replacement + path.fromFirstOccurrenceOf(normalized, false, false);
  }
  for (auto &entry : entries) {
    if (entry.folder == normalized)
      entry.folder = replacement;
    else if (entry.folder.startsWith(normalized + "/"))
      entry.folder =
          replacement +
          entry.folder.fromFirstOccurrenceOf(normalized, false, false);
  }
  std::sort(folders.begin(), folders.end());
  return save();
}

bool UserBoxLibrary::removeFolder(const juce::String &folder, bool deleteContents,
                                  juce::String &error) {
  const auto normalized = normalizeFolder(folder, error);
  if (error.isNotEmpty() || normalized.isEmpty()) {
    if (error.isEmpty())
      error = "Cannot delete the library root";
    return false;
  }
  std::vector<juce::String> containedIds;
  for (const auto &entry : entries) {
    if (folderIsUnder(entry.folder, normalized))
      containedIds.push_back(entry.id);
  }
  if (!containedIds.empty() && !deleteContents) {
    error = "Folder is not empty";
    return false;
  }
  for (const auto &id : containedIds)
    removeEntry(id);
  folders.erase(std::remove_if(folders.begin(), folders.end(),
                               [&normalized](const juce::String &path) {
                                 return folderIsUnder(path, normalized);
                               }),
                folders.end());
  return save();
}

bool UserBoxLibrary::copyArtifactsIntoEntry(juce::ValueTree &snapshot,
                                            const juce::File &artifactsDirectory,
                                            juce::String &error) const {
  juce::StringArray paths;
  collectArtifactPaths(snapshot, paths);
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
  rewriteArtifactPaths(snapshot, [&rewritten](const juce::String &path) {
    const auto found = rewritten.find(path);
    return found != rewritten.end() ? found->second : path;
  });
  return true;
}

bool UserBoxLibrary::copyArtifactsIntoProject(
    juce::ValueTree &snapshot, const juce::File &payloadDirectory,
    juce::String &error) const {
  juce::StringArray paths;
  collectArtifactPaths(snapshot, paths);
  std::unordered_map<juce::String, juce::String> rewritten;
  const auto weights = weightsDirectory();
  for (const auto &path : paths) {
    auto source = juce::File(path);
    if (!juce::File::isAbsolutePath(path))
      source = payloadDirectory.getChildFile(path);
    if (!source.existsAsFile()) {
      error = "Box artifact is missing: " + source.getFileName();
      return false;
    }
    const auto fileName = copyUniqueFile(source, weights, error);
    if (fileName.isEmpty())
      return false;
    rewritten[path] = weights.getChildFile(fileName).getFullPathName();
  }
  rewriteArtifactPaths(snapshot, [&rewritten](const juce::String &path) {
    const auto found = rewritten.find(path);
    return found != rewritten.end() ? found->second : path;
  });
  return true;
}

void UserBoxLibrary::ensureFolderListed(const juce::String &folder) {
  if (folder.isEmpty())
    return;
  juce::String prefix;
  for (const auto &part :
       juce::StringArray::fromTokens(folder, "/", "")) {
    prefix = prefix.isEmpty() ? part : prefix + "/" + part;
    if (std::find(folders.begin(), folders.end(), prefix) == folders.end())
      folders.push_back(prefix);
  }
  std::sort(folders.begin(), folders.end());
}
} // namespace openyourbox::library
