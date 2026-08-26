#include "PatchSnapshot.h"

#include "graph/NodeGraph.h"
#include "params/ParamIDs.h"

namespace openyourbox::state {

bool PatchSnapshot::isValid() const {
  return parameterState.isValid() && graphDocument.isValid() &&
         graphDocument.hasType("GraphDocument") &&
         schemaVersion >= 1 && schemaVersion <= patchSnapshotSchemaVersion;
}

std::unique_ptr<juce::XmlElement> PatchSnapshot::toXml() const {
  auto xml = parameterState.isValid() ? parameterState.createXml()
                                      : std::make_unique<juce::XmlElement>(
                                            openyourbox::params::stateType);
  if (xml == nullptr)
    xml = std::make_unique<juce::XmlElement>(openyourbox::params::stateType);
  xml->setAttribute("schemaVersion", schemaVersion);
  xml->setAttribute("randomizationCounter", juce::String(randomizationCounter));
  xml->setAttribute("lastTrainObjective", lastTrainObjective);
  if (hasWeights) {
    xml->setAttribute("architectureHash", architectureHash);
    xml->setAttribute("weights", weightsBlob.toBase64Encoding());
  }
  if (graphDocument.isValid()) {
    auto *existing = xml->getChildByName("GraphDocument");
    while (existing != nullptr) {
      xml->removeChildElement(existing, true);
      existing = xml->getChildByName("GraphDocument");
    }
    if (auto graphXml = graphDocument.createXml())
      xml->addChildElement(graphXml.release());
  }
  return xml;
}

std::optional<PatchSnapshot> PatchSnapshot::fromXml(const juce::XmlElement &xml) {
  const auto schema = xml.getIntAttribute("schemaVersion", 1);
  if (schema < 1 || schema > patchSnapshotSchemaVersion)
    return std::nullopt;

  auto restored = juce::ValueTree::fromXml(xml);
  if (!restored.isValid())
    return std::nullopt;

  PatchSnapshot snapshot;
  snapshot.schemaVersion = schema;
  snapshot.graphDocument = restored.getChildWithName("GraphDocument");
  if (snapshot.graphDocument.isValid())
    restored.removeChild(snapshot.graphDocument, nullptr);
  else
    snapshot.graphDocument = juce::ValueTree{"GraphDocument"};
  snapshot.parameterState = std::move(restored);
  snapshot.randomizationCounter = static_cast<std::uint64_t>(
      xml.getStringAttribute("randomizationCounter", "0").getLargeIntValue());
  snapshot.lastTrainObjective =
      xml.getStringAttribute("lastTrainObjective", "mapping");
  snapshot.architectureHash = xml.getStringAttribute("architectureHash");
  if (xml.hasAttribute("weights")) {
    snapshot.hasWeights =
        snapshot.weightsBlob.fromBase64Encoding(xml.getStringAttribute("weights"));
    if (!snapshot.hasWeights)
      snapshot.weightsBlob.reset();
  }
  return snapshot;
}

juce::String PatchSnapshot::sonicFingerprint() const {
  auto xml = toXml();
  if (xml == nullptr)
    return {};
  if (auto *graphXml = xml->getChildByName("GraphDocument")) {
    graphXml->removeAttribute("panX");
    graphXml->removeAttribute("panY");
    graphXml->removeAttribute("zoom");
    graphXml->removeAttribute("mapVisible");
    graphXml->removeAttribute("focusedGroupId");
  }
  const auto text = xml->toString();
  return juce::String::toHexString(static_cast<juce::int64>(text.hashCode64())) +
         "-" + juce::String(text.length());
}

void PatchSnapshot::copyViewportFrom(const juce::ValueTree &viewportSource) {
  if (!graphDocument.isValid() || !viewportSource.isValid())
    return;
  for (const auto *key : {"panX", "panY", "zoom", "mapVisible", "focusedGroupId"}) {
    if (viewportSource.hasProperty(key))
      graphDocument.setProperty(key, viewportSource.getProperty(key), nullptr);
    else
      graphDocument.removeProperty(key, nullptr);
  }
}

namespace {
/**
 * @brief Walks @p tree collecting non-empty artifact path properties.
 * @param tree Graph subtree.
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
} // namespace

bool PatchSnapshot::referencedArtifactsExist(juce::String &error) const {
  error.clear();
  juce::StringArray paths;
  collectArtifactPaths(graphDocument, paths);
  for (const auto &path : paths) {
    if (!juce::File::isAbsolutePath(path))
      continue;
    if (!juce::File(path).existsAsFile()) {
      error = "Preset artifact is missing: " + juce::File(path).getFileName();
      return false;
    }
  }
  return true;
}

bool graphDocumentIsRestorable(const juce::ValueTree &tree, juce::String &error) {
  error.clear();
  if (!tree.isValid() || !tree.hasType("GraphDocument")) {
    error = "Preset graph document is missing or corrupt";
    return false;
  }
  return openyourbox::graph::NodeGraph::documentIsRestorable(tree, error);
}
} // namespace openyourbox::state
