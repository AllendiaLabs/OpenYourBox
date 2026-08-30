#include "PatchSnapshot.h"

#include "graph/NodeGraph.h"
#include "params/ParamIDs.h"

namespace openyourbox::state {
namespace {
/**
 * @brief Removes view-only attributes from @p element and its descendants.
 * @param element Graph XML node to strip; ignored when null.
 */
void stripViewOnlyAttributes(juce::XmlElement *element) {
  if (element == nullptr)
    return;
  if (element->hasTagName("GraphDocument")) {
    for (const auto *key : {"panX", "panY", "zoom", "mapVisible",
                            "focusedGroupId", "stickySpine"})
      element->removeAttribute(key);
  }
  if (element->hasTagName("Group")) {
    for (const auto *key : {"viewPanX", "viewPanY", "viewZoom"})
      element->removeAttribute(key);
  }
  for (auto *child = element->getFirstChildElement(); child != nullptr;
       child = child->getNextElement())
    stripViewOnlyAttributes(child);
}

/**
 * @brief Copies per-group camera fields from @p source onto matching groups.
 * @param destination Graph document whose group views should be replaced.
 * @param source Graph document providing the live cameras.
 */
void copyGroupViews(juce::ValueTree destination, const juce::ValueTree &source) {
  if (!destination.isValid() || !source.isValid())
    return;
  for (int index = 0; index < destination.getNumChildren(); ++index) {
    auto destGroup = destination.getChild(index);
    if (!destGroup.hasType("Group"))
      continue;
    const auto id = destGroup.getProperty("id");
    for (int sourceIndex = 0; sourceIndex < source.getNumChildren();
         ++sourceIndex) {
      const auto sourceGroup = source.getChild(sourceIndex);
      if (!sourceGroup.hasType("Group") || sourceGroup.getProperty("id") != id)
        continue;
      for (const auto *key : {"viewPanX", "viewPanY", "viewZoom"}) {
        if (sourceGroup.hasProperty(key))
          destGroup.setProperty(key, sourceGroup.getProperty(key), nullptr);
        else
          destGroup.removeProperty(key, nullptr);
      }
      break;
    }
  }
}
} // namespace

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
  stripViewOnlyAttributes(xml->getChildByName("GraphDocument"));
  const auto text = xml->toString();
  return juce::String::toHexString(static_cast<juce::int64>(text.hashCode64())) +
         "-" + juce::String(text.length());
}

void PatchSnapshot::copyViewportFrom(const juce::ValueTree &viewportSource) {
  if (!graphDocument.isValid() || !viewportSource.isValid())
    return;
  for (const auto *key : {"panX", "panY", "zoom", "mapVisible", "focusedGroupId",
                          "stickySpine"}) {
    if (viewportSource.hasProperty(key))
      graphDocument.setProperty(key, viewportSource.getProperty(key), nullptr);
    else
      graphDocument.removeProperty(key, nullptr);
  }
  copyGroupViews(graphDocument, viewportSource);
}

namespace {
/**
 * @brief Walks @p tree collecting non-empty artifact path properties.
 * @param tree Graph subtree.
 * @param paths Destination list.
 */
void collectArtifactPaths(const juce::ValueTree &tree, juce::StringArray &paths) {
  const auto origin = tree.getProperty("blackBoxOrigin").toString();
  const bool externalLoad = origin == "external_load";
  if (!externalLoad) {
    for (const auto *key : {"weightsPath", "artifactPath"}) {
      if (!tree.hasProperty(key))
        continue;
      const auto current = tree.getProperty(key).toString();
      if (current.isNotEmpty())
        paths.addIfNotAlreadyThere(current);
    }
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
