#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace openyourbox::state {
/** @brief Current PatchSnapshot schema; readers refuse unknown future majors. */
inline constexpr int patchSnapshotSchemaVersion = 1;

/**
 * @struct ApplyOptions
 * @brief Controls how a snapshot is restored onto a live processor.
 */
struct ApplyOptions {
  /**
   * @enum WeightPolicy
   * @brief How unrestorable weight archives are handled.
   */
  enum class WeightPolicy {
    /** @brief Keep the rebuilt seed/counter model (host session restore). */
    hostFallback,
    /** @brief Refuse the entire apply when claimed weights cannot load. */
    failClosed
  };

  /** @brief Weight restore policy for this apply. */
  WeightPolicy weightPolicy = WeightPolicy::hostFallback;
  /** @brief When true, keep the live canvas pan/zoom instead of the snapshot's. */
  bool preserveViewport = false;
};

/**
 * @struct PatchSnapshot
 * @brief Full sonic plugin patch shared by host state, presets, and undo.
 */
struct PatchSnapshot {
  /** @brief Snapshot format version. */
  int schemaVersion = patchSnapshotSchemaVersion;
  /** @brief APVTS tree without the GraphDocument child. */
  juce::ValueTree parameterState;
  /** @brief Complete graph document (nodes, links, groups, layout). */
  juce::ValueTree graphDocument{"GraphDocument"};
  /** @brief Serialized LibTorch archive when a published model exists. */
  juce::MemoryBlock weightsBlob;
  /** @brief True when @ref weightsBlob holds a weight archive. */
  bool hasWeights = false;
  /** @brief Hex architecture hash used to validate weight load. */
  juce::String architectureHash;
  /** @brief Randomization counter for deterministic seed recall. */
  std::uint64_t randomizationCounter = 0;
  /** @brief Last train objective name persisted with the patch. */
  juce::String lastTrainObjective{"mapping"};

  /**
   * @brief Returns true when parameter and graph documents are usable.
   */
  [[nodiscard]] bool isValid() const;

  /**
   * @brief Serializes this snapshot to the host-state XML document.
   * @return Root element matching today's `getStateInformation` shape.
   */
  [[nodiscard]] std::unique_ptr<juce::XmlElement> toXml() const;

  /**
   * @brief Parses a host-state or preset XML document.
   * @param xml Root element produced by @ref toXml or legacy host state.
   * @return Snapshot, or no value when the document is unusable.
   */
  [[nodiscard]] static std::optional<PatchSnapshot>
  fromXml(const juce::XmlElement &xml);

  /**
   * @brief Hash of sonic-relevant contents (excludes canvas pan/zoom).
   *
   * Used to detect dirty presets and skip no-op history steps.
   */
  [[nodiscard]] juce::String sonicFingerprint() const;

  /**
   * @brief Copies pan/zoom/map fields from @p viewportSource into the graph.
   * @param viewportSource Graph document whose view should be kept.
   */
  void copyViewportFrom(const juce::ValueTree &viewportSource);

  /**
   * @brief Verifies every non-empty artifact/weights path exists on disk.
   * @param error Receives a user-facing refusal when a file is missing.
   */
  [[nodiscard]] bool referencedArtifactsExist(juce::String &error) const;
};

/**
 * @brief Validates a GraphDocument for restore (known types, well-formed).
 * @param tree Candidate graph document.
 * @param error Receives a user-facing refusal.
 * @return True when the document can be applied without a partial graph.
 */
bool graphDocumentIsRestorable(const juce::ValueTree &tree, juce::String &error);
} // namespace openyourbox::state
