#pragma once

#include "../graph/GraphTypes.h"
#include "../library/TrainingLibrary.h"

#include <JuceHeader.h>

#include <vector>

namespace openyourbox::train {
/**
 * @struct CloudCorpusFile
 * @brief One audio part included in a cloud job package.
 */
struct CloudCorpusFile {
  /** @brief Multipart part name without the `file:` prefix. */
  juce::String partName;
  /** @brief Absolute local path of the audio file. */
  juce::File file;
};

/**
 * @struct CloudJobPackage
 * @brief Manifest plus selected library files for Cloud submit.
 */
struct CloudJobPackage {
  /** @brief `manifest.json` object (schema_version 1). */
  juce::var manifest;
  /** @brief Files to upload, or empty when reusing `corpus_id`. */
  std::vector<CloudCorpusFile> files;
  /** @brief Sum of file sizes that would be uploaded. */
  juce::int64 totalBytes = 0;
  /** @brief Retained corpus reference when not uploading. */
  juce::String corpusId;
};

/**
 * @brief Returns true when @p totalBytes exceeds the soft warning threshold.
 * @param totalBytes Sum of selected upload file sizes.
 * @param thresholdBytes Soft warning limit (default 2 GiB).
 */
[[nodiscard]] inline bool exceedsSoftUploadWarn(juce::int64 totalBytes,
                                                juce::int64 thresholdBytes) noexcept {
  const auto limit = thresholdBytes > 0 ? thresholdBytes
                                        : graph::defaultCloudSoftUploadWarnBytes;
  return totalBytes > limit;
}

/**
 * @brief Builds a cloud job package from a local train request and library.
 * @param localRequest Local `train_steerable` JSON already assembled for Run.
 * @param library Training library (selected entries supply audio bytes).
 * @param clientInstanceId Submitter instance id written into the manifest.
 * @param pluginVersion Plugin version string.
 * @param corpusId Optional retained corpus; when set, files are omitted.
 * @return Package ready for multipart or JSON reuse submit.
 */
[[nodiscard]] CloudJobPackage assembleCloudJobPackage(
    const graph::TrainJobRequest &localRequest,
    const library::TrainingLibrary &library, const juce::String &clientInstanceId,
    const juce::String &pluginVersion, const juce::String &corpusId = {});
} // namespace openyourbox::train
