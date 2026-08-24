#pragma once

#include "../graph/GraphTypes.h"

#include <JuceHeader.h>

#include <functional>
#include <string>

namespace openyourbox::dsp {
/**
 * @class WeightLoader
 * @brief Off-thread compatible weight-file load with atomic provenance update.
 *
 * Incompatible files leave the active weights unchanged and report a clear
 * error. Loading is never performed on the audio thread.
 */
class WeightLoader {
public:
  /**
   * @brief Callback invoked on the message thread after a successful load.
   * @param nodeId Target graph node.
   * @param path Canonical loaded file path.
   */
  using LoadedCallback =
      std::function<void(std::int32_t nodeId, const std::string &path)>;

  /**
   * @brief Callback invoked on the message thread after a rejected load.
   * @param nodeId Target graph node.
   * @param error User-facing incompatibility reason.
   */
  using FailedCallback =
      std::function<void(std::int32_t nodeId, const std::string &error)>;

  /**
   * @brief Attempts to load a weight file for one node on a background thread.
   * @param node Target weight-bearing graph node.
   * @param file User-selected file.
   * @param onLoaded Message-thread success callback.
   * @param onFailed Message-thread failure callback.
   */
  static void loadAsync(const graph::GraphNode &node, const juce::File &file,
                        LoadedCallback onLoaded, FailedCallback onFailed);

  /**
   * @brief Synchronously validates and describes a candidate weight file.
   * @param node Target weight-bearing graph node.
   * @param file User-selected file.
   * @param error Receives a user-facing rejection.
   * @return True when the file is a compatible TorchScript or weight archive.
   */
  static bool isCompatible(const graph::GraphNode &node, const juce::File &file,
                           std::string &error);
};
} // namespace openyourbox::dsp
