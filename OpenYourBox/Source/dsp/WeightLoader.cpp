#include "WeightLoader.h"

#include "../library/UserDataPaths.h"
#include "TorchScriptBlackBox.h"

namespace openyourbox::dsp {
namespace {
/**
 * @brief Copies @p file into the Weights folder when it lives elsewhere.
 * @param file User-selected weight file.
 * @return Canonical path stored for provenance.
 */
juce::File persistWeightFile(const juce::File &file) {
  const auto directory = library::weightsDirectory();
  if (file.isAChildOf(directory) || file.getParentDirectory() == directory)
    return file;
  auto destination = directory.getNonexistentChildFile(
      file.getFileNameWithoutExtension(), file.getFileExtension());
  if (file.copyFileTo(destination))
    return destination;
  return file;
}
} // namespace

bool WeightLoader::isCompatible(const graph::GraphNode &node,
                                const juce::File &file, std::string &error) {
  error.clear();
  if (!node.hasWeights && node.type != graph::NodeType::blackBox) {
    error = "This element does not own weights";
    return false;
  }
  if (!file.existsAsFile()) {
    error = "Weight file does not exist";
    return false;
  }
  const auto extension = file.getFileExtension().toLowerCase();
  if (extension != ".pt" && extension != ".pth" && extension != ".bin") {
    error = "Weights must be a TorchScript .pt file";
    return false;
  }
  try {
    auto module = torch::jit::load(file.getFullPathName().toStdString(), torch::kCPU);
    module.eval();
    (void)module;
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

void WeightLoader::loadAsync(const graph::GraphNode &node, const juce::File &file,
                             LoadedCallback onLoaded, FailedCallback onFailed) {
  const auto nodeId = node.id;
  juce::Thread::launch([node, file, nodeId, onLoaded, onFailed]() {
    std::string error;
    const auto ok = isCompatible(node, file, error);
    const auto stored = ok ? persistWeightFile(file) : file;
    const auto storedPath = stored.getFullPathName().toStdString();
    juce::MessageManager::callAsync(
        [ok, nodeId, storedPath, error, onLoaded, onFailed]() {
          if (ok) {
            if (onLoaded)
              onLoaded(nodeId, storedPath);
          } else if (onFailed) {
            onFailed(nodeId, error.empty() ? "Incompatible weight file" : error);
          }
        });
  });
}
} // namespace openyourbox::dsp
