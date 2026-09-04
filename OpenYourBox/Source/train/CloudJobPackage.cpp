#include "CloudJobPackage.h"

namespace openyourbox::train {
CloudJobPackage assembleCloudJobPackage(const graph::TrainJobRequest &localRequest,
                                        const library::TrainingLibrary &library,
                                        const juce::String &clientInstanceId,
                                        const juce::String &pluginVersion,
                                        const juce::String &corpusId) {
  CloudJobPackage package;
  package.corpusId = corpusId;
  auto parsed = juce::JSON::parse(juce::String(localRequest.graphFragment));
  auto *root = parsed.getDynamicObject();
  if (root == nullptr)
    root = new juce::DynamicObject();

  auto client = std::make_unique<juce::DynamicObject>();
  client->setProperty("plugin_version", pluginVersion);
  client->setProperty("client_instance_id", clientInstanceId);
  root->setProperty("client", juce::var(client.release()));
  root->setProperty("schema_version", graph::trainGraphSchemaVersion);
  if (root->getProperty("operation").toString().isEmpty())
    root->setProperty("operation", "train_graph");
  if (corpusId.isNotEmpty())
    root->setProperty("corpus_id", corpusId);

  auto *capture = root->getProperty("capture_set").getDynamicObject();
  juce::Array<juce::var> pairParts;
  juce::Array<juce::var> clipParts;
  if (corpusId.isEmpty()) {
    for (const auto &entry : library.getEntries()) {
      if (!entry.selectedForTrain)
        continue;
      if (entry.kind == library::LibraryEntryKind::clip) {
        const juce::File file(entry.xPath);
        const auto name = file.getFileName();
        package.files.push_back({name, file});
        package.totalBytes += file.getSize();
        auto clip = std::make_unique<juce::DynamicObject>();
        clip->setProperty("clip_id", entry.id);
        clip->setProperty("name", name);
        clip->setProperty("kind", "clip");
        clipParts.add(juce::var(clip.release()));
        continue;
      }
      const juce::File xFile(entry.xPath);
      const juce::File yFile(entry.yPath);
      const auto xName = xFile.getFileName();
      const auto yName = yFile.getFileName();
      package.files.push_back({xName, xFile});
      package.files.push_back({yName, yFile});
      package.totalBytes += xFile.getSize() + yFile.getSize();
      auto pair = std::make_unique<juce::DynamicObject>();
      pair->setProperty("pair_id", entry.id);
      pair->setProperty("x_name", xName);
      pair->setProperty("y_name", yName);
      pair->setProperty("kind", "pair");
      pairParts.add(juce::var(pair.release()));
    }
    auto captureSet = std::make_unique<juce::DynamicObject>();
    if (capture != nullptr) {
      for (const auto &name : capture->getProperties())
        captureSet->setProperty(name.name, name.value);
    }
    captureSet->setProperty("pairs", pairParts);
    captureSet->setProperty("clips", clipParts);
    root->setProperty("capture_set", juce::var(captureSet.release()));
  }

  if (corpusId.isEmpty()) {
    auto *bindings = root->getProperty("data_loader_bindings").getDynamicObject();
    if (bindings != nullptr) {
      for (const auto &loaderProp : bindings->getProperties()) {
        auto *perLoader = loaderProp.value.getDynamicObject();
        if (perLoader == nullptr)
          continue;
        for (const auto &outProp : perLoader->getProperties()) {
          auto *output = outProp.value.getDynamicObject();
          if (output == nullptr)
            continue;
          auto *items = output->getProperty("items").getArray();
          if (items == nullptr)
            continue;
          for (auto &itemVar : *items) {
            auto *item = itemVar.getDynamicObject();
            if (item == nullptr)
              continue;
            const auto path = item->getProperty("path").toString();
            if (path.isEmpty())
              continue;
            const juce::File file(path);
            if (!file.existsAsFile())
              continue;
            const auto name = file.getFileName();
            package.files.push_back({name, file});
            package.totalBytes += file.getSize();
            item->setProperty("path", name);
          }
        }
      }
    }
  }

  package.manifest = juce::var(root);
  return package;
}
} // namespace openyourbox::train
