#include "library/UserPresetLibrary.h"
#include "state/PatchSnapshot.h"

#include <JuceHeader.h>

#include <iostream>

namespace {
/**
 * @brief Reports a failed library invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Creates an isolated temporary presets directory.
 */
juce::File makeTempPresetsRoot() {
  auto root = juce::File::getSpecialLocation(juce::File::tempDirectory)
                  .getChildFile("OpenYourBoxPresetLibraryTests")
                  .getChildFile(juce::Uuid().toDashedString());
  root.deleteRecursively();
  root.createDirectory();
  return root;
}

/**
 * @brief Builds a minimal restorable snapshot for catalog tests.
 * @param marker Distinguishing integer stored on the graph document.
 */
openyourbox::state::PatchSnapshot makeSnapshot(int marker) {
  openyourbox::state::PatchSnapshot snapshot;
  snapshot.parameterState = juce::ValueTree{"OpenYourBoxState"};
  snapshot.parameterState.setProperty("marker", marker, nullptr);
  snapshot.graphDocument = juce::ValueTree{"GraphDocument"};
  snapshot.graphDocument.setProperty("version", 2, nullptr);
  snapshot.graphDocument.setProperty("marker", marker, nullptr);
  snapshot.trainConfigJson = "{}";
  return snapshot;
}
} // namespace

/**
 * @brief Runs preset catalog CRUD and overwrite checks.
 * @return Zero when every invariant passes.
 */
int main() {
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;
  const auto root = makeTempPresetsRoot();
  openyourbox::library::UserPresetLibrary library(root);
  juce::String error;

  passed &= expect(!library.saveAs(makeSnapshot(1), "  ", false, error).has_value(),
                   "empty names must be refused");
  error.clear();

  const auto saved =
      library.saveAs(makeSnapshot(1), "PatchA", false, error);
  passed &= expect(saved.has_value() && saved->name == "PatchA",
                   "Save As unique name must create an entry");
  error.clear();
  passed &= expect(!library.saveAs(makeSnapshot(2), "PatchA", false, error).has_value(),
                   "colliding Save As must require overwrite");
  error.clear();
  const auto overwritten =
      library.saveAs(makeSnapshot(2), "PatchA", true, error);
  passed &= expect(overwritten.has_value() && overwritten->id == saved->id,
                   "confirmed overwrite must replace the same entry");

  error.clear();
  const auto second = library.saveAs(makeSnapshot(3), "PatchB", false, error);
  passed &= expect(second.has_value(), "second unique name must save");

  error.clear();
  const auto loaded = library.loadSnapshot(saved->id, error);
  passed &= expect(loaded.has_value() &&
                       static_cast<int>(loaded->graphDocument["marker"]) == 2,
                   "load must return the overwritten snapshot");

  error.clear();
  passed &= expect(library.rename(second->id, "PatchB2", error),
                   "rename to a unique name must succeed");
  error.clear();
  passed &= expect(!library.rename(second->id, "PatchA", error),
                   "rename collision must be refused");

  openyourbox::library::UserPresetLibrary reloaded(root);
  passed &= expect(reloaded.findEntryByName("PatchA") != nullptr &&
                       reloaded.findEntryByName("PatchB2") != nullptr &&
                       reloaded.findEntryByName("PatchB") == nullptr,
                   "index must survive reload after rename");

  return passed ? 0 : 1;
}
