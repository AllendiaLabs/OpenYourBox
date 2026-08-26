#pragma once

#include <JuceHeader.h>

namespace openyourbox::library {
/**
 * @brief Returns the plugin's persistent user-data root.
 *
 * macOS: `~/Library/Audio/Presets/Allendia/OpenYourBox`
 * Windows: `Documents/Allendia/OpenYourBox`
 * Linux: `~/.local/share/allendia/OpenYourBox`
 */
inline juce::File userDataRoot() {
#if JUCE_MAC
  return juce::File::getSpecialLocation(juce::File::userHomeDirectory)
      .getChildFile("Library")
      .getChildFile("Audio")
      .getChildFile("Presets")
      .getChildFile("Allendia")
      .getChildFile("OpenYourBox");
#elif JUCE_WINDOWS
  return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
      .getChildFile("Allendia")
      .getChildFile("OpenYourBox");
#else
  return juce::File::getSpecialLocation(juce::File::userHomeDirectory)
      .getChildFile(".local")
      .getChildFile("share")
      .getChildFile("allendia")
      .getChildFile("OpenYourBox");
#endif
}

/**
 * @brief Ensures @p directory exists and returns it.
 * @param directory Folder to create if missing.
 */
inline juce::File ensureDirectory(const juce::File &directory) {
  directory.createDirectory();
  return directory;
}

/**
 * @brief Directory where browsed and trained weight files are stored.
 */
inline juce::File weightsDirectory() {
  return ensureDirectory(userDataRoot().getChildFile("Weights"));
}

/**
 * @brief Directory where the training sample library is stored.
 *
 * Falls back to the legacy Application Support library when the new folder
 * has no index yet so existing captures are not orphaned.
 */
inline juce::File samplesDirectory() {
  const auto samples = userDataRoot().getChildFile("Samples");
  const auto index = samples.getChildFile("index.json");
  if (index.existsAsFile())
    return ensureDirectory(samples);
  const auto legacy =
      juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
          .getChildFile("Allendia")
          .getChildFile("OpenYourBox")
          .getChildFile("TrainingLibrary");
  if (legacy.getChildFile("index.json").existsAsFile())
    return legacy;
  return ensureDirectory(samples);
}

/**
 * @brief Directory where saved user boxes (elements and groups) are stored.
 */
inline juce::File boxesDirectory() {
  return ensureDirectory(userDataRoot().getChildFile("Boxes"));
}

/**
 * @brief Directory where named full-patch user presets are stored.
 *
 * Distinct from @ref boxesDirectory (reusable components) and from the DAW
 * project state owned by the host.
 */
inline juce::File presetsDirectory() {
  return ensureDirectory(userDataRoot().getChildFile("UserPresets"));
}
} // namespace openyourbox::library
