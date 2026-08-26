#pragma once

#include "../library/TrainingLibrary.h"
#include "../graph/GraphTypes.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <array>
#include <functional>
#include <string>

namespace openyourbox::ui {
/**
 * @class TrainingLibraryPanel
 * @brief List+detail ImGui browser for the master's training library.
 */
class TrainingLibraryPanel {
public:
  /** @brief Message-thread actions emitted by the library panel. */
  struct Callbacks {
    /** @brief Import a clean/processed file pair chosen by the user. */
    std::function<void()> importPair;
    /** @brief Import a single unpaired clip. */
    std::function<void()> importClip;
    /** @brief Rename the focused entry. */
    std::function<void(const juce::String &id, const juce::String &name)> rename;
    /** @brief Delete the focused entry after the panel has confirmed. */
    std::function<void(const juce::String &id)> removeEntry;
    /** @brief Preview playback of x or y for the focused entry. */
    std::function<void(const juce::String &id, bool playX)> preview;
    /** @brief Stop in-plugin preview playback. */
    std::function<void()> stopPreview;
  };

  /**
   * @brief Draws the library list and detail inspector.
   * @param library Mutable master library store.
   * @param callbacks Editor-owned actions.
   * @param previewPlaying True while x or y preview is audible.
   */
  void render(library::TrainingLibrary &library, const Callbacks &callbacks,
              bool previewPlaying);

  /** @brief Current Train objective used for warn/filter copy. */
  graph::TrainObjective objective = graph::TrainObjective::mapping;

private:
  /** @brief Identifier of the row shown in the detail pane. */
  juce::String focusedId;
  /** @brief Editable display-name buffer for the focused row. */
  std::array<char, 128> nameBuffer{};
  /** @brief True after the user has requested delete confirmation. */
  bool confirmDelete = false;
};
} // namespace openyourbox::ui
