#pragma once

#include "../dsp/LiveGraphEngine.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <cstdint>
#include <functional>

namespace openyourbox::ui {
/**
 * @struct AnalysisPanelState
 * @brief Message-thread state for the per-element analysis panel.
 */
struct AnalysisPanelState {
  /** @brief Node currently shown, or zero when no analysis is open. */
  std::int32_t nodeId = 0;
  /** @brief Selected plot family. */
  graph::AnalysisView view = graph::AnalysisView::transfer;
  /** @brief Latest snapshot consumed by the renderer. */
  dsp::AnalysisSnapshot snapshot;
  /** @brief True when the host transport is playing. */
  bool playbackActive = false;
};

/**
 * @class InfoPanel
 * @brief Renders architecture metrics and dual-family analysis plots.
 */
class InfoPanel {
public:
  /**
   * @brief Draws metrics and optional analysis for the selected node.
   * @param receptiveFieldSamples Receptive field in samples.
   * @param sampleRate Current host sample rate.
   * @param parameterCount Number of model parameters.
   * @param buildError Latest asynchronous model error.
   * @param analysis Current analysis panel state; empty `nodeId` hides plots.
   * @param viewChanged Invoked when the user selects a different analysis view.
   * @param viewError Opens the copyable error dialog for @p buildError.
   */
  void render(std::uint64_t receptiveFieldSamples, double sampleRate,
              std::uint64_t parameterCount, const juce::String &buildError,
              AnalysisPanelState &analysis,
              const std::function<void(graph::AnalysisView)> &viewChanged,
              const std::function<void()> &viewError = {}) const;

private:
  /**
   * @brief Draws chain and element-only traces for the active view.
   * @param analysis Panel state containing the snapshot to render.
   * @param plotSize Size of the plot region in pixels.
   */
  void renderPlot(const AnalysisPanelState &analysis, ImVec2 plotSize) const;
};
} // namespace openyourbox::ui
