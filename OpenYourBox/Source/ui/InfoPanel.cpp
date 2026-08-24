#include "InfoPanel.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace openyourbox::ui {
namespace {
ImU32 channelColour(int channel, int channelCount, bool chain) {
  static const ImU32 colours[] = {
      IM_COL32(80, 180, 255, 255),  IM_COL32(255, 140, 80, 255),
      IM_COL32(120, 220, 140, 255), IM_COL32(220, 120, 220, 255),
      IM_COL32(255, 220, 80, 255),  IM_COL32(80, 220, 220, 255),
      IM_COL32(255, 100, 140, 255), IM_COL32(180, 180, 255, 255)};
  const auto index = static_cast<unsigned>(channel) %
                     (sizeof(colours) / sizeof(colours[0]));
  auto colour = colours[index];
  if (!chain) {
    const auto r = (colour >> 0) & 0xFFu;
    const auto g = (colour >> 8) & 0xFFu;
    const auto b = (colour >> 16) & 0xFFu;
    colour = IM_COL32((r + 40) / 2, (g + 40) / 2, (b + 40) / 2, 220);
  }
  (void)channelCount;
  return colour;
}

ImVec2 mapPoint(const ImVec2 &origin, const ImVec2 &size, float x, float y,
                float minX, float maxX, float minY, float maxY, bool logX) {
  float nx = 0.0f;
  if (logX) {
    const auto safeMin = std::max(minX, 1.0f);
    const auto safeMax = std::max(maxX, safeMin * 1.01f);
    const auto safeX = std::max(x, safeMin);
    nx = static_cast<float>((std::log(safeX) - std::log(safeMin)) /
                            (std::log(safeMax) - std::log(safeMin)));
  } else {
    nx = maxX > minX ? (x - minX) / (maxX - minX) : 0.0f;
  }
  const auto ny = maxY > minY ? (y - minY) / (maxY - minY) : 0.0f;
  return {origin.x + nx * size.x, origin.y + (1.0f - ny) * size.y};
}
} // namespace

void InfoPanel::render(std::uint64_t receptiveFieldSamples, double sampleRate,
                       std::uint64_t parameterCount,
                       const juce::String &buildError,
                       AnalysisPanelState &analysis,
                       const std::function<void(graph::AnalysisView)>
                           &viewChanged,
                       const std::function<void()> &viewError) const {
  const auto milliseconds =
      sampleRate > 0.0
          ? static_cast<double>(receptiveFieldSamples) * 1000.0 / sampleRate
          : 0.0;

  ImGui::Text("Receptive field: %llu samples (%.2f ms)",
              static_cast<unsigned long long>(receptiveFieldSamples),
              milliseconds);
  ImGui::Text("Trainable parameters: %llu",
              static_cast<unsigned long long>(parameterCount));

  if (milliseconds > 1000.0) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.2f, 1.0f));
    ImGui::TextWrapped("Warning: this receptive field exceeds one second and "
                       "may be too expensive for real-time playback.");
    ImGui::PopStyleColor();
  }

  if (buildError.isNotEmpty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    ImGui::TextUnformatted("Model error");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::SmallButton("View") && viewError)
      viewError();
  }

  ImGui::Separator();
  if (analysis.nodeId == 0) {
    ImGui::TextDisabled("Select Analyze on a Blue or Gold node.");
    return;
  }

  ImGui::Text("Analysis");
  const char *views[] = {"Transfer", "Frequency", "Phase", "Oscilloscope"};
  auto viewIndex = static_cast<int>(analysis.view);
  if (ImGui::Combo("View", &viewIndex, views, 4)) {
    analysis.view = static_cast<graph::AnalysisView>(viewIndex);
    if (viewChanged)
      viewChanged(analysis.view);
  }

  switch (analysis.snapshot.status) {
  case dsp::AnalysisStatus::live:
    ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.45f, 1.0f), "Source: live audio");
    break;
  case dsp::AnalysisStatus::probeFallback:
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                       "Source: probe fallback");
    break;
  case dsp::AnalysisStatus::disconnected:
    ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f),
                       "Source: disconnected");
    break;
  case dsp::AnalysisStatus::unavailable:
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                       "Analysis unavailable");
    break;
  }
  if (analysis.snapshot.isStale)
    ImGui::TextDisabled("Snapshot is stale; refreshing...");

  const auto plotHeight = std::max(180.0f, ImGui::GetContentRegionAvail().y - 90.0f);
  renderPlot(analysis, ImVec2(ImGui::GetContentRegionAvail().x, plotHeight));

  ImGui::Text("Legend");
  const auto channelCount = analysis.snapshot.channelCount;
  const auto legendLimit = std::min(channelCount, 16);
  const auto oscilloscope =
      analysis.view == graph::AnalysisView::oscilloscope;
  for (int channel = 0; channel < legendLimit; ++channel) {
    ImGui::PushID(channel);
    if (oscilloscope) {
      ImGui::ColorButton("##trace",
                         ImGui::ColorConvertU32ToFloat4(
                             channelColour(channel, channelCount, false)),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10, 10));
      ImGui::SameLine();
      ImGui::Text("%s",
                  dsp::analysisChannelLabel(channel, channelCount).c_str());
    } else {
      ImGui::ColorButton("##chain",
                         ImGui::ColorConvertU32ToFloat4(
                             channelColour(channel, channelCount, true)),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10, 10));
      ImGui::SameLine();
      ImGui::Text("%s chain",
                  dsp::analysisChannelLabel(channel, channelCount).c_str());
      ImGui::SameLine();
      ImGui::ColorButton("##elem",
                         ImGui::ColorConvertU32ToFloat4(
                             channelColour(channel, channelCount, false)),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10, 10));
      ImGui::SameLine();
      ImGui::Text("%s element",
                  dsp::analysisChannelLabel(channel, channelCount).c_str());
    }
    ImGui::PopID();
  }
  if (channelCount > legendLimit)
    ImGui::TextDisabled("... %d more dimensions (thinner traces)",
                        channelCount - legendLimit);
}

void InfoPanel::renderPlot(const AnalysisPanelState &analysis,
                           ImVec2 plotSize) const {
  const auto origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##analysisplot", plotSize);
  auto *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin,
                      ImVec2(origin.x + plotSize.x, origin.y + plotSize.y),
                      IM_COL32(18, 22, 30, 255), 4.0f);
  draw->AddRect(origin, ImVec2(origin.x + plotSize.x, origin.y + plotSize.y),
                IM_COL32(70, 80, 95, 255), 4.0f);

  const auto &snapshot = analysis.snapshot;
  auto boundsOf = [](const std::vector<dsp::AnalysisSeries> &family, float &minX,
                     float &maxX, float &minY, float &maxY) {
    for (const auto &series : family) {
      for (std::size_t index = 0; index < series.x.size(); ++index) {
        minX = std::min(minX, series.x[index]);
        maxX = std::max(maxX, series.x[index]);
        if (index < series.y.size()) {
          minY = std::min(minY, series.y[index]);
          maxY = std::max(maxY, series.y[index]);
        }
      }
    }
  };
  float minX = 0.0f;
  float maxX = 1.0f;
  float minY = 0.0f;
  float maxY = 1.0f;
  if (snapshot.view == graph::AnalysisView::transfer) {
    minX = -1.0f;
    maxX = 1.0f;
    minY = 0.0f;
    maxY = 1.0f;
  }
  if (snapshot.view == graph::AnalysisView::oscilloscope) {
    minY = -1.0f;
    maxY = 1.0f;
    boundsOf(snapshot.elementOnlySeries, minX, maxX, minY, maxY);
  } else {
    boundsOf(snapshot.chainSeries, minX, maxX, minY, maxY);
    boundsOf(snapshot.elementOnlySeries, minX, maxX, minY, maxY);
  }
  if (maxY - minY < 1.0e-4f)
    maxY = minY + 1.0f;
  if (maxX - minX < 1.0e-4f)
    maxX = minX + 1.0f;
  const auto logX = snapshot.view != graph::AnalysisView::transfer &&
                    snapshot.view != graph::AnalysisView::oscilloscope;
  const auto thickness = snapshot.channelCount > 16 ? 1.0f : 1.6f;

  auto drawFamily = [&](const std::vector<dsp::AnalysisSeries> &family,
                        bool chain) {
    for (int channel = 0; channel < static_cast<int>(family.size());
         ++channel) {
      const auto &series = family[static_cast<std::size_t>(channel)];
      if (series.x.size() < 2 || series.x.size() != series.y.size())
        continue;
      const auto colour =
          channelColour(channel, snapshot.channelCount, chain);
      for (std::size_t index = 1; index < series.x.size(); ++index) {
        const auto from =
            mapPoint(origin, plotSize, series.x[index - 1], series.y[index - 1],
                     minX, maxX, minY, maxY, logX);
        const auto to = mapPoint(origin, plotSize, series.x[index],
                                 series.y[index], minX, maxX, minY, maxY, logX);
        draw->AddLine(from, to, colour, chain ? thickness : thickness * 0.85f);
      }
    }
  };
  if (snapshot.view == graph::AnalysisView::oscilloscope)
    drawFamily(snapshot.elementOnlySeries, false);
  else {
    drawFamily(snapshot.elementOnlySeries, false);
    drawFamily(snapshot.chainSeries, true);
  }

  if (analysis.playbackActive && snapshot.transferMarker.has_value() &&
      snapshot.view == graph::AnalysisView::transfer) {
    const auto &marker = *snapshot.transferMarker;
    const auto point =
        mapPoint(origin, plotSize, marker.inputLevel, marker.outputLevel, minX,
                 maxX, minY, maxY, false);
    draw->AddCircleFilled(point, 4.5f, IM_COL32(255, 255, 255, 240));
    draw->AddCircle(point, 4.5f, IM_COL32(20, 20, 20, 255), 12, 1.5f);
  }
}
} // namespace openyourbox::ui
