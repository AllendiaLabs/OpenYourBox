#include "InstrumentWidgets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace openyourbox::ui {
namespace {
/**
 * @brief Converts a token to ImGui's float4 colour.
 * @param colour Token RGBA.
 * @return ImGui colour.
 */
ImVec4 vec(const Rgba &colour) noexcept {
  return {colour.r, colour.g, colour.b, colour.a};
}

/**
 * @brief Packed 32-bit colour for draw-list calls.
 * @param colour Token RGBA.
 * @param alpha Optional alpha override in 0–1; negative keeps the token alpha.
 * @return IM_COL32 packed colour.
 */
ImU32 packed(const Rgba &colour, float alpha = -1.0f) noexcept {
  const auto a = alpha >= 0.0f ? alpha : colour.a;
  return ImGui::ColorConvertFloat4ToU32({colour.r, colour.g, colour.b, a});
}
} // namespace

bool InstrumentWidgets::tabBar(const char *id, const std::vector<Tab> &tabs,
                               int *activeIndex, int pendingSelect) {
  if (activeIndex == nullptr || tabs.empty())
    return false;
  if (pendingSelect >= 0)
    *activeIndex = pendingSelect;

  ImGui::PushID(id);
  const auto origin = ImGui::GetCursorScreenPos();
  const float height = ImGui::GetFrameHeight();
  auto *draw = ImGui::GetWindowDrawList();
  float x = 0.0f;
  bool changed = false;
  for (int i = 0; i < static_cast<int>(tabs.size()); ++i) {
    const auto &tab = tabs[static_cast<std::size_t>(i)];
    VisualLanguage::pushStrong();
    const auto textSize = ImGui::CalcTextSize(tab.label);
    VisualLanguage::popFont();
    const ImVec2 tabSize{textSize.x + 18.0f, height};
    ImGui::SetCursorScreenPos(ImVec2(origin.x + x, origin.y));
    ImGui::PushID(i);
    if (ImGui::InvisibleButton("##tab", tabSize)) {
      *activeIndex = tab.index;
      changed = true;
    }
    const bool selected = *activeIndex == tab.index;
    const bool hovered = ImGui::IsItemHovered();
    const auto min = ImGui::GetItemRectMin();
    const auto max = ImGui::GetItemRectMax();
    if (hovered && !selected)
      draw->AddRectFilled(min, max, packed(VisualLanguage::Surface::raised, 0.55f),
                          4.0f);
    VisualLanguage::pushStrong();
    const auto colour =
        selected ? VisualLanguage::Text::primary : VisualLanguage::Text::muted;
    const auto labelPos =
        ImVec2(min.x + 9.0f, min.y + (tabSize.y - textSize.y) * 0.5f);
    draw->AddText(labelPos, packed(colour), tab.label);
    VisualLanguage::popFont();
    if (selected)
      draw->AddLine(ImVec2(min.x + 6.0f, max.y - 1.0f),
                    ImVec2(max.x - 6.0f, max.y - 1.0f),
                    packed(VisualLanguage::accent), 2.0f);
    ImGui::PopID();
    x += tabSize.x + 2.0f;
  }
  ImGui::SetCursorScreenPos(origin);
  ImGui::Dummy(ImVec2(x, height));
  ImGui::PopID();
  return changed;
}

bool InstrumentWidgets::dryWetSlider(const char *id, float *percent) {
  if (percent == nullptr)
    return false;
  ImGui::PushID(id);
  const auto width = std::max(80.0f, ImGui::GetContentRegionAvail().x);
  const auto height = ImGui::GetFrameHeight();
  const auto origin = ImGui::GetCursorScreenPos();
  const bool changedHeld = ImGui::InvisibleButton("##track", ImVec2(width, height));
  const bool active = ImGui::IsItemActive();
  const bool hovered = ImGui::IsItemHovered();
  bool changed = false;
  const float valueWidth = 44.0f;
  const float trackWidth = std::max(48.0f, width - valueWidth);
  if (active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    const auto mouse = ImGui::GetIO().MousePos.x;
    const auto t = std::clamp((mouse - origin.x) / trackWidth, 0.0f, 1.0f);
    *percent = t * 100.0f;
    changed = true;
  }
  (void)changedHeld;
  auto *draw = ImGui::GetWindowDrawList();
  const float y = origin.y + height * 0.5f;
  const float trackH = 3.0f;
  const float fill = std::clamp(*percent / 100.0f, 0.0f, 1.0f);
  draw->AddRectFilled(ImVec2(origin.x, y - trackH * 0.5f),
                      ImVec2(origin.x + trackWidth, y + trackH * 0.5f),
                      packed(VisualLanguage::Surface::raised), 2.0f);
  draw->AddRectFilled(ImVec2(origin.x, y - trackH * 0.5f),
                      ImVec2(origin.x + trackWidth * fill, y + trackH * 0.5f),
                      packed(VisualLanguage::accent), 2.0f);
  const auto thumb = ImVec2(origin.x + trackWidth * fill, y);
  const float radius = hovered || active ? 7.0f : 6.0f;
  draw->AddCircleFilled(thumb, radius, packed(VisualLanguage::Text::primary));
  draw->AddCircle(thumb, radius, packed(VisualLanguage::accent), 0, 1.25f);
  char caption[16];
  std::snprintf(caption, sizeof caption, "%.0f%%", static_cast<double>(*percent));
  draw->AddText(ImVec2(origin.x + trackWidth + 6.0f,
                       origin.y + (height - ImGui::GetTextLineHeight()) * 0.5f),
                packed(VisualLanguage::Text::muted), caption);
  ImGui::PopID();
  return changed;
}

bool InstrumentWidgets::circularKnob(const char *id, float *value, float minimum,
                                     float maximum) {
  if (value == nullptr)
    return false;
  const float span = maximum - minimum;
  ImGui::PushID(id);
  const ImVec2 hit{36.0f, 72.0f};
  const auto origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##knob", hit);
  bool changed = false;
  if (ImGui::IsItemActive() && span > 0.0f) {
    const auto delta = -ImGui::GetIO().MouseDelta.y / (hit.y * 0.85f);
    *value = std::clamp(*value + delta * span, minimum, maximum);
    changed = delta != 0.0f;
  }
  const auto t =
      span > 0.0f ? std::clamp((*value - minimum) / span, 0.0f, 1.0f) : 0.0f;
  auto *draw = ImGui::GetWindowDrawList();
  const auto centre = ImVec2(origin.x + hit.x * 0.5f, origin.y + 28.0f);
  const float radius = 16.0f;
  draw->AddCircleFilled(centre, radius, packed(VisualLanguage::Surface::raised));
  draw->AddCircle(centre, radius, packed(VisualLanguage::border), 0, 1.2f);
  constexpr float pi = 3.14159265f;
  const float a0 = pi * 0.75f;
  const float a1 = a0 + t * pi * 1.5f;
  draw->PathArcTo(centre, radius - 2.5f, a0, a1, 24);
  draw->PathStroke(packed(VisualLanguage::accent), 0, 2.4f);
  const auto needle = ImVec2(centre.x + std::cos(a1) * (radius - 5.0f),
                             centre.y + std::sin(a1) * (radius - 5.0f));
  draw->AddLine(centre, needle, packed(VisualLanguage::Text::primary), 1.6f);
  draw->AddCircleFilled(centre, 2.4f, packed(VisualLanguage::Text::primary));
  ImGui::PopID();
  return changed;
}

bool InstrumentWidgets::xyPad(const char *id, float *x, float *y, float minimum,
                              float maximum, ImVec2 size) {
  if (x == nullptr || y == nullptr)
    return false;
  const float span = maximum - minimum;
  ImGui::PushID(id);
  const auto origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##xy", size);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  bool changed = false;
  auto normalize = [minimum, span](float value) {
    return span > 0.0f ? std::clamp((value - minimum) / span, 0.0f, 1.0f) : 0.0f;
  };
  if ((hovered || active) && ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
      span > 0.0f) {
    const auto mouse = ImGui::GetIO().MousePos;
    const auto nx = std::clamp((mouse.x - origin.x) / size.x, 0.0f, 1.0f);
    const auto ny = std::clamp(1.0f - (mouse.y - origin.y) / size.y, 0.0f, 1.0f);
    *x = minimum + nx * span;
    *y = minimum + ny * span;
    changed = true;
  }
  auto *draw = ImGui::GetWindowDrawList();
  const auto max = ImVec2(origin.x + size.x, origin.y + size.y);
  draw->AddRectFilled(origin, max, packed(VisualLanguage::Surface::raised), 8.0f);
  draw->AddRect(origin, max, packed(VisualLanguage::border), 8.0f);
  const auto midX = origin.x + size.x * 0.5f;
  const auto midY = origin.y + size.y * 0.5f;
  draw->AddLine(ImVec2(midX, origin.y + 8.0f), ImVec2(midX, max.y - 8.0f),
                packed(VisualLanguage::border, 0.55f));
  draw->AddLine(ImVec2(origin.x + 8.0f, midY), ImVec2(max.x - 8.0f, midY),
                packed(VisualLanguage::border, 0.55f));
  const auto handle =
      ImVec2(origin.x + normalize(*x) * size.x,
             origin.y + (1.0f - normalize(*y)) * size.y);
  draw->AddCircleFilled(handle, 6.0f, packed(VisualLanguage::accent));
  draw->AddCircle(handle, 6.0f, packed(VisualLanguage::Text::primary), 0, 1.2f);
  ImGui::PopID();
  return changed;
}

bool InstrumentWidgets::button(const char *label, ImVec2 size,
                               InstrumentButtonKind kind) {
  Rgba fill = VisualLanguage::Surface::raised;
  if (kind == InstrumentButtonKind::primary)
    fill = VisualLanguage::accent;
  else if (kind == InstrumentButtonKind::danger)
    fill = VisualLanguage::danger;
  ImGui::PushStyleColor(ImGuiCol_Button, vec(fill));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        vec(Rgba{std::min(1.0f, fill.r + 0.08f),
                                 std::min(1.0f, fill.g + 0.08f),
                                 std::min(1.0f, fill.b + 0.08f), fill.a}));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                        vec(Rgba{fill.r * 0.85f, fill.g * 0.85f, fill.b * 0.85f,
                                 fill.a}));
  if (kind != InstrumentButtonKind::secondary)
    ImGui::PushStyleColor(ImGuiCol_Text, vec(VisualLanguage::Surface::page));
  if (kind != InstrumentButtonKind::secondary)
    VisualLanguage::pushStrong();
  const auto pressed = ImGui::Button(label, size);
  if (kind != InstrumentButtonKind::secondary)
    VisualLanguage::popFont();
  ImGui::PopStyleColor(kind != InstrumentButtonKind::secondary ? 4 : 3);
  return pressed;
}

bool InstrumentWidgets::field(const char *label, char *buffer,
                              std::size_t bufferSize, ImGuiInputTextFlags flags) {
  ImGui::PushStyleColor(ImGuiCol_FrameBg, vec(VisualLanguage::Surface::raised));
  ImGui::PushStyleColor(ImGuiCol_Border, vec(VisualLanguage::border));
  const auto changed =
      ImGui::InputText(label, buffer, bufferSize, flags);
  ImGui::PopStyleColor(2);
  return changed;
}

bool InstrumentWidgets::checkbox(const char *label, bool *checked) {
  ImGui::PushStyleColor(ImGuiCol_CheckMark, vec(VisualLanguage::accent));
  ImGui::PushStyleColor(ImGuiCol_FrameBg, vec(VisualLanguage::Surface::raised));
  const auto changed = ImGui::Checkbox(label, checked);
  ImGui::PopStyleColor(2);
  return changed;
}

void InstrumentWidgets::analysisWell(ImVec2 origin, ImVec2 size) {
  auto *draw = ImGui::GetWindowDrawList();
  const auto max = ImVec2(origin.x + size.x, origin.y + size.y);
  draw->AddRectFilled(origin, max, packed(VisualLanguage::Surface::raised), 8.0f);
  draw->AddRect(origin, max, packed(VisualLanguage::border), 8.0f);
}

void InstrumentWidgets::pushModalCard() {
  ImGui::PushStyleColor(ImGuiCol_PopupBg, vec(VisualLanguage::Surface::panel));
  ImGui::PushStyleColor(ImGuiCol_Border, vec(VisualLanguage::border));
  ImGui::PushStyleColor(ImGuiCol_TitleBg, vec(VisualLanguage::Surface::panel));
  ImGui::PushStyleColor(ImGuiCol_TitleBgActive,
                        vec(VisualLanguage::Surface::panel));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 14.0f));
}

void InstrumentWidgets::popModalCard() {
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(4);
}

void InstrumentWidgets::pushTreeStyle() {
  ImGui::PushStyleColor(ImGuiCol_Header,
                        ImVec4(VisualLanguage::accent.r, VisualLanguage::accent.g,
                               VisualLanguage::accent.b, 0.12f));
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                        ImVec4(VisualLanguage::accent.r, VisualLanguage::accent.g,
                               VisualLanguage::accent.b, 0.18f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                        ImVec4(VisualLanguage::accent.r, VisualLanguage::accent.g,
                               VisualLanguage::accent.b, 0.28f));
}

void InstrumentWidgets::popTreeStyle() { ImGui::PopStyleColor(3); }
} // namespace openyourbox::ui
