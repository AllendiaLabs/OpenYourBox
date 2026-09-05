#include "VisualLanguage.h"

#include <BinaryData.h>
#include <imgui.h>
#include <imgui_node_editor.h>

namespace ed = ax::NodeEditor;

namespace openyourbox::ui {
namespace {
/** @brief Inter Regular atlas font, or null before @ref VisualLanguage::loadFonts. */
ImFont *bodyFont = nullptr;
/** @brief Inter SemiBold atlas font, or null before @ref VisualLanguage::loadFonts. */
ImFont *strongFont = nullptr;

/**
 * @brief Converts a token to ImGui's float4 colour.
 * @param colour Token RGBA.
 * @return ImGui colour.
 */
ImVec4 toImVec4(const Rgba &colour) noexcept {
  return {colour.r, colour.g, colour.b, colour.a};
}

/**
 * @brief Mixes two tokens for hover / active fills.
 * @param from Base colour.
 * @param toward Tint colour.
 * @param amount Blend toward @p toward in 0–1.
 * @return Mixed colour.
 */
Rgba mix(const Rgba &from, const Rgba &toward, float amount) noexcept {
  return {from.r + (toward.r - from.r) * amount,
          from.g + (toward.g - from.g) * amount,
          from.b + (toward.b - from.b) * amount,
          from.a + (toward.a - from.a) * amount};
}

/**
 * @brief Merges Phosphor Regular into the font most recently added to the atlas.
 * @param io ImGui IO that owns the atlas.
 */
void mergePhosphor(ImGuiIO &io) {
  ImFontConfig icons;
  icons.MergeMode = true;
  icons.PixelSnapH = true;
  icons.FontDataOwnedByAtlas = false;
  icons.GlyphMinAdvanceX = 16.0f;
  static const ImWchar ranges[] = {0xE000, 0xF8FF, 0};
  icons.GlyphRanges = ranges;
  io.Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(BinaryData::PhosphorRegular_ttf),
      BinaryData::PhosphorRegular_ttfSize, 16.0f, &icons);
}
} // namespace

void VisualLanguage::loadFonts() {
  auto &io = ImGui::GetIO();
  io.Fonts->Clear();
  ImFontConfig regular;
  regular.FontDataOwnedByAtlas = false;
  bodyFont = io.Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(BinaryData::InterRegular_ttf),
      BinaryData::InterRegular_ttfSize, 16.0f, &regular);
  mergePhosphor(io);

  ImFontConfig semibold;
  semibold.FontDataOwnedByAtlas = false;
  strongFont = io.Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(BinaryData::InterSemiBold_ttf),
      BinaryData::InterSemiBold_ttfSize, 16.0f, &semibold);
  mergePhosphor(io);
  io.FontDefault = bodyFont;
}

void VisualLanguage::applyStyle() {
  ImGui::GetStyle() = ImGuiStyle{};
  auto &style = ImGui::GetStyle();
  style.WindowRounding = 0.0f;
  style.ChildRounding = 6.0f;
  style.FrameRounding = 5.0f;
  style.PopupRounding = 8.0f;
  style.GrabRounding = 8.0f;
  style.TabRounding = 4.0f;
  style.ScrollbarRounding = 7.0f;
  style.WindowPadding = ImVec2(10.0f, 8.0f);
  style.FramePadding = ImVec2(10.0f, 6.0f);
  style.ItemSpacing = ImVec2(8.0f, 6.0f);
  style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
  style.GrabMinSize = 12.0f;
  style.FrameBorderSize = 1.0f;
  style.WindowBorderSize = 0.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupBorderSize = 1.0f;
  style.TabBarBorderSize = 0.0f;
  style.ScrollbarSize = 12.0f;

  auto *colours = style.Colors;
  const auto page = toImVec4(Surface::page);
  const auto canvas = toImVec4(Surface::canvas);
  const auto panel = toImVec4(Surface::panel);
  const auto raised = toImVec4(Surface::raised);
  const auto text = toImVec4(Text::primary);
  const auto muted = toImVec4(Text::muted);
  const auto accentVec = toImVec4(accent);
  const auto dangerVec = toImVec4(danger);
  const auto borderVec = toImVec4(border);
  const auto hover = toImVec4(mix(Surface::raised, Text::primary, 0.08f));
  const auto active = toImVec4(mix(Surface::raised, accent, 0.28f));
  auto header = toImVec4(mix(Surface::panel, accent, 0.12f));
  header.w = 0.55f;

  colours[ImGuiCol_Text] = text;
  colours[ImGuiCol_TextDisabled] = muted;
  colours[ImGuiCol_WindowBg] = page;
  colours[ImGuiCol_ChildBg] = canvas;
  colours[ImGuiCol_PopupBg] = panel;
  colours[ImGuiCol_Border] = borderVec;
  colours[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colours[ImGuiCol_FrameBg] = raised;
  colours[ImGuiCol_FrameBgHovered] = hover;
  colours[ImGuiCol_FrameBgActive] = active;
  colours[ImGuiCol_TitleBg] = panel;
  colours[ImGuiCol_TitleBgActive] = panel;
  colours[ImGuiCol_TitleBgCollapsed] = page;
  colours[ImGuiCol_MenuBarBg] = panel;
  colours[ImGuiCol_ScrollbarBg] = page;
  colours[ImGuiCol_ScrollbarGrab] = raised;
  colours[ImGuiCol_ScrollbarGrabHovered] = hover;
  colours[ImGuiCol_ScrollbarGrabActive] = active;
  colours[ImGuiCol_CheckMark] = accentVec;
  colours[ImGuiCol_SliderGrab] = accentVec;
  colours[ImGuiCol_SliderGrabActive] = toImVec4(mix(accent, Text::primary, 0.2f));
  colours[ImGuiCol_Button] = raised;
  colours[ImGuiCol_ButtonHovered] = hover;
  colours[ImGuiCol_ButtonActive] = active;
  colours[ImGuiCol_Header] = header;
  colours[ImGuiCol_HeaderHovered] = toImVec4(mix(Surface::raised, accent, 0.18f));
  colours[ImGuiCol_HeaderActive] = active;
  colours[ImGuiCol_Separator] = borderVec;
  colours[ImGuiCol_SeparatorHovered] = accentVec;
  colours[ImGuiCol_SeparatorActive] = accentVec;
  colours[ImGuiCol_ResizeGrip] = raised;
  colours[ImGuiCol_ResizeGripHovered] = hover;
  colours[ImGuiCol_ResizeGripActive] = active;
  colours[ImGuiCol_Tab] = panel;
  colours[ImGuiCol_TabHovered] = hover;
  colours[ImGuiCol_TabSelected] = raised;
  colours[ImGuiCol_TabSelectedOverline] = accentVec;
  colours[ImGuiCol_TabDimmed] = panel;
  colours[ImGuiCol_TabDimmedSelected] = raised;
  colours[ImGuiCol_TabDimmedSelectedOverline] = borderVec;
  colours[ImGuiCol_PlotLines] = accentVec;
  colours[ImGuiCol_PlotLinesHovered] = toImVec4(warning);
  colours[ImGuiCol_PlotHistogram] = toImVec4(Family::helper);
  colours[ImGuiCol_PlotHistogramHovered] = toImVec4(warning);
  colours[ImGuiCol_TableHeaderBg] = panel;
  colours[ImGuiCol_TableBorderStrong] = borderVec;
  colours[ImGuiCol_TableBorderLight] = borderVec;
  colours[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colours[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.02f);
  colours[ImGuiCol_TextLink] = accentVec;
  colours[ImGuiCol_TextSelectedBg] = ImVec4(accent.r, accent.g, accent.b, 0.28f);
  colours[ImGuiCol_DragDropTarget] = accentVec;
  colours[ImGuiCol_NavCursor] = accentVec;
  colours[ImGuiCol_NavWindowingHighlight] = ImVec4(accent.r, accent.g, accent.b, 0.35f);
  colours[ImGuiCol_NavWindowingDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.45f);
  colours[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.55f);
  (void)dangerVec;
}

void VisualLanguage::applyNodeEditorStyle() {
  auto &style = ed::GetStyle();
  style.NodeRounding = 6.0f;
  style.NodeBorderWidth = 1.35f;
  style.HoveredNodeBorderWidth = 1.85f;
  style.SelectedNodeBorderWidth = 2.1f;
  style.PinRounding = 3.0f;
  style.PinBorderWidth = 1.0f;
  style.Colors[ed::StyleColor_Bg] = toImVec4(Surface::canvas);
  style.Colors[ed::StyleColor_Grid] =
      ImVec4(border.r, border.g, border.b, 0.35f);
  style.Colors[ed::StyleColor_NodeBg] =
      ImVec4(Surface::panel.r, Surface::panel.g, Surface::panel.b, 0.92f);
  style.Colors[ed::StyleColor_NodeBorder] = toImVec4(border);
  style.Colors[ed::StyleColor_HovNodeBorder] = toImVec4(accent);
  style.Colors[ed::StyleColor_SelNodeBorder] = toImVec4(accent);
  style.Colors[ed::StyleColor_NodeSelRect] =
      ImVec4(accent.r, accent.g, accent.b, 0.12f);
  style.Colors[ed::StyleColor_NodeSelRectBorder] =
      ImVec4(accent.r, accent.g, accent.b, 0.45f);
  style.Colors[ed::StyleColor_HovLinkBorder] = toImVec4(accent);
  style.Colors[ed::StyleColor_SelLinkBorder] = toImVec4(accent);
  style.Colors[ed::StyleColor_HighlightLinkBorder] = toImVec4(warning);
  style.Colors[ed::StyleColor_PinRect] =
      ImVec4(accent.r, accent.g, accent.b, 0.18f);
  style.Colors[ed::StyleColor_PinRectBorder] = toImVec4(accent);
  style.Colors[ed::StyleColor_Flow] = toImVec4(live);
  style.Colors[ed::StyleColor_FlowMarker] = toImVec4(warning);
  style.Colors[ed::StyleColor_GroupBg] =
      ImVec4(Surface::panel.r, Surface::panel.g, Surface::panel.b, 0.35f);
  style.Colors[ed::StyleColor_GroupBorder] = toImVec4(border);
}

void VisualLanguage::pushBody() {
  if (bodyFont != nullptr)
    ImGui::PushFont(bodyFont, 0.0f);
  else
    ImGui::PushFont(nullptr, 0.0f);
}

void VisualLanguage::pushStrong() {
  if (strongFont != nullptr)
    ImGui::PushFont(strongFont, 0.0f);
  else
    ImGui::PushFont(nullptr, 0.0f);
}

void VisualLanguage::popFont() { ImGui::PopFont(); }
} // namespace openyourbox::ui
