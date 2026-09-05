#pragma once

#include "VisualLanguage.h"

#include <imgui.h>

#include <cstddef>
#include <vector>

namespace openyourbox::ui {
/**
 * @brief Visual kind for instrument buttons.
 */
enum class InstrumentButtonKind {
  /** @brief Quiet raised control. */
  secondary,
  /** @brief Accent fill for the primary action in a group. */
  primary,
  /** @brief Danger fill for destructive or blocking actions. */
  danger
};

/**
 * @class InstrumentWidgets
 * @brief Custom instrument chrome for tabs, trees, buttons, fields, and pads.
 *
 * Layout stays Dear ImGui immediate-mode. Hit targets are at least as large as
 * the previous stock controls (Dry/Wet frame height, knob 36×72, XY 118×118).
 */
class InstrumentWidgets {
public:
  /**
   * @brief One inspector tab: label plus the stable side-panel index.
   */
  struct Tab {
    /** @brief Visible label. */
    const char *label = "";
    /** @brief Index stored in @c PluginEditor::sidePanelTab. */
    int index = 0;
  };

  /**
   * @brief Horizontal instrument tab strip (underline, not raised toolkit tabs).
   * @param id ImGui ID.
   * @param tabs Visible tabs in order.
   * @param activeIndex In/out selected @c Tab::index.
   * @param pendingSelect When >= 0, selects that index this frame (Parameters-on-select).
   * @return True when the user clicked a different tab.
   */
  static bool tabBar(const char *id, const std::vector<Tab> &tabs, int *activeIndex,
                     int pendingSelect = -1);

  /**
   * @brief Thin-track Dry/Wet slider with a round thumb.
   * @param id ImGui ID.
   * @param percent Mix in 0–100.
   * @return True while the value changes.
   */
  static bool dryWetSlider(const char *id, float *percent);

  /**
   * @brief Circular Knob Input; vertical drag, hit target 36×72.
   * @param id ImGui ID.
   * @param value In/out scalar.
   * @param minimum Inclusive lower bound.
   * @param maximum Inclusive upper bound.
   * @return True when the value changes.
   */
  static bool circularKnob(const char *id, float *value, float minimum,
                           float maximum);

  /**
   * @brief Dark rounded XY well with a visible handle.
   * @param id ImGui ID.
   * @param x In/out horizontal value.
   * @param y In/out vertical value.
   * @param minimum Inclusive lower bound.
   * @param maximum Inclusive upper bound.
   * @param size Hit target (default matches today's 118×118 pad).
   * @return True when the value changes.
   */
  static bool xyPad(const char *id, float *x, float *y, float minimum,
                    float maximum, ImVec2 size = ImVec2(118.0f, 118.0f));

  /**
   * @brief Crafted button using token fills.
   * @param label Caption and ImGui ID.
   * @param size Explicit size, or (0,0) to size to the label.
   * @param kind Primary, secondary, or danger chrome.
   * @return True when clicked.
   */
  static bool button(const char *label, ImVec2 size = ImVec2(0.0f, 0.0f),
                     InstrumentButtonKind kind = InstrumentButtonKind::secondary);

  /**
   * @brief Inset text field on @c surface.raised.
   * @param label Caption; empty or @c ## hides the label.
   * @param buffer UTF-8 buffer.
   * @param bufferSize Buffer capacity.
   * @param flags ImGui input flags.
   * @return True when the text changes.
   */
  static bool field(const char *label, char *buffer, std::size_t bufferSize,
                    ImGuiInputTextFlags flags = 0);

  /**
   * @brief Crafted checkbox with a token mark.
   * @param label Caption.
   * @param checked In/out state.
   * @return True when the state changes.
   */
  static bool checkbox(const char *label, bool *checked);

  /**
   * @brief Analysis plot well (dark inset card).
   * @param origin Top-left of the well.
   * @param size Well size.
   */
  static void analysisWell(ImVec2 origin, ImVec2 size);

  /**
   * @brief Pushes modal-card colours (panel surface, token border).
   */
  static void pushModalCard();
  /** @brief Pops @ref pushModalCard. */
  static void popModalCard();

  /**
   * @brief Pushes tree-row header colours (no default toolkit-blue Header fill).
   */
  static void pushTreeStyle();
  /** @brief Pops @ref pushTreeStyle. */
  static void popTreeStyle();
};
} // namespace openyourbox::ui
