#pragma once

#include <cmath>

namespace openyourbox::ui {
/**
 * @brief Linear RGBA colour used by the OpenYourBox visual language.
 *
 * Values are in 0–1. Tokens stay dark-mode only; there is no light theme.
 */
struct Rgba {
  /** @brief Red channel. */
  float r = 0.0f;
  /** @brief Green channel. */
  float g = 0.0f;
  /** @brief Blue channel. */
  float b = 0.0f;
  /** @brief Alpha channel. */
  float a = 1.0f;
};

/**
 * @class VisualLanguage
 * @brief Named dark-mode tokens, type ramp, and ImGui / node-editor application.
 *
 * Token RGB is the source of truth for chrome, Live, Frozen, families, and
 * danger. @c applyStyle() replaces @c ImGui::StyleColorsDark(). Enumerators
 * such as @c NodeState::liveBlue stay; only the painted RGB changes.
 */
class VisualLanguage {
public:
  /** @brief Page, canvas, inspector, and control-well surfaces. */
  struct Surface {
    /** @brief Full-window / host clear. */
    static constexpr Rgba page{0.055f, 0.059f, 0.071f, 1.0f};
    /** @brief Graph well, slightly lifted over the page. */
    static constexpr Rgba canvas{0.086f, 0.094f, 0.114f, 1.0f};
    /** @brief Inspector and overlay library. */
    static constexpr Rgba panel{0.110f, 0.122f, 0.149f, 1.0f};
    /** @brief Buttons, fields, and inset wells. */
    static constexpr Rgba raised{0.149f, 0.165f, 0.204f, 1.0f};
  };

  /** @brief Primary and secondary type colours. */
  struct Text {
    /** @brief Labels, trees, pins, and numbers. */
    static constexpr Rgba primary{0.910f, 0.918f, 0.933f, 1.0f};
    /** @brief Secondary and disabled copy. */
    static constexpr Rgba muted{0.541f, 0.573f, 0.620f, 1.0f};
  };

  /**
   * @brief Chrome focus, active tab edge, and primary actions.
   *
   * Must not be used as Live or Frozen box fill.
   */
  static constexpr Rgba accent{0.165f, 0.769f, 0.722f, 1.0f};
  /** @brief Learned Live box fill (cool blue role). */
  static constexpr Rgba live{0.282f, 0.580f, 0.910f, 1.0f};
  /** @brief Frozen / Black Box fill (warm gold role). */
  static constexpr Rgba frozen{0.839f, 0.643f, 0.251f, 1.0f};
  /** @brief Illegal cable and blocking error chrome (unambiguously red). */
  static constexpr Rgba danger{0.910f, 0.282f, 0.306f, 1.0f};
  /** @brief Graph warning / muted-input caution. */
  static constexpr Rgba warning{0.910f, 0.690f, 0.251f, 1.0f};
  /** @brief 1 px separators and field outlines. */
  static constexpr Rgba border{0.204f, 0.227f, 0.275f, 1.0f};

  /** @brief Family fills for live boxes that are not learned layers. */
  struct Family {
    /** @brief Audio / group I/O. */
    static constexpr Rgba audioIo{0.220f, 0.690f, 0.580f, 1.0f};
    /** @brief Knob Input and XY Trackpad. */
    static constexpr Rgba conditioning{0.612f, 0.463f, 0.910f, 1.0f};
    /** @brief Helpers / DSP utilities. */
    static constexpr Rgba helper{0.580f, 0.729f, 0.361f, 1.0f};
    /** @brief Data Loader and Loss. */
    static constexpr Rgba trainOnly{0.878f, 0.478f, 0.282f, 1.0f};
  };

  /**
   * @brief Bundled Inter ramp (Regular + SemiBold only).
   *
   * Weights are recorded here so CTest can assert the two-weight contract
   * without loading the ImGui atlas.
   */
  struct Type {
    /** @brief Family name loaded as the default face. */
    static constexpr const char *face = "Inter";
    /** @brief Body / numbers / pins / trees. */
    static constexpr int bodyWeight = 400;
    /** @brief Titles, tabs, and primary buttons. */
    static constexpr int strongWeight = 600;
  };

  /**
   * @brief Phosphor Regular PUA glyphs merged into the Inter atlas.
   *
   * UTF-8 sequences for outline icons. Not Codicons and not emoji.
   */
  struct Icon {
    /** @brief Folder outline (library). */
    static constexpr const char *folder = "\xEE\x89\x8A";
    /** @brief Lock outline (Frozen). */
    static constexpr const char *lock = "\xEE\x8B\xBA";
    /** @brief Warning outline. */
    static constexpr const char *warning = "\xEE\x93\xA0";
    /** @brief Snowflake outline (freeze affordance). */
    static constexpr const char *snowflake = "\xEE\x96\xAA";
  };

  /**
   * @brief sRGB relative luminance (WCAG).
   * @param colour Linear-sRGB 0–1 colour (treated as sRGB encoded).
   * @return Luminance in 0–1.
   */
  static float relativeLuminance(const Rgba &colour) noexcept {
    const auto linearise = [](float channel) {
      return channel <= 0.04045f
                 ? channel / 12.92f
                 : std::pow((channel + 0.055f) / 1.055f, 2.4f);
    };
    return 0.2126f * linearise(colour.r) + 0.7152f * linearise(colour.g) +
           0.0722f * linearise(colour.b);
  }

  /**
   * @brief True when @p colour is a dark surface (not a light/day page).
   * @param colour Surface token.
   * @return True when luminance is well below a light page.
   */
  static bool isDarkSurface(const Rgba &colour) noexcept {
    return relativeLuminance(colour) < 0.18f;
  }

  /**
   * @brief Euclidean RGB distance for arm’s-length distinctness tests.
   * @param left First colour.
   * @param right Second colour.
   * @return Distance in 0–√3.
   */
  static float rgbDistance(const Rgba &left, const Rgba &right) noexcept {
    const auto dr = left.r - right.r;
    const auto dg = left.g - right.g;
    const auto db = left.b - right.b;
    return std::sqrt(dr * dr + dg * dg + db * db);
  }

  /**
   * @brief Loads Inter Regular + SemiBold and merges Phosphor Regular.
   *
   * Replaces the toolkit default face. Call from
   * @c ImGuiHost::newOpenGLContextCreated after creating the ImGui context.
   */
  static void loadFonts();

  /**
   * @brief Maps ImGui style (rounding, padding, @c ImGuiCol_*) onto tokens.
   *
   * Must be the effective theme; do not leave @c StyleColorsDark() in place.
   */
  static void applyStyle();

  /**
   * @brief Maps imgui-node-editor background, grid, node, pin, and select colours.
   *
   * Call while an editor context is current.
   */
  static void applyNodeEditorStyle();

  /** @brief Pushes Inter Regular for body copy, numbers, and trees. */
  static void pushBody();
  /** @brief Pushes Inter SemiBold for titles, tabs, and primary actions. */
  static void pushStrong();
  /** @brief Pops a font pushed by @ref pushBody or @ref pushStrong. */
  static void popFont();
};
} // namespace openyourbox::ui
