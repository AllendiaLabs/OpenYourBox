#pragma once

#include <imgui.h>
#include <JuceHeader.h>

#include <vector>

namespace openyourbox::ui {
/**
 * @class ErrorModal
 * @brief Blocking error dialog with selectable text and clipboard copy.
 *
 * Host DAWs typically swallow Cmd/Ctrl+C, so the dialog exposes an explicit
 * Copy button that writes through the JUCE system clipboard.
 */
class ErrorModal {
public:
  /**
   * @brief Opens (or refreshes) the dialog with a new error payload.
   * @param message Human-readable error text.
   */
  void show(const juce::String &message);

  /**
   * @brief Draws the modal when an error is waiting to be acknowledged.
   */
  void render();

  /**
   * @brief True while the dialog is visible.
   */
  [[nodiscard]] bool isVisible() const noexcept { return visible; }

private:
  /** @brief Copies @p text to the host system clipboard. */
  static void copyToClipboard(const juce::String &text);

  /** @brief True while the popup should stay open. */
  bool visible = false;
  /** @brief Displayed error body. */
  juce::String message;
  /** @brief Mutable UTF-8 buffer backing the selectable text field. */
  std::vector<char> buffer;
};
} // namespace openyourbox::ui
