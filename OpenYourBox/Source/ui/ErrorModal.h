#pragma once

#include <imgui.h>
#include <JuceHeader.h>

#include <vector>

namespace openyourbox::ui {
/**
 * @class ErrorModal
 * @brief Blocking error or warning dialog with selectable text and clipboard copy.
 *
 * Host DAWs typically swallow Cmd/Ctrl+C, so the dialog exposes an explicit
 * Copy button that writes through the JUCE system clipboard. Error and warning
 * presentations use distinct popup ids so both can exist in one editor frame.
 */
class ErrorModal {
public:
  /** @brief Severity presented by this dialog instance. */
  enum class Kind { error, warning };

  /**
   * @brief Opens (or refreshes) the dialog with a new error payload.
   * @param message Human-readable error text.
   */
  void show(const juce::String &message);

  /**
   * @brief Opens (or refreshes) the dialog with a graph warning payload.
   * @param message Human-readable warning text.
   */
  void showWarning(const juce::String &message);

  /**
   * @brief Closes the dialog without requiring the Close button.
   */
  void dismiss();

  /**
   * @brief Draws the modal when an error or warning is waiting to be acknowledged.
   */
  void render();

  /**
   * @brief True while the dialog is visible.
   */
  [[nodiscard]] bool isVisible() const noexcept { return visible; }

private:
  /** @brief Copies @p text to the host system clipboard. */
  static void copyToClipboard(const juce::String &text);

  /**
   * @brief Opens the dialog with @p text under @p nextKind.
   * @param text Human-readable body.
   * @param nextKind Error or warning chrome.
   */
  void show(const juce::String &text, Kind nextKind);

  /** @brief True while the popup should stay open. */
  bool visible = false;
  /** @brief Popup title and heading currently presented. */
  Kind kind = Kind::error;
  /** @brief Displayed error or warning body. */
  juce::String message;
  /** @brief Mutable UTF-8 buffer backing the selectable text field. */
  std::vector<char> buffer;
};
} // namespace openyourbox::ui
