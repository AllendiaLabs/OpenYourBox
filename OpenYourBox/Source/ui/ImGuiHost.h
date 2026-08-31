#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <functional>
#include <string>
#include <vector>

struct ImGuiContext;

namespace openyourbox::ui {
/**
 * @class ImGuiHost
 * @brief Owns a per-editor Dear ImGui context rendered by JUCE OpenGL.
 */
class ImGuiHost final : public juce::Component, private juce::OpenGLRenderer {
public:
  /** @brief User drawing callback executed between ImGui NewFrame and Render.
   */
  using RenderCallback = std::function<void()>;

  /**
   * @brief Attaches an OpenGL context and starts continuous rendering.
   * @param callback Function that builds the current ImGui frame.
   */
  explicit ImGuiHost(RenderCallback callback);

  /** @brief Detaches OpenGL and releases the associated ImGui context. */
  ~ImGuiHost() override;

  /** @brief Updates ImGui pointer position and button state. */
  void mouseMove(const juce::MouseEvent &event) override;
  /** @brief Updates ImGui drag pointer state. */
  void mouseDrag(const juce::MouseEvent &event) override;
  /** @brief Forwards a mouse-button press to ImGui. */
  void mouseDown(const juce::MouseEvent &event) override;
  /** @brief Forwards a mouse-button release to ImGui. */
  void mouseUp(const juce::MouseEvent &event) override;
  /** @brief Forwards wheel deltas to ImGui. */
  void mouseWheelMove(const juce::MouseEvent &event,
                      const juce::MouseWheelDetails &wheel) override;
  /**
   * @brief Queues a native trackpad pinch as a smooth graph zoom gesture.
   * @param event Gesture location.
   * @param scaleFactor Relative magnification supplied by JUCE.
   */
  void mouseMagnify(const juce::MouseEvent &event, float scaleFactor) override;
  /**
   * @brief Returns and clears the accumulated pinch magnification.
   * @return Relative scale factor, or 1 when no pinch occurred.
   */
  [[nodiscard]] float takeMagnification() noexcept;
  /** @brief Forwards typed text and queues one-frame ImGui key pulses. */
  bool keyPressed(const juce::KeyPress &key) override;
  /** @brief Forwards navigation-key press and release state to ImGui. */
  bool keyStateChanged(bool isKeyDown) override;

private:
  /**
   * @struct PendingInputState
   * @brief Message-thread input accumulated for the next OpenGL frame.
   */
  struct PendingInputState {
    /** @brief Latest local pointer position. */
    juce::Point<float> mousePosition;
    /** @brief Latest left, right, and middle button states. */
    std::array<bool, 3> mouseButtons{};
    /** @brief Accumulated horizontal wheel delta. */
    float wheelX = 0.0f;
    /** @brief Accumulated vertical wheel delta. */
    float wheelY = 0.0f;
    /** @brief Latest Ctrl, Shift, Alt, and Command states. */
    std::array<bool, 4> modifiers{};
    /** @brief Latest supported navigation key states. */
    std::array<bool, 15> navigationKeys{};
    /**
     * @brief Letter/digit keys to press for one ImGui frame.
     *
     * macOS synthesizes an immediate key-up for Cmd+letter, so polling
     * `isKeyCurrentlyDown` always sees those keys as released.
     */
    std::vector<int> keyPulses;
    /** @brief Unicode characters typed since the previous frame. */
    std::vector<unsigned int> characters;
    /** @brief Whether a pointer update is waiting to be forwarded. */
    bool mousePositionChanged = false;
    /** @brief Whether button states are waiting to be forwarded. */
    bool mouseButtonsChanged = false;
    /** @brief Whether modifier states are waiting to be forwarded. */
    bool modifiersChanged = false;
    /** @brief Whether navigation key states are waiting to be forwarded. */
    bool navigationKeysChanged = false;
  };

  void newOpenGLContextCreated() override;
  void renderOpenGL() override;
  void openGLContextClosing() override;
  /** @brief Forwards queued message-thread input to the ImGui context. */
  void drainPendingInput();
  void updateMouse(const juce::MouseEvent &event);
  void updateButtons(const juce::ModifierKeys &modifiers);
  /**
   * @brief Copies the system clipboard into @ref clipboardUtf8 on the message
   *        thread so paste can run on the OpenGL thread.
   */
  void snapshotSystemClipboard();
  /** @brief Wires ImGui clipboard callbacks to the JUCE system clipboard. */
  void bindClipboardHandlers();
  /**
   * @brief Queues a one-frame ImGui key press from a JUCE key code.
   * @param keyCode JUCE key code, typically `'c'` for Cmd+C on macOS.
   */
  void queueKeyPulse(int keyCode);
  /**
   * @brief True when @p key is a clipboard or undo chord this host must consume.
   * @param key Incoming JUCE key press.
   */
  static bool isEditShortcut(const juce::KeyPress &key) noexcept;

  juce::OpenGLContext openGLContext;
  RenderCallback renderCallback;
  ImGuiContext *imguiContext = nullptr;
  /** @brief Latest native pinch scale awaiting the OpenGL frame. */
  std::atomic<float> pendingMagnification{1.0f};
  /** @brief Protects message-thread input pending for the render thread. */
  juce::CriticalSection inputLock;
  /** @brief Input snapshot consumed at the beginning of each frame. */
  PendingInputState pendingInput;
  /** @brief Last render-thread keyboard capture decision. */
  std::atomic<bool> wantsKeyboardCapture{false};
  /**
   * @brief UTF-8 clipboard snapshot readable from ImGui's GetClipboardText
   *        callback. Updated on the message thread and on copy.
   */
  std::string clipboardUtf8;
  /**
   * @brief ImGui keys pulsed last frame that must be released before the
   *        next NewFrame. Touched only on the OpenGL thread.
   */
  std::vector<int> imguiKeysToRelease;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImGuiHost)
};
} // namespace openyourbox::ui
