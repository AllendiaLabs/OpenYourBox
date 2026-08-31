#include "ImGuiHost.h"

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <cmath>
#include <utility>

namespace openyourbox::ui {
ImGuiHost::ImGuiHost(RenderCallback callback)
    : renderCallback(std::move(callback)) {
  setOpaque(true);
  setWantsKeyboardFocus(true);
  openGLContext.setRenderer(this);
  openGLContext.setPreferredVersion(juce::OpenGLVersion{3, 2});
  openGLContext.setContinuousRepainting(true);
  openGLContext.setMultisamplingEnabled(true);
  openGLContext.attachTo(*this);
}

ImGuiHost::~ImGuiHost() { openGLContext.detach(); }

void ImGuiHost::mouseMove(const juce::MouseEvent &event) { updateMouse(event); }

void ImGuiHost::mouseDrag(const juce::MouseEvent &event) { updateMouse(event); }

void ImGuiHost::mouseDown(const juce::MouseEvent &event) {
  grabKeyboardFocus();
  snapshotSystemClipboard();
  updateMouse(event);
}

void ImGuiHost::mouseUp(const juce::MouseEvent &event) { updateMouse(event); }

void ImGuiHost::mouseWheelMove(const juce::MouseEvent &event,
                               const juce::MouseWheelDetails &wheel) {
  updateMouse(event);
  const juce::ScopedLock lock(inputLock);
  pendingInput.wheelX += wheel.deltaX;
  pendingInput.wheelY += wheel.deltaY;
}

void ImGuiHost::mouseMagnify(const juce::MouseEvent &event, float scaleFactor) {
  updateMouse(event);
  if (!std::isfinite(scaleFactor) || scaleFactor <= 0.0f)
    return;
  auto current = pendingMagnification.load(std::memory_order_relaxed);
  while (!pendingMagnification.compare_exchange_weak(
      current, current * scaleFactor, std::memory_order_acq_rel,
      std::memory_order_relaxed)) {
  }
}

float ImGuiHost::takeMagnification() noexcept {
  return pendingMagnification.exchange(1.0f, std::memory_order_acq_rel);
}

bool ImGuiHost::keyPressed(const juce::KeyPress &key) {
  snapshotSystemClipboard();
  const auto modifiers = key.getModifiers();
  {
    const juce::ScopedLock lock(inputLock);
    pendingInput.modifiers = {modifiers.isCtrlDown(), modifiers.isShiftDown(),
                               modifiers.isAltDown(), modifiers.isCommandDown()};
    pendingInput.modifiersChanged = true;
    queueKeyPulse(key.getKeyCode());
    auto character = key.getTextCharacter();
    // AZERTY and similar layouts treat `^` as a dead key, so getTextCharacter()
    // is often 0 even though the physical key code is '^'. Inject the ASCII
    // caret so expression fields can still accept power notation.
    if (character == 0 && key.getKeyCode() == '^' &&
        !modifiers.isCommandDown() && !modifiers.isCtrlDown())
      character = '^';
    // Cmd/Ctrl shortcuts must not insert control characters into InputText;
    // ImGui handles copy/paste/select-all from the matching key events.
    if (character >= 32 && !modifiers.isCommandDown() &&
        !modifiers.isCtrlDown())
      pendingInput.characters.push_back(static_cast<unsigned int>(character));
  }
  return isEditShortcut(key) ||
         wantsKeyboardCapture.load(std::memory_order_acquire);
}

bool ImGuiHost::keyStateChanged(bool isKeyDown) {
  juce::ignoreUnused(isKeyDown);
  const std::array<int, 15> juceKeys{
      juce::KeyPress::tabKey,      juce::KeyPress::leftKey,
      juce::KeyPress::rightKey,    juce::KeyPress::upKey,
      juce::KeyPress::downKey,     juce::KeyPress::pageUpKey,
      juce::KeyPress::pageDownKey, juce::KeyPress::homeKey,
      juce::KeyPress::endKey,      juce::KeyPress::insertKey,
      juce::KeyPress::deleteKey,   juce::KeyPress::backspaceKey,
      juce::KeyPress::spaceKey,    juce::KeyPress::returnKey,
      juce::KeyPress::escapeKey};
  {
    const juce::ScopedLock lock(inputLock);
    for (std::size_t index = 0; index < juceKeys.size(); ++index)
      pendingInput.navigationKeys[index] =
          juce::KeyPress::isKeyCurrentlyDown(juceKeys[index]);
    pendingInput.navigationKeysChanged = true;
  }
  return wantsKeyboardCapture.load(std::memory_order_acquire);
}

void ImGuiHost::newOpenGLContextCreated() {
  imguiContext = ImGui::CreateContext();
  ImGui::SetCurrentContext(imguiContext);
  ImGui::StyleColorsDark();
  auto &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  bindClipboardHandlers();
  ImGui_ImplOpenGL3_Init(nullptr);
}

void ImGuiHost::renderOpenGL() {
  if (imguiContext == nullptr)
    return;

  ImGui::SetCurrentContext(imguiContext);
  ImGui_ImplOpenGL3_NewFrame();

  auto &io = ImGui::GetIO();
  const auto scale = static_cast<float>(openGLContext.getRenderingScale());
  io.DisplaySize =
      ImVec2(static_cast<float>(getWidth()), static_cast<float>(getHeight()));
  io.DisplayFramebufferScale = ImVec2(scale, scale);
  drainPendingInput();

  ImGui::NewFrame();
  if (renderCallback)
    renderCallback();
  ImGui::Render();
  wantsKeyboardCapture.store(io.WantCaptureKeyboard || io.WantTextInput,
                             std::memory_order_release);

  juce::OpenGLHelpers::clear(juce::Colour(20, 23, 30));
  juce::gl::glViewport(
      0, 0, juce::roundToInt(static_cast<float>(getWidth()) * scale),
      juce::roundToInt(static_cast<float>(getHeight()) * scale));
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiHost::openGLContextClosing() {
  if (imguiContext == nullptr)
    return;

  ImGui::SetCurrentContext(imguiContext);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui::DestroyContext(imguiContext);
  imguiContext = nullptr;
}

void ImGuiHost::drainPendingInput() {
  const auto modifiers = juce::ModifierKeys::getCurrentModifiersRealtime();
  PendingInputState input;
  {
    const juce::ScopedLock lock(inputLock);
    if (pendingInput.keyPulses.empty()) {
      pendingInput.modifiers = {modifiers.isCtrlDown(), modifiers.isShiftDown(),
                               modifiers.isAltDown(), modifiers.isCommandDown()};
    }
    pendingInput.modifiersChanged = true;
    input = std::move(pendingInput);
    pendingInput.mousePosition = input.mousePosition;
    pendingInput.mouseButtons = input.mouseButtons;
    pendingInput.modifiers = input.modifiers;
    pendingInput.navigationKeys = input.navigationKeys;
    pendingInput.characters.clear();
    pendingInput.keyPulses.clear();
    pendingInput.wheelX = 0.0f;
    pendingInput.wheelY = 0.0f;
    pendingInput.mousePositionChanged = false;
    pendingInput.mouseButtonsChanged = false;
    pendingInput.modifiersChanged = false;
    pendingInput.navigationKeysChanged = false;
  }

  auto &io = ImGui::GetIO();
  if (input.mousePositionChanged)
    io.AddMousePosEvent(input.mousePosition.x, input.mousePosition.y);
  if (input.mouseButtonsChanged) {
    for (std::size_t index = 0; index < input.mouseButtons.size(); ++index)
      io.AddMouseButtonEvent(static_cast<int>(index),
                             input.mouseButtons[index]);
  }
  if (input.wheelX != 0.0f || input.wheelY != 0.0f)
    io.AddMouseWheelEvent(input.wheelX, input.wheelY);
  if (input.modifiersChanged) {
    io.AddKeyEvent(ImGuiMod_Ctrl, input.modifiers[0]);
    io.AddKeyEvent(ImGuiMod_Shift, input.modifiers[1]);
    io.AddKeyEvent(ImGuiMod_Alt, input.modifiers[2]);
    io.AddKeyEvent(ImGuiMod_Super, input.modifiers[3]);
  }
  if (input.navigationKeysChanged) {
    const std::array<ImGuiKey, 15> keys{
        ImGuiKey_Tab,      ImGuiKey_LeftArrow, ImGuiKey_RightArrow,
        ImGuiKey_UpArrow,  ImGuiKey_DownArrow, ImGuiKey_PageUp,
        ImGuiKey_PageDown, ImGuiKey_Home,      ImGuiKey_End,
        ImGuiKey_Insert,   ImGuiKey_Delete,    ImGuiKey_Backspace,
        ImGuiKey_Space,    ImGuiKey_Enter,     ImGuiKey_Escape};
    for (std::size_t index = 0; index < keys.size(); ++index)
      io.AddKeyEvent(keys[index], input.navigationKeys[index]);
  }
  for (const auto imguiKey : imguiKeysToRelease)
    io.AddKeyEvent(static_cast<ImGuiKey>(imguiKey), false);
  imguiKeysToRelease.clear();
  imguiKeysToRelease.reserve(input.keyPulses.size());
  for (const auto imguiKey : input.keyPulses) {
    io.AddKeyEvent(static_cast<ImGuiKey>(imguiKey), true);
    imguiKeysToRelease.push_back(imguiKey);
  }
  for (const auto character : input.characters)
    io.AddInputCharacter(character);
}

void ImGuiHost::queueKeyPulse(int keyCode) {
  auto normalised = keyCode;
  if (normalised >= 'a' && normalised <= 'z')
    normalised -= 'a' - 'A';
  int imguiKey = ImGuiKey_None;
  if (normalised >= 'A' && normalised <= 'Z')
    imguiKey = ImGuiKey_A + (normalised - 'A');
  else if (normalised >= '0' && normalised <= '9')
    imguiKey = ImGuiKey_0 + (normalised - '0');
  if (imguiKey == ImGuiKey_None)
    return;
  pendingInput.keyPulses.push_back(imguiKey);
}

bool ImGuiHost::isEditShortcut(const juce::KeyPress &key) noexcept {
  const auto modifiers = key.getModifiers();
  if (!(modifiers.isCommandDown() || modifiers.isCtrlDown()) ||
      modifiers.isAltDown())
    return false;
  auto code = key.getKeyCode();
  if (code >= 'a' && code <= 'z')
    code -= 'a' - 'A';
  return code == 'A' || code == 'C' || code == 'V' || code == 'X' ||
         code == 'Y' || code == 'Z';
}

void ImGuiHost::snapshotSystemClipboard() {
  const auto text = juce::SystemClipboard::getTextFromClipboard();
  const juce::ScopedLock lock(inputLock);
  clipboardUtf8 = text.toStdString();
}

void ImGuiHost::bindClipboardHandlers() {
  auto &io = ImGui::GetIO();
  io.ClipboardUserData = this;
  io.SetClipboardTextFn = [](void *user, const char *text) {
    auto *host = static_cast<ImGuiHost *>(user);
    const juce::String copy(text != nullptr ? text : "");
    {
      const juce::ScopedLock lock(host->inputLock);
      host->clipboardUtf8 = copy.toStdString();
    }
    juce::MessageManager::callAsync([copy] {
      juce::SystemClipboard::copyTextToClipboard(copy);
    });
  };
  io.GetClipboardTextFn = [](void *user) -> const char * {
    auto *host = static_cast<ImGuiHost *>(user);
    const juce::ScopedLock lock(host->inputLock);
    return host->clipboardUtf8.c_str();
  };
}

void ImGuiHost::updateMouse(const juce::MouseEvent &event) {
  {
    const juce::ScopedLock lock(inputLock);
    pendingInput.mousePosition = event.position;
    pendingInput.mousePositionChanged = true;
  }
  updateButtons(event.mods);
}

void ImGuiHost::updateButtons(const juce::ModifierKeys &modifiers) {
  const juce::ScopedLock lock(inputLock);
  pendingInput.mouseButtons = {modifiers.isLeftButtonDown(),
                               modifiers.isRightButtonDown(),
                               modifiers.isMiddleButtonDown()};
  pendingInput.mouseButtonsChanged = true;
}
} // namespace openyourbox::ui
