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
  const auto modifiers = key.getModifiers();
  {
    const juce::ScopedLock lock(inputLock);
    pendingInput.modifiers = {modifiers.isCtrlDown(), modifiers.isShiftDown(),
                              modifiers.isAltDown(), modifiers.isCommandDown()};
    pendingInput.modifiersChanged = true;
    auto character = key.getTextCharacter();
    // AZERTY and similar layouts treat `^` as a dead key, so getTextCharacter()
    // is often 0 even though the physical key code is '^'. Inject the ASCII
    // caret so expression fields can still accept power notation.
    if (character == 0 && key.getKeyCode() == '^' &&
        !modifiers.isCommandDown() && !modifiers.isCtrlDown())
      character = '^';
    if (character > 0)
      pendingInput.characters.push_back(static_cast<unsigned int>(character));
  }
  return wantsKeyboardCapture.load(std::memory_order_acquire);
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
  wantsKeyboardCapture.store(io.WantCaptureKeyboard, std::memory_order_release);

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
  PendingInputState input;
  {
    const juce::ScopedLock lock(inputLock);
    input = std::move(pendingInput);
    pendingInput.mousePosition = input.mousePosition;
    pendingInput.mouseButtons = input.mouseButtons;
    pendingInput.modifiers = input.modifiers;
    pendingInput.navigationKeys = input.navigationKeys;
    pendingInput.characters.clear();
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
  for (const auto character : input.characters)
    io.AddInputCharacter(character);
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
