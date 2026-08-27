#include "ErrorModal.h"

#include <algorithm>
#include <cstring>

namespace openyourbox::ui {
void ErrorModal::show(const juce::String &text) {
  show(text, Kind::error);
}

void ErrorModal::showWarning(const juce::String &text) {
  show(text, Kind::warning);
}

void ErrorModal::dismiss() {
  visible = false;
}

void ErrorModal::show(const juce::String &text, Kind nextKind) {
  kind = nextKind;
  message = text;
  visible = true;
  const auto utf8Bytes =
      static_cast<std::size_t>(message.getNumBytesAsUTF8());
  const auto bytes = std::max(utf8Bytes, std::size_t{1}) + 1;
  buffer.assign(std::max(bytes, static_cast<std::size_t>(4096)), '\0');
  const auto *utf8 = message.toRawUTF8();
  std::strncpy(buffer.data(), utf8 != nullptr ? utf8 : "", buffer.size() - 1);
}

void ErrorModal::copyToClipboard(const juce::String &text) {
  juce::SystemClipboard::copyTextToClipboard(text);
  ImGui::SetClipboardText(text.toRawUTF8());
}

void ErrorModal::render() {
  if (!visible)
    return;

  const auto *popupId = kind == Kind::warning ? "Warning" : "Error";
  ImGui::OpenPopup(popupId);
  const auto center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(640.0f, 360.0f), ImGuiCond_Appearing);
  bool open = true;
  if (ImGui::BeginPopupModal(popupId, &open, ImGuiWindowFlags_None)) {
    ImGui::TextUnformatted(kind == Kind::warning ? "A warning occurred:"
                                                 : "An error occurred:");
    const auto buttonRow =
        ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    const auto fieldHeight =
        std::max(80.0f, ImGui::GetContentRegionAvail().y - buttonRow);
    if (buffer.empty())
      buffer.assign(1, '\0');
    ImGui::InputTextMultiline("##errorText", buffer.data(), buffer.size(),
                              ImVec2(-1.0f, fieldHeight),
                              ImGuiInputTextFlags_ReadOnly);
    if (ImGui::Button("Copy to clipboard"))
      copyToClipboard(message);
    ImGui::SameLine();
    if (ImGui::Button("Close"))
      open = false;
    if (!open) {
      visible = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else {
    visible = false;
  }
}
} // namespace openyourbox::ui
