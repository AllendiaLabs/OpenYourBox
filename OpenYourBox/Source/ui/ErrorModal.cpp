#include "ErrorModal.h"
#include "InstrumentWidgets.h"
#include "VisualLanguage.h"

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
  InstrumentWidgets::pushModalCard();
  ImGui::OpenPopup(popupId);
  const auto center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(640.0f, 360.0f), ImGuiCond_Appearing);
  bool open = true;
  if (ImGui::BeginPopupModal(popupId, &open, ImGuiWindowFlags_None)) {
    const auto heading = kind == Kind::warning ? VisualLanguage::warning
                                               : VisualLanguage::danger;
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImVec4(heading.r, heading.g, heading.b, 1.0f));
    ImGui::TextUnformatted(kind == Kind::warning ? "A warning occurred:"
                                                 : "An error occurred:");
    ImGui::PopStyleColor();
    const auto buttonRow =
        ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    const auto fieldHeight =
        std::max(80.0f, ImGui::GetContentRegionAvail().y - buttonRow);
    if (buffer.empty())
      buffer.assign(1, '\0');
    ImGui::PushStyleColor(ImGuiCol_FrameBg,
                          ImVec4(VisualLanguage::Surface::raised.r,
                                 VisualLanguage::Surface::raised.g,
                                 VisualLanguage::Surface::raised.b, 1.0f));
    ImGui::InputTextMultiline("##errorText", buffer.data(), buffer.size(),
                              ImVec2(-1.0f, fieldHeight),
                              ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor();
    if (InstrumentWidgets::button("Copy to clipboard", ImVec2(0.0f, 0.0f),
                                  InstrumentButtonKind::primary))
      copyToClipboard(message);
    ImGui::SameLine();
    if (InstrumentWidgets::button("Close"))
      open = false;
    if (!open) {
      visible = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else {
    visible = false;
  }
  InstrumentWidgets::popModalCard();
}
} // namespace openyourbox::ui
