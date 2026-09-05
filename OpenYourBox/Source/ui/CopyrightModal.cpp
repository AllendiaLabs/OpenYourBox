#include "CopyrightModal.h"
#include "InstrumentWidgets.h"

namespace openyourbox::ui {
bool CopyrightModal::render(library::CopyrightAcknowledgment &acknowledgment,
                            bool visible,
                            const std::function<void()> &onConfirmed) {
  if (!visible || acknowledgment.isAcknowledged())
    return false;

  InstrumentWidgets::pushModalCard();
  ImGui::OpenPopup("Copyright acknowledgment");
  const auto center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  bool open = true;
  if (ImGui::BeginPopupModal("Copyright acknowledgment", &open,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", acknowledgment.getCertificationText().toRawUTF8());
    InstrumentWidgets::checkbox("I certify this statement", &certified);
    if (!certified)
      ImGui::BeginDisabled();
    if (InstrumentWidgets::button("Confirm", ImVec2(0.0f, 0.0f),
                                  InstrumentButtonKind::primary) &&
        acknowledgment.acknowledge()) {
      if (onConfirmed)
        onConfirmed();
      certified = false;
      ImGui::CloseCurrentPopup();
      open = false;
    }
    if (!certified)
      ImGui::EndDisabled();
    ImGui::SameLine();
    if (InstrumentWidgets::button("Cancel")) {
      certified = false;
      ImGui::CloseCurrentPopup();
      open = false;
    }
    ImGui::EndPopup();
  }
  InstrumentWidgets::popModalCard();
  return open;
}
} // namespace openyourbox::ui
