#include "CopyrightModal.h"

namespace openyourbox::ui {
bool CopyrightModal::render(library::CopyrightAcknowledgment &acknowledgment,
                            bool visible,
                            const std::function<void()> &onConfirmed) {
  if (!visible || acknowledgment.isAcknowledged())
    return false;

  ImGui::OpenPopup("Copyright acknowledgment");
  const auto center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  bool open = true;
  if (ImGui::BeginPopupModal("Copyright acknowledgment", &open,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", acknowledgment.getCertificationText().toRawUTF8());
    ImGui::Checkbox("I certify this statement", &certified);
    if (!certified)
      ImGui::BeginDisabled();
    if (ImGui::Button("Confirm") && acknowledgment.acknowledge()) {
      if (onConfirmed)
        onConfirmed();
      certified = false;
      ImGui::CloseCurrentPopup();
      open = false;
    }
    if (!certified)
      ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      certified = false;
      ImGui::CloseCurrentPopup();
      open = false;
    }
    ImGui::EndPopup();
  }
  return open;
}
} // namespace openyourbox::ui
