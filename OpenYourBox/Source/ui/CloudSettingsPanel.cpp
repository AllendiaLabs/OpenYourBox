#include "CloudSettingsPanel.h"

namespace openyourbox::ui {
void CloudSettingsPanel::render(const train::CloudSettings &settings,
                                const LinkFlow &flow,
                                const Callbacks &callbacks) {
  ImGui::TextUnformatted("Allendia account");
  const auto status = settings.getLinkStatus();
  if (status == train::CloudSettings::LinkStatus::linked)
    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.5f, 1.0f), "Status: Signed in (%s)",
                       settings.getMaskedToken().toRawUTF8());
  else if (status == train::CloudSettings::LinkStatus::authError)
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
                       "Status: Auth error — sign in again.");
  else
    ImGui::TextDisabled("Status: Not signed in");

  const auto entitlement = settings.getEntitlementStatus();
  const auto hint = settings.getEntitlementHint();
  if (entitlement == train::CloudSettings::EntitlementStatus::available) {
    if (hint.isNotEmpty())
      ImGui::Text("Credits: available (%s)", hint.toRawUTF8());
    else
      ImGui::TextUnformatted("Credits: available");
  } else if (entitlement == train::CloudSettings::EntitlementStatus::unavailable)
    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                       "Credits: unavailable");
  else if (status == train::CloudSettings::LinkStatus::linked)
    ImGui::TextDisabled("Credits: unknown");

  if (flow.active) {
    ImGui::Text("Enter this code on Allendia: %s", flow.userCode.toRawUTF8());
    if (flow.message.isNotEmpty())
      ImGui::TextDisabled("%s", flow.message.toRawUTF8());
    if (ImGui::Button("Cancel") && callbacks.cancelLink)
      callbacks.cancelLink();
  } else if (status != train::CloudSettings::LinkStatus::linked) {
    if (ImGui::Button("Sign in with Allendia") && callbacks.linkAccount)
      callbacks.linkAccount();
  } else if (ImGui::Button("Sign out") && callbacks.disconnect) {
    callbacks.disconnect();
  }
  ImGui::SameLine();
  if (ImGui::Button("Manage account") && callbacks.openStorefront)
    callbacks.openStorefront();
}
} // namespace openyourbox::ui
