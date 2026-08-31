#include "CloudSettingsPanel.h"

#include <cstring>

namespace openyourbox::ui {
namespace {
void copyToBuffer(std::array<char, 512> &buffer, const juce::String &text) {
  const auto utf8 = text.toRawUTF8();
  const auto length =
      std::min(static_cast<size_t>(std::strlen(utf8)), buffer.size() - 1);
  std::memcpy(buffer.data(), utf8, length);
  buffer[length] = '\0';
}
} // namespace

void CloudSettingsPanel::syncEndpointBuffers(const train::CloudSettings &settings) {
  const auto api = settings.getApiBaseUrlOverride();
  const auto storefront = settings.getStorefrontUrlOverride();
  if (!endpointBuffersReady || api != lastSyncedApiOverride) {
    copyToBuffer(apiBaseUrlBuffer, api);
    lastSyncedApiOverride = api;
  }
  if (!endpointBuffersReady || storefront != lastSyncedStorefrontOverride) {
    copyToBuffer(storefrontUrlBuffer, storefront);
    lastSyncedStorefrontOverride = storefront;
  }
  endpointBuffersReady = true;
}

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

  ImGui::Separator();
  if (ImGui::CollapsingHeader("Cloud endpoint (staging / RunPod)")) {
    syncEndpointBuffers(settings);
    ImGui::TextDisabled("Leave empty to use product defaults.");
    ImGui::TextDisabled("Effective API: %s", settings.getApiBaseUrl().toRawUTF8());

    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##cloudApiBase",
                             "API base URL (e.g. https://POD-8787.proxy.runpod.net)",
                             apiBaseUrlBuffer.data(), apiBaseUrlBuffer.size());
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint(
        "##cloudStorefront",
        "Storefront / link URL (often same as API for staging)",
        storefrontUrlBuffer.data(), storefrontUrlBuffer.size());

    if (ImGui::Button("Use same as API")) {
      std::memcpy(storefrontUrlBuffer.data(), apiBaseUrlBuffer.data(),
                  storefrontUrlBuffer.size());
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear to defaults")) {
      apiBaseUrlBuffer[0] = '\0';
      storefrontUrlBuffer[0] = '\0';
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply endpoint") && callbacks.applyEndpointOverrides) {
      const juce::String api(apiBaseUrlBuffer.data());
      const juce::String storefront(storefrontUrlBuffer.data());
      callbacks.applyEndpointOverrides(api, storefront);
      lastSyncedApiOverride = api.trim();
      lastSyncedStorefrontOverride = storefront.trim();
      copyToBuffer(apiBaseUrlBuffer, lastSyncedApiOverride);
      copyToBuffer(storefrontUrlBuffer, lastSyncedStorefrontOverride);
    }
  }
}
} // namespace openyourbox::ui
