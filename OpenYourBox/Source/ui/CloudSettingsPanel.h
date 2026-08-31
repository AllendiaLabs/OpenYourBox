#pragma once

#include "../train/CloudSettings.h"
#include "../train/CloudTrainClient.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <array>
#include <functional>

namespace openyourbox::ui {
/**
 * @class CloudSettingsPanel
 * @brief Sign in / sign out of an Allendia account and optional staging endpoints.
 *
 * No checkout, cart, or payment controls are provided. Advanced API / storefront
 * URL fields let operators point Cloud Train at a staging or RunPod host.
 */
class CloudSettingsPanel {
public:
  /** @brief Message-thread actions for the Allendia account surface. */
  struct Callbacks {
    /** @brief Start Allendia device-code sign-in. */
    std::function<void()> linkAccount;
    /** @brief Cancel an in-progress sign-in poll. */
    std::function<void()> cancelLink;
    /** @brief Sign out and clear local credentials. */
    std::function<void()> disconnect;
    /** @brief Launch the Allendia account page in the system browser. */
    std::function<void()> openStorefront;
    /**
     * @brief Persist API and storefront URL overrides (empty = product default).
     * @param apiBaseUrlOverride Staging/RunPod API origin, or empty.
     * @param storefrontUrlOverride Storefront/link origin, or empty.
     */
    std::function<void(const juce::String &apiBaseUrlOverride,
                       const juce::String &storefrontUrlOverride)>
        applyEndpointOverrides;
  };

  /** @brief In-progress device-code sign-in shown in the panel. */
  struct LinkFlow {
    bool active = false;
    juce::String userCode;
    juce::String verificationUrl;
    juce::String message;
  };

  /**
   * @brief Draws Allendia account status, endpoint overrides, and sign-in controls.
   * @param settings Persisted cloud settings.
   * @param flow Current sign-in bootstrap state.
   * @param callbacks Editor-owned actions.
   */
  void render(const train::CloudSettings &settings, const LinkFlow &flow,
              const Callbacks &callbacks);

private:
  /**
   * @brief Copies persisted overrides into ImGui edit buffers when not editing.
   * @param settings Source of persisted override strings.
   */
  void syncEndpointBuffers(const train::CloudSettings &settings);

  std::array<char, 512> apiBaseUrlBuffer{};
  std::array<char, 512> storefrontUrlBuffer{};
  juce::String lastSyncedApiOverride;
  juce::String lastSyncedStorefrontOverride;
  bool endpointBuffersReady = false;
};
} // namespace openyourbox::ui
