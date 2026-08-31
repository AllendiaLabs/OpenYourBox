#pragma once

#include "../train/CloudSettings.h"
#include "../train/CloudTrainClient.h"

#include <imgui.h>
#include <JuceHeader.h>

#include <functional>

namespace openyourbox::ui {
/**
 * @class CloudSettingsPanel
 * @brief Sign in / sign out of an Allendia account and open account management.
 *
 * No checkout, cart, payment, or staging URL controls are provided.
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
  };

  /** @brief In-progress device-code sign-in shown in the panel. */
  struct LinkFlow {
    bool active = false;
    juce::String userCode;
    juce::String verificationUrl;
    juce::String message;
  };

  /**
   * @brief Draws Allendia account status, entitlement hint, and sign-in controls.
   * @param settings Persisted cloud settings.
   * @param flow Current sign-in bootstrap state.
   * @param callbacks Editor-owned actions.
   */
  void render(const train::CloudSettings &settings, const LinkFlow &flow,
              const Callbacks &callbacks);
};
} // namespace openyourbox::ui
