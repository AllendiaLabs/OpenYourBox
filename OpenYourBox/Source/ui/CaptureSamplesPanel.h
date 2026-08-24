#pragma once

#include "../capture/CapturePairing.h"

#include <imgui.h>

#include <functional>
#include <string>
#include <vector>

namespace openyourbox::ui {
/**
 * @class CaptureSamplesPanel
 * @brief In-window Capture Samples menu with master and reduced slave layouts.
 */
class CaptureSamplesPanel {
public:
  /** @brief Message-thread actions emitted by the capture panel. */
  struct Callbacks {
    /** @brief Start searching so this instance is discoverable. */
    std::function<void()> beginDiscovery;
    /** @brief Stop searching and retract the advertisement. */
    std::function<void()> stopDiscovery;
    /** @brief Connect to a listed searching peer (this instance becomes master). */
    std::function<void(const capture::DiscoveredInstance &)> pairWith;
    /** @brief Close the pairing session. */
    std::function<void()> unpair;
    /** @brief Assign Clean or Processed. */
    std::function<void(capture::CaptureRole)> setRole;
    /** @brief Toggle capture bypass. */
    std::function<void(bool)> setBypass;
    /** @brief Whether Record should start host playback when stopped. */
    std::function<bool()> getStartTransportOnRecord;
    /** @brief Persist the Record transport-start preference. */
    std::function<void(bool)> setStartTransportOnRecord;
    /** @brief Start a synchronized take. */
    std::function<void()> startRecording;
    /** @brief Stop the current take and ingest into the library. */
    std::function<void()> stopRecording;
    /** @brief Open the Training Library panel. */
    std::function<void()> openLibrary;
  };

  /**
   * @brief Draws master or reduced slave capture controls.
   * @param pairing Live pairing endpoint for this instance.
   * @param peers Currently searching peers (ignored unless this instance is searching).
   * @param callbacks Editor-owned actions.
   */
  void render(const capture::CapturePairing &pairing,
              const std::vector<capture::DiscoveredInstance> &peers,
              const Callbacks &callbacks);
};
} // namespace openyourbox::ui
