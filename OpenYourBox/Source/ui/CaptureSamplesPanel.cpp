#include "CaptureSamplesPanel.h"

namespace openyourbox::ui {
void CaptureSamplesPanel::render(
    const capture::CapturePairing &pairing,
    const std::vector<capture::DiscoveredInstance> &peers,
    const Callbacks &callbacks) {
  const auto role = pairing.getPairingRole();
  const auto sync = pairing.getSyncState();
  const bool slave = role == capture::PairingRole::slave;
  const bool searching = sync == capture::SyncState::discovering;
  ImGui::TextUnformatted(slave ? "Capture Samples (slave)"
                               : "Capture Samples");
  ImGui::Text("This instance: %s",
              capture::CapturePairing::shortInstanceLabel(
                  pairing.getInstanceId())
                  .toRawUTF8());

  const char *syncLabel = "Unpaired";
  if (searching)
    syncLabel = "Searching...";
  else if (sync == capture::SyncState::paired)
    syncLabel = "Paired";
  else if (sync == capture::SyncState::recording)
    syncLabel = "Recording";
  else if (sync == capture::SyncState::error)
    syncLabel = "Error";
  ImGui::Text("Sync: %s", syncLabel);

  if (!slave && role == capture::PairingRole::unpaired) {
    if (!searching) {
      if (ImGui::Button("Search for instances") && callbacks.beginDiscovery)
        callbacks.beginDiscovery();
      ImGui::TextWrapped(
          "Instances appear here only while they are searching. "
          "Click Search for instances on both, then Connect.");
    } else {
      if (ImGui::Button("Stop searching") && callbacks.stopDiscovery)
        callbacks.stopDiscovery();
      if (peers.empty())
        ImGui::TextWrapped(
            "No other instances are searching. On the other instance, "
            "open Capture and click Search for instances.");
      else
        ImGui::TextUnformatted("Searching instances on this machine:");
      for (const auto &peer : peers) {
        ImGui::PushID(peer.instanceId.toRawUTF8());
        ImGui::TextUnformatted(
            capture::CapturePairing::shortInstanceLabel(peer.instanceId)
                .toRawUTF8());
        ImGui::SameLine();
        if (ImGui::SmallButton("Connect") && callbacks.pairWith)
          callbacks.pairWith(peer);
        ImGui::PopID();
      }
    }
  }

  if (sync == capture::SyncState::paired ||
      sync == capture::SyncState::recording) {
    const auto captureRole = pairing.getCaptureRole();
    ImGui::Text("Role: %s",
                captureRole == capture::CaptureRole::clean
                    ? "Clean (x)"
                    : captureRole == capture::CaptureRole::processed
                          ? "Processed (y)"
                          : "Unassigned");
    if (!slave) {
      if (ImGui::Button("This instance: Clean") && callbacks.setRole)
        callbacks.setRole(capture::CaptureRole::clean);
      ImGui::SameLine();
      if (ImGui::Button("This instance: Processed") && callbacks.setRole)
        callbacks.setRole(capture::CaptureRole::processed);
    }
    bool bypass = pairing.isCaptureBypassEnabled();
    if (ImGui::Checkbox("Bypass graph (default)", &bypass) &&
        callbacks.setBypass)
      callbacks.setBypass(bypass);
    bool startTransport = true;
    if (callbacks.getStartTransportOnRecord)
      startTransport = callbacks.getStartTransportOnRecord();
    if (ImGui::Checkbox("Start transport if stopped", &startTransport) &&
        callbacks.setStartTransportOnRecord)
      callbacks.setStartTransportOnRecord(startTransport);

    if (!slave) {
      if (sync != capture::SyncState::recording) {
        const bool canRecord =
            captureRole != capture::CaptureRole::unassigned;
        if (!canRecord)
          ImGui::BeginDisabled();
        if (ImGui::Button("Record") && callbacks.startRecording)
          callbacks.startRecording();
        if (!canRecord) {
          ImGui::EndDisabled();
          ImGui::TextDisabled("Assign complementary Clean/Processed roles.");
        }
      } else if (ImGui::Button("Stop") && callbacks.stopRecording) {
        callbacks.stopRecording();
      }
    } else if (sync == capture::SyncState::recording) {
      ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.25f, 1.0f),
                         "Recording under master control");
    }

    if (ImGui::Button("Unpair") && callbacks.unpair)
      callbacks.unpair();
  }

  if (!slave && ImGui::Button("Open Library") && callbacks.openLibrary)
    callbacks.openLibrary();
}
} // namespace openyourbox::ui
