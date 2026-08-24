#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace openyourbox::capture {
/** @brief Master/slave role for a Capture Samples pairing session. */
enum class PairingRole {
  /** @brief Not currently in a pairing session. */
  unpaired,
  /** @brief Instance that initiated pairing; owns library and Train. */
  master,
  /** @brief Paired peer with a reduced capture menu. */
  slave
};

/** @brief Complementary recording assignment independent of master/slave. */
enum class CaptureRole {
  /** @brief No Clean/Processed assignment yet. */
  unassigned,
  /** @brief Records this instance's input as clean x. */
  clean,
  /** @brief Records this instance's input as processed y. */
  processed
};

/** @brief High-level pairing lifecycle shown in the Capture UI. */
enum class SyncState {
  unpaired,
  discovering,
  paired,
  recording,
  error
};

/**
 * @struct DiscoveredInstance
 * @brief One localhost peer advertised in the discovery registry.
 */
struct DiscoveredInstance {
  /** @brief Stable instance identifier. */
  juce::String instanceId;
  /** @brief Loopback port the peer listens on. */
  int port = 0;
  /** @brief Human-readable plugin version string. */
  juce::String pluginVersion;
};

/**
 * @class CapturePairing
 * @brief Localhost discovery plus loopback InterprocessConnection control.
 *
 * Each instance listens with `InterprocessConnectionServer` and owns one
 * active `InterprocessConnection` channel. Connection setup, registry I/O,
 * and file-path exchange run on the message thread. The audio thread only
 * reads atomic role/bypass/recording flags.
 */
class CapturePairing final : private juce::Timer {
public:
  /** @brief Invoked on the message thread when a control message arrives. */
  using MessageHandler = std::function<void(const juce::var &)>;

  /** @brief Creates a pairing endpoint with a unique instance identifier. */
  CapturePairing();

  /** @brief Unregisters discovery and closes the control channel. */
  ~CapturePairing() override;

  /** @brief Returns this instance's stable identifier. */
  [[nodiscard]] juce::String getInstanceId() const;

  /**
   * @brief Returns a short hex prefix of an instance UUID for UI labels.
   * @param id Full dashed or compact UUID.
   */
  [[nodiscard]] static juce::String shortInstanceLabel(const juce::String &id);

  /** @brief Returns the current master/slave role. */
  [[nodiscard]] PairingRole getPairingRole() const noexcept;

  /** @brief Returns the current Clean/Processed assignment. */
  [[nodiscard]] CaptureRole getCaptureRole() const noexcept;

  /** @brief Returns the UI-facing sync state. */
  [[nodiscard]] SyncState getSyncState() const noexcept;

  /** @brief Returns true while a pairing session is using default bypass. */
  [[nodiscard]] bool isCaptureBypassEnabled() const noexcept;

  /** @brief Returns true while a synchronized recording is active. */
  [[nodiscard]] bool isRecording() const noexcept;

  /** @brief Returns the current session identifier, or empty when unpaired. */
  [[nodiscard]] juce::String getSessionId() const;

  /** @brief Returns the paired peer identifier, or empty when unpaired. */
  [[nodiscard]] juce::String getPeerInstanceId() const;

  /**
   * @brief Starts advertising so other searching instances can list this peer.
   *
   * Idle loaded instances are not discoverable. The listen socket stays open
   * so a searching instance can still be paired after its editor is closed.
   */
  void beginDiscovery();

  /**
   * @brief Stops advertising. Does not drop an already-established pair.
   */
  void stopDiscovery();

  /**
   * @brief Returns peers that are currently searching, excluding self.
   *
   * Only instances that called `beginDiscovery()` write registry entries.
   */
  [[nodiscard]] std::vector<DiscoveredInstance> listPeers() const;

  /**
   * @brief Connects to a discovered peer and becomes master.
   * @param peer Selected peer from `listPeers()`.
   * @return False when the control channel could not be opened.
   */
  bool pairWith(const DiscoveredInstance &peer);

  /** @brief Closes the control channel and returns to unpaired. */
  void unpair();

  /**
   * @brief Assigns complementary Clean/Processed roles.
   * @param localRole Role for this instance; the peer receives the complement.
   * @return False when the role is unassigned.
   */
  bool setCaptureRole(CaptureRole localRole);

  /**
   * @brief Updates the capture bypass flag (default true).
   * @param enabled Whether graph processing is bypassed during capture.
   */
  void setCaptureBypass(bool enabled);

  /**
   * @brief Sends record_start to the peer and marks local recording active.
   * @param pairId Identifier used for the take.
   * @param sampleRate Host sample rate used for the WAV header.
   * @return False when roles are not complementary or the peer is missing.
   */
  bool startRecording(const juce::String &pairId, double sampleRate);

  /** @brief Sends record_stop and clears the local recording flag. */
  void stopRecording();

  /**
   * @brief Notifies the master that a slave clip file is ready.
   * @param pairId Take identifier.
   * @param path Absolute WAV path.
   */
  void sendClipReady(const juce::String &pairId, const juce::File &path);

  /** @brief Installs a message-thread handler for inbound control JSON. */
  void setMessageHandler(MessageHandler handler);

  /**
   * @brief Returns true when Record is legal (paired + complementary roles).
   */
  [[nodiscard]] bool canRecord() const noexcept;

private:
  class Channel;
  class Server;

  void timerCallback() override;

  /**
   * @brief Writes this instance's advertisement into the discovery registry.
   */
  void publishAdvertisement();

  /** @brief Removes this instance's advertisement file. */
  void retractAdvertisement();

  /** @brief Stops the heartbeat timer and retracts the registry entry. */
  void stopAdvertising();

  /**
   * @brief Sends one JSON control object on the InterprocessConnection.
   * @param object JSON payload.
   */
  void sendJson(const juce::var &object);

  /**
   * @brief Creates a channel for an inbound socket accepted by the server.
   * @return Connection object owned by this pairing endpoint.
   */
  juce::InterprocessConnection *createIncomingChannel();

  /**
   * @brief Handles a newly established channel.
   * @param who Channel that connected.
   */
  void onChannelMade(Channel *who);

  /**
   * @brief Handles a dropped channel.
   * @param who Channel that disconnected.
   */
  void onChannelLost(Channel *who);

  /**
   * @brief Dispatches one inbound control payload.
   * @param who Channel that received the payload.
   * @param message Packed UTF-8 JSON.
   */
  void onChannelMessage(Channel *who, const juce::MemoryBlock &message);

  /** @brief Loopback acceptor that creates inbound channels. */
  std::unique_ptr<Server> server;
  /** @brief Active paired control channel. */
  std::unique_ptr<Channel> channel;
  /** @brief Inbound sockets rejected because a session is already active. */
  std::vector<std::unique_ptr<Channel>> extraChannels;
  /** @brief Protects channel pointer swaps. */
  juce::CriticalSection channelLock;

  /** @brief Stable identifier for this plugin instance. */
  juce::String instanceId;
  /** @brief Current pairing session, empty when unpaired. */
  juce::String sessionId;
  /** @brief Connected peer identifier. */
  juce::String peerInstanceId;
  /** @brief Loopback listen port for incoming slave connections. */
  int listenPort = 0;
  /** @brief Master/slave role. */
  std::atomic<int> pairingRole{static_cast<int>(PairingRole::unpaired)};
  /** @brief Clean/Processed assignment. */
  std::atomic<int> captureRole{static_cast<int>(CaptureRole::unassigned)};
  /** @brief UI-facing sync state. */
  std::atomic<int> syncState{static_cast<int>(SyncState::unpaired)};
  /** @brief Default-on bypass while a capture session is active. */
  std::atomic<bool> captureBypass{true};
  /** @brief True while a synchronized take is being written. */
  std::atomic<bool> recording{false};
  /** @brief Message-thread JSON handler. */
  MessageHandler messageHandler;
  /** @brief Protects session identifiers. */
  mutable juce::CriticalSection stateLock;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CapturePairing)
};
} // namespace openyourbox::capture
