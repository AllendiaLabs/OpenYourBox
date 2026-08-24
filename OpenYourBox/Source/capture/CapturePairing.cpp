#include "CapturePairing.h"

#include <algorithm>
#include <cmath>

namespace openyourbox::capture {
namespace {
/** @brief Magic number identifying OpenYourBox pairing messages. */
constexpr juce::uint32 pairingMagic = 0x4f594250; // 'OYBP'

/**
 * @brief Returns the discovery registry directory in the system temp folder.
 * @return `OpenYourBox/Pairing` under the temp directory.
 */
juce::File discoveryDirectory() {
  return juce::File::getSpecialLocation(juce::File::tempDirectory)
      .getChildFile("OpenYourBox")
      .getChildFile("Pairing");
}

/**
 * @brief Serializes a JSON object to an InterprocessConnection memory block.
 * @param object JSON payload.
 * @return Packed UTF-8 block.
 */
juce::MemoryBlock jsonToBlock(const juce::var &object) {
  const auto text = juce::JSON::toString(object, true);
  juce::MemoryBlock block;
  block.append(text.toRawUTF8(), static_cast<size_t>(text.getNumBytesAsUTF8()));
  return block;
}
} // namespace

/**
 * @class CapturePairing::Channel
 * @brief One localhost InterprocessConnection owned by the pairing endpoint.
 */
class CapturePairing::Channel final : public juce::InterprocessConnection {
public:
  /**
   * @brief Binds this socket to a pairing endpoint.
   * @param ownerToUse Endpoint that receives connection callbacks.
   */
  explicit Channel(CapturePairing &ownerToUse)
      : juce::InterprocessConnection(true, pairingMagic), owner(ownerToUse) {}

  /** @brief Disconnects before destruction as required by JUCE. */
  ~Channel() override {
    disconnect(-1, juce::InterprocessConnection::Notify::no);
  }

  /** @brief Forwards an established connection to the pairing endpoint. */
  void connectionMade() override { owner.onChannelMade(this); }

  /** @brief Forwards a dropped connection to the pairing endpoint. */
  void connectionLost() override { owner.onChannelLost(this); }

  /**
   * @brief Forwards one packed JSON payload to the pairing endpoint.
   * @param message Packed UTF-8 control JSON.
   */
  void messageReceived(const juce::MemoryBlock &message) override {
    owner.onChannelMessage(this, message);
  }

private:
  /** @brief Pairing endpoint that owns this channel. */
  CapturePairing &owner;
};

/**
 * @class CapturePairing::Server
 * @brief Loopback acceptor that creates inbound pairing channels.
 */
class CapturePairing::Server final : public juce::InterprocessConnectionServer {
public:
  /**
   * @brief Binds this acceptor to a pairing endpoint.
   * @param ownerToUse Endpoint that adopts inbound channels.
   */
  explicit Server(CapturePairing &ownerToUse) : owner(ownerToUse) {}

  /**
   * @brief Creates an inbound channel when a peer connects.
   * @return Connection object owned by the pairing endpoint.
   */
  juce::InterprocessConnection *createConnectionObject() override {
    return owner.createIncomingChannel();
  }

private:
  /** @brief Pairing endpoint that adopts inbound channels. */
  CapturePairing &owner;
};

CapturePairing::CapturePairing() {
  instanceId = juce::Uuid().toDashedString();
  server = std::make_unique<Server>(*this);
  for (int port = 49152; port < 49252; ++port) {
    if (server->beginWaitingForSocket(port, "127.0.0.1")) {
      listenPort = port;
      break;
    }
  }
}

CapturePairing::~CapturePairing() {
  stopTimer();
  retractAdvertisement();
  if (server != nullptr)
    server->stop();
  std::unique_ptr<Channel> active;
  std::vector<std::unique_ptr<Channel>> extras;
  {
    const juce::ScopedLock lock(channelLock);
    active = std::move(channel);
    extras = std::move(extraChannels);
  }
}

juce::String CapturePairing::getInstanceId() const { return instanceId; }

juce::String CapturePairing::shortInstanceLabel(const juce::String &id) {
  const auto compact = id.removeCharacters("-");
  return compact.substring(0, juce::jmin(8, compact.length()));
}

PairingRole CapturePairing::getPairingRole() const noexcept {
  return static_cast<PairingRole>(pairingRole.load(std::memory_order_acquire));
}

CaptureRole CapturePairing::getCaptureRole() const noexcept {
  return static_cast<CaptureRole>(captureRole.load(std::memory_order_acquire));
}

SyncState CapturePairing::getSyncState() const noexcept {
  return static_cast<SyncState>(syncState.load(std::memory_order_acquire));
}

bool CapturePairing::isCaptureBypassEnabled() const noexcept {
  return captureBypass.load(std::memory_order_acquire);
}

bool CapturePairing::isRecording() const noexcept {
  return recording.load(std::memory_order_acquire);
}

juce::String CapturePairing::getSessionId() const {
  const juce::ScopedLock lock(stateLock);
  return sessionId;
}

juce::String CapturePairing::getPeerInstanceId() const {
  const juce::ScopedLock lock(stateLock);
  return peerInstanceId;
}

void CapturePairing::beginDiscovery() {
  if (getPairingRole() != PairingRole::unpaired)
    return;
  syncState.store(static_cast<int>(SyncState::discovering),
                  std::memory_order_release);
  publishAdvertisement();
  startTimer(500);
}

void CapturePairing::stopDiscovery() {
  stopAdvertising();
  if (getSyncState() == SyncState::discovering)
    syncState.store(static_cast<int>(SyncState::unpaired),
                    std::memory_order_release);
}

void CapturePairing::publishAdvertisement() {
  if (listenPort <= 0)
    return;
  const auto directory = discoveryDirectory();
  directory.createDirectory();
  auto object = std::make_unique<juce::DynamicObject>();
  object->setProperty("instanceId", instanceId);
  object->setProperty("port", listenPort);
  object->setProperty("pluginVersion", juce::String("1.0.0"));
  object->setProperty("searching", true);
  object->setProperty("updatedAt", juce::Time::currentTimeMillis());
  directory.getChildFile(instanceId + ".json")
      .replaceWithText(juce::JSON::toString(juce::var(object.release()), true));
}

void CapturePairing::retractAdvertisement() {
  discoveryDirectory().getChildFile(instanceId + ".json").deleteFile();
}

void CapturePairing::stopAdvertising() {
  stopTimer();
  retractAdvertisement();
}

std::vector<DiscoveredInstance> CapturePairing::listPeers() const {
  std::vector<DiscoveredInstance> peers;
  const auto directory = discoveryDirectory();
  if (!directory.isDirectory())
    return peers;
  for (const auto &file :
       directory.findChildFiles(juce::File::findFiles, false, "*.json")) {
    const auto parsed = juce::JSON::parse(file.loadFileAsString());
    if (!parsed.isObject())
      continue;
    DiscoveredInstance peer;
    peer.instanceId = parsed.getProperty("instanceId", {}).toString();
    peer.port = static_cast<int>(parsed.getProperty("port", 0));
    peer.pluginVersion = parsed.getProperty("pluginVersion", {}).toString();
    const auto updated =
        static_cast<juce::int64>(parsed.getProperty("updatedAt", 0));
    if (peer.instanceId == instanceId || peer.port <= 0)
      continue;
    if (!static_cast<bool>(parsed.getProperty("searching", false)))
      continue;
    if (std::abs(juce::Time::currentTimeMillis() - updated) > 5000)
      continue;
    peers.push_back(std::move(peer));
  }
  return peers;
}

bool CapturePairing::pairWith(const DiscoveredInstance &peer) {
  if (peer.port <= 0)
    return false;
  pairingRole.store(static_cast<int>(PairingRole::master), std::memory_order_release);
  {
    const juce::ScopedLock lock(stateLock);
    peerInstanceId = peer.instanceId;
    if (sessionId.isEmpty())
      sessionId = juce::Uuid().toDashedString();
  }
  auto outbound = std::make_unique<Channel>(*this);
  auto *raw = outbound.get();
  {
    const juce::ScopedLock lock(channelLock);
    channel = std::move(outbound);
  }
  if (!raw->connectToSocket("127.0.0.1", peer.port, 1000)) {
    const juce::ScopedLock lock(channelLock);
    if (channel.get() == raw)
      channel.reset();
    return false;
  }
  auto hello = std::make_unique<juce::DynamicObject>();
  hello->setProperty("type", "pair");
  hello->setProperty("sessionId", getSessionId());
  hello->setProperty("instanceId", instanceId);
  hello->setProperty("pluginVersion", juce::String("1.0.0"));
  hello->setProperty("role", "master");
  sendJson(juce::var(hello.release()));
  stopAdvertising();
  syncState.store(static_cast<int>(SyncState::paired), std::memory_order_release);
  return true;
}

void CapturePairing::unpair() {
  recording.store(false, std::memory_order_release);
  std::unique_ptr<Channel> active;
  {
    const juce::ScopedLock lock(channelLock);
    active = std::move(channel);
  }
  if (active != nullptr)
    active->disconnect(-1, juce::InterprocessConnection::Notify::no);
  {
    const juce::ScopedLock lock(stateLock);
    peerInstanceId.clear();
    sessionId.clear();
  }
  pairingRole.store(static_cast<int>(PairingRole::unpaired),
                    std::memory_order_release);
  captureRole.store(static_cast<int>(CaptureRole::unassigned),
                    std::memory_order_release);
  syncState.store(static_cast<int>(SyncState::unpaired),
                  std::memory_order_release);
  stopAdvertising();
}

bool CapturePairing::setCaptureRole(CaptureRole localRole) {
  if (localRole == CaptureRole::unassigned)
    return false;
  captureRole.store(static_cast<int>(localRole), std::memory_order_release);
  auto message = std::make_unique<juce::DynamicObject>();
  message->setProperty("type", "set_capture_role");
  message->setProperty("role", localRole == CaptureRole::clean ? "clean"
                                                               : "processed");
  sendJson(juce::var(message.release()));
  return true;
}

void CapturePairing::setCaptureBypass(bool enabled) {
  captureBypass.store(enabled, std::memory_order_release);
  auto message = std::make_unique<juce::DynamicObject>();
  message->setProperty("type", "set_bypass");
  message->setProperty("enabled", enabled);
  sendJson(juce::var(message.release()));
}

bool CapturePairing::canRecord() const noexcept {
  const auto local = getCaptureRole();
  return getSyncState() == SyncState::paired &&
         (local == CaptureRole::clean || local == CaptureRole::processed);
}

bool CapturePairing::startRecording(const juce::String &pairId, double sampleRate) {
  if (!canRecord())
    return false;
  recording.store(true, std::memory_order_release);
  syncState.store(static_cast<int>(SyncState::recording),
                  std::memory_order_release);
  auto message = std::make_unique<juce::DynamicObject>();
  message->setProperty("type", "record_start");
  message->setProperty("pairId", pairId);
  message->setProperty("sampleRate", sampleRate);
  sendJson(juce::var(message.release()));
  return true;
}

void CapturePairing::stopRecording() {
  recording.store(false, std::memory_order_release);
  if (getSyncState() == SyncState::recording)
    syncState.store(static_cast<int>(SyncState::paired),
                    std::memory_order_release);
  auto message = std::make_unique<juce::DynamicObject>();
  message->setProperty("type", "record_stop");
  sendJson(juce::var(message.release()));
}

void CapturePairing::sendClipReady(const juce::String &pairId,
                                   const juce::File &path) {
  auto message = std::make_unique<juce::DynamicObject>();
  message->setProperty("type", "clip_ready");
  message->setProperty("pairId", pairId);
  message->setProperty("role", getCaptureRole() == CaptureRole::clean
                                   ? "clean"
                                   : "processed");
  message->setProperty("path", path.getFullPathName());
  sendJson(juce::var(message.release()));
}

void CapturePairing::setMessageHandler(MessageHandler handler) {
  messageHandler = std::move(handler);
}

void CapturePairing::sendJson(const juce::var &object) {
  const juce::ScopedLock lock(channelLock);
  if (channel == nullptr || !channel->isConnected())
    return;
  channel->sendMessage(jsonToBlock(object));
}

juce::InterprocessConnection *CapturePairing::createIncomingChannel() {
  auto incoming = std::make_unique<Channel>(*this);
  auto *raw = incoming.get();
  const juce::ScopedLock lock(channelLock);
  if (channel == nullptr || !channel->isConnected())
    channel = std::move(incoming);
  else
    extraChannels.push_back(std::move(incoming));
  return raw;
}

void CapturePairing::onChannelMade(Channel *who) {
  bool extra = false;
  {
    const juce::ScopedLock lock(channelLock);
    extra = channel.get() != who;
  }
  if (extra) {
    who->disconnect(-1, juce::InterprocessConnection::Notify::no);
    return;
  }
  if (getPairingRole() != PairingRole::master)
    pairingRole.store(static_cast<int>(PairingRole::slave),
                      std::memory_order_release);
  if (getSyncState() == SyncState::unpaired ||
      getSyncState() == SyncState::discovering) {
    stopAdvertising();
    syncState.store(static_cast<int>(SyncState::paired),
                    std::memory_order_release);
  }
}

void CapturePairing::onChannelLost(Channel *who) {
  {
    const juce::ScopedLock lock(channelLock);
    if (channel.get() != who)
      return;
  }
  recording.store(false, std::memory_order_release);
  syncState.store(static_cast<int>(SyncState::unpaired),
                  std::memory_order_release);
  pairingRole.store(static_cast<int>(PairingRole::unpaired),
                    std::memory_order_release);
  if (messageHandler) {
    auto lost = std::make_unique<juce::DynamicObject>();
    lost->setProperty("type", "peer_lost");
    messageHandler(juce::var(lost.release()));
  }
}

void CapturePairing::onChannelMessage(Channel *who,
                                      const juce::MemoryBlock &message) {
  {
    const juce::ScopedLock lock(channelLock);
    if (channel.get() != who)
      return;
  }
  const auto parsed = juce::JSON::parse(message.toString());
  if (!parsed.isObject())
    return;
  const auto type = parsed.getProperty("type", {}).toString();
  if (type == "pair") {
    stopAdvertising();
    pairingRole.store(static_cast<int>(PairingRole::slave),
                      std::memory_order_release);
    syncState.store(static_cast<int>(SyncState::paired),
                    std::memory_order_release);
    captureBypass.store(true, std::memory_order_release);
    const juce::ScopedLock lock(stateLock);
    sessionId = parsed.getProperty("sessionId", {}).toString();
    peerInstanceId = parsed.getProperty("instanceId", {}).toString();
  } else if (type == "set_capture_role") {
    const auto role = parsed.getProperty("role", {}).toString();
    captureRole.store(static_cast<int>(role == "clean" ? CaptureRole::processed
                                                       : CaptureRole::clean),
                      std::memory_order_release);
  } else if (type == "set_bypass") {
    captureBypass.store(static_cast<bool>(parsed.getProperty("enabled", true)),
                        std::memory_order_release);
  } else if (type == "record_start") {
    recording.store(true, std::memory_order_release);
    syncState.store(static_cast<int>(SyncState::recording),
                    std::memory_order_release);
  } else if (type == "record_stop") {
    recording.store(false, std::memory_order_release);
    syncState.store(static_cast<int>(SyncState::paired),
                    std::memory_order_release);
  }
  if (messageHandler)
    messageHandler(parsed);
}

void CapturePairing::timerCallback() {
  if (getSyncState() == SyncState::discovering && listenPort > 0)
    publishAdvertisement();
  const juce::ScopedLock lock(channelLock);
  extraChannels.erase(std::remove_if(extraChannels.begin(), extraChannels.end(),
                                     [](const std::unique_ptr<Channel> &item) {
                                       return item == nullptr ||
                                              !item->isConnected();
                                     }),
                      extraChannels.end());
}
} // namespace openyourbox::capture
