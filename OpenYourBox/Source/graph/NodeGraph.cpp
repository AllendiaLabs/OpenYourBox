#include "NodeGraph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <torch/script.h>

namespace {
void migrateLegacyMixerNode(openyourbox::graph::GraphNode &node,
                            const juce::String &storedType);

void normalizeMergeNodeProperties(openyourbox::graph::GraphNode &node);

void refreshPropagatedPinShapes(openyourbox::graph::NodeGraph &graph);
void refreshPropagatedPinShapesCore(openyourbox::graph::NodeGraph &graph);
void refreshOutputCopyShapes(openyourbox::graph::NodeGraph &graph);

/**
 * @brief Returns true when @p id is a member of @p group, including nested groups.
 * @param graph Graph owning groups and nodes.
 * @param groupId Ancestor group.
 * @param id Node or nested group identifier.
 */
bool groupOwnsId(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
                 std::int32_t id) {
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return false;
  for (const auto member : group->memberIds) {
    if (member == id)
      return true;
    if (graph.findGroup(member) != nullptr && groupOwnsId(graph, member, id))
      return true;
  }
  return false;
}

/**
 * @brief Walks parent groups from @p groupId toward the canvas root.
 * @param graph Graph owning groups.
 * @param groupId Starting group.
 * @return Depth 1 for a root group.
 */
int groupDepth(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId) {
  int depth = 0;
  auto current = groupId;
  std::unordered_set<std::int32_t> visiting;
  while (const auto *group = graph.findGroup(current)) {
    if (!visiting.insert(current).second)
      break;
    ++depth;
    if (!group->parentGroupId.has_value())
      break;
    current = *group->parentGroupId;
  }
  return depth;
}

/**
 * @brief Collects leaf node ids owned by @p groupId into @p out.
 * @param graph Graph owning groups and nodes.
 * @param groupId Ancestor group.
 * @param out Destination list.
 */
void collectLeaves(const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
                   std::vector<std::int32_t> &out) {
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return;
  for (const auto member : group->memberIds) {
    if (graph.findNode(member) != nullptr)
      out.push_back(member);
    else
      collectLeaves(graph, member, out);
  }
}

/**
 * @brief Collects every node and nested group owned by @p groupId.
 * @param graph Graph owning membership.
 * @param groupId Ancestor group.
 * @param nodeIds Destination node identifiers.
 * @param groupIds Destination group identifiers, including @p groupId.
 */
void collectGroupSubtree(const openyourbox::graph::NodeGraph &graph,
                         std::int32_t groupId,
                         std::unordered_set<std::int32_t> &nodeIds,
                         std::unordered_set<std::int32_t> &groupIds) {
  if (!groupIds.insert(groupId).second)
    return;
  const auto *group = graph.findGroup(groupId);
  if (group == nullptr)
    return;
  for (const auto member : group->memberIds) {
    if (graph.findNode(member) != nullptr)
      nodeIds.insert(member);
    else
      collectGroupSubtree(graph, member, nodeIds, groupIds);
  }
}

/**
 * @brief Signal I/O used to stack independent group copies in series.
 *
 * Uses the group's interface pins (unconnected internally) rather than
 * currently attached external cables, and omits TCN/BlackBox control inputs
 * so optional conditioning does not block a legal audio through-path.
 * @param graph Graph owning membership and pins.
 * @param groupId Target group.
 * @param inputs Destination list of chainable external inputs.
 * @param outputs Destination list of chainable external outputs.
 */
void appendSerialChainPorts(
    const openyourbox::graph::NodeGraph &graph, std::int32_t groupId,
    std::vector<openyourbox::graph::GroupBoundaryPort> &inputs,
    std::vector<openyourbox::graph::GroupBoundaryPort> &outputs) {
  using openyourbox::graph::PinKind;
  using openyourbox::graph::isControlInputPin;
  for (const auto &port : graph.groupInterfacePorts(groupId)) {
    const auto *pin = graph.findPin(port.memberPinId);
    if (pin != nullptr && isControlInputPin(*pin))
      continue;
    if (port.kind == PinKind::input)
      inputs.push_back(port);
    else
      outputs.push_back(port);
  }
}

/**
 * @brief User-facing reason when copies cannot form a serial stack.
 * @param inputs Chainable external inputs.
 * @param outputs Chainable external outputs.
 * @return Empty when pairing is legal.
 */
std::string serialChainRefusal(
    const std::vector<openyourbox::graph::GroupBoundaryPort> &inputs,
    const std::vector<openyourbox::graph::GroupBoundaryPort> &outputs) {
  if (inputs.empty() || outputs.empty() || inputs.size() != outputs.size())
    return "Copies requires matching external inputs and outputs that can "
           "chain in series";
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!outputs[index].shape.isCompatibleWith(inputs[index].shape))
      return "Copies cannot chain: group outputs are not compatible with "
             "group inputs";
  }
  return {};
}

/**
 * @brief Returns the parent group of a node or nested group.
 * @param graph Graph owning membership.
 * @param memberId Node or group identifier.
 */
std::optional<std::int32_t>
parentGroupOf(const openyourbox::graph::NodeGraph &graph, std::int32_t memberId) {
  if (const auto *node = graph.findNode(memberId))
    return node->parentGroupId;
  if (const auto *group = graph.findGroup(memberId))
    return group->parentGroupId;
  return std::nullopt;
}

/**
 * @brief Axis-aligned bounds of a node or nested group.
 * @param graph Graph owning layout.
 * @param memberId Node or group identifier.
 */
std::optional<std::pair<juce::Point<float>, juce::Point<float>>>
memberBounds(const openyourbox::graph::NodeGraph &graph, std::int32_t memberId) {
  if (const auto *node = graph.findNode(memberId))
    return std::make_pair(node->position, node->size);
  if (const auto *group = graph.findGroup(memberId))
    return std::make_pair(group->position, group->size);
  return std::nullopt;
}

/**
 * @brief Writes the stored position of a node or nested group.
 * @param graph Graph to mutate.
 * @param memberId Node or group identifier.
 * @param position New stored position.
 */
void setItemStoredPosition(openyourbox::graph::NodeGraph &graph,
                           std::int32_t memberId, juce::Point<float> position) {
  if (auto *node = graph.findNode(memberId))
    node->position = position;
  else if (auto *group = graph.findGroup(memberId))
    group->position = position;
}

/**
 * @brief Canvas-space origin of a node or nested group.
 * @param graph Graph owning layout.
 * @param memberId Node or group identifier.
 */
juce::Point<float> itemWorldPosition(const openyourbox::graph::NodeGraph &graph,
                                     std::int32_t memberId) {
  if (graph.findNode(memberId) != nullptr)
    return graph.worldPositionOfNode(memberId);
  return graph.worldPositionOfGroup(memberId);
}

/**
 * @brief Writes a canvas-space origin into the item's current parent space.
 * @param graph Graph to mutate.
 * @param memberId Node or group identifier.
 * @param worldPoint Canvas-space origin.
 * @param parent Group that currently owns the item, or empty at the root.
 */
void storeWorldPosition(openyourbox::graph::NodeGraph &graph,
                        std::int32_t memberId, juce::Point<float> worldPoint,
                        std::optional<std::int32_t> parent) {
  const auto stored =
      parent.has_value() ? graph.worldToGroupLocal(*parent, worldPoint)
                         : worldPoint;
  setItemStoredPosition(graph, memberId, stored);
}

/**
 * @brief Ensures Activation and TCN nodes expose a validated Gain property.
 * @param node Loaded or newly created processing node.
 */
void normalizeGainProperty(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  if (node.type != NodeType::activation && node.type != NodeType::tcn)
    return;
  for (auto &property : node.properties) {
    if (property.key != "gain")
      continue;
    property.kind = PropertyKind::real;
    property.label = "Gain";
    property.floatMinimum = openyourbox::graph::gainMinimum;
    property.floatMaximum = openyourbox::graph::gainMaximum;
    if (property.floatValue <= 0.0f)
      property.floatValue = openyourbox::graph::gainDefault;
    property.setFloatValue(property.floatValue);
    return;
  }
  openyourbox::graph::NodeProperty gain;
  gain.key = "gain";
  gain.label = "Gain";
  gain.kind = PropertyKind::real;
  gain.floatValue = openyourbox::graph::gainDefault;
  gain.floatMinimum = openyourbox::graph::gainMinimum;
  gain.floatMaximum = openyourbox::graph::gainMaximum;
  node.properties.push_back(std::move(gain));
}

/**
 * @brief Ensures Phase 3 TCN/Activation fields exist on loaded nodes.
 * @param node Loaded or newly created processing node.
 */
void normalizePhase3Node(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::defaultDilationGrowth;
  using openyourbox::graph::isTrainableType;
  using openyourbox::graph::minimumDilationGrowth;
  using openyourbox::graph::unlimitedPropertyMaximum;
  using openyourbox::graph::widenIntegerPropertyBounds;

  auto ensureActivationChoices = [](NodeProperty &property) {
    if (property.key != "activation")
      return;
    property.kind = PropertyKind::choice;
    property.choices = {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"};
    property.minimum = 0;
    property.maximum = 4;
    property.setValue(property.value);
  };

  if (node.type == NodeType::activation) {
    for (auto &property : node.properties)
      ensureActivationChoices(property);
    if (node.hasWeights)
      node.armedForTraining = node.armedForTraining || true;
  }

  if (node.type != NodeType::tcn)
    return;

  auto ensureInt = [&node](const char *key, const char *label, int value,
                           int minimum, int maximum) {
    for (auto &property : node.properties) {
      if (property.key == key) {
        property.minimum = minimum;
        property.maximum = maximum;
        property.setValue(property.value);
        return;
      }
    }
    NodeProperty property;
    property.key = key;
    property.label = label;
    property.value = value;
    property.minimum = minimum;
    property.maximum = maximum;
    node.properties.push_back(property);
  };
  ensureInt("dilation_growth", "Dilation growth", defaultDilationGrowth,
            minimumDilationGrowth, unlimitedPropertyMaximum);
  ensureInt("residual", "Residual", 0, 0, 1);
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
  node.properties.erase(
      std::remove_if(node.properties.begin(), node.properties.end(),
                     [](const NodeProperty &property) {
                       return property.key == "dilation";
                     }),
      node.properties.end());
  for (auto &property : node.properties)
    ensureActivationChoices(property);
  if (node.hasWeights)
    node.armedForTraining = true;
}

/**
 * @brief Restores Knob/XY widths and TCN/BlackBox control pin roles.
 * @param node Loaded source node.
 */
void normalizeConditioningPins(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::controlPinLabel;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  if (openyourbox::graph::isConditioningSourceType(node.type)) {
    for (auto &pin : node.outputs) {
      if (pin.shape.channels <= 0)
        pin.shape.channels = 1;
    }
    return;
  }
  if (node.type != openyourbox::graph::NodeType::tcn &&
      node.type != openyourbox::graph::NodeType::blackBox)
    return;
  for (auto &pin : node.inputs) {
    if (!isControlInputPin(pin))
      continue;
    pin.label = controlPinLabel;
    pin.shape = flexibleTensorShape();
  }
}

/**
 * @brief Reads an integer node property when present.
 * @param node Graph node to inspect.
 * @param key Property key.
 * @param fallback Value returned when the property is absent.
 * @return Stored property value or @p fallback.
 */
int readNodeProperty(const openyourbox::graph::GraphNode &node, const char *key,
                     int fallback = 0) {
  const auto property = std::find_if(
      node.properties.begin(), node.properties.end(),
      [key](const openyourbox::graph::NodeProperty &candidate) {
        return candidate.key == key;
      });
  return property == node.properties.end() ? fallback : property->value;
}

/**
 * @brief Returns the configured merge operating mode for one node.
 * @param node Merge or legacy mixer node.
 * @return Merge mode index.
 */
int mergeModeFor(const openyourbox::graph::GraphNode &node) {
  return std::clamp(readNodeProperty(node, "mode", 0), 0, 2);
}

/**
 * @brief Returns whether two Merge input widths can share add/multiply.
 * @param left First width, or zero when unknown.
 * @param right Second width, or zero when unknown.
 * @return True when the widths match or either side can broadcast from 1.
 */
bool channelsAreBroadcastCompatible(int left, int right) noexcept {
  return left == 0 || right == 0 || left == right || left == 1 || right == 1;
}

int resolvePinChannels(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t pinId,
                       std::unordered_set<std::int32_t> &visiting);

/**
 * @brief Computes merge output channels from currently connected inputs.
 * @param graph Graph document used for upstream resolution.
 * @param node Merge node to inspect.
 * @param visiting Active pin-resolution stack for cycle detection.
 * @return Output channel count, or zero when unknown.
 */
int computeMergeOutputChannels(const openyourbox::graph::NodeGraph &graph,
                               const openyourbox::graph::GraphNode &node,
                               std::unordered_set<std::int32_t> &visiting) {
  using openyourbox::graph::MergeMode;
  const auto mode = mergeModeFor(node);
  std::vector<int> connectedChannels;
  connectedChannels.reserve(node.inputs.size());
  for (const auto &inputPin : node.inputs) {
    for (const auto &link : graph.getLinks()) {
      if (link.destinationPinId != inputPin.id)
        continue;
      visiting.clear();
      const auto channels =
          resolvePinChannels(graph, link.sourcePinId, visiting);
      if (channels > 0)
        connectedChannels.push_back(channels);
      break;
    }
  }
  if (connectedChannels.empty())
    return 0;
  if (mode == static_cast<int>(MergeMode::concatenate)) {
    int outputChannels = 0;
    for (const auto channels : connectedChannels)
      outputChannels += channels;
    return outputChannels;
  }
  int width = 0;
  for (const auto channels : connectedChannels) {
    if (!channelsAreBroadcastCompatible(width, channels))
      return 0;
    width = std::max(width, channels);
  }
  return width;
}

int resolvePinChannels(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t pinId,
                       std::unordered_set<std::int32_t> &visiting) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PinKind;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  if (!visiting.insert(pinId).second)
    return 0;

  const auto *pin = graph.findPin(pinId);
  if (pin == nullptr)
    return 0;
  if (pin->shape.channels > 0)
    return pin->shape.channels;

  const auto ownerId = graph.findNodeForPin(pinId);
  if (!ownerId.has_value())
    return 0;
  const auto *node = graph.findNode(*ownerId);
  if (node == nullptr)
    return 0;

  if (pin->kind == PinKind::input) {
    for (const auto &link : graph.getLinks()) {
      if (link.destinationPinId != pinId)
        continue;
      return resolvePinChannels(graph, link.sourcePinId, visiting);
    }
    return 0;
  }

  switch (node->type) {
  case NodeType::audioInput:
    return openyourbox::graph::hostIoChannelsFromChoice(
        readNodeProperty(*node, "channels",
                         openyourbox::graph::hostIoChoiceFromChannels(2)));
  case NodeType::linear: {
    const auto features = readNodeProperty(*node, "features", 0);
    return features > 0 ? features : 0;
  }
  case NodeType::convolution: {
    const auto channels = readNodeProperty(*node, "channels", 0);
    return channels > 0 ? channels : 0;
  }
  case NodeType::convTranspose: {
    const auto channels = readNodeProperty(*node, "channels", 0);
    return channels > 0 ? channels : 0;
  }
  case NodeType::merge:
    visiting.erase(pinId);
    return computeMergeOutputChannels(graph, *node, visiting);
  case NodeType::activation:
  case NodeType::tcn:
  case NodeType::batchNorm:
  case NodeType::audioOutput:
    if (node->inputs.empty())
      return 0;
    return resolvePinChannels(graph, node->inputs.front().id, visiting);
  case NodeType::knobInput:
  case NodeType::xyTrackpad:
    return 1;
  case NodeType::pqmfAnalysis: {
    const auto nBand = readNodeProperty(*node, "n_band", defaultPqmfBands);
    const auto audioChannels =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return nBand * std::max(1, audioChannels);
  }
  case NodeType::pqmfSynthesis: {
    const auto nBand = readNodeProperty(*node, "n_band", defaultPqmfBands);
    const auto bands =
        node->inputs.empty()
            ? 0
            : resolvePinChannels(graph, node->inputs.front().id, visiting);
    return nBand > 0 ? std::max(1, bands / nBand) : bands;
  }
  case NodeType::rateConv: {
    const auto channels = readNodeProperty(*node, "channels", 0);
    return channels > 0 ? channels : 0;
  }
  case NodeType::variationalBottleneck:
    return readNodeProperty(*node, "latent_size", defaultLatentSize);
  case NodeType::noiseSynthesizer:
    if (node->inputs.empty())
      return 0;
    return resolvePinChannels(graph, node->inputs.front().id, visiting);
  default:
    return 0;
  }
}

/**
 * @brief Updates the declared output channels on one merge node.
 * @param graph Graph document to mutate.
 * @param node Merge node to refresh.
 */
void updateMergeOutputShape(openyourbox::graph::NodeGraph &graph,
                            openyourbox::graph::GraphNode &node) {
  if (node.outputs.empty())
    return;
  std::unordered_set<std::int32_t> visiting;
  node.outputs.front().shape.channels =
      computeMergeOutputChannels(graph, node, visiting);
}

/**
 * @brief Refreshes output channel declarations on every merge node.
 * @param graph Graph document to mutate.
 */
void refreshAllMergeOutputShapes(openyourbox::graph::NodeGraph &graph) {
  refreshPropagatedPinShapes(graph);
}

/**
 * @brief Returns true when downstream consumers accept the merge output width.
 * @param graph Graph document to inspect.
 * @param merge Merge node whose output width was computed.
 * @return False when a connected destination requires a different channel count.
 */
bool mergeDownstreamIsCompatible(const openyourbox::graph::NodeGraph &graph,
                               const openyourbox::graph::GraphNode &merge) {
  if (merge.outputs.empty())
    return true;
  const auto outputChannels = merge.outputs.front().shape.channels;
  if (outputChannels == 0)
    return true;

  for (const auto &link : graph.getLinks()) {
    if (link.sourcePinId != merge.outputs.front().id)
      continue;
    const auto *destinationPin = graph.findPin(link.destinationPinId);
    if (destinationPin == nullptr)
      continue;
    std::unordered_set<std::int32_t> visiting;
    const auto destinationChannels =
        destinationPin->shape.channels > 0
            ? destinationPin->shape.channels
            : resolvePinChannels(graph, link.destinationPinId, visiting);
    if (destinationChannels > 0 && destinationChannels != outputChannels)
      return false;
  }
  return true;
}

/**
 * @brief Validates a new merge input for add/multiply channel agreement.
 * @param graph Graph document to inspect.
 * @param merge Destination merge node.
 * @param newSourcePinId Source pin proposed for one merge input.
 * @return False when add/multiply mode would mix incompatible channel counts.
 */
bool mergeInputConnectionIsValid(const openyourbox::graph::NodeGraph &graph,
                                 const openyourbox::graph::GraphNode &merge,
                                 std::int32_t newSourcePinId) {
  using openyourbox::graph::MergeMode;
  const auto concatenate =
      mergeModeFor(merge) == static_cast<int>(MergeMode::concatenate);
  std::unordered_set<std::int32_t> visiting;
  const auto newChannels = resolvePinChannels(graph, newSourcePinId, visiting);
  const auto *newSourcePin = graph.findPin(newSourcePinId);
  for (const auto &inputPin : merge.inputs) {
    for (const auto &link : graph.getLinks()) {
      if (link.destinationPinId != inputPin.id)
        continue;
      visiting.clear();
      const auto existingChannels =
          resolvePinChannels(graph, link.sourcePinId, visiting);
      if (!concatenate && newChannels > 0 && existingChannels > 0 &&
          !channelsAreBroadcastCompatible(existingChannels, newChannels))
        return false;
      const auto *existingSource = graph.findPin(link.sourcePinId);
      if (newSourcePin != nullptr && existingSource != nullptr) {
        auto left = existingSource->shape;
        auto right = newSourcePin->shape;
        left.channels = 0;
        right.channels = 0;
        if (!left.isCompatibleWith(right))
          return false;
      }
      break;
    }
  }
  return true;
}

const openyourbox::graph::Pin *
findConnectedSourcePin(const openyourbox::graph::NodeGraph &graph,
                       std::int32_t destinationPinId) {
  for (const auto &link : graph.getLinks()) {
    if (link.destinationPinId == destinationPinId)
      return graph.findPin(link.sourcePinId);
  }
  return nullptr;
}

/**
 * @brief Returns the first connected tensor source shape, or a wildcard.
 * @param graph Graph document to inspect.
 * @param node Node whose non-control inputs are scanned.
 */
openyourbox::graph::ShapeSignature
firstConnectedTensorSource(const openyourbox::graph::NodeGraph &graph,
                           const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  for (const auto &pin : node.inputs) {
    if (isControlInputPin(pin))
      continue;
    if (const auto *source = findConnectedSourcePin(graph, pin.id))
      return source->shape;
  }
  return flexibleTensorShape();
}

/**
 * @brief Copies inherited hop rate and nBand onto one tensor pin.
 * @param pin Pin to update.
 * @param incoming Upstream shape.
 * @param copyChannels When true, also rewrite @c pin.shape.channels.
 */
void inheritTensorFields(openyourbox::graph::Pin &pin,
                         const openyourbox::graph::ShapeSignature &incoming,
                         bool copyChannels = false) {
  pin.shape.temporalRate = incoming.temporalRate;
  pin.shape.nBand = incoming.nBand;
  if (copyChannels)
    pin.shape.channels = incoming.channels;
}

/**
 * @brief Writes Conv1D detail text from stride.
 * @param node Convolution node to update.
 */
void updateConv1dDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::isConvolutionType;
  if (!isConvolutionType(node.type))
    return;
  const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
  if (stride <= 1)
    node.detail = "Temporal convolution";
  else
    node.detail = "Downsample x" + std::to_string(stride);
}

/**
 * @brief Writes ConvTranspose1d detail text from stride.
 * @param node Transposed convolution node to update.
 */
void updateConvTransposeDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  if (node.type != NodeType::convTranspose)
    return;
  const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
  if (stride <= 1)
    node.detail = "Temporal upsampling";
  else
    node.detail = "Upsample x" + std::to_string(stride);
}

/**
 * @brief Returns the declared host I/O mode on an Audio Input/Output node.
 * @param node Host boundary node.
 */
openyourbox::graph::HostIoMode
readHostIoMode(const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::HostIoMode;
  using openyourbox::graph::hostIoModeFromChannels;
  using openyourbox::graph::hostIoModeFromChoice;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return HostIoMode::stereo;
  for (const auto &property : node.properties) {
    if (property.key != "channels")
      continue;
    if (property.kind == openyourbox::graph::PropertyKind::choice) {
      const auto legacyPair =
          property.maximum <= 1 || property.choices.size() == 2;
      return hostIoModeFromChoice(property.value, legacyPair);
    }
    return hostIoModeFromChannels(std::clamp(property.value, 1, 2));
  }
  for (const auto &pin : node.outputs) {
    if (pin.shape.channels == 1 || pin.shape.channels == 2)
      return hostIoModeFromChannels(pin.shape.channels);
  }
  for (const auto &pin : node.inputs) {
    if (pin.shape.channels == 1 || pin.shape.channels == 2)
      return hostIoModeFromChannels(pin.shape.channels);
  }
  return HostIoMode::stereo;
}

/**
 * @brief Returns the mono/stereo width declared on a host I/O node.
 * @param node Audio Input or Audio Output node.
 */
int readHostIoChannels(const openyourbox::graph::GraphNode &node) {
  return openyourbox::graph::hostIoChannelsFromMode(readHostIoMode(node));
}

/**
 * @brief Writes host I/O detail text from the Channels choice.
 * @param node Audio Input or Audio Output node.
 */
void updateHostIoDetail(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::hostIoModeDetail;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  node.detail =
      hostIoModeDetail(readHostIoMode(node), node.type == NodeType::audioInput);
}

/**
 * @brief Applies the Channels property to host I/O pin shapes and detail.
 * @param node Audio Input or Audio Output node.
 */
void applyHostIoChannels(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  const auto channels = readHostIoChannels(node);
  for (auto &pin : node.inputs) {
    pin.shape.channels = channels;
    pin.shape.temporalRate = 1;
    pin.shape.nBand = 0;
  }
  for (auto &pin : node.outputs) {
    pin.shape.channels = channels;
    pin.shape.temporalRate = 1;
    pin.shape.nBand = 0;
  }
  updateHostIoDetail(node);
}

/**
 * @brief Ensures Audio Input/Output expose Mono|Mirrored|Stereo Channels.
 * @param node Host boundary node to normalize.
 */
void normalizeHostIoProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::HostIoMode;
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::hostIoChoiceFromMode;
  using openyourbox::graph::hostIoModeFromChannels;
  using openyourbox::graph::hostIoModeFromChoice;
  using openyourbox::graph::isFixedIoType;
  if (!isFixedIoType(node.type))
    return;
  const auto inferred = readHostIoMode(node);
  for (auto &property : node.properties) {
    if (property.key != "channels")
      continue;
    const auto wasChoice = property.kind == PropertyKind::choice;
    const auto legacyPair =
        wasChoice && (property.maximum <= 1 || property.choices.size() == 2);
    HostIoMode mode = inferred;
    if (wasChoice)
      mode = hostIoModeFromChoice(property.value, legacyPair);
    else if (property.value == 1 || property.value == 2)
      mode = hostIoModeFromChannels(property.value);
    property.label = "Mode";
    property.kind = PropertyKind::choice;
    property.choices = {"Mono", "Mirrored", "Stereo"};
    property.minimum = 0;
    property.maximum = 2;
    property.setValue(hostIoChoiceFromMode(mode));
    applyHostIoChannels(node);
    return;
  }
  NodeProperty channels;
  channels.key = "channels";
  channels.label = "Mode";
  channels.kind = PropertyKind::choice;
  channels.choices = {"Mono", "Mirrored", "Stereo"};
  channels.minimum = 0;
  channels.maximum = 2;
  channels.setValue(hostIoChoiceFromMode(inferred));
  node.properties.push_back(std::move(channels));
  applyHostIoChannels(node);
}

/**
 * @brief Migrates legacy Rate Conv / Direction upsample nodes and ensures
 *        Conv1D stride properties exist.
 * @param node Loaded or newly created convolution node.
 */
void storeFloatVector(juce::ValueTree &tree, const char *key,
                      const std::vector<float> &values) {
  if (values.empty())
    return;
  juce::MemoryBlock block(values.data(), values.size() * sizeof(float));
  tree.setProperty(key, block.toBase64Encoding(), nullptr);
}

void loadFloatVector(const juce::ValueTree &tree, const char *key,
                     std::vector<float> &values) {
  if (!tree.hasProperty(key))
    return;
  juce::MemoryBlock block;
  if (!block.fromBase64Encoding(tree[key].toString()))
    return;
  const auto count = block.getSize() / sizeof(float);
  values.resize(count);
  if (count > 0)
    std::memcpy(values.data(), block.getData(), count * sizeof(float));
}

void copyTensorToVector(const torch::Tensor &tensor, std::vector<float> &out) {
  if (!tensor.defined() || tensor.numel() < 1) {
    out.clear();
    return;
  }
  auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
  const auto *data = cpu.data_ptr<float>();
  out.assign(data, data + cpu.numel());
}

void copyCompactnessFromArtifact(openyourbox::graph::GraphNode &node,
                                 const std::string &path) {
  using openyourbox::graph::clearCompactness;
  if (path.empty()) {
    clearCompactness(node);
    return;
  }
  try {
    auto module = torch::jit::load(path, torch::kCPU);
    module.eval();
    bool ready = false;
    if (module.hasattr("compactness_ready"))
      ready = module.attr("compactness_ready").toTensor().item<float>() > 0.5f;
    node.compactnessReady = ready;
    if (!ready) {
      node.latentMean.clear();
      node.latentPca.clear();
      node.cumulativeVariance.clear();
      return;
    }
    if (module.hasattr("latent_mean"))
      copyTensorToVector(module.attr("latent_mean").toTensor(), node.latentMean);
    if (module.hasattr("latent_pca"))
      copyTensorToVector(module.attr("latent_pca").toTensor(), node.latentPca);
    if (module.hasattr("cumulative_variance"))
      copyTensorToVector(module.attr("cumulative_variance").toTensor(),
                         node.cumulativeVariance);
  } catch (const std::exception &) {
    clearCompactness(node);
  }
}

void normalizeBottleneckProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultBottleneckKernelSize;
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  if (node.type != NodeType::variationalBottleneck)
    return;
  for (auto &property : node.properties) {
    if (property.key == "kernel_size") {
      property.minimum = minimumPositiveProperty;
      property.maximum = unlimitedPropertyMaximum;
      if (property.value < 1)
        property.value = defaultBottleneckKernelSize;
      return;
    }
  }
  NodeProperty kernel;
  kernel.key = "kernel_size";
  kernel.label = "Kernel Size";
  kernel.value = defaultBottleneckKernelSize;
  kernel.minimum = minimumPositiveProperty;
  kernel.maximum = unlimitedPropertyMaximum;
  node.properties.push_back(std::move(kernel));
}

/**
 * @brief Removes persisted upper bounds on numeric node properties.
 * @param node Loaded or newly created graph node.
 */
void normalizePropertyBounds(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::widenIntegerPropertyBounds;
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
}

void normalizeConvolutionProperties(openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeProperty;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::PropertyKind;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isShapePassthroughType;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::minimumPositiveProperty;
  using openyourbox::graph::unlimitedPropertyMaximum;
  using openyourbox::graph::widenIntegerPropertyBounds;
  if (node.type == NodeType::rateConv) {
    node.type = NodeType::convolution;
    if (node.label == "Rate Conv")
      node.label = "Conv1D";
  }
  if (node.type == NodeType::convolution) {
    int direction = 0;
    for (const auto &property : node.properties) {
      if (property.key == "direction")
        direction = property.value;
    }
    if (direction != 0) {
      node.type = NodeType::convTranspose;
      node.label = "ConvTranspose1d";
    }
    node.properties.erase(
        std::remove_if(node.properties.begin(), node.properties.end(),
                       [](const NodeProperty &property) {
                         return property.key == "direction";
                       }),
        node.properties.end());
  }
  auto resetFlexiblePin = [&](openyourbox::graph::Pin &pin) {
    if (isControlInputPin(pin))
      return;
    if (pin.shape.temporalRate == 1 && pin.shape.nBand == 0) {
      const auto channels = pin.shape.channels;
      pin.shape = flexibleTensorShape(channels);
    }
  };
  if (isShapePassthroughType(node.type) ||
      node.type == NodeType::variationalBottleneck ||
      node.type == NodeType::batchNorm) {
    for (auto &pin : node.inputs)
      resetFlexiblePin(pin);
    for (auto &pin : node.outputs)
      resetFlexiblePin(pin);
  }
  if (node.type == NodeType::convTranspose) {
    for (auto &pin : node.inputs)
      resetFlexiblePin(pin);
    for (auto &pin : node.outputs)
      resetFlexiblePin(pin);
    auto ensureInt = [&node](const char *key, const char *label, int value,
                             int minimum) {
      for (auto &property : node.properties) {
        if (property.key == key) {
          property.minimum = minimum;
          property.maximum = unlimitedPropertyMaximum;
          return;
        }
      }
      node.properties.push_back(
          NodeProperty{key, label, value, minimum, unlimitedPropertyMaximum});
    };
    ensureInt("channels", "Channels", 2, minimumPositiveProperty);
    ensureInt("kernel_size", "Kernel Size", 3, minimumPositiveProperty);
    ensureInt("dilation", "Dilation", 1, minimumPositiveProperty);
    ensureInt("stride", "Stride", 1, minimumPositiveProperty);
    {
      bool hasWeightNorm = false;
      for (auto &property : node.properties) {
        if (property.key == "weight_norm") {
          property.minimum = 0;
          property.maximum = 1;
          hasWeightNorm = true;
          break;
        }
      }
      if (!hasWeightNorm)
        node.properties.push_back(
            NodeProperty{"weight_norm", "Weight Norm", 0, 0, 1});
    }
    for (auto &property : node.properties)
      widenIntegerPropertyBounds(property);
    updateConvTransposeDetail(node);
    return;
  }
  if (node.type != NodeType::convolution)
    return;

  auto ensureInt = [&node](const char *key, const char *label, int value,
                           int minimum,
                           PropertyKind kind = PropertyKind::integer,
                           std::vector<std::string> choices = {}) {
    for (auto &property : node.properties) {
      if (property.key == key) {
        property.minimum = minimum;
        property.maximum = unlimitedPropertyMaximum;
        return;
      }
    }
    node.properties.push_back(NodeProperty{key, label, value, minimum,
                                           unlimitedPropertyMaximum, kind,
                                           std::move(choices)});
  };
  ensureInt("channels", "Channels", 2, minimumPositiveProperty);
  ensureInt("kernel_size", "Kernel Size", 3, minimumPositiveProperty);
  ensureInt("dilation", "Dilation", 1, minimumPositiveProperty);
  ensureInt("stride", "Stride", 1, minimumPositiveProperty);
  {
    bool hasWeightNorm = false;
    for (auto &property : node.properties) {
      if (property.key == "weight_norm") {
        property.minimum = 0;
        property.maximum = 1;
        hasWeightNorm = true;
        break;
      }
    }
    if (!hasWeightNorm)
      node.properties.push_back(
          NodeProperty{"weight_norm", "Weight Norm", 0, 0, 1});
  }
  for (auto &property : node.properties)
    widenIntegerPropertyBounds(property);
  updateConv1dDetail(node);
}

void applyNodePinShapes(openyourbox::graph::NodeGraph &graph,
                        openyourbox::graph::GraphNode &node);

/**
 * @brief Refreshes inherited hop rate and nBand on every node (first copy only).
 * @param graph Graph document to mutate.
 */
void refreshPropagatedPinShapesCore(openyourbox::graph::NodeGraph &graph) {
  std::unordered_map<std::int32_t, int> indegree;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  for (const auto &node : graph.getNodes())
    indegree[node.id] = 0;
  for (const auto &link : graph.getLinks()) {
    const auto source = graph.findNodeForPin(link.sourcePinId);
    const auto destination = graph.findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value() ||
        *source == *destination)
      continue;
    outgoing[*source].push_back(*destination);
    ++indegree[*destination];
  }
  std::queue<std::int32_t> ready;
  std::unordered_set<std::int32_t> visited;
  for (const auto &entry : indegree) {
    if (entry.second == 0)
      ready.push(entry.first);
  }
  auto apply = [&graph](std::int32_t nodeId) {
    if (auto *node = graph.findNode(nodeId))
      applyNodePinShapes(graph, *node);
  };
  while (!ready.empty()) {
    const auto nodeId = ready.front();
    ready.pop();
    if (!visited.insert(nodeId).second)
      continue;
    apply(nodeId);
    for (const auto next : outgoing[nodeId]) {
      if (--indegree[next] == 0)
        ready.push(next);
    }
  }
  for (auto &node : graph.getNodes()) {
    if (visited.count(node.id) == 0)
      applyNodePinShapes(graph, node);
  }
}

/**
 * @brief Fills @c Pin::copyShapes from an unrolled serial stack.
 * @param graph Editable graph whose first-copy shapes are already current.
 */
void refreshOutputCopyShapes(openyourbox::graph::NodeGraph &graph) {
  // Materialize restores via ValueTree, which refreshes shapes again. Skip the
  // nested copy-shape pass so we only unroll once per outer refresh.
  static thread_local int depth = 0;
  if (depth > 0)
    return;
  struct DepthGuard {
    int &value;
    explicit DepthGuard(int &depthValue) : value(depthValue) { ++value; }
    ~DepthGuard() { --value; }
  } guard(depth);

  for (auto &node : graph.getNodes()) {
    for (auto &pin : node.outputs)
      pin.copyShapes.clear();
    for (auto &pin : node.inputs)
      pin.copyShapes.clear();
  }
  bool anyMultiCopy = false;
  for (const auto &group : graph.getGroups()) {
    if (group.copies > 1) {
      anyMultiCopy = true;
      break;
    }
  }
  if (!anyMultiCopy)
    return;

  std::unordered_map<std::int32_t, std::pair<std::int32_t, int>> provenance;
  auto expanded = graph.withInvisibleCopiesMaterialized(&provenance);
  refreshPropagatedPinShapesCore(expanded);

  std::unordered_map<std::int64_t, const openyourbox::graph::GraphNode *>
      nodeAtSlot;
  const auto slotKey = [](std::int32_t originalId, int slot) -> std::int64_t {
    return (static_cast<std::int64_t>(originalId) << 32) |
           static_cast<std::uint32_t>(slot);
  };
  for (const auto &entry : provenance) {
    const auto *expandedNode = expanded.findNode(entry.first);
    if (expandedNode == nullptr)
      continue;
    nodeAtSlot[slotKey(entry.second.first, entry.second.second)] = expandedNode;
  }

  for (auto &node : graph.getNodes()) {
    const auto copies = graph.effectiveCopyCount(node.id);
    if (copies <= 1)
      continue;
    for (std::size_t pinIndex = 0; pinIndex < node.outputs.size(); ++pinIndex) {
      auto &pin = node.outputs[pinIndex];
      pin.copyShapes.assign(static_cast<std::size_t>(copies), pin.shape);
      for (int slot = 0; slot < copies; ++slot) {
        const auto found = nodeAtSlot.find(slotKey(node.id, slot));
        if (found == nodeAtSlot.end() || found->second == nullptr)
          continue;
        if (pinIndex >= found->second->outputs.size())
          continue;
        pin.copyShapes[static_cast<std::size_t>(slot)] =
            found->second->outputs[pinIndex].shape;
      }
    }
    for (std::size_t pinIndex = 0; pinIndex < node.inputs.size(); ++pinIndex) {
      auto &pin = node.inputs[pinIndex];
      pin.copyShapes.assign(static_cast<std::size_t>(copies), pin.shape);
      for (int slot = 0; slot < copies; ++slot) {
        const auto found = nodeAtSlot.find(slotKey(node.id, slot));
        if (found == nodeAtSlot.end() || found->second == nullptr)
          continue;
        if (pinIndex >= found->second->inputs.size())
          continue;
        pin.copyShapes[static_cast<std::size_t>(slot)] =
            found->second->inputs[pinIndex].shape;
      }
    }
  }
}

/**
 * @brief Refreshes first-copy pin shapes and per-copy shape lists.
 * @param graph Graph document to mutate.
 */
void refreshPropagatedPinShapes(openyourbox::graph::NodeGraph &graph) {
  refreshPropagatedPinShapesCore(graph);
  refreshOutputCopyShapes(graph);
}

/**
 * @brief Applies hop-rate and nBand rules for one node after its producers.
 * @param graph Graph document used to resolve upstream cables.
 * @param node Node whose pins are rewritten.
 */
void applyNodePinShapes(openyourbox::graph::NodeGraph &graph,
                        openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::convolutionOutputTemporalRate;
  using openyourbox::graph::defaultLatentSize;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::flexibleTensorShape;
  using openyourbox::graph::isControlInputPin;
  using openyourbox::graph::isConvolutionType;
  using openyourbox::graph::isConvTransposeType;
  using openyourbox::graph::isShapePassthroughType;
  const auto incoming = firstConnectedTensorSource(graph, node);
  auto inheritInputs = [&]() {
    for (auto &pin : node.inputs) {
      if (isControlInputPin(pin))
        continue;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id))
        inheritTensorFields(pin, connected->shape, true);
      else
        pin.shape = flexibleTensorShape();
    }
  };

  switch (node.type) {
  case NodeType::pqmfAnalysis: {
    const auto nBand =
        std::max(2, readNodeProperty(node, "n_band", defaultPqmfBands));
    int audioChannels = 0;
    for (auto &pin : node.inputs) {
      pin.shape.temporalRate = 1;
      pin.shape.nBand = 0;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id)) {
        pin.shape.channels = connected->shape.channels;
        if (connected->shape.channels > 0)
          audioChannels = connected->shape.channels;
      } else {
        pin.shape.channels = 0;
      }
    }
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = nBand;
      pin.shape.nBand = nBand;
      pin.shape.channels =
          audioChannels > 0 ? audioChannels * nBand : 0;
    }
    break;
  }
  case NodeType::pqmfSynthesis: {
    const auto nBand =
        std::max(2, readNodeProperty(node, "n_band", defaultPqmfBands));
    int bandChannels = 0;
    for (auto &pin : node.inputs) {
      pin.shape.temporalRate = nBand;
      pin.shape.nBand = nBand;
      if (const auto *connected = findConnectedSourcePin(graph, pin.id)) {
        pin.shape.channels = connected->shape.channels;
        if (connected->shape.channels > 0)
          bandChannels = connected->shape.channels;
      } else {
        pin.shape.channels = 0;
      }
    }
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = 1;
      pin.shape.nBand = 0;
      pin.shape.channels =
          bandChannels > 0 ? std::max(1, bandChannels / nBand) : 0;
    }
    break;
  }
  case NodeType::variationalBottleneck:
    inheritInputs();
    for (auto &pin : node.outputs) {
      pin.shape.temporalRate = incoming.temporalRate;
      pin.shape.nBand = incoming.nBand;
      const auto latent = readNodeProperty(node, "latent_size", defaultLatentSize);
      if (latent > 0)
        pin.shape.channels = latent;
    }
    break;
  default: {
    if (!isShapePassthroughType(node.type) && !isConvolutionType(node.type) &&
        !isConvTransposeType(node.type))
      break;
    inheritInputs();
    auto outgoing = incoming;
    if (isConvolutionType(node.type)) {
      const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
      const auto rate =
          convolutionOutputTemporalRate(incoming.temporalRate, stride, false);
      outgoing.temporalRate = rate < 0 ? 0 : rate;
      const auto channels = readNodeProperty(node, "channels", 0);
      if (channels > 0)
        outgoing.channels = channels;
      updateConv1dDetail(node);
    } else if (isConvTransposeType(node.type)) {
      const auto stride = std::max(1, readNodeProperty(node, "stride", 1));
      const auto rate =
          convolutionOutputTemporalRate(incoming.temporalRate, stride, true);
      outgoing.temporalRate = rate < 0 ? 0 : rate;
      const auto channels = readNodeProperty(node, "channels", 0);
      if (channels > 0)
        outgoing.channels = channels;
      updateConvTransposeDetail(node);
    } else if (node.type == NodeType::linear) {
      const auto features = readNodeProperty(node, "features", 0);
      if (features > 0)
        outgoing.channels = features;
    } else if (node.type == NodeType::merge) {
      updateMergeOutputShape(graph, node);
      outgoing.channels =
          node.outputs.empty() ? 0 : node.outputs.front().shape.channels;
    }
    for (auto &pin : node.outputs)
      inheritTensorFields(pin, outgoing, true);
    break;
  }
  }
}

/**
 * @brief Returns a tooltip when any committed cable is now illegal.
 * @param graph Graph document to inspect.
 */
std::string firstIncompatibleLinkMessage(const openyourbox::graph::NodeGraph &graph) {
  using openyourbox::graph::convolutionRateIsError;
  using openyourbox::graph::convolutionRateMessage;
  using openyourbox::graph::defaultPqmfBands;
  using openyourbox::graph::isConvolutionType;
  using openyourbox::graph::isConvTransposeType;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::pqmfSynthesisChannelIsError;
  using openyourbox::graph::pqmfSynthesisChannelMessage;
  using openyourbox::graph::variationalBottleneckChannelIsError;
  using openyourbox::graph::variationalBottleneckChannelMessage;
  for (const auto &link : graph.getLinks()) {
    const auto *source = graph.findPin(link.sourcePinId);
    const auto *destination = graph.findPin(link.destinationPinId);
    if (source == nullptr || destination == nullptr)
      continue;
    if (!source->shape.isCompatibleWith(destination->shape)) {
      auto message = source->shape.incompatibilityMessage(destination->shape);
      if (message.empty())
        message = "Shape mismatch: channel counts are incompatible";
      return message;
    }
    const auto destNode = graph.findNodeForPin(destination->id);
    if (!destNode.has_value())
      continue;
    const auto *node = graph.findNode(*destNode);
    if (node == nullptr)
      continue;
    if (node->type == NodeType::pqmfSynthesis &&
        pqmfSynthesisChannelIsError(source->shape.channels,
                                    std::max(2, readNodeProperty(
                                                    *node, "n_band",
                                                    defaultPqmfBands))))
      return pqmfSynthesisChannelMessage(
          source->shape.channels,
          std::max(2, readNodeProperty(*node, "n_band", defaultPqmfBands)));
    if (node->type == NodeType::variationalBottleneck &&
        variationalBottleneckChannelIsError(source->shape.channels))
      return variationalBottleneckChannelMessage(source->shape.channels);
    if (!isConvolutionType(node->type) && !isConvTransposeType(node->type))
      continue;
    const auto stride = std::max(1, readNodeProperty(*node, "stride", 1));
    const auto upsample = isConvTransposeType(node->type);
    if (convolutionRateIsError(stride, upsample, source->shape.temporalRate))
      return convolutionRateMessage(stride, upsample,
                                    source->shape.temporalRate);
  }
  return {};
}

juce::Colour colourForType(openyourbox::graph::NodeType type,
                           openyourbox::graph::NodeState state) {
  using openyourbox::graph::NodeType;
  if (state == openyourbox::graph::NodeState::frozenGold ||
      type == NodeType::blackBox)
    return openyourbox::graph::frozenGoldColour;
  switch (type) {
  case NodeType::audioInput:
    return openyourbox::graph::audioInputColour;
  case NodeType::audioOutput:
    return openyourbox::graph::audioOutputColour;
  case NodeType::knobInput:
  case NodeType::xyTrackpad:
    return openyourbox::graph::conditioningColour;
  default:
    return openyourbox::graph::liveBlueColour;
  }
}

const char *nodeTypeName(openyourbox::graph::NodeType type) noexcept {
  using openyourbox::graph::NodeType;
  switch (type) {
  case NodeType::audioInput:
    return "audio_input";
  case NodeType::audioOutput:
    return "audio_output";
  case NodeType::linear:
    return "linear";
  case NodeType::convolution:
    return "conv1d";
  case NodeType::activation:
    return "activation";
  case NodeType::tcn:
    return "tcn";
  case NodeType::merge:
    return "merge";
  case NodeType::blackBox:
    return "blackbox";
  case NodeType::knobInput:
    return "knob_input";
  case NodeType::xyTrackpad:
    return "xy_trackpad";
  case NodeType::pqmfAnalysis:
    return "pqmf_analysis";
  case NodeType::pqmfSynthesis:
    return "pqmf_synthesis";
  case NodeType::rateConv:
    return "rate_conv";
  case NodeType::variationalBottleneck:
    return "variational_bottleneck";
  case NodeType::noiseSynthesizer:
    return "noise_synthesizer";
  case NodeType::convTranspose:
    return "conv_transpose1d";
  case NodeType::batchNorm:
    return "batch_norm";
  }
  return "tcn";
}

openyourbox::graph::NodeType nodeTypeFromName(const juce::String &name) {
  using openyourbox::graph::NodeType;
  if (name == "audio_input")
    return NodeType::audioInput;
  if (name == "audio_output")
    return NodeType::audioOutput;
  if (name == "linear")
    return NodeType::linear;
  if (name == "activation")
    return NodeType::activation;
  if (name == "sum" || name == "multiply" || name == "concatenate" ||
      name == "merge")
    return NodeType::merge;
  if (name == "blackbox")
    return NodeType::blackBox;
  if (name == "knob_input")
    return NodeType::knobInput;
  if (name == "xy_trackpad")
    return NodeType::xyTrackpad;
  if (name == "pqmf_analysis")
    return NodeType::pqmfAnalysis;
  if (name == "pqmf_synthesis")
    return NodeType::pqmfSynthesis;
  if (name == "rate_conv" || name == "conv1d")
    return NodeType::convolution;
  if (name == "conv_transpose1d")
    return NodeType::convTranspose;
  if (name == "batch_norm")
    return NodeType::batchNorm;
  if (name == "variational_bottleneck")
    return NodeType::variationalBottleneck;
  if (name == "noise_synthesizer")
    return NodeType::noiseSynthesizer;
  return NodeType::tcn;
}

/**
 * @brief Returns true when @p name is a persisted element type this build knows.
 * @param name ValueTree `type` token.
 */
bool isKnownPersistedNodeType(const juce::String &name) {
  static const std::array<const char *, 20> known{
      "audio_input",
      "audio_output",
      "linear",
      "activation",
      "sum",
      "multiply",
      "concatenate",
      "merge",
      "blackbox",
      "knob_input",
      "xy_trackpad",
      "pqmf_analysis",
      "pqmf_synthesis",
      "rate_conv",
      "conv1d",
      "conv_transpose1d",
      "batch_norm",
      "variational_bottleneck",
      "noise_synthesizer",
      "tcn"};
  for (const auto *token : known) {
    if (name == token)
      return true;
  }
  return false;
}

juce::ValueTree nodeToTree(const openyourbox::graph::GraphNode &node) {
  juce::ValueTree tree{"Node"};
  tree.setProperty("id", node.id, nullptr);
  tree.setProperty("label", juce::String(node.label), nullptr);
  tree.setProperty("detail", juce::String(node.detail), nullptr);
  tree.setProperty("type", nodeTypeName(node.type), nullptr);
  tree.setProperty("state",
                   node.state == openyourbox::graph::NodeState::frozenGold
                       ? "frozen_gold"
                       : "live_blue",
                   nullptr);
  tree.setProperty("x", node.position.x, nullptr);
  tree.setProperty("y", node.position.y, nullptr);
  tree.setProperty("width", node.size.x, nullptr);
  tree.setProperty("height", node.size.y, nullptr);
  tree.setProperty("hasWeights", node.hasWeights, nullptr);
  tree.setProperty("seed", node.seed, nullptr);
  tree.setProperty("explicitSeed", node.explicitSeed, nullptr);
  tree.setProperty("useExplicitSeed", node.useExplicitSeed, nullptr);
  tree.setProperty("artifactPath", juce::String(node.artifactPath), nullptr);
  tree.setProperty("sourceSubgraph", juce::String(node.sourceSubgraph),
                   nullptr);
  tree.setProperty("analysisView", static_cast<int>(node.selectedAnalysisView),
                   nullptr);
  tree.setProperty("conditioningValue", node.conditioningValue, nullptr);
  tree.setProperty("conditioningX", node.conditioningX, nullptr);
  tree.setProperty("conditioningY", node.conditioningY, nullptr);
  tree.setProperty("armedForTraining", node.armedForTraining, nullptr);
  tree.setProperty("residual", node.residual, nullptr);
  tree.setProperty("dilationGrowth", node.dilationGrowth, nullptr);
  tree.setProperty("weightsProvenance",
                   node.weightsProvenance ==
                           openyourbox::graph::WeightsProvenance::file
                       ? "file"
                       : "random",
                   nullptr);
  tree.setProperty("weightsPath", juce::String(node.weightsPath), nullptr);
  tree.setProperty("blackBoxOrigin",
                   node.blackBoxOrigin ==
                           openyourbox::graph::BlackBoxOrigin::trainAutoload
                       ? "train_autoload"
                       : "manual_freeze",
                   nullptr);
  tree.setProperty("fidelityPercent", node.fidelityPercent, nullptr);
  tree.setProperty("compactnessReady", node.compactnessReady, nullptr);
  storeFloatVector(tree, "latentMean", node.latentMean);
  storeFloatVector(tree, "latentPca", node.latentPca);
  storeFloatVector(tree, "cumulativeVariance", node.cumulativeVariance);
  if (node.parentGroupId.has_value())
    tree.setProperty("parentGroupId", *node.parentGroupId, nullptr);

  for (const auto &slot : node.copySlots) {
    juce::ValueTree child{"CopySlot"};
    child.setProperty("seed", slot.seed, nullptr);
    child.setProperty("weightsProvenance",
                      slot.provenance ==
                              openyourbox::graph::WeightsProvenance::file
                          ? "file"
                          : "random",
                      nullptr);
    child.setProperty("weightsPath", juce::String(slot.weightsPath), nullptr);
    child.setProperty("artifactPath", juce::String(slot.artifactPath), nullptr);
    tree.appendChild(child, nullptr);
  }

  if (node.metrics.has_value()) {
    tree.setProperty("compileMs", node.metrics->compileTimeMilliseconds,
                     nullptr);
    tree.setProperty("inferenceMs", node.metrics->inferenceTimeMilliseconds,
                     nullptr);
  }

  const auto appendPins =
      [&tree](const std::vector<openyourbox::graph::Pin> &pins,
              const char *kind) {
        for (const auto &pin : pins) {
          juce::ValueTree child{"Pin"};
          child.setProperty("id", pin.id, nullptr);
          child.setProperty("label", juce::String(pin.label), nullptr);
          child.setProperty("kind", kind, nullptr);
          child.setProperty("channels", pin.shape.channels, nullptr);
          child.setProperty("temporalRate", pin.shape.temporalRate, nullptr);
          child.setProperty("nBand", pin.shape.nBand, nullptr);
          tree.appendChild(child, nullptr);
        }
      };
  appendPins(node.inputs, "input");
  appendPins(node.outputs, "output");

  for (const auto &property : node.properties) {
    juce::ValueTree child{"Property"};
    child.setProperty("key", juce::String(property.key), nullptr);
    child.setProperty("label", juce::String(property.label), nullptr);
    child.setProperty("value", property.value, nullptr);
    child.setProperty("minimum", property.minimum, nullptr);
    child.setProperty("maximum", property.maximum, nullptr);
    child.setProperty("kind", static_cast<int>(property.kind), nullptr);
    juce::StringArray choices;
    for (const auto &choice : property.choices)
      choices.add(choice);
    child.setProperty("choices", choices.joinIntoString("|"), nullptr);
    child.setProperty("floatValue", property.floatValue, nullptr);
    child.setProperty("floatMinimum", property.floatMinimum, nullptr);
    child.setProperty("floatMaximum", property.floatMaximum, nullptr);
    if (!property.copyIntValues.empty()) {
      juce::StringArray ints;
      for (const auto value : property.copyIntValues)
        ints.add(juce::String(value));
      child.setProperty("copyIntValues", ints.joinIntoString(","), nullptr);
    }
    if (!property.copyFloatValues.empty()) {
      juce::StringArray reals;
      for (const auto value : property.copyFloatValues)
        reals.add(juce::String(value, 6));
      child.setProperty("copyFloatValues", reals.joinIntoString(","), nullptr);
    }
    tree.appendChild(child, nullptr);
  }
  return tree;
}

openyourbox::graph::GraphNode nodeFromTree(const juce::ValueTree &tree) {
  using namespace openyourbox::graph;
  GraphNode node;
  node.id = static_cast<std::int32_t>(tree["id"]);
  node.label = tree["label"].toString().toStdString();
  node.detail = tree["detail"].toString().toStdString();
  node.type = nodeTypeFromName(tree["type"].toString());
  node.state = tree["state"].toString() == "frozen_gold" ? NodeState::frozenGold
                                                         : NodeState::liveBlue;
  node.colour = colourForType(node.type, node.state);
  node.position = {static_cast<float>(tree["x"]),
                   static_cast<float>(tree["y"])};
  node.size = {static_cast<float>(tree.getProperty("width", 180.0f)),
               static_cast<float>(tree.getProperty("height", 120.0f))};
  node.hasWeights = static_cast<bool>(tree["hasWeights"]);
  node.seed = openyourbox::graph::clampSeed(
      static_cast<std::int32_t>(tree.getProperty("seed", 42)));
  node.explicitSeed = openyourbox::graph::clampSeed(static_cast<std::int32_t>(
      tree.getProperty("explicitSeed", node.seed)));
  node.useExplicitSeed =
      static_cast<bool>(tree.getProperty("useExplicitSeed", false));
  node.artifactPath = tree["artifactPath"].toString().toStdString();
  node.sourceSubgraph = tree["sourceSubgraph"].toString().toStdString();
  node.selectedAnalysisView = static_cast<AnalysisView>(std::clamp(
      static_cast<int>(tree.getProperty("analysisView", 0)), 0, 3));
  node.conditioningValue = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningValue", 0.0)));
  node.conditioningX = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningX", 0.0)));
  node.conditioningY = openyourbox::graph::clampConditioning(
      static_cast<float>(tree.getProperty("conditioningY", 0.0)));
  node.armedForTraining =
      static_cast<bool>(tree.getProperty("armedForTraining", true));
  node.residual = static_cast<bool>(tree.getProperty("residual", false));
  node.dilationGrowth = std::max(
      static_cast<int>(tree.getProperty("dilationGrowth", defaultDilationGrowth)),
      minimumDilationGrowth);
  node.weightsProvenance =
      tree.getProperty("weightsProvenance", "random").toString() == "file"
          ? WeightsProvenance::file
          : WeightsProvenance::random;
  node.weightsPath = tree["weightsPath"].toString().toStdString();
  node.blackBoxOrigin =
      tree.getProperty("blackBoxOrigin", "manual_freeze").toString() ==
              "train_autoload"
          ? BlackBoxOrigin::trainAutoload
          : BlackBoxOrigin::manualFreeze;
  node.fidelityPercent = openyourbox::graph::clampFidelity(static_cast<float>(
      tree.getProperty("fidelityPercent", defaultFidelityPercent)));
  node.compactnessReady =
      static_cast<bool>(tree.getProperty("compactnessReady", false));
  loadFloatVector(tree, "latentMean", node.latentMean);
  loadFloatVector(tree, "latentPca", node.latentPca);
  loadFloatVector(tree, "cumulativeVariance", node.cumulativeVariance);
  if (tree.hasProperty("parentGroupId"))
    node.parentGroupId =
        static_cast<std::int32_t>(tree.getProperty("parentGroupId"));

  if (tree.hasProperty("inferenceMs")) {
    node.metrics =
        NodeMetrics{static_cast<double>(tree.getProperty("compileMs", 0.0)),
                    static_cast<double>(tree.getProperty("inferenceMs", 0.0))};
  }

  for (const auto child : tree) {
    if (child.hasType("Pin")) {
      Pin pin;
      pin.id = static_cast<std::int32_t>(child["id"]);
      pin.label = child["label"].toString().toStdString();
      pin.kind = child["kind"].toString() == "output" ? PinKind::output
                                                      : PinKind::input;
      pin.shape.channels = static_cast<int>(child["channels"]);
      pin.shape.temporalRate =
          static_cast<int>(child.getProperty("temporalRate", 1));
      pin.shape.nBand = static_cast<int>(child.getProperty("nBand", 0));
      (pin.kind == PinKind::input ? node.inputs : node.outputs)
          .push_back(std::move(pin));
    } else if (child.hasType("CopySlot")) {
      CopyWeightSlot slot;
      slot.seed = openyourbox::graph::clampSeed(
          static_cast<std::int32_t>(child.getProperty("seed", 42)));
      slot.provenance =
          child.getProperty("weightsProvenance", "random").toString() == "file"
              ? WeightsProvenance::file
              : WeightsProvenance::random;
      slot.weightsPath = child["weightsPath"].toString().toStdString();
      slot.artifactPath = child["artifactPath"].toString().toStdString();
      node.copySlots.push_back(std::move(slot));
    } else if (child.hasType("Property")) {
      NodeProperty property;
      property.key = child["key"].toString().toStdString();
      property.label = child["label"].toString().toStdString();
      property.value = static_cast<int>(child["value"]);
      property.minimum = static_cast<int>(child["minimum"]);
      property.maximum = static_cast<int>(child["maximum"]);
      property.kind =
          static_cast<PropertyKind>(static_cast<int>(child["kind"]));
      const auto choices =
          juce::StringArray::fromTokens(child["choices"].toString(), "|", "");
      for (const auto &choice : choices)
        property.choices.push_back(choice.toStdString());
      property.floatValue =
          static_cast<float>(child.getProperty("floatValue", 0.0));
      property.floatMinimum =
          static_cast<float>(child.getProperty("floatMinimum", 0.0));
      property.floatMaximum =
          static_cast<float>(child.getProperty("floatMaximum", 1.0));
      if (child.hasProperty("copyIntValues")) {
        const auto tokens = juce::StringArray::fromTokens(
            child["copyIntValues"].toString(), ",", "");
        for (const auto &token : tokens)
          property.copyIntValues.push_back(token.getIntValue());
      }
      if (child.hasProperty("copyFloatValues")) {
        const auto tokens = juce::StringArray::fromTokens(
            child["copyFloatValues"].toString(), ",", "");
        for (const auto &token : tokens)
          property.copyFloatValues.push_back(token.getFloatValue());
      }
      node.properties.push_back(std::move(property));
    }
  }
  migrateLegacyMixerNode(node, tree["type"].toString());
  normalizeMergeNodeProperties(node);
  normalizeGainProperty(node);
  normalizeConditioningPins(node);
  normalizePhase3Node(node);
  normalizeConvolutionProperties(node);
  normalizeBottleneckProperties(node);
  normalizePropertyBounds(node);
  normalizeHostIoProperties(node);
  return node;
}

/**
 * @brief Serializes one group container into a value tree.
 * @param group Group to persist.
 */
juce::ValueTree groupToTree(const openyourbox::graph::GraphGroup &group) {
  juce::ValueTree tree{"Group"};
  tree.setProperty("id", group.id, nullptr);
  tree.setProperty("name", juce::String(group.name), nullptr);
  if (group.parentGroupId.has_value())
    tree.setProperty("parentGroupId", *group.parentGroupId, nullptr);
  tree.setProperty("collapsed", group.collapsed, nullptr);
  tree.setProperty("copies", group.copies, nullptr);
  tree.setProperty("x", group.position.x, nullptr);
  tree.setProperty("y", group.position.y, nullptr);
  tree.setProperty("width", group.size.x, nullptr);
  tree.setProperty("height", group.size.y, nullptr);
  tree.setProperty("viewPanX", group.viewPan.x, nullptr);
  tree.setProperty("viewPanY", group.viewPan.y, nullptr);
  tree.setProperty("viewZoom", group.viewZoom, nullptr);
  for (const auto member : group.memberIds) {
    juce::ValueTree child{"Member"};
    child.setProperty("id", member, nullptr);
    tree.appendChild(child, nullptr);
  }
  return tree;
}

/**
 * @brief Restores one group container from a value tree.
 * @param tree Serialized group.
 */
openyourbox::graph::GraphGroup groupFromTree(const juce::ValueTree &tree) {
  openyourbox::graph::GraphGroup group;
  group.id = static_cast<std::int32_t>(tree["id"]);
  group.name = tree.getProperty("name", "Group").toString().toStdString();
  if (group.name.empty())
    group.name = "Group";
  if (tree.hasProperty("parentGroupId"))
    group.parentGroupId =
        static_cast<std::int32_t>(tree.getProperty("parentGroupId"));
  group.collapsed = static_cast<bool>(tree.getProperty("collapsed", false));
  group.copies = std::max(1, static_cast<int>(tree.getProperty(
                                   "copies", openyourbox::graph::defaultGroupCopies)));
  group.position = {static_cast<float>(tree.getProperty("x", 0.0f)),
                    static_cast<float>(tree.getProperty("y", 0.0f))};
  group.size = {static_cast<float>(tree.getProperty("width", 320.0f)),
                static_cast<float>(tree.getProperty("height", 220.0f))};
  group.viewPan = {static_cast<float>(tree.getProperty("viewPanX", 0.0f)),
                   static_cast<float>(tree.getProperty("viewPanY", 0.0f))};
  group.viewZoom = std::clamp(
      static_cast<float>(tree.getProperty("viewZoom", 1.0f)),
      openyourbox::graph::minimumZoom, openyourbox::graph::maximumZoom);
  for (const auto child : tree) {
    if (!child.hasType("Member"))
      continue;
    group.memberIds.push_back(static_cast<std::int32_t>(child["id"]));
  }
  return group;
}

/**
 * @brief Detaches @p memberId from every group's member list.
 * @param groups Group collection to mutate.
 * @param memberId Node or nested group identifier.
 */
void eraseMemberFromParents(std::vector<openyourbox::graph::GraphGroup> &groups,
                            std::int32_t memberId) {
  for (auto &group : groups) {
    group.memberIds.erase(std::remove(group.memberIds.begin(),
                                      group.memberIds.end(), memberId),
                          group.memberIds.end());
  }
}

/**
 * @brief Writes parentGroupId onto a node or nested group.
 * @param graph Graph to mutate.
 * @param memberId Node or nested group identifier.
 * @param parent New parent, or empty for the canvas root.
 */
void assignParent(openyourbox::graph::NodeGraph &graph, std::int32_t memberId,
                  std::optional<std::int32_t> parent) {
  if (auto *node = graph.findNode(memberId))
    node->parentGroupId = parent;
  else if (auto *group = graph.findGroup(memberId))
    group->parentGroupId = parent;
}

/**
 * @brief Resizes copy slots for every leaf of @p groupId.
 * @param graph Graph to mutate.
 * @param groupId Group whose descendants need slot updates.
 */
void refreshCopySlotsForGroup(openyourbox::graph::NodeGraph &graph,
                              std::int32_t groupId) {
  for (const auto nodeId : graph.collectLeafNodeIds(groupId)) {
    if (auto *node = graph.findNode(nodeId)) {
      const auto copies = graph.effectiveCopyCount(nodeId);
      openyourbox::graph::ensureCopySlotCount(*node, copies);
      openyourbox::graph::ensureNodePropertyCopyCounts(*node, copies);
    }
  }
}

/**
 * @brief Writes a randomization seed onto a node and its copy slots.
 * @param node Weighted live element to update.
 * @param primarySeed Seed applied to the visible element (slot 0).
 * @param independentExtraSlots True to derive `primarySeed + i` per copy slot.
 */
void writeRandomizedWeightSlots(openyourbox::graph::GraphNode &node,
                                std::int32_t primarySeed,
                                bool independentExtraSlots) {
  using openyourbox::graph::WeightsProvenance;
  using openyourbox::graph::clampSeed;
  using openyourbox::graph::copySlotFromNode;
  using openyourbox::graph::seedForCopySlot;
  node.seed = clampSeed(primarySeed);
  node.weightsProvenance = WeightsProvenance::random;
  node.weightsPath.clear();
  node.artifactPath.clear();
  if (node.copySlots.empty())
    node.copySlots.push_back(copySlotFromNode(node));
  else
    node.copySlots.front() = copySlotFromNode(node);
  for (std::size_t index = 0; index < node.copySlots.size(); ++index) {
    auto &slot = node.copySlots[index];
    slot.seed = independentExtraSlots ? seedForCopySlot(node.seed, index)
                                      : node.seed;
    slot.provenance = WeightsProvenance::random;
    slot.weightsPath.clear();
    slot.artifactPath.clear();
  }
  if (node.type == openyourbox::graph::NodeType::variationalBottleneck ||
      node.type == openyourbox::graph::NodeType::blackBox)
    openyourbox::graph::clearCompactness(node);
}

/**
 * @brief Upgrades persisted mixer nodes to the unified Merge element.
 * @param node Loaded graph node to normalize in place.
 * @param storedType Serialized type name from the graph document.
 */
void migrateLegacyMixerNode(openyourbox::graph::GraphNode &node,
                            const juce::String &storedType) {
  using openyourbox::graph::MergeMode;
  using openyourbox::graph::NodeType;
  int legacyMode = -1;
  if (storedType == "sum")
    legacyMode = static_cast<int>(MergeMode::add);
  else if (storedType == "multiply")
    legacyMode = static_cast<int>(MergeMode::multiply);
  else if (storedType == "concatenate")
    legacyMode = static_cast<int>(MergeMode::concatenate);
  else
    return;

  node.type = NodeType::merge;
  if (node.label == "Sum" || node.label == "Multiply" ||
      node.label == "Concatenate")
    node.label = "Merge";

  for (auto &property : node.properties) {
    if (property.key == "mode") {
      property.value = legacyMode;
      return;
    }
  }

  openyourbox::graph::NodeProperty modeProperty;
  modeProperty.key = "mode";
  modeProperty.label = "Mode";
  modeProperty.value = legacyMode;
  modeProperty.minimum = 0;
  modeProperty.maximum = 2;
  modeProperty.kind = openyourbox::graph::PropertyKind::choice;
  modeProperty.choices = {"Add", "Multiply", "Concatenate"};
  node.properties.insert(node.properties.begin(), std::move(modeProperty));
}

/**
 * @brief Ensures persisted merge nodes expose the current mode choices.
 * @param node Loaded or migrated merge node.
 */
void normalizeMergeNodeProperties(openyourbox::graph::GraphNode &node) {
  if (node.type != openyourbox::graph::NodeType::merge)
    return;
  for (auto &property : node.properties) {
    if (property.key != "mode")
      continue;
    property.maximum = 2;
    property.choices = {"Add", "Multiply", "Concatenate"};
    property.value = std::clamp(property.value, property.minimum, property.maximum);
    return;
  }
}

juce::ValueTree linkToTree(const openyourbox::graph::GraphLink &link) {
  juce::ValueTree tree{"Link"};
  tree.setProperty("id", link.id, nullptr);
  tree.setProperty("sourcePin", link.sourcePinId, nullptr);
  tree.setProperty("destinationPin", link.destinationPinId, nullptr);
  return tree;
}

openyourbox::graph::GraphLink linkFromTree(const juce::ValueTree &tree) {
  return {static_cast<std::int32_t>(tree["id"]),
          static_cast<std::int32_t>(tree["sourcePin"]),
          static_cast<std::int32_t>(tree["destinationPin"])};
}
} // namespace

namespace openyourbox::graph {
void NodeGraph::rebuildFromModel(const dsp::TCNConfiguration &configuration) {
  nodes.clear();
  links.clear();
  nextNodeId = 1;
  nextPinId = 1001;
  nextLinkId = 2001;

  const auto inputId = addNode(NodeType::audioInput, {24.0f, 130.0f});
  const auto tcnId = addNode(NodeType::tcn, {250.0f, 90.0f});
  const auto outputId = addNode(NodeType::audioOutput, {500.0f, 130.0f});
  auto *tcn = findNode(tcnId);
  if (tcn != nullptr) {
    setProperty(tcnId, "depth", configuration.depth);
    setProperty(tcnId, "kernel_size", configuration.kernelSize);
    setProperty(tcnId, "channels", configuration.channels);
    setProperty(tcnId, "activation",
                static_cast<int>(configuration.activation));
    tcn->detail = std::to_string(configuration.channels) + " ch, RF model";
  }

  auto *input = findNode(inputId);
  auto *output = findNode(outputId);
  tcn = findNode(tcnId);
  if (input != nullptr && tcn != nullptr && output != nullptr) {
    setProperty(inputId, "channels",
                hostIoChoiceFromChannels(configuration.inputChannels));
    setProperty(outputId, "channels",
                hostIoChoiceFromChannels(configuration.outputChannels));
    tcn->inputs.front().shape.channels = configuration.inputChannels;
    tcn->outputs.front().shape.channels = configuration.outputChannels;
    connect(input->outputs.front().id, tcn->inputs.front().id);
    connect(tcn->outputs.front().id, output->inputs.front().id);
  }
  ensureFixedHostIo();
}

std::int32_t NodeGraph::addNode(NodeType type, juce::Point<float> position,
                                std::optional<std::int32_t> parentGroupId) {
  auto node = makeNode(type, position);
  if (isFixedIoType(type))
    parentGroupId.reset();
  if (parentGroupId.has_value() && findGroup(*parentGroupId) == nullptr)
    parentGroupId.reset();
  node.parentGroupId = parentGroupId;
  const auto id = node.id;
  nodes.push_back(std::move(node));
  if (parentGroupId.has_value()) {
    if (auto *group = findGroup(*parentGroupId)) {
      if (std::find(group->memberIds.begin(), group->memberIds.end(), id) ==
          group->memberIds.end())
        group->memberIds.push_back(id);
    }
    if (auto *created = findNode(id)) {
      const auto copies = effectiveCopyCount(id);
      ensureCopySlotCount(*created, copies);
      ensureNodePropertyCopyCounts(*created, copies);
    }
  }
  return id;
}

void NodeGraph::ensureFixedHostIo() {
  GraphNode *input = nullptr;
  GraphNode *output = nullptr;
  for (auto &node : nodes) {
    if (node.type == NodeType::audioInput && input == nullptr)
      input = &node;
    else if (node.type == NodeType::audioOutput && output == nullptr)
      output = &node;
  }
  if (input == nullptr)
    addNode(NodeType::audioInput, {24.0f, 130.0f});
  if (output == nullptr)
    addNode(NodeType::audioOutput, {500.0f, 130.0f});
  for (auto &node : nodes) {
    if (!isFixedIoType(node.type))
      continue;
    normalizeHostIoProperties(node);
    node.colour = colourForType(node.type, node.state);
  }
}

std::optional<std::int32_t>
NodeGraph::insertNodeOnLink(std::int32_t linkId, NodeType type,
                            juce::Point<float> position) {
  if (isFixedIoType(type) || type == NodeType::blackBox ||
      isConditioningSourceType(type))
    return std::nullopt;
  const auto *link = findLink(linkId);
  if (link == nullptr)
    return std::nullopt;
  const auto sourcePinId = link->sourcePinId;
  const auto destinationPinId = link->destinationPinId;
  if (!removeLink(linkId))
    return std::nullopt;

  const auto sourceOwner = findNodeForPin(sourcePinId);
  const auto destOwner = findNodeForPin(destinationPinId);
  std::optional<std::int32_t> parent;
  const auto *sourceNode =
      sourceOwner.has_value() ? findNode(*sourceOwner) : nullptr;
  const auto *destNode =
      destOwner.has_value() ? findNode(*destOwner) : nullptr;
  if (sourceNode != nullptr)
    parent = sourceNode->parentGroupId;
  if (destNode != nullptr && parent != destNode->parentGroupId)
    parent = viewport.focusedGroupId;

  const auto nodeId = addNode(type, position, parent);
  auto *node = findNode(nodeId);
  if (node == nullptr || node->inputs.empty() || node->outputs.empty()) {
    connect(sourcePinId, destinationPinId);
    if (nodeId != 0)
      removeNode(nodeId);
    return std::nullopt;
  }
  const auto inputResult = connect(sourcePinId, node->inputs.front().id);
  const auto outputResult = connect(node->outputs.front().id, destinationPinId);
  if (!inputResult.accepted || !outputResult.accepted) {
    removeNode(nodeId);
    connect(sourcePinId, destinationPinId);
    return std::nullopt;
  }
  return nodeId;
}

std::optional<std::int32_t>
NodeGraph::attachNodeToPin(std::int32_t pinId, NodeType type,
                           juce::Point<float> position) {
  if (isFixedIoType(type) || type == NodeType::blackBox)
    return std::nullopt;
  const auto resolvedPinId = resolveCollapsedPin(pinId);
  const auto *pin = findPin(resolvedPinId);
  if (pin == nullptr || isPinConnected(resolvedPinId))
    return std::nullopt;

  std::optional<std::int32_t> parent = viewport.focusedGroupId;
  if (!isCollapsedGroupPin(pinId)) {
    if (const auto ownerId = findNodeForPin(resolvedPinId);
        ownerId.has_value()) {
      if (const auto *owner = findNode(*ownerId))
        parent = owner->parentGroupId;
    }
  }

  const auto nodeId = addNode(type, position, parent);
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return std::nullopt;
  if (pin->kind == PinKind::output) {
    if (node->inputs.empty()) {
      removeNode(nodeId);
      return std::nullopt;
    }
  } else if (node->outputs.empty()) {
    removeNode(nodeId);
    return std::nullopt;
  }

  const auto result =
      pin->kind == PinKind::output
          ? connect(resolvedPinId, node->inputs.front().id)
          : connect(node->outputs.front().id, resolvedPinId);
  if (!result.accepted) {
    removeNode(nodeId);
    return std::nullopt;
  }
  return nodeId;
}

bool NodeGraph::removeNode(std::int32_t nodeId) {
  const auto *node = findNode(nodeId);
  if (node == nullptr || isFixedIoType(node->type) ||
      (node->state == NodeState::frozenGold &&
       node->type != NodeType::blackBox))
    return false;

  std::unordered_set<std::int32_t> pins;
  for (const auto &pin : node->inputs)
    pins.insert(pin.id);
  for (const auto &pin : node->outputs)
    pins.insert(pin.id);
  links.erase(std::remove_if(links.begin(), links.end(),
                             [&pins](const GraphLink &link) {
                               return pins.count(link.sourcePinId) != 0 ||
                                      pins.count(link.destinationPinId) != 0;
                             }),
              links.end());
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [nodeId](const GraphNode &candidate) {
                               return candidate.id == nodeId;
                             }),
              nodes.end());
  eraseMemberFromParents(groups, nodeId);
  return true;
}

void NodeGraph::moveNode(std::int32_t nodeId, juce::Point<float> position) {
  if (auto *node = findNode(nodeId))
    node->position = position;
}

const std::vector<GraphGroup> &NodeGraph::getGroups() const noexcept {
  return groups;
}

std::vector<GraphGroup> &NodeGraph::getGroups() noexcept { return groups; }

GraphGroup *NodeGraph::findGroup(std::int32_t groupId) noexcept {
  const auto found = std::find_if(
      groups.begin(), groups.end(),
      [groupId](const GraphGroup &group) { return group.id == groupId; });
  return found != groups.end() ? &*found : nullptr;
}

const GraphGroup *NodeGraph::findGroup(std::int32_t groupId) const noexcept {
  const auto found = std::find_if(
      groups.begin(), groups.end(),
      [groupId](const GraphGroup &group) { return group.id == groupId; });
  return found != groups.end() ? &*found : nullptr;
}

juce::Point<float>
NodeGraph::groupLocalToWorld(std::int32_t groupId,
                             juce::Point<float> localPoint) const {
  auto point = localPoint;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    point = group->position + groupContentOffset() +
            (point - group->viewPan) * group->viewZoom;
    current = group->parentGroupId;
  }
  return point;
}

juce::Point<float>
NodeGraph::worldToGroupLocal(std::int32_t groupId,
                             juce::Point<float> worldPoint) const {
  std::vector<const GraphGroup *> chain;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    chain.push_back(group);
    current = group->parentGroupId;
  }
  auto point = worldPoint;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const auto *group = *it;
    const auto zoom = std::max(0.01f, group->viewZoom);
    const auto origin = group->position + groupContentOffset();
    point = group->viewPan + (point - origin) / zoom;
  }
  return point;
}

juce::Point<float>
NodeGraph::worldPositionOfGroup(std::int32_t groupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr)
    return {};
  if (!group->parentGroupId.has_value())
    return group->position;
  return groupLocalToWorld(*group->parentGroupId, group->position);
}

juce::Point<float> NodeGraph::worldPositionOfNode(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return {};
  if (!node->parentGroupId.has_value())
    return node->position;
  return groupLocalToWorld(*node->parentGroupId, node->position);
}

float NodeGraph::groupContentToCanvasScale(std::int32_t groupId) const {
  float scale = 1.0f;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    scale *= std::max(0.01f, group->viewZoom);
    current = group->parentGroupId;
  }
  return scale;
}

void NodeGraph::setGroupView(std::int32_t groupId, juce::Point<float> pan,
                             float zoom) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->viewPan = pan;
  group->viewZoom =
      std::clamp(zoom, minimumZoom, maximumZoom);
}

std::vector<std::int32_t>
NodeGraph::collectLeafNodeIds(std::int32_t groupId) const {
  std::vector<std::int32_t> leaves;
  collectLeaves(*this, groupId, leaves);
  return leaves;
}

int NodeGraph::effectiveCopyCount(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr)
    return 1;
  int count = 1;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    count *= std::max(1, group->copies);
    parent = group->parentGroupId;
  }
  return std::max(1, count);
}

bool NodeGraph::isNodeHiddenByCollapse(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return false;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    if (group->collapsed)
      return true;
    parent = group->parentGroupId;
  }
  return false;
}

bool NodeGraph::isGroupHiddenByCollapse(std::int32_t groupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr || !group->parentGroupId.has_value())
    return false;
  auto parent = group->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    const auto *ancestor = findGroup(*parent);
    if (ancestor == nullptr)
      break;
    if (ancestor->collapsed)
      return true;
    parent = ancestor->parentGroupId;
  }
  return false;
}

bool NodeGraph::isNodeClippedByGroup(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return false;
  const auto *group = findGroup(*node->parentGroupId);
  if (group == nullptr || group->collapsed)
    return false;
  const auto origin = worldPositionOfGroup(group->id);
  const auto parentScale =
      group->parentGroupId.has_value()
          ? groupContentToCanvasScale(*group->parentGroupId)
          : 1.0f;
  const auto contentMinX = origin.x + groupContentPad * parentScale;
  const auto contentMinY = origin.y + groupHeaderHeight * parentScale;
  const auto contentMaxX =
      origin.x + (group->size.x - groupContentPad) * parentScale;
  const auto contentMaxY =
      origin.y + (group->size.y - groupContentPad) * parentScale;
  const auto world = worldPositionOfNode(nodeId);
  const auto width = std::max(8.0f, node->size.x);
  const auto height = std::max(8.0f, node->size.y);
  return world.x + width < contentMinX || world.x > contentMaxX ||
         world.y + height < contentMinY || world.y > contentMaxY;
}

bool NodeGraph::isNodeOnFocusedCanvas(
    std::int32_t nodeId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *node = findNode(nodeId);
  return node != nullptr && node->parentGroupId == focusedGroupId;
}

bool NodeGraph::isGroupOnFocusedCanvas(
    std::int32_t groupId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *group = findGroup(groupId);
  if (group == nullptr || group->id == focusedGroupId)
    return false;
  return group->parentGroupId == focusedGroupId;
}

std::optional<std::int32_t> NodeGraph::focusedCanvasHostGroup(
    std::int32_t nodeId, std::optional<std::int32_t> focusedGroupId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return std::nullopt;
  if (node->parentGroupId == focusedGroupId)
    return std::nullopt;
  auto current = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    if (group->parentGroupId == focusedGroupId)
      return group->id;
    current = group->parentGroupId;
  }
  return std::nullopt;
}

std::vector<std::int32_t>
NodeGraph::groupAncestorChain(std::int32_t groupId) const {
  std::vector<std::int32_t> chain;
  auto current = std::optional<std::int32_t>(groupId);
  std::unordered_set<std::int32_t> visiting;
  while (current.has_value()) {
    if (!visiting.insert(*current).second)
      break;
    const auto *group = findGroup(*current);
    if (group == nullptr)
      break;
    chain.push_back(*current);
    current = group->parentGroupId;
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

std::optional<std::int32_t>
NodeGraph::findExpandedGroupAt(juce::Point<float> canvasPoint) const {
  std::optional<std::int32_t> best;
  int bestDepth = -1;
  for (const auto &group : groups) {
    if (group.collapsed || isGroupHiddenByCollapse(group.id))
      continue;
    const auto origin = worldPositionOfGroup(group.id);
    const auto scale = group.parentGroupId.has_value()
                           ? groupContentToCanvasScale(*group.parentGroupId)
                           : 1.0f;
    const auto size = juce::Point<float>(group.size.x * scale, group.size.y * scale);
    if (canvasPoint.x < origin.x || canvasPoint.y < origin.y ||
        canvasPoint.x > origin.x + size.x || canvasPoint.y > origin.y + size.y)
      continue;
    const auto depth = groupDepth(*this, group.id);
    if (depth >= bestDepth) {
      bestDepth = depth;
      best = group.id;
    }
  }
  return best;
}

std::vector<GroupBoundaryPort>
NodeGraph::groupBoundaryPorts(std::int32_t groupId) const {
  std::unordered_set<std::int32_t> leaves;
  for (const auto id : collectLeafNodeIds(groupId))
    leaves.insert(id);
  std::vector<GroupBoundaryPort> ports;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto sourceInside = leaves.count(*source) != 0;
    const auto destInside = leaves.count(*destination) != 0;
    if (sourceInside == destInside)
      continue;
    GroupBoundaryPort port;
    if (destInside) {
      const auto *pin = findPin(link.destinationPinId);
      const auto *node = findNode(*destination);
      if (pin == nullptr || node == nullptr)
        continue;
      port.memberPinId = pin->id;
      port.memberNodeId = node->id;
      port.kind = PinKind::input;
      port.shape = pin->shape;
      port.label = pin->label;
    } else {
      const auto *pin = findPin(link.sourcePinId);
      const auto *node = findNode(*source);
      if (pin == nullptr || node == nullptr)
        continue;
      port.memberPinId = pin->id;
      port.memberNodeId = node->id;
      port.kind = PinKind::output;
      port.shape = pin->shape;
      port.label = pin->label;
    }
    ports.push_back(std::move(port));
  }
  std::sort(ports.begin(), ports.end(),
            [](const GroupBoundaryPort &left, const GroupBoundaryPort &right) {
              if (left.kind != right.kind)
                return left.kind == PinKind::input;
              if (left.memberNodeId != right.memberNodeId)
                return left.memberNodeId < right.memberNodeId;
              return left.memberPinId < right.memberPinId;
            });
  return ports;
}

std::vector<GroupBoundaryPort>
NodeGraph::groupInterfacePorts(std::int32_t groupId) const {
  std::unordered_set<std::int32_t> leaves;
  for (const auto id : collectLeafNodeIds(groupId))
    leaves.insert(id);
  std::unordered_set<std::int32_t> internallyFedInputs;
  std::unordered_set<std::int32_t> internallyUsedOutputs;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (leaves.count(*source) == 0 || leaves.count(*destination) == 0)
      continue;
    internallyUsedOutputs.insert(link.sourcePinId);
    internallyFedInputs.insert(link.destinationPinId);
  }
  std::vector<GroupBoundaryPort> ports;
  const auto appendPin = [&](const GraphNode &node, const Pin &pin) {
    GroupBoundaryPort port;
    port.memberPinId = pin.id;
    port.memberNodeId = node.id;
    port.kind = pin.kind;
    port.shape = pin.shape;
    port.label = node.label.empty() ? pin.label : node.label + " " + pin.label;
    ports.push_back(std::move(port));
  };
  for (const auto id : collectLeafNodeIds(groupId)) {
    const auto *node = findNode(id);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs) {
      if (internallyFedInputs.count(pin.id) == 0)
        appendPin(*node, pin);
    }
    for (const auto &pin : node->outputs) {
      if (internallyUsedOutputs.count(pin.id) == 0)
        appendPin(*node, pin);
    }
  }
  std::sort(ports.begin(), ports.end(),
            [](const GroupBoundaryPort &left, const GroupBoundaryPort &right) {
              if (left.kind != right.kind)
                return left.kind == PinKind::input;
              if (left.memberNodeId != right.memberNodeId)
                return left.memberNodeId < right.memberNodeId;
              return left.memberPinId < right.memberPinId;
            });
  return ports;
}

std::optional<std::int32_t>
NodeGraph::innermostVisibleGroupOf(std::int32_t nodeId) const {
  const auto *node = findNode(nodeId);
  if (node == nullptr || !node->parentGroupId.has_value())
    return std::nullopt;
  auto parent = node->parentGroupId;
  std::unordered_set<std::int32_t> visiting;
  while (parent.has_value()) {
    if (!visiting.insert(*parent).second)
      break;
    if (!isGroupHiddenByCollapse(*parent))
      return parent;
    const auto *group = findGroup(*parent);
    if (group == nullptr)
      break;
    parent = group->parentGroupId;
  }
  return std::nullopt;
}

void NodeGraph::fitGroupToMembers(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr || group->memberIds.empty())
    return;
  const auto oldPosition = group->position;
  const auto oldPan = group->viewPan;
  const auto oldZoom = std::max(0.01f, group->viewZoom);
  auto min = juce::Point<float>(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
  auto max = juce::Point<float>(std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest());
  bool any = false;
  struct ParentRect {
    std::int32_t id = 0;
    juce::Point<float> position;
    juce::Point<float> size;
  };
  std::vector<ParentRect> parentRects;
  parentRects.reserve(group->memberIds.size());
  for (const auto member : group->memberIds) {
    const auto bounds = memberBounds(*this, member);
    if (!bounds.has_value())
      continue;
    const auto parentPosition =
        oldPosition + groupContentOffset() +
        (bounds->first - oldPan) * oldZoom;
    const auto parentSize =
        juce::Point<float>(bounds->second.x * oldZoom, bounds->second.y * oldZoom);
    parentRects.push_back({member, parentPosition, parentSize});
    any = true;
    min.x = std::min(min.x, parentPosition.x);
    min.y = std::min(min.y, parentPosition.y);
    max.x = std::max(max.x, parentPosition.x + parentSize.x);
    max.y = std::max(max.y, parentPosition.y + parentSize.y);
  }
  if (!any)
    return;
  group->position = {min.x - groupFitPadding,
                     min.y - groupFitPadding - groupHeaderHeight};
  group->size = {std::max(160.0f, max.x - min.x + groupFitPadding * 2.0f),
                 std::max(120.0f, max.y - min.y + groupFitPadding * 2.0f +
                                      groupHeaderHeight)};
  group->viewPan = {};
  group->viewZoom = 1.0f;
  const auto origin = group->position + groupContentOffset();
  for (const auto &rect : parentRects)
    setItemStoredPosition(*this, rect.id, rect.position - origin);
}

void NodeGraph::setGroupBounds(std::int32_t groupId, juce::Point<float> position,
                               juce::Point<float> size) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->position = position;
  group->size = {std::max(80.0f, size.x), std::max(60.0f, size.y)};
}

void NodeGraph::moveGroup(std::int32_t groupId, juce::Point<float> position) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return;
  group->position = position;
}

bool NodeGraph::renameGroup(std::int32_t groupId, const std::string &name) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  auto trimmed = name;
  while (!trimmed.empty() &&
         (trimmed.back() == ' ' || trimmed.back() == '\t'))
    trimmed.pop_back();
  if (trimmed.empty())
    return false;
  group->name = trimmed;
  return true;
}

bool NodeGraph::setGroupCollapsed(std::int32_t groupId, bool collapsed) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  group->collapsed = collapsed;
  return true;
}

bool NodeGraph::toggleGroupCollapsed(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return false;
  group->collapsed = !group->collapsed;
  return true;
}

GroupActionResult
NodeGraph::createGroup(const std::vector<std::int32_t> &memberIds) {
  std::vector<std::int32_t> unique;
  unique.reserve(memberIds.size());
  std::unordered_set<std::int32_t> seen;
  std::optional<std::int32_t> sharedParent;
  bool parentInitialized = false;
  for (const auto id : memberIds) {
    if (!seen.insert(id).second)
      continue;
    const auto *node = findNode(id);
    const auto *group = findGroup(id);
    if (node == nullptr && group == nullptr)
      return {false, "Selection includes an unknown box", 0};
    if (node != nullptr && isFixedIoType(node->type))
      return {false, "Audio Input and Audio Output cannot join a group", 0};
    const auto parent = parentGroupOf(*this, id);
    if (!parentInitialized) {
      sharedParent = parent;
      parentInitialized = true;
    } else if (parent != sharedParent) {
      return {false, "Group members must share the same parent group", 0};
    }
    unique.push_back(id);
  }
  if (unique.size() < 2)
    return {false, "Select at least two boxes to create a group", 0};

  const auto parentDepth =
      sharedParent.has_value() ? groupDepth(*this, *sharedParent) : 0;
  if (parentDepth + 1 > maximumGroupNestingDepth)
    return {false, "Group nesting is limited to 8 levels", 0};

  std::vector<juce::Point<float>> memberWorlds;
  memberWorlds.reserve(unique.size());
  for (const auto id : unique)
    memberWorlds.push_back(itemWorldPosition(*this, id));

  GraphGroup group;
  group.id = nextNodeId++;
  group.name = "Group";
  group.parentGroupId = sharedParent;
  group.memberIds = unique;
  groups.push_back(group);
  for (const auto id : unique) {
    eraseMemberFromParents(groups, id);
    assignParent(*this, id, group.id);
  }
  if (auto *created = findGroup(group.id))
    created->memberIds = unique;
  if (sharedParent.has_value()) {
    if (auto *parent = findGroup(*sharedParent))
      parent->memberIds.push_back(group.id);
  }

  auto min = juce::Point<float>(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
  auto max = juce::Point<float>(std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest());
  bool any = false;
  for (std::size_t index = 0; index < unique.size(); ++index) {
    const auto bounds = memberBounds(*this, unique[index]);
    auto parentSpace = memberWorlds[index];
    if (sharedParent.has_value())
      parentSpace = worldToGroupLocal(*sharedParent, memberWorlds[index]);
    const auto size = bounds.has_value() ? bounds->second
                                         : juce::Point<float>(8.0f, 8.0f);
    any = true;
    min.x = std::min(min.x, parentSpace.x);
    min.y = std::min(min.y, parentSpace.y);
    max.x = std::max(max.x, parentSpace.x + size.x);
    max.y = std::max(max.y, parentSpace.y + size.y);
  }
  if (any) {
    if (auto *created = findGroup(group.id)) {
      created->position = {min.x - groupFitPadding,
                           min.y - groupFitPadding - groupHeaderHeight};
      created->size = {
          std::max(160.0f, max.x - min.x + groupFitPadding * 2.0f),
          std::max(120.0f, max.y - min.y + groupFitPadding * 2.0f +
                               groupHeaderHeight)};
      created->viewPan = {};
      created->viewZoom = 1.0f;
    }
  }
  for (std::size_t index = 0; index < unique.size(); ++index)
    storeWorldPosition(*this, unique[index], memberWorlds[index], group.id);
  refreshCopySlotsForGroup(*this, group.id);
  return {true, {}, group.id};
}

GroupActionResult NodeGraph::ungroup(std::int32_t groupId) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Group no longer exists", 0};
  const auto parent = group->parentGroupId;
  const auto members = group->memberIds;
  std::vector<juce::Point<float>> memberWorlds;
  memberWorlds.reserve(members.size());
  for (const auto member : members)
    memberWorlds.push_back(itemWorldPosition(*this, member));
  for (std::size_t index = 0; index < members.size(); ++index) {
    assignParent(*this, members[index], parent);
    storeWorldPosition(*this, members[index], memberWorlds[index], parent);
  }
  if (parent.has_value()) {
    if (auto *parentGroup = findGroup(*parent)) {
      parentGroup->memberIds.erase(std::remove(parentGroup->memberIds.begin(),
                                               parentGroup->memberIds.end(),
                                               groupId),
                                   parentGroup->memberIds.end());
      parentGroup->memberIds.insert(parentGroup->memberIds.end(),
                                    members.begin(), members.end());
    }
  }
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [groupId](const GraphGroup &candidate) {
                                return candidate.id == groupId;
                              }),
               groups.end());
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::deleteGroup(std::int32_t groupId) {
  if (findGroup(groupId) == nullptr)
    return {false, "Group no longer exists", 0};
  std::unordered_set<std::int32_t> nodeIds;
  std::unordered_set<std::int32_t> groupIds;
  collectGroupSubtree(*this, groupId, nodeIds, groupIds);
  std::unordered_set<std::int32_t> pins;
  for (const auto nodeId : nodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      continue;
    for (const auto &pin : node->inputs)
      pins.insert(pin.id);
    for (const auto &pin : node->outputs)
      pins.insert(pin.id);
  }
  links.erase(std::remove_if(links.begin(), links.end(),
                             [&pins](const GraphLink &link) {
                               return pins.count(link.sourcePinId) != 0 ||
                                      pins.count(link.destinationPinId) != 0;
                             }),
              links.end());
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [&nodeIds](const GraphNode &candidate) {
                               return nodeIds.count(candidate.id) != 0;
                             }),
              nodes.end());
  eraseMemberFromParents(groups, groupId);
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [&groupIds](const GraphGroup &candidate) {
                                return groupIds.count(candidate.id) != 0;
                              }),
               groups.end());
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::addToGroup(std::int32_t groupId,
                                        std::int32_t memberId,
                                        bool preserveStoredPosition) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Drop target group no longer exists", 0};
  if (memberId == groupId)
    return {false, "A group cannot contain itself", 0};
  if (const auto *node = findNode(memberId)) {
    if (isFixedIoType(node->type))
      return {false, "Audio Input and Audio Output cannot join a group", 0};
  } else if (findGroup(memberId) == nullptr) {
    return {false, "Drop source is not a graph box", 0};
  }
  if (findGroup(memberId) != nullptr && groupOwnsId(*this, memberId, groupId))
    return {false, "A group cannot be nested inside one of its descendants", 0};
  const auto newDepth = groupDepth(*this, groupId) +
                        (findGroup(memberId) != nullptr
                             ? groupDepth(*this, memberId)
                             : 1);
  if (newDepth > maximumGroupNestingDepth)
    return {false, "Group nesting is limited to 8 levels", 0};
  if (std::find(group->memberIds.begin(), group->memberIds.end(), memberId) !=
      group->memberIds.end())
    return {true, {}, groupId};

  const auto world = itemWorldPosition(*this, memberId);
  eraseMemberFromParents(groups, memberId);
  assignParent(*this, memberId, groupId);
  group->memberIds.push_back(memberId);
  if (!preserveStoredPosition)
    storeWorldPosition(*this, memberId, world, groupId);
  refreshCopySlotsForGroup(*this, groupId);
  return {true, {}, groupId};
}

GroupActionResult NodeGraph::removeFromGroup(std::int32_t memberId) {
  const auto parent = parentGroupOf(*this, memberId);
  if (!parent.has_value())
    return {false, "Box is not inside a group", 0};
  auto *group = findGroup(*parent);
  if (group == nullptr)
    return {false, "Parent group no longer exists", 0};
  const auto grandparent = group->parentGroupId;
  const auto world = itemWorldPosition(*this, memberId);
  eraseMemberFromParents(groups, memberId);
  assignParent(*this, memberId, grandparent);
  storeWorldPosition(*this, memberId, world, grandparent);
  if (grandparent.has_value()) {
    if (auto *ancestor = findGroup(*grandparent))
      ancestor->memberIds.push_back(memberId);
  }
  if (findNode(memberId) != nullptr) {
    auto *node = findNode(memberId);
    const auto copies = effectiveCopyCount(memberId);
    ensureCopySlotCount(*node, copies);
    ensureNodePropertyCopyCounts(*node, copies);
  } else if (findGroup(memberId) != nullptr)
    refreshCopySlotsForGroup(*this, memberId);
  return {true, {}, *parent};
}

std::vector<std::int32_t> NodeGraph::expandSelectionToFreezableLeaves(
    const std::vector<std::int32_t> &selectedIds) const {
  std::vector<std::int32_t> leaves;
  std::unordered_set<std::int32_t> seen;
  const auto appendNode = [&](std::int32_t nodeId) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return;
    if (seen.insert(nodeId).second)
      leaves.push_back(nodeId);
  };
  for (const auto id : selectedIds) {
    if (findGroup(id) != nullptr) {
      for (const auto leaf : collectLeafNodeIds(id))
        appendNode(leaf);
    } else {
      appendNode(id);
    }
  }
  return leaves;
}

GroupActionResult NodeGraph::setGroupCopies(std::int32_t groupId, int copies) {
  auto *group = findGroup(groupId);
  if (group == nullptr)
    return {false, "Group no longer exists", 0};
  if (copies < 1)
    copies = 1;
  if (copies == group->copies)
    return {true, {}, groupId};
  if (copies > 1) {
    std::vector<GroupBoundaryPort> inputs;
    std::vector<GroupBoundaryPort> outputs;
    appendSerialChainPorts(*this, groupId, inputs, outputs);
    if (const auto message = serialChainRefusal(inputs, outputs);
        !message.empty())
      return {false, message, 0};
  }
  group->copies = copies;
  refreshCopySlotsForGroup(*this, groupId);
  refreshPropagatedPinShapes(*this);
  return {true, {}, groupId};
}

bool NodeGraph::randomizeGroupWeights(std::int32_t groupId) {
  if (findGroup(groupId) == nullptr)
    return false;
  bool any = false;
  for (const auto nodeId : collectLeafNodeIds(groupId)) {
    auto *node = findNode(nodeId);
    if (node == nullptr || !node->hasWeights ||
        node->state == NodeState::frozenGold)
      continue;
    if (node->type == NodeType::batchNorm)
      writeRandomizedWeightSlots(*node, 0, false);
    else if (node->useExplicitSeed)
      writeRandomizedWeightSlots(*node, node->explicitSeed, true);
    else {
      const auto seed =
          juce::Random::getSystemRandom().nextInt(maximumSeed + 1);
      writeRandomizedWeightSlots(*node, seed, true);
    }
    any = true;
  }
  return any;
}

NodeGraph NodeGraph::withInvisibleCopiesMaterialized() const {
  return withInvisibleCopiesMaterialized(nullptr);
}

NodeGraph NodeGraph::withInvisibleCopiesMaterialized(
    std::unordered_map<std::int32_t, std::pair<std::int32_t, int>> *provenance)
    const {
  NodeGraph expanded;
  expanded.restoreFromValueTree(toValueTree());
  if (provenance != nullptr)
    provenance->clear();
  std::vector<std::int32_t> ordered;
  ordered.reserve(expanded.groups.size());
  for (const auto &group : expanded.groups)
    ordered.push_back(group.id);
  std::sort(ordered.begin(), ordered.end(),
            [&expanded](std::int32_t left, std::int32_t right) {
              return groupDepth(expanded, left) > groupDepth(expanded, right);
            });

  struct WorkingNode {
    std::int32_t originalId = 0;
    int slotIndex = 0;
  };
  std::unordered_map<std::int32_t, WorkingNode> working;
  for (const auto &node : expanded.nodes) {
    working[node.id] = WorkingNode{node.id, 0};
    if (provenance != nullptr)
      (*provenance)[node.id] = {node.id, 0};
  }

  for (const auto groupId : ordered) {
    auto *group = expanded.findGroup(groupId);
    if (group == nullptr || group->copies <= 1)
      continue;
    const auto copies = group->copies;
    const auto templateNodes = expanded.collectLeafNodeIds(groupId);
    if (templateNodes.empty())
      continue;

    const auto innerProduct = [&](std::int32_t nodeId) {
      const auto found = working.find(nodeId);
      const auto originalId =
          found != working.end() ? found->second.originalId : nodeId;
      const auto *original = findNode(originalId);
      if (original == nullptr)
        return 1;
      int product = 1;
      auto parent = original->parentGroupId;
      std::unordered_set<std::int32_t> visiting;
      while (parent.has_value() && *parent != groupId) {
        if (!visiting.insert(*parent).second)
          break;
        const auto *ancestor = findGroup(*parent);
        if (ancestor == nullptr)
          break;
        product *= std::max(1, ancestor->copies);
        parent = ancestor->parentGroupId;
      }
      return std::max(1, product);
    };

    std::unordered_set<std::int32_t> templateSet(templateNodes.begin(),
                                                 templateNodes.end());
    std::vector<GraphLink> internalLinks;
    for (const auto &link : expanded.links) {
      const auto sourceNode = expanded.findNodeForPin(link.sourcePinId);
      const auto dest = expanded.findNodeForPin(link.destinationPinId);
      if (sourceNode.has_value() && dest.has_value() &&
          templateSet.count(*sourceNode) != 0 && templateSet.count(*dest) != 0)
        internalLinks.push_back(link);
    }
    std::vector<GroupBoundaryPort> inputs;
    std::vector<GroupBoundaryPort> outputs;
    appendSerialChainPorts(expanded, groupId, inputs, outputs);

    struct CopyInstance {
      std::unordered_map<std::int32_t, std::int32_t> pinMap;
      std::unordered_map<std::int32_t, std::int32_t> nodeMap;
    };
    std::vector<CopyInstance> instances(static_cast<std::size_t>(copies));
    for (const auto nodeId : templateNodes) {
      instances[0].nodeMap[nodeId] = nodeId;
      if (const auto *node = expanded.findNode(nodeId)) {
        for (const auto &pin : node->inputs)
          instances[0].pinMap[pin.id] = pin.id;
        for (const auto &pin : node->outputs)
          instances[0].pinMap[pin.id] = pin.id;
      }
      if (auto *node = expanded.findNode(nodeId)) {
        const auto slot = working[nodeId].slotIndex;
        const auto originalId = working[nodeId].originalId;
        if (const auto *original = findNode(originalId))
          node->properties = original->properties;
        ensureCopySlotCount(
            *node, std::max(slot + 1, effectiveCopyCount(originalId)));
        if (slot < static_cast<int>(node->copySlots.size()))
          applyCopySlot(*node, node->copySlots[static_cast<std::size_t>(slot)]);
        applyCopyPropertyValues(*node, slot);
        if (provenance != nullptr)
          (*provenance)[nodeId] = {originalId, slot};
      }
    }

    for (int copy = 1; copy < copies; ++copy) {
      auto &instance = instances[static_cast<std::size_t>(copy)];
      for (const auto nodeId : templateNodes) {
        const auto *templateNode = expanded.findNode(nodeId);
        if (templateNode == nullptr)
          continue;
        GraphNode clone = *templateNode;
        clone.id = expanded.nextNodeId++;
        clone.parentGroupId = templateNode->parentGroupId;
        for (auto &pin : clone.inputs) {
          const auto originalPin = pin.id;
          pin.id = expanded.nextPinId++;
          instance.pinMap[originalPin] = pin.id;
        }
        for (auto &pin : clone.outputs) {
          const auto originalPin = pin.id;
          pin.id = expanded.nextPinId++;
          instance.pinMap[originalPin] = pin.id;
        }
        const auto slot =
            working[nodeId].slotIndex + copy * innerProduct(nodeId);
        const auto originalId = working[nodeId].originalId;
        if (const auto *original = findNode(originalId)) {
          clone.copySlots = original->copySlots;
          clone.properties = original->properties;
          ensureCopySlotCount(
              clone, std::max(slot + 1,
                              static_cast<int>(original->copySlots.size())));
          if (slot < static_cast<int>(clone.copySlots.size()))
            applyCopySlot(clone,
                          clone.copySlots[static_cast<std::size_t>(slot)]);
          applyCopyPropertyValues(clone, slot);
        }
        instance.nodeMap[nodeId] = clone.id;
        working[clone.id] = WorkingNode{originalId, slot};
        if (provenance != nullptr)
          (*provenance)[clone.id] = {originalId, slot};
        expanded.nodes.push_back(std::move(clone));
      }
      for (const auto &link : internalLinks) {
        const auto sourcePin = instance.pinMap.find(link.sourcePinId);
        const auto destPin = instance.pinMap.find(link.destinationPinId);
        if (sourcePin == instance.pinMap.end() ||
            destPin == instance.pinMap.end())
          continue;
        GraphLink cloned;
        cloned.id = expanded.nextLinkId++;
        cloned.sourcePinId = sourcePin->second;
        cloned.destinationPinId = destPin->second;
        expanded.links.push_back(cloned);
      }
    }

    // Retarget external outs onto the last copy before serial chain links are
    // inserted. Doing this afterward would rewrite copy[i]->copy[i+1] edges
    // onto the last copy and create a directed cycle.
    const auto &last = instances.back();
    for (auto &link : expanded.links) {
      const auto sourceNode = expanded.findNodeForPin(link.sourcePinId);
      if (!sourceNode.has_value() || templateSet.count(*sourceNode) == 0)
        continue;
      const auto dest = expanded.findNodeForPin(link.destinationPinId);
      if (dest.has_value() && templateSet.count(*dest) != 0)
        continue;
      const auto mapped = last.pinMap.find(link.sourcePinId);
      if (mapped != last.pinMap.end())
        link.sourcePinId = mapped->second;
    }

    const auto pairCount = std::min(inputs.size(), outputs.size());
    for (int copy = 0; copy + 1 < copies; ++copy) {
      const auto &current = instances[static_cast<std::size_t>(copy)];
      const auto &next = instances[static_cast<std::size_t>(copy + 1)];
      for (std::size_t index = 0; index < pairCount; ++index) {
        const auto outPin = current.pinMap.find(outputs[index].memberPinId);
        const auto inPin = next.pinMap.find(inputs[index].memberPinId);
        if (outPin == current.pinMap.end() || inPin == next.pinMap.end())
          continue;
        GraphLink chain;
        chain.id = expanded.nextLinkId++;
        chain.sourcePinId = outPin->second;
        chain.destinationPinId = inPin->second;
        expanded.links.push_back(chain);
      }
    }
    group->copies = 1;
  }
  return expanded;
}

ConnectionResult NodeGraph::connect(std::int32_t firstPinId,
                                    std::int32_t secondPinId) {
  firstPinId = resolveCollapsedPin(firstPinId);
  secondPinId = resolveCollapsedPin(secondPinId);
  const auto *first = findPin(firstPinId);
  const auto *second = findPin(secondPinId);
  if (first == nullptr || second == nullptr)
    return {false, "Connection endpoint no longer exists"};
  if (first->kind == second->kind)
    return {false, "Connect an output port to an input port"};

  const auto *source = first->kind == PinKind::output ? first : second;
  const auto *destination = first->kind == PinKind::input ? first : second;

  const auto sourceNode = findNodeForPin(source->id);
  const auto destinationNode = findNodeForPin(destination->id);
  if (!sourceNode.has_value() || !destinationNode.has_value())
    return {false, "Connection endpoint has no owning element"};
  if (*sourceNode == *destinationNode ||
      wouldCreateCycle(*sourceNode, *destinationNode))
    return {false, "Cycles are not allowed in the audio graph"};

  const auto *destinationNodePtr = findNode(*destinationNode);
  const auto *sourceNodePtr = findNode(*sourceNode);

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::merge &&
      destination->kind == PinKind::input &&
      !mergeInputConnectionIsValid(*this, *destinationNodePtr, source->id))
    return {false,
            "Merge inputs must share temporal rate, band count, and channel count"};

  if (sourceNodePtr != nullptr && sourceNodePtr->type == NodeType::merge)
    updateMergeOutputShape(*this, *const_cast<GraphNode *>(sourceNodePtr));

  if (!source->shape.isCompatibleWith(destination->shape)) {
    if (sourceNodePtr != nullptr && sourceNodePtr->type == NodeType::merge) {
      std::unordered_set<std::int32_t> visiting;
      const auto outputChannels =
          computeMergeOutputChannels(*this, *sourceNodePtr, visiting);
      ShapeSignature mergeShape = source->shape;
      if (outputChannels > 0)
        mergeShape.channels = outputChannels;
      if (!mergeShape.isCompatibleWith(destination->shape)) {
        auto message = mergeShape.incompatibilityMessage(destination->shape);
        if (message.empty())
          message = "Shape mismatch: channel counts are incompatible";
        return {false, message};
      }
    } else {
      auto message = source->shape.incompatibilityMessage(destination->shape);
      if (message.empty())
        message = "Shape mismatch: channel counts are incompatible";
      return {false, message};
    }
  }

  if (destinationNodePtr != nullptr &&
      (isConvolutionType(destinationNodePtr->type) ||
       isConvTransposeType(destinationNodePtr->type))) {
    const auto stride =
        std::max(1, readNodeProperty(*destinationNodePtr, "stride", 1));
    const auto upsample = isConvTransposeType(destinationNodePtr->type);
    if (convolutionRateIsError(stride, upsample, source->shape.temporalRate))
      return {false, convolutionRateMessage(stride, upsample,
                                            source->shape.temporalRate)};
  }

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::pqmfSynthesis &&
      pqmfSynthesisChannelIsError(
          source->shape.channels,
          std::max(2, readNodeProperty(*destinationNodePtr, "n_band",
                                       defaultPqmfBands))))
    return {false, pqmfSynthesisChannelMessage(
                       source->shape.channels,
                       std::max(2, readNodeProperty(*destinationNodePtr,
                                                      "n_band",
                                                      defaultPqmfBands)))};

  if (destinationNodePtr != nullptr &&
      destinationNodePtr->type == NodeType::variationalBottleneck &&
      variationalBottleneckChannelIsError(source->shape.channels))
    return {false,
            variationalBottleneckChannelMessage(source->shape.channels)};

  const auto duplicate = std::any_of(
      links.begin(), links.end(), [source, destination](const GraphLink &link) {
        return link.sourcePinId == source->id &&
               link.destinationPinId == destination->id;
      });
  if (duplicate)
    return {false, "These ports are already connected"};

  const auto occupied = std::any_of(
      links.begin(), links.end(), [destination](const GraphLink &link) {
        return link.destinationPinId == destination->id;
      });
  if (occupied)
    return {false, "This input already has a connection"};

  links.push_back({nextLinkId++, source->id, destination->id});
  refreshPropagatedPinShapes(*this);
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    links.pop_back();
    refreshPropagatedPinShapes(*this);
    return {false, incompatible};
  }
  return {true, {}};
}

bool NodeGraph::removeLink(std::int32_t linkId) {
  const auto oldSize = links.size();
  links.erase(std::remove_if(links.begin(), links.end(),
                             [linkId](const GraphLink &link) {
                               return link.id == linkId;
                             }),
              links.end());
  if (links.size() != oldSize)
    refreshAllMergeOutputShapes(*this);
  return links.size() != oldSize;
}

bool NodeGraph::setProperty(std::int32_t nodeId, const std::string &key,
                            int value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end())
    return false;
  const auto previousValue = property->value;
  const auto previousCopyValues = property->copyIntValues;
  property->setValue(value);

  const auto syncCopyValues = [&]() {
    if (!propertySupportsCopyValueList(*property))
      return;
    ensurePropertyCopyCount(*property, effectiveCopyCount(nodeId));
    std::fill(property->copyIntValues.begin(), property->copyIntValues.end(),
              property->value);
  };
  const auto restoreCopyValues = [&]() {
    property->copyIntValues = previousCopyValues;
  };
  syncCopyValues();

  if (key == "channels" && isFixedIoType(node->type)) {
    property->setValue(std::clamp(property->value, 0, 2));
    const auto choice = property->value;
    int pairedPrevious = choice;
    GraphNode *paired = nullptr;
    for (auto &candidate : nodes) {
      if (!isFixedIoType(candidate.type) || candidate.id == node->id)
        continue;
      paired = &candidate;
      for (auto &pairedProperty : candidate.properties) {
        if (pairedProperty.key != "channels")
          continue;
        pairedPrevious = pairedProperty.value;
        pairedProperty.setValue(choice);
        break;
      }
      applyHostIoChannels(candidate);
    }
    applyHostIoChannels(*node);
    refreshPropagatedPinShapes(*this);
    const auto incompatible = firstIncompatibleLinkMessage(*this);
    if (!incompatible.empty()) {
      property->setValue(previousValue);
      restoreCopyValues();
      applyHostIoChannels(*node);
      if (paired != nullptr) {
        for (auto &pairedProperty : paired->properties) {
          if (pairedProperty.key != "channels")
            continue;
          pairedProperty.setValue(pairedPrevious);
          break;
        }
        applyHostIoChannels(*paired);
      }
      refreshPropagatedPinShapes(*this);
      return false;
    }
    syncCopyValues();
    return true;
  } else if (key == "channels" && isConvolutionType(node->type)) {
    for (auto &pin : node->outputs)
      pin.shape.channels = property->value;
  } else if (key == "features" && node->type == NodeType::linear) {
    for (auto &pin : node->outputs)
      pin.shape.channels = property->value;
  } else if (key == "inputs" && isMixerType(node->type)) {
    setMixerInputCount(*node, property->value);
  } else if (key == "mode" && node->type == NodeType::merge) {
    updateMergeOutputShape(*this, *node);
    if (!mergeDownstreamIsCompatible(*this, *node)) {
      property->setValue(previousValue);
      restoreCopyValues();
      updateMergeOutputShape(*this, *node);
      return false;
    }
  } else if (key == "activation" && node->type == NodeType::activation &&
             property->kind == PropertyKind::choice && property->value >= 0 &&
             property->value < static_cast<int>(property->choices.size())) {
    node->detail = property->choices[static_cast<std::size_t>(property->value)];
    node->hasWeights = property->value == 4;
    if (node->hasWeights)
      node->armedForTraining = true;
  } else if (key == "residual" && node->type == NodeType::tcn) {
    node->residual = property->value != 0;
  } else if (key == "dilation_growth" && node->type == NodeType::tcn) {
    node->dilationGrowth =
        std::max(minimumDilationGrowth, property->value);
    property->value = node->dilationGrowth;
  } else if (key == "n_band" &&
             (node->type == NodeType::pqmfAnalysis ||
              node->type == NodeType::pqmfSynthesis)) {
    property->value = std::max(minimumPqmfBands, property->value);
  } else if (key == "latent_size" &&
             node->type == NodeType::variationalBottleneck) {
    if (variationalBottleneckLatentIsError(property->value)) {
      property->setValue(previousValue);
      restoreCopyValues();
      return false;
    }
    for (auto &pin : node->outputs) {
      pin.shape.channels = property->value;
    }
  } else if (key == "stride" &&
             (isConvolutionType(node->type) ||
              isConvTransposeType(node->type))) {
    property->value = std::max(1, property->value);
    if (isConvolutionType(node->type))
      updateConv1dDetail(*node);
    else
      updateConvTransposeDetail(*node);
  }
  refreshPropagatedPinShapes(*this);
  const auto incompatible = firstIncompatibleLinkMessage(*this);
  if (!incompatible.empty()) {
    property->setValue(previousValue);
    restoreCopyValues();
    if (isConvolutionType(node->type))
      updateConv1dDetail(*node);
    else if (isConvTransposeType(node->type))
      updateConvTransposeDetail(*node);
    refreshPropagatedPinShapes(*this);
    return false;
  }
  syncCopyValues();
  return true;
}

bool NodeGraph::setFloatProperty(std::int32_t nodeId, const std::string &key,
                                 float value) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  const bool fidelityOnGold =
      key == "fidelity" && node->state == NodeState::frozenGold;
  if (node->state == NodeState::frozenGold && !fidelityOnGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      property->kind != PropertyKind::real)
    return false;
  property->setFloatValue(value);
  if (key == "fidelity")
    node->fidelityPercent = clampFidelity(property->floatValue);
  if (propertySupportsCopyValueList(*property)) {
    ensurePropertyCopyCount(*property, effectiveCopyCount(nodeId));
    std::fill(property->copyFloatValues.begin(),
              property->copyFloatValues.end(), property->floatValue);
  }
  return true;
}

bool NodeGraph::setPropertyCopyValues(std::int32_t nodeId,
                                      const std::string &key,
                                      const std::vector<int> &values) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->state == NodeState::frozenGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      !propertySupportsCopyValueList(*property) ||
      property->kind != PropertyKind::integer)
    return false;
  const auto copies = effectiveCopyCount(nodeId);
  if (static_cast<int>(values.size()) != copies)
    return false;
  std::vector<int> clamped;
  clamped.reserve(values.size());
  for (const auto value : values) {
    if (value < property->minimum || value > property->maximum)
      return false;
    clamped.push_back(std::clamp(value, property->minimum, property->maximum));
  }
  if (!setProperty(nodeId, key, clamped.front()))
    return false;
  property->copyIntValues = std::move(clamped);
  property->value = property->copyIntValues.front();
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::setFloatPropertyCopyValues(std::int32_t nodeId,
                                           const std::string &key,
                                           const std::vector<float> &values) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  const bool fidelityOnGold =
      key == "fidelity" && node->state == NodeState::frozenGold;
  if (node->state == NodeState::frozenGold && !fidelityOnGold)
    return false;
  const auto property = std::find_if(
      node->properties.begin(), node->properties.end(),
      [&key](const NodeProperty &candidate) { return candidate.key == key; });
  if (property == node->properties.end() ||
      !propertySupportsCopyValueList(*property) ||
      property->kind != PropertyKind::real)
    return false;
  const auto copies = effectiveCopyCount(nodeId);
  if (static_cast<int>(values.size()) != copies)
    return false;
  std::vector<float> clamped;
  clamped.reserve(values.size());
  for (const auto value : values) {
    if (value < property->floatMinimum || value > property->floatMaximum)
      return false;
    clamped.push_back(
        std::clamp(value, property->floatMinimum, property->floatMaximum));
  }
  if (!setFloatProperty(nodeId, key, clamped.front()))
    return false;
  property->copyFloatValues = std::move(clamped);
  property->floatValue = property->copyFloatValues.front();
  if (key == "fidelity")
    node->fidelityPercent = clampFidelity(property->floatValue);
  refreshPropagatedPinShapes(*this);
  return true;
}

bool NodeGraph::setConditioningValue(std::int32_t nodeId, float value) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->type != NodeType::knobInput)
    return false;
  node->conditioningValue = clampConditioning(value);
  return true;
}

bool NodeGraph::setConditioningPad(std::int32_t nodeId, float x, float y) {
  auto *node = findNode(nodeId);
  if (node == nullptr || node->type != NodeType::xyTrackpad)
    return false;
  node->conditioningX = clampConditioning(x);
  node->conditioningY = clampConditioning(y);
  return true;
}

bool NodeGraph::setSelectedAnalysisView(std::int32_t nodeId,
                                        AnalysisView view) {
  auto *node = findNode(nodeId);
  if (node == nullptr)
    return false;
  node->selectedAnalysisView = view;
  return true;
}

bool NodeGraph::setSeed(std::int32_t nodeId, std::int32_t seed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights ||
      node->state == NodeState::frozenGold)
    return false;
  node->seed = clampSeed(seed);
  if (node->useExplicitSeed)
    node->explicitSeed = node->seed;
  if (!node->copySlots.empty() &&
      node->weightsProvenance == WeightsProvenance::random) {
    for (std::size_t index = 0; index < node->copySlots.size(); ++index) {
      auto &slot = node->copySlots[index];
      if (slot.provenance != WeightsProvenance::random)
        continue;
      slot.seed = seedForCopySlot(node->seed, index);
    }
    node->copySlots.front() = copySlotFromNode(*node);
  }
  return true;
}

std::optional<std::int32_t>
NodeGraph::freezeSelection(const std::vector<std::int32_t> &selectedNodeIds,
                           const FreezeSelectionResult &result) {
  if (!result.succeeded || selectedNodeIds.empty() ||
      result.inputChannels < 1 || result.outputChannels < 1 ||
      !selectionIsConnected(selectedNodeIds))
    return std::nullopt;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return std::nullopt;
  }

  for (const auto nodeId : selectedNodeIds) {
    auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    node->state = NodeState::frozenGold;
    node->colour = colourForType(node->type, node->state);
    node->artifactPath = result.artifactPath;
    node->metrics.reset();
    node->sourceSubgraph.clear();
  }

  std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                            selectedNodeIds.end());
  std::int32_t sinkId = 0;
  for (const auto nodeId : selectedNodeIds) {
    bool hasSuccessorInSelection = false;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (source.has_value() && *source == nodeId &&
          destination.has_value() && selected.count(*destination) != 0) {
        hasSuccessorInSelection = true;
        break;
      }
    }
    if (!hasSuccessorInSelection) {
      sinkId = nodeId;
      break;
    }
  }
  if (auto *sink = findNode(sinkId != 0 ? sinkId : selectedNodeIds.back()))
    sink->metrics = result.baselineMetrics;
  return selectedNodeIds.front();
}

bool NodeGraph::unfreeze(std::int32_t nodeId) {
  const auto *node = findNode(nodeId);
  if (node == nullptr || node->state != NodeState::frozenGold)
    return false;

  if (node->type == NodeType::blackBox) {
    const auto fragmentXml = juce::XmlDocument::parse(node->sourceSubgraph);
    if (fragmentXml == nullptr)
      return false;
    const auto fragment = juce::ValueTree::fromXml(*fragmentXml);
    if (!fragment.hasType("GraphFragment"))
      return false;
    const auto trainedPath =
        node->weightsPath.empty() ? node->artifactPath : node->weightsPath;
    const auto keepTrainedWeights =
        node->blackBoxOrigin == BlackBoxOrigin::trainAutoload ||
        node->weightsProvenance == WeightsProvenance::file;
    const auto trainOrigin =
        node->blackBoxOrigin == BlackBoxOrigin::trainAutoload;
    const auto fidelity = node->fidelityPercent;

    removeNode(nodeId);
    for (const auto child : fragment) {
      if (!child.hasType("Node"))
        continue;
      auto restored = nodeFromTree(child);
      nextNodeId = std::max(nextNodeId, restored.id + 1);
      for (const auto &pin : restored.inputs)
        nextPinId = std::max(nextPinId, pin.id + 1);
      for (const auto &pin : restored.outputs)
        nextPinId = std::max(nextPinId, pin.id + 1);
      if (keepTrainedWeights && restored.hasWeights) {
        restored.weightsProvenance = WeightsProvenance::file;
        restored.weightsPath = trainedPath;
        restored.artifactPath = trainedPath;
        if (trainOrigin)
          restored.blackBoxOrigin = BlackBoxOrigin::trainAutoload;
      }
      if (restored.type == NodeType::variationalBottleneck) {
        restored.fidelityPercent = fidelity;
        for (auto &property : restored.properties) {
          if (property.key == "fidelity")
            property.floatValue = fidelity;
        }
        copyCompactnessFromArtifact(restored, trainedPath);
      }
      nodes.push_back(std::move(restored));
    }
    for (const auto child : fragment) {
      if (!child.hasType("Link"))
        continue;
      auto restored = linkFromTree(child);
      if (findPin(restored.sourcePinId) != nullptr &&
          findPin(restored.destinationPinId) != nullptr) {
        nextLinkId = std::max(nextLinkId, restored.id + 1);
        links.push_back(restored);
      }
    }
    return true;
  }

  const auto artifactPath = node->artifactPath;
  for (auto &candidate : nodes) {
    if (candidate.state != NodeState::frozenGold)
      continue;
    const auto sameGroup =
        artifactPath.empty() ? candidate.id == nodeId
                             : candidate.artifactPath == artifactPath;
    if (!sameGroup)
      continue;
    if (!artifactPath.empty()) {
      candidate.weightsProvenance = WeightsProvenance::file;
      candidate.weightsPath = artifactPath;
    }
    candidate.state = NodeState::liveBlue;
    candidate.colour = colourForType(candidate.type, candidate.state);
    candidate.artifactPath.clear();
    candidate.sourceSubgraph.clear();
    candidate.metrics.reset();
  }
  return true;
}

std::optional<FreezeSelectionRequest> NodeGraph::createFreezeRequest(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  if (!selectionIsConnected(selectedNodeIds))
    return std::nullopt;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr || isFixedIoType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return std::nullopt;
  }

  const std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                                  selectedNodeIds.end());
  FreezeSelectionRequest request;
  request.requestId = juce::Uuid().toString().toStdString();
  request.selectedNodeIds = selectedNodeIds;

  auto root = std::make_unique<juce::DynamicObject>();
  root->setProperty("request_id", juce::String(request.requestId));
  root->setProperty("operation", "freeze_selection");

  juce::Array<juce::var> selectedIds;
  juce::Array<juce::var> elements;
  for (const auto nodeId : selectedNodeIds) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    selectedIds.add(nodeId);

    auto element = std::make_unique<juce::DynamicObject>();
    element->setProperty("id", node->id);
    element->setProperty("type", nodeTypeName(node->type));
    element->setProperty("label", juce::String(node->label));
    element->setProperty("seed", node->seed);
    juce::Array<juce::var> properties;
    for (const auto &property : node->properties) {
      auto serialized = std::make_unique<juce::DynamicObject>();
      serialized->setProperty("key", juce::String(property.key));
      serialized->setProperty("value", property.value);
      properties.add(juce::var(serialized.release()));
    }
    element->setProperty("properties", properties);
    elements.add(juce::var(element.release()));
  }
  root->setProperty("selected_element_ids", selectedIds);

  auto fragment = std::make_unique<juce::DynamicObject>();
  fragment->setProperty("elements", elements);
  juce::Array<juce::var> connections;
  juce::Array<juce::var> boundaryInputs;
  juce::Array<juce::var> boundaryOutputs;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto sourceSelected = selected.count(*source) != 0;
    const auto destinationSelected = selected.count(*destination) != 0;
    if (!sourceSelected && !destinationSelected)
      continue;

    auto serialized = std::make_unique<juce::DynamicObject>();
    serialized->setProperty("connection_id", link.id);
    serialized->setProperty("source_element_id", *source);
    serialized->setProperty("source_pin_id", link.sourcePinId);
    serialized->setProperty("destination_element_id", *destination);
    serialized->setProperty("destination_pin_id", link.destinationPinId);
    if (sourceSelected && destinationSelected)
      connections.add(juce::var(serialized.release()));
    else if (destinationSelected)
      boundaryInputs.add(juce::var(serialized.release()));
    else
      boundaryOutputs.add(juce::var(serialized.release()));
  }
  fragment->setProperty("connections", connections);
  auto boundary = std::make_unique<juce::DynamicObject>();
  boundary->setProperty("inputs", boundaryInputs);
  boundary->setProperty("outputs", boundaryOutputs);
  fragment->setProperty("io_boundary", juce::var(boundary.release()));
  root->setProperty("graph_fragment", juce::var(fragment.release()));

  auto options = std::make_unique<juce::DynamicObject>();
  auto sourceIds = selected;
  auto sinkIds = selected;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (source.has_value() && destination.has_value() &&
        selected.count(*source) != 0 && selected.count(*destination) != 0) {
      sourceIds.erase(*destination);
      sinkIds.erase(*source);
    }
  }
  if (sourceIds.size() != 1 || sinkIds.size() != 1)
    return std::nullopt;
  const auto *sourceNode = findNode(*sourceIds.begin());
  const auto *sinkNode = findNode(*sinkIds.begin());
  if (sourceNode == nullptr || sinkNode == nullptr)
    return std::nullopt;
  const auto inputChannels =
      !sourceNode->inputs.empty()
          ? sourceNode->inputs.front().shape.channels
          : (!sourceNode->outputs.empty()
                 ? sourceNode->outputs.front().shape.channels
                 : 0);
  const auto outputChannels =
      !sinkNode->outputs.empty()
          ? sinkNode->outputs.front().shape.channels
          : (!sinkNode->inputs.empty() ? sinkNode->inputs.front().shape.channels
                                       : 0);
  options->setProperty("mode", "manual_freeze");
  options->setProperty("host_input_channels", inputChannels);
  options->setProperty("host_output_channels", outputChannels);
  options->setProperty("example_samples", 256);
  int condDim = 0;
  for (const auto &link : links) {
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!destination.has_value() || selected.count(*destination) == 0)
      continue;
    const auto *destinationPin = findPin(link.destinationPinId);
    if (destinationPin == nullptr || !isControlInputPin(*destinationPin))
      continue;
    const auto *sourcePin = findPin(link.sourcePinId);
    if (sourcePin != nullptr && sourcePin->shape.channels > 0)
      condDim = sourcePin->shape.channels;
  }
  options->setProperty("cond_dim", condDim);
  options->setProperty("conditioning", condDim > 0);
  root->setProperty("compile_options", juce::var(options.release()));
  request.graphFragment =
      juce::JSON::toString(juce::var(root.release()), true).toStdString();
  return request;
}

const std::vector<GraphNode> &NodeGraph::getNodes() const noexcept {
  return nodes;
}

std::vector<GraphNode> &NodeGraph::getNodes() noexcept { return nodes; }

const std::vector<GraphLink> &NodeGraph::getLinks() const noexcept {
  return links;
}

std::vector<GraphLink> &NodeGraph::getLinks() noexcept { return links; }

GraphNode *NodeGraph::findNode(std::int32_t nodeId) noexcept {
  const auto found =
      std::find_if(nodes.begin(), nodes.end(), [nodeId](const GraphNode &node) {
        return node.id == nodeId;
      });
  return found != nodes.end() ? &*found : nullptr;
}

const GraphNode *NodeGraph::findNode(std::int32_t nodeId) const noexcept {
  const auto found =
      std::find_if(nodes.begin(), nodes.end(), [nodeId](const GraphNode &node) {
        return node.id == nodeId;
      });
  return found != nodes.end() ? &*found : nullptr;
}

const Pin *NodeGraph::findPin(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  for (const auto &node : nodes) {
    const auto input =
        std::find_if(node.inputs.begin(), node.inputs.end(),
                     [pinId](const Pin &pin) { return pin.id == pinId; });
    if (input != node.inputs.end())
      return &*input;
    const auto output =
        std::find_if(node.outputs.begin(), node.outputs.end(),
                     [pinId](const Pin &pin) { return pin.id == pinId; });
    if (output != node.outputs.end())
      return &*output;
  }
  return nullptr;
}

const GraphLink *NodeGraph::findLink(std::int32_t linkId) const noexcept {
  const auto found =
      std::find_if(links.begin(), links.end(), [linkId](const GraphLink &link) {
        return link.id == linkId;
      });
  return found != links.end() ? &*found : nullptr;
}

bool NodeGraph::isFixedIoNode(std::int32_t nodeId) const noexcept {
  const auto *node = findNode(nodeId);
  return node != nullptr && isFixedIoType(node->type);
}

std::optional<std::int32_t>
NodeGraph::findNodeForPin(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  for (const auto &node : nodes) {
    const auto owns = [pinId](const Pin &pin) { return pin.id == pinId; };
    if (std::any_of(node.inputs.begin(), node.inputs.end(), owns) ||
        std::any_of(node.outputs.begin(), node.outputs.end(), owns))
      return node.id;
  }
  return std::nullopt;
}

ViewportState &NodeGraph::getViewport() noexcept { return viewport; }

const ViewportState &NodeGraph::getViewport() const noexcept {
  return viewport;
}

juce::ValueTree NodeGraph::toValueTree() const {
  juce::ValueTree tree{"GraphDocument"};
  tree.setProperty("version", 2, nullptr);
  tree.setProperty("panX", viewport.pan.x, nullptr);
  tree.setProperty("panY", viewport.pan.y, nullptr);
  tree.setProperty("zoom", viewport.zoom, nullptr);
  tree.setProperty("mapVisible", viewport.mapVisible, nullptr);
  if (viewport.focusedGroupId.has_value())
    tree.setProperty("focusedGroupId", *viewport.focusedGroupId, nullptr);
  for (const auto &node : nodes)
    tree.appendChild(nodeToTree(node), nullptr);
  for (const auto &group : groups)
    tree.appendChild(groupToTree(group), nullptr);
  for (const auto &link : links)
    tree.appendChild(linkToTree(link), nullptr);
  return tree;
}

bool NodeGraph::documentIsRestorable(const juce::ValueTree &tree,
                                     juce::String &error) {
  error.clear();
  if (!tree.hasType("GraphDocument")) {
    error = "Preset graph document is missing or corrupt";
    return false;
  }
  for (const auto child : tree) {
    if (!child.hasType("Node"))
      continue;
    const auto typeName = child["type"].toString();
    if (!isKnownPersistedNodeType(typeName)) {
      error = "Preset uses unknown element type '" + typeName + "'";
      return false;
    }
  }
  return true;
}

bool NodeGraph::restoreFromValueTree(const juce::ValueTree &tree) {
  juce::String error;
  if (!documentIsRestorable(tree, error))
    return false;

  std::vector<GraphNode> restoredNodes;
  std::vector<GraphGroup> restoredGroups;
  std::vector<GraphLink> restoredLinks;
  std::int32_t restoredNextNode = 1;
  std::int32_t restoredNextPin = 1001;
  std::int32_t restoredNextLink = 2001;
  for (const auto child : tree) {
    if (child.hasType("Node")) {
      auto node = nodeFromTree(child);
      restoredNextNode = std::max(restoredNextNode, node.id + 1);
      for (const auto &pin : node.inputs)
        restoredNextPin = std::max(restoredNextPin, pin.id + 1);
      for (const auto &pin : node.outputs)
        restoredNextPin = std::max(restoredNextPin, pin.id + 1);
      restoredNodes.push_back(std::move(node));
    } else if (child.hasType("Group")) {
      auto group = groupFromTree(child);
      restoredNextNode = std::max(restoredNextNode, group.id + 1);
      restoredGroups.push_back(std::move(group));
    } else if (child.hasType("Link")) {
      auto link = linkFromTree(child);
      restoredNextLink = std::max(restoredNextLink, link.id + 1);
      restoredLinks.push_back(link);
    }
  }
  nodes = std::move(restoredNodes);
  groups = std::move(restoredGroups);
  links = std::move(restoredLinks);
  nextNodeId = restoredNextNode;
  nextPinId = restoredNextPin;
  nextLinkId = restoredNextLink;
  viewport.pan = {static_cast<float>(tree.getProperty("panX", 0.0f)),
                  static_cast<float>(tree.getProperty("panY", 0.0f))};
  viewport.zoom = std::clamp(static_cast<float>(tree.getProperty("zoom", 1.0f)),
                             minimumZoom, maximumZoom);
  viewport.mapVisible = static_cast<bool>(tree.getProperty("mapVisible", true));
  if (tree.hasProperty("focusedGroupId"))
    viewport.focusedGroupId =
        static_cast<std::int32_t>(tree.getProperty("focusedGroupId"));
  else
    viewport.focusedGroupId.reset();
  if (viewport.focusedGroupId.has_value() &&
      findGroup(*viewport.focusedGroupId) == nullptr)
    viewport.focusedGroupId.reset();

  links.erase(std::remove_if(links.begin(), links.end(),
                             [this](const GraphLink &link) {
                               return findPin(link.sourcePinId) == nullptr ||
                                      findPin(link.destinationPinId) == nullptr;
                             }),
              links.end());
  ensureFixedHostIo();
  refreshAllMergeOutputShapes(*this);
  for (auto &node : nodes) {
    if (node.type != NodeType::tcn && node.type != NodeType::blackBox)
      continue;
    bool hasControl = false;
    for (auto &pin : node.inputs) {
      if (isControlInputPin(pin)) {
        pin.label = controlPinLabel;
        pin.shape = flexibleTensorShape();
        hasControl = true;
      }
    }
    if (!hasControl)
      node.inputs.push_back({nextPinId++, controlPinLabel, PinKind::input,
                             flexibleTensorShape()});
  }
  return true;
}

juce::ValueTree NodeGraph::exportBox(std::int32_t boxId,
                                     juce::String &error) const {
  error.clear();
  juce::ValueTree tree{"BoxSnapshot"};
  tree.setProperty("version", 1, nullptr);
  tree.setProperty("rootId", boxId, nullptr);
  if (const auto *node = findNode(boxId)) {
    if (isFixedIoType(node->type)) {
      error = "Audio Input and Audio Output cannot be saved to the box library";
      return {};
    }
    auto cloned = nodeToTree(*node);
    cloned.removeProperty("parentGroupId", nullptr);
    tree.setProperty("kind", "element", nullptr);
    tree.setProperty("rootTypeHint", nodeTypeName(node->type), nullptr);
    tree.appendChild(cloned, nullptr);
    return tree;
  }
  if (findGroup(boxId) != nullptr) {
    std::unordered_set<std::int32_t> nodeIds;
    std::unordered_set<std::int32_t> groupIds;
    collectGroupSubtree(*this, boxId, nodeIds, groupIds);
    for (const auto nodeId : nodeIds) {
      const auto *member = findNode(nodeId);
      if (member != nullptr && isFixedIoType(member->type)) {
        error =
            "Audio Input and Audio Output cannot be saved to the box library";
        return {};
      }
    }
    tree.setProperty("kind", "group", nullptr);
    tree.setProperty("rootTypeHint", "group", nullptr);
    for (const auto nodeId : nodeIds) {
      if (const auto *member = findNode(nodeId))
        tree.appendChild(nodeToTree(*member), nullptr);
    }
    for (const auto groupId : groupIds) {
      if (const auto *nested = findGroup(groupId)) {
        auto cloned = groupToTree(*nested);
        if (groupId == boxId)
          cloned.removeProperty("parentGroupId", nullptr);
        tree.appendChild(cloned, nullptr);
      }
    }
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (nodeIds.count(*source) == 0 || nodeIds.count(*destination) == 0)
        continue;
      tree.appendChild(linkToTree(link), nullptr);
    }
    return tree;
  }
  error = "Save to Box Library applies to one element or one group";
  return {};
}

std::optional<std::int32_t>
NodeGraph::importBox(const juce::ValueTree &snapshot, juce::Point<float> position,
                     bool collapseGroups, juce::String &error) {
  error.clear();
  if (!snapshot.hasType("BoxSnapshot")) {
    error = "Box library entry is not a valid snapshot";
    return std::nullopt;
  }
  for (const auto child : snapshot) {
    if (!child.hasType("Node"))
      continue;
    const auto typeName = child["type"].toString();
    if (!isKnownPersistedNodeType(typeName)) {
      error = "Box uses unknown element type '" + typeName + "'";
      return std::nullopt;
    }
    if (isFixedIoType(nodeTypeFromName(typeName))) {
      error = "Audio Input and Audio Output cannot be inserted from the library";
      return std::nullopt;
    }
  }

  const auto backup = toValueTree();
  const auto originalRoot = static_cast<std::int32_t>(snapshot["rootId"]);
  std::unordered_map<std::int32_t, std::int32_t> idMap;
  std::unordered_map<std::int32_t, std::int32_t> pinMap;
  std::vector<GraphNode> importedNodes;
  std::vector<GraphGroup> importedGroups;
  std::vector<GraphLink> importedLinks;

  const auto remapId = [this, &idMap](std::int32_t oldId) {
    const auto found = idMap.find(oldId);
    if (found != idMap.end())
      return found->second;
    const auto fresh = nextNodeId++;
    idMap.emplace(oldId, fresh);
    return fresh;
  };

  for (const auto child : snapshot) {
    if (child.hasType("Node")) {
      auto node = nodeFromTree(child);
      node.id = remapId(node.id);
      if (node.parentGroupId.has_value())
        node.parentGroupId = remapId(*node.parentGroupId);
      for (auto &pin : node.inputs) {
        const auto fresh = nextPinId++;
        pinMap.emplace(pin.id, fresh);
        pin.id = fresh;
      }
      for (auto &pin : node.outputs) {
        const auto fresh = nextPinId++;
        pinMap.emplace(pin.id, fresh);
        pin.id = fresh;
      }
      importedNodes.push_back(std::move(node));
    } else if (child.hasType("Group")) {
      auto group = groupFromTree(child);
      group.id = remapId(group.id);
      if (group.parentGroupId.has_value())
        group.parentGroupId = remapId(*group.parentGroupId);
      if (collapseGroups)
        group.collapsed = true;
      importedGroups.push_back(std::move(group));
    }
  }

  for (auto &group : importedGroups) {
    for (auto &member : group.memberIds) {
      const auto found = idMap.find(member);
      if (found == idMap.end()) {
        error = "Box snapshot has a missing group member";
        restoreFromValueTree(backup);
        return std::nullopt;
      }
      member = found->second;
    }
  }

  for (const auto child : snapshot) {
    if (!child.hasType("Link"))
      continue;
    auto link = linkFromTree(child);
    const auto source = pinMap.find(link.sourcePinId);
    const auto destination = pinMap.find(link.destinationPinId);
    if (source == pinMap.end() || destination == pinMap.end()) {
      error = "Box snapshot has a connection to a missing pin";
      restoreFromValueTree(backup);
      return std::nullopt;
    }
    link.id = nextLinkId++;
    link.sourcePinId = source->second;
    link.destinationPinId = destination->second;
    importedLinks.push_back(link);
  }

  const auto rootFound = idMap.find(originalRoot);
  if (rootFound == idMap.end()) {
    error = "Box snapshot is missing its root box";
    restoreFromValueTree(backup);
    return std::nullopt;
  }
  const auto newRoot = rootFound->second;

  for (auto &node : importedNodes) {
    if (node.id == newRoot)
      node.position = position;
    nodes.push_back(std::move(node));
  }
  for (auto &group : importedGroups) {
    if (group.id == newRoot)
      group.position = position;
    groups.push_back(std::move(group));
  }
  for (const auto &link : importedLinks)
    links.push_back(link);

  refreshAllMergeOutputShapes(*this);
  return newRoot;
}

std::string NodeGraph::toJson() const {
  auto root = std::make_unique<juce::DynamicObject>();
  juce::Array<juce::var> nodeArray;
  for (const auto &node : nodes) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", node.id);
    object->setProperty("type", nodeTypeName(node.type));
    object->setProperty("label", juce::String(node.label));
    object->setProperty("x", node.position.x);
    object->setProperty("y", node.position.y);
    object->setProperty("seed", node.seed);
    object->setProperty("state", node.state == NodeState::frozenGold
                                     ? "frozen_gold"
                                     : "live_blue");
    juce::Array<juce::var> inputs;
    for (const auto &pin : node.inputs) {
      auto endpoint = std::make_unique<juce::DynamicObject>();
      endpoint->setProperty("id", pin.id);
      endpoint->setProperty("channels", pin.shape.channels);
      inputs.add(juce::var(endpoint.release()));
    }
    object->setProperty("inputs", inputs);
    juce::Array<juce::var> outputs;
    for (const auto &pin : node.outputs) {
      auto endpoint = std::make_unique<juce::DynamicObject>();
      endpoint->setProperty("id", pin.id);
      endpoint->setProperty("channels", pin.shape.channels);
      outputs.add(juce::var(endpoint.release()));
    }
    object->setProperty("outputs", outputs);
    juce::Array<juce::var> properties;
    for (const auto &property : node.properties) {
      auto value = std::make_unique<juce::DynamicObject>();
      value->setProperty("key", juce::String(property.key));
      value->setProperty("value", property.value);
      properties.add(juce::var(value.release()));
    }
    object->setProperty("properties", properties);
    nodeArray.add(juce::var(object.release()));
  }
  juce::Array<juce::var> linkArray;
  for (const auto &link : links) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("id", link.id);
    object->setProperty("source_pin", link.sourcePinId);
    object->setProperty("destination_pin", link.destinationPinId);
    linkArray.add(juce::var(object.release()));
  }
  root->setProperty("elements", nodeArray);
  root->setProperty("connections", linkArray);
  return juce::JSON::toString(juce::var(root.release()), true).toStdString();
}

GraphNode NodeGraph::makeNode(NodeType type, juce::Point<float> position) {
  if (type == NodeType::rateConv)
    type = NodeType::convolution;
  GraphNode node;
  node.id = nextNodeId++;
  node.type = type;
  node.position = position;
  node.state =
      type == NodeType::blackBox ? NodeState::frozenGold : NodeState::liveBlue;
  node.colour = colourForType(type, node.state);

  const auto addInput = [&](const char *label = "in", int channels = 0) {
    node.inputs.push_back(
        {nextPinId++, label, PinKind::input, {channels}});
  };
  const auto addOutput = [&](const char *label = "out", int channels = 0) {
    node.outputs.push_back(
        {nextPinId++, label, PinKind::output, {channels}});
  };
  const auto property = [](std::string key, std::string label, int value,
                           int minimum, int maximum,
                           PropertyKind kind = PropertyKind::integer,
                           std::vector<std::string> choices = {}) {
    return NodeProperty{std::move(key),    std::move(label), value,
                        minimum,           maximum,          kind,
                        std::move(choices)};
  };
  const auto gainProperty = []() {
    NodeProperty gain;
    gain.key = "gain";
    gain.label = "Gain";
    gain.kind = PropertyKind::real;
    gain.floatValue = gainDefault;
    gain.floatMinimum = gainMinimum;
    gain.floatMaximum = gainMaximum;
    return gain;
  };
  const auto fidelityProperty = []() {
    NodeProperty fidelity;
    fidelity.key = "fidelity";
    fidelity.label = "Fidelity";
    fidelity.kind = PropertyKind::real;
    fidelity.floatValue = defaultFidelityPercent;
    fidelity.floatMinimum = fidelityMinimum;
    fidelity.floatMaximum = fidelityMaximum;
    return fidelity;
  };

  switch (type) {
  case NodeType::audioInput:
    node.label = "Audio Input";
    node.detail = hostIoModeDetail(HostIoMode::stereo, true);
    addOutput("out", 2);
    node.properties.push_back(property("channels", "Mode", 2, 0, 2,
                                       PropertyKind::choice,
                                       {"Mono", "Mirrored", "Stereo"}));
    applyHostIoChannels(node);
    break;
  case NodeType::audioOutput:
    node.label = "Audio Output";
    node.detail = hostIoModeDetail(HostIoMode::stereo, false);
    addInput("in", 2);
    node.properties.push_back(property("channels", "Mode", 2, 0, 2,
                                       PropertyKind::choice,
                                       {"Mono", "Mirrored", "Stereo"}));
    applyHostIoChannels(node);
    break;
  case NodeType::linear:
    node.label = "Linear";
    node.detail = "Weighted projection";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("features", "Features", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::convolution:
  case NodeType::rateConv:
    node.label = "Conv1D";
    node.detail = "Temporal convolution";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("channels", "Channels", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation", "Dilation", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("stride", "Stride", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("weight_norm", "Weight Norm", 0, 0, 1));
    break;
  case NodeType::convTranspose:
    node.label = "ConvTranspose1d";
    node.detail = "Temporal upsampling";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("channels", "Channels", 2,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation", "Dilation", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("stride", "Stride", 1,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("weight_norm", "Weight Norm", 0, 0, 1));
    break;
  case NodeType::batchNorm:
    node.label = "BatchNorm1d";
    node.detail = "Affine normalization";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    break;
  case NodeType::activation:
    node.label = "Activation";
    node.detail = "ReLU";
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("activation", "Function", 0, 0, 4, PropertyKind::choice,
                 {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}));
    node.properties.push_back(gainProperty());
    break;
  case NodeType::tcn:
    node.label = "TCN";
    node.detail = "Live modular network";
    node.hasWeights = true;
    node.armedForTraining = true;
    node.dilationGrowth = defaultDilationGrowth;
    addInput();
    addInput(controlPinLabel);
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.inputs.back().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(property("depth", "Depth", 4, minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size", 3,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("channels", "Channels", 16,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("dilation_growth", "Dilation growth",
                                       defaultDilationGrowth,
                                       minimumDilationGrowth,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("residual", "Residual", 0, 0, 1));
    node.properties.push_back(
        property("activation", "Activation", 0, 0, 4, PropertyKind::choice,
                 {"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}));
    node.properties.push_back(gainProperty());
    break;
  case NodeType::merge:
    node.label = "Merge";
    node.detail = "Elementwise combine";
    addOutput();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("mode", "Mode", 0, 0, 2, PropertyKind::choice,
                 {"Add", "Multiply", "Concatenate"}));
    node.properties.push_back(property("inputs", "Inputs", 2, 2,
                                       unlimitedPropertyMaximum));
    setMixerInputCount(node, 2);
    break;
  case NodeType::knobInput:
    node.label = "Knob Input";
    node.detail = "1D conditioning";
    addOutput("c", 1);
    break;
  case NodeType::xyTrackpad:
    node.label = "XY Trackpad";
    node.detail = "Independent X and Y outputs";
    addOutput("x", 1);
    addOutput("y", 1);
    break;
  case NodeType::blackBox:
    node.label = "Frozen Selection";
    node.detail = "Locked";
    node.hasWeights = true;
    addInput();
    addInput(controlPinLabel);
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.inputs.back().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(fidelityProperty());
    break;
  case NodeType::pqmfAnalysis:
    node.label = "PQMF Analysis";
    node.detail = "Audio to multiband";
    addInput("in", 0);
    addOutput("bands", 0);
    node.outputs.front().shape.temporalRate = defaultPqmfBands;
    node.outputs.front().shape.nBand = defaultPqmfBands;
    node.properties.push_back(property("n_band", "nBand", defaultPqmfBands,
                                       minimumPqmfBands,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::pqmfSynthesis:
    node.label = "PQMF Synthesis";
    node.detail = "Multiband to audio";
    addInput("bands", 0);
    addOutput("out", 0);
    node.inputs.front().shape.temporalRate = defaultPqmfBands;
    node.inputs.front().shape.nBand = defaultPqmfBands;
    node.properties.push_back(property("n_band", "nBand", defaultPqmfBands,
                                       minimumPqmfBands,
                                       unlimitedPropertyMaximum));
    break;
  case NodeType::variationalBottleneck:
    node.label = "Variational Bottleneck";
    node.detail = "Latent sample";
    node.hasWeights = true;
    node.armedForTraining = true;
    node.fidelityPercent = defaultFidelityPercent;
    addInput("features", 0);
    addOutput("z", defaultLatentSize);
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape.temporalRate = 0;
    node.properties.push_back(property("latent_size", "Latent", defaultLatentSize,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(property("kernel_size", "Kernel Size",
                                       defaultBottleneckKernelSize,
                                       minimumPositiveProperty,
                                       unlimitedPropertyMaximum));
    node.properties.push_back(fidelityProperty());
    break;
  case NodeType::noiseSynthesizer:
    node.label = "Noise Synth";
    node.detail = "Filtered noise addend";
    node.hasWeights = true;
    node.armedForTraining = true;
    addInput();
    addOutput();
    node.inputs.front().shape = flexibleTensorShape();
    node.outputs.front().shape = flexibleTensorShape();
    node.properties.push_back(
        property("noise_bands", "Noise bands", defaultNoiseBands,
                 minimumPositiveProperty, unlimitedPropertyMaximum));
    break;
  }
  return node;
}

void NodeGraph::setMixerInputCount(GraphNode &node, int inputCount) {
  const auto count = std::max(inputCount, 2);
  while (static_cast<int>(node.inputs.size()) > count) {
    const auto pinId = node.inputs.back().id;
    links.erase(std::remove_if(links.begin(), links.end(),
                               [pinId](const GraphLink &link) {
                                 return link.sourcePinId == pinId ||
                                        link.destinationPinId == pinId;
                               }),
                links.end());
    node.inputs.pop_back();
  }
  while (static_cast<int>(node.inputs.size()) < count) {
    const auto index = static_cast<int>(node.inputs.size()) + 1;
    node.inputs.push_back({nextPinId++, "in " + std::to_string(index),
                           PinKind::input, flexibleTensorShape()});
  }
  for (int index = 0; index < static_cast<int>(node.inputs.size()); ++index)
    node.inputs[static_cast<std::size_t>(index)].label =
        "in " + std::to_string(index + 1);
}

bool NodeGraph::wouldCreateCycle(std::int32_t sourceNodeId,
                                 std::int32_t destinationNodeId) const {
  std::queue<std::int32_t> pending;
  std::unordered_set<std::int32_t> visited;
  pending.push(destinationNodeId);
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (current == sourceNodeId)
      return true;
    if (!visited.insert(current).second)
      continue;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (source.has_value() && destination.has_value() && *source == current)
        pending.push(*destination);
    }
  }
  return false;
}

bool NodeGraph::selectionIsConnected(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  if (selectedNodeIds.empty())
    return false;
  const std::unordered_set<std::int32_t> selected(selectedNodeIds.begin(),
                                                  selectedNodeIds.end());
  for (const auto id : selected) {
    const auto *node = findNode(id);
    if (node == nullptr || node->state != NodeState::liveBlue)
      return false;
  }
  if (selected.size() == 1)
    return true;

  std::queue<std::int32_t> pending;
  std::unordered_set<std::int32_t> visited;
  pending.push(*selected.begin());
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop();
    if (!visited.insert(current).second)
      continue;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (*source == current && selected.count(*destination) != 0)
        pending.push(*destination);
      if (*destination == current && selected.count(*source) != 0)
        pending.push(*source);
    }
  }
  return visited.size() == selected.size();
}

bool NodeGraph::isPinConnected(std::int32_t pinId) const noexcept {
  pinId = resolveCollapsedPin(pinId);
  return std::any_of(links.begin(), links.end(),
                     [pinId](const GraphLink &link) {
                       return link.sourcePinId == pinId ||
                              link.destinationPinId == pinId;
                     });
}

std::vector<std::vector<std::int32_t>> NodeGraph::partitionFreezeChains(
    const std::vector<std::int32_t> &selectedNodeIds) const {
  std::unordered_set<std::int32_t> selected;
  for (const auto id : selectedNodeIds) {
    const auto *node = findNode(id);
    if (node == nullptr || isFixedIoType(node->type) ||
        isConditioningSourceType(node->type) ||
        node->state != NodeState::liveBlue)
      return {};
    selected.insert(id);
  }
  if (selected.empty())
    return {};

  std::unordered_map<std::int32_t, std::vector<std::int32_t>> undirected;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (selected.count(*source) != 0 && selected.count(*destination) != 0) {
      undirected[*source].push_back(*destination);
      undirected[*destination].push_back(*source);
    }
  }

  std::unordered_set<std::int32_t> visited;
  std::vector<std::vector<std::int32_t>> chains;
  for (const auto id : selectedNodeIds) {
    if (selected.count(id) == 0 || visited.count(id) != 0)
      continue;
    std::vector<std::int32_t> component;
    std::queue<std::int32_t> pending;
    pending.push(id);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      component.push_back(current);
      for (const auto neighbor : undirected[current])
        pending.push(neighbor);
    }

    const std::unordered_set<std::int32_t> members(component.begin(),
                                                   component.end());
    auto sources = members;
    auto sinks = members;
    for (const auto &link : links) {
      const auto source = findNodeForPin(link.sourcePinId);
      const auto destination = findNodeForPin(link.destinationPinId);
      if (!source.has_value() || !destination.has_value())
        continue;
      if (members.count(*source) != 0 && members.count(*destination) != 0) {
        sources.erase(*destination);
        sinks.erase(*source);
      }
    }
    if (sources.size() != 1 || sinks.size() != 1)
      return {};
    chains.push_back(std::move(component));
  }
  return chains;
}

std::vector<std::int32_t> NodeGraph::getArmedTrainableNodeIds() const {
  std::vector<std::int32_t> armed;
  for (const auto &node : nodes) {
    if (node.state != NodeState::liveBlue)
      continue;
    if (isControlSourceType(node.type) || !node.hasWeights)
      continue;
    if (node.armedForTraining)
      armed.push_back(node.id);
  }
  return armed;
}

bool NodeGraph::setArmedForTraining(std::int32_t nodeId, bool armed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights || isControlSourceType(node->type) ||
      node->state == NodeState::frozenGold)
    return false;
  node->armedForTraining = armed;
  return true;
}

bool NodeGraph::setWeightsPath(std::int32_t nodeId, const std::string &path) {
  auto *node = findNode(nodeId);
  if (node == nullptr || (!node->hasWeights && node->type != NodeType::blackBox))
    return false;
  node->weightsProvenance = WeightsProvenance::file;
  node->weightsPath = path;
  if (node->type == NodeType::blackBox)
    node->artifactPath = path;
  return true;
}

bool NodeGraph::clearWeightsToSeed(std::int32_t nodeId, std::int32_t seed) {
  auto *node = findNode(nodeId);
  if (node == nullptr || !node->hasWeights ||
      node->state == NodeState::frozenGold)
    return false;
  ensureCopySlotCount(*node, effectiveCopyCount(nodeId));
  // BatchNorm keeps a shared identity seed; every other weighted member derives
  // `seed + i` so materialized copies do not share live audition weights.
  const bool independentExtraSlots = node->type != NodeType::batchNorm;
  writeRandomizedWeightSlots(*node, seed, independentExtraSlots);
  node->blackBoxOrigin = BlackBoxOrigin::manualFreeze;
  return true;
}

std::optional<TrainJobRequest> NodeGraph::createTrainRequest() const {
  for (const auto &group : groups) {
    if (group.copies > 1) {
      auto expanded = withInvisibleCopiesMaterialized();
      return expanded.createTrainRequest();
    }
  }
  const auto armed = getArmedTrainableNodeIds();
  if (armed.empty())
    return std::nullopt;

  TrainJobRequest request;
  request.requestId = juce::Uuid().toString().toStdString();
  request.armedNodeIds = armed;

  auto root = std::make_unique<juce::DynamicObject>();
  root->setProperty("request_id", juce::String(request.requestId));
  root->setProperty("operation", "train_steerable");
  root->setProperty("command", "start");

  juce::Array<juce::var> armedIds;
  juce::Array<juce::var> elements;
  std::unordered_set<std::int32_t> selected(armed.begin(), armed.end());
  std::int32_t inputId = 0;
  for (const auto &node : nodes) {
    if (node.type == NodeType::audioInput)
      inputId = node.id;
  }
  if (inputId != 0) {
    std::queue<std::int32_t> pending;
    pending.push(inputId);
    std::unordered_set<std::int32_t> visited;
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      const auto *node = findNode(current);
      if (node != nullptr && !isControlSourceType(node->type) &&
          node->state == NodeState::liveBlue)
        selected.insert(current);
      for (const auto &link : links) {
        const auto source = findNodeForPin(link.sourcePinId);
        const auto destination = findNodeForPin(link.destinationPinId);
        if (source.has_value() && *source == current && destination.has_value())
          pending.push(*destination);
      }
    }
  }
  const std::unordered_set<std::int32_t> snapshot(selected.begin(),
                                                  selected.end());
  for (const auto nodeId : armed)
    armedIds.add(nodeId);
  for (const auto nodeId : snapshot) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      continue;
    auto element = std::make_unique<juce::DynamicObject>();
    element->setProperty("id", node->id);
    element->setProperty("type", nodeTypeName(node->type));
    element->setProperty("label", juce::String(node->label));
    element->setProperty("seed", node->seed);
    juce::Array<juce::var> properties;
    for (const auto &property : node->properties) {
      auto serialized = std::make_unique<juce::DynamicObject>();
      serialized->setProperty("key", juce::String(property.key));
      serialized->setProperty("value", property.value);
      if (property.kind == PropertyKind::real)
        serialized->setProperty("float_value", property.floatValue);
      properties.add(juce::var(serialized.release()));
    }
    element->setProperty("properties", properties);
    elements.add(juce::var(element.release()));
  }
  root->setProperty("armed_element_ids", armedIds);

  auto fragment = std::make_unique<juce::DynamicObject>();
  fragment->setProperty("elements", elements);
  juce::Array<juce::var> connections;
  juce::Array<juce::var> audioInputs;
  juce::Array<juce::var> audioOutputs;
  juce::Array<juce::var> conditioningInputs;
  for (const auto &link : links) {
    const auto sourceNode = findNodeForPin(link.sourcePinId);
    const auto destinationNode = findNodeForPin(link.destinationPinId);
    if (!sourceNode.has_value() || !destinationNode.has_value())
      continue;
    const auto sourceArmed = snapshot.count(*sourceNode) != 0;
    const auto destinationArmed = snapshot.count(*destinationNode) != 0;
    if (sourceArmed && destinationArmed) {
      auto connection = std::make_unique<juce::DynamicObject>();
      connection->setProperty("source_element_id", *sourceNode);
      connection->setProperty("destination_element_id", *destinationNode);
      connections.add(juce::var(connection.release()));
    } else if (!sourceArmed && destinationArmed)
      audioInputs.add(*destinationNode);
    else if (sourceArmed && !destinationArmed)
      audioOutputs.add(*sourceNode);
  }
  fragment->setProperty("connections", connections);
  auto io = std::make_unique<juce::DynamicObject>();
  io->setProperty("audio_inputs", audioInputs);
  io->setProperty("audio_outputs", audioOutputs);
  io->setProperty("conditioning_inputs", conditioningInputs);
  fragment->setProperty("io_boundary", juce::var(io.release()));
  root->setProperty("graph_fragment", juce::var(fragment.release()));
  request.graphFragment =
      juce::JSON::toString(juce::var(root.release()), true).toStdString();
  return request;
}

std::optional<std::int32_t>
NodeGraph::absorbArmedChain(const TrainJobResult &result) {
  if (result.artifactPath.empty())
    return std::nullopt;
  const auto armed = getArmedTrainableNodeIds();
  if (armed.empty())
    return std::nullopt;

  juce::ValueTree fragment{"GraphFragment"};
  const std::unordered_set<std::int32_t> selected(armed.begin(), armed.end());
  juce::Point<float> centroid;
  for (const auto nodeId : armed) {
    const auto *node = findNode(nodeId);
    if (node == nullptr)
      return std::nullopt;
    fragment.appendChild(nodeToTree(*node), nullptr);
    centroid += node->position;
  }
  centroid /= static_cast<float>(armed.size());
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    if (selected.count(*source) != 0 || selected.count(*destination) != 0)
      fragment.appendChild(linkToTree(link), nullptr);
  }

  struct Boundary {
    std::int32_t sourcePin = 0;
    std::int32_t destinationPin = 0;
    bool control = false;
  };
  std::vector<Boundary> incoming;
  std::vector<Boundary> outgoing;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    const auto *destinationPin = findPin(link.destinationPinId);
    const bool destControl =
        destinationPin != nullptr && isControlInputPin(*destinationPin);
    if (selected.count(*source) == 0 && selected.count(*destination) != 0)
      incoming.push_back({link.sourcePinId, 0, destControl});
    else if (selected.count(*source) != 0 && selected.count(*destination) == 0)
      outgoing.push_back({0, link.destinationPinId, false});
  }

  auto box = makeNode(NodeType::blackBox, centroid);
  box.label = result.hasEncodeDecode ? "Trained RAVE" : "Trained Steerable";
  box.detail = "Locked";
  box.state = NodeState::frozenGold;
  box.colour = colourForType(box.type, box.state);
  box.artifactPath = result.artifactPath;
  box.weightsPath = result.artifactPath;
  box.weightsProvenance = WeightsProvenance::file;
  box.blackBoxOrigin = BlackBoxOrigin::trainAutoload;
  box.hasWeights = true;
  box.compactnessReady = result.compactnessReady;
  if (result.compactnessReady)
    copyCompactnessFromArtifact(box, result.artifactPath);
  box.sourceSubgraph = fragment.toXmlString().toStdString();
  if (result.hasEncodeDecode) {
    box.outputs.push_back({nextPinId++, latentPinLabel, PinKind::output,
                           flexibleTensorShape(defaultLatentSize)});
    box.inputs.push_back({nextPinId++, latentPinLabel, PinKind::input,
                          flexibleTensorShape(defaultLatentSize)});
    box.fidelityPercent = defaultFidelityPercent;
  }
  const auto boxId = box.id;
  const auto audioIn = box.inputs.front().id;
  std::int32_t controlIn = 0;
  for (const auto &pin : box.inputs) {
    if (isControlInputPin(pin))
      controlIn = pin.id;
  }
  const auto audioOut = box.outputs.front().id;
  nodes.push_back(std::move(box));

  bool wiredAudioIn = false;
  bool wiredControl = false;
  for (const auto &edge : incoming) {
    if (edge.control && controlIn != 0 && !wiredControl) {
      links.push_back({nextLinkId++, edge.sourcePin, controlIn});
      wiredControl = true;
    } else if (!edge.control && !wiredAudioIn) {
      links.push_back({nextLinkId++, edge.sourcePin, audioIn});
      wiredAudioIn = true;
    }
  }
  bool wiredAudioOut = false;
  for (const auto &edge : outgoing) {
    if (!wiredAudioOut) {
      links.push_back({nextLinkId++, audioOut, edge.destinationPin});
      wiredAudioOut = true;
    }
  }

  for (auto nodeId : armed)
    removeNode(nodeId);
  return boxId;
}

bool NodeGraph::hasReconstructionTrainPath() const {
  return reconstructionGateMessage().empty();
}

std::string NodeGraph::reconstructionGateMessage() const {
  const GraphNode *bottleneck = nullptr;
  for (const auto &node : nodes) {
    if (node.type == NodeType::variationalBottleneck &&
        node.state == NodeState::liveBlue && node.armedForTraining) {
      bottleneck = &node;
      break;
    }
  }
  if (bottleneck == nullptr)
    return "Reconstruction requires an armed variational bottleneck";

  std::unordered_map<std::int32_t, std::vector<std::int32_t>> outgoing;
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> incoming;
  for (const auto &link : links) {
    const auto source = findNodeForPin(link.sourcePinId);
    const auto destination = findNodeForPin(link.destinationPinId);
    if (!source.has_value() || !destination.has_value())
      continue;
    outgoing[*source].push_back(*destination);
    incoming[*destination].push_back(*source);
  }

  const auto reaches = [&](std::int32_t from, NodeType targetType) {
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visited;
    pending.push(from);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      const auto *node = findNode(current);
      if (node != nullptr && node->type == targetType)
        return true;
      for (const auto next : outgoing[current])
        pending.push(next);
    }
    return false;
  };
  const auto reachedFrom = [&](NodeType sourceType, std::int32_t to) {
    std::queue<std::int32_t> pending;
    std::unordered_set<std::int32_t> visited;
    pending.push(to);
    while (!pending.empty()) {
      const auto current = pending.front();
      pending.pop();
      if (!visited.insert(current).second)
        continue;
      const auto *node = findNode(current);
      if (node != nullptr && node->type == sourceType)
        return true;
      for (const auto prev : incoming[current])
        pending.push(prev);
    }
    return false;
  };

  if (!reachedFrom(NodeType::audioInput, bottleneck->id))
    return "Reconstruction requires a path from Audio Input to the bottleneck";
  if (!reaches(bottleneck->id, NodeType::audioOutput))
    return "Reconstruction requires a decode path from the bottleneck to Audio Output";
  return {};
}
} // namespace openyourbox::graph
