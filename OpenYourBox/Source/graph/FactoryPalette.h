#pragma once

#include "GraphTypes.h"
#include "NodeGraph.h"

#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>

namespace openyourbox::graph {
/**
 * @struct PaletteItem
 * @brief One Factory catalog entry shared by the left palette and context menus.
 */
struct PaletteItem {
  /** @brief User-visible Factory row label. */
  const char *label = "";
  /** @brief Element type placed when this item is chosen. */
  NodeType type = NodeType::activation;
  /** @brief True when this item places a TorchScript Load Gold node. */
  bool externalLoad = false;
};

/**
 * @struct PaletteCategory
 * @brief Named group of Factory catalog items.
 */
struct PaletteCategory {
  /** @brief Category header shown in the Factory tree and menus. */
  const char *label = "";
  /** @brief First item in the category. */
  const PaletteItem *items = nullptr;
  /** @brief Number of items in the category. */
  std::size_t count = 0;
};

/** @brief DDSP / time-domain effect Factory items. */
inline constexpr PaletteItem effectsPaletteItems[] = {
    {"ExpDecayReverb", NodeType::expDecayReverb},
    {"FilteredNoiseReverb", NodeType::filteredNoiseReverb},
    {"FIRFilter", NodeType::firFilter},
    {"ModDelay", NodeType::modDelay},
    {"Reverb", NodeType::reverb},
};

/** @brief Sequence-model and external-load Factory items. */
inline constexpr PaletteItem neuralPaletteItems[] = {
    {"LSTM", NodeType::lstm},
    {"RNN", NodeType::rnn},
    {"TorchScript Load", NodeType::blackBox, true},
};

/** @brief Layer / DSP-block Factory items. */
inline constexpr PaletteItem layerPaletteItems[] = {
    {"Activation", NodeType::activation},
    {"BatchNorm1d", NodeType::batchNorm},
    {"Bottleneck", NodeType::variationalBottleneck},
    {"Conv1D", NodeType::convolution},
    {"ConvTranspose1d", NodeType::convTranspose},
    {"Linear", NodeType::linear},
    {"Math Expression", NodeType::mathExpression},
    {"Noise Synth", NodeType::noiseSynthesizer},
    {"PQMF Analysis", NodeType::pqmfAnalysis},
    {"PQMF Synthesis", NodeType::pqmfSynthesis},
    {"TCN", NodeType::tcn},
    {"Utility", NodeType::merge},
};

/** @brief Conditioning-source Factory items. */
inline constexpr PaletteItem sourcePaletteItems[] = {
    {"Knob Input", NodeType::knobInput},
    {"XY Trackpad", NodeType::xyTrackpad},
};

/** @brief Ordered Factory categories; left panel and context menus share this. */
inline constexpr PaletteCategory paletteCategories[] = {
    {"Effects", effectsPaletteItems, std::size(effectsPaletteItems)},
    {"Neural / Sequence", neuralPaletteItems, std::size(neuralPaletteItems)},
    {"Layers", layerPaletteItems, std::size(layerPaletteItems)},
    {"Sources", sourcePaletteItems, std::size(sourcePaletteItems)},
};

/**
 * @brief Invokes @p visitor for every Factory palette item.
 * @tparam Visitor Callable `(const PaletteItem &)`.
 * @param visitor Called once per catalog item in category order.
 */
template <typename Visitor> void forEachPaletteItem(Visitor &&visitor) {
  for (const auto &category : paletteCategories) {
    for (std::size_t index = 0; index < category.count; ++index)
      visitor(category.items[index]);
  }
}

/**
 * @brief Returns true when a palette type can be inserted onto an existing cable.
 * @param type Palette element type.
 * @return False for source-only Knob/XY elements and TorchScript Load.
 */
inline bool canInsertOnLink(NodeType type) noexcept {
  return !isConditioningSourceType(type) && type != NodeType::blackBox;
}

/**
 * @brief Places a Factory palette item onto the focused canvas or group.
 * @param graph Editable graph document.
 * @param item Palette payload.
 * @param position Destination canvas coordinates.
 * @param parentGroupId Destination group, or empty for the root.
 * @return Stable identifier of the new node.
 */
inline std::int32_t placePaletteItem(NodeGraph &graph, const PaletteItem &item,
                                     juce::Point<float> position,
                                     std::optional<std::int32_t> parentGroupId) {
  if (item.externalLoad)
    return graph.addExternalTorchScriptLoadNode(position, parentGroupId);
  return graph.addNode(item.type, position, parentGroupId);
}
} // namespace openyourbox::graph
