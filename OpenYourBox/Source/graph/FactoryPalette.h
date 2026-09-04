#pragma once

#include "GraphTypes.h"

#include <cstddef>
#include <iterator>

namespace openyourbox::graph {
/**
 * @struct PaletteItem
 * @brief One Factory catalog entry shared by the left Library and context menus.
 */
struct PaletteItem {
  /** @brief User-visible element name. */
  const char *label = nullptr;
  /** @brief Graph node type to place. */
  NodeType type = NodeType::linear;
  /** @brief True when this item places a TorchScript Load Gold node. */
  bool externalLoad = false;
};

/**
 * @struct PaletteCategory
 * @brief Named group of Factory palette items.
 */
struct PaletteCategory {
  /** @brief User-visible category heading. */
  const char *label = nullptr;
  /** @brief Contiguous item array for this category. */
  const PaletteItem *items = nullptr;
  /** @brief Number of items in @ref items. */
  std::size_t count = 0;
};

/** @brief Factory Effects category. */
inline constexpr PaletteItem effectsPaletteItems[] = {
    {"ExpDecayReverb", NodeType::expDecayReverb},
    {"FilteredNoiseReverb", NodeType::filteredNoiseReverb},
    {"FIRFilter", NodeType::firFilter},
    {"ModDelay", NodeType::modDelay},
    {"Reverb", NodeType::reverb},
};

/** @brief Factory Neural / Sequence category. */
inline constexpr PaletteItem neuralPaletteItems[] = {
    {"LSTM", NodeType::lstm},
    {"RNN", NodeType::rnn},
    {"TorchScript Load", NodeType::blackBox, true},
};

/** @brief Factory Layers category. */
inline constexpr PaletteItem layerPaletteItems[] = {
    {"Activation", NodeType::activation},
    {"BatchNorm1d", NodeType::batchNorm},
    {"Bottleneck", NodeType::variationalBottleneck},
    {"Conv1D", NodeType::convolution},
    {"ConvTranspose1d", NodeType::convTranspose},
    {"Math Expression", NodeType::mathExpression},
    {"Noise Synth", NodeType::noiseSynthesizer},
    {"PQMF Analysis", NodeType::pqmfAnalysis},
    {"PQMF Synthesis", NodeType::pqmfSynthesis},
    {"Utility", NodeType::merge},
};

/** @brief Factory Sources category. */
inline constexpr PaletteItem sourcePaletteItems[] = {
    {"Knob Input", NodeType::knobInput},
    {"XY Trackpad", NodeType::xyTrackpad},
};

/** @brief Factory Training category (live-inaudible). */
inline constexpr PaletteItem trainingPaletteItems[] = {
    {"Data Loader", NodeType::dataLoader},
    {"Loss", NodeType::loss},
};

/** @brief Ordered Factory catalog categories (single source of truth). */
inline constexpr PaletteCategory paletteCategories[] = {
    {"Effects", effectsPaletteItems, std::size(effectsPaletteItems)},
    {"Neural / Sequence", neuralPaletteItems, std::size(neuralPaletteItems)},
    {"Layers", layerPaletteItems, std::size(layerPaletteItems)},
    {"Sources", sourcePaletteItems, std::size(sourcePaletteItems)},
    {"Training", trainingPaletteItems, std::size(trainingPaletteItems)},
};

/**
 * @brief Invokes @p visitor for every Factory palette item.
 * @tparam Visitor Callable `(const PaletteItem &)`.
 * @param visitor Visitor invoked once per catalog item.
 */
template <typename Visitor> void forEachPaletteItem(Visitor &&visitor) {
  for (const auto &category : paletteCategories) {
    for (std::size_t index = 0; index < category.count; ++index)
      visitor(category.items[index]);
  }
}
} // namespace openyourbox::graph
