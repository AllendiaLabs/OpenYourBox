#pragma once

#include "NodeGraph.h"

#include <string>

namespace openyourbox::graph {
/** @brief Published RAVE starting graphs. */
enum class RaveLayoutId { original, latestContinuous };

/**
 * @brief Inserts an original or latest-continuous RAVE graph.
 *
 * Existing processing nodes (except host Audio Input/Output) are left in
 * place; the layout is added to the right of Audio Input. Channel width is
 * 1 (mono) or 2 (stereo). Refused while a nested group canvas is focused.
 * Host vs layout width mismatches remain a shape error unless the user wires
 * adapters.
 *
 * @param graph Document to mutate.
 * @param layout Published lineage.
 * @param channelWidth 1 or 2.
 * @return Empty on success; otherwise a user-facing refusal.
 */
std::string insertRaveLayout(NodeGraph &graph, RaveLayoutId layout,
                             int channelWidth);

/**
 * @brief Returns the causal delay of a RAVE processing node in samples.
 * @param node Graph node.
 * @return Delay at host rate, or 0 when the type is not rate-reducing.
 */
[[nodiscard]] std::uint64_t raveNodeDelaySamples(const GraphNode &node);
} // namespace openyourbox::graph
