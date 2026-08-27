#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace openyourbox::graph {
/**
 * @brief Directed cable between two displayed sibling boxes.
 */
struct BoxFlowEdge {
  /** @brief Upstream sibling box identifier. */
  std::int32_t source = 0;
  /** @brief Downstream sibling box identifier. */
  std::int32_t destination = 0;
};

/**
 * @brief Orders sibling boxes by information-flow rank, then name, then id.
 *
 * Rank is the longest path from sources in the sibling-link DAG after
 * contracting each strongly connected component (feedback loop) to one node.
 * Members of a loop share a rank and sort by case-insensitive display name,
 * then id. Boxes with no sibling links sit with sources (rank 0).
 *
 * @param boxIds Candidate sibling identifiers; self-loops and unknown endpoints
 *               in @p edges are ignored.
 * @param names Display names keyed by box id; missing names sort as empty.
 * @param edges Directed sibling cables, including duplicates.
 * @return @p boxIds sorted by rank, case-insensitive name, then id.
 */
[[nodiscard]] std::vector<std::int32_t> orderBoxesByFlowRank(
    std::vector<std::int32_t> boxIds,
    const std::unordered_map<std::int32_t, juce::String> &names,
    const std::vector<BoxFlowEdge> &edges);
} // namespace openyourbox::graph
