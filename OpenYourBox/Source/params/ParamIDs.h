#pragma once

#include <JuceHeader.h>

/**
 * @namespace openyourbox::params
 * @brief Stable host-facing parameter identifiers and value constraints.
 */
namespace openyourbox::params {
inline constexpr auto stateType = "OpenYourBoxState";
inline constexpr auto depth = "depth";
inline constexpr auto kernelSize = "kernel_size";
inline constexpr auto channels = "channels";
inline constexpr auto activation = "activation";
inline constexpr auto randomize = "randomize";
inline constexpr auto randomizeCC = "randomize_cc";
inline constexpr auto globalSeed = "global_seed";
inline constexpr auto dryWet = "dry_wet";

inline constexpr int defaultDepth = 4;
inline constexpr int defaultKernelSize = 3;
inline constexpr int defaultChannels = 16;
inline constexpr int defaultSeed = 42;
inline constexpr int defaultRandomizeCC = 64;

/** @brief Persisted graph document schema version. */
inline constexpr int graphStateVersion = 1;

/**
 * @brief Creates a versioned parameter identifier for stable AU ordering.
 * @param identifier Stable string identifier exposed to hosts.
 * @return A JUCE parameter identifier with version hint 1.
 */
inline juce::ParameterID makeID(const char *identifier) {
  return {identifier, 1};
}
} // namespace openyourbox::params
