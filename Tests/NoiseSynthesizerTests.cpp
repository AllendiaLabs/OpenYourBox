#include "dsp/NoiseSynthesizer.h"
#include "graph/GraphTypes.h"
#include "graph/NodeGraph.h"

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {
/**
 * @brief Reports a failed Noise Synth invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Looks up an integer property on @p node.
 * @param node Graph element.
 * @param key Canonical property key.
 * @return Pointer to the property, or nullptr.
 */
const openyourbox::graph::NodeProperty *
findProperty(const openyourbox::graph::GraphNode &node, const std::string &key) {
  for (const auto &property : node.properties) {
    if (property.key == key)
      return &property;
  }
  return nullptr;
}
} // namespace

/**
 * @brief Runs acids-rave IR×noise parity and Noise Synth graph coverage.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::dsp::NoiseSynthesizer;
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultNoiseBands;
  using openyourbox::graph::defaultNoiseWindowSize;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  auto amp = torch::linspace(0.1, 0.9, 160,
                             torch::TensorOptions().dtype(torch::kFloat32))
                 .reshape({1, 80, 2});
  auto spectrum = amp.permute({0, 2, 1}).reshape({1, 2, 16, 5});
  const auto ir = NoiseSynthesizer::ampToImpulseResponse(spectrum, 64);
  passed &= expect(ir.dim() == 4 && ir.size(0) == 1 && ir.size(1) == 2 &&
                       ir.size(2) == 16 && ir.size(3) == 64,
                   "IR shape is [batch, frames, data, window]");
  const std::array<float, 8> irHead{0.12012579f, -0.00733135f, 0.0f,
                                    -0.00021582f, 0.0f,        0.0f,
                                    0.0f,         0.0f};
  for (int index = 0; index < 8; ++index) {
    passed &= expect(std::abs(ir[0][0][0][index].item<float>() - irHead[static_cast<std::size_t>(index)]) <
                         1.0e-5f,
                     "amp_to_impulse_response matches acids-rave");
  }

  torch::manual_seed(0);
  auto noise = torch::rand_like(ir) * 2.0 - 1.0;
  const auto composed =
      NoiseSynthesizer::fftConvolve(noise, ir).permute({0, 2, 1, 3}).reshape(
          {1, 16, 128});
  torch::manual_seed(0);
  const auto processed = NoiseSynthesizer::process(amp, 5, 64);
  passed &= expect(processed.dim() == 3 && processed.size(0) == 1 &&
                       processed.size(1) == 16 && processed.size(2) == 128,
                   "filtered noise is [batch, data_size, frames * window]");
  passed &= expect(torch::allclose(processed, composed, 1e-5, 1e-5),
                   "process equals IR × injected-style uniform noise");

  const std::array<float, 8> outHead{
      -0.00089936f, 0.06449560f, -0.10280180f, -0.08236959f,
      -0.04098731f, 0.03521394f, -0.00418720f, 0.09547485f};
  for (int index = 0; index < 8; ++index) {
    passed &= expect(std::abs(processed[0][0][index].item<float>() -
                              outHead[static_cast<std::size_t>(index)]) < 1.0e-5f,
                     "fft_convolve matches acids-rave with the same noise");
  }

  const auto injected = NoiseSynthesizer::process(amp, 5, 64, noise);
  passed &= expect(torch::allclose(injected, composed, 1e-5, 1e-5),
                   "injected noise path is deterministic");

  NodeGraph graph;
  const auto noiseId = graph.addNode(NodeType::noiseSynthesizer, {180.0f, 0.0f});
  const auto *node = graph.findNode(noiseId);
  passed &= expect(node != nullptr && !node->hasWeights &&
                       !node->armedForTraining,
                   "Noise Synth has no learned weights");
  const auto *bands = node != nullptr ? findProperty(*node, "noise_bands") : nullptr;
  const auto *window =
      node != nullptr ? findProperty(*node, "window_size") : nullptr;
  passed &= expect(bands != nullptr && bands->value == defaultNoiseBands,
                   "default noise bands is 5");
  passed &= expect(window != nullptr && window->value == defaultNoiseWindowSize,
                   "default window size is 64");

  NodeGraph refuseGraph;
  const auto audioIn = refuseGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto refuseNoise =
      refuseGraph.addNode(NodeType::noiseSynthesizer, {180.0f, 0.0f});
  const auto refused = refuseGraph.connect(
      refuseGraph.findNode(audioIn)->outputs.front().id,
      refuseGraph.findNode(refuseNoise)->inputs.front().id);
  passed &= expect(!refused.accepted,
                   "audio-rate stereo is refused (bands and hop)");

  if (!passed)
    return 1;
  std::cout << "NoiseSynthesizerTests passed\n";
  return 0;
}
