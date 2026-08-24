#include "dsp/TCNModel.h"

#include <torch/torch.h>

#include <iostream>

namespace {
/**
 * @brief Reports a failed model invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}
} // namespace

/**
 * @brief Runs deterministic TCN architecture and inference checks.
 * @return Zero when every invariant passes.
 */
int main() {
  using namespace openyourbox::dsp;
  TCNConfiguration configuration;
  configuration.depth = 4;
  configuration.kernelSize = 3;
  configuration.channels = 8;
  configuration.inputChannels = 2;
  configuration.outputChannels = 2;

  auto first = std::make_shared<TCNModel>(configuration);
  auto second = std::make_shared<TCNModel>(configuration);
  first->randomizeWeights(1234);
  second->randomizeWeights(1234);

  bool passed = true;
  passed &= expect(first->getReceptiveField() == 31,
                   "depth=4, kernel=3 must have a 31-sample receptive field");
  passed &= expect(first->getParameterCount() > 0,
                   "model must own trainable parameters");

  const auto firstParameters = first->parameters();
  const auto secondParameters = second->parameters();
  passed &=
      expect(firstParameters.size() == secondParameters.size(),
             "matching architectures must expose matching parameter lists");
  for (std::size_t index = 0; index < firstParameters.size(); ++index)
    passed &= expect(
        torch::equal(firstParameters[index], secondParameters[index]),
        "equal seed and architecture must produce byte-identical weights");

  torch::InferenceMode inferenceGuard;
  const auto silence = torch::zeros({1, 2, 128}, torch::kFloat32);
  const auto output = first->forward(silence);
  passed &= expect(output.sizes() == silence.sizes(),
                   "causal forward pass must preserve tensor shape");
  passed &= expect(output.abs().max().item<float>() == 0.0f,
                   "bias-free model must preserve digital silence");

  if (passed)
    std::cout << "OpenYourBox TCN model tests passed\n";
  return passed ? 0 : 1;
}
