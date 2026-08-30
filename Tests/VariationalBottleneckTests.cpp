#include "dsp/VariationalBottleneck.h"

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <iostream>
#include <utility>

namespace {
/**
 * @brief Reports a failed bottleneck invariant.
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

int main() {
  bool passed = true;
  torch::manual_seed(0);

  const int inChannels = 8;
  const int latent = 4;
  const int kernel = 5;
  auto weight = torch::randn({2 * latent, inChannels / 2, kernel});
  auto features = torch::randn({1, inChannels, 32});
  auto padded = torch::nn::functional::pad(
      features, torch::nn::functional::PadFuncOptions({kernel - 1, 0}));

  const auto first = openyourbox::dsp::VariationalBottleneck::encodeMean(
      padded, weight, 99.0f, false, {}, {}, {});
  const auto second = openyourbox::dsp::VariationalBottleneck::encodeMean(
      padded, weight, 99.0f, false, {}, {}, {});
  passed &= expect(first.defined() && first.size(1) == latent,
                   "encodeMean must emit the latent width");
  passed &= expect(first.size(2) == features.size(2),
                   "encodeMean must keep the unpadded time length");
  passed &= expect(torch::equal(first, second),
                   "consecutive encodeMean calls must be bit-identical");

  auto projected = torch::conv1d(padded, weight, std::optional<torch::Tensor>{},
                                 std::array<std::int64_t, 1>{1},
                                 std::array<std::int64_t, 1>{0},
                                 std::array<std::int64_t, 1>{1}, 2);
  auto mean = projected.narrow(1, 0, latent);
  passed &= expect(torch::allclose(first, mean),
                   "encodeMean must match the grouped-conv μ half");

  auto distribution = openyourbox::dsp::VariationalBottleneck::encodeDistribution(
      padded, weight, 99.0f, false, {}, {}, {});
  passed &= expect(distribution.first.defined() && distribution.second.defined(),
                   "encodeDistribution must return μ and σ");
  passed &= expect(torch::equal(distribution.first, first),
                   "encodeDistribution μ must match encodeMean");
  passed &= expect(torch::all(distribution.second > 0).item<bool>(),
                   "encodeDistribution σ must be strictly positive");
  const auto restd = openyourbox::dsp::VariationalBottleneck::softplusStd(
      projected.narrow(1, latent, latent));
  passed &= expect(torch::allclose(distribution.second, restd),
                   "encodeDistribution σ must be softplus(scale)+eps");

  auto cumulative = torch::tensor({0.4f, 0.7f, 0.9f, 1.0f});
  const auto keepHigh =
      openyourbox::dsp::VariationalBottleneck::keptRank(90.0f, cumulative);
  const auto keepLow =
      openyourbox::dsp::VariationalBottleneck::keptRank(50.0f, cumulative);
  passed &= expect(keepHigh >= keepLow,
                   "higher fidelity must keep at least as many dimensions");
  passed &= expect(keepHigh == 3, "90% fidelity should keep three components");
  passed &= expect(keepLow == 2, "50% fidelity should keep two components");

  if (passed)
    std::cout << "OpenYourBox variational bottleneck tests passed\n";
  return passed ? 0 : 1;
}
