#include "VariationalBottleneck.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace openyourbox::dsp {
namespace {
/**
 * @brief Runs the acids-rave grouped causal head with padding already applied.
 * @param features `[batch, in, time]` including `kernel−1` left context.
 * @param groupedWeight `[2 * latent, in/2, kernel]`.
 * @return Concatenated `[μ, scale]` of width `2 * latent`.
 */
torch::Tensor groupedHead(const torch::Tensor &features,
                          const torch::Tensor &groupedWeight) {
  const std::array<std::int64_t, 1> stride{1};
  const std::array<std::int64_t, 1> padding{0};
  const std::array<std::int64_t, 1> dilation{1};
  return torch::conv1d(features, groupedWeight, std::optional<torch::Tensor>{},
                       stride, padding, dilation, 2);
}
} // namespace

int VariationalBottleneck::keptRank(float fidelityPercent,
                                    const torch::Tensor &cumulativeVariance) {
  if (!cumulativeVariance.defined() || cumulativeVariance.numel() < 1)
    return 1;
  const auto width = static_cast<int>(cumulativeVariance.numel());
  const auto target = std::clamp(fidelityPercent, 0.0f, 100.0f) / 100.0f;
  int keep = width;
  for (int index = 0; index < width; ++index) {
    if (cumulativeVariance[index].item<float>() >= target) {
      keep = index + 1;
      break;
    }
  }
  return std::clamp(keep, 1, width);
}

torch::Tensor VariationalBottleneck::applyFidelity(
    const torch::Tensor &latent, float fidelityPercent,
    const torch::Tensor &latentMean, const torch::Tensor &latentPca,
    const torch::Tensor &cumulativeVariance) {
  if (!latent.defined() || !latentMean.defined() || !latentPca.defined() ||
      !cumulativeVariance.defined())
    return latent;
  const auto width = latent.size(1);
  if (latentMean.numel() != width || latentPca.size(0) != width ||
      cumulativeVariance.numel() != width)
    return latent;
  const auto keep = keptRank(fidelityPercent, cumulativeVariance);
  auto centered = latent - latentMean.view({1, width, 1});
  auto frames = centered.permute({0, 2, 1}).reshape({-1, width});
  auto codes = torch::matmul(frames, latentPca.transpose(0, 1));
  if (keep < width)
    codes.narrow(1, keep, width - keep)
        .copy_(torch::randn_like(codes.narrow(1, keep, width - keep)));
  auto restored = torch::matmul(codes, latentPca)
                      .reshape({latent.size(0), latent.size(2), width})
                      .permute({0, 2, 1})
                      .contiguous();
  return restored + latentMean.view({1, width, 1});
}

torch::Tensor VariationalBottleneck::softplusStd(const torch::Tensor &scale) {
  if (!scale.defined())
    return {};
  return torch::softplus(scale) + softplusEpsilon;
}

std::pair<torch::Tensor, torch::Tensor> VariationalBottleneck::encodeDistribution(
    const torch::Tensor &features, const torch::Tensor &groupedWeight,
    float fidelityPercent, bool compactnessReady,
    const torch::Tensor &latentMean, const torch::Tensor &latentPca,
    const torch::Tensor &cumulativeVariance) {
  auto projected = groupedHead(features, groupedWeight);
  const auto latentWidth = projected.size(1) / 2;
  auto mean = projected.narrow(1, 0, latentWidth);
  auto std = softplusStd(projected.narrow(1, latentWidth, latentWidth));
  if (compactnessReady)
    mean = applyFidelity(mean, fidelityPercent, latentMean, latentPca,
                         cumulativeVariance);
  return {mean, std};
}

torch::Tensor VariationalBottleneck::encodeMean(
    const torch::Tensor &features, const torch::Tensor &groupedWeight,
    float fidelityPercent, bool compactnessReady,
    const torch::Tensor &latentMean, const torch::Tensor &latentPca,
    const torch::Tensor &cumulativeVariance) {
  return encodeDistribution(features, groupedWeight, fidelityPercent,
                            compactnessReady, latentMean, latentPca,
                            cumulativeVariance)
      .first;
}
} // namespace openyourbox::dsp
