#include "VariationalBottleneck.h"

#include <algorithm>
#include <cmath>

namespace openyourbox::dsp {
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
  const auto target = std::clamp(fidelityPercent, 0.0f, 100.0f) / 100.0f;
  int keep = width;
  for (int index = 0; index < width; ++index) {
    if (cumulativeVariance[index].item<float>() >= target) {
      keep = index + 1;
      break;
    }
  }
  keep = std::clamp(keep, 1, static_cast<int>(width));
  auto centered = latent - latentMean.view({1, width, 1});
  auto frames = centered.permute({0, 2, 1}).reshape({-1, width});
  auto codes = torch::matmul(frames, latentPca.transpose(0, 1));
  if (keep < width)
    codes.narrow(1, keep, width - keep).copy_(torch::randn_like(codes.narrow(1, keep, width - keep)));
  auto restored = torch::matmul(codes, latentPca)
                      .reshape({latent.size(0), latent.size(2), width})
                      .permute({0, 2, 1})
                      .contiguous();
  return restored + latentMean.view({1, width, 1});
}

torch::Tensor VariationalBottleneck::encode(
    const torch::Tensor &features, const torch::Tensor &meanWeight,
    const torch::Tensor &logVarWeight, float fidelityPercent,
    bool compactnessReady, const torch::Tensor &latentMean,
    const torch::Tensor &latentPca, const torch::Tensor &cumulativeVariance) {
  auto mean = torch::conv1d(features, meanWeight);
  auto logVar = torch::conv1d(features, logVarWeight).clamp(-8.0, 8.0);
  auto latent = mean + torch::exp(0.5 * logVar) * torch::randn_like(mean);
  if (compactnessReady)
    latent = applyFidelity(latent, fidelityPercent, latentMean, latentPca,
                           cumulativeVariance);
  return latent;
}
} // namespace openyourbox::dsp
