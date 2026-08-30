#pragma once

#include <torch/torch.h>

#include <utility>

namespace openyourbox::dsp {
/**
 * @class VariationalBottleneck
 * @brief Acids-rave v1 grouped variational head with optional compactness.
 *
 * Live, Gold, and eval paths historically returned the mean branch only
 * (`encodeMean`). Prior-mix on RAVE Gold uses `encodeDistribution` so scale
 * and sampling see the same softplus-std convention as training
 * (`σ = softplus(scale) + ε`). Compactness (when buffers are ready) keeps
 * the leading PCA dimensions implied by `fidelityPercent` and fills the
 * remainder with prior noise so the latent port width stays fixed.
 */
class VariationalBottleneck {
public:
  /** @brief Softplus floor matching acids-rave `std = softplus(scale) + 1e-4`. */
  static constexpr double softplusEpsilon = 1e-4;

  /**
   * @brief Converts a raw scale head to a strictly positive standard deviation.
   * @param scale Raw grouped-conv scale branch `[batch, latent, time]`.
   * @return `softplus(scale) + softplusEpsilon`.
   */
  [[nodiscard]] static torch::Tensor softplusStd(const torch::Tensor &scale);

  /**
   * @brief Encodes encoder features to `(μ, σ)` without sampling.
   *
   * @p features must already include causal left context of `kernel−1`
   * samples. Compactness, when ready, is applied to `μ` only (existing
   * `applyFidelity` semantics); `σ` is the encoder spread.
   *
   * @param features Encoder tensor `[batch, inChannels, time]`.
   * @param groupedWeight Grouped conv weight `[2 * latent, in/2, kernel]`.
   * @param fidelityPercent 0–100 compactness keep ratio.
   * @param compactnessReady True when PCA buffers are defined.
   * @param latentMean Optional `[latent]` mean.
   * @param latentPca Optional `[latent, latent]` basis (rows = components).
   * @param cumulativeVariance Optional `[latent]` cumulative singular-value ratio.
   * @return Pair `(μ, σ)` each `[batch, latent, time]`.
   */
  [[nodiscard]] static std::pair<torch::Tensor, torch::Tensor> encodeDistribution(
      const torch::Tensor &features, const torch::Tensor &groupedWeight,
      float fidelityPercent, bool compactnessReady,
      const torch::Tensor &latentMean, const torch::Tensor &latentPca,
      const torch::Tensor &cumulativeVariance);

  /**
   * @brief Encodes encoder features to the mean latent (no sampling).
   *
   * @p features must already include causal left context of `kernel−1`
   * samples (live streaming history or an explicit pad). The grouped conv
   * uses PyTorch layout `[2 * latent, in/2, kernel]` with `groups=2`.
   *
   * @param features Encoder tensor `[batch, inChannels, time]`.
   * @param groupedWeight Grouped conv weight `[2 * latent, in/2, kernel]`.
   * @param fidelityPercent 0–100 compactness keep ratio.
   * @param compactnessReady True when PCA buffers are defined.
   * @param latentMean Optional `[latent]` mean.
   * @param latentPca Optional `[latent, latent]` basis (rows = components).
   * @param cumulativeVariance Optional `[latent]` cumulative singular-value ratio.
   * @return Latent tensor `[batch, latent, time]`.
   */
  [[nodiscard]] static torch::Tensor encodeMean(
      const torch::Tensor &features, const torch::Tensor &groupedWeight,
      float fidelityPercent, bool compactnessReady,
      const torch::Tensor &latentMean, const torch::Tensor &latentPca,
      const torch::Tensor &cumulativeVariance);

  /**
   * @brief Applies compactness crop in the PCA basis then projects back.
   * @param latent `[batch, latent, time]`.
   * @param fidelityPercent 0–100 keep ratio.
   * @param latentMean `[latent]`.
   * @param latentPca `[latent, latent]`.
   * @param cumulativeVariance `[latent]`.
   * @return Same-width latent tensor.
   */
  [[nodiscard]] static torch::Tensor
  applyFidelity(const torch::Tensor &latent, float fidelityPercent,
                const torch::Tensor &latentMean, const torch::Tensor &latentPca,
                const torch::Tensor &cumulativeVariance);

  /**
   * @brief Rank kept by cumulative singular-value thresholding.
   * @param fidelityPercent 0–100 keep ratio.
   * @param cumulativeVariance `[latent]` ratios in `[0, 1]`.
   * @return Kept component count in `[1, latent]`.
   */
  [[nodiscard]] static int keptRank(float fidelityPercent,
                                    const torch::Tensor &cumulativeVariance);
};
} // namespace openyourbox::dsp
