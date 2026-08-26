#pragma once

#include <torch/torch.h>

namespace openyourbox::dsp {
/**
 * @class VariationalBottleneck
 * @brief Encoder-head reparameterization with optional compactness fidelity.
 *
 * Live inference samples `z = μ + σ ⊙ ε`. Compactness (when buffers are
 * present) keeps the leading PCA dimensions implied by `fidelityPercent` and
 * fills the remainder with prior noise so the latent port width stays fixed.
 */
class VariationalBottleneck {
public:
  /**
   * @brief Reparameterizes encoder features into a latent trajectory.
   * @param features Encoder tensor `[batch, inChannels, time]`.
   * @param meanWeight 1×1 conv weight `[latent, in, 1]`.
   * @param logVarWeight 1×1 conv weight `[latent, in, 1]`.
   * @param fidelityPercent 0–100 compactness keep ratio.
   * @param compactnessReady True when PCA buffers are defined.
   * @param latentMean Optional `[latent]` mean.
   * @param latentPca Optional `[latent, latent]` basis (rows = components).
   * @param cumulativeVariance Optional `[latent]` cumulative explained ratio.
   * @return Latent tensor `[batch, latent, time]`.
   */
  [[nodiscard]] static torch::Tensor encode(
      const torch::Tensor &features, const torch::Tensor &meanWeight,
      const torch::Tensor &logVarWeight, float fidelityPercent,
      bool compactnessReady, const torch::Tensor &latentMean,
      const torch::Tensor &latentPca, const torch::Tensor &cumulativeVariance);

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
};
} // namespace openyourbox::dsp
