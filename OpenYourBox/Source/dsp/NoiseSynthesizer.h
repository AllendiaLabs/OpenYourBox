#pragma once

#include <torch/torch.h>

namespace openyourbox::dsp {
/**
 * @class NoiseSynthesizer
 * @brief Learned filtered-noise addend (DDSP-style amplitude → IR).
 */
class NoiseSynthesizer {
public:
  /**
   * @brief Filters unit noise by a 1×1 projection of the input amplitude.
   * @param conditioner Driving tensor `[batch, inChannels, time]`.
   * @param weight Bias-free conv `[noiseBands, inChannels, 1]`.
   * @return Noise addend `[batch, inChannels, time]` (bands mixed back).
   */
  [[nodiscard]] static torch::Tensor
  process(const torch::Tensor &conditioner, const torch::Tensor &weight);
};
} // namespace openyourbox::dsp
