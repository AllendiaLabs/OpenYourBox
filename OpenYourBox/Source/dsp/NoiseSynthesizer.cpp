#include "NoiseSynthesizer.h"

namespace openyourbox::dsp {
torch::Tensor NoiseSynthesizer::process(const torch::Tensor &conditioner,
                                        const torch::Tensor &weight) {
  if (!conditioner.defined() || conditioner.dim() != 3 || !weight.defined())
    return conditioner;
  auto bands = torch::sigmoid(torch::conv1d(conditioner, weight));
  auto noise = torch::randn_like(bands);
  auto mixed = (bands * noise).mean(1, true);
  if (conditioner.size(1) == mixed.size(1))
    return mixed;
  return mixed.expand({conditioner.size(0), conditioner.size(1),
                       conditioner.size(2)})
      .contiguous();
}
} // namespace openyourbox::dsp
