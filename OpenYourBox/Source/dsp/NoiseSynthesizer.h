#pragma once

#include <torch/torch.h>

namespace openyourbox::dsp {
/**
 * @class NoiseSynthesizer
 * @brief Acids-rave v1 IR-filtered white noise (no amplitude conv stack).
 *
 * Input amplitudes are `[batch, dataSize * noiseBands, frames]`. The last
 * `noiseBands` bins of each channel group are treated as a real zero-phase
 * spectrum, converted to a hop-length impulse response, and FFT-convolved
 * with uniform white noise in `[-1, 1)`, matching `rave.core.amp_to_impulse_response`
 * and `rave.core.fft_convolve`. Output is `[batch, dataSize, frames * windowSize]`.
 *
 * The acids-rave `NoiseGenerator` conv/LeakyReLU stack and `mod_sigmoid(x - 5)`
 * stay outside this element so users can wire them from existing graph nodes.
 */
class NoiseSynthesizer {
public:
  /**
   * @brief Converts a real amplitude spectrum on the last dim into a windowed IR.
   * @param amplitudes Real bins `[..., noiseBands]` (imaginary part is zero).
   * @param targetSize Hop / IR length (`prod(ratios)` in acids-rave v1).
   * @return Impulse responses `[..., targetSize]`.
   */
  [[nodiscard]] static torch::Tensor
  ampToImpulseResponse(const torch::Tensor &amplitudes, int targetSize);

  /**
   * @brief Linear convolution on the last dimension via rFFT multiply.
   * @param signal Driving signal `[..., hop]`.
   * @param kernel Impulse response `[..., hop]`.
   * @return Convolved signal `[..., hop]` (second half of the full linear conv).
   */
  [[nodiscard]] static torch::Tensor fftConvolve(const torch::Tensor &signal,
                                                 const torch::Tensor &kernel);

  /**
   * @brief Filters unit-uniform noise by IRs derived from @p amplitudes.
   * @param amplitudes Conditioner `[batch, dataSize * noiseBands, frames]`.
   * @param noiseBands IR frequency bins per output channel (≥ 2).
   * @param windowSize IR / hop length in samples (≥ IR length `2*(bands-1)`).
   * @param noise Optional pre-rolled noise `[batch, frames, dataSize, windowSize]`.
   *             When undefined, draws `rand * 2 - 1` like acids-rave.
   * @return Filtered noise `[batch, dataSize, frames * windowSize]`.
   */
  [[nodiscard]] static torch::Tensor process(const torch::Tensor &amplitudes,
                                             int noiseBands, int windowSize,
                                             const torch::Tensor &noise = {});
};
} // namespace openyourbox::dsp
