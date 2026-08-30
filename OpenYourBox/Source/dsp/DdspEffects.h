#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace openyourbox::dsp {
/**
 * @class DdspEffects
 * @brief Magenta DDSP-style effect helpers for live and freeze/train parity.
 *
 * Algorithms follow `ddsp/effects.py` parameter roles (exponential-decay IR,
 * FFT FIR reverb, filtered-noise IR, LTV-FIR, modulated delay) implemented
 * with LibTorch. All tensors are CPU float. History buffers are allocated
 * off the audio thread and only written in place during process.
 */
class DdspEffects {
public:
  /**
   * @brief Magenta `core.exp_sigmoid` used to scale effect gains.
   * @param value Unconstrained control tensor or scalar tensor.
   * @return Positive scaled value with a small floor.
   */
  [[nodiscard]] static torch::Tensor expSigmoid(const torch::Tensor &value);

  /**
   * @brief Scalar Magenta `exp_sigmoid`.
   * @param value Unconstrained control.
   * @return Positive scaled value with a small floor.
   */
  [[nodiscard]] static float expSigmoid(float value) noexcept;

  /**
   * @brief Builds an exponential-decay white-noise impulse response.
   * @param gain Raw Magenta gain control (scaled by exp_sigmoid).
   * @param decay Raw Magenta decay control.
   * @param length Impulse-response length in samples (≥ 1).
   * @param seed Deterministic noise seed.
   * @return 1-D IR tensor of length @p length.
   */
  [[nodiscard]] static torch::Tensor
  expDecayImpulseResponse(float gain, float decay, int length,
                          std::int32_t seed);

  /**
   * @brief Zeros the first IR tap so the dry path is not double-counted.
   * @param ir Impulse response, 1-D or batched.
   * @return IR with the first sample set to zero.
   */
  [[nodiscard]] static torch::Tensor maskDryIr(const torch::Tensor &ir);

  /**
   * @brief Pads or crops @p source to @p length samples.
   * @param source Time-domain IR.
   * @param length Target length (≥ 1).
   * @return 1-D IR of @p length.
   */
  [[nodiscard]] static torch::Tensor matchIrLength(const torch::Tensor &source,
                                                   int length);

  /**
   * @brief Linear blend of internal and external impulse responses.
   * @param internal Internal/trainable IR.
   * @param external External IR (may be empty).
   * @param blend 0 = internal, 1 = external.
   * @return Blended 1-D IR matching the internal length.
   */
  [[nodiscard]] static torch::Tensor
  blendImpulseResponses(const torch::Tensor &internal,
                        const torch::Tensor &external, float blend);

  /**
   * @brief Causal FIR convolution with a shared mono IR on every channel.
   * @param audio Input `[1, channels, time]`.
   * @param ir 1-D impulse response.
   * @param history Causal prefix `[1, channels, ir_len-1]`, updated in place.
   * @return Wet audio `[1, channels, time]`.
   */
  [[nodiscard]] static torch::Tensor streamingFir(const torch::Tensor &audio,
                                                  const torch::Tensor &ir,
                                                  torch::Tensor &history);

  /**
   * @brief Synthesizes a filtered-noise IR from a magnitude grid.
   * @param magnitudes `[n_frames, n_filter_banks]` (pre-sigmoid).
   * @param windowSize Odd FIR window used per frame.
   * @param reverbLength Output IR length in samples.
   * @param seed Deterministic noise seed.
   * @return 1-D IR of length @p reverbLength.
   */
  [[nodiscard]] static torch::Tensor
  filteredNoiseImpulseResponse(const torch::Tensor &magnitudes, int windowSize,
                               int reverbLength, std::int32_t seed);

  /**
   * @brief Applies an LTV-FIR frequency filter from a magnitude grid.
   * @param audio Input `[1, channels, time]`.
   * @param magnitudes `[n_frames, n_filter_banks]` (pre-sigmoid).
   * @param windowSize Odd FIR window size.
   * @param history Causal prefix sized to `windowSize-1`, updated in place.
   * @return Filtered audio `[1, channels, time]`.
   */
  [[nodiscard]] static torch::Tensor
  frequencyFilter(const torch::Tensor &audio, const torch::Tensor &magnitudes,
                  int windowSize, torch::Tensor &history);

  /**
   * @brief Variable-length modulated delay with linear interpolation.
   * @param audio Input `[1, channels, time]`.
   * @param centerMs Center delay in milliseconds.
   * @param depthMs Modulation depth in milliseconds.
   * @param gain Raw Magenta wet gain (exp_sigmoid).
   * @param phase Normalized delay position in `[0, 1]` after sigmoid mapping.
   * @param sampleRate Host sample rate.
   * @param addDry When true, mix dry with wet.
   * @param delayLine Circular buffer `[1, channels, maxDelay+1]`.
   * @param writeIndex Circular write head, updated in place.
   * @return Processed audio `[1, channels, time]`.
   */
  [[nodiscard]] static torch::Tensor
  modDelay(const torch::Tensor &audio, float centerMs, float depthMs,
           float gain, float phase, double sampleRate, bool addDry,
           torch::Tensor &delayLine, int &writeIndex);

  /**
   * @brief Maximum delay-line length in samples for the given center/depth.
   * @param centerMs Center delay in milliseconds.
   * @param depthMs Modulation depth in milliseconds.
   * @param sampleRate Host sample rate.
   * @return Buffer length including one extra sample for interpolation.
   */
  [[nodiscard]] static int delayLineLength(float centerMs, float depthMs,
                                           double sampleRate) noexcept;
};
} // namespace openyourbox::dsp
