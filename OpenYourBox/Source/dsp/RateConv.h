#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace openyourbox::dsp {
/** @brief Downsample (strided conv) or upsample (zero-insert + conv). */
enum class RateConvMode { downsample, upsample };

/**
 * @class RateConv
 * @brief Causal integer-stride convolution with preallocated leftover rings.
 */
class RateConv {
public:
  /**
   * @brief Stores convolution hyperparameters.
   * @param stride Integer hop ≥ 1.
   * @param kernelSize Temporal kernel ≥ 1.
   * @param dilation Temporal dilation ≥ 1.
   * @param mode Downsample or upsample.
   */
  RateConv(int stride, int kernelSize, int dilation, RateConvMode mode);

  /** @brief Returns the configured stride. */
  [[nodiscard]] int getStride() const noexcept;

  /** @brief Returns the configured kernel size. */
  [[nodiscard]] int getKernelSize() const noexcept;

  /** @brief Causal delay at the input rate in samples. */
  [[nodiscard]] std::uint64_t getCausalDelaySamples() const noexcept;

  /**
   * @brief Applies the layer to a contiguous tensor (training/tests).
   * @param input `[batch, inChannels, time]`.
   * @param weight Bias-free conv weight `[out, in, K]`.
   * @return Output tensor at the destination rate.
   */
  [[nodiscard]] torch::Tensor process(const torch::Tensor &input,
                                      const torch::Tensor &weight) const;

  /**
   * @brief Streaming process with leftover hop buffering.
   * @param input Current block.
   * @param weight Convolution kernel.
   * @param leftover Incomplete hop prepared off the audio thread.
   * @return Emitted frames (time may be 0).
   */
  [[nodiscard]] torch::Tensor processStreaming(const torch::Tensor &input,
                                               const torch::Tensor &weight,
                                               torch::Tensor &leftover) const;

private:
  /** @brief Integer stride. */
  int stride = 1;
  /** @brief Kernel size. */
  int kernelSize = 3;
  /** @brief Dilation. */
  int dilation = 1;
  /** @brief Downsample or upsample. */
  RateConvMode mode = RateConvMode::downsample;
};

/**
 * @class ConvTranspose1d
 * @brief Causal transposed convolution for integer-rate upsampling.
 */
class ConvTranspose1d {
public:
  /**
   * @brief Stores transposed-convolution hyperparameters.
   * @param stride Integer upsampling factor ≥ 1.
   * @param kernelSize Temporal kernel ≥ 1 (RAVE default: `2 * stride`).
   * @param dilation Temporal dilation ≥ 1.
   */
  ConvTranspose1d(int stride, int kernelSize, int dilation);

  /** @brief Returns the configured stride. */
  [[nodiscard]] int getStride() const noexcept;

  /** @brief Returns the configured kernel size. */
  [[nodiscard]] int getKernelSize() const noexcept;

  /** @brief Causal delay at the input rate in samples. */
  [[nodiscard]] std::uint64_t getCausalDelaySamples() const noexcept;

  /**
   * @brief Applies the layer to a contiguous tensor (training/tests).
   * @param input `[batch, inChannels, time]`.
   * @param weight Stored as `[outChannels, inChannels, kernel]`; transposed for
   *        `conv_transpose1d`.
   * @return Upsampled tensor `[batch, outChannels, time * stride]`.
   */
  [[nodiscard]] torch::Tensor process(const torch::Tensor &input,
                                      const torch::Tensor &weight) const;

  /**
   * @brief Streaming upsample with leftover hop buffering.
   * @param input Current block.
   * @param weight Convolution kernel stored as `[out, in, kernel]`.
   * @param leftover Incomplete hop prepared off the audio thread.
   * @return Emitted frames (time may be 0).
   */
  [[nodiscard]] torch::Tensor processStreaming(const torch::Tensor &input,
                                               const torch::Tensor &weight,
                                               torch::Tensor &leftover) const;

private:
  /** @brief Integer upsampling stride. */
  int stride = 1;
  /** @brief Kernel size. */
  int kernelSize = 3;
  /** @brief Dilation. */
  int dilation = 1;
};
} // namespace openyourbox::dsp
