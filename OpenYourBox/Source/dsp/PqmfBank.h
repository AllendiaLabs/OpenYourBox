#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace openyourbox::dsp {
/**
 * @class PqmfBank
 * @brief Causal Kaiser cosine-modulated PQMF matching acids-rave coefficients.
 *
 * Analysis maps host-rate audio `[1, C, T]` to multiband `[1, C * nBand, T']`
 * with leftover hop buffering. Synthesis inverts that mapping. Filter design
 * runs off the audio thread; `process*` reuse preallocated leftovers.
 */
class PqmfBank {
public:
  /**
   * @brief Designs analysis and synthesis banks.
   * @param nBand Number of sub-bands (power of two recommended).
   * @param attenuationDb Stopband attenuation used by the Kaiser prototype.
   */
  explicit PqmfBank(int nBand = 16, float attenuationDb = 100.0f);

  /** @brief Returns the configured band count. */
  [[nodiscard]] int getBandCount() const noexcept;

  /** @brief Returns the prototype/filter length in taps. */
  [[nodiscard]] int getKernelSize() const noexcept;

  /** @brief Causal delay of analysis or synthesis in host-rate samples. */
  [[nodiscard]] std::uint64_t getCausalDelaySamples() const noexcept;

  /**
   * @brief Runs analysis on a contiguous (non-streaming) tensor.
   * @param audio Input shaped `[batch, audioChannels, time]`.
   * @return Multiband tensor shaped `[batch, audioChannels * nBand, time/nBand]`.
   */
  [[nodiscard]] torch::Tensor analyse(const torch::Tensor &audio) const;

  /**
   * @brief Runs synthesis on a contiguous (non-streaming) tensor.
   * @param bands Multiband tensor from `analyse`.
   * @param audioChannels Host audio width used when stacking bands.
   * @return Audio tensor at host rate.
   */
  [[nodiscard]] torch::Tensor synthesise(const torch::Tensor &bands,
                                         int audioChannels) const;

  /**
   * @brief Streaming analysis using a leftover hop ring.
   * @param audio Current block `[1, C, T]`.
   * @param leftover Mutable incomplete hop, prepared off the audio thread.
   * @return Emitted multiband frames (time may be 0 when the hop is incomplete).
   */
  [[nodiscard]] torch::Tensor
  analyseStreaming(const torch::Tensor &audio, torch::Tensor &leftover) const;

  /**
   * @brief Streaming synthesis using a leftover upsampled-band ring.
   * @param bands Current multiband block.
   * @param leftover Mutable incomplete upsampled hop (`[1, C * nBand, kernel-1]`).
   * @param audioChannels Host audio width.
   * @return Emitted host-rate audio (`[1, C, bandFrames * nBand]`).
   */
  [[nodiscard]] torch::Tensor synthesiseStreaming(const torch::Tensor &bands,
                                                  torch::Tensor &leftover,
                                                  int audioChannels) const;

  /**
   * @brief Allocates a zero leftover ring for streaming.
   * @param channels Channel count stored in the leftover (audio or bands).
   * @return Zero tensor shaped `[1, channels, kernel-1]`.
   */
  [[nodiscard]] torch::Tensor makeLeftover(int channels) const;

private:
  /** @brief Configured band count. */
  int nBand = 16;
  /** @brief Odd Kaiser prototype length. */
  int kernelSize = 1;
  /** @brief Analysis filters `[nBand, 1, K]`. */
  torch::Tensor analysisKernels;
  /** @brief Synthesis filters `[1, nBand, K]` (time-reversed analysis). */
  torch::Tensor synthesisKernels;
};
} // namespace openyourbox::dsp
