#include "NoiseSynthesizer.h"

#include <algorithm>

namespace openyourbox::dsp {
namespace {
/**
 * @brief Returns a zero tensor matching @p amplitudes' batch and dtype.
 * @param amplitudes Conditioner used for options and batch size.
 * @param channels Output channel count.
 * @param frames Output time length.
 */
torch::Tensor zerosLikeRank3(const torch::Tensor &amplitudes, int64_t channels,
                             int64_t frames) {
  return torch::zeros({amplitudes.size(0), channels, frames},
                      amplitudes.options());
}
} // namespace

torch::Tensor
NoiseSynthesizer::ampToImpulseResponse(const torch::Tensor &amplitudes,
                                       int targetSize) {
  if (!amplitudes.defined() || amplitudes.size(-1) < 2 || targetSize < 1)
    return amplitudes;
  auto spectrum = torch::stack({amplitudes, torch::zeros_like(amplitudes)}, -1);
  auto ir = torch::fft::irfft(torch::view_as_complex(spectrum));
  const auto filterSize = ir.size(-1);
  ir = torch::roll(ir, {filterSize / 2}, {-1});
  auto window = torch::hann_window(filterSize, /*periodic=*/true, ir.options());
  ir = ir * window;
  const auto pad = static_cast<int64_t>(targetSize) - filterSize;
  ir = torch::nn::functional::pad(
      ir, torch::nn::functional::PadFuncOptions({0, pad}));
  ir = torch::roll(ir, {-filterSize / 2}, {-1});
  return ir;
}

torch::Tensor NoiseSynthesizer::fftConvolve(const torch::Tensor &signal,
                                            const torch::Tensor &kernel) {
  if (!signal.defined() || !kernel.defined() || signal.size(-1) < 1 ||
      kernel.size(-1) < 1)
    return signal;
  const auto length = signal.size(-1);
  auto paddedSignal = torch::nn::functional::pad(
      signal, torch::nn::functional::PadFuncOptions({0, length}));
  auto paddedKernel = torch::nn::functional::pad(
      kernel, torch::nn::functional::PadFuncOptions({length, 0}));
  auto output = torch::fft::irfft(torch::fft::rfft(paddedSignal) *
                                  torch::fft::rfft(paddedKernel));
  return output.narrow(-1, output.size(-1) / 2, output.size(-1) / 2);
}

torch::Tensor NoiseSynthesizer::process(const torch::Tensor &amplitudes,
                                        int noiseBands, int windowSize,
                                        const torch::Tensor &noise) {
  const auto bands = std::max(2, noiseBands);
  const auto hop = std::max(1, windowSize);
  if (!amplitudes.defined() || amplitudes.dim() != 3)
    return amplitudes;
  const auto batch = amplitudes.size(0);
  const auto channels = amplitudes.size(1);
  const auto frames = amplitudes.size(2);
  if (channels < bands || channels % bands != 0)
    return zerosLikeRank3(amplitudes, std::max<int64_t>(1, channels / bands),
                          frames * hop);
  const auto dataSize = channels / bands;
  auto amp = amplitudes.permute({0, 2, 1}).reshape({batch, frames, dataSize, bands});
  auto ir = ampToImpulseResponse(amp, hop);
  torch::Tensor driving = noise;
  if (!driving.defined())
    driving = torch::rand_like(ir) * 2.0 - 1.0;
  auto filtered = fftConvolve(driving, ir).permute({0, 2, 1, 3});
  return filtered.reshape({batch, dataSize, frames * hop});
}
} // namespace openyourbox::dsp
