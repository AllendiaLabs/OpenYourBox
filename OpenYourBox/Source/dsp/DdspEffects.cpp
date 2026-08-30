#include "DdspEffects.h"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
/** @brief Advances a SplitMix64 generator. */
std::uint64_t splitMix64(std::uint64_t &state) noexcept {
  auto value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

/** @brief Signed unit uniform sample matching live weight randomization. */
float uniformSigned(std::uint64_t &state) noexcept {
  constexpr auto inverse = 1.0 / static_cast<double>(std::uint64_t{1} << 53U);
  const auto unit = static_cast<double>(splitMix64(state) >> 11U) * inverse;
  return static_cast<float>(unit * 2.0 - 1.0);
}

/** @brief Magenta exp_sigmoid constants. */
constexpr float expSigmoidMax = 2.0f;
constexpr float expSigmoidExponent = 10.0f;
constexpr float expSigmoidFloor = 1.0e-7f;

/** @brief Flattens an IR tensor to 1-D CPU float. */
torch::Tensor flattenIr(const torch::Tensor &ir) {
  if (!ir.defined() || ir.numel() < 1)
    return {};
  auto flat = ir.contiguous().to(torch::kFloat32).reshape({ir.numel()});
  return flat;
}

/** @brief Causal FIR of @p audio with a mono kernel, updating @p history. */
torch::Tensor convolveMonoIr(const torch::Tensor &audio, const torch::Tensor &ir,
                             torch::Tensor &history) {
  auto kernel = flattenIr(ir);
  if (!audio.defined() || audio.dim() != 3 || !kernel.defined())
    return audio;
  const auto channels = audio.size(1);
  const auto samples = audio.size(2);
  const auto irLen = std::max<std::int64_t>(1, kernel.size(0));
  const auto historyLen = irLen - 1;
  torch::Tensor extended = audio;
  if (historyLen > 0) {
    if (!history.defined() || history.dim() != 3 ||
        history.size(1) != channels || history.size(2) != historyLen) {
      history = torch::zeros({1, channels, historyLen}, audio.options());
    }
    extended = torch::cat({history, audio}, 2);
    history.copy_(extended.narrow(2, extended.size(2) - historyLen, historyLen));
  }
  auto weight = kernel.flip(0).view({1, 1, irLen}).repeat({channels, 1, 1});
  auto wet = torch::nn::functional::conv1d(
      extended, weight,
      torch::nn::functional::Conv1dFuncOptions().groups(channels));
  if (wet.size(2) > samples)
    wet = wet.narrow(2, wet.size(2) - samples, samples);
  return wet;
}
} // namespace

namespace openyourbox::dsp {
torch::Tensor DdspEffects::expSigmoid(const torch::Tensor &value) {
  const auto exponent = std::log(static_cast<double>(expSigmoidExponent));
  return expSigmoidMax * torch::sigmoid(value).pow(exponent) + expSigmoidFloor;
}

float DdspEffects::expSigmoid(float value) noexcept {
  const auto sigmoid = 1.0 / (1.0 + std::exp(static_cast<double>(-value)));
  const auto exponent = std::log(static_cast<double>(expSigmoidExponent));
  return static_cast<float>(expSigmoidMax * std::pow(sigmoid, exponent) +
                            expSigmoidFloor);
}

torch::Tensor DdspEffects::expDecayImpulseResponse(float gain, float decay,
                                                   int length,
                                                   std::int32_t seed) {
  const auto irLen = std::max(1, length);
  auto ir = torch::empty({irLen}, torch::kFloat32);
  auto *data = ir.data_ptr<float>();
  auto state =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) ^
      0xa0761d6478bd642fULL;
  const auto scaledGain = expSigmoid(gain);
  const auto decayExponent = 2.0 + std::exp(static_cast<double>(decay));
  const auto denom = static_cast<double>(std::max(irLen - 1, 1));
  for (int index = 0; index < irLen; ++index) {
    const auto time = irLen == 1 ? 0.0 : static_cast<double>(index) / denom;
    data[index] = scaledGain *
                  static_cast<float>(std::exp(-decayExponent * time)) *
                  uniformSigned(state);
  }
  return ir;
}

torch::Tensor DdspEffects::maskDryIr(const torch::Tensor &ir) {
  auto masked = flattenIr(ir);
  if (!masked.defined() || masked.numel() < 1)
    return masked;
  masked = masked.clone();
  masked[0] = 0.0f;
  return masked;
}

torch::Tensor DdspEffects::matchIrLength(const torch::Tensor &source,
                                         int length) {
  const auto irLen = std::max(1, length);
  auto flat = flattenIr(source);
  if (!flat.defined())
    return torch::zeros({irLen}, torch::kFloat32);
  if (flat.size(0) == irLen)
    return flat;
  if (flat.size(0) > irLen)
    return flat.narrow(0, 0, irLen);
  return torch::nn::functional::pad(
      flat, torch::nn::functional::PadFuncOptions({0, irLen - flat.size(0)}));
}

torch::Tensor DdspEffects::blendImpulseResponses(const torch::Tensor &internal,
                                                 const torch::Tensor &external,
                                                 float blend) {
  auto left = flattenIr(internal);
  if (!left.defined())
    return {};
  const auto amount = std::clamp(blend, 0.0f, 1.0f);
  auto right = flattenIr(external);
  if (!right.defined() || right.numel() < 1)
    return left;
  right = matchIrLength(right, static_cast<int>(left.size(0)));
  return left * (1.0f - amount) + right * amount;
}

torch::Tensor DdspEffects::streamingFir(const torch::Tensor &audio,
                                        const torch::Tensor &ir,
                                        torch::Tensor &history) {
  return convolveMonoIr(audio, ir, history);
}

torch::Tensor DdspEffects::filteredNoiseImpulseResponse(
    const torch::Tensor &magnitudes, int windowSize, int reverbLength,
    std::int32_t seed) {
  const auto irLen = std::max(1, reverbLength);
  const auto frames = magnitudes.defined() && magnitudes.dim() >= 2
                          ? std::max<std::int64_t>(1, magnitudes.size(0))
                          : 1;
  const auto banks = magnitudes.defined() && magnitudes.dim() >= 2
                         ? std::max<std::int64_t>(1, magnitudes.size(1))
                         : 1;
  auto grid = magnitudes.defined()
                  ? magnitudes.to(torch::kFloat32).reshape({frames, banks})
                  : torch::zeros({frames, banks}, torch::kFloat32);
  grid = expSigmoid(grid);
  const auto oddWindow = std::max(1, windowSize | 1);
  const auto nFft = oddWindow;
  const auto nBins = nFft / 2 + 1;
  auto spec = torch::zeros({frames, nBins}, torch::kFloat32);
  const auto copyBins = std::min<std::int64_t>(banks, nBins);
  spec.narrow(1, 0, copyBins).copy_(grid.narrow(1, 0, copyBins));
  auto irFrames = torch::fft::irfft(spec.to(torch::kComplexFloat), nFft, -1);
  auto window = torch::hann_window(nFft, /*periodic=*/false);
  irFrames = irFrames * window;
  auto noise = torch::empty({irLen}, torch::kFloat32);
  auto state =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) ^
      0x9e3779b97f4a7c15ULL;
  auto *noisePtr = noise.data_ptr<float>();
  for (int index = 0; index < irLen; ++index)
    noisePtr[index] = uniformSigned(state);
  auto envelope = torch::zeros({irLen}, torch::kFloat32);
  const auto hop = std::max<std::int64_t>(1, irLen / frames);
  for (int frame = 0; frame < static_cast<int>(frames); ++frame) {
    const auto start = static_cast<int>(std::min<std::int64_t>(
        irLen - 1, static_cast<std::int64_t>(frame) * hop));
    const auto end = frame + 1 == static_cast<int>(frames)
                         ? irLen
                         : static_cast<int>(std::min<std::int64_t>(
                               irLen, (frame + 1) * hop));
    auto frameIr = irFrames[frame];
    const auto scale = frameIr.abs().mean().item<float>();
    envelope.narrow(0, start, std::max(1, end - start)).fill_(scale);
  }
  auto ir = noise * envelope;
  const auto peak = ir.abs().max().item<float>();
  if (peak > 1.0e-6f)
    ir = ir * (0.25f / peak);
  return ir;
}

torch::Tensor DdspEffects::frequencyFilter(const torch::Tensor &audio,
                                           const torch::Tensor &magnitudes,
                                           int windowSize,
                                           torch::Tensor &history) {
  const auto oddWindow = std::max(1, windowSize | 1);
  const auto frames = magnitudes.defined() && magnitudes.dim() >= 2
                          ? std::max<std::int64_t>(1, magnitudes.size(0))
                          : 1;
  const auto banks = magnitudes.defined() && magnitudes.dim() >= 2
                         ? std::max<std::int64_t>(1, magnitudes.size(1))
                         : 1;
  auto grid = magnitudes.defined()
                  ? magnitudes.to(torch::kFloat32).reshape({frames, banks})
                  : torch::ones({1, banks}, torch::kFloat32);
  grid = expSigmoid(grid);
  const auto nBins = oddWindow / 2 + 1;
  auto meanMag = grid.mean(0);
  auto spec = torch::zeros({nBins}, torch::kFloat32);
  const auto copyBins = std::min<std::int64_t>(banks, nBins);
  spec.narrow(0, 0, copyBins).copy_(meanMag.narrow(0, 0, copyBins));
  auto ir = torch::fft::irfft(spec.to(torch::kComplexFloat), oddWindow, -1);
  ir = ir * torch::hann_window(oddWindow, /*periodic=*/false);
  ir = torch::roll(ir, oddWindow / 2, 0);
  return convolveMonoIr(audio, ir, history);
}

int DdspEffects::delayLineLength(float centerMs, float depthMs,
                                 double sampleRate) noexcept {
  const auto maxMs = std::max(0.0f, centerMs) + std::max(0.0f, depthMs);
  const auto samples =
      static_cast<int>(std::ceil(std::max(1.0, sampleRate) * maxMs / 1000.0));
  return std::max(2, samples + 1);
}

torch::Tensor DdspEffects::modDelay(const torch::Tensor &audio, float centerMs,
                                    float depthMs, float gain, float phase,
                                    double sampleRate, bool addDry,
                                    torch::Tensor &delayLine, int &writeIndex) {
  if (!audio.defined() || audio.dim() != 3)
    return audio;
  const auto channels = audio.size(1);
  const auto samples = audio.size(2);
  const auto maxLen = delayLineLength(centerMs, depthMs, sampleRate);
  if (!delayLine.defined() || delayLine.dim() != 3 ||
      delayLine.size(1) != channels || delayLine.size(2) != maxLen) {
    delayLine = torch::zeros({1, channels, maxLen}, audio.options());
    writeIndex = 0;
  }
  const auto wetGain = expSigmoid(gain);
  const auto maxDelayMs = std::max(0.0f, centerMs) + std::max(0.0f, depthMs);
  const auto maxDelaySamples =
      std::max(1.0, sampleRate * static_cast<double>(maxDelayMs) / 1000.0);
  const auto depthPhase =
      maxDelayMs > 0.0f ? std::max(0.0f, depthMs) / maxDelayMs : 0.0f;
  const auto centerPhase =
      maxDelayMs > 0.0f ? std::max(0.0f, centerMs) / maxDelayMs : 0.0f;
  const auto mappedPhase =
      std::clamp(phase, 0.0f, 1.0f) * depthPhase + centerPhase;
  const auto delaySamples = std::clamp(
      mappedPhase * static_cast<float>(maxDelaySamples), 0.0f,
      static_cast<float>(maxLen - 2));
  auto output = torch::empty_like(audio);
  const auto *src = audio.contiguous().data_ptr<float>();
  auto *dst = output.data_ptr<float>();
  auto *line = delayLine.data_ptr<float>();
  const auto lineStride = maxLen;
  for (int sample = 0; sample < static_cast<int>(samples); ++sample) {
    for (int channel = 0; channel < static_cast<int>(channels); ++channel) {
      const auto inIndex =
          channel * static_cast<int>(samples) + sample;
      const auto lineBase = channel * lineStride;
      line[lineBase + writeIndex] = src[inIndex];
      auto readPos = static_cast<float>(writeIndex) - delaySamples;
      while (readPos < 0.0f)
        readPos += static_cast<float>(maxLen);
      const auto i0 = static_cast<int>(readPos) % maxLen;
      const auto i1 = (i0 + 1) % maxLen;
      const auto frac = readPos - std::floor(readPos);
      const auto delayed =
          line[lineBase + i0] * (1.0f - frac) + line[lineBase + i1] * frac;
      dst[inIndex] = delayed * wetGain;
      if (addDry)
        dst[inIndex] += src[inIndex];
    }
    writeIndex = (writeIndex + 1) % maxLen;
  }
  return output;
}
} // namespace openyourbox::dsp
