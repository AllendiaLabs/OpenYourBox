#include "PqmfBank.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace openyourbox::dsp {
namespace {
constexpr double kPi = 3.14159265358979323846;

/**
 * @brief Modified Bessel function I0 used by the Kaiser window.
 * @param x Argument.
 * @return I0(x).
 */
double besselI0(double x) noexcept {
  double sum = 1.0;
  double term = 1.0;
  const double half = x * 0.5;
  for (int k = 1; k < 50; ++k) {
    term *= (half / static_cast<double>(k)) * (half / static_cast<double>(k));
    sum += term;
    if (term < 1.0e-12 * sum)
      break;
  }
  return sum;
}

/**
 * @brief Kaiser window sample.
 * @param n Tap index in `[0, N)`.
 * @param length Window length N.
 * @param beta Shape parameter.
 */
double kaiserSample(int n, int length, double beta) noexcept {
  if (length <= 1)
    return 1.0;
  const auto alpha = static_cast<double>(length - 1) * 0.5;
  const auto arg = (static_cast<double>(n) - alpha) / alpha;
  const auto inside = 1.0 - arg * arg;
  if (inside <= 0.0)
    return 0.0;
  return besselI0(beta * std::sqrt(inside)) / besselI0(beta);
}

/**
 * @brief Designs a Kaiser low-pass prototype for an M-band PQMF.
 * @param nBand Sub-band count M.
 * @param attenuationDb Positive stopband attenuation.
 * @return Odd-length prototype.
 */
std::vector<float> designPrototype(int nBand, float attenuationDb) {
  const auto atten = std::max(20.0, static_cast<double>(attenuationDb));
  const auto width = 1.0 / static_cast<double>(std::max(2, nBand));
  auto length = static_cast<int>(std::ceil((atten - 8.0) / (2.285 * width * kPi)));
  length = std::max(length, 4 * nBand + 1);
  if ((length % 2) == 0)
    ++length;
  double beta = 0.0;
  if (atten > 50.0)
    beta = 0.1102 * (atten - 8.7);
  else if (atten >= 21.0)
    beta = 0.5842 * std::pow(atten - 21.0, 0.4) + 0.07886 * (atten - 21.0);

  const auto cutoff = kPi / static_cast<double>(nBand);
  const auto mid = (length - 1) / 2;
  std::vector<float> prototype(static_cast<std::size_t>(length), 0.0f);
  for (int n = 0; n < length; ++n) {
    const auto t = static_cast<double>(n - mid);
    const auto sinc =
        t == 0.0 ? cutoff / kPi : std::sin(cutoff * t) / (kPi * t);
    prototype[static_cast<std::size_t>(n)] =
        static_cast<float>(sinc * kaiserSample(n, length, beta));
  }
  return prototype;
}

/**
 * @brief Cosine-modulates a prototype into an M-band analysis bank.
 * @param prototype Low-pass prototype.
 * @param nBand Band count.
 * @return Tensor `[nBand, 1, K]`.
 */
torch::Tensor modulateBank(const std::vector<float> &prototype, int nBand) {
  const auto length = static_cast<int>(prototype.size());
  auto bank = torch::zeros(
      {nBand, 1, length},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const auto mid = length / 2;
  for (int k = 0; k < nBand; ++k) {
    const auto phase = std::pow(-1.0, k) * kPi / 4.0;
    for (int n = 0; n < length; ++n) {
      const auto t = static_cast<double>(n - mid);
      const auto mod =
          std::cos((2.0 * k + 1.0) * kPi / (2.0 * nBand) * t + phase);
      bank[k][0][n] = 2.0f * prototype[static_cast<std::size_t>(n)] *
                      static_cast<float>(mod);
    }
  }
  return bank;
}

/**
 * @brief Alternating-sign half-band flip used by acids-rave PQMF.
 * @param bands Multiband tensor.
 */
torch::Tensor reverseHalf(torch::Tensor bands) {
  auto mask = torch::ones_like(bands);
  auto accessor = mask.accessor<float, 3>();
  for (int b = 0; b < mask.size(0); ++b) {
    for (int c = 1; c < mask.size(1); c += 2) {
      for (int t = 0; t < mask.size(2); t += 2)
        accessor[b][c][t] = -1.0f;
    }
  }
  return bands * mask;
}
} // namespace

PqmfBank::PqmfBank(int nBand, float attenuationDb)
    : nBand(std::max(1, nBand)) {
  const auto prototype = designPrototype(this->nBand, attenuationDb);
  kernelSize = static_cast<int>(prototype.size());
  analysisKernels = modulateBank(prototype, this->nBand).contiguous();
  synthesisKernels = analysisKernels.flip(-1).permute({1, 0, 2}).contiguous();
}

int PqmfBank::getBandCount() const noexcept { return nBand; }

int PqmfBank::getKernelSize() const noexcept { return kernelSize; }

std::uint64_t PqmfBank::getCausalDelaySamples() const noexcept {
  return static_cast<std::uint64_t>(std::max(0, kernelSize - 1));
}

torch::Tensor PqmfBank::makeLeftover(int channels) const {
  const auto keep = std::max(1, kernelSize - 1);
  return torch::zeros(
      {1, std::max(1, channels), keep},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

torch::Tensor PqmfBank::analyse(const torch::Tensor &audio) const {
  if (!audio.defined() || audio.dim() != 3)
    throw std::invalid_argument("PQMF analysis expects [B, C, T]");
  if (nBand == 1)
    return audio;
  const auto audioChannels = static_cast<int>(audio.size(1));
  const auto pad = kernelSize / 2;
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<std::size_t>(audioChannels));
  for (int channel = 0; channel < audioChannels; ++channel) {
    auto mono = audio.narrow(1, channel, 1);
    auto padded = torch::nn::functional::pad(
        mono, torch::nn::functional::PadFuncOptions({pad, pad})
                  .mode(torch::kConstant)
                  .value(0.0));
    auto bands = torch::conv1d(padded, analysisKernels,
                               std::optional<torch::Tensor>{},
                               std::array<std::int64_t, 1>{nBand},
                               std::array<std::int64_t, 1>{0});
    parts.push_back(reverseHalf(bands));
  }
  return torch::cat(parts, 1);
}

torch::Tensor PqmfBank::synthesise(const torch::Tensor &bands,
                                   int audioChannels) const {
  if (!bands.defined() || bands.dim() != 3)
    throw std::invalid_argument("PQMF synthesis expects [B, C, T]");
  if (nBand == 1)
    return bands;
  audioChannels = std::max(1, audioChannels);
  const auto pad = kernelSize / 2;
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<std::size_t>(audioChannels));
  for (int channel = 0; channel < audioChannels; ++channel) {
    auto slice = reverseHalf(bands.narrow(1, channel * nBand, nBand));
    auto up = torch::zeros(
        {slice.size(0), nBand, slice.size(2) * nBand}, slice.options());
    up.slice(2, 0, up.size(2), nBand) = slice * static_cast<float>(nBand);
    auto padded = torch::nn::functional::pad(
        up, torch::nn::functional::PadFuncOptions({pad, pad})
                .mode(torch::kConstant)
                .value(0.0));
    auto reconstructed =
        torch::conv1d(padded, synthesisKernels, std::optional<torch::Tensor>{});
    parts.push_back(reconstructed);
  }
  return torch::cat(parts, 1);
}

torch::Tensor PqmfBank::analyseStreaming(const torch::Tensor &audio,
                                         torch::Tensor &leftover) const {
  if (!audio.defined() || audio.dim() != 3)
    return audio;
  if (nBand == 1)
    return audio;
  const auto history = std::max<std::int64_t>(1, kernelSize - 1);
  if (!leftover.defined() || leftover.dim() != 3 ||
      leftover.size(1) != audio.size(1)) {
    leftover = torch::zeros({audio.size(0), audio.size(1), history},
                            audio.options());
  }
  auto extended = torch::cat({leftover, audio}, 2);
  leftover = extended.narrow(2, extended.size(2) - history, history).clone();
  const auto available = extended.size(2);
  const auto hops = available / nBand;
  if (hops < 1)
    return torch::zeros({audio.size(0), audio.size(1) * nBand, 0},
                        audio.options());
  const auto usable = hops * nBand;
  auto window = extended.narrow(2, 0, usable);
  const auto audioChannels = static_cast<int>(window.size(1));
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<std::size_t>(audioChannels));
  for (int channel = 0; channel < audioChannels; ++channel) {
    auto mono = window.narrow(1, channel, 1);
    auto bands = torch::conv1d(mono, analysisKernels, {},
                               std::array<std::int64_t, 1>{nBand});
    parts.push_back(reverseHalf(bands));
  }
  return torch::cat(parts, 1);
}

torch::Tensor PqmfBank::synthesiseStreaming(const torch::Tensor &bands,
                                            torch::Tensor &leftover,
                                            int audioChannels) const {
  if (!bands.defined() || bands.dim() != 3)
    return bands;
  if (nBand == 1)
    return bands;
  audioChannels = std::max(1, audioChannels);
  const auto history = std::max<std::int64_t>(1, kernelSize - 1);
  const auto bandFrames = bands.size(2);
  if (bandFrames < 1)
    return torch::zeros({bands.size(0), audioChannels, 0}, bands.options());

  const auto emitSamples = bandFrames * static_cast<std::int64_t>(nBand);
  const auto bandPlanes =
      static_cast<std::int64_t>(audioChannels) * static_cast<std::int64_t>(nBand);
  if (!leftover.defined() || leftover.dim() != 3 ||
      leftover.size(1) != bandPlanes) {
    leftover =
        torch::zeros({bands.size(0), bandPlanes, history}, bands.options());
  }

  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<std::size_t>(audioChannels));
  for (int channel = 0; channel < audioChannels; ++channel) {
    auto slice = reverseHalf(bands.narrow(1, channel * nBand, nBand));
    auto up = torch::zeros({slice.size(0), nBand, emitSamples}, slice.options());
    up.slice(2, 0, up.size(2), nBand) = slice * static_cast<float>(nBand);

    const auto plane = static_cast<std::int64_t>(channel) * nBand;
    auto channelHistory = leftover.narrow(1, plane, nBand);
    auto extended = torch::cat({channelHistory, up}, 2);
    channelHistory.copy_(extended.narrow(2, extended.size(2) - history, history));
    parts.push_back(torch::conv1d(extended, synthesisKernels,
                                  std::optional<torch::Tensor>{}));
  }
  return torch::cat(parts, 1);
}
} // namespace openyourbox::dsp
