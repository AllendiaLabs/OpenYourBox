#include "RateConv.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>

namespace openyourbox::dsp {
RateConv::RateConv(int stride, int kernelSize, int dilation, RateConvMode mode)
    : stride(std::max(1, stride)), kernelSize(std::max(1, kernelSize)),
      dilation(std::max(1, dilation)), mode(mode) {}

int RateConv::getStride() const noexcept { return stride; }

int RateConv::getKernelSize() const noexcept { return kernelSize; }

std::uint64_t RateConv::getCausalDelaySamples() const noexcept {
  return static_cast<std::uint64_t>(std::max(0, kernelSize - 1)) *
         static_cast<std::uint64_t>(dilation);
}

torch::Tensor RateConv::process(const torch::Tensor &input,
                                const torch::Tensor &weight) const {
  if (!input.defined() || input.dim() != 3 || !weight.defined())
    throw std::invalid_argument("rateConv expects [B, C, T] and a conv weight");
  const auto leftPad =
      static_cast<std::int64_t>(kernelSize - 1) * static_cast<std::int64_t>(dilation);
  auto padded = torch::nn::functional::pad(
      input, torch::nn::functional::PadFuncOptions({leftPad, 0})
                 .mode(torch::kConstant)
                 .value(0.0));
  const std::array<std::int64_t, 1> dilationValues{
      static_cast<std::int64_t>(dilation)};
  const std::array<std::int64_t, 1> padding{0};
  if (mode == RateConvMode::downsample) {
    const std::array<std::int64_t, 1> strideValues{
        static_cast<std::int64_t>(stride)};
    return torch::conv1d(padded, weight, std::optional<torch::Tensor>{},
                         strideValues, padding, dilationValues, 1);
  }
  auto up = torch::zeros({input.size(0), input.size(1), input.size(2) * stride},
                         input.options());
  up.slice(2, 0, up.size(2), stride) = input;
  auto upPadded = torch::nn::functional::pad(
      up, torch::nn::functional::PadFuncOptions({leftPad, 0})
              .mode(torch::kConstant)
              .value(0.0));
  const std::array<std::int64_t, 1> unitStride{1};
  return torch::conv1d(upPadded, weight, std::optional<torch::Tensor>{},
                       unitStride, padding, dilationValues, 1);
}

torch::Tensor RateConv::processStreaming(const torch::Tensor &input,
                                         const torch::Tensor &weight,
                                         torch::Tensor &leftover) const {
  if (!input.defined() || input.dim() != 3)
    return input;
  const auto history = static_cast<std::int64_t>(getCausalDelaySamples());
  const auto hop = mode == RateConvMode::downsample
                       ? static_cast<std::int64_t>(stride)
                       : 1;
  if (!leftover.defined() || leftover.dim() != 3 ||
      leftover.size(1) != input.size(1)) {
    leftover = torch::zeros({input.size(0), input.size(1), std::max(history, hop)},
                            input.options());
  }
  auto extended = torch::cat({leftover, input}, 2);
  leftover = extended
                 .narrow(2, extended.size(2) - leftover.size(2), leftover.size(2))
                 .clone();
  if (mode == RateConvMode::downsample) {
    const auto hops = extended.size(2) / hop;
    if (hops < 1)
      return torch::zeros({input.size(0), weight.size(0), 0}, input.options());
    const auto usable = hops * hop;
    auto window = extended.narrow(2, 0, usable);
    return process(window, weight);
  }
  return process(input, weight);
}

ConvTranspose1d::ConvTranspose1d(int stride, int kernelSize, int dilation)
    : stride(std::max(1, stride)), kernelSize(std::max(1, kernelSize)),
      dilation(std::max(1, dilation)) {}

int ConvTranspose1d::getStride() const noexcept { return stride; }

int ConvTranspose1d::getKernelSize() const noexcept { return kernelSize; }

std::uint64_t ConvTranspose1d::getCausalDelaySamples() const noexcept {
  return static_cast<std::uint64_t>(std::max(0, kernelSize - 1)) *
         static_cast<std::uint64_t>(dilation);
}

torch::Tensor ConvTranspose1d::process(const torch::Tensor &input,
                                       const torch::Tensor &weight) const {
  if (!input.defined() || input.dim() != 3 || !weight.defined())
    throw std::invalid_argument(
        "ConvTranspose1d expects [B, C, T] and a conv weight");
  const auto leftPad =
      static_cast<std::int64_t>(kernelSize - 1) * static_cast<std::int64_t>(dilation);
  auto padded = torch::nn::functional::pad(
      input, torch::nn::functional::PadFuncOptions({leftPad, 0})
                 .mode(torch::kConstant)
                 .value(0.0));
  const std::array<std::int64_t, 1> strideValues{
      static_cast<std::int64_t>(stride)};
  const std::array<std::int64_t, 1> paddingValues{
      static_cast<std::int64_t>(stride / 2)};
  const std::array<std::int64_t, 1> dilationValues{
      static_cast<std::int64_t>(dilation)};
  const auto transposeWeight = weight.permute({1, 0, 2});
  return torch::conv_transpose1d(
      padded, transposeWeight, std::optional<torch::Tensor>{}, strideValues,
      paddingValues, std::array<std::int64_t, 1>{0}, 1, dilationValues);
}

torch::Tensor ConvTranspose1d::processStreaming(const torch::Tensor &input,
                                                const torch::Tensor &weight,
                                                torch::Tensor &leftover) const {
  if (!input.defined() || input.dim() != 3)
    return input;
  const auto history = static_cast<std::int64_t>(getCausalDelaySamples());
  if (!leftover.defined() || leftover.dim() != 3 ||
      leftover.size(1) != input.size(1)) {
    leftover = torch::zeros({input.size(0), input.size(1),
                             std::max<std::int64_t>(history, 1)},
                            input.options());
  }
  auto extended = torch::cat({leftover, input}, 2);
  leftover = extended
                 .narrow(2, extended.size(2) - leftover.size(2), leftover.size(2))
                 .clone();
  return process(extended, weight);
}
} // namespace openyourbox::dsp
