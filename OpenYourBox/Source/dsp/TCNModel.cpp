#include "TCNModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
std::uint64_t splitMix64(std::uint64_t &state) noexcept {
  auto value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

float uniformSigned(std::uint64_t &state) noexcept {
  constexpr auto inverse = 1.0 / static_cast<double>(std::uint64_t{1} << 53U);
  const auto unit = static_cast<double>(splitMix64(state) >> 11U) * inverse;
  return static_cast<float>(unit * 2.0 - 1.0);
}

void hashCombine(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
}
} // namespace

namespace openyourbox::dsp {
bool TCNConfiguration::isValid() const noexcept {
  if (depth < 1 || depth > 30 || kernelSize < 2 || channels < 1 ||
      dilationGrowth < 1 || dilationGrowth > 16 ||
      inputChannels < 1 || inputChannels > 2 ||
      outputChannels != inputChannels)
    return false;
  long long value = 1;
  for (int layer = 0; layer < depth; ++layer) {
    if (value > std::numeric_limits<int>::max())
      return false;
    value *= dilationGrowth;
  }
  return true;
}

TCNModel::TCNModel(TCNConfiguration configuration) : config(configuration) {
  if (!config.isValid())
    throw std::invalid_argument("Unsupported TCN configuration");

  inputProjection = register_module(
      "input_projection",
      torch::nn::Conv1d(
          torch::nn::Conv1dOptions(config.inputChannels, config.channels, 1)
              .bias(false)));

  convolutions = register_module("convolutions", torch::nn::ModuleList());
  residualProjections =
      register_module("residual_projections", torch::nn::ModuleList());
  preluActivations = register_module("prelu_activations", torch::nn::ModuleList());
  receptiveField = 1;

  long long growthPower = 1;
  for (int layer = 0; layer < config.depth; ++layer) {
    const auto dilation = static_cast<int>(
        std::min<long long>(growthPower, std::numeric_limits<int>::max()));
    convolutions->push_back(torch::nn::Conv1d(
        torch::nn::Conv1dOptions(config.channels, config.channels,
                                 config.kernelSize)
            .dilation(dilation)
            .bias(false)));
    if (config.residual) {
      residualProjections->push_back(torch::nn::Conv1d(
          torch::nn::Conv1dOptions(config.channels, config.channels, 1)
              .bias(false)));
    }
    if (config.activation == ActivationType::prelu) {
      preluActivations->push_back(
          torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(config.channels)));
    }

    const auto contribution =
        static_cast<std::uint64_t>(config.kernelSize - 1) *
        static_cast<std::uint64_t>(dilation);
    receptiveField = contribution > std::numeric_limits<std::uint64_t>::max() -
                                        receptiveField
                         ? std::numeric_limits<std::uint64_t>::max()
                         : receptiveField + contribution;
    if (growthPower > std::numeric_limits<long long>::max() / std::max(1, config.dilationGrowth))
      growthPower = std::numeric_limits<long long>::max();
    else
      growthPower *= std::max(1, config.dilationGrowth);
  }

  outputProjection = register_module(
      "output_projection",
      torch::nn::Conv1d(
          torch::nn::Conv1dOptions(config.channels, config.outputChannels, 1)
              .bias(false)));

  eval();
}

torch::Tensor TCNModel::forward(const torch::Tensor &input) {
  auto value = inputProjection->forward(input);

  for (std::size_t layer = 0; layer < convolutions->size(); ++layer) {
    long long growthPower = 1;
    for (std::size_t index = 0; index < layer; ++index)
      growthPower *= std::max(1, config.dilationGrowth);
    const auto dilation = static_cast<std::int64_t>(
        std::min<long long>(growthPower, std::numeric_limits<int>::max()));
    const auto leftPadding =
        static_cast<std::int64_t>(config.kernelSize - 1) * dilation;
    auto residual = value;
    value = torch::nn::functional::pad(
        value, torch::nn::functional::PadFuncOptions({leftPadding, 0})
                   .mode(torch::kConstant)
                   .value(0.0));
    value = convolutions[layer]->as<torch::nn::Conv1d>()->forward(value);
    value = applyActivation(std::move(value), static_cast<int>(layer));
    if (config.residual && layer < residualProjections->size()) {
      auto projected =
          residualProjections[layer]->as<torch::nn::Conv1d>()->forward(residual);
      value = value + projected.narrow(2, projected.size(2) - value.size(2),
                                       value.size(2));
    }
  }

  return outputProjection->forward(value);
}

void TCNModel::randomizeWeights(std::uint64_t seed) {
  torch::NoGradGuard noGrad;
  auto state = seed;

  for (auto &parameter : parameters()) {
    auto contiguous = parameter.contiguous();
    auto *data = contiguous.data_ptr<float>();
    const auto count = contiguous.numel();
    const auto fanIn =
        count > 0 ? std::max<std::int64_t>(
                        1, parameter.size(-1) *
                               (parameter.dim() > 1 ? parameter.size(1) : 1))
                  : 1;
    const auto scale =
        static_cast<float>(std::sqrt(6.0 / static_cast<double>(fanIn)));

    for (std::int64_t index = 0; index < count; ++index)
      data[index] = uniformSigned(state) * scale;

    if (!parameter.is_contiguous())
      parameter.copy_(contiguous);
  }
}

const TCNConfiguration &TCNModel::getConfiguration() const noexcept {
  return config;
}

std::uint64_t TCNModel::getReceptiveField() const noexcept {
  return receptiveField;
}

double
TCNModel::getReceptiveFieldMilliseconds(double sampleRate) const noexcept {
  return sampleRate > 0.0
             ? static_cast<double>(receptiveField) * 1000.0 / sampleRate
             : 0.0;
}

std::uint64_t TCNModel::getParameterCount() const noexcept {
  std::uint64_t count = 0;
  for (const auto &parameter : parameters())
    count += static_cast<std::uint64_t>(parameter.numel());
  return count;
}

std::uint64_t TCNModel::getArchitectureHash() const noexcept {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  hashCombine(hash, static_cast<std::uint64_t>(config.depth));
  hashCombine(hash, static_cast<std::uint64_t>(config.kernelSize));
  hashCombine(hash, static_cast<std::uint64_t>(config.channels));
  hashCombine(hash, static_cast<std::uint64_t>(config.inputChannels));
  hashCombine(hash, static_cast<std::uint64_t>(config.outputChannels));
  hashCombine(hash, static_cast<std::uint64_t>(config.activation));
  hashCombine(hash, static_cast<std::uint64_t>(config.dilationGrowth));
  hashCombine(hash, static_cast<std::uint64_t>(config.residual ? 1 : 0));
  hashCombine(hash, static_cast<std::uint64_t>(
                        static_cast<int>(config.gain * 1000.0f)));
  hashCombine(hash, static_cast<std::uint64_t>(
                        static_cast<int>(config.negativeSlope * 1000.0f)));
  return hash;
}

/**
 * @brief Applies the configured TCN activation for @p layer.
 * @param value Pre-activation tensor.
 * @param layer Temporal block index (used for PReLU parameters).
 */
torch::Tensor TCNModel::applyActivation(torch::Tensor value, int layer) {
  if (std::abs(config.gain - 1.0f) > 1.0e-6f)
    value = value * config.gain;
  switch (config.activation) {
  case ActivationType::relu:
    return torch::relu(value);
  case ActivationType::sigmoid:
    return torch::sigmoid(value);
  case ActivationType::tanh:
    return torch::tanh(value);
  case ActivationType::leakyRelu:
    return torch::leaky_relu(value, config.negativeSlope);
  case ActivationType::prelu:
    if (layer >= 0 && static_cast<std::size_t>(layer) < preluActivations->size())
      return preluActivations[static_cast<std::size_t>(layer)]
          ->as<torch::nn::PReLU>()
          ->forward(value);
    return torch::prelu(value, torch::tensor({0.25f}));
  }

  return value;
}
} // namespace openyourbox::dsp
