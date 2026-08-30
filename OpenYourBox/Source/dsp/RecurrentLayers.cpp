#include "RecurrentLayers.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
/**
 * @brief Applies Activation/TCN-style Gain then nonlinearity.
 * @param value Pre-activation tensor.
 * @param activation Selected in-cell function.
 * @param gain Pre-nonlinearity slope.
 * @param negativeSlope LeakyReLU slope.
 */
torch::Tensor applyCellActivation(torch::Tensor value,
                                  openyourbox::dsp::ActivationType activation,
                                  float gain, float negativeSlope) {
  if (std::abs(gain - 1.0f) > 1.0e-6f)
    value = value * gain;
  switch (activation) {
  case openyourbox::dsp::ActivationType::relu:
    return torch::relu(value);
  case openyourbox::dsp::ActivationType::sigmoid:
    return torch::where(value == 0.0, torch::zeros_like(value),
                        torch::sigmoid(value));
  case openyourbox::dsp::ActivationType::tanh:
    return torch::tanh(value);
  case openyourbox::dsp::ActivationType::leakyRelu:
    return torch::leaky_relu(value, negativeSlope);
  case openyourbox::dsp::ActivationType::prelu:
    return torch::prelu(value, torch::full({1}, 0.25f, value.options()));
  }
  return value;
}

/**
 * @brief Linear map `x @ W^T + b` for a `[B, in]` step.
 * @param step Input features `[B, in]`.
 * @param weight `[out, in]`.
 * @param bias Optional `[out]`.
 */
torch::Tensor linearStep(const torch::Tensor &step, const torch::Tensor &weight,
                         const torch::Tensor &bias) {
  if (bias.defined())
    return torch::linear(step, weight, bias);
  return torch::linear(step, weight);
}

/**
 * @brief Runs one unidirectional RNN or LSTM over time.
 * @param input `[B, C, T]`.
 * @param weights Forward (or backward) weight tensors.
 * @param config Shared architecture.
 * @param hidden Hidden `[B, H]`, updated.
 * @param cell LSTM cell `[B, H]`, updated.
 * @param reverse When true, iterate time backward.
 */
torch::Tensor runDirection(const torch::Tensor &input,
                           const std::vector<torch::Tensor> &weights,
                           const openyourbox::dsp::RecurrentConfig &config,
                           torch::Tensor &hidden, torch::Tensor &cell,
                           bool reverse) {
  const auto batch = input.size(0);
  const auto time = input.size(2);
  const auto hiddenSize = std::max(1, config.hiddenSize);
  if (!hidden.defined() || hidden.size(0) != batch ||
      hidden.size(1) != hiddenSize)
    hidden = torch::zeros({batch, hiddenSize}, input.options());
  if (config.lstm &&
      (!cell.defined() || cell.size(0) != batch || cell.size(1) != hiddenSize))
    cell = torch::zeros({batch, hiddenSize}, input.options());

  const auto &weightIh = weights[0];
  const auto &weightHh = weights[1];
  torch::Tensor biasIh;
  torch::Tensor biasHh;
  if (config.bias && weights.size() >= 4) {
    biasIh = weights[2];
    biasHh = weights[3];
  }

  std::vector<torch::Tensor> steps;
  steps.reserve(static_cast<std::size_t>(time));
  for (int index = 0; index < static_cast<int>(time); ++index) {
    const auto t = reverse ? static_cast<int>(time) - 1 - index : index;
    auto x = input.select(2, t);
    auto recurrentWeight = weightHh;
    if (std::abs(config.recurrentWeightScale - 1.0f) > 1.0e-6f)
      recurrentWeight = weightHh * config.recurrentWeightScale;
    auto prevHidden = hidden;
    auto gate = linearStep(x, weightIh, biasIh) +
                linearStep(hidden, recurrentWeight, biasHh);
    if (config.lstm) {
      auto chunks = gate.chunk(4, 1);
      auto inputGate = torch::sigmoid(chunks[0]);
      auto forgetGate = torch::sigmoid(chunks[1]);
      auto candidate = applyCellActivation(chunks[2], config.activation,
                                           config.gain, config.negativeSlope);
      auto outputGate = torch::sigmoid(chunks[3]);
      cell = forgetGate * cell + inputGate * candidate;
      hidden = outputGate * applyCellActivation(cell, config.activation,
                                                config.gain, config.negativeSlope);
    } else {
      hidden = applyCellActivation(gate, config.activation, config.gain,
                                   config.negativeSlope);
    }
    const auto leak = std::clamp(config.leakRate, 0.0f, 1.0f);
    if (std::abs(leak - 1.0f) > 1.0e-6f)
      hidden = prevHidden * (1.0f - leak) + hidden * leak;
    steps.push_back(hidden);
  }
  if (reverse)
    std::reverse(steps.begin(), steps.end());
  return torch::stack(steps, 2);
}
} // namespace

namespace openyourbox::dsp {
int RecurrentLayers::outputChannels(const RecurrentConfig &config) noexcept {
  const auto hidden = std::max(1, config.hiddenSize);
  return config.bidirectional ? hidden * 2 : hidden;
}

int RecurrentLayers::weightTensorCount(const RecurrentConfig &config) noexcept {
  const auto perDirection = config.bias ? 4 : 2;
  return config.bidirectional ? perDirection * 2 : perDirection;
}

torch::Tensor RecurrentLayers::process(const torch::Tensor &input,
                                       const std::vector<torch::Tensor> &weights,
                                       const RecurrentConfig &config,
                                       torch::Tensor &hidden,
                                       torch::Tensor &cell) {
  if (!input.defined() || input.dim() != 3 || weights.size() < 2)
    return input;
  const auto perDirection = config.bias ? 4 : 2;
  std::vector<torch::Tensor> forwardWeights(weights.begin(),
                                            weights.begin() + perDirection);
  torch::Tensor forwardHidden = hidden.defined() && hidden.dim() == 2 &&
                                        hidden.size(1) == config.hiddenSize * 2
                                    ? hidden.narrow(1, 0, config.hiddenSize)
                                    : hidden;
  torch::Tensor forwardCell = cell.defined() && cell.dim() == 2 &&
                                      cell.size(1) == config.hiddenSize * 2
                                  ? cell.narrow(1, 0, config.hiddenSize)
                                  : cell;
  auto forward = runDirection(input, forwardWeights, config, forwardHidden,
                              forwardCell, false);
  if (!config.bidirectional) {
    hidden = forwardHidden;
    cell = forwardCell;
    return forward;
  }
  std::vector<torch::Tensor> backwardWeights(
      weights.begin() + perDirection, weights.end());
  torch::Tensor backwardHidden;
  torch::Tensor backwardCell;
  if (hidden.defined() && hidden.dim() == 2 &&
      hidden.size(1) == config.hiddenSize * 2)
    backwardHidden = hidden.narrow(1, config.hiddenSize, config.hiddenSize);
  if (cell.defined() && cell.dim() == 2 &&
      cell.size(1) == config.hiddenSize * 2)
    backwardCell = cell.narrow(1, config.hiddenSize, config.hiddenSize);
  auto backward = runDirection(input, backwardWeights, config, backwardHidden,
                               backwardCell, true);
  hidden = torch::cat({forwardHidden, backwardHidden}, 1);
  if (config.lstm)
    cell = torch::cat({forwardCell, backwardCell}, 1);
  return torch::cat({forward, backward}, 1);
}
} // namespace openyourbox::dsp
