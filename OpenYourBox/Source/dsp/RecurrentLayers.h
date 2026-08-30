#pragma once

#include "TCNModel.h"

#include <torch/torch.h>

#include <cstdint>

namespace openyourbox::dsp {
/**
 * @struct RecurrentConfig
 * @brief Single-layer LSTM/RNN settings shared by live compile and process.
 */
struct RecurrentConfig {
  /** @brief Input feature size inferred from the upstream channel count. */
  int inputSize = 1;
  /** @brief Hidden size of one unidirectional layer. */
  int hiddenSize = 16;
  /** @brief When true, concatenate forward and backward hidden states. */
  bool bidirectional = false;
  /** @brief When true, cells include learnable bias terms. */
  bool bias = true;
  /** @brief In-cell primary nonlinearity. */
  ActivationType activation = ActivationType::tanh;
  /** @brief Pre-nonlinearity Gain applied to the in-cell activation. */
  float gain = 1.0f;
  /** @brief LeakyReLU negative slope when that activation is selected. */
  float negativeSlope = 0.01f;
  /** @brief True for LSTM (gates + cell); false for vanilla RNN. */
  bool lstm = false;
  /**
   * @brief Leaky-integrator mix in `[0, 1]`.
   *
   * `1` keeps the standard cell output. Smaller values mix the previous
   * hidden state: `h_t = (1 - leak) * h_{t-1} + leak * h_new`.
   */
  float leakRate = 1.0f;
  /**
   * @brief Multiplier applied to hidden-to-hidden (`W_hh`) weights.
   *
   * `1` leaves recurrent weights unchanged. Does not scale input weights or
   * bias terms.
   */
  float recurrentWeightScale = 1.0f;
};

/**
 * @class RecurrentLayers
 * @brief Custom single-layer RNN/LSTM with Activation/TCN in-cell nonlinearity.
 *
 * Hidden (and LSTM cell) state is carried across live buffers via the
 * caller-owned tensors. Bidirectional processing concatenates a reverse pass
 * along the channel axis.
 */
class RecurrentLayers {
public:
  /**
   * @brief Output channel count for @p config.
   * @param config Recurrent element configuration.
   * @return `hiddenSize` or `2 * hiddenSize` when bidirectional.
   */
  [[nodiscard]] static int outputChannels(const RecurrentConfig &config) noexcept;

  /**
   * @brief Number of parameter tensors stored in compiled `weights`.
   * @param config Recurrent element configuration.
   * @return Count of weight/bias tensors in execution order.
   */
  [[nodiscard]] static int weightTensorCount(const RecurrentConfig &config) noexcept;

  /**
   * @brief Runs one buffer of full-sequence recurrence.
   * @param input Sequence `[1, inputSize, time]`.
   * @param weights Compiled weight tensors from `randomizeElementWeights`.
   * @param config Architecture and activation settings.
   * @param hidden Forward (and optional backward) hidden state, updated.
   * @param cell LSTM cell state, updated when `config.lstm` is true.
   * @return Output `[1, outputChannels, time]`.
   */
  [[nodiscard]] static torch::Tensor
  process(const torch::Tensor &input, const std::vector<torch::Tensor> &weights,
          const RecurrentConfig &config, torch::Tensor &hidden,
          torch::Tensor &cell);
};
} // namespace openyourbox::dsp
