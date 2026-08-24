#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace openyourbox::dsp {
/** @brief Non-linearities available after each temporal convolution. */
enum class ActivationType { relu = 0, sigmoid, tanh, leakyRelu, prelu };

/** @brief Inclusive maximum activation enum index (PReLU). */
inline constexpr int maximumActivationIndex = 4;

/** @brief Immutable architecture settings used to construct a TCN. */
struct TCNConfiguration {
  int depth = 4;
  int kernelSize = 3;
  int channels = 16;
  /** @brief Dilation growth G so layer n uses dilation G^n. */
  int dilationGrowth = 2;
  int inputChannels = 2;
  int outputChannels = 2;
  ActivationType activation = ActivationType::relu;
  /** @brief Pre-nonlinearity Gain applied to each TCN activation stage. */
  float gain = 1.0f;
  /** @brief Whether each temporal block adds a residual path. */
  bool residual = false;

  /** @brief Returns true when all values can be represented by LibTorch Conv1d.
   */
  [[nodiscard]] bool isValid() const noexcept;
};

/**
 * @class TCNModel
 * @brief Runtime-constructed causal temporal convolution network.
 *
 * The module owns one input projection, a dilation-doubling convolution stack,
 * and one output projection. Biases are disabled to guarantee silence-in,
 * silence-out behavior.
 */
class TCNModel final : public torch::nn::Module {
public:
  /** @brief Constructs an inference-ready model for the supplied architecture.
   */
  explicit TCNModel(TCNConfiguration configuration);

  /**
   * @brief Runs causal inference on a tensor shaped [batch, channels, samples].
   * @param input Input tensor on CPU.
   * @return Output tensor with the same temporal length as input.
   */
  torch::Tensor forward(const torch::Tensor &input);

  /**
   * @brief Deterministically replaces every trainable weight.
   * @param seed Per-instance randomization seed.
   */
  void randomizeWeights(std::uint64_t seed);

  /** @brief Returns the immutable model configuration. */
  [[nodiscard]] const TCNConfiguration &getConfiguration() const noexcept;

  /** @brief Returns the causal receptive field in samples, saturated on
   * overflow. */
  [[nodiscard]] std::uint64_t getReceptiveField() const noexcept;

  /** @brief Returns the receptive field duration for a sample rate. */
  [[nodiscard]] double
  getReceptiveFieldMilliseconds(double sampleRate) const noexcept;

  /** @brief Returns the total number of trainable scalar parameters. */
  [[nodiscard]] std::uint64_t getParameterCount() const noexcept;

  /** @brief Returns a stable hash of architecture values, excluding weights. */
  [[nodiscard]] std::uint64_t getArchitectureHash() const noexcept;

private:
  torch::Tensor applyActivation(torch::Tensor value, int layer);

  TCNConfiguration config;
  torch::nn::Conv1d inputProjection{nullptr};
  torch::nn::ModuleList convolutions;
  torch::nn::ModuleList residualProjections;
  torch::nn::ModuleList preluActivations;
  torch::nn::Conv1d outputProjection{nullptr};
  std::uint64_t receptiveField = 1;
};
} // namespace openyourbox::dsp
