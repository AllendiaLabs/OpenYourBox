#pragma once

#include "LiveGraphEngine.h"

#include <torch/script.h>

#include <cstdint>
#include <memory>
#include <string>

namespace openyourbox::dsp {
/**
 * @class TorchScriptBlackBoxFactory
 * @brief Validated immutable metadata for one frozen TorchScript artifact.
 */
class TorchScriptBlackBoxFactory final : public FrozenBlackBoxFactory {
public:
  /**
   * @brief Loads, warms, and validates a local TorchScript artifact.
   * @param artifactPath Absolute local artifact path.
   * @param inputChannels Host or upstream channels supplied to the artifact.
   * @param receptiveFieldSamples Causal context declared by the worker.
   * @param error Receives a user-facing validation failure.
   * @param acceptsConditioning True when the module expects (audio, cond).
   *   When false, load still retries a conditioned forward if `forward(audio)`
   *   fails because `cond` is required (training checkpoints).
   * @param condDim Trained FiLM width, or 0 to detect during validation.
   */
  [[nodiscard]] static std::shared_ptr<const TorchScriptBlackBoxFactory>
  load(const std::string &artifactPath, int inputChannels,
       std::uint64_t receptiveFieldSamples, std::string &error,
       bool requireSilencePreservation = true,
       bool acceptsConditioning = false, int condDim = 0);

  /** @brief Returns the artifact's validated input channel count. */
  [[nodiscard]] int getInputChannels() const noexcept override;

  /** @brief Returns the artifact's validated output channel count. */
  [[nodiscard]] int getOutputChannels() const noexcept override;

  /** @brief Returns the artifact's declared causal receptive field. */
  [[nodiscard]] std::uint64_t getReceptiveField() const noexcept override;

  /** @brief Returns the artifact's scalar parameter count. */
  [[nodiscard]] std::uint64_t getParameterCount() const noexcept override;

  /** @brief Reports whether validated zero input produced exact zero output. */
  [[nodiscard]] bool preservesSilence() const noexcept override;

  /** @brief Loads a runtime-local TorchScript module off the audio thread. */
  [[nodiscard]] std::unique_ptr<FrozenBlackBoxKernel>
  createKernel() const override;

  /** @brief True when encode and decode methods were found on the artifact. */
  [[nodiscard]] bool hasEncodeDecode() const noexcept override;

  /** @brief Returns the canonical artifact path. */
  [[nodiscard]] const std::string &getArtifactPath() const noexcept;

private:
  /**
   * @brief Stores validated artifact metadata.
   * @param path Canonical artifact path.
   * @param inputs Validated input channels.
   * @param outputs Validated output channels.
   * @param field Causal receptive field.
   * @param parameters Scalar parameter count.
   * @param silence Whether zero input remains exactly zero.
   * @param conditioned True when the module accepts a control tensor.
   * @param condDim Validated FiLM control width.
   * @param encodeDecode True when encode and decode methods exist.
   */
  TorchScriptBlackBoxFactory(std::string path, int inputs, int outputs,
                             std::uint64_t field, std::uint64_t parameters,
                             bool silence, bool conditioned, int condDim,
                             bool encodeDecode);

  /** @brief Canonical local TorchScript file path. */
  std::string artifactPath;
  /** @brief Exact accepted input channels. */
  int validatedInputChannels = 0;
  /** @brief Exact produced output channels. */
  int validatedOutputChannels = 0;
  /** @brief Causal receptive field supplied by artifact metadata. */
  std::uint64_t receptiveField = 1;
  /** @brief Scalar parameter count read from the module. */
  std::uint64_t parameterCount = 0;
  /** @brief Exact-silence validation result. */
  bool silencePreserving = false;
  /** @brief True when the scripted module accepts (audio, cond). */
  bool conditioned = false;
  /** @brief FiLM control width used at trace and runtime. */
  int conditioningDim = 2;
  /** @brief True when encode/decode methods were validated. */
  bool encodeDecode = false;
};
} // namespace openyourbox::dsp
