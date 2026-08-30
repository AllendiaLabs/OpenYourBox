#pragma once

#include "LiveGraphEngine.h"

#include <torch/script.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
   *
   * Probes 256 samples first, then a `_b<n>_` filename hint (RAVE streaming
   * exports such as `…_b2048_r48000_z16.ts`), then 512…16384. `forward` schema
   * is inspected so 1-tensor TraceModels are never called with a cond tensor.
   * Rate-changing RAVE graphs may return a different time length; live playback
   * crops or left-pads with matchTimeLength. Models that reject short probes
   * are accumulated to the working hop inside the kernel.
   */
  [[nodiscard]] static std::shared_ptr<const TorchScriptBlackBoxFactory>
  load(const std::string &artifactPath, int inputChannels,
       std::uint64_t receptiveFieldSamples, std::string &error,
       bool requireSilencePreservation = true,
       bool acceptsConditioning = false, int condDim = 0,
       double hostSampleRate = 0.0);

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

  /** @brief True when load probed a conditioned `forward`. */
  [[nodiscard]] bool acceptsConditioning() const noexcept override;

  /** @brief Encode output width, or 0 when encode/decode is absent. */
  [[nodiscard]] int getLatentChannels() const noexcept override;

  /** @brief Audio samples accepted by one validated inference call. */
  [[nodiscard]] int getInferenceBlockSamples() const noexcept override;

  /** @brief Latent frames produced by one validated encode call. */
  [[nodiscard]] int getLatentFramesPerBlock() const noexcept override;

  /** @brief True when encode/decode require complete inference blocks. */
  [[nodiscard]] bool requiresFixedInferenceBlock() const noexcept override;

  /** @brief True when compactness PCA attrs were copied from the artifact. */
  [[nodiscard]] bool compactnessReady() const noexcept override;

  /** @brief Compactness mean copied at load. */
  [[nodiscard]] const std::vector<float> &compactnessMean() const override;

  /** @brief Compactness PCA copied at load. */
  [[nodiscard]] const std::vector<float> &compactnessPca() const override;

  /** @brief Cumulative variance copied at load. */
  [[nodiscard]] const std::vector<float> &
  compactnessCumulative() const override;

  /** @brief Sample-rate mismatch notice, or empty. */
  [[nodiscard]] const std::string &sampleRateWarning() const override;

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
   * @param latentChannels Encode output width, or 0.
   * @param latentFrames Encode output time length for one probe block.
   * @param compactnessReady True when PCA attrs were present.
   * @param latentMean Compactness mean buffer.
   * @param latentPca Compactness PCA buffer.
   * @param cumulativeVariance Compactness cumulative ratios.
   * @param sampleRateWarning Best-effort rate mismatch text.
   * @param hopSamples Time length of the successful load probe.
   * @param requiresFixedHop True when shorter probes failed.
   */
  TorchScriptBlackBoxFactory(std::string path, int inputs, int outputs,
                             std::uint64_t field, std::uint64_t parameters,
                             bool silence, bool conditioned, int condDim,
                             bool encodeDecode, int latentChannels,
                             int latentFrames,
                             bool compactnessReady,
                             std::vector<float> latentMean,
                             std::vector<float> latentPca,
                             std::vector<float> cumulativeVariance,
                             std::string sampleRateWarning, int hopSamples,
                             bool requiresFixedHop);

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
  /** @brief Encode output channel count, or 0. */
  int latentChannels = 0;
  /** @brief Encode output frames for one validated inference block. */
  int latentFramesPerBlock = 1;
  /** @brief True when compactness attrs were copied. */
  bool compactnessBuffersReady = false;
  /** @brief Compactness mean `[latent]`. */
  std::vector<float> latentMean;
  /** @brief Compactness PCA `[latent × latent]` row-major. */
  std::vector<float> latentPca;
  /** @brief Cumulative singular-value ratios. */
  std::vector<float> cumulativeVariance;
  /** @brief Sample-rate mismatch notice. */
  std::string rateWarning;
  /** @brief Time length of the successful load probe. */
  int inferenceBlockSamples = 256;
  /** @brief True when live audio must be accumulated to `inferenceBlockSamples`. */
  bool requiresFixedHop = false;
};
} // namespace openyourbox::dsp
