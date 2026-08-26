#include "TorchScriptBlackBox.h"

#include <algorithm>
#include <limits>
#include <utility>

#include <torch/nn/functional.h>

namespace {
/**
 * @brief Repeats, averages, or crops channels to a target width.
 * @param input Tensor shaped `[batch, channels, time]`.
 * @param channels Desired channel count.
 * @return Tensor with `channels` in dimension 1.
 */
torch::Tensor matchChannelCount(const torch::Tensor &input, int channels) {
  if (!input.defined() || input.dim() != 3 || channels < 1)
    return input;
  if (input.size(1) == channels)
    return input;
  if (channels == 1)
    return input.mean(1, true);
  if (input.size(1) == 1)
    return input.repeat({1, channels, 1});
  if (input.size(1) > channels)
    return input.narrow(1, 0, channels);
  auto padded = torch::zeros({input.size(0), channels, input.size(2)},
                             input.options());
  padded.narrow(1, 0, input.size(1)).copy_(input);
  return padded;
}

/**
 * @brief Shapes a live control tensor to `[batch, condDim, time]` for FiLM.
 *
 * Time is preserved so a ramped Knob/XY is not flattened to one value per
 * audio buffer (that zipper is heard as a low buzz at `sampleRate / blockSize`).
 * When @p targetLength is set, a shorter trajectory is left-padded with its
 * first frame and a longer one is cropped to the tail.
 *
 * @param conditioning Live control or undefined placeholder.
 * @param batch Batch size of the audio input.
 * @param condDim Trained control width.
 * @param options Tensor options copied from the audio input.
 * @param targetLength Desired time length, or 0 to keep the source length.
 */
torch::Tensor normalizeConditioning(const torch::Tensor &conditioning,
                                    std::int64_t batch, int condDim,
                                    const torch::TensorOptions &options,
                                    std::int64_t targetLength = 0) {
  const auto width = std::max(1, condDim);
  const auto fallbackTime = std::max<std::int64_t>(1, targetLength);
  if (!conditioning.defined())
    return torch::zeros({batch, width, fallbackTime}, options);
  auto cond = conditioning;
  if (cond.dim() == 1)
    cond = cond.unsqueeze(0).unsqueeze(-1);
  else if (cond.dim() == 2)
    cond = cond.unsqueeze(-1);
  if (cond.dim() != 3)
    return torch::zeros({batch, width, fallbackTime}, options);
  if (cond.size(0) != batch)
    cond = cond.repeat({batch, 1, 1});
  if (cond.size(1) < width) {
    auto padded = torch::zeros({cond.size(0), width, cond.size(2)}, cond.options());
    padded.narrow(1, 0, cond.size(1)).copy_(cond);
    cond = padded;
  } else if (cond.size(1) > width)
    cond = cond.narrow(1, 0, width);
  if (targetLength < 1 || cond.size(2) == targetLength)
    return cond;
  if (cond.size(2) > targetLength)
    return cond.narrow(2, cond.size(2) - targetLength, targetLength);
  const auto missing = targetLength - cond.size(2);
  auto head = cond.narrow(2, 0, 1).expand({cond.size(0), cond.size(1), missing});
  return torch::cat({head.contiguous(), cond}, 2);
}

/**
 * @brief Reports whether the module accepts a control tensor whose time
 * dimension matches the audio block.
 * @param module Loaded TorchScript module.
 * @param silence Example audio of the desired time length.
 * @param condDim Control width.
 * @return True when a matching-length `cond` forwards successfully.
 */
bool probeTimeVaryingConditioning(torch::jit::Module &module,
                                  const torch::Tensor &silence, int condDim) {
  try {
    const auto zeros = torch::zeros(
        {1, std::max(1, condDim), silence.size(2)}, silence.options());
    const auto value = module.forward({silence, zeros});
    return value.isTensor() && value.toTensor().defined() &&
           value.toTensor().size(2) == silence.size(2);
  } catch (const std::exception &) {
    return false;
  }
}

/**
 * @brief Forwards silence plus a control tensor of width @p condDim.
 * @return True when the module accepts that control width.
 */
bool probeConditioningWidth(torch::jit::Module &module,
                            const torch::Tensor &silence, int condDim) {
  const auto width = std::max(1, condDim);
  try {
    const auto timed =
        torch::zeros({1, width, silence.size(2)}, silence.options());
    if (module.forward({silence, timed}).isTensor())
      return true;
  } catch (const std::exception &) {
  }
  const auto zeros = torch::zeros({1, width, 1}, silence.options());
  const auto value = module.forward({silence, zeros});
  return value.isTensor();
}

/**
 * @brief Finds a control width the scripted module will accept.
 * @param preferred Width from train metadata, or 0 when unknown.
 */
int detectConditioningDim(torch::jit::Module &module,
                          const torch::Tensor &silence, int preferred) {
  if (preferred > 0) {
    try {
      if (probeConditioningWidth(module, silence, preferred))
        return preferred;
    } catch (const std::exception &) {
    }
  }
  constexpr int candidates[] = {2, 1, 4, 8, 16};
  for (const auto width : candidates) {
    if (width == preferred)
      continue;
    try {
      if (probeConditioningWidth(module, silence, width))
        return width;
    } catch (const std::exception &) {
    }
  }
  return std::max(1, preferred);
}

/**
 * @class TorchScriptKernel
 * @brief Runtime-local executor for one validated frozen artifact.
 */
class TorchScriptKernel final : public openyourbox::dsp::FrozenBlackBoxKernel {
public:
  /**
   * @brief Adopts a runtime-local inference module.
   * @param moduleToAdopt Loaded and evaluated TorchScript module.
   * @param conditioned True when the module expects an (audio, cond) pair.
   * @param modelChannels Channel count the module was traced with.
   * @param condDim Control width the module was traced with.
   */
  TorchScriptKernel(torch::jit::Module moduleToAdopt, bool conditioned,
                    int modelChannels, int condDim)
      : module(std::move(moduleToAdopt)), acceptsConditioning(conditioned),
        inputChannels(std::max(1, modelChannels)),
        conditioningDim(std::max(1, condDim)) {
    module.eval();
    if (acceptsConditioning) {
      const auto probe = torch::zeros(
          {1, inputChannels, 32},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      acceptsTimeVaryingConditioning =
          probeTimeVaryingConditioning(module, probe, conditioningDim);
    }
    encodeDecode = module.find_method("encode").has_value() &&
                   module.find_method("decode").has_value();
    try {
      if (module.hasattr("latent_mean"))
        latentMean = module.attr("latent_mean").toTensor().contiguous();
      if (module.hasattr("latent_pca"))
        latentPca = module.attr("latent_pca").toTensor().contiguous();
      if (module.hasattr("cumulative_variance"))
        cumulativeVariance =
            module.attr("cumulative_variance").toTensor().contiguous();
    } catch (const std::exception &) {
    }
  }

  /** @brief Executes one frozen inference call. */
  torch::Tensor forward(const torch::Tensor &input) override {
    torch::InferenceMode inferenceGuard;
    if (encodeDecode) {
      auto latent = encode(input);
      if (latent.defined())
        return decode(latent);
    }
    if (!acceptsConditioning)
      return module.forward({input}).toTensor();
    auto audio = matchChannelCount(input, inputChannels);
    auto zeros = normalizeConditioning({}, audio.size(0), conditioningDim,
                                       audio.options(),
                                       condTimeLength(audio.size(2)));
    return matchChannelCount(module.forward({audio, zeros}).toTensor(),
                             static_cast<int>(input.size(1)));
  }

  /** @brief Executes frozen inference with live conditioning. */
  torch::Tensor forwardWithConditioning(
      const torch::Tensor &input, const torch::Tensor &conditioning) override {
    torch::InferenceMode inferenceGuard;
    if (encodeDecode)
      return forward(input);
    if (!acceptsConditioning)
      return forward(input);
    auto audio = matchChannelCount(input, inputChannels);
    auto cond = normalizeConditioning(conditioning, audio.size(0),
                                      conditioningDim, audio.options(),
                                      condTimeLength(audio.size(2)));
    return matchChannelCount(module.forward({audio, cond}).toTensor(),
                             static_cast<int>(input.size(1)));
  }

  torch::Tensor encode(const torch::Tensor &input) override {
    torch::InferenceMode inferenceGuard;
    if (!encodeDecode)
      return {};
    try {
      return module.get_method("encode")({input}).toTensor();
    } catch (const std::exception &) {
      return {};
    }
  }

  torch::Tensor decode(const torch::Tensor &latent) override {
    torch::InferenceMode inferenceGuard;
    if (!encodeDecode)
      return {};
    try {
      return module.get_method("decode")({latent}).toTensor();
    } catch (const std::exception &) {
      return {};
    }
  }

  bool hasEncodeDecode() const noexcept override { return encodeDecode; }

  torch::Tensor compactnessMean() const override { return latentMean; }
  torch::Tensor compactnessPca() const override { return latentPca; }
  torch::Tensor compactnessCumulative() const override {
    return cumulativeVariance;
  }

private:
  /**
   * @brief Returns the control time length the artifact will accept.
   * @param audioSamples Time length of the audio argument.
   * @return @p audioSamples when the module is time-varying, otherwise 1.
   */
  [[nodiscard]] std::int64_t
  condTimeLength(std::int64_t audioSamples) const noexcept {
    return acceptsTimeVaryingConditioning ? std::max<std::int64_t>(1, audioSamples)
                                          : 1;
  }

  /** @brief Runtime-local module never mutated after construction. */
  torch::jit::Module module;
  /** @brief True when the scripted module accepts a conditioning argument. */
  bool acceptsConditioning = false;
  /** @brief True when `cond` may share the audio time dimension. */
  bool acceptsTimeVaryingConditioning = false;
  /** @brief Channel count the module was traced against. */
  int inputChannels = 1;
  /** @brief Flattened control width the module was traced against. */
  int conditioningDim = 2;
  /** @brief True when encode and decode methods exist. */
  bool encodeDecode = false;
  /** @brief Optional compactness mean loaded from the artifact. */
  torch::Tensor latentMean;
  /** @brief Optional compactness PCA loaded from the artifact. */
  torch::Tensor latentPca;
  /** @brief Optional cumulative variance loaded from the artifact. */
  torch::Tensor cumulativeVariance;
};
} // namespace

namespace openyourbox::dsp {
std::shared_ptr<const TorchScriptBlackBoxFactory>
TorchScriptBlackBoxFactory::load(const std::string &artifactPath,
                                 int inputChannels,
                                 std::uint64_t receptiveFieldSamples,
                                 std::string &error,
                                 bool requireSilencePreservation,
                                 bool acceptsConditioning, int condDim) {
  error.clear();
  if (artifactPath.empty() || inputChannels < 1 || receptiveFieldSamples < 1) {
    error = "Frozen artifact path, channels, and receptive field must be valid";
    return {};
  }

  try {
    auto module = torch::jit::load(artifactPath, torch::kCPU);
    module.eval();
    torch::InferenceMode inferenceGuard;
    const auto silence = torch::zeros(
        {1, inputChannels, 256},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    c10::IValue value;
    auto resolvedCondDim = std::max(1, condDim);
    auto conditioned = acceptsConditioning;
    const auto forwardUnconditioned = [&]() -> bool {
      try {
        value = module.forward({silence});
        return value.isTensor();
      } catch (const std::exception &) {
        return false;
      }
    };
    const auto forwardConditioned = [&]() {
      resolvedCondDim = detectConditioningDim(module, silence, condDim);
      try {
        const auto timed = torch::zeros(
            {1, resolvedCondDim, silence.size(2)}, silence.options());
        value = module.forward({silence, timed});
      } catch (const std::exception &) {
        const auto zeros =
            torch::zeros({1, resolvedCondDim, 1}, silence.options());
        value = module.forward({silence, zeros});
      }
      conditioned = true;
    };
    if (acceptsConditioning || !forwardUnconditioned())
      forwardConditioned();
    if (!value.isTensor()) {
      error = "Frozen artifact did not return an audio tensor";
      return {};
    }
    const auto output = value.toTensor();
    if (!output.defined() || output.device().type() != torch::kCPU ||
        output.scalar_type() != torch::kFloat32 || output.dim() != 3 ||
        output.size(0) != 1 || output.size(1) < 1 ||
        output.size(1) > std::numeric_limits<int>::max() ||
        output.size(2) != silence.size(2)) {
      error = "Frozen artifact returned an invalid audio tensor shape";
      return {};
    }

    std::uint64_t parameters = 0;
    for (const auto &parameter : module.parameters())
      parameters += static_cast<std::uint64_t>(parameter.numel());
    const auto preserves =
        torch::count_nonzero(output).item<std::int64_t>() == 0;
    if (requireSilencePreservation && !preserves) {
      error = "Frozen artifact does not preserve digital silence";
      return {};
    }

    const auto encodeDecode =
        module.find_method("encode").has_value() &&
        module.find_method("decode").has_value();

    return std::shared_ptr<const TorchScriptBlackBoxFactory>(
        new TorchScriptBlackBoxFactory(
            artifactPath, inputChannels, static_cast<int>(output.size(1)),
            receptiveFieldSamples, parameters, preserves, conditioned,
            resolvedCondDim, encodeDecode));
  } catch (const std::exception &exception) {
    error = exception.what();
    return {};
  }
}

int TorchScriptBlackBoxFactory::getInputChannels() const noexcept {
  return validatedInputChannels;
}

int TorchScriptBlackBoxFactory::getOutputChannels() const noexcept {
  return validatedOutputChannels;
}

std::uint64_t TorchScriptBlackBoxFactory::getReceptiveField() const noexcept {
  return receptiveField;
}

std::uint64_t TorchScriptBlackBoxFactory::getParameterCount() const noexcept {
  return parameterCount;
}

bool TorchScriptBlackBoxFactory::preservesSilence() const noexcept {
  return silencePreserving;
}

std::unique_ptr<FrozenBlackBoxKernel>
TorchScriptBlackBoxFactory::createKernel() const {
  try {
    auto module = torch::jit::load(artifactPath, torch::kCPU);
    return std::make_unique<TorchScriptKernel>(
        std::move(module), conditioned, validatedInputChannels,
        conditioningDim);
  } catch (...) {
    return {};
  }
}

bool TorchScriptBlackBoxFactory::hasEncodeDecode() const noexcept {
  return encodeDecode;
}

const std::string &
TorchScriptBlackBoxFactory::getArtifactPath() const noexcept {
  return artifactPath;
}

TorchScriptBlackBoxFactory::TorchScriptBlackBoxFactory(std::string path,
                                                       int inputs, int outputs,
                                                       std::uint64_t field,
                                                       std::uint64_t parameters,
                                                       bool silence,
                                                       bool conditionedModule,
                                                       int condDim,
                                                       bool encodeDecodeMethods)
    : artifactPath(std::move(path)), validatedInputChannels(inputs),
      validatedOutputChannels(outputs), receptiveField(field),
      parameterCount(parameters), silencePreserving(silence),
      conditioned(conditionedModule), conditioningDim(std::max(1, condDim)),
      encodeDecode(encodeDecodeMethods) {}
} // namespace openyourbox::dsp
