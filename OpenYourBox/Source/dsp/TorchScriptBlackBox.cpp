#include "TorchScriptBlackBox.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

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
 * @brief Concatenates two `[batch, channels, time]` tensors along time.
 * @param head Leading tensor, or undefined / empty.
 * @param tail Trailing tensor, or undefined / empty.
 * @return Defined concatenation, or an undefined tensor when both are empty.
 */
torch::Tensor concatTime(const torch::Tensor &head, const torch::Tensor &tail) {
  const bool haveHead = head.defined() && head.dim() == 3 && head.size(2) > 0;
  const bool haveTail = tail.defined() && tail.dim() == 3 && tail.size(2) > 0;
  if (!haveHead)
    return haveTail ? tail : torch::Tensor();
  if (!haveTail)
    return head;
  return torch::cat({head, tail}, 2);
}

/**
 * @brief Reads a RAVE-style `_b<block>_` streaming size from a file name.
 * @param path Artifact path, for example `guitar_iil_b2048_r48000_z16.ts`.
 * @return Positive block size, or 0 when the name does not advertise one.
 */
int streamingBlockHintFromPath(const std::string &path) {
  const auto slash = path.find_last_of("/\\");
  const auto name =
      slash == std::string::npos ? path : path.substr(slash + 1);
  const auto marker = name.find("_b");
  if (marker == std::string::npos || marker + 2 >= name.size())
    return 0;
  if (name[marker + 2] < '0' || name[marker + 2] > '9')
    return 0;
  int value = 0;
  std::size_t index = marker + 2;
  while (index < name.size() && name[index] >= '0' && name[index] <= '9') {
    value = value * 10 + (name[index] - '0');
    ++index;
  }
  if (value < 1 || index >= name.size() || name[index] != '_')
    return 0;
  return value;
}

/**
 * @brief Ordered probe lengths: 256 first, then a filename hint, then hops.
 * @param artifactPath Path used to parse an optional `_b<n>_` hint.
 */
std::vector<int> buildProbeLengths(const std::string &artifactPath) {
  std::vector<int> lengths = {256};
  const auto hint = streamingBlockHintFromPath(artifactPath);
  if (hint > 0 && hint != 256)
    lengths.push_back(hint);
  for (int candidate : {512, 1024, 2048, 4096, 8192, 16384}) {
    if (std::find(lengths.begin(), lengths.end(), candidate) == lengths.end())
      lengths.push_back(candidate);
  }
  return lengths;
}

/**
 * @brief User-facing `forward` arity excluding `self`.
 */
struct ForwardArity {
  /** @brief Arguments without defaults, or -1 when unknown. */
  int required = -1;
  /** @brief Total user arguments including optionals, or -1 when unknown. */
  int total = -1;
};

/**
 * @brief Reads `forward` schema (or `num_inputs`) to decide cond probing.
 * @param module Loaded TorchScript module.
 */
ForwardArity inspectForwardArity(torch::jit::Module &module) {
  ForwardArity arity;
  try {
    const auto &schema = module.get_method("forward").function().getSchema();
    int required = 0;
    int total = 0;
    for (const auto &argument : schema.arguments()) {
      if (argument.name() == "self")
        continue;
      ++total;
      if (!argument.default_value().has_value())
        ++required;
    }
    arity.required = required;
    arity.total = total;
    return arity;
  } catch (const std::exception &) {
  }
  try {
    const auto inputs = module.get_method("forward").num_inputs();
    const auto user =
        inputs > 0 ? static_cast<int>(inputs) - 1 : 0;
    arity.required = user;
    arity.total = user;
  } catch (const std::exception &) {
  }
  return arity;
}

/**
 * @brief True when a probe `IValue` is a CPU float `[1, C, T]` audio tensor.
 * @param value Module forward result.
 */
bool isValidAudioTensor(const c10::IValue &value) {
  if (!value.isTensor())
    return false;
  const auto output = value.toTensor();
  return output.defined() && output.device().type() == torch::kCPU &&
         output.scalar_type() == torch::kFloat32 && output.dim() == 3 &&
         output.size(0) == 1 && output.size(1) >= 1 &&
         output.size(1) <= std::numeric_limits<int>::max() &&
         output.size(2) >= 1;
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
  try {
    const auto zeros = torch::zeros({1, width, 1}, silence.options());
    const auto value = module.forward({silence, zeros});
    return value.isTensor();
  } catch (const std::exception &) {
    return false;
  }
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
   * @param hopSamples Time length used by a successful load probe.
   * @param fixedBlock True when shorter probes failed and audio must be
   *   accumulated to @p hopSamples before each `forward`.
   */
  TorchScriptKernel(torch::jit::Module moduleToAdopt, bool conditioned,
                    int modelChannels, int condDim, int hopSamples,
                    bool fixedBlock)
      : module(std::move(moduleToAdopt)), acceptsConditioning(conditioned),
        inputChannels(std::max(1, modelChannels)),
        conditioningDim(std::max(1, condDim)),
        inferenceBlock(std::max(0, hopSamples)),
        requiresFixedBlock(fixedBlock && inferenceBlock > 1) {
    module.eval();
    if (acceptsConditioning) {
      const auto probeLen =
          requiresFixedBlock ? std::max(1, inferenceBlock) : 32;
      const auto probe = torch::zeros(
          {1, inputChannels, probeLen},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      acceptsTimeVaryingConditioning =
          probeTimeVaryingConditioning(module, probe, conditioningDim);
    }
    encodeDecode = module.find_method("encode").has_value() &&
                   module.find_method("decode").has_value();
    hasEncodeDistribution =
        module.find_method("encode_distribution").has_value();
    try {
      if (module.hasattr("latent_mean"))
        latentMean = module.attr("latent_mean").toTensor().contiguous();
      if (module.hasattr("latent_pca"))
        latentPca = module.attr("latent_pca").toTensor().contiguous();
      if (module.hasattr("cumulative_variance"))
        cumulativeVariance =
            module.attr("cumulative_variance").toTensor().contiguous();
      if (module.hasattr("compactness_ready"))
        compactnessBuffersReady =
            module.attr("compactness_ready").toTensor().item<float>() > 0.5f;
    } catch (const std::exception &) {
    }
  }

  /** @brief Executes one frozen inference call. */
  torch::Tensor forward(const torch::Tensor &input) override {
    torch::InferenceMode inferenceGuard;
    return emitBlock(input, {});
  }

  /** @brief Executes frozen inference with live conditioning. */
  torch::Tensor forwardWithConditioning(
      const torch::Tensor &input, const torch::Tensor &conditioning) override {
    torch::InferenceMode inferenceGuard;
    return emitBlock(input, conditioning);
  }

  torch::Tensor encode(const torch::Tensor &input) override {
    torch::InferenceMode inferenceGuard;
    if (!encodeDecode)
      return {};
    if (requiresFixedBlock && input.size(2) != inferenceBlock)
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

  bool encodeDistribution(const torch::Tensor &input, torch::Tensor &mean,
                          torch::Tensor &std) override {
    torch::InferenceMode inferenceGuard;
    mean = torch::Tensor{};
    std = torch::Tensor{};
    if (!encodeDecode)
      return false;
    if (requiresFixedBlock && input.size(2) != inferenceBlock)
      return false;
    try {
      if (hasEncodeDistribution) {
        auto packed = module.get_method("encode_distribution")({input});
        if (packed.isTensor()) {
          auto tensor = packed.toTensor();
          if (tensor.defined() && tensor.dim() == 3 && tensor.size(1) >= 2 &&
              tensor.size(1) % 2 == 0) {
            const auto half = tensor.size(1) / 2;
            mean = tensor.narrow(1, 0, half);
            std = tensor.narrow(1, half, half);
            return true;
          }
        } else if (packed.isTuple()) {
          const auto tuple = packed.toTuple();
          if (tuple->elements().size() >= 2) {
            mean = tuple->elements()[0].toTensor();
            std = tuple->elements()[1].toTensor();
            return mean.defined();
          }
        }
      }
    } catch (const std::exception &) {
    }
    mean = encode(input);
    return mean.defined();
  }

  bool hasEncodeDecode() const noexcept override { return encodeDecode; }

  torch::Tensor compactnessMean() const override { return latentMean; }
  torch::Tensor compactnessPca() const override { return latentPca; }
  torch::Tensor compactnessCumulative() const override {
    return cumulativeVariance;
  }

  bool compactnessReady() const noexcept override {
    return compactnessBuffersReady;
  }

private:
  /**
   * @brief Runs one native-length module call (no hop accumulation).
   * @param input Audio chunk, possibly already hop-sized.
   * @param conditioning Live control, or undefined.
   */
  torch::Tensor invoke(const torch::Tensor &input,
                       const torch::Tensor &conditioning) {
    if (encodeDecode && !requiresFixedBlock) {
      auto latent = encode(input);
      if (latent.defined())
        return decode(latent);
    }
    auto audio = matchChannelCount(input, inputChannels);
    if (!acceptsConditioning)
      return matchChannelCount(module.forward({audio}).toTensor(),
                               static_cast<int>(input.size(1)));
    auto cond = normalizeConditioning(conditioning, audio.size(0),
                                      conditioningDim, audio.options(),
                                      condTimeLength(audio.size(2)));
    return matchChannelCount(module.forward({audio, cond}).toTensor(),
                             static_cast<int>(input.size(1)));
  }

  /**
   * @brief Forwards @p input, accumulating to the probed hop when required.
   * @param input Current audio block.
   * @param conditioning Live control for this block, or undefined.
   * @return Tensor with the same time length as @p input (leading silence
   *   until one hop of output is queued).
   */
  torch::Tensor emitBlock(const torch::Tensor &input,
                          const torch::Tensor &conditioning) {
    if (!requiresFixedBlock)
      return invoke(input, conditioning);
    const auto hop = static_cast<std::int64_t>(inferenceBlock);
    const auto emit = input.size(2);
    pendingAudio = concatTime(pendingAudio, input);
    if (conditioning.defined())
      pendingCond = concatTime(pendingCond, conditioning);
    std::vector<torch::Tensor> chunks;
    while (pendingAudio.defined() && pendingAudio.size(2) >= hop) {
      auto audioChunk = pendingAudio.narrow(2, 0, hop).contiguous();
      torch::Tensor condChunk;
      if (pendingCond.defined() && pendingCond.size(2) >= hop)
        condChunk = pendingCond.narrow(2, 0, hop).contiguous();
      chunks.push_back(invoke(audioChunk, condChunk));
      const auto remain = pendingAudio.size(2) - hop;
      pendingAudio =
          remain > 0 ? pendingAudio.narrow(2, hop, remain).contiguous()
                     : torch::Tensor();
      if (pendingCond.defined()) {
        const auto condRemain =
            pendingCond.size(2) > hop ? pendingCond.size(2) - hop : 0;
        pendingCond =
            condRemain > 0
                ? pendingCond.narrow(2, hop, condRemain).contiguous()
                : torch::Tensor();
      }
    }
    if (!chunks.empty())
      pendingOutput = concatTime(pendingOutput, torch::cat(chunks, 2));
    const auto outChannels =
        pendingOutput.defined() ? pendingOutput.size(1) : input.size(1);
    if (pendingOutput.defined() && pendingOutput.size(2) >= emit) {
      auto out = pendingOutput.narrow(2, 0, emit).contiguous();
      const auto rest = pendingOutput.size(2) - emit;
      pendingOutput =
          rest > 0 ? pendingOutput.narrow(2, emit, rest).contiguous()
                   : torch::Tensor();
      return out;
    }
    auto silence =
        torch::zeros({input.size(0), outChannels, emit}, input.options());
    if (pendingOutput.defined() && pendingOutput.size(2) > 0) {
      silence.narrow(2, 0, pendingOutput.size(2)).copy_(pendingOutput);
      pendingOutput = torch::Tensor();
    }
    return silence;
  }

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
  /** @brief Probe time length that succeeded at load. */
  int inferenceBlock = 256;
  /** @brief True when live audio must be accumulated to `inferenceBlock`. */
  bool requiresFixedBlock = false;
  /** @brief Audio samples waiting for a full hop. */
  torch::Tensor pendingAudio;
  /** @brief Control samples waiting for a full hop. */
  torch::Tensor pendingCond;
  /** @brief Produced audio not yet emitted to the host block. */
  torch::Tensor pendingOutput;
  /** @brief True when encode and decode methods exist. */
  bool encodeDecode = false;
  /** @brief Cached at load; never call `find_method` on the audio thread. */
  bool hasEncodeDistribution = false;
  /** @brief Optional compactness mean loaded from the artifact. */
  torch::Tensor latentMean;
  /** @brief Optional compactness PCA loaded from the artifact. */
  torch::Tensor latentPca;
  /** @brief Optional cumulative variance loaded from the artifact. */
  torch::Tensor cumulativeVariance;
  /** @brief True when the artifact marked compactness PCA as ready. */
  bool compactnessBuffersReady = false;
};
} // namespace

namespace openyourbox::dsp {
std::shared_ptr<const TorchScriptBlackBoxFactory>
TorchScriptBlackBoxFactory::load(const std::string &artifactPath,
                                 int inputChannels,
                                 std::uint64_t receptiveFieldSamples,
                                 std::string &error,
                                 bool requireSilencePreservation,
                                 bool acceptsConditioning, int condDim,
                                 double hostSampleRate) {
  error.clear();
  if (artifactPath.empty() || inputChannels < 1 || receptiveFieldSamples < 1) {
    error = "Frozen artifact path, channels, and receptive field must be valid";
    return {};
  }

  try {
    auto module = torch::jit::load(artifactPath, torch::kCPU);
    module.eval();
    torch::InferenceMode inferenceGuard;
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    c10::IValue value;
    auto resolvedCondDim = std::max(1, condDim);
    auto conditioned = false;
    std::string lastError;
    const auto arity = inspectForwardArity(module);
    const bool mayUnconditioned = arity.required <= 1;
    const bool mayConditioned = arity.total != 1;
    int hopSamples = 256;
    bool requiresFixedBlock = false;
    torch::Tensor silence;
    auto tryUnconditioned = [&](const torch::Tensor &probe) -> bool {
      try {
        value = module.forward({probe});
        if (isValidAudioTensor(value)) {
          conditioned = false;
          return true;
        }
        lastError = "Frozen artifact returned an invalid audio tensor shape";
      } catch (const std::exception &exception) {
        lastError = exception.what();
      }
      return false;
    };
    auto tryConditioned = [&](const torch::Tensor &probe) -> bool {
      try {
        resolvedCondDim = detectConditioningDim(module, probe, condDim);
        try {
          const auto timed = torch::zeros(
              {1, resolvedCondDim, probe.size(2)}, probe.options());
          value = module.forward({probe, timed});
        } catch (const std::exception &) {
          const auto zeros =
              torch::zeros({1, resolvedCondDim, 1}, probe.options());
          value = module.forward({probe, zeros});
        }
        if (isValidAudioTensor(value)) {
          conditioned = true;
          return true;
        }
        lastError = "Frozen artifact returned an invalid audio tensor shape";
      } catch (const std::exception &exception) {
        lastError = exception.what();
      }
      return false;
    };
    bool probed = false;
    bool failedShorterLength = false;
    for (const auto length : buildProbeLengths(artifactPath)) {
      silence = torch::zeros({1, inputChannels, length}, options);
      bool ok = false;
      if (acceptsConditioning && mayConditioned)
        ok = tryConditioned(silence) ||
             (mayUnconditioned && tryUnconditioned(silence));
      else
        ok = (mayUnconditioned && tryUnconditioned(silence)) ||
             (mayConditioned && tryConditioned(silence));
      if (ok) {
        hopSamples = length;
        requiresFixedBlock = failedShorterLength;
        probed = true;
        break;
      }
      failedShorterLength = true;
    }
    if (!probed || !isValidAudioTensor(value)) {
      error = lastError.empty() ? "Frozen artifact did not return an audio tensor"
                                : lastError;
      return {};
    }
    const auto output = value.toTensor();
    // Rate-changing RAVE graphs (PQMF, strided conv, conv-transpose) may
    // emit a different time length than the probe. Live playback already
    // crops or left-pads with matchTimeLength. Streaming exports that reject
    // short probes are accumulated to hopSamples inside the kernel.

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
    int latentChannels = 0;
    int latentFrames = 1;
    if (encodeDecode) {
      bool encodeFailedShorterLength = false;
      for (const auto length : buildProbeLengths(artifactPath)) {
        try {
          auto encodeInput =
              torch::zeros({1, inputChannels, length}, options);
          auto encoded =
              module.get_method("encode")({encodeInput}).toTensor();
          if (encoded.defined() && encoded.dim() >= 2 &&
              encoded.size(1) > 0) {
            latentChannels = static_cast<int>(encoded.size(1));
            if (encoded.dim() >= 3 && encoded.size(2) > 0)
              latentFrames = static_cast<int>(encoded.size(2));
            if (encodeFailedShorterLength) {
              hopSamples = length;
              requiresFixedBlock = true;
            }
            break;
          }
        } catch (const std::exception &) {
        }
        encodeFailedShorterLength = true;
      }
    }

    bool compactnessReady = false;
    std::vector<float> latentMean;
    std::vector<float> latentPca;
    std::vector<float> cumulativeVariance;
    const auto copyAttr = [](torch::jit::Module &loaded, const char *name,
                             std::vector<float> &out) {
      if (!loaded.hasattr(name))
        return;
      try {
        auto tensor = loaded.attr(name).toTensor().contiguous().to(torch::kCPU).to(
            torch::kFloat32);
        if (!tensor.defined() || tensor.numel() < 1)
          return;
        const auto *data = tensor.data_ptr<float>();
        out.assign(data, data + tensor.numel());
      } catch (const std::exception &) {
      }
    };
    try {
      if (module.hasattr("compactness_ready"))
        compactnessReady =
            module.attr("compactness_ready").toTensor().item<float>() > 0.5f;
    } catch (const std::exception &) {
    }
    if (compactnessReady) {
      copyAttr(module, "latent_mean", latentMean);
      copyAttr(module, "latent_pca", latentPca);
      copyAttr(module, "cumulative_variance", cumulativeVariance);
    }

    std::string rateWarning;
    if (hostSampleRate > 0.0) {
      const auto readRate = [&module]() -> double {
        for (const char *name : {"sample_rate", "sr", "sampling_rate"}) {
          if (!module.hasattr(name))
            continue;
          try {
            auto attr = module.attr(name);
            if (attr.isDouble())
              return attr.toDouble();
            if (attr.isInt())
              return static_cast<double>(attr.toInt());
            if (attr.isTensor()) {
              auto tensor = attr.toTensor();
              if (tensor.defined() && tensor.numel() > 0)
                return static_cast<double>(
                    tensor.to(torch::kCPU).item<float>());
            }
          } catch (const std::exception &) {
          }
        }
        return 0.0;
      };
      const auto advertised = readRate();
      if (advertised > 0.0 &&
          std::abs(advertised - hostSampleRate) > 1.0) {
        rateWarning = "Checkpoint sample rate " +
                      std::to_string(static_cast<int>(advertised)) +
                      " Hz differs from the host rate";
      }
    }

    return std::shared_ptr<const TorchScriptBlackBoxFactory>(
        new TorchScriptBlackBoxFactory(
            artifactPath, inputChannels, static_cast<int>(output.size(1)),
            receptiveFieldSamples, parameters, preserves, conditioned,
            resolvedCondDim, encodeDecode, latentChannels, latentFrames,
            compactnessReady, std::move(latentMean), std::move(latentPca),
            std::move(cumulativeVariance), std::move(rateWarning), hopSamples,
            requiresFixedBlock));
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
        conditioningDim, inferenceBlockSamples, requiresFixedHop);
  } catch (...) {
    return {};
  }
}

bool TorchScriptBlackBoxFactory::hasEncodeDecode() const noexcept {
  return encodeDecode;
}

bool TorchScriptBlackBoxFactory::acceptsConditioning() const noexcept {
  return conditioned;
}

int TorchScriptBlackBoxFactory::getLatentChannels() const noexcept {
  return latentChannels;
}

int TorchScriptBlackBoxFactory::getInferenceBlockSamples() const noexcept {
  return inferenceBlockSamples;
}

int TorchScriptBlackBoxFactory::getLatentFramesPerBlock() const noexcept {
  return latentFramesPerBlock;
}

bool TorchScriptBlackBoxFactory::requiresFixedInferenceBlock() const noexcept {
  return requiresFixedHop;
}

bool TorchScriptBlackBoxFactory::compactnessReady() const noexcept {
  return compactnessBuffersReady;
}

const std::vector<float> &
TorchScriptBlackBoxFactory::compactnessMean() const {
  return latentMean;
}

const std::vector<float> &TorchScriptBlackBoxFactory::compactnessPca() const {
  return latentPca;
}

const std::vector<float> &
TorchScriptBlackBoxFactory::compactnessCumulative() const {
  return cumulativeVariance;
}

const std::string &
TorchScriptBlackBoxFactory::sampleRateWarning() const {
  return rateWarning;
}

const std::string &
TorchScriptBlackBoxFactory::getArtifactPath() const noexcept {
  return artifactPath;
}

TorchScriptBlackBoxFactory::TorchScriptBlackBoxFactory(
    std::string path, int inputs, int outputs, std::uint64_t field,
    std::uint64_t parameters, bool silence, bool conditionedModule, int condDim,
    bool encodeDecodeMethods, int latentWidth, int latentFrames,
    bool compactness,
    std::vector<float> mean, std::vector<float> pca,
    std::vector<float> cumulative, std::string warning, int hopSamples,
    bool fixedHop)
    : artifactPath(std::move(path)), validatedInputChannels(inputs),
      validatedOutputChannels(outputs), receptiveField(field),
      parameterCount(parameters), silencePreserving(silence),
      conditioned(conditionedModule), conditioningDim(std::max(1, condDim)),
      encodeDecode(encodeDecodeMethods),
      latentChannels(std::max(0, latentWidth)),
      latentFramesPerBlock(std::max(1, latentFrames)),
      compactnessBuffersReady(compactness), latentMean(std::move(mean)),
      latentPca(std::move(pca)), cumulativeVariance(std::move(cumulative)),
      rateWarning(std::move(warning)),
      inferenceBlockSamples(std::max(1, hopSamples)),
      requiresFixedHop(fixedHop && inferenceBlockSamples > 1) {}
} // namespace openyourbox::dsp
