#include "RaveLayouts.h"

#include <algorithm>
#include <cmath>

namespace openyourbox::graph {
namespace {
/** @brief Activation enum index for LeakyReLU in the palette. */
constexpr int leakyReluActivation = 3;

int readInt(const GraphNode &node, const char *key, int fallback) {
  for (const auto &property : node.properties) {
    if (property.key == key)
      return property.value;
  }
  return fallback;
}

std::int32_t audioPin(const GraphNode &node, bool input) {
  const auto &pins = input ? node.inputs : node.outputs;
  for (const auto &pin : pins) {
    if (!isControlInputPin(pin))
      return pin.id;
  }
  return pins.empty() ? 0 : pins.front().id;
}

void configureDownConv(NodeGraph &graph, std::int32_t id, int stride, int channels,
                       int kernelSize) {
  graph.setProperty(id, "stride", stride);
  graph.setProperty(id, "channels", channels);
  graph.setProperty(id, "kernel_size", kernelSize);
  graph.setProperty(id, "dilation", 1);
}

void configureUpConv(NodeGraph &graph, std::int32_t id, int stride, int channels,
                     int kernelSize) {
  graph.setProperty(id, "stride", stride);
  graph.setProperty(id, "channels", channels);
  graph.setProperty(id, "kernel_size", kernelSize);
  graph.setProperty(id, "dilation", 1);
}
} // namespace

std::uint64_t raveNodeDelaySamples(const GraphNode &node) {
  switch (node.type) {
  case NodeType::pqmfAnalysis:
  case NodeType::pqmfSynthesis: {
    const auto nBand = std::max(2, readInt(node, "n_band", defaultPqmfBands));
    return static_cast<std::uint64_t>(4 * nBand);
  }
  case NodeType::convolution:
  case NodeType::rateConv:
  case NodeType::convTranspose: {
    const auto kernel = std::max(1, readInt(node, "kernel_size", 3));
    const auto dilation = std::max(1, readInt(node, "dilation", 1));
    const auto stride = std::max(1, readInt(node, "stride", 1));
    return static_cast<std::uint64_t>(kernel - 1) * static_cast<std::uint64_t>(dilation) *
           static_cast<std::uint64_t>(stride);
  }
  case NodeType::variationalBottleneck: {
    const auto kernel =
        std::max(1, readInt(node, "kernel_size", defaultBottleneckKernelSize));
    return static_cast<std::uint64_t>(kernel - 1);
  }
  case NodeType::noiseSynthesizer:
  case NodeType::batchNorm:
  case NodeType::activation:
    return 0;
  default:
    return 0;
  }
}

std::string insertRaveLayout(NodeGraph &graph, RaveLayoutId layout,
                             int channelWidth) {
  if (channelWidth != 1 && channelWidth != 2)
    return "RAVE layout channel width must be mono (1) or stereo (2)";
  if (graph.getViewport().focusedGroupId.has_value())
    return "RAVE layouts can only be inserted on the root canvas "
           "(open Graph from the breadcrumb first)";

  graph.ensureFixedHostIo();
  const GraphNode *input = nullptr;
  const GraphNode *output = nullptr;
  for (const auto &node : graph.getNodes()) {
    if (node.type == NodeType::audioInput)
      input = &node;
    if (node.type == NodeType::audioOutput)
      output = &node;
  }
  if (input == nullptr || output == nullptr)
    return "Host Audio Input and Output are required before inserting a layout";

  const auto inputId = input->id;
  const auto outputId = output->id;
  if (!graph.setProperty(inputId, "channels",
                         hostIoChoiceFromChannels(channelWidth)) ||
      !graph.setProperty(outputId, "channels",
                         hostIoChoiceFromChannels(channelWidth)))
    return "Could not set Audio Input/Output to the chosen mono/stereo width";
  const auto originX = input->position.x + 220.0f;
  const auto originY = input->position.y - 40.0f;
  const auto latest = layout == RaveLayoutId::latestContinuous;
  float x = originX;
  const auto place = [&](NodeType type) {
    const auto id = graph.addNode(type, {x, originY});
    x += 190.0f;
    return id;
  };
  const auto placeLeakyRelu = [&]() {
    const auto id = place(NodeType::activation);
    graph.setProperty(id, "activation", leakyReluActivation);
    return id;
  };

  const auto analysisId = place(NodeType::pqmfAnalysis);
  graph.setProperty(analysisId, "n_band", defaultPqmfBands);

  const int strides[4] = {4, 4, 4, 2};
  const int encoderChannels[4] = {128, 256, 512, defaultLatentSize};
  std::vector<std::int32_t> encoder;
  encoder.reserve(16);
  for (int stage = 0; stage < 4; ++stage) {
    if (latest) {
      const auto tcnId = place(NodeType::tcn);
      graph.setProperty(tcnId, "channels", 64);
      graph.setProperty(tcnId, "depth", 2);
      graph.setProperty(tcnId, "residual", 1);
      graph.setProperty(tcnId, "activation", leakyReluActivation);
      encoder.push_back(tcnId);
    }
    if (!latest) {
      const auto batchNormId = place(NodeType::batchNorm);
      encoder.push_back(batchNormId);
    }
    encoder.push_back(placeLeakyRelu());
    const auto convId = place(NodeType::convolution);
    const auto kernel = stage == 0 ? 7 : 2 * strides[stage] + 1;
    configureDownConv(graph, convId, strides[stage],
                      latest ? (stage < 3 ? 64 : defaultLatentSize)
                             : encoderChannels[stage],
                      kernel);
    encoder.push_back(convId);
  }

  const auto bottleneckId = place(NodeType::variationalBottleneck);
  graph.setProperty(bottleneckId, "latent_size", defaultLatentSize);
  graph.setProperty(bottleneckId, "kernel_size", defaultBottleneckKernelSize);

  std::vector<std::int32_t> decoder;
  const int upStrides[4] = {2, 4, 4, 4};
  const int decoderChannels[4] = {defaultLatentSize, 512, 256, 128};
  for (int stage = 0; stage < 4; ++stage) {
    decoder.push_back(placeLeakyRelu());
    const auto convId = place(NodeType::convTranspose);
    const auto kernel = 2 * upStrides[stage];
    configureUpConv(graph, convId, upStrides[stage],
                    latest ? (stage == 3 ? channelWidth * defaultPqmfBands : 64)
                           : (stage == 3 ? channelWidth * defaultPqmfBands
                                         : decoderChannels[stage]),
                    kernel);
    decoder.push_back(convId);
    if (latest) {
      const auto tcnId = place(NodeType::tcn);
      graph.setProperty(tcnId, "channels", 64);
      graph.setProperty(tcnId, "depth", 2);
      graph.setProperty(tcnId, "residual", 1);
      graph.setProperty(tcnId, "activation", leakyReluActivation);
      decoder.push_back(tcnId);
    }
  }

  const auto noiseId = latest ? 0 : place(NodeType::noiseSynthesizer);
  const auto mergeId = latest ? 0 : place(NodeType::merge);
  if (mergeId != 0)
    graph.setProperty(mergeId, "mode", 0);
  const auto synthesisId = place(NodeType::pqmfSynthesis);
  graph.setProperty(synthesisId, "n_band", defaultPqmfBands);

  const auto connect = [&](std::int32_t sourceId, std::int32_t destId) {
    const auto *source = graph.findNode(sourceId);
    const auto *dest = graph.findNode(destId);
    if (source == nullptr || dest == nullptr)
      return;
    graph.connect(audioPin(*source, false), audioPin(*dest, true));
  };

  connect(inputId, analysisId);
  std::int32_t previous = analysisId;
  for (const auto id : encoder) {
    connect(previous, id);
    previous = id;
  }
  connect(previous, bottleneckId);
  previous = bottleneckId;
  for (const auto id : decoder) {
    connect(previous, id);
    previous = id;
  }
  if (noiseId != 0 && mergeId != 0) {
    const auto *merge = graph.findNode(mergeId);
    const auto *decoderNode = graph.findNode(previous);
    const auto *noise = graph.findNode(noiseId);
    if (merge != nullptr && decoderNode != nullptr && noise != nullptr &&
        merge->inputs.size() >= 2) {
      graph.connect(audioPin(*decoderNode, false), merge->inputs[0].id);
      graph.connect(audioPin(*decoderNode, false), audioPin(*noise, true));
      graph.connect(audioPin(*noise, false), merge->inputs[1].id);
      previous = mergeId;
    }
  }
  connect(previous, synthesisId);
  connect(synthesisId, outputId);
  return {};
}
} // namespace openyourbox::graph
