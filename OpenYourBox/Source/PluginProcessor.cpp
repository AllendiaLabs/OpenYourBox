#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "capture/HostTransport.h"
#include "params/ParamIDs.h"
#include "params/ParamLayout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace {
constexpr std::array<const char *, 9> listenedParameterIDs{
    openyourbox::params::depth,       openyourbox::params::kernelSize,
    openyourbox::params::channels,    openyourbox::params::dilation,
    openyourbox::params::activation,  openyourbox::params::randomize,
    openyourbox::params::randomizeCC, openyourbox::params::globalSeed,
    openyourbox::params::dryWet};

constexpr std::uint64_t maximumRealtimeHistory = 1U << 20U;
constexpr int minimumPreparedBlockSize = 8192;
constexpr int modelCrossfadeSamples = 64;

bool sameConfiguration(
    const openyourbox::dsp::TCNConfiguration &left,
    const openyourbox::dsp::TCNConfiguration &right) noexcept {
  return left.depth == right.depth && left.kernelSize == right.kernelSize &&
         left.channels == right.channels && left.dilation == right.dilation &&
         left.inputChannels == right.inputChannels &&
         left.outputChannels == right.outputChannels &&
         left.activation == right.activation;
}
} // namespace

OpenYourBoxAudioProcessor::OpenYourBoxAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, openyourbox::params::stateType,
                 openyourbox::params::createParameterLayout())
#else
    : parameters(*this, nullptr, openyourbox::params::stateType,
                 openyourbox::params::createParameterLayout())
#endif
{
  for (const auto *identifier : listenedParameterIDs)
    parameters.addParameterListener(identifier, this);

  modelBuilder.setPublishCallback([this] {
    publishRuntime(modelBuilder.getPublishedModel());
    resetRandomizeParameter();
  });
  capturePairing.setMessageHandler([this](const juce::var &message) {
    handlePairingMessage(message);
  });
  startTimerHz(20);
}

OpenYourBoxAudioProcessor::~OpenYourBoxAudioProcessor() {
  stopTimer();
  cancelPendingUpdate();
  modelBuilder.setPublishCallback({});
  for (const auto *identifier : listenedParameterIDs)
    parameters.removeParameterListener(identifier, this);
}

const juce::String OpenYourBoxAudioProcessor::getName() const {
  return JucePlugin_Name;
}

bool OpenYourBoxAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
  return true;
#else
  return false;
#endif
}

bool OpenYourBoxAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
  return true;
#else
  return false;
#endif
}

bool OpenYourBoxAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
  return true;
#else
  return false;
#endif
}

double OpenYourBoxAudioProcessor::getTailLengthSeconds() const {
  const auto rate = currentSampleRate.load(std::memory_order_acquire);
  const auto field = getReceptiveFieldSamples();
  if (rate <= 0.0 || field <= 1)
    return 0.0;
  return static_cast<double>(field - 1) / rate;
}

int OpenYourBoxAudioProcessor::getNumPrograms() { return 1; }

int OpenYourBoxAudioProcessor::getCurrentProgram() { return 0; }

void OpenYourBoxAudioProcessor::setCurrentProgram(int index) {
  juce::ignoreUnused(index);
}

const juce::String OpenYourBoxAudioProcessor::getProgramName(int index) {
  juce::ignoreUnused(index);
  return "Default";
}

void OpenYourBoxAudioProcessor::changeProgramName(int index,
                                                 const juce::String &newName) {
  juce::ignoreUnused(index, newName);
}

void OpenYourBoxAudioProcessor::prepareToPlay(double sampleRate,
                                             int samplesPerBlock) {
  currentSampleRate.store(sampleRate, std::memory_order_release);
  constexpr auto dcBlockerCutoffHz = 20.0;
  dcBlockerCoefficient.store(
      static_cast<float>(
          std::exp(-juce::MathConstants<double>::twoPi * dcBlockerCutoffHz /
                   std::max(1.0, sampleRate))),
      std::memory_order_release);
  preparedBlockSize.store(std::max(samplesPerBlock, minimumPreparedBlockSize),
                          std::memory_order_release);
  const auto maximumBlock = preparedBlockSize.load(std::memory_order_acquire);
  const auto outputChannels = std::max(1, getTotalNumOutputChannels());
  graphWetBuffer.setSize(outputChannels, maximumBlock, false, false, true);
  previousGraphWetBuffer.setSize(outputChannels, maximumBlock, false, false,
                                 true);
  prepared.store(true, std::memory_order_release);

  dryWetSmoother.reset(sampleRate,
                       openyourbox::dsp::controlRampSecondsDefault);
  if (auto *mix = parameters.getRawParameterValue(openyourbox::params::dryWet))
    dryWetSmoother.setCurrentAndTargetValue(mix->load());

  auto snapshot = modelBuilder.getPublishedModel();
  const auto configuration = getRequestedConfiguration();
  if (snapshot == nullptr ||
      !sameConfiguration(snapshot->model->getConfiguration(), configuration)) {
    const auto seed = static_cast<std::uint64_t>(
        parameters.getRawParameterValue(openyourbox::params::globalSeed)
            ->load());
    const auto counter = randomizationCounter.load(std::memory_order_acquire);
    snapshot = modelBuilder.buildNow(configuration, seed + counter, counter);
  }
  publishRuntime(snapshot);
  requestGraphCompile();
}

void OpenYourBoxAudioProcessor::releaseResources() {
  prepared.store(false, std::memory_order_release);
  transportPlaying.store(false, std::memory_order_release);
  std::atomic_store_explicit(&publishedRuntime, std::shared_ptr<RuntimeState>{},
                             std::memory_order_release);
  if (auto graphRuntime = graphPublisher.getPublishedRuntime())
    graphRuntime->reset();
  activeRuntime.reset();
  previousRuntime.reset();
  activeGraphRuntime.reset();
  previousGraphRuntime.reset();
  graphWetBuffer.setSize(0, 0);
  previousGraphWetBuffer.setSize(0, 0);
  graphDcInput.fill(0.0f);
  graphDcOutput.fill(0.0f);
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool OpenYourBoxAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const {
#if JucePlugin_IsMidiEffect
  juce::ignoreUnused(layouts);
  return true;
#else
  const auto output = layouts.getMainOutputChannelSet();
  if (output != juce::AudioChannelSet::mono() &&
      output != juce::AudioChannelSet::stereo())
    return false;

#if !JucePlugin_IsSynth
  if (output != layouts.getMainInputChannelSet())
    return false;
#endif

  return true;
#endif
}
#endif

void OpenYourBoxAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                            juce::MidiBuffer &midiMessages) {
  juce::ScopedNoDenormals noDenormals;
  bool playing = false;
  if (auto *head = getPlayHead()) {
    if (const auto position = head->getPosition())
      playing = position->getIsPlaying();
    if (pendingTransportPlay.exchange(false, std::memory_order_acq_rel) &&
        !playing) {
      head->transportPlay(true);
      playing = true;
    }
  }
  transportPlaying.store(playing, std::memory_order_release);
  const auto inputChannels = getTotalNumInputChannels();
  const auto outputChannels = getTotalNumOutputChannels();
  const auto numSamples = buffer.getNumSamples();

  for (auto channel = inputChannels; channel < outputChannels; ++channel)
    buffer.clear(channel, 0, numSamples);

  const auto randomizeCC = juce::roundToInt(
      parameters.getRawParameterValue(openyourbox::params::randomizeCC)->load());
  for (const auto metadata : midiMessages) {
    const auto message = metadata.getMessage();
    if (message.isController() &&
        message.getControllerNumber() == randomizeCC &&
        message.getControllerValue() >= 64) {
      midiRandomizePending.store(true, std::memory_order_release);
      triggerAsyncUpdate();
    }
  }

  syncDryWetSmoother();

  const auto captureInput = inputCapture.isActive();
  if (captureInput) {
    const float *channels[2] = {
        inputChannels > 0 ? buffer.getReadPointer(0) : nullptr,
        inputChannels > 1 ? buffer.getReadPointer(1) : nullptr};
    inputCapture.pushInput(channels, inputChannels, numSamples);
  }

  if (captureBypass.load(std::memory_order_acquire)) {
    for (int channel = inputChannels; channel < outputChannels; ++channel)
      buffer.clear(channel, 0, numSamples);
    mixPreview(buffer, outputChannels, numSamples);
    return;
  }

  if (processLiveGraph(buffer, inputChannels, numSamples)) {
    mixPreview(buffer, outputChannels, numSamples);
    return;
  }

  graphDcInput.fill(0.0f);
  graphDcOutput.fill(0.0f);
  for (int sample = 0; sample < numSamples; ++sample) {
    const auto dryWet = dryWetSmoother.getNextValue();
    const auto dryGain = 1.0f - dryWet;
    for (int channel = 0; channel < outputChannels; ++channel)
      buffer.setSample(channel, sample,
                       buffer.getSample(channel, sample) * dryGain);
  }
  mixPreview(buffer, outputChannels, numSamples);
}

bool OpenYourBoxAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor *OpenYourBoxAudioProcessor::createEditor() {
  return new OpenYourBoxAudioProcessorEditor(*this);
}

void OpenYourBoxAudioProcessor::getStateInformation(
    juce::MemoryBlock &destData) {
  auto xml = parameters.copyState().createXml();
  const auto snapshot = modelBuilder.getPublishedModel();
  {
    const juce::ScopedLock lock(graphStateLock);
    if (persistedGraphState.isValid())
      xml->addChildElement(persistedGraphState.createXml().release());
  }

  if (snapshot != nullptr && snapshot->model != nullptr) {
    std::ostringstream stream(std::ios::binary);
    torch::serialize::OutputArchive archive;
    snapshot->model->save(archive);
    archive.save_to(stream);
    const auto bytes = stream.str();
    const juce::MemoryBlock weights(bytes.data(), bytes.size());

    xml->setAttribute("architectureHash",
                      juce::String::toHexString(static_cast<juce::int64>(
                          snapshot->model->getArchitectureHash())));
    xml->setAttribute("randomizationCounter",
                      juce::String(snapshot->randomizationCounter));
    xml->setAttribute("weights", weights.toBase64Encoding());
  }

  copyXmlToBinary(*xml, destData);
}

void OpenYourBoxAudioProcessor::setStateInformation(const void *data,
                                                   int sizeInBytes) {
  const auto xml = getXmlFromBinary(data, sizeInBytes);
  if (xml == nullptr || !xml->hasTagName(parameters.state.getType().toString()))
    return;

  restoringState.store(true, std::memory_order_release);
  auto restoredState = juce::ValueTree::fromXml(*xml);
  const auto restoredGraph = restoredState.getChildWithName("GraphDocument");
  if (restoredGraph.isValid()) {
    const juce::ScopedLock lock(graphStateLock);
    persistedGraphState = restoredGraph.createCopy();
    restoredState.removeChild(restoredGraph, nullptr);
  }
  parameters.replaceState(restoredState);
  restoringState.store(false, std::memory_order_release);

  const auto counter = static_cast<std::uint64_t>(
      xml->getStringAttribute("randomizationCounter", "0").getLargeIntValue());
  randomizationCounter.store(counter, std::memory_order_release);
  const auto seed = static_cast<std::uint64_t>(
      parameters.getRawParameterValue(openyourbox::params::globalSeed)->load());
  auto snapshot = modelBuilder.buildNow(getRequestedConfiguration(),
                                        seed + counter, counter);

  if (snapshot != nullptr && xml->hasAttribute("weights")) {
    const auto expectedHash = static_cast<std::uint64_t>(
        xml->getStringAttribute("architectureHash").getHexValue64());
    juce::MemoryBlock weights;

    if (expectedHash == snapshot->model->getArchitectureHash() &&
        weights.fromBase64Encoding(xml->getStringAttribute("weights"))) {
      try {
        const std::string bytes(static_cast<const char *>(weights.getData()),
                                weights.getSize());
        std::istringstream stream(bytes, std::ios::binary);
        torch::serialize::InputArchive archive;
        archive.load_from(stream);
        snapshot->model->load(archive);
      } catch (const std::exception &) {
        // The deterministic seed/counter model remains a valid fallback.
      }
    }
  }

  publishRuntime(snapshot);
  requestGraphCompile();
}

juce::AudioProcessorValueTreeState &
OpenYourBoxAudioProcessor::getParameterState() noexcept {
  return parameters;
}

openyourbox::dsp::TCNConfiguration
OpenYourBoxAudioProcessor::getRequestedConfiguration() const noexcept {
  openyourbox::dsp::TCNConfiguration configuration;
  configuration.depth = juce::roundToInt(
      parameters.getRawParameterValue(openyourbox::params::depth)->load());
  configuration.kernelSize = juce::roundToInt(
      parameters.getRawParameterValue(openyourbox::params::kernelSize)->load());
  configuration.channels = juce::roundToInt(
      parameters.getRawParameterValue(openyourbox::params::channels)->load());
  configuration.dilation = juce::roundToInt(
      parameters.getRawParameterValue(openyourbox::params::dilation)->load());
  configuration.activation =
      static_cast<openyourbox::dsp::ActivationType>(juce::roundToInt(
          parameters.getRawParameterValue(openyourbox::params::activation)
              ->load()));
  configuration.inputChannels = std::max(1, getTotalNumInputChannels());
  configuration.outputChannels = configuration.inputChannels;
  return configuration;
}

std::uint64_t
OpenYourBoxAudioProcessor::getReceptiveFieldSamples() const noexcept {
  const auto graphRuntime = graphPublisher.getPublishedRuntime();
  if (graphRuntime != nullptr)
    return graphRuntime->getSnapshot()->getReceptiveField();
  return 0;
}

std::uint64_t
OpenYourBoxAudioProcessor::getModelParameterCount() const noexcept {
  const auto graphRuntime = graphPublisher.getPublishedRuntime();
  if (graphRuntime != nullptr)
    return graphRuntime->getSnapshot()->getParameterCount();
  return 0;
}

double OpenYourBoxAudioProcessor::getFrozenInferenceTimeMilliseconds(
    std::int32_t nodeId) const noexcept {
  const auto runtime = graphPublisher.getPublishedRuntime();
  if (runtime == nullptr)
    return 0.0;
  return runtime->getFrozenInferenceTimeMilliseconds(nodeId);
}

double OpenYourBoxAudioProcessor::getCurrentSampleRate() const noexcept {
  return currentSampleRate.load(std::memory_order_acquire);
}

juce::String OpenYourBoxAudioProcessor::getModelError() const {
  const auto graphError = graphPublisher.getLastError();
  if (graphError.isNotEmpty())
    return graphError;
  const juce::ScopedLock lock(errorLock);
  return runtimeError.isNotEmpty() ? runtimeError : modelBuilder.getLastError();
}

void OpenYourBoxAudioProcessor::applyGraphConfiguration(
    const openyourbox::dsp::TCNConfiguration &configuration) {
  if (!configuration.isValid())
    return;
  const auto update = [this](const char *identifier, float value) {
    if (auto *parameter = parameters.getParameter(identifier)) {
      parameter->beginChangeGesture();
      parameter->setValueNotifyingHost(parameter->convertTo0to1(value));
      parameter->endChangeGesture();
    }
  };
  update(openyourbox::params::depth, static_cast<float>(configuration.depth));
  update(openyourbox::params::kernelSize,
         static_cast<float>(configuration.kernelSize));
  update(openyourbox::params::channels,
         static_cast<float>(configuration.channels));
  update(openyourbox::params::dilation,
         static_cast<float>(configuration.dilation));
  update(openyourbox::params::activation,
         static_cast<float>(configuration.activation));
}

void OpenYourBoxAudioProcessor::randomizeGraphElement(std::int32_t nodeId,
                                                     std::int32_t seed) {
  if (const auto runtime = graphPublisher.getPublishedRuntime()) {
    const auto &statistics = runtime->getSnapshot()->getElementStatistics();
    const auto compiled = std::find_if(
        statistics.begin(), statistics.end(),
        [nodeId](const openyourbox::dsp::LiveGraphElementStatistics &stats) {
          return stats.nodeId == nodeId && stats.randomizable;
        });
    if (compiled != statistics.end())
      graphPublisher.requestRandomization(nodeId, seed);
    return;
  }
  const auto unsignedSeed =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed));
  const auto elementSalt =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(nodeId)) << 32U;
  modelBuilder.requestBuild(
      getRequestedConfiguration(), unsignedSeed ^ elementSalt,
      randomizationCounter.load(std::memory_order_acquire));
}

juce::ValueTree OpenYourBoxAudioProcessor::getGraphState() const {
  const juce::ScopedLock lock(graphStateLock);
  return persistedGraphState.createCopy();
}

void OpenYourBoxAudioProcessor::setGraphState(const juce::ValueTree &graphState,
                                             bool compileRuntime) {
  if (!graphState.hasType("GraphDocument"))
    return;
  {
    const juce::ScopedLock lock(graphStateLock);
    persistedGraphState = graphState.createCopy();
  }
  graphRevision.fetch_add(1, std::memory_order_acq_rel);
  if (compileRuntime)
    requestGraphCompile();
}

void OpenYourBoxAudioProcessor::setRuntimeControls(
    const openyourbox::dsp::RuntimeControlState &controls) {
  auto published =
      std::make_shared<const openyourbox::dsp::RuntimeControlState>(controls);
  std::atomic_store_explicit(&publishedControls, std::move(published),
                             std::memory_order_release);
  graphRevision.fetch_add(1, std::memory_order_acq_rel);
}

std::uint64_t OpenYourBoxAudioProcessor::getGraphRevision() const noexcept {
  return graphRevision.load(std::memory_order_acquire);
}

bool OpenYourBoxAudioProcessor::isTransportPlaying() const noexcept {
  if (auto *head = getPlayHead()) {
    if (const auto position = head->getPosition())
      return position->getIsPlaying();
  }
  return transportPlaying.load(std::memory_order_acquire);
}

void OpenYourBoxAudioProcessor::copyLiveCapture(
    float &inputPeak, float &outputPeak, bool &suitable, float *const *input,
    int maxSamples, int &channels, int &samples) const noexcept {
  const auto index = liveCaptureIndex.load(std::memory_order_acquire);
  const auto &slot =
      liveCaptureSlots[static_cast<std::size_t>(index & 1)];
  inputPeak = slot.inputPeak;
  outputPeak = slot.outputPeak;
  suitable = slot.suitable;
  channels = slot.channels;
  samples = std::min(slot.samples, std::max(0, maxSamples));
  if (input == nullptr)
    return;
  for (int channel = 0; channel < channels; ++channel) {
    if (input[channel] == nullptr)
      continue;
    std::memcpy(input[channel], slot.input[static_cast<std::size_t>(channel)].data(),
                static_cast<std::size_t>(samples) * sizeof(float));
  }
}

bool OpenYourBoxAudioProcessor::getAnalysisTapPeaks(std::int32_t nodeId,
                                                   float &inputPeak,
                                                   float &outputPeak) const
    noexcept {
  auto runtime = graphPublisher.getPublishedRuntime();
  if (runtime == nullptr)
    return false;
  return runtime->getTapPeaks(nodeId, inputPeak, outputPeak);
}

bool OpenYourBoxAudioProcessor::prepareFrozenArtifact(
    const openyourbox::graph::FreezeSelectionResult &result,
    std::string &error) {
  if (result.inputChannels < 1 || result.outputChannels < 1) {
    error = "Compiled artifact has an invalid channel signature";
    return false;
  }

  const auto factory = openyourbox::dsp::TorchScriptBlackBoxFactory::load(
      result.artifactPath, result.inputChannels, result.receptiveFieldSamples,
      error, !result.acceptsConditioning, result.acceptsConditioning,
      result.condDim);
  if (factory == nullptr ||
      factory->getOutputChannels() != result.outputChannels)
    return false;

  const auto current = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                 std::memory_order_acquire);
  auto replacement = current != nullptr
                         ? std::make_shared<FrozenArtifactRegistry>(*current)
                         : std::make_shared<FrozenArtifactRegistry>();
  replacement->artifacts[result.artifactPath] = factory;
  std::atomic_store_explicit(
      &publishedFrozenArtifacts,
      std::shared_ptr<const FrozenArtifactRegistry>(std::move(replacement)),
      std::memory_order_release);
  return true;
}

void OpenYourBoxAudioProcessor::releaseFrozenArtifact(
    const std::string &artifactPath) {
  const auto current = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                 std::memory_order_acquire);
  if (current == nullptr || current->artifacts.count(artifactPath) == 0)
    return;
  auto replacement = std::make_shared<FrozenArtifactRegistry>(*current);
  replacement->artifacts.erase(artifactPath);
  std::atomic_store_explicit(
      &publishedFrozenArtifacts,
      std::shared_ptr<const FrozenArtifactRegistry>(std::move(replacement)),
      std::memory_order_release);
}

bool OpenYourBoxAudioProcessor::hasPreparedFrozenArtifact(
    const std::string &artifactPath) const noexcept {
  const auto current = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                 std::memory_order_acquire);
  return current != nullptr && current->artifacts.count(artifactPath) != 0;
}

void OpenYourBoxAudioProcessor::parameterChanged(const juce::String &parameterID,
                                                float newValue) {
  if (restoringState.load(std::memory_order_acquire))
    return;

  if (parameterID == openyourbox::params::randomize) {
    const auto isActive = newValue >= 0.5f;
    const auto wasActive =
        lastRandomizeValue.exchange(isActive, std::memory_order_acq_rel);
    if (isActive && !wasActive)
      randomizePending.store(true, std::memory_order_release);
  } else if (parameterID == openyourbox::params::depth ||
             parameterID == openyourbox::params::kernelSize ||
             parameterID == openyourbox::params::channels ||
             parameterID == openyourbox::params::dilation ||
             parameterID == openyourbox::params::activation) {
    architectureChangePending.store(true, std::memory_order_release);
  } else {
    return;
  }

  triggerAsyncUpdate();
}

void OpenYourBoxAudioProcessor::handleAsyncUpdate() {
  const auto shouldRandomize =
      randomizePending.exchange(false, std::memory_order_acq_rel) ||
      midiRandomizePending.exchange(false, std::memory_order_acq_rel);
  const auto architectureChanged =
      architectureChangePending.exchange(false, std::memory_order_acq_rel);

  if (shouldRandomize)
    requestCurrentArchitecture(true);
  else if (architectureChanged)
    requestCurrentArchitecture(false);
}

void OpenYourBoxAudioProcessor::publishRuntime(
    const std::shared_ptr<const openyourbox::dsp::ModelSnapshot> &snapshot) {
  if (snapshot == nullptr || !prepared.load(std::memory_order_acquire))
    return;

  if (auto runtime = createRuntime(snapshot)) {
    std::atomic_store_explicit(&publishedRuntime, std::move(runtime),
                               std::memory_order_release);
    const juce::ScopedLock lock(errorLock);
    runtimeError.clear();
  }
}

std::shared_ptr<OpenYourBoxAudioProcessor::RuntimeState>
OpenYourBoxAudioProcessor::createRuntime(
    const std::shared_ptr<const openyourbox::dsp::ModelSnapshot> &snapshot) {
  const auto receptiveField = snapshot->model->getReceptiveField();
  if (receptiveField == 0 || receptiveField - 1 > maximumRealtimeHistory) {
    const juce::ScopedLock lock(errorLock);
    runtimeError = "Receptive field exceeds the real-time workspace limit";
    return {};
  }

  try {
    auto runtime = std::make_shared<RuntimeState>();
    runtime->snapshot = snapshot;
    runtime->maximumBlockSize =
        preparedBlockSize.load(std::memory_order_acquire);
    const auto history = static_cast<std::size_t>(receptiveField - 1);
    const auto channels = snapshot->model->getConfiguration().inputChannels;
    runtime->lookback.resize(channels, history);
    runtime->inputTensor = torch::zeros(
        {1, channels,
         static_cast<std::int64_t>(history) + runtime->maximumBlockSize},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    return runtime;
  } catch (const std::exception &error) {
    const juce::ScopedLock lock(errorLock);
    runtimeError = error.what();
  }

  return {};
}

torch::Tensor
OpenYourBoxAudioProcessor::runModel(RuntimeState &runtime,
                                   const juce::AudioBuffer<float> &input,
                                   int numSamples) {
  runtime.lookback.prependToTensor(runtime.inputTensor, input, numSamples);
  const auto validSamples =
      static_cast<std::int64_t>(runtime.lookback.getHistorySamples()) +
      numSamples;
  torch::InferenceMode inferenceGuard;
  return runtime.snapshot->model->forward(
      runtime.inputTensor.narrow(2, 0, validSamples));
}

bool OpenYourBoxAudioProcessor::processLiveGraph(
    juce::AudioBuffer<float> &buffer, int inputChannels,
    int numSamples) noexcept {
  auto latest = graphPublisher.getPublishedRuntime();
  if (latest != activeGraphRuntime) {
    previousGraphRuntime = activeGraphRuntime;
    activeGraphRuntime = std::move(latest);
    if (activeGraphRuntime != nullptr)
      activeGraphRuntime->reset();
    graphCrossfadeSamplesRemaining =
        previousGraphRuntime != nullptr && activeGraphRuntime != nullptr
            ? modelCrossfadeSamples
            : 0;
  }
  if (activeGraphRuntime == nullptr ||
      numSamples > graphWetBuffer.getNumSamples())
    return false;

  if (auto controls = std::atomic_load_explicit(&publishedControls,
                                                std::memory_order_acquire))
    activeGraphRuntime->bindControls(controls);

  LiveCaptureSlot capture;
  capture.channels = std::min(2, inputChannels);
  capture.samples = std::min(512, numSamples);
  for (int channel = 0; channel < capture.channels; ++channel) {
    const auto *source = buffer.getReadPointer(channel);
    std::memcpy(capture.input[static_cast<std::size_t>(channel)].data(),
                source, static_cast<std::size_t>(capture.samples) * sizeof(float));
    capture.inputPeak =
        std::max(capture.inputPeak,
                 buffer.getMagnitude(channel, 0, numSamples));
  }

  const auto outputChannels = getTotalNumOutputChannels();
  if (!activeGraphRuntime->processHost(buffer.getArrayOfReadPointers(),
                                       static_cast<std::size_t>(inputChannels),
                                       graphWetBuffer.getArrayOfWritePointers(),
                                       static_cast<std::size_t>(outputChannels),
                                       static_cast<std::size_t>(numSamples)))
    return false;

  auto useCrossfade =
      previousGraphRuntime != nullptr && graphCrossfadeSamplesRemaining > 0;
  if (useCrossfade && !previousGraphRuntime->processHost(
                          buffer.getArrayOfReadPointers(),
                          static_cast<std::size_t>(inputChannels),
                          previousGraphWetBuffer.getArrayOfWritePointers(),
                          static_cast<std::size_t>(outputChannels),
                          static_cast<std::size_t>(numSamples))) {
    useCrossfade = false;
    previousGraphRuntime.reset();
    graphCrossfadeSamplesRemaining = 0;
  }

  for (int sample = 0; sample < numSamples; ++sample) {
    const auto fade =
        useCrossfade && graphCrossfadeSamplesRemaining > 0
            ? 1.0f - static_cast<float>(graphCrossfadeSamplesRemaining) /
                         static_cast<float>(modelCrossfadeSamples)
            : 1.0f;
    const auto mix = dryWetSmoother.getNextValue();
    for (int channel = 0; channel < outputChannels; ++channel) {
      auto processed = graphWetBuffer.getSample(channel, sample);
      if (useCrossfade) {
        const auto previous = previousGraphWetBuffer.getSample(channel, sample);
        processed = previous + fade * (processed - previous);
      }
      const auto dry = buffer.getSample(channel, sample);
      buffer.setSample(channel, sample, dry + mix * (processed - dry));
    }
    if (graphCrossfadeSamplesRemaining > 0)
      --graphCrossfadeSamplesRemaining;
  }
  if (graphCrossfadeSamplesRemaining == 0)
    previousGraphRuntime.reset();

  applyDcBlocker(graphDcInput, graphDcOutput, buffer, outputChannels,
                 numSamples);
  capture.outputPeak = buffer.getMagnitude(0, 0, numSamples);
  capture.suitable = capture.inputPeak > 1.0e-5f;
  const auto writeIndex =
      1 - (liveCaptureIndex.load(std::memory_order_relaxed) & 1);
  liveCaptureSlots[static_cast<std::size_t>(writeIndex)] = capture;
  liveCaptureIndex.store(writeIndex, std::memory_order_release);
  return true;
}

void OpenYourBoxAudioProcessor::applyDcBlocker(RuntimeState &runtime,
                                              juce::AudioBuffer<float> &buffer,
                                              int channels,
                                              int samples) noexcept {
  applyDcBlocker(runtime.dcInput, runtime.dcOutput, buffer, channels, samples);
}

void OpenYourBoxAudioProcessor::applyDcBlocker(std::array<float, 2> &inputState,
                                              std::array<float, 2> &outputState,
                                              juce::AudioBuffer<float> &buffer,
                                              int channels,
                                              int samples) noexcept {
  const auto coefficient = dcBlockerCoefficient.load(std::memory_order_relaxed);
  const auto processedChannels =
      std::min(channels, static_cast<int>(inputState.size()));
  for (int channel = 0; channel < processedChannels; ++channel) {
    auto previousInput = inputState[static_cast<std::size_t>(channel)];
    auto previousOutput = outputState[static_cast<std::size_t>(channel)];
    auto *samplesData = buffer.getWritePointer(channel);
    for (int sample = 0; sample < samples; ++sample) {
      const auto input = samplesData[sample];
      const auto output = input - previousInput + coefficient * previousOutput;
      samplesData[sample] = output;
      previousInput = input;
      previousOutput = output;
    }
    inputState[static_cast<std::size_t>(channel)] = previousInput;
    outputState[static_cast<std::size_t>(channel)] = previousOutput;
  }
}

void OpenYourBoxAudioProcessor::syncDryWetSmoother() noexcept {
  if (auto *mix = parameters.getRawParameterValue(openyourbox::params::dryWet))
    dryWetSmoother.setTargetValue(mix->load());
}

void OpenYourBoxAudioProcessor::requestGraphCompile() {
  if (!prepared.load(std::memory_order_acquire))
    return;
  auto graphState = getGraphState();
  if (!graphState.isValid() || graphState.getNumChildren() == 0)
    return;

  openyourbox::dsp::LiveGraphCompileOptions options;
  options.hostInputChannels = std::max(1, getTotalNumInputChannels());
  options.hostOutputChannels = std::max(1, getTotalNumOutputChannels());
  options.maximumBlockSize = preparedBlockSize.load(std::memory_order_acquire);
  options.maximumHistorySamples = maximumRealtimeHistory;
  options.sampleRate =
      std::max(1.0, currentSampleRate.load(std::memory_order_acquire));
  {
    const juce::ScopedLock lock(trainingPreviewLock);
    if (!trainingPreviewPath.empty()) {
      const std::unordered_set<std::int32_t> preview(
          trainingPreviewNodeIds.begin(), trainingPreviewNodeIds.end());
      for (int index = 0; index < graphState.getNumChildren(); ++index) {
        auto child = graphState.getChild(index);
        if (!child.hasType("Node"))
          continue;
        const auto nodeId = static_cast<std::int32_t>(child["id"]);
        if (preview.count(nodeId) == 0)
          continue;
        child.setProperty("state", "frozen_gold", nullptr);
        child.setProperty("artifactPath", juce::String(trainingPreviewPath),
                          nullptr);
        child.setProperty("blackBoxOrigin", "train_autoload", nullptr);
      }
    }
  }
  graphPublisher.requestCompile(
      graphState, options, [this](const openyourbox::graph::GraphNode &node) {
        return resolveFrozenBlackBox(node);
      });
}

std::shared_ptr<const openyourbox::dsp::FrozenBlackBoxFactory>
OpenYourBoxAudioProcessor::resolveFrozenBlackBoxForAnalysis(
    const openyourbox::graph::GraphNode &node) const {
  return resolveFrozenBlackBox(node);
}

std::shared_ptr<const openyourbox::dsp::FrozenBlackBoxFactory>
OpenYourBoxAudioProcessor::resolveFrozenBlackBox(
    const openyourbox::graph::GraphNode &node) const {
  if (node.artifactPath.empty())
    return {};
  const auto registry = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                  std::memory_order_acquire);
  if (registry == nullptr)
    return {};
  const auto found = registry->artifacts.find(node.artifactPath);
  return found != registry->artifacts.end() ? found->second : nullptr;
}

void OpenYourBoxAudioProcessor::requestCurrentArchitecture(
    bool randomizeWeights) {
  const auto configuration = getRequestedConfiguration();
  const auto globalSeed = static_cast<std::uint64_t>(
      parameters.getRawParameterValue(openyourbox::params::globalSeed)->load());

  if (randomizeWeights) {
    const auto counter =
        randomizationCounter.fetch_add(1, std::memory_order_acq_rel) + 1;
    modelBuilder.requestRandomization(configuration, globalSeed, counter);
  } else {
    const auto counter = randomizationCounter.load(std::memory_order_acquire);
    modelBuilder.requestBuild(configuration, globalSeed + counter, counter);
  }
}

void OpenYourBoxAudioProcessor::resetRandomizeParameter() {
  auto *parameter = parameters.getParameter(openyourbox::params::randomize);
  if (parameter != nullptr && parameter->getValue() >= 0.5f) {
    parameter->beginChangeGesture();
    parameter->setValueNotifyingHost(0.0f);
    parameter->endChangeGesture();
  }
  lastRandomizeValue.store(false, std::memory_order_release);
}

bool OpenYourBoxAudioProcessor::prepareTrainedArtifact(
    const openyourbox::graph::TrainJobResult &result, std::string &error) {
  const auto channels = result.inputChannels > 0 ? result.inputChannels
                                                 : getTotalNumInputChannels();
  const auto field =
      result.receptiveFieldSamples > 0 ? result.receptiveFieldSamples : 1;
  auto loadChannels = channels;
  const auto hostChannels = std::max(1, getTotalNumInputChannels());
  const auto factory = openyourbox::dsp::TorchScriptBlackBoxFactory::load(
      result.artifactPath, loadChannels, field, error, false,
      result.acceptsConditioning, result.condDim);
  if (factory == nullptr && loadChannels != hostChannels) {
    error.clear();
    loadChannels = hostChannels;
    const auto retry = openyourbox::dsp::TorchScriptBlackBoxFactory::load(
        result.artifactPath, loadChannels, field, error, false,
        result.acceptsConditioning, result.condDim);
    if (retry == nullptr)
      return false;
    const auto current = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                   std::memory_order_acquire);
    auto replacement = current != nullptr
                           ? std::make_shared<FrozenArtifactRegistry>(*current)
                           : std::make_shared<FrozenArtifactRegistry>();
    replacement->artifacts[result.artifactPath] = retry;
    std::atomic_store_explicit(
        &publishedFrozenArtifacts,
        std::shared_ptr<const FrozenArtifactRegistry>(std::move(replacement)),
        std::memory_order_release);
    return true;
  }
  if (factory == nullptr)
    return false;
  const auto current = std::atomic_load_explicit(&publishedFrozenArtifacts,
                                                 std::memory_order_acquire);
  auto replacement = current != nullptr
                         ? std::make_shared<FrozenArtifactRegistry>(*current)
                         : std::make_shared<FrozenArtifactRegistry>();
  replacement->artifacts[result.artifactPath] = factory;
  std::atomic_store_explicit(
      &publishedFrozenArtifacts,
      std::shared_ptr<const FrozenArtifactRegistry>(std::move(replacement)),
      std::memory_order_release);
  return true;
}

void OpenYourBoxAudioProcessor::setTrainingPreview(
    const std::string &artifactPath, const std::vector<std::int32_t> &nodeIds) {
  {
    const juce::ScopedLock lock(trainingPreviewLock);
    trainingPreviewPath = artifactPath;
    trainingPreviewNodeIds = nodeIds;
  }
  requestGraphCompile();
}

void OpenYourBoxAudioProcessor::clearTrainingPreview() {
  bool hadPreview = false;
  {
    const juce::ScopedLock lock(trainingPreviewLock);
    hadPreview = !trainingPreviewPath.empty();
    trainingPreviewPath.clear();
    trainingPreviewNodeIds.clear();
  }
  if (hadPreview)
    requestGraphCompile();
}

void OpenYourBoxAudioProcessor::setCaptureBypass(bool enabled) noexcept {
  captureBypass.store(enabled, std::memory_order_release);
}

bool OpenYourBoxAudioProcessor::isCaptureBypassEnabled() const noexcept {
  return captureBypass.load(std::memory_order_acquire);
}

bool OpenYourBoxAudioProcessor::startInputCapture(const juce::File &destination,
                                                  double sampleRate,
                                                  int channels) {
  return inputCapture.start(destination, sampleRate, channels);
}

void OpenYourBoxAudioProcessor::drainInputCapture() { inputCapture.drain(); }

juce::File OpenYourBoxAudioProcessor::stopInputCapture() {
  return inputCapture.stop();
}

bool OpenYourBoxAudioProcessor::isInputCaptureActive() const noexcept {
  return inputCapture.isActive();
}

openyourbox::capture::CapturePairing &
OpenYourBoxAudioProcessor::getCapturePairing() noexcept {
  return capturePairing;
}

const openyourbox::capture::CapturePairing &
OpenYourBoxAudioProcessor::getCapturePairing() const noexcept {
  return capturePairing;
}

openyourbox::library::TrainingLibrary &
OpenYourBoxAudioProcessor::getTrainingLibrary() noexcept {
  return trainingLibrary;
}

const openyourbox::library::TrainingLibrary &
OpenYourBoxAudioProcessor::getTrainingLibrary() const noexcept {
  return trainingLibrary;
}

void OpenYourBoxAudioProcessor::setStartTransportOnRecord(bool enabled) noexcept {
  startTransportOnRecord.store(enabled, std::memory_order_release);
}

bool OpenYourBoxAudioProcessor::getStartTransportOnRecord() const noexcept {
  return startTransportOnRecord.load(std::memory_order_acquire);
}

void OpenYourBoxAudioProcessor::timerCallback() { inputCapture.drain(); }

void OpenYourBoxAudioProcessor::requestTransportStartIfNeeded() noexcept {
  if (!startTransportOnRecord.load(std::memory_order_acquire) ||
      isTransportPlaying())
    return;
  pendingTransportPlay.store(true, std::memory_order_release);
  juce::MessageManager::callAsync(
      [] { openyourbox::capture::requestHostTransportStart(); });
}

void OpenYourBoxAudioProcessor::setCaptureStatusMessage(
    const juce::String &message) {
  const juce::ScopedLock lock(captureStateLock);
  captureStatusMessage = message;
}

juce::String OpenYourBoxAudioProcessor::takeCaptureStatusMessage() {
  const juce::ScopedLock lock(captureStateLock);
  return std::exchange(captureStatusMessage, {});
}

bool OpenYourBoxAudioProcessor::takeLibraryFocusRequest() noexcept {
  const juce::ScopedLock lock(captureStateLock);
  const auto requested = libraryFocusRequested;
  libraryFocusRequested = false;
  return requested;
}

bool OpenYourBoxAudioProcessor::startLocalCapture(const juce::String &pairId,
                                                  const juce::String &suffix) {
  const auto directory =
      juce::File::getSpecialLocation(juce::File::tempDirectory)
          .getChildFile("OpenYourBox")
          .getChildFile("Capture");
  directory.createDirectory();
  const auto localFile = directory.getChildFile(pairId + suffix + ".wav");
  setCaptureBypass(capturePairing.isCaptureBypassEnabled());
  return startInputCapture(localFile, getCurrentSampleRate(),
                           getTotalNumInputChannels());
}

void OpenYourBoxAudioProcessor::startPairedRecording() {
  if (!capturePairing.canRecord()) {
    setCaptureStatusMessage("Assign complementary Clean/Processed roles.");
    return;
  }
  const auto pairId = juce::Uuid().toDashedString();
  {
    const juce::ScopedLock lock(captureStateLock);
    activeCapturePairId = pairId;
    localCaptureClip = juce::File();
    pendingPeerClip = juce::File();
    waitingForPeerClip = false;
  }
  if (!startLocalCapture(pairId, "_local")) {
    setCaptureStatusMessage("Could not start capture");
    return;
  }
  capturePairing.startRecording(pairId, getCurrentSampleRate());
  requestTransportStartIfNeeded();
}

void OpenYourBoxAudioProcessor::stopPairedRecording() {
  capturePairing.stopRecording();
  const auto clip = stopInputCapture();
  setCaptureBypass(false);
  {
    const juce::ScopedLock lock(captureStateLock);
    localCaptureClip = clip;
    waitingForPeerClip = true;
  }
  if (!clip.existsAsFile()) {
    {
      const juce::ScopedLock lock(captureStateLock);
      waitingForPeerClip = false;
    }
    setCaptureStatusMessage("Capture discarded: no local audio was recorded");
    return;
  }
  tryAssembleCapturedPair();
}

void OpenYourBoxAudioProcessor::tryAssembleCapturedPair() {
  juce::File xFile;
  juce::File yFile;
  {
    const juce::ScopedLock lock(captureStateLock);
    if (!localCaptureClip.existsAsFile() || !pendingPeerClip.existsAsFile()) {
      if (waitingForPeerClip)
        captureStatusMessage = "Waiting for peer clip...";
      return;
    }
    const auto localRole = capturePairing.getCaptureRole();
    if (localRole == openyourbox::capture::CaptureRole::clean) {
      xFile = localCaptureClip;
      yFile = pendingPeerClip;
    } else {
      yFile = localCaptureClip;
      xFile = pendingPeerClip;
    }
    waitingForPeerClip = false;
  }
  juce::String error;
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(xFile));
  const auto duration =
      reader != nullptr && reader->sampleRate > 0.0
          ? static_cast<double>(reader->lengthInSamples) / reader->sampleRate
          : 0.0;
  const auto name = juce::String("Capture ") +
                    juce::Time::getCurrentTime().formatted("%Y-%m-%d %H:%M");
  if (!trainingLibrary.addCapturedPair(
          name, xFile, yFile,
          reader != nullptr ? reader->sampleRate : getCurrentSampleRate(),
          reader != nullptr ? static_cast<int>(reader->numChannels)
                            : std::max(1, getTotalNumInputChannels()),
          duration, error)) {
    setCaptureStatusMessage(error);
    return;
  }
  {
    const juce::ScopedLock lock(captureStateLock);
    libraryFocusRequested = true;
    captureStatusMessage = "Capture added to Training Library";
    localCaptureClip = juce::File();
    pendingPeerClip = juce::File();
  }
}

void OpenYourBoxAudioProcessor::handlePairingMessage(const juce::var &message) {
  const auto type = message.getProperty("type", {}).toString();
  if (type == "clip_ready") {
    {
      const juce::ScopedLock lock(captureStateLock);
      pendingPeerClip = juce::File(message.getProperty("path", {}).toString());
    }
    tryAssembleCapturedPair();
    return;
  }
  if (type == "record_start" &&
      capturePairing.getPairingRole() ==
          openyourbox::capture::PairingRole::slave) {
    const auto pairId = message.getProperty("pairId", {}).toString();
    {
      const juce::ScopedLock lock(captureStateLock);
      activeCapturePairId = pairId;
    }
    const auto rate = static_cast<double>(
        message.getProperty("sampleRate", getCurrentSampleRate()));
    (void)rate;
    if (!startLocalCapture(pairId, "_slave"))
      setCaptureStatusMessage("Could not start slave capture");
    else
      requestTransportStartIfNeeded();
    return;
  }
  if (type == "record_stop" &&
      capturePairing.getPairingRole() ==
          openyourbox::capture::PairingRole::slave) {
    const auto clip = stopInputCapture();
    juce::String pairId;
    {
      const juce::ScopedLock lock(captureStateLock);
      pairId = activeCapturePairId;
    }
    if (clip.existsAsFile())
      capturePairing.sendClipReady(pairId, clip);
    setCaptureBypass(false);
    return;
  }
  if (type == "set_bypass")
    setCaptureBypass(capturePairing.isCaptureBypassEnabled());
  else if (type == "peer_lost") {
    if (isInputCaptureActive())
      stopInputCapture();
    setCaptureBypass(false);
    waitingForPeerClip = false;
    setCaptureStatusMessage("Peer disconnected");
  }
}

bool OpenYourBoxAudioProcessor::startPreviewFile(const juce::File &file) {
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr)
    return false;
  auto preview = std::make_shared<PreviewState>();
  const auto length = static_cast<int>(
      std::min<juce::int64>(reader->lengthInSamples, 1 << 24));
  preview->samples.setSize(static_cast<int>(reader->numChannels), length);
  reader->read(&preview->samples, 0, length, 0, true, true);
  preview->position.store(0, std::memory_order_release);
  preview->active.store(true, std::memory_order_release);
  std::atomic_store_explicit(&previewPlayback, preview,
                             std::memory_order_release);
  return true;
}

void OpenYourBoxAudioProcessor::stopPreview() {
  if (auto preview = std::atomic_load_explicit(&previewPlayback,
                                               std::memory_order_acquire))
    preview->active.store(false, std::memory_order_release);
}

bool OpenYourBoxAudioProcessor::isPreviewPlaying() const noexcept {
  if (auto preview = std::atomic_load_explicit(&previewPlayback,
                                               std::memory_order_acquire))
    return preview->active.load(std::memory_order_acquire);
  return false;
}

void OpenYourBoxAudioProcessor::mixPreview(juce::AudioBuffer<float> &buffer,
                                           int channels, int samples) noexcept {
  auto preview = std::atomic_load_explicit(&previewPlayback,
                                           std::memory_order_acquire);
  if (preview == nullptr || !preview->active.load(std::memory_order_acquire))
    return;
  auto position = preview->position.load(std::memory_order_relaxed);
  const auto total = preview->samples.getNumSamples();
  const auto srcChannels = preview->samples.getNumChannels();
  if (position >= total) {
    preview->active.store(false, std::memory_order_release);
    return;
  }
  for (int sample = 0; sample < samples && position < total; ++sample, ++position) {
    for (int channel = 0; channel < channels; ++channel) {
      const auto sourceChannel = std::min(channel, srcChannels - 1);
      buffer.addSample(channel, sample,
                       preview->samples.getSample(sourceChannel, position));
    }
  }
  preview->position.store(position, std::memory_order_release);
  if (position >= total)
    preview->active.store(false, std::memory_order_release);
}

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new OpenYourBoxAudioProcessor();
}
