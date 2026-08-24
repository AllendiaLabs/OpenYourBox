#include "TrainCoordinator.h"

#include "../library/UserDataPaths.h"
#include <BinaryData.h>

#include <array>
#include <limits>
#include <utility>

#ifndef OPENYOURBOX_PYTHON_EXECUTABLE
#define OPENYOURBOX_PYTHON_EXECUTABLE "python3"
#endif

namespace {
/**
 * @brief Reads a string property from a parsed JUCE JSON object.
 * @param object Parsed JSON object.
 * @param key Property name.
 * @return UTF-8 property value.
 */
std::string jsonString(const juce::var &object, const juce::Identifier &key) {
  return object.getProperty(key, {}).toString().toStdString();
}

/**
 * @brief Writes the embedded Python train worker to a private temporary location.
 * @param error Receives a user-facing filesystem failure.
 * @return Runnable script file, or an invalid file on failure.
 */
juce::File materializeTrainWorker(std::string &error) {
  const auto workerDirectory =
      juce::File::getSpecialLocation(juce::File::tempDirectory)
          .getChildFile("OpenYourBox")
          .getChildFile("Worker");
  if (workerDirectory.createDirectory().failed()) {
    error = "Could not create the embedded worker directory";
    return {};
  }

  const auto worker = workerDirectory.getChildFile("train_worker.py");
  if (!worker.replaceWithData(BinaryData::train_worker_py,
                              BinaryData::train_worker_pySize)) {
    error = "Could not materialize the embedded train worker";
    return {};
  }
  return worker;
}

/**
 * @brief Parses every complete JSON object from accumulated worker output.
 * @param buffer In/out remaining unparsed text.
 * @param objects Receives newly parsed objects.
 * @param diagnostics Receives non-JSON stdout/stderr lines (tracebacks, mlflow).
 */
void consumeJsonLines(juce::String &buffer, juce::Array<juce::var> &objects,
                      juce::String &diagnostics) {
  while (buffer.containsChar('\n')) {
    const auto line = buffer.upToFirstOccurrenceOf("\n", false, false);
    buffer = buffer.fromFirstOccurrenceOf("\n", false, false);
    const auto trimmed = line.trim();
    if (trimmed.isEmpty())
      continue;
    const auto parsed = juce::JSON::parse(trimmed);
    if (parsed.isObject())
      objects.add(parsed);
    else {
      diagnostics << trimmed << "\n";
      constexpr int maximumDiagnostics = 24000;
      if (diagnostics.length() > maximumDiagnostics)
        diagnostics = diagnostics.substring(diagnostics.length() -
                                            maximumDiagnostics);
    }
  }
}

/**
 * @brief Appends captured worker text onto a primary failure reason.
 * @param primary JSON `error_message` or C++ preparation failure.
 * @param diagnostics Non-JSON worker stdout/stderr.
 * @return Combined user-facing failure text.
 */
std::string composeWorkerFailure(const std::string &primary,
                                 const juce::String &diagnostics) {
  juce::String text(primary);
  if (text.isEmpty())
    text = "Train worker rejected the request";
  if (diagnostics.isNotEmpty() && !text.contains(diagnostics.trimEnd())) {
    text << "\n\nWorker output:\n" << diagnostics;
  }
  return text.toStdString();
}
} // namespace

namespace openyourbox::train {
TrainCoordinator::TrainCoordinator(ArtifactPreparer artifactPreparer)
    : Thread("OpenYourBox Train Worker"),
      prepareArtifact(std::move(artifactPreparer)) {}

TrainCoordinator::~TrainCoordinator() {
  signalThreadShouldExit();
  writeCommand("stop");
  {
    const juce::ScopedLock lock(stateLock);
    if (childProcess != nullptr)
      childProcess->kill();
  }
  stopThread(5000);
}

bool TrainCoordinator::start(const graph::TrainJobRequest &request) {
  if (request.graphFragment.empty() ||
      status.load(std::memory_order_acquire) != TrainStatus::idle)
    return false;

  {
    const juce::ScopedLock lock(stateLock);
    pendingRequest = request;
    completedResult.reset();
    latestProgress = {};
    latestProgress.requestId = request.requestId;
    latestProgress.status = "running";
    latestProgress.totalSteps = 2500;
    lossHistory.clear();
    statusMessage = "Training...";
  }
  status.store(TrainStatus::running, std::memory_order_release);
  startThread();
  return true;
}

void TrainCoordinator::pause() {
  if (status.load(std::memory_order_acquire) != TrainStatus::running)
    return;
  writeCommand("pause");
}

void TrainCoordinator::resume() {
  if (status.load(std::memory_order_acquire) != TrainStatus::paused)
    return;
  writeCommand("resume");
}

void TrainCoordinator::stop() {
  const auto current = status.load(std::memory_order_acquire);
  if (current != TrainStatus::running && current != TrainStatus::paused)
    return;
  writeCommand("stop");
}

bool TrainCoordinator::retryLoad(const graph::TrainJobResult &result) {
  if (prepareArtifact == nullptr || result.artifactPath.empty() ||
      !juce::File(result.artifactPath).existsAsFile())
    return false;
  std::string error;
  return prepareArtifact(result, error);
}

TrainStatus TrainCoordinator::getStatus() const noexcept {
  return status.load(std::memory_order_acquire);
}

juce::String TrainCoordinator::getStatusMessage() const {
  const juce::ScopedLock lock(stateLock);
  return statusMessage;
}

graph::TrainJobResult TrainCoordinator::getProgress() const {
  const juce::ScopedLock lock(stateLock);
  return latestProgress;
}

std::vector<float> TrainCoordinator::getLossHistory() const {
  const juce::ScopedLock lock(stateLock);
  return lossHistory;
}

std::optional<graph::TrainJobResult> TrainCoordinator::takeResult() {
  const juce::ScopedLock lock(stateLock);
  auto result = std::move(completedResult);
  completedResult.reset();
  if (result.has_value())
    status.store(TrainStatus::idle, std::memory_order_release);
  return result;
}

void TrainCoordinator::writeCommand(const juce::String &command) {
  juce::File file;
  std::string requestId;
  {
    const juce::ScopedLock lock(stateLock);
    file = commandFile;
    requestId = pendingRequest.requestId;
  }
  if (!file.getFullPathName().isEmpty()) {
    auto object = std::make_unique<juce::DynamicObject>();
    object->setProperty("request_id", juce::String(requestId));
    object->setProperty("operation", "train_steerable");
    object->setProperty("command", command);
    file.replaceWithText(juce::JSON::toString(juce::var(object.release()), true));
  }
}

void TrainCoordinator::applyProgressObject(const juce::var &parsed) {
  graph::TrainJobResult progress;
  {
    const juce::ScopedLock lock(stateLock);
    progress = latestProgress;
  }
  progress.requestId = jsonString(parsed, "request_id");
  progress.status = jsonString(parsed, "status");
  progress.step = static_cast<int>(parsed.getProperty("step", progress.step));
  progress.totalSteps =
      static_cast<int>(parsed.getProperty("total_steps", progress.totalSteps));
  progress.loss = static_cast<double>(parsed.getProperty("loss", progress.loss));
  progress.bestLoss = static_cast<double>(
      parsed.getProperty("best_loss", progress.bestLoss));
  progress.learningRate = static_cast<double>(
      parsed.getProperty("learning_rate", progress.learningRate));
  progress.artifactPath = jsonString(parsed, "artifact_path");
  progress.errorMessage = jsonString(parsed, "error_message");
  if (progress.errorMessage.empty())
    progress.errorMessage = jsonString(parsed, "message");
  if (!progress.artifactPath.empty())
    progress.acceptsConditioning = true;
  const auto metadata = parsed.getProperty("blackbox_metadata", {});
  if (metadata.isObject()) {
    progress.acceptsConditioning =
        static_cast<bool>(metadata.getProperty("conditioning", true));
    progress.condDim = static_cast<int>(metadata.getProperty("cond_dim", 0));
    progress.receptiveFieldSamples = static_cast<std::uint64_t>(
        static_cast<juce::int64>(
            metadata.getProperty("receptive_field_samples", 0)));
    const auto shape = metadata.getProperty("shape_signature", {});
    progress.inputChannels =
        static_cast<int>(shape.getProperty("input_channels", 0));
    progress.outputChannels =
        static_cast<int>(shape.getProperty("output_channels", 0));
  }

  {
    const juce::ScopedLock lock(stateLock);
    latestProgress = progress;
    if (progress.status == "running" && progress.step > 0)
      lossHistory.push_back(static_cast<float>(progress.loss));
    if (lossHistory.size() > 2048)
      lossHistory.erase(lossHistory.begin());
    if (progress.status == "paused") {
      statusMessage = "Paused";
      status.store(TrainStatus::paused, std::memory_order_release);
    } else if (progress.status == "running" || progress.status == "stopping") {
      statusMessage = "Training...";
      if (status.load(std::memory_order_acquire) == TrainStatus::paused)
        status.store(TrainStatus::running, std::memory_order_release);
    }
  }
}

void TrainCoordinator::run() {
  graph::TrainJobRequest request;
  {
    const juce::ScopedLock lock(stateLock);
    request = pendingRequest;
  }

  std::string workerError;
  const auto worker = materializeTrainWorker(workerError);
  if (!worker.existsAsFile()) {
    publishFailure(request.requestId, workerError);
    return;
  }

  auto requestFile = juce::File::createTempFile("openyourbox-train.json");
  if (!requestFile.replaceWithText(request.graphFragment, false, false, "\n")) {
    publishFailure(request.requestId, "Could not write the train request");
    return;
  }

  const auto artifactDirectory = library::weightsDirectory();
  if (artifactDirectory.createDirectory().failed()) {
    publishFailure(request.requestId, "Could not create the artifact directory");
    requestFile.deleteFile();
    return;
  }

  auto commands = juce::File::createTempFile("openyourbox-train-cmd.json");
  {
    const juce::ScopedLock lock(stateLock);
    commandFile = commands;
  }

  juce::StringArray command;
  command.add(OPENYOURBOX_PYTHON_EXECUTABLE);
  command.add(worker.getFullPathName());
  command.add("--request");
  command.add(requestFile.getFullPathName());
  command.add("--artifact-dir");
  command.add(artifactDirectory.getFullPathName());
  command.add("--command-file");
  command.add(commands.getFullPathName());

  auto process = std::make_unique<juce::ChildProcess>();
  if (!process->start(command, juce::ChildProcess::wantStdOut |
                                   juce::ChildProcess::wantStdErr)) {
    publishFailure(request.requestId, "Could not launch the train worker");
    requestFile.deleteFile();
    commands.deleteFile();
    return;
  }
  {
    const juce::ScopedLock lock(stateLock);
    childProcess = std::move(process);
  }

  juce::String output;
  juce::String workerDiagnostics;
  for (;;) {
    if (threadShouldExit()) {
      writeCommand("stop");
      const juce::ScopedLock lock(stateLock);
      if (childProcess != nullptr)
        childProcess->kill();
      requestFile.deleteFile();
      commands.deleteFile();
      return;
    }

    juce::ChildProcess *processHandle = nullptr;
    {
      const juce::ScopedLock lock(stateLock);
      processHandle = childProcess.get();
    }
    if (processHandle == nullptr)
      break;

    std::array<char, 4096> outputBuffer{};
    const auto bytesRead = processHandle->readProcessOutput(
        outputBuffer.data(), static_cast<int>(outputBuffer.size()));
    if (bytesRead > 0)
      output += juce::String::fromUTF8(outputBuffer.data(), bytesRead);
    juce::Array<juce::var> objects;
    consumeJsonLines(output, objects, workerDiagnostics);
    for (const auto &object : objects)
      applyProgressObject(object);

    if (!processHandle->isRunning())
      break;
    wait(20);
  }

  auto exitCode = std::numeric_limits<juce::uint32>::max();
  {
    const juce::ScopedLock lock(stateLock);
    if (childProcess != nullptr) {
      output += childProcess->readAllProcessOutput();
      exitCode = childProcess->getExitCode();
      childProcess.reset();
    }
  }
  juce::Array<juce::var> objects;
  consumeJsonLines(output, objects, workerDiagnostics);
  const auto leftover = juce::JSON::parse(output.trim());
  if (leftover.isObject())
    objects.add(leftover);
  else if (output.trim().isNotEmpty())
    workerDiagnostics << output.trim() << "\n";
  for (const auto &object : objects)
    applyProgressObject(object);

  requestFile.deleteFile();
  commands.deleteFile();

  graph::TrainJobResult result;
  {
    const juce::ScopedLock lock(stateLock);
    result = latestProgress;
  }
  if (result.status == "stopped") {
    result.requestId = request.requestId;
    {
      const juce::ScopedLock lock(stateLock);
      completedResult = result;
      statusMessage = "Stopped";
      status.store(TrainStatus::stopped, std::memory_order_release);
    }
    return;
  }

  if (exitCode != 0 || result.status != "success" || result.artifactPath.empty()) {
    publishFailure(request.requestId,
                   composeWorkerFailure(result.errorMessage, workerDiagnostics));
    return;
  }

  std::string preparationError;
  if (prepareArtifact != nullptr &&
      !prepareArtifact(result, preparationError)) {
    result.errorMessage = composeWorkerFailure(
        preparationError.empty() ? "Trained artifact could not be prepared"
                                 : preparationError,
        workerDiagnostics);
    {
      const juce::ScopedLock lock(stateLock);
      completedResult = result;
      statusMessage = result.errorMessage;
      status.store(TrainStatus::failed, std::memory_order_release);
    }
    return;
  }

  {
    const juce::ScopedLock lock(stateLock);
    completedResult = result;
    statusMessage = "Training succeeded";
    status.store(TrainStatus::succeeded, std::memory_order_release);
  }
}

void TrainCoordinator::publishFailure(const std::string &requestId,
                                      const std::string &message) {
  graph::TrainJobResult result;
  result.requestId = requestId;
  result.status = "failure";
  result.errorMessage = message;
  {
    const juce::ScopedLock lock(stateLock);
    childProcess.reset();
    latestProgress = result;
    completedResult = result;
    statusMessage = message;
    status.store(TrainStatus::failed, std::memory_order_release);
  }
}
} // namespace openyourbox::train
