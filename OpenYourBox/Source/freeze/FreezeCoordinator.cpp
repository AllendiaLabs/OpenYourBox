#include "FreezeCoordinator.h"

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
 * @brief Writes the embedded Python worker to a private temporary location.
 * @param error Receives a user-facing filesystem failure.
 * @return Runnable script file, or an invalid file on failure.
 */
juce::File materializeWorkerScript(std::string &error) {
  const auto workerDirectory =
      juce::File::getSpecialLocation(juce::File::tempDirectory)
          .getChildFile("OpenYourBox")
          .getChildFile("Worker");
  if (workerDirectory.createDirectory().failed()) {
    error = "Could not create the embedded worker directory";
    return {};
  }

  const auto worker = workerDirectory.getChildFile("freeze_worker.py");
  if (!worker.replaceWithData(BinaryData::freeze_worker_py,
                              BinaryData::freeze_worker_pySize)) {
    error = "Could not materialize the embedded freeze worker";
    return {};
  }
  return worker;
}

/**
 * @brief Parses the final JSON object from worker standard output.
 * @param output Complete captured standard output.
 * @return Parsed object, or void when no valid JSON object was emitted.
 *
 * Parsing the final valid line tolerates incidental standard-output warnings;
 * standard error is intentionally not merged into this stream.
 */
juce::var parseWorkerResponse(const juce::String &output) {
  const auto wholeOutput = juce::JSON::parse(output);
  if (wholeOutput.isObject())
    return wholeOutput;

  juce::StringArray lines;
  lines.addLines(output);
  for (auto index = lines.size(); --index >= 0;) {
    const auto candidate = juce::JSON::parse(lines[index].trim());
    if (candidate.isObject())
      return candidate;
  }
  return {};
}
} // namespace

namespace openyourbox::freeze {
FreezeCoordinator::FreezeCoordinator(ArtifactPreparer artifactPreparer)
    : Thread("OpenYourBox Freeze Worker"),
      prepareArtifact(std::move(artifactPreparer)) {}

FreezeCoordinator::~FreezeCoordinator() {
  signalThreadShouldExit();
  {
    const juce::ScopedLock lock(stateLock);
    if (childProcess != nullptr)
      childProcess->kill();
  }
  stopThread(5000);
}

bool FreezeCoordinator::start(const graph::FreezeSelectionRequest &request) {
  if (request.graphFragment.empty() ||
      status.load(std::memory_order_acquire) != FreezeStatus::idle)
    return false;

  {
    const juce::ScopedLock lock(stateLock);
    pendingRequest = request;
    completedResult.reset();
    statusMessage = "Compiling selected graph...";
  }
  status.store(FreezeStatus::compiling, std::memory_order_release);
  startThread();
  return true;
}

FreezeStatus FreezeCoordinator::getStatus() const noexcept {
  return status.load(std::memory_order_acquire);
}

juce::String FreezeCoordinator::getStatusMessage() const {
  const juce::ScopedLock lock(stateLock);
  return statusMessage;
}

std::optional<graph::FreezeSelectionResult> FreezeCoordinator::takeResult() {
  const juce::ScopedLock lock(stateLock);
  auto result = std::move(completedResult);
  completedResult.reset();
  if (result.has_value())
    status.store(FreezeStatus::idle, std::memory_order_release);
  return result;
}

void FreezeCoordinator::run() {
  graph::FreezeSelectionRequest request;
  {
    const juce::ScopedLock lock(stateLock);
    request = pendingRequest;
  }

  std::string workerError;
  const auto worker = materializeWorkerScript(workerError);
  if (!worker.existsAsFile()) {
    publishFailure(request.requestId, workerError);
    return;
  }

  auto requestFile = juce::File::createTempFile("openyourbox-freeze.json");
  if (!requestFile.replaceWithText(request.graphFragment, false, false, "\n")) {
    publishFailure(request.requestId, "Could not write the freeze request");
    return;
  }

  const auto artifactDirectory =
      juce::File::getSpecialLocation(juce::File::tempDirectory)
          .getChildFile("OpenYourBox")
          .getChildFile("FrozenArtifacts");
  if (artifactDirectory.createDirectory().failed()) {
    publishFailure(request.requestId,
                   "Could not create the artifact directory");
    requestFile.deleteFile();
    return;
  }

  juce::StringArray command;
  command.add(OPENYOURBOX_PYTHON_EXECUTABLE);
  command.add(worker.getFullPathName());
  command.add("--request");
  command.add(requestFile.getFullPathName());
  command.add("--artifact-dir");
  command.add(artifactDirectory.getFullPathName());

  auto process = std::make_unique<juce::ChildProcess>();
  if (!process->start(command, juce::ChildProcess::wantStdOut)) {
    publishFailure(request.requestId, "Could not launch the freeze worker");
    requestFile.deleteFile();
    return;
  }
  {
    const juce::ScopedLock lock(stateLock);
    childProcess = std::move(process);
  }

  juce::String output;
  for (;;) {
    if (threadShouldExit()) {
      const juce::ScopedLock lock(stateLock);
      if (childProcess != nullptr)
        childProcess->kill();
      requestFile.deleteFile();
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
    const auto running = processHandle->isRunning();
    if (!running)
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
  requestFile.deleteFile();

  const auto parsed = parseWorkerResponse(output);
  if (!parsed.isObject()) {
    publishFailure(request.requestId,
                   output.trim().isNotEmpty()
                       ? output.trim().toStdString()
                       : "Freeze worker returned an invalid response");
    return;
  }

  graph::FreezeSelectionResult result;
  result.requestId = jsonString(parsed, "request_id");
  if (exitCode != 0 || result.requestId != request.requestId ||
      jsonString(parsed, "status") != "success") {
    publishFailure(request.requestId,
                   jsonString(parsed, "error_message").empty()
                       ? "Freeze worker rejected the request"
                       : jsonString(parsed, "error_message"));
    return;
  }

  result.artifactPath = jsonString(parsed, "artifact_path");
  const auto metadata = parsed.getProperty("blackbox_metadata", {});
  const auto metrics = metadata.getProperty("baseline_metrics", {});
  const auto shape = metadata.getProperty("shape_signature", {});
  result.baselineMetrics.compileTimeMilliseconds =
      static_cast<double>(metrics.getProperty("compile_time_ms", 0.0));
  result.baselineMetrics.inferenceTimeMilliseconds =
      static_cast<double>(metrics.getProperty("estimated_latency_ms", 0.0));
  result.inputChannels =
      static_cast<int>(shape.getProperty("input_channels", 0));
  result.outputChannels =
      static_cast<int>(shape.getProperty("output_channels", 0));
  result.receptiveFieldSamples =
      static_cast<std::uint64_t>(static_cast<juce::int64>(
          metadata.getProperty("receptive_field_samples", 0)));
  result.acceptsConditioning =
      static_cast<bool>(metadata.getProperty("conditioning", false));
  result.condDim = static_cast<int>(metadata.getProperty("cond_dim", 0));

  std::string preparationError;
  if (result.artifactPath.empty() || prepareArtifact == nullptr ||
      !prepareArtifact(result, preparationError)) {
    publishFailure(request.requestId,
                   preparationError.empty()
                       ? "Compiled artifact could not be prepared"
                       : preparationError);
    return;
  }

  result.succeeded = true;
  {
    const juce::ScopedLock lock(stateLock);
    completedResult = std::move(result);
    statusMessage = "Freeze succeeded";
    status.store(FreezeStatus::succeeded, std::memory_order_release);
  }
}

void FreezeCoordinator::publishFailure(const std::string &requestId,
                                       const std::string &message) {
  graph::FreezeSelectionResult result;
  result.requestId = requestId;
  result.errorMessage = message;
  {
    const juce::ScopedLock lock(stateLock);
    childProcess.reset();
    completedResult = std::move(result);
    statusMessage = message;
    status.store(FreezeStatus::failed, std::memory_order_release);
  }
}
} // namespace openyourbox::freeze
