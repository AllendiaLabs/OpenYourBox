#pragma once

#include "../graph/GraphTypes.h"

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace openyourbox::train {
/** @brief User-visible lifecycle of one Train job. */
enum class TrainStatus {
  idle,
  running,
  paused,
  succeeded,
  failed,
  stopped
};

/**
 * @class TrainCoordinator
 * @brief Runs the Python train worker off-thread with pause/resume/stop.
 *
 * Progress events are parsed on the worker thread and published for the
 * message thread. Artifact preparation is optional and never runs on the
 * audio thread. Stop and failure never request a model swap.
 */
class TrainCoordinator final : private juce::Thread {
public:
  /**
   * @brief Callback that validates a trained artifact off-thread.
   * @param result Worker success payload.
   * @param error Receives a human-readable preparation failure.
   * @return True only after publication has completed.
   */
  using ArtifactPreparer =
      std::function<bool(const graph::TrainJobResult &result, std::string &error)>;

  /**
   * @brief Creates a coordinator for one editor instance.
   * @param artifactPreparer Off-thread processor artifact preparation callback.
   */
  explicit TrainCoordinator(ArtifactPreparer artifactPreparer);

  /** @brief Stops the worker process and joins its thread. */
  ~TrainCoordinator() override;

  /**
   * @brief Starts one train job without blocking the caller.
   * @param request Complete JSON worker request.
   * @return False when another job is already active.
   */
  bool start(const graph::TrainJobRequest &request);

  /** @brief Requests the worker to pause optimization. */
  void pause();

  /** @brief Requests the worker to resume a paused job. */
  void resume();

  /** @brief Requests the worker to stop without exporting a replacement model. */
  void stop();

  /**
   * @brief Retries off-thread load of a previously exported artifact.
   * @param result Prior success payload whose `artifactPath` still exists.
   * @return False when no preparer is installed or the file is missing.
   */
  bool retryLoad(const graph::TrainJobResult &result);

  /** @brief Returns the current lifecycle state without blocking. */
  [[nodiscard]] TrainStatus getStatus() const noexcept;

  /** @brief Returns a short user-facing status message. */
  [[nodiscard]] juce::String getStatusMessage() const;

  /** @brief Returns the latest streamed progress snapshot. */
  [[nodiscard]] graph::TrainJobResult getProgress() const;

  /**
   * @brief Returns the recorded loss curve for the active or last job.
   * @return Copy of scalar losses in step order.
   */
  [[nodiscard]] std::vector<float> getLossHistory() const;

  /**
   * @brief Takes the completed worker result once.
   * @return Terminal result, or no value while work is pending.
   */
  [[nodiscard]] std::optional<graph::TrainJobResult> takeResult();

private:
  void run() override;

  /**
   * @brief Writes a pause/resume/stop command for the worker to poll.
   * @param command Command verb (`pause`, `resume`, or `stop`).
   */
  void writeCommand(const juce::String &command);

  /**
   * @brief Publishes one worker or preparation failure for message-thread use.
   * @param requestId Correlation identifier of the failed request.
   * @param message Human-readable failure detail.
   */
  void publishFailure(const std::string &requestId, const std::string &message);

  /**
   * @brief Applies one streamed JSON object to the published progress snapshot.
   * @param parsed Parsed worker JSON line.
   */
  void applyProgressObject(const juce::var &parsed);

  /** @brief Processor callback used after worker export succeeds. */
  ArtifactPreparer prepareArtifact;
  /** @brief Lock-free lifecycle state polled by the editor. */
  std::atomic<TrainStatus> status{TrainStatus::idle};
  /** @brief Protects request, result, message, and process ownership. */
  mutable juce::CriticalSection stateLock;
  /** @brief Immutable request copied before the worker thread starts. */
  graph::TrainJobRequest pendingRequest;
  /** @brief Latest streamed progress copied for UI. */
  graph::TrainJobResult latestProgress;
  /** @brief Scalar losses appended on the worker thread for the live plot. */
  std::vector<float> lossHistory;
  /** @brief Single completion consumed by the message thread. */
  std::optional<graph::TrainJobResult> completedResult;
  /** @brief User-facing lifecycle or failure text. */
  juce::String statusMessage{"Ready"};
  /** @brief Command file polled by the worker. */
  juce::File commandFile;
  /** @brief Killable child process owned for the active request. */
  std::unique_ptr<juce::ChildProcess> childProcess;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TrainCoordinator)
};
} // namespace openyourbox::train
