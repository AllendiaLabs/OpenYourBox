#pragma once

#include "../graph/GraphTypes.h"
#include "CloudJobPackage.h"
#include "CloudSettings.h"
#include "CloudTrainClient.h"

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
  queued,
  running,
  paused,
  succeeded,
  failed,
  stopped
};

/**
 * @class TrainCoordinator
 * @brief Runs the Python train worker off-thread with run/stop.
 *
 * Progress events are parsed on the worker thread and published for the
 * message thread. Artifact preparation is optional and never runs on the
 * audio thread. Stop and failure never request a model swap. Cloud jobs use
 * the same busy gate and status atom as the local ChildProcess path.
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
   * @brief Binds cloud HTTP client and persisted settings (editor lifecycle).
   * @param client HTTPS client, or null to disable cloud.
   * @param cloudSettings Linked session and submitter map.
   */
  void setCloudDependencies(CloudTrainClient *client, CloudSettings *cloudSettings);

  /**
   * @brief Sets Local vs Cloud destination for the next Run.
   * @param destination Train destination.
   */
  void setDestination(graph::TrainDestination destination) noexcept;

  /** @brief Returns the destination of the active or last job. */
  [[nodiscard]] graph::TrainDestination getDestination() const noexcept;

  /**
   * @brief Starts one local train job without blocking the caller.
   * @param request Complete JSON worker request.
   * @return False when another job is already active.
   */
  bool start(const graph::TrainJobRequest &request);

  /**
   * @brief Packages and submits a cloud job on a background thread.
   * @param package Manifest plus corpus files.
   * @return False when another job is already active or cloud is unbound.
   */
  bool startCloud(CloudJobPackage package);

  /**
   * @brief Attaches to an existing account job (rediscovery).
   * @param jobId Server job id.
   * @param snapshot Optional last-known snapshot.
   * @return False when another job is already active.
   */
  bool attachCloudJob(const juce::String &jobId,
                      const CloudJobSnapshot &snapshot = {});

  /** @brief Polls cloud status when a remote job is attached (message thread). */
  void pollCloud();

  /** @brief Requests the worker to stop without exporting a replacement model. */
  void stop();

  /**
   * @brief Retries off-thread load of a previously exported artifact.
   * @param result Prior success payload whose `artifactPath` still exists.
   * @return False when no preparer is installed or the file is missing.
   */
  bool retryLoad(const graph::TrainJobResult &result);

  /**
   * @brief Downloads a cloud checkpoint and optionally exposes it as progress.
   * @param checkpointId Server checkpoint id.
   * @return Local file when the download succeeded.
   */
  std::optional<juce::File> downloadCloudCheckpoint(const juce::String &checkpointId);

  /** @brief Returns published cloud checkpoints for the attached job. */
  [[nodiscard]] std::vector<CloudCheckpointInfo> getCloudCheckpoints() const;

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

  /** @brief True when this instance submitted the attached cloud job. */
  [[nodiscard]] bool isCloudSubmitter() const;

  /** @brief True when success should auto-load Gold (local or submitter). */
  [[nodiscard]] bool shouldAutoLoadOnSuccess() const;

  /** @brief True when cloud poll failed due to network (job not cancelled). */
  [[nodiscard]] bool isOffline() const noexcept;

  /** @brief Packaging/upload progress for Cloud Run. */
  [[nodiscard]] CloudSubmitProgress getSubmitProgress() const;

  /** @brief Attached cloud job id, or empty. */
  [[nodiscard]] juce::String getCloudJobId() const;

  /** @brief True when non-submitter success offers manual download/load. */
  [[nodiscard]] bool hasManualCloudLoad() const noexcept;

private:
  void run() override;

  /**
   * @brief Writes a stop command for the worker to poll.
   * @param command Command verb (`stop`).
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

  /**
   * @brief Maps a remote job snapshot onto TrainStatus and progress.
   * @param snapshot Cloud job payload.
   */
  void applyCloudSnapshot(const CloudJobSnapshot &snapshot);
  /**
   * @brief Downloads (when submitter) and publishes a successful cloud result.
   * @param artifact Local final artifact, or empty for non-submitter success.
   */
  void finishCloudSuccess(const juce::File &artifact);
  /**
   * @brief Returns true for queued or running.
   * @param value Lifecycle state.
   */
  [[nodiscard]] bool isBusyStatus(TrainStatus value) const noexcept;
  /**
   * @brief Sends stop to the cloud API off-thread.
   */
  void cloudStop();

  /** @brief Processor callback used after worker export succeeds. */
  ArtifactPreparer prepareArtifact;
  CloudTrainClient *cloudClient = nullptr;
  CloudSettings *cloudSettings = nullptr;
  std::atomic<graph::TrainDestination> destination{graph::TrainDestination::local};
  /** @brief Lock-free lifecycle state polled by the editor. */
  std::atomic<TrainStatus> status{TrainStatus::idle};
  std::atomic<bool> offline{false};
  std::atomic<bool> cloudPollInFlight{false};
  std::atomic<bool> manualCloudLoad{false};
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
  juce::String cloudJobId;
  CloudSubmitProgress submitProgress;
  std::vector<CloudCheckpointInfo> cloudCheckpoints;
  juce::int64 lastCloudPollMs = 0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TrainCoordinator)
};
} // namespace openyourbox::train
