#pragma once

#include "CloudJobPackage.h"
#include "CloudSettings.h"

#include <JuceHeader.h>

#include <functional>
#include <optional>
#include <vector>

namespace openyourbox::train {
/**
 * @struct CloudApiError
 * @brief Parsed `{error_code, error_message}` from the cloud API.
 */
struct CloudApiError {
  /** @brief Contract error_code token. */
  juce::String code;
  /** @brief User-facing error_message. */
  juce::String message;
  /** @brief HTTP status, or 0 when the request did not complete. */
  int httpStatus = 0;
};

/**
 * @struct CloudLinkStart
 * @brief Device-code bootstrap payload.
 */
struct CloudLinkStart {
  juce::String deviceCode;
  juce::String userCode;
  juce::String verificationUrl;
  int expiresIn = 600;
  int intervalSeconds = 5;
};

/**
 * @struct CloudEntitlement
 * @brief Result of GET /v1/entitlement.
 */
struct CloudEntitlement {
  bool sufficient = false;
  juce::String balanceHint;
};

/**
 * @struct CloudJobSnapshot
 * @brief One job from list or detail.
 */
struct CloudJobSnapshot {
  juce::String jobId;
  juce::String status;
  juce::String objective;
  int step = 0;
  int totalSteps = 0;
  juce::String stage;
  double loss = 0.0;
  juce::String corpusId;
  juce::String errorCode;
  juce::String errorMessage;
  juce::String createdAt;
  juce::String updatedAt;
  bool hasFinalArtifact = false;
};

/**
 * @struct CloudCheckpointInfo
 * @brief One published checkpoint.
 */
struct CloudCheckpointInfo {
  juce::String checkpointId;
  int step = 0;
  juce::String stage;
  juce::String createdAt;
};

/**
 * @struct CloudSubmitProgress
 * @brief Packaging/upload progress for the Train panel (message thread).
 */
struct CloudSubmitProgress {
  enum class Phase { idle, packaging, uploading, accepted };
  Phase phase = Phase::idle;
  juce::int64 bytesSent = 0;
  juce::int64 bytesTotal = 0;
};

/**
 * @class CloudTrainClient
 * @brief HTTPS job API client. All network I/O is off the audio thread.
 */
class CloudTrainClient {
public:
  /**
   * @brief Binds the client to persisted cloud settings.
   * @param settings Linked session and URL overrides.
   */
  explicit CloudTrainClient(CloudSettings &settings);

  /**
   * @brief POST /v1/auth/link/start.
   * @param error Receives a failure.
   */
  [[nodiscard]] std::optional<CloudLinkStart>
  startLink(CloudApiError &error);

  /**
   * @brief POST /v1/auth/link/token and persist the session on success.
   * @param deviceCode Device code from startLink.
   * @param error Receives pending/expired/other failures.
   * @return True when linked.
   */
  bool pollLinkToken(const juce::String &deviceCode, CloudApiError &error);

  /** @brief POST /v1/auth/logout then disconnect locally. */
  void logout();

  /**
   * @brief GET /v1/entitlement and update the local cache.
   * @param error Receives a transport or auth failure.
   */
  [[nodiscard]] std::optional<CloudEntitlement>
  probeEntitlement(CloudApiError &error);

  /**
   * @brief POST /v1/jobs with multipart files or JSON corpus reuse.
   * @param package Assembled job package.
   * @param error Receives gate or transport failures.
   * @param onProgress Optional upload-progress callback (background thread).
   */
  [[nodiscard]] std::optional<CloudJobSnapshot>
  submitJob(const CloudJobPackage &package, CloudApiError &error,
            const std::function<void(juce::int64, juce::int64)> &onProgress = {});

  /**
   * @brief GET /v1/jobs for the linked account.
   * @param error Receives a failure.
   */
  [[nodiscard]] std::optional<std::vector<CloudJobSnapshot>>
  listJobs(CloudApiError &error);

  /**
   * @brief GET /v1/jobs/{id}.
   * @param jobId Job identifier.
   * @param error Receives a failure.
   */
  [[nodiscard]] std::optional<CloudJobSnapshot>
  getJob(const juce::String &jobId, CloudApiError &error);

  /**
   * @brief POST pause/resume/stop.
   * @param jobId Job identifier.
   * @param verb `pause`, `resume`, or `stop`.
   * @param error Receives conflict or transport failures.
   */
  [[nodiscard]] std::optional<CloudJobSnapshot>
  controlJob(const juce::String &jobId, const juce::String &verb,
             CloudApiError &error);

  /**
   * @brief GET /v1/jobs/{id}/checkpoints.
   * @param jobId Job identifier.
   * @param error Receives a failure.
   */
  [[nodiscard]] std::optional<std::vector<CloudCheckpointInfo>>
  listCheckpoints(const juce::String &jobId, CloudApiError &error);

  /**
   * @brief Downloads a checkpoint to the cloud artifact directory.
   * @param jobId Job identifier.
   * @param checkpointId Checkpoint identifier.
   * @param error Receives a failure.
   * @return Local file path on success.
   */
  [[nodiscard]] std::optional<juce::File>
  downloadCheckpoint(const juce::String &jobId, const juce::String &checkpointId,
                     CloudApiError &error);

  /**
   * @brief Downloads the final artifact after success.
   * @param jobId Job identifier.
   * @param error Receives a failure.
   * @return Local file path on success.
   */
  [[nodiscard]] std::optional<juce::File>
  downloadArtifact(const juce::String &jobId, CloudApiError &error);

  /** @brief Opens the storefront URL in the system browser. */
  void openStorefront() const;

  /** @brief Opens an arbitrary http(s) URL in the system browser. */
  static void openExternalUrl(const juce::String &url);

private:
  struct HttpResult {
    int status = 0;
    juce::String body;
    juce::MemoryBlock binary;
    juce::String transportError;
  };

  [[nodiscard]] HttpResult request(const juce::String &method,
                                   const juce::String &path,
                                   const juce::String &jsonBody,
                                   bool withAuth) const;
  [[nodiscard]] HttpResult requestMultipart(const juce::String &path,
                                            const CloudJobPackage &package,
                                            const std::function<void(juce::int64, juce::int64)>
                                                &onProgress) const;
  [[nodiscard]] HttpResult downloadUrl(const juce::String &url) const;
  static bool parseError(const HttpResult &result, CloudApiError &error);
  static CloudJobSnapshot parseJob(const juce::var &object);
  CloudSettings &settings;
};
} // namespace openyourbox::train
