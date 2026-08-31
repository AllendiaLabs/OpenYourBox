#include "CloudTrainClient.h"

#include "../library/UserDataPaths.h"

#include <cstring>

namespace openyourbox::train {
namespace {
juce::String jsonString(const juce::var &object, const char *key) {
  return object.getProperty(key, {}).toString();
}

void applyAuthHeader(juce::String &headers, const CloudSettings &settings, bool withAuth) {
  if (!withAuth)
    return;
  const auto token = settings.getAccessToken();
  if (token.isNotEmpty())
    headers << "Authorization: Bearer " << token << "\r\n";
}

juce::MemoryBlock buildMultipart(const CloudJobPackage &package, juce::String &contentType) {
  const auto boundary = "----OpenYourBoxCloud" + juce::Uuid().toString().removeCharacters("{}-");
  contentType = "multipart/form-data; boundary=" + boundary;
  juce::MemoryOutputStream stream;
  const auto dash = "--" + boundary;
  auto writePart = [&](const juce::String &disposition, const void *data, size_t size,
                       const juce::String &partType) {
    stream << dash << "\r\n";
    stream << "Content-Disposition: form-data; " << disposition << "\r\n";
    if (partType.isNotEmpty())
      stream << "Content-Type: " << partType << "\r\n";
    stream << "\r\n";
    stream.write(data, size);
    stream << "\r\n";
  };
  const auto manifestText = juce::JSON::toString(package.manifest, true);
  const auto manifestUtf8 = manifestText.toRawUTF8();
  writePart("name=\"manifest\"", manifestUtf8, std::strlen(manifestUtf8), "application/json");
  if (package.corpusId.isNotEmpty()) {
    const auto idUtf8 = package.corpusId.toRawUTF8();
    writePart("name=\"corpus_id\"", idUtf8, std::strlen(idUtf8), "text/plain");
  }
  for (const auto &part : package.files) {
    juce::MemoryBlock fileData;
    part.file.loadFileAsData(fileData);
    const auto disposition = "name=\"file:" + part.partName + "\"; filename=\"" +
                             part.partName + "\"";
    writePart(disposition, fileData.getData(), fileData.getSize(),
              "application/octet-stream");
  }
  stream << dash << "--\r\n";
  return stream.getMemoryBlock();
}
} // namespace

CloudTrainClient::CloudTrainClient(CloudSettings &settingsToUse)
    : settings(settingsToUse) {}

CloudTrainClient::HttpResult
CloudTrainClient::request(const juce::String &method, const juce::String &path,
                          const juce::String &jsonBody, bool withAuth) const {
  HttpResult result;
  auto url = juce::URL(settings.getApiBaseUrl() + path);
  juce::String headers;
  applyAuthHeader(headers, settings, withAuth);
  juce::URL::ParameterHandling handling = juce::URL::ParameterHandling::inAddress;
  if (jsonBody.isNotEmpty()) {
    headers << "Content-Type: application/json\r\n";
    url = url.withPOSTData(jsonBody);
    handling = juce::URL::ParameterHandling::inPostData;
  }
  const auto options = juce::URL::InputStreamOptions(handling)
                           .withExtraHeaders(headers)
                           .withConnectionTimeoutMs(30000)
                           .withHttpRequestCmd(method)
                           .withStatusCode(&result.status);
  if (auto stream = url.createInputStream(options)) {
    result.body = stream->readEntireStreamAsString();
  } else {
    result.transportError = "Could not reach the cloud training service.";
  }
  return result;
}

CloudTrainClient::HttpResult CloudTrainClient::requestMultipart(
    const juce::String &path, const CloudJobPackage &package,
    const std::function<void(juce::int64, juce::int64)> &onProgress) const {
  HttpResult result;
  juce::String contentType;
  const auto body = buildMultipart(package, contentType);
  const auto total = static_cast<juce::int64>(body.getSize());
  if (onProgress)
    onProgress(0, total);
  juce::String headers;
  applyAuthHeader(headers, settings, true);
  headers << "Content-Type: " << contentType << "\r\n";
  auto posting = juce::URL(settings.getApiBaseUrl() + path).withPOSTData(body);
  const auto options =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inPostData)
          .withExtraHeaders(headers)
          .withConnectionTimeoutMs(120000)
          .withHttpRequestCmd("POST")
          .withStatusCode(&result.status);
  if (auto stream = posting.createInputStream(options)) {
    result.body = stream->readEntireStreamAsString();
    if (onProgress)
      onProgress(total, total);
  } else {
    result.transportError = "Could not upload the cloud training package.";
  }
  return result;
}

CloudTrainClient::HttpResult
CloudTrainClient::downloadUrl(const juce::String &urlText) const {
  HttpResult result;
  auto url = juce::URL(urlText);
  const auto options =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
          .withConnectionTimeoutMs(60000)
          .withHttpRequestCmd("GET")
          .withStatusCode(&result.status);
  if (auto stream = url.createInputStream(options)) {
    juce::MemoryBlock block;
    stream->readIntoMemoryBlock(block);
    result.body = juce::String::fromUTF8(static_cast<const char *>(block.getData()),
                                         static_cast<int>(block.getSize()));
    result.binary = std::move(block);
  } else {
    result.transportError = "Could not download the cloud artifact.";
  }
  return result;
}

bool CloudTrainClient::parseError(const HttpResult &result, CloudApiError &error) {
  error.httpStatus = result.status;
  if (result.transportError.isNotEmpty() && result.body.isEmpty()) {
    error.code = "capacity";
    error.message = result.transportError;
    return true;
  }
  const auto parsed = juce::JSON::parse(result.body);
  error.code = jsonString(parsed, "error_code");
  error.message = jsonString(parsed, "error_message");
  if (result.status >= 400 || result.status == 0) {
    if (error.code.isEmpty() && result.status == 401)
      error.code = graph::cloudErrorCode::unauthorized;
    if (error.message.isEmpty())
      error.message = result.transportError.isNotEmpty()
                          ? result.transportError
                          : "Cloud request failed.";
    return true;
  }
  return false;
}

CloudJobSnapshot CloudTrainClient::parseJob(const juce::var &object) {
  CloudJobSnapshot job;
  job.jobId = jsonString(object, "job_id");
  job.status = jsonString(object, "status");
  job.objective = jsonString(object, "objective");
  job.step = static_cast<int>(object.getProperty("step", 0));
  job.totalSteps = static_cast<int>(object.getProperty("total_steps", 0));
  job.stage = jsonString(object, "stage");
  job.loss = static_cast<double>(object.getProperty("loss", 0.0));
  job.corpusId = jsonString(object, "corpus_id");
  job.errorCode = jsonString(object, "error_code");
  job.errorMessage = jsonString(object, "error_message");
  job.createdAt = jsonString(object, "created_at");
  job.updatedAt = jsonString(object, "updated_at");
  job.hasFinalArtifact =
      static_cast<bool>(object.getProperty("has_final_artifact", false));
  return job;
}

std::optional<CloudLinkStart> CloudTrainClient::startLink(CloudApiError &error) {
  const auto result = request("POST", "/v1/auth/link/start", "{}", false);
  if (parseError(result, error))
    return std::nullopt;
  const auto parsed = juce::JSON::parse(result.body);
  CloudLinkStart start;
  start.deviceCode = jsonString(parsed, "device_code");
  start.userCode = jsonString(parsed, "user_code");
  start.verificationUrl = jsonString(parsed, "verification_url");
  start.expiresIn = static_cast<int>(parsed.getProperty("expires_in", 600));
  start.intervalSeconds = static_cast<int>(parsed.getProperty("interval", 5));
  return start;
}

bool CloudTrainClient::pollLinkToken(const juce::String &deviceCode,
                                     CloudApiError &error) {
  auto body = std::make_unique<juce::DynamicObject>();
  body->setProperty("device_code", deviceCode);
  const auto result =
      request("POST", "/v1/auth/link/token",
              juce::JSON::toString(juce::var(body.release())), false);
  if (parseError(result, error))
    return false;
  const auto parsed = juce::JSON::parse(result.body);
  const auto access = jsonString(parsed, "access_token");
  if (access.isEmpty()) {
    error.code = graph::cloudErrorCode::linkPending;
    error.message = "Waiting for Allendia verification.";
    return false;
  }
  settings.saveLinkedSession(access, jsonString(parsed, "refresh_token"),
                             jsonString(parsed, "customer_id"));
  return true;
}

void CloudTrainClient::logout() {
  if (settings.getAccessToken().isNotEmpty()) {
    CloudApiError ignored;
    (void)request("POST", "/v1/auth/logout", "{}", true);
  }
  settings.disconnect();
}

std::optional<CloudEntitlement>
CloudTrainClient::probeEntitlement(CloudApiError &error) {
  const auto result = request("GET", "/v1/entitlement", {}, true);
  if (parseError(result, error)) {
    if (error.code == graph::cloudErrorCode::unauthorized)
      settings.markAuthError();
    settings.clearEntitlementCache();
    return std::nullopt;
  }
  const auto parsed = juce::JSON::parse(result.body);
  CloudEntitlement entitlement;
  entitlement.sufficient =
      static_cast<bool>(parsed.getProperty("sufficient", false));
  entitlement.balanceHint = jsonString(parsed, "balance_hint");
  settings.cacheEntitlement(entitlement.sufficient, entitlement.balanceHint);
  return entitlement;
}

std::optional<CloudJobSnapshot>
CloudTrainClient::submitJob(const CloudJobPackage &package, CloudApiError &error,
                            const std::function<void(juce::int64, juce::int64)> &onProgress) {
  HttpResult result;
  if (package.corpusId.isNotEmpty() && package.files.empty()) {
    auto body = std::make_unique<juce::DynamicObject>();
    body->setProperty("manifest", package.manifest);
    body->setProperty("corpus_id", package.corpusId);
    result = request("POST", "/v1/jobs", juce::JSON::toString(juce::var(body.release())),
                     true);
  } else {
    result = requestMultipart("/v1/jobs", package, onProgress);
  }
  if (parseError(result, error)) {
    if (error.code == graph::cloudErrorCode::unauthorized)
      settings.markAuthError();
    return std::nullopt;
  }
  const auto parsed = juce::JSON::parse(result.body);
  auto job = parseJob(parsed);
  if (job.jobId.isEmpty())
    job.jobId = jsonString(parsed, "job_id");
  if (job.status.isEmpty())
    job.status = jsonString(parsed, "status");
  if (job.corpusId.isEmpty())
    job.corpusId = jsonString(parsed, "corpus_id");
  if (job.jobId.isEmpty()) {
    error.code = graph::cloudErrorCode::validationFailed;
    error.message = "Cloud submit did not return a job id.";
    return std::nullopt;
  }
  settings.rememberSubmitter(job.jobId);
  return job;
}

std::optional<std::vector<CloudJobSnapshot>>
CloudTrainClient::listJobs(CloudApiError &error) {
  const auto result = request("GET", "/v1/jobs", {}, true);
  if (parseError(result, error)) {
    if (error.code == graph::cloudErrorCode::unauthorized)
      settings.markAuthError();
    return std::nullopt;
  }
  const auto parsed = juce::JSON::parse(result.body);
  std::vector<CloudJobSnapshot> jobs;
  if (auto *list = parsed.getProperty("jobs", {}).getArray()) {
    for (const auto &item : *list)
      jobs.push_back(parseJob(item));
  }
  return jobs;
}

std::optional<CloudJobSnapshot> CloudTrainClient::getJob(const juce::String &jobId,
                                                         CloudApiError &error) {
  const auto result = request("GET", "/v1/jobs/" + jobId, {}, true);
  if (parseError(result, error)) {
    if (error.code == graph::cloudErrorCode::unauthorized)
      settings.markAuthError();
    return std::nullopt;
  }
  return parseJob(juce::JSON::parse(result.body));
}

std::optional<CloudJobSnapshot>
CloudTrainClient::controlJob(const juce::String &jobId, const juce::String &verb,
                             CloudApiError &error) {
  const auto result =
      request("POST", "/v1/jobs/" + jobId + "/" + verb, "{}", true);
  if (parseError(result, error)) {
    if (error.code == graph::cloudErrorCode::unauthorized)
      settings.markAuthError();
    return std::nullopt;
  }
  return parseJob(juce::JSON::parse(result.body));
}

std::optional<std::vector<CloudCheckpointInfo>>
CloudTrainClient::listCheckpoints(const juce::String &jobId, CloudApiError &error) {
  const auto result = request("GET", "/v1/jobs/" + jobId + "/checkpoints", {}, true);
  if (parseError(result, error))
    return std::nullopt;
  const auto parsed = juce::JSON::parse(result.body);
  std::vector<CloudCheckpointInfo> checkpoints;
  if (auto *list = parsed.getProperty("checkpoints", {}).getArray()) {
    for (const auto &item : *list) {
      CloudCheckpointInfo info;
      info.checkpointId = jsonString(item, "checkpoint_id");
      info.step = static_cast<int>(item.getProperty("step", 0));
      info.stage = jsonString(item, "stage");
      info.createdAt = jsonString(item, "created_at");
      checkpoints.push_back(std::move(info));
    }
  }
  return checkpoints;
}

std::optional<juce::File>
CloudTrainClient::downloadCheckpoint(const juce::String &jobId,
                                     const juce::String &checkpointId,
                                     CloudApiError &error) {
  const auto result = request(
      "GET", "/v1/jobs/" + jobId + "/checkpoints/" + checkpointId + "/download", {},
      true);
  if (parseError(result, error))
    return std::nullopt;
  const auto parsed = juce::JSON::parse(result.body);
  const auto signedUrl = jsonString(parsed, "url");
  if (signedUrl.isEmpty()) {
    error.message = "Checkpoint download URL missing.";
    return std::nullopt;
  }
  const auto bytes = downloadUrl(signedUrl);
  if (bytes.binary.isEmpty() && bytes.body.isEmpty()) {
    error.message = bytes.transportError.isNotEmpty()
                        ? bytes.transportError
                        : "Could not download the checkpoint.";
    return std::nullopt;
  }
  auto file = library::cloudArtifactsDirectory().getChildFile(jobId).getChildFile(
      checkpointId + ".pt");
  file.getParentDirectory().createDirectory();
  const auto *data =
      bytes.binary.getSize() > 0 ? bytes.binary.getData() : bytes.body.toRawUTF8();
  const auto size = bytes.binary.getSize() > 0
                        ? bytes.binary.getSize()
                        : static_cast<size_t>(bytes.body.getNumBytesAsUTF8());
  file.replaceWithData(data, size);
  return file;
}

std::optional<juce::File>
CloudTrainClient::downloadArtifact(const juce::String &jobId, CloudApiError &error) {
  const auto result =
      request("GET", "/v1/jobs/" + jobId + "/artifact/download", {}, true);
  if (parseError(result, error))
    return std::nullopt;
  const auto parsed = juce::JSON::parse(result.body);
  const auto signedUrl = jsonString(parsed, "url");
  if (signedUrl.isEmpty()) {
    error.message = "Artifact download URL missing.";
    return std::nullopt;
  }
  const auto bytes = downloadUrl(signedUrl);
  if (bytes.binary.isEmpty() && bytes.body.isEmpty()) {
    error.message = "Could not download the trained artifact. Retry.";
    return std::nullopt;
  }
  auto file = library::cloudArtifactsDirectory().getChildFile(jobId).getChildFile(
      jobId + ".pt");
  file.getParentDirectory().createDirectory();
  const auto *data =
      bytes.binary.getSize() > 0 ? bytes.binary.getData() : bytes.body.toRawUTF8();
  const auto size = bytes.binary.getSize() > 0
                        ? bytes.binary.getSize()
                        : static_cast<size_t>(bytes.body.getNumBytesAsUTF8());
  if (!file.replaceWithData(data, size)) {
    error.message = "Could not download the trained artifact. Retry.";
    return std::nullopt;
  }
  return file;
}

void CloudTrainClient::openStorefront() const {
  openExternalUrl(settings.getStorefrontUrl());
}

void CloudTrainClient::openExternalUrl(const juce::String &url) {
  juce::URL(url).launchInDefaultBrowser();
}
} // namespace openyourbox::train
