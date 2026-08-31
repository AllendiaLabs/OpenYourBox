#include "CloudSettings.h"

#include "../library/UserDataPaths.h"

namespace openyourbox::train {
namespace {
juce::File defaultStorageFile() {
  return library::userDataRoot().getChildFile("cloud.xml");
}

juce::String maskToken(const juce::String &token) {
  if (token.length() < 8)
    return token.isEmpty() ? juce::String() : juce::String("••••");
  return token.substring(0, 4) + "…" + token.substring(token.length() - 4);
}

juce::String loadString(const juce::XmlElement *root, const juce::Identifier &key) {
  if (root == nullptr)
    return {};
  return root->getStringAttribute(key);
}
} // namespace

CloudSettings::CloudSettings() : CloudSettings(defaultStorageFile()) {}

CloudSettings::CloudSettings(juce::File storage)
    : storageFile(std::move(storage)) {
  storageFile.getParentDirectory().createDirectory();
  if (auto xml = juce::XmlDocument::parse(storageFile)) {
    accessToken = loadString(xml.get(), "accessToken");
    refreshToken = loadString(xml.get(), "refreshToken");
    customerIdHint = loadString(xml.get(), "customerIdHint");
    apiBaseUrlOverride = loadString(xml.get(), "apiBaseUrlOverride");
    storefrontUrlOverride = loadString(xml.get(), "storefrontUrlOverride");
    clientInstanceId = loadString(xml.get(), "clientInstanceId");
    submitterJobIds.addTokens(loadString(xml.get(), "submitterJobIds"), ",", {});
    submitterJobIds.trim();
    submitterJobIds.removeEmptyStrings();
    const auto warn = xml->getStringAttribute("softUploadWarnBytes");
    if (warn.isNotEmpty())
      softUploadWarnBytes = warn.getLargeIntValue();
    if (softUploadWarnBytes <= 0)
      softUploadWarnBytes = graph::defaultCloudSoftUploadWarnBytes;
  }
  loadOrCreateInstanceId();
}

void CloudSettings::loadOrCreateInstanceId() {
  if (clientInstanceId.isNotEmpty())
    return;
  clientInstanceId = juce::Uuid().toString();
  persist();
}

void CloudSettings::persist() const {
  juce::XmlElement root("cloud");
  root.setAttribute("accessToken", accessToken);
  root.setAttribute("refreshToken", refreshToken);
  root.setAttribute("customerIdHint", customerIdHint);
  root.setAttribute("apiBaseUrlOverride", apiBaseUrlOverride);
  root.setAttribute("storefrontUrlOverride", storefrontUrlOverride);
  root.setAttribute("clientInstanceId", clientInstanceId);
  root.setAttribute("submitterJobIds", submitterJobIds.joinIntoString(","));
  root.setAttribute("softUploadWarnBytes", juce::String(softUploadWarnBytes));
  storageFile.getParentDirectory().createDirectory();
  root.writeTo(storageFile);
}

bool CloudSettings::isLinked() const { return accessToken.isNotEmpty() && !authError; }

CloudSettings::LinkStatus CloudSettings::getLinkStatus() const {
  if (authError)
    return LinkStatus::authError;
  if (accessToken.isNotEmpty())
    return LinkStatus::linked;
  return LinkStatus::notLinked;
}

juce::String CloudSettings::getMaskedToken() const { return maskToken(accessToken); }

juce::String CloudSettings::getCustomerIdHint() const { return customerIdHint; }

juce::String CloudSettings::getAccessToken() const { return accessToken; }

juce::String CloudSettings::getRefreshToken() const { return refreshToken; }

void CloudSettings::saveLinkedSession(const juce::String &access,
                                      const juce::String &refresh,
                                      const juce::String &hint) {
  accessToken = access;
  refreshToken = refresh;
  customerIdHint = hint;
  authError = false;
  persist();
}

void CloudSettings::disconnect() {
  accessToken.clear();
  refreshToken.clear();
  customerIdHint.clear();
  authError = false;
  entitlementStatus = EntitlementStatus::unknown;
  entitlementHint.clear();
  persist();
}

void CloudSettings::markAuthError() { authError = true; }

juce::String CloudSettings::getApiBaseUrl() const {
  const auto trimmed = apiBaseUrlOverride.trim();
  return trimmed.isNotEmpty() ? trimmed.trimCharactersAtEnd("/")
                              : juce::String(graph::defaultCloudApiBaseUrl);
}

juce::String CloudSettings::getApiBaseUrlOverride() const { return apiBaseUrlOverride; }

void CloudSettings::setApiBaseUrlOverride(const juce::String &url) {
  apiBaseUrlOverride = url.trim();
  persist();
}

juce::String CloudSettings::getStorefrontUrl() const {
  const auto trimmed = storefrontUrlOverride.trim();
  return trimmed.isNotEmpty() ? trimmed.trimCharactersAtEnd("/")
                              : juce::String(graph::defaultStorefrontUrl);
}

juce::String CloudSettings::getStorefrontUrlOverride() const {
  return storefrontUrlOverride;
}

void CloudSettings::setStorefrontUrlOverride(const juce::String &url) {
  storefrontUrlOverride = url.trim();
  persist();
}

juce::int64 CloudSettings::getSoftUploadWarnBytes() const { return softUploadWarnBytes; }

void CloudSettings::setSoftUploadWarnBytes(juce::int64 bytes) {
  softUploadWarnBytes =
      bytes > 0 ? bytes : graph::defaultCloudSoftUploadWarnBytes;
  persist();
}

juce::String CloudSettings::getClientInstanceId() const { return clientInstanceId; }

void CloudSettings::rememberSubmitter(const juce::String &jobId) {
  if (jobId.isEmpty() || submitterJobIds.contains(jobId))
    return;
  submitterJobIds.add(jobId);
  persist();
}

bool CloudSettings::isSubmitter(const juce::String &jobId) const {
  return jobId.isNotEmpty() && submitterJobIds.contains(jobId);
}

CloudSettings::EntitlementStatus
CloudSettings::getEntitlementStatus() const noexcept {
  return entitlementStatus;
}

juce::String CloudSettings::getEntitlementHint() const { return entitlementHint; }

void CloudSettings::cacheEntitlement(bool sufficient, const juce::String &hint) {
  entitlementStatus = sufficient ? EntitlementStatus::available
                                 : EntitlementStatus::unavailable;
  entitlementHint = hint;
}

void CloudSettings::clearEntitlementCache() {
  entitlementStatus = EntitlementStatus::unknown;
  entitlementHint.clear();
}
} // namespace openyourbox::train
