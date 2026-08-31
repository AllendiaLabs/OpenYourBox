#pragma once

#include "../graph/GraphTypes.h"

#include <JuceHeader.h>

#include <vector>

namespace openyourbox::train {
/**
 * @class CloudSettings
 * @brief Persists a masked platform-account link, URL overrides, and submitter ids.
 *
 * Credentials are stored in a PropertiesFile under the plugin user-data root.
 * The UI must never display a full access token after save. All I/O is
 * message-thread / background only — never call from processBlock.
 */
class CloudSettings {
public:
  /** @brief Last entitlement probe result used for UI hints only. */
  enum class EntitlementStatus {
    unknown,
    available,
    unavailable
  };

  /** @brief Account-link status shown in settings. */
  enum class LinkStatus {
    notLinked,
    linked,
    authError
  };

  /**
   * @brief Loads persisted cloud settings from the default user-data file.
   */
  CloudSettings();

  /**
   * @brief Loads persisted cloud settings from an explicit file (tests).
   * @param storageFile Properties XML file.
   */
  explicit CloudSettings(juce::File storageFile);

  /** @brief True when a non-empty access token is stored. */
  [[nodiscard]] bool isLinked() const;

  /** @brief User-visible link status. */
  [[nodiscard]] LinkStatus getLinkStatus() const;

  /** @brief Masked token preview (`abcd…wxyz`) or empty when unlinked. */
  [[nodiscard]] juce::String getMaskedToken() const;

  /** @brief Opaque customer hint for debug UI; never a password. */
  [[nodiscard]] juce::String getCustomerIdHint() const;

  /** @brief Raw access token for Authorization headers; do not log. */
  [[nodiscard]] juce::String getAccessToken() const;

  /** @brief Optional refresh token. */
  [[nodiscard]] juce::String getRefreshToken() const;

  /**
   * @brief Saves a newly linked session and clears auth-error state.
   * @param accessToken Bearer credential.
   * @param refreshToken Optional refresh credential.
   * @param customerIdHint Optional opaque customer id.
   */
  void saveLinkedSession(const juce::String &accessToken,
                         const juce::String &refreshToken,
                         const juce::String &customerIdHint);

  /** @brief Clears credentials and disables Cloud Run until re-linked. */
  void disconnect();

  /** @brief Records that the stored session was rejected as unauthorized. */
  void markAuthError();

  /** @brief Effective API base URL (override or product default). */
  [[nodiscard]] juce::String getApiBaseUrl() const;

  /** @brief User override; empty means product default. */
  [[nodiscard]] juce::String getApiBaseUrlOverride() const;

  /**
   * @brief Stores an optional API base URL override.
   * @param url Staging origin, or empty to restore the product default.
   */
  void setApiBaseUrlOverride(const juce::String &url);

  /** @brief Effective storefront URL (override or product default). */
  [[nodiscard]] juce::String getStorefrontUrl() const;

  /** @brief User override; empty means product default. */
  [[nodiscard]] juce::String getStorefrontUrlOverride() const;

  /**
   * @brief Stores an optional storefront URL override.
   * @param url Staging origin, or empty to restore the product default.
   */
  void setStorefrontUrlOverride(const juce::String &url);

  /** @brief Soft-upload warning threshold in bytes. */
  [[nodiscard]] juce::int64 getSoftUploadWarnBytes() const;

  /**
   * @brief Updates the soft-upload warning threshold.
   * @param bytes Threshold; values ≤ 0 restore the 2 GiB default.
   */
  void setSoftUploadWarnBytes(juce::int64 bytes);

  /** @brief Stable id for this plugin instance (submitter metadata). */
  [[nodiscard]] juce::String getClientInstanceId() const;

  /**
   * @brief Records that this instance submitted @p jobId.
   * @param jobId Server-assigned job identifier.
   */
  void rememberSubmitter(const juce::String &jobId);

  /**
   * @brief Returns true when this instance submitted @p jobId.
   * @param jobId Server-assigned job identifier.
   */
  [[nodiscard]] bool isSubmitter(const juce::String &jobId) const;

  /** @brief Advisory entitlement status from the last successful probe. */
  [[nodiscard]] EntitlementStatus getEntitlementStatus() const noexcept;

  /** @brief Optional balance hint from the last probe. */
  [[nodiscard]] juce::String getEntitlementHint() const;

  /**
   * @brief Updates the local entitlement cache (never authoritative).
   * @param sufficient Server `sufficient` flag.
   * @param balanceHint Optional display string.
   */
  void cacheEntitlement(bool sufficient, const juce::String &balanceHint);

  /** @brief Marks entitlement unknown (disconnected or probe failed). */
  void clearEntitlementCache();

private:
  void loadOrCreateInstanceId();
  void persist() const;
  [[nodiscard]] juce::StringArray submitterIds() const;

  juce::File storageFile;
  juce::String accessToken;
  juce::String refreshToken;
  juce::String customerIdHint;
  juce::String apiBaseUrlOverride;
  juce::String storefrontUrlOverride;
  juce::String clientInstanceId;
  juce::StringArray submitterJobIds;
  juce::int64 softUploadWarnBytes = graph::defaultCloudSoftUploadWarnBytes;
  bool authError = false;
  EntitlementStatus entitlementStatus = EntitlementStatus::unknown;
  juce::String entitlementHint;
};
} // namespace openyourbox::train
