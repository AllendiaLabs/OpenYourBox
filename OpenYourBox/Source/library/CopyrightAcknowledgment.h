#pragma once

#include <JuceHeader.h>

namespace openyourbox::library {
/** @brief Disclaimer revision stored with the local acknowledgment log. */
inline constexpr const char *copyrightTextVersion = "1";

/**
 * @class CopyrightAcknowledgment
 * @brief Local persistent gate required before the first Train session.
 *
 * The acknowledgment is stored under plugin user data and is never uploaded.
 */
class CopyrightAcknowledgment {
public:
  /** @brief Binds the log to the default user-data folder. */
  CopyrightAcknowledgment();

  /**
   * @brief Binds the log to an explicit directory (tests).
   * @param rootDirectory Directory that will contain `copyright-ack.json`.
   */
  explicit CopyrightAcknowledgment(juce::File rootDirectory);

  /** @brief Returns true when a matching local acknowledgment exists. */
  [[nodiscard]] bool isAcknowledged() const;

  /**
   * @brief Persists a confirmed acknowledgment for the current disclaimer.
   * @return True when the log was written.
   */
  bool acknowledge();

  /** @brief Certification text shown in the blocking modal. */
  [[nodiscard]] static juce::String getCertificationText();

private:
  /** @brief JSON log file path. */
  juce::File logFile;
};
} // namespace openyourbox::library
