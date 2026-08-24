#include "CopyrightAcknowledgment.h"
#include "UserDataPaths.h"

namespace openyourbox::library {
CopyrightAcknowledgment::CopyrightAcknowledgment()
    : CopyrightAcknowledgment(userDataRoot()) {}

CopyrightAcknowledgment::CopyrightAcknowledgment(juce::File rootDirectory)
    : logFile(rootDirectory.getChildFile("copyright-ack.json")) {}

bool CopyrightAcknowledgment::isAcknowledged() const {
  if (!logFile.existsAsFile())
    return false;
  const auto parsed = juce::JSON::parse(logFile.loadFileAsString());
  if (!parsed.isObject())
    return false;
  return parsed.getProperty("acknowledged", false) &&
         parsed.getProperty("textVersion", {}).toString() ==
             juce::String(copyrightTextVersion);
}

bool CopyrightAcknowledgment::acknowledge() {
  logFile.getParentDirectory().createDirectory();
  auto object = std::make_unique<juce::DynamicObject>();
  object->setProperty("acknowledged", true);
  object->setProperty("textVersion", juce::String(copyrightTextVersion));
  object->setProperty(
      "acknowledgedAt",
      juce::Time::getCurrentTime().toISO8601(true));
  return logFile.replaceWithText(
      juce::JSON::toString(juce::var(object.release()), true));
}

juce::String CopyrightAcknowledgment::getCertificationText() {
  return "I certify that all audio samples captured or imported for this "
         "training are my original work or royalty-free.";
}
} // namespace openyourbox::library
