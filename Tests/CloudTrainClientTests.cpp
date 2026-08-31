#include "graph/GraphTypes.h"
#include "train/CloudJobPackage.h"
#include "train/CloudSettings.h"

#include <iostream>

namespace {
/**
 * @brief Reports a failed invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}
} // namespace

/**
 * @brief Covers submitter-only auto-load persistence and soft-warn byte math.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::defaultCloudSoftUploadWarnBytes;
  using openyourbox::train::CloudSettings;
  using openyourbox::train::exceedsSoftUploadWarn;

  bool passed = true;
  passed &= expect(!exceedsSoftUploadWarn(0, defaultCloudSoftUploadWarnBytes),
                   "empty corpus is under the 2 GiB warn");
  passed &= expect(!exceedsSoftUploadWarn(defaultCloudSoftUploadWarnBytes,
                                          defaultCloudSoftUploadWarnBytes),
                   "exactly 2 GiB does not warn");
  passed &= expect(exceedsSoftUploadWarn(defaultCloudSoftUploadWarnBytes + 1,
                                         defaultCloudSoftUploadWarnBytes),
                   "one byte over 2 GiB warns");
  passed &= expect(exceedsSoftUploadWarn(100, 50),
                   "custom threshold warns when exceeded");
  passed &= expect(!exceedsSoftUploadWarn(100, 0),
                   "non-positive threshold falls back to 2 GiB default");

  const auto temp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                        .getChildFile("OpenYourBoxCloudTests")
                        .getChildFile(juce::Uuid().toString())
                        .getChildFile("cloud.xml");
  temp.getParentDirectory().createDirectory();
  CloudSettings settings(temp);
  passed &= expect(!settings.isSubmitter("job-1"),
                   "unknown job is not a submitter");
  settings.rememberSubmitter("job-1");
  passed &= expect(settings.isSubmitter("job-1"),
                   "submitting instance remembers job-1");
  passed &= expect(!settings.isSubmitter("job-2"),
                   "other jobs stay non-submitter");

  CloudSettings reloaded(temp);
  passed &= expect(reloaded.isSubmitter("job-1"),
                   "submitter flag survives reload");
  passed &= expect(!reloaded.isSubmitter("job-2"),
                   "non-submitter stays false after reload");
  reloaded.disconnect();
  passed &= expect(!reloaded.isLinked(), "disconnect clears the linked session");

  temp.getParentDirectory().deleteRecursively();
  if (!passed)
    return 1;
  std::cout << "CloudTrainClientTests passed\n";
  return 0;
}
