#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <cstdint>

namespace openyourbox::capture {
/**
 * @class CaptureRecorder
 * @brief Real-time-safe input tap that drains to a WAV file on the message thread.
 *
 * The audio thread only writes into a preallocated ring. The message thread
 * drains that ring into a growing WAV file so capture length is unbounded
 * without audio-thread allocations.
 */
class CaptureRecorder {
public:
  /** @brief Number of stereo frames held in the lock-free capture ring. */
  static constexpr int ringFrames = 192000;

  /** @brief Preallocates the capture ring. */
  CaptureRecorder();

  /**
   * @brief Begins writing a new WAV take.
   * @param destination Output file (overwritten).
   * @param sampleRate Host sample rate.
   * @param channels Host input channel count (1 or 2).
   * @return False when the WAV writer could not be opened.
   */
  bool start(const juce::File &destination, double sampleRate, int channels);

  /**
   * @brief Copies planar host input into the ring. Audio-thread safe.
   * @param input Channel pointers from `processBlock`.
   * @param channelCount Number of readable channels.
   * @param numSamples Block size.
   */
  void pushInput(const float *const *input, int channelCount,
                 int numSamples) noexcept;

  /**
   * @brief Drains the ring onto the WAV writer. Message-thread only.
   * @return True while recording is still active.
   */
  bool drain();

  /**
   * @brief Flushes remaining samples and closes the WAV file.
   * @return Destination file when at least one sample was written.
   */
  juce::File stop();

  /** @brief Returns true between `start` and `stop`. */
  [[nodiscard]] bool isActive() const noexcept;

  /** @brief Returns captured duration in seconds after stop, or 0 while recording. */
  [[nodiscard]] double getDurationSeconds() const noexcept;

  /** @brief Returns the destination file set by `start`. */
  [[nodiscard]] juce::File getDestination() const;

  /** @brief Sample rate used for the current take. */
  [[nodiscard]] double getSampleRate() const noexcept;

  /** @brief Channel count used for the current take. */
  [[nodiscard]] int getChannels() const noexcept;

private:
  /** @brief Planar stereo ring storage. */
  std::array<std::array<float, ringFrames>, 2> ring{};
  /** @brief Absolute frames written by the audio thread. */
  std::atomic<std::uint64_t> writeFrames{0};
  /** @brief Absolute frames drained by the message thread. */
  std::atomic<std::uint64_t> readFrames{0};
  /** @brief True while a take is open. */
  std::atomic<bool> active{false};
  /** @brief Destination WAV. */
  juce::File destination;
  /** @brief Host sample rate for the take. */
  double sampleRate = 44100.0;
  /** @brief Channel count for the take. */
  int channels = 2;
  /** @brief Frames actually written to disk. */
  std::uint64_t writtenFrames = 0;
  /** @brief WAV output stream owned for the take. */
  std::unique_ptr<juce::FileOutputStream> outputStream;
  /** @brief WAV writer owned for the take. */
  std::unique_ptr<juce::AudioFormatWriter> writer;
};
} // namespace openyourbox::capture
