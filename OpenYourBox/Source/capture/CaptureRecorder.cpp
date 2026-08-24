#include "CaptureRecorder.h"

namespace openyourbox::capture {
CaptureRecorder::CaptureRecorder() = default;

bool CaptureRecorder::start(const juce::File &file, double rate, int channelCount) {
  stop();
  destination = file;
  sampleRate = rate > 0.0 ? rate : 44100.0;
  channels = std::clamp(channelCount, 1, 2);
  writtenFrames = 0;
  writeFrames.store(0, std::memory_order_release);
  readFrames.store(0, std::memory_order_release);
  destination.getParentDirectory().createDirectory();
  destination.deleteFile();
  std::unique_ptr<juce::OutputStream> stream = destination.createOutputStream();
  if (stream == nullptr)
    return false;
  juce::WavAudioFormat wav;
  auto options =
      juce::AudioFormatWriterOptions{}
          .withSampleRate(sampleRate)
          .withNumChannels(channels)
          .withBitsPerSample(32)
          .withSampleFormat(
              juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
  writer = wav.createWriterFor(stream, options);
  if (writer == nullptr)
    return false;
  outputStream.reset();
  active.store(true, std::memory_order_release);
  return true;
}

void CaptureRecorder::pushInput(const float *const *input, int channelCount,
                                int numSamples) noexcept {
  if (!active.load(std::memory_order_acquire) || input == nullptr ||
      numSamples <= 0)
    return;
  const auto usableChannels = std::min(channelCount, channels);
  auto write = writeFrames.load(std::memory_order_relaxed);
  const auto read = readFrames.load(std::memory_order_acquire);
  if (write - read + static_cast<std::uint64_t>(numSamples) >
      static_cast<std::uint64_t>(ringFrames - 1))
    return;

  for (int sample = 0; sample < numSamples; ++sample) {
    const auto index = static_cast<int>(write % static_cast<std::uint64_t>(ringFrames));
    ring[0][static_cast<std::size_t>(index)] =
        usableChannels > 0 && input[0] != nullptr ? input[0][sample] : 0.0f;
    ring[1][static_cast<std::size_t>(index)] =
        usableChannels > 1 && input[1] != nullptr ? input[1][sample]
                                                  : ring[0][static_cast<std::size_t>(index)];
    ++write;
  }
  writeFrames.store(write, std::memory_order_release);
}

bool CaptureRecorder::drain() {
  if (!active.load(std::memory_order_acquire) || writer == nullptr)
    return false;
  const auto write = writeFrames.load(std::memory_order_acquire);
  auto read = readFrames.load(std::memory_order_relaxed);
  if (read >= write)
    return true;

  const auto available = static_cast<int>(write - read);
  juce::AudioBuffer<float> block(channels, available);
  for (int sample = 0; sample < available; ++sample) {
    const auto index =
        static_cast<int>(read % static_cast<std::uint64_t>(ringFrames));
    block.setSample(0, sample, ring[0][static_cast<std::size_t>(index)]);
    if (channels > 1)
      block.setSample(1, sample, ring[1][static_cast<std::size_t>(index)]);
    ++read;
  }
  writer->writeFromAudioSampleBuffer(block, 0, available);
  writtenFrames += static_cast<std::uint64_t>(available);
  readFrames.store(read, std::memory_order_release);
  return true;
}

juce::File CaptureRecorder::stop() {
  drain();
  writer.reset();
  outputStream.reset();
  active.store(false, std::memory_order_release);
  return writtenFrames > 0 ? destination : juce::File{};
}

bool CaptureRecorder::isActive() const noexcept {
  return active.load(std::memory_order_acquire);
}

double CaptureRecorder::getDurationSeconds() const noexcept {
  return sampleRate > 0.0 ? static_cast<double>(writtenFrames) / sampleRate : 0.0;
}

juce::File CaptureRecorder::getDestination() const { return destination; }

double CaptureRecorder::getSampleRate() const noexcept { return sampleRate; }

int CaptureRecorder::getChannels() const noexcept { return channels; }
} // namespace openyourbox::capture
