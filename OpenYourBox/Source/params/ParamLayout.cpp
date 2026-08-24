#include "ParamLayout.h"
#include "ParamIDs.h"

#include <limits>

namespace openyourbox::params {
juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout() {
  juce::AudioProcessorValueTreeState::ParameterLayout layout;

  layout.add(std::make_unique<juce::AudioParameterInt>(
      makeID(depth), "Depth", 1, 999, defaultDepth,
      juce::AudioParameterIntAttributes().withLabel("layers")));
  layout.add(std::make_unique<juce::AudioParameterInt>(
      makeID(kernelSize), "Kernel Size", 2, 65, defaultKernelSize,
      juce::AudioParameterIntAttributes().withLabel("samples")));
  layout.add(std::make_unique<juce::AudioParameterInt>(
      makeID(channels), "Channels", 1, 512, defaultChannels));
  layout.add(std::make_unique<juce::AudioParameterChoice>(
      makeID(activation), "Activation",
      juce::StringArray{"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "PReLU"}, 0));
  layout.add(std::make_unique<juce::AudioParameterBool>(makeID(randomize),
                                                        "Randomize", false));
  layout.add(std::make_unique<juce::AudioParameterInt>(
      makeID(randomizeCC), "Randomize MIDI CC", 0, 127, defaultRandomizeCC));
  layout.add(std::make_unique<juce::AudioParameterInt>(
      makeID(globalSeed), "Seed", 0, std::numeric_limits<int>::max(),
      defaultSeed));
  layout.add(std::make_unique<juce::AudioParameterFloat>(
      makeID(dryWet), "Dry/Wet",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f,
      juce::AudioParameterFloatAttributes()
          .withLabel("%")
          .withStringFromValueFunction(
              [](float value, int) { return juce::String(value * 100.0f, 1); })
          .withValueFromStringFunction([](const juce::String &text) {
            return text.getFloatValue() / 100.0f;
          })));

  return layout;
}
} // namespace openyourbox::params
