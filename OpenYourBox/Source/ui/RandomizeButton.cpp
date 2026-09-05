#include "RandomizeButton.h"
#include "../params/ParamIDs.h"
#include "InstrumentWidgets.h"

namespace openyourbox::ui {
void RandomizeButton::render(juce::AudioProcessorValueTreeState &state) const {
  if (!InstrumentWidgets::button("Randomize Weights", ImVec2(-1.0f, 36.0f),
                                 InstrumentButtonKind::primary))
    return;

  if (auto *parameter = state.getParameter(params::randomize)) {
    parameter->beginChangeGesture();
    parameter->setValueNotifyingHost(1.0f);
    parameter->endChangeGesture();
  }
}
} // namespace openyourbox::ui
