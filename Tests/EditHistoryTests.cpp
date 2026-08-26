#include "state/EditHistory.h"
#include "state/PatchSnapshot.h"

#include <JuceHeader.h>

#include <iostream>

namespace {
/**
 * @brief Reports a failed history invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Builds a distinguishable in-memory snapshot.
 * @param marker Integer stored on both parameter and graph trees.
 */
openyourbox::state::PatchSnapshot makeSnapshot(int marker) {
  openyourbox::state::PatchSnapshot snapshot;
  snapshot.parameterState = juce::ValueTree{"OpenYourBoxState"};
  snapshot.parameterState.setProperty("marker", marker, nullptr);
  snapshot.graphDocument = juce::ValueTree{"GraphDocument"};
  snapshot.graphDocument.setProperty("marker", marker, nullptr);
  return snapshot;
}

/**
 * @brief Builds a current-preset association for history tests.
 * @param name Display name.
 * @param dirty Dirty flag.
 */
openyourbox::state::CurrentPresetState makeCurrent(const juce::String &name,
                                                   bool dirty) {
  openyourbox::state::CurrentPresetState current;
  current.entryId = name.isEmpty() ? juce::String{} : "id-" + name;
  current.name = name;
  current.dirty = dirty;
  current.baselineFingerprint = name;
  return current;
}
} // namespace

/**
 * @brief Runs EditHistory depth, coalesce, redo-clear, and CurrentPreset tests.
 * @return Zero when every invariant passes.
 */
int main() {
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  openyourbox::state::PatchSnapshot live = makeSnapshot(0);
  openyourbox::state::CurrentPresetState liveCurrent;
  openyourbox::state::EditHistory history(50);
  history.setApplyFn([&](const openyourbox::state::PatchSnapshot &snapshot,
                         const openyourbox::state::CurrentPresetState &current) {
    live = snapshot;
    liveCurrent = current;
    return true;
  });

  for (int step = 1; step <= 5; ++step) {
    history.pushStep("edit", makeSnapshot(step - 1), makeSnapshot(step), {}, {});
  }
  passed &= expect(history.getUndoDepth() == 5, "five discrete edits must stack");
  passed &= expect(history.canUndo() && !history.canRedo(),
                   "fresh history can undo and cannot redo");

  passed &= expect(history.undo() &&
                       static_cast<int>(live.graphDocument["marker"]) == 4,
                   "undo must apply the previous snapshot");
  passed &= expect(history.undo() &&
                       static_cast<int>(live.graphDocument["marker"]) == 3,
                   "second undo must walk back one more step");
  passed &= expect(history.canRedo(), "undo must enable redo");
  passed &= expect(history.redo() &&
                       static_cast<int>(live.graphDocument["marker"]) == 4,
                   "redo must restore the undone snapshot");

  history.pushStep("new", makeSnapshot(4), makeSnapshot(10), {}, {});
  passed &= expect(!history.canRedo(), "a new edit after undo must clear redo");
  passed &= expect(history.undo() &&
                       static_cast<int>(live.graphDocument["marker"]) == 4,
                   "undo after a new edit must restore the pre-edit snapshot");

  openyourbox::state::EditHistory gestures(50);
  openyourbox::state::PatchSnapshot gestureLive = makeSnapshot(0);
  gestures.setApplyFn([&](const openyourbox::state::PatchSnapshot &snapshot,
                          const openyourbox::state::CurrentPresetState &) {
    gestureLive = snapshot;
    return true;
  });
  gestures.beginGesture("knob", makeSnapshot(0), {});
  gestures.endGesture(makeSnapshot(7), {});
  passed &= expect(gestures.getUndoDepth() == 1,
                   "a coalesced gesture must be one step");
  passed &= expect(gestures.undo() &&
                       static_cast<int>(gestureLive.graphDocument["marker"]) == 0,
                   "undo of a gesture must restore the pre-gesture snapshot");

  gestures.beginGesture("cancelled", makeSnapshot(0), {});
  gestures.cancelGesture();
  passed &= expect(gestures.getUndoDepth() == 0,
                   "a cancelled gesture must not push a step");

  openyourbox::state::EditHistory capped(50);
  openyourbox::state::PatchSnapshot cappedLive = makeSnapshot(0);
  capped.setApplyFn([&](const openyourbox::state::PatchSnapshot &snapshot,
                        const openyourbox::state::CurrentPresetState &) {
    cappedLive = snapshot;
    return true;
  });
  for (int step = 1; step <= 55; ++step)
    capped.pushStep("cap", makeSnapshot(step - 1), makeSnapshot(step), {}, {});
  passed &= expect(capped.getMaxDepth() == 50, "default depth must be 50");
  passed &= expect(capped.getUndoDepth() == 50, "oldest steps must drop at cap");
  int undoCount = 0;
  while (capped.canUndo()) {
    if (!capped.undo())
      break;
    ++undoCount;
  }
  passed &= expect(undoCount == 50, "exactly 50 steps must remain after capping");
  passed &= expect(static_cast<int>(cappedLive.graphDocument["marker"]) == 5,
                   "dropping oldest steps must keep the newest 50");

  openyourbox::state::EditHistory presetHistory(50);
  openyourbox::state::PatchSnapshot presetLive = makeSnapshot(1);
  openyourbox::state::CurrentPresetState presetCurrent = makeCurrent("A", true);
  presetHistory.setApplyFn([&](const openyourbox::state::PatchSnapshot &snapshot,
                               const openyourbox::state::CurrentPresetState
                                   &current) {
    presetLive = snapshot;
    presetCurrent = current;
    return true;
  });
  presetHistory.pushStep("Load preset B", makeSnapshot(1), makeSnapshot(2),
                         makeCurrent("A", true), makeCurrent("B", false));
  passed &= expect(presetHistory.undo() && presetCurrent.name == "A" &&
                       presetCurrent.dirty &&
                       static_cast<int>(presetLive.graphDocument["marker"]) == 1,
                   "undo of preset load must restore prior patch and current/dirty");
  passed &= expect(presetHistory.redo() && presetCurrent.name == "B" &&
                       !presetCurrent.dirty &&
                       static_cast<int>(presetLive.graphDocument["marker"]) == 2,
                   "redo of preset load must restore the loaded association");

  openyourbox::state::EditHistory suppressed(50);
  suppressed.setApplyFn([](const openyourbox::state::PatchSnapshot &,
                           const openyourbox::state::CurrentPresetState &) {
    return true;
  });
  suppressed.setSuppressed(true);
  suppressed.pushStep("nope", makeSnapshot(0), makeSnapshot(1), {}, {});
  passed &= expect(!suppressed.canUndo(),
                   "suppressed apply must not push history steps");

  return passed ? 0 : 1;
}
