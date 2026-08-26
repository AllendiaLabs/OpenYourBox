#include "EditHistory.h"

namespace openyourbox::state {
/**
 * @class EditHistory::SnapshotAction
 * @brief Stores before/after snapshots; first perform is a no-op apply.
 */
class EditHistory::SnapshotAction final : public juce::UndoableAction {
public:
  /**
   * @brief Creates one undoable patch replacement.
   * @param owner History that owns the apply callback.
   * @param stepLabel User-facing name.
   * @param beforeSnapshot State prior to the change.
   * @param afterSnapshot State after the change.
   * @param beforePreset Association prior to the change.
   * @param afterPreset Association after the change.
   */
  SnapshotAction(EditHistory &owner, juce::String stepLabel,
                 PatchSnapshot beforeSnapshot, PatchSnapshot afterSnapshot,
                 CurrentPresetState beforePreset,
                 CurrentPresetState afterPreset)
      : history(owner), label(std::move(stepLabel)),
        before(std::move(beforeSnapshot)), after(std::move(afterSnapshot)),
        beforeCurrent(std::move(beforePreset)),
        afterCurrent(std::move(afterPreset)) {}

  bool perform() override {
    if (!appliedOnce) {
      appliedOnce = true;
      return true;
    }
    if (history.applyFn == nullptr)
      return false;
    const juce::ScopedValueSetter<bool> suppress(history.suppressed, true);
    return history.applyFn(after, afterCurrent);
  }

  bool undo() override {
    if (history.applyFn == nullptr)
      return false;
    const juce::ScopedValueSetter<bool> suppress(history.suppressed, true);
    return history.applyFn(before, beforeCurrent);
  }

  int getSizeInUnits() override { return 1; }

  /** @brief History that performs apply. */
  EditHistory &history;
  /** @brief Step label retained for debugging. */
  juce::String label;
  /** @brief Patch prior to the change. */
  PatchSnapshot before;
  /** @brief Patch after the change. */
  PatchSnapshot after;
  /** @brief Association prior to the change. */
  CurrentPresetState beforeCurrent;
  /** @brief Association after the change. */
  CurrentPresetState afterCurrent;
  /** @brief False until the first UndoManager perform (already applied). */
  bool appliedOnce = false;
};

EditHistory::EditHistory(int depth) : maxDepth(std::max(1, depth)) {
  undoManager.setMaxNumberOfStoredUnits(maxDepth, maxDepth);
}

void EditHistory::setApplyFn(ApplyFn apply) { applyFn = std::move(apply); }

void EditHistory::pushStep(const juce::String &label, PatchSnapshot before,
                           PatchSnapshot after,
                           CurrentPresetState beforeCurrent,
                           CurrentPresetState afterCurrent) {
  if (suppressed || gestureOpen)
    return;
  if (before.sonicFingerprint() == after.sonicFingerprint() &&
      beforeCurrent.entryId == afterCurrent.entryId &&
      beforeCurrent.name == afterCurrent.name &&
      beforeCurrent.dirty == afterCurrent.dirty)
    return;
  undoManager.beginNewTransaction(label);
  if (undoManager.perform(new SnapshotAction(*this, label, std::move(before),
                                             std::move(after),
                                             std::move(beforeCurrent),
                                             std::move(afterCurrent))))
    storedUndoSteps = std::min(storedUndoSteps + 1, maxDepth);
}

void EditHistory::beginGesture(const juce::String &label, PatchSnapshot before,
                               CurrentPresetState beforeCurrent) {
  if (suppressed)
    return;
  if (gestureOpen)
    cancelGesture();
  gestureOpen = true;
  gestureLabel = label;
  gestureBefore = std::move(before);
  gestureBeforeCurrent = std::move(beforeCurrent);
}

void EditHistory::endGesture(PatchSnapshot after,
                             CurrentPresetState afterCurrent) {
  if (!gestureOpen) {
    return;
  }
  gestureOpen = false;
  auto before = std::move(gestureBefore);
  auto beforeCurrent = std::move(gestureBeforeCurrent);
  const auto label = gestureLabel;
  gestureLabel.clear();
  if (suppressed)
    return;
  pushStep(label, std::move(before), std::move(after), std::move(beforeCurrent),
           std::move(afterCurrent));
}

void EditHistory::cancelGesture() {
  gestureOpen = false;
  gestureLabel.clear();
  gestureBefore = {};
  gestureBeforeCurrent = {};
}

bool EditHistory::undo() {
  if (gestureOpen)
    cancelGesture();
  if (!undoManager.canUndo())
    return false;
  if (!undoManager.undo())
    return false;
  storedUndoSteps = std::max(0, storedUndoSteps - 1);
  return true;
}

bool EditHistory::redo() {
  if (gestureOpen)
    cancelGesture();
  if (!undoManager.canRedo())
    return false;
  if (!undoManager.redo())
    return false;
  storedUndoSteps = std::min(storedUndoSteps + 1, maxDepth);
  return true;
}

bool EditHistory::canUndo() const noexcept { return undoManager.canUndo(); }

bool EditHistory::canRedo() const noexcept { return undoManager.canRedo(); }

int EditHistory::getMaxDepth() const noexcept { return maxDepth; }

int EditHistory::getUndoDepth() const noexcept { return storedUndoSteps; }

bool EditHistory::isSuppressed() const noexcept { return suppressed; }

void EditHistory::setSuppressed(bool next) noexcept { suppressed = next; }

bool EditHistory::isGestureOpen() const noexcept { return gestureOpen; }
} // namespace openyourbox::state
