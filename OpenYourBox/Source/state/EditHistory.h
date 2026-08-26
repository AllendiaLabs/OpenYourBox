#pragma once

#include "PatchSnapshot.h"

#include <JuceHeader.h>

#include <functional>
#include <utility>

namespace openyourbox::state {
/**
 * @struct CurrentPresetState
 * @brief Session association of the live instance to a named catalog entry.
 */
struct CurrentPresetState {
  /** @brief Catalog UUID after load/Save/Save As, or empty. */
  juce::String entryId;
  /** @brief Display name shown in chrome, or empty when unassociated. */
  juce::String name;
  /** @brief True after an undoable patch edit while associated. */
  bool dirty = false;
  /** @brief Sonic fingerprint of the last loaded or saved snapshot. */
  juce::String baselineFingerprint;

  /**
   * @brief Returns true when a catalog name is associated with this instance.
   */
  [[nodiscard]] bool isAssociated() const noexcept { return name.isNotEmpty(); }
};

/**
 * @class EditHistory
 * @brief Session-scoped undo/redo façade over JUCE UndoManager.
 *
 * Stores before/after @ref PatchSnapshot plus @ref CurrentPresetState. Apply
 * is delegated to the owner so snapshots are restored on the GUI thread.
 */
class EditHistory {
public:
  /**
   * @brief Applies a snapshot and current-preset association.
   * @param snapshot Patch to restore.
   * @param currentPreset Association to restore with the patch.
   * @return True when apply succeeded.
   */
  using ApplyFn = std::function<bool(const PatchSnapshot &snapshot,
                                     const CurrentPresetState &currentPreset)>;

  /**
   * @brief Constructs history with a default depth of 50 steps.
   * @param maxDepth Maximum stored steps (FR-016).
   */
  explicit EditHistory(int maxDepth = 50);

  /**
   * @brief Sets the owner callback used for undo and redo apply.
   * @param apply Owner apply function.
   */
  void setApplyFn(ApplyFn apply);

  /**
   * @brief Pushes one discrete history step.
   *
   * No-ops when suppressed, when a gesture is open, or when fingerprints match.
   *
   * @param label User-facing step name.
   * @param before State prior to the change.
   * @param after State after the change.
   * @param beforeCurrent Association prior to the change.
   * @param afterCurrent Association after the change.
   */
  void pushStep(const juce::String &label, PatchSnapshot before,
                PatchSnapshot after, CurrentPresetState beforeCurrent,
                CurrentPresetState afterCurrent);

  /**
   * @brief Begins a coalesced gesture; captures @p before as the step start.
   * @param label User-facing step name.
   * @param before State prior to the gesture.
   * @param beforeCurrent Association prior to the gesture.
   */
  void beginGesture(const juce::String &label, PatchSnapshot before,
                    CurrentPresetState beforeCurrent);

  /**
   * @brief Ends a gesture and pushes one step when the patch changed.
   * @param after State after the gesture.
   * @param afterCurrent Association after the gesture.
   */
  void endGesture(PatchSnapshot after, CurrentPresetState afterCurrent);

  /** @brief Cancels an open gesture without pushing a step. */
  void cancelGesture();

  /**
   * @brief Restores the previous step.
   * @return True when a step was undone.
   */
  bool undo();

  /**
   * @brief Restores the next previously undone step.
   * @return True when a step was redone.
   */
  bool redo();

  /** @brief Returns true when undo is available. */
  [[nodiscard]] bool canUndo() const noexcept;

  /** @brief Returns true when redo is available. */
  [[nodiscard]] bool canRedo() const noexcept;

  /** @brief Returns the configured maximum depth. */
  [[nodiscard]] int getMaxDepth() const noexcept;

  /** @brief Returns the number of undoable steps currently stored. */
  [[nodiscard]] int getUndoDepth() const noexcept;

  /** @brief Returns true while snapshot apply should not push new steps. */
  [[nodiscard]] bool isSuppressed() const noexcept;

  /**
   * @brief Sets whether new pushes are ignored (apply / host restore).
   * @param suppressed True to ignore push/gesture commits.
   */
  void setSuppressed(bool suppressed) noexcept;

  /** @brief Returns true while a coalesce gesture is open. */
  [[nodiscard]] bool isGestureOpen() const noexcept;

private:
  /**
   * @class SnapshotAction
   * @brief Undoable replace of the live patch and current-preset association.
   */
  class SnapshotAction;

  /** @brief Owner apply callback used by undo/redo. */
  ApplyFn applyFn;
  /** @brief JUCE undo stack; unit size 1 per step, capped at @ref maxDepth. */
  juce::UndoManager undoManager;
  /** @brief Maximum stored steps. */
  int maxDepth = 50;
  /** @brief True while applying a snapshot so recursive pushes are ignored. */
  bool suppressed = false;
  /** @brief True while a coalesce gesture is open. */
  bool gestureOpen = false;
  /** @brief Count of undoable steps (redo excluded); capped at @ref maxDepth. */
  int storedUndoSteps = 0;
  /** @brief Label for the open gesture. */
  juce::String gestureLabel;
  /** @brief Snapshot captured at @ref beginGesture. */
  PatchSnapshot gestureBefore;
  /** @brief Current-preset captured at @ref beginGesture. */
  CurrentPresetState gestureBeforeCurrent;
};
} // namespace openyourbox::state
