#pragma once

#include "../library/CopyrightAcknowledgment.h"

#include <imgui.h>

#include <functional>

namespace openyourbox::ui {
/**
 * @class CopyrightModal
 * @brief Blocking first-Train certification dialog with a local log gate.
 */
class CopyrightModal {
public:
  /**
   * @brief Draws the modal when acknowledgment is still required.
   * @param acknowledgment Persistent local log helper.
   * @param visible Whether the editor is requesting the modal.
   * @param onConfirmed Invoked after a successful local persist.
   * @return True while the modal should remain open.
   */
  bool render(library::CopyrightAcknowledgment &acknowledgment, bool visible,
              const std::function<void()> &onConfirmed);

private:
  /** @brief Checkbox state for the certification text. */
  bool certified = false;
};
} // namespace openyourbox::ui
