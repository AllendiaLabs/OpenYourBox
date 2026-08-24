#pragma once

namespace openyourbox::capture {
/**
 * @brief Asks the hosting DAW to start playback.
 *
 * Ableton and other macOS hosts ignore `AudioPlayHead::transportPlay`. This
 * posts the Space key equivalent through the host application menu, which Live
 * binds to Play. No-op on platforms without a native implementation.
 */
#if defined(__APPLE__)
void requestHostTransportStart();
#else
inline void requestHostTransportStart() {}
#endif
} // namespace openyourbox::capture
