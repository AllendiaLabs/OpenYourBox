#import <AppKit/AppKit.h>

#include "HostTransport.h"

namespace openyourbox::capture {
void requestHostTransportStart() {
  @autoreleasepool {
    NSEvent *event =
        [NSEvent keyEventWithType:NSEventTypeKeyDown
                         location:NSZeroPoint
                    modifierFlags:0
                        timestamp:NSProcessInfo.processInfo.systemUptime
                     windowNumber:0
                          context:nil
                       characters:@" "
      charactersIgnoringModifiers:@" "
                        isARepeat:NO
                          keyCode:49];
    if (event == nil)
      return;
    if (![NSApp.mainMenu performKeyEquivalent:event])
      [NSApp postEvent:event atStart:NO];
  }
}
} // namespace openyourbox::capture
