# Feature Specification: Modern Editor UI

**Feature Branch**: `020-modern-editor-ui`

**Created**: 2026-09-05

**Status**: Draft

**Input**: User description: "make oyb UI look modern and beautiful, find examples online to depart from current appearance which looks too much like default imgui. make it more like modern software like vs code. build standalone and take screenshots whenever necessary, especially during implementation"

## Clarifications

### Session 2026-09-05

- Q: Which typefaces should OpenYourBox use for editor chrome (titles, tabs, buttons, trees) versus numbers, status, and pin text? → A: One bundled contemporary UI sans for chrome and numbers (regular + semibold only)
- Q: How should the workbench’s cool focus accent relate to Live box blue so selection and Live/Frozen stay obvious at a glance? → A: Change everything to a modern colour family (chrome, accent, Live, Frozen, and canvas family roles all remap together; Live vs Frozen stay instantly distinct roles, current toolkit blue/gold RGB is not locked)
- Q: Should OpenYourBox keep today’s left library / centre graph / right inspector arrangement, or add VS Code–like chrome such as an icon rail or a bottom status bar? → A: **Superseded.** VS Code is inspiration only. Do what is best for OpenYourBox as a graph instrument plugin (see the following clarification). Do not clone a code-editor workbench.
- Q: Which icons should the new workbench rail and actions use? → A: Generic open UI icons (not Codicons), one style across chrome actions. An icon rail is not required.
- Q: Should Dry/Wet, Knob Input, and the XY trackpad look like VS Code settings widgets, or stay tactile instrument controls inside the new workbench? → A: Custom instrument look for every control, including trees and tabs
- Q: Dark mode or a light/day page? → A: Dark mode only — never a day, light, cream, or pale-plaza page
- Q: How strictly should OpenYourBox follow VS Code? → A: VS Code is inspiration only. Do what is best for OpenYourBox. Dark modern materials, professional type, and accent-on-focus may be borrowed; layout, widgets, and identity MUST serve the graph instrument, not a code editor.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First glance reads as OpenYourBox, not a toolkit or a code editor (Priority: P1)

A producer opens OpenYourBox. Within a second the window reads as a **dark modern instrument editor**: the graph is the hero, library and parameters are at hand, type is professional, controls look crafted. It may remind someone of contemporary dark software (VS Code is one inspiration among others). It is never a light or day page, never the stock toolkit demo, and never “VS Code with a graph.”

**Why this priority**: The current editor uses the toolkit’s built-in dark theme (saturated blue fills, pixel-like type, heavy window frames). That first impression is the problem this feature exists to fix. Copying a code editor would swap one wrong identity for another.

**Independent Test**: Open the editor as its own window (no music host), capture the idle graph view at a typical plugin size, and place it beside a stored “before” still and beside a public VS Code still. Reviewers must pick OpenYourBox as a modern instrument editor — not as “default toolkit” and not as a VS Code clone.

**Acceptance Scenarios**:

1. **Given** the editor at a typical **plugin** window size, **When** a reviewer sees the idle view, **Then** the graph is the dominant working surface; library and inspector are reachable without crowding the canvas into a leftover pane; the page is dark (not light/day/cream)
2. **Given** that same view, **When** the reviewer inspects type, icons, and chrome, **Then** labels, tabs, buttons, trees, numbers, status, and pin text all use the same bundled contemporary UI sans in regular or semibold only; action icons use one generic open UI icon style (not VS Code Codicons, not emoji); tabs, trees, buttons, fields, and sliders look like custom instrument parts, not unmodified stock toolkit widgets and not a code-editor settings page
3. **Given** hover, selection, and primary actions, **When** the reviewer watches interaction, **Then** a single accent from the modern colour family appears on focus/selection/primary actions only — not as a default fill on every widget, and not as a substitute for Live/Frozen node colour
4. **Given** a stored before-still of the current editor, **When** it is placed beside the restyled idle view, **Then** the restyled view is immediately distinguishable by palette, type, density, and control chrome — not only by tinting the same widgets

---

### User Story 2 - Public visual references stay in the loop (Priority: P1)

An implementer does not invent a look from memory. They collect a small **public reference board** of modern dark software and of toolkit-based apps that already left the default toolkit appearance, then restyle OpenYourBox against those stills **in service of this product**. VS Code is on the board as inspiration for dark layered surfaces and professional chrome, not as a layout to reproduce. During every visual slice they rebuild the windowed host, capture OpenYourBox stills, and compare them to the board.

**Why this priority**: “Make it beautiful” fails when the only reference is the current editor. Public examples are how the team leaves the default toolkit look — they are not a license to clone another product.

**Independent Test**: Confirm a dated reference board exists (URLs plus saved stills) covering VS Code’s current dark workbench as **one** inspiration plus at least two toolkit-based apps with custom chrome; confirm each implementation slice that changes appearance has a matching OpenYourBox still taken from the windowed host.

**Acceptance Scenarios**:

1. **Given** implementation of this feature has started, **When** a reviewer inspects the private visual-reference folder, **Then** they find a board of **public** examples that includes Visual Studio Code’s current dark workbench as inspiration, not as a screenshot to match pixel-for-pixel
2. **Given** that board, **When** the reviewer looks for toolkit-based apps that no longer look like the default toolkit, **Then** they find at least two such examples (expected starting set: ImHex’s custom window chrome; Tracy Profiler’s typography). DearSQL is optional. None of these is the product to copy
3. **Given** a visual change (palette, type, tabs, trees, buttons, graph chrome, or modals), **When** that change is considered done, **Then** the windowed host has been built and launched, and a new still of the affected surface has been saved
4. **Given** an OpenYourBox still and a VS Code still on the board, **When** a reviewer compares them, **Then** they may name shared **materials** (dark layered surfaces, quiet separators, accent-on-focus, professional type) but MUST NOT conclude that OpenYourBox copied VS Code’s activity bar, file tabs, terminal panel, or status-bar layout

---

### User Story 3 - Existing workflows remain discoverable (Priority: P1)

A sound designer still places boxes, wires Audio In → process → Audio Out, edits Parameters, freezes, captures, and trains. Those capabilities all remain. Surfaces MAY move if the new placement is better for a graph instrument in a plugin window; they MUST NOT disappear. Selecting a box still reveals Parameters. Live boxes stay obviously Live; Frozen boxes stay obviously Frozen (with lock). Hues sit in the same modern colour family as the chrome.

**Why this priority**: A prettier editor that hides Train, Parameters, or freeze would fail the product. Relocating them is allowed when it helps OpenYourBox; copying another app’s chrome to host them is not the goal.

**Independent Test**: Walk the first-session path (place a box, wire Audio In → process → Audio Out, open Parameters, move Dry/Wet) and a freeze/unfreeze path; confirm every former inspector surface is reachable without a tutorial.

**Acceptance Scenarios**:

1. **Given** the editor, **When** the user looks for Library, Project structure, Info, Parameters, Capture, Train, and Presets, **Then** each former surface is reachable in at most a small number of obvious clicks (they MAY leave today’s left/right tab locations if the new home is clearer for a graph instrument)
2. **Given** a Live learned box and a Frozen box, **When** the user looks at the canvas without reading labels, **Then** they can tell Live from Frozen in under one second (two distinct modern-family hues + lock on Frozen), and neither hue is the same as the selection accent
3. **Given** an illegal connection attempt, **When** the cable is refused, **Then** the failure colour remains unambiguously warning-red
4. **Given** Capture, Train, copyright, and error notices, **When** those surfaces appear, **Then** they use the new materials but keep the same blocking and acknowledgement behaviour
5. **Given** a box on the canvas, **When** the user selects or clicks it, **Then** Parameters for that box become visible without hunting through a hidden menu

---

### User Story 4 - Every control is an instrument part (Priority: P2)

A performer trims Dry/Wet, browses the library tree, and edits parameters. Trees, tabs, buttons, fields, sliders, knobs, and the XY surface all share one **custom instrument** language — tactile shapes, inset wells, crafted indicators. Nothing looks like a stock toolkit widget or a code-editor settings clone. Graph boxes stay slim (name + pins) and pick up the same materials.

**Why this priority**: Recolouring stock trees and sliders still reads as a restyled demo. Instrument chrome on every control is what makes the product feel designed.

**Independent Test**: Capture the idle editor, Library tree, inspector tabs, Parameters (including Dry/Wet), a Knob Input, an XY Trackpad, and a populated canvas; confirm every control class uses the instrument language.

**Acceptance Scenarios**:

1. **Given** the Library, **When** Factory and User Library are expanded, **Then** the tree uses the same instrument chrome as the rest of the editor (crafted rows, hover, selection) and does not use the toolkit’s default bright-blue header fill or a plain file-explorer clone as the visible default
2. **Given** tabs and section switches, **When** the user changes view, **Then** those controls are instrument-crafted (not raised toolkit tabs); the active item is indicated with a quiet accent; action icons are the same generic open style throughout
3. **Given** Dry/Wet, Knob Input, XY Trackpad, parameter fields, checkboxes, and buttons, **When** the user views them, **Then** they share the instrument language (knobs and XY remain tactile instrument shapes; fields and buttons are crafted, not stock bars)
4. **Given** the graph canvas, **When** boxes and cables are shown, **Then** box chrome uses the same instrument materials; Live, Frozen, and family roles use distinct hues from the same modern colour family as the chrome
5. **Given** a small plugin window, **When** side panels are narrow, **Then** they may collapse or stack so the graph stays usable; labels that stay visible remain legible

---

### User Story 5 - Windowed host screenshots at every visual step (Priority: P2)

An implementer does not wait until the end of the feature to look at the UI. Whenever appearance changes — and especially while implementing — they build the existing windowed host (the plugin running as its own application, without a DAW), open it, and save dated stills of the editor. Those stills are the evidence that the OpenYourBox language still holds.

**Why this priority**: Beauty regressions hide in DAW-hosted plugin windows. A launchable windowed host plus stills is the only reliable visual QA loop.

**Independent Test**: For each implementation slice that changes pixels, show a new still captured from the windowed host of the affected surface; reject slices that only claim a restyle in text.

**Acceptance Scenarios**:

1. **Given** a visual implementation slice, **When** it is marked done, **Then** the windowed host was built and launched and at least one new still of the changed surface exists
2. **Given** a still used for QA, **When** a reviewer opens it, **Then** it shows the editor window only (no desktop, menu bar of other apps, or unrelated windows)
3. **Given** the idle editor at plugin size, a populated mixed Live/Frozen canvas, Library, Parameters, Capture, Train, Presets, analysis, and at least one modal, **When** the feature is ready for review, **Then** a still exists for each of those surfaces
4. **Given** the private stills folder, **When** the repository is inspected, **Then** those stills are not required to build the product and are not committed

---

### Edge Cases

- High-DPI / retina: strokes and 1-pixel separators stay crisp; the intended look is not blurry toolkit artefacts.
- Colour-vision deficiency: Live vs Frozen vs accent vs illegal-cable red must still pass a “tell apart at arm’s length” check after the modern-family remap.
- Host scaling inside a DAW: the restyle MUST look like the same OpenYourBox editor in the plugin window as in the windowed host; density may scale but materials must not revert to the stock toolkit theme.
- Very small editor size: side panels MAY collapse or stack; the graph MUST remain the usable centre; chrome must not clip into unusable slivers. Do not keep an icon rail “because code editors have one” if it steals canvas at plugin size.
- Selection, hover, and drag on boxes, pins, and splitters remain as reliable as today (visual restyle must not shrink hit targets below current usability).
- Modal notices restyle to instrument cards but keep blocking/acknowledgement behaviour.
- Training-path highlights (when Train is open) stay visible on the dark editor.
- Visual-QA stills that include the desktop or other apps are invalid and MUST be recaptured as window-only.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The editor MUST present as a **dark modern OpenYourBox instrument editor**: graph-dominant, layered dark opaque surfaces, professional type, crafted controls, a single accent reserved for focus/selection/primary actions. The page MUST be dark mode. It MUST NOT be a light, day, cream, or pale-plaza page, MUST NOT present as the stock toolkit dark theme, and MUST NOT present as a code-editor workbench with a graph dropped in
- **FR-002**: Visual Studio Code’s current dark workbench MAY inspire **materials** (dark layered surfaces, quiet separators, accent-on-focus, professional type). It MUST NOT dictate layout, widget shapes, iconography, or information architecture. OpenYourBox MUST NOT clone VS Code’s product logo, Codicons, activity bar, file tabs, terminal panel, command palette, or status-bar layout
- **FR-003**: Before restyling surfaces, the team MUST collect a **public reference board** of online examples. The board MUST include VS Code’s current dark workbench as **one inspiration** plus at least two toolkit-based applications that already left the default toolkit look. Starting set: [VS Code theme colours](https://code.visualstudio.com/api/references/theme-color) (materials only), [ImHex](https://github.com/WerWolv/ImHex), [Tracy Profiler UI notes](https://bolu.dev/programming/2024/12/29/imgui-starter-tracy.html). Optional: [DearSQL](https://github.com/dunkbing/dearsql). Planning MUST prefer choices that serve a graph instrument in a plugin window over choices that only increase likeness to a reference
- **FR-004**: Implementation MUST consult that board when choosing materials. A restyle that only recolours stock widgets without changing type, density, separators, and control chrome DOES NOT meet this feature. A restyle that copies another product’s layout also DOES NOT meet this feature
- **FR-005**: Layout MUST be whatever is best for OpenYourBox as a **graph instrument plugin**, judged at typical DAW plugin sizes as well as a large windowed host. Non-negotiable layout outcomes:
  - The **graph canvas** is the centre working surface and MUST keep the majority of the window at typical plugin size
  - **Library** (Factory + User) and **Project structure** stay immediately reachable
  - **Parameters** (and Info) appear when a box is selected, without replacing the graph
  - **Capture, Train, and Presets** stay discoverable
  - **Analysis** stays associated with the selected box / Info, not relocated into a fake “terminal” because another app has a bottom panel
  - Freeze/train/audio/muted status stays visible
  Slim boxes (name + pins) remain. Icon rail, extra sidebars, bottom panels, and status bars are **allowed only if** they improve those outcomes in a plugin window; they are **not required**
- **FR-006**: Graph editing behaviour (select, drag, wire, groups, freeze/unfreeze, train, capture, presets, undo/redo) MUST remain available and discoverable. This feature MAY relocate those surfaces when the new home is clearer for OpenYourBox; it MUST NOT remove them
- **FR-007**: Live versus Frozen MUST remain instantly distinguishable **roles** (Live = editable/glitchable, Frozen = compiled/stable with the existing lock affordance). Their hues MUST come from the same modern colour family as the chrome and MAY leave the current toolkit blue/gold. Live, Frozen, and the accent MUST be pairwise distinct at arm’s length; the accent MUST NOT be used as the Live or Frozen fill
- **FR-008**: Existing family roles on the canvas (audio I/O, conditioning, helpers, learned Live, Frozen, training-only) MUST remain uniquely coloured. Those hues MUST be remapped into the same modern colour family as the chrome (not left as leftover toolkit primaries beside a restyled editor)
- **FR-009**: **Every** visible control — tabs, trees, buttons, fields, checkboxes, sliders, menus, modals, Dry/Wet, Knob Input, and XY Trackpad — MUST use one **custom instrument** chrome (tactile shapes, inset wells, crafted hover/selection). Unmodified stock toolkit widgets MUST NOT remain as the visible default. Code-editor settings widgets MUST NOT be the visible default
- **FR-010**: Typography MUST use **one bundled contemporary UI sans** for all editor chrome (titles, tabs, buttons, trees) **and** for numbers, status, and pin text, in a **two-weight ramp only** (regular + semibold). The toolkit’s default typeface MUST NOT appear in editor chrome. Distinctive display faces (including Bounded and Computer Says No) MUST NOT be used. Planning MAY pick any license-respecting family that meets this ramp; it MUST be bundled with the plugin so every host looks the same
- **FR-016**: Action and chrome icons MUST use **one generic open UI icon set** (license-respecting, bundled), the same style everywhere. VS Code Codicons, emoji-as-icons, and original hand-drawn pictograms are out. Planning MAY pick the specific set. Icons do not imply that an activity-bar rail must exist
- **FR-011**: UI MUST remain responsive at the constitution target of 60 frames per second while audio is running
- **FR-012**: During implementation (and whenever appearance changes at any speckit stage), visual claims MUST be verified by **building and launching the existing windowed host** (plugin as its own application, without a DAW) and saving **window-only** stills under `.ignore/visual-refs/` with dated or abstract filenames. Full-desktop captures are invalid. Stills MUST include a typical plugin-sized window, not only a maximized code-editor-like frame
- **FR-013**: Private stills and the reference board MAY live under `.ignore/` for implementers; they MUST NOT be required to build or run the shipped plugin and MUST NOT be committed
- **FR-014**: Illegal-connection warning MUST stay unambiguously red (safety outranks accent)
- **FR-015**: The same OpenYourBox language MUST apply inside a DAW-hosted plugin window as in the windowed host; the windowed host is for visual QA, not a second product look

### Key Entities

- **OpenYourBox language**: Dark modern instrument-editor materials (layered dark surfaces, quiet separators, one UI sans, accent-on-focus, crafted controls) judged for a graph plugin — not a code-editor skin
- **Inspiration board**: Public examples (including VS Code dark workbench) used for materials and craft, never as a layout or identity to copy
- **Instrument chrome**: Custom control craft applied to trees, tabs, buttons, fields, sliders, knobs, and XY — not stock toolkit widgets and not code-editor settings clones
- **Icon set**: One generic open UI icon family used on actions and chrome (not Codicons, not emoji, not original pictograms)
- **Canvas role colour**: Semantic box colour role (audio I/O, conditioning, helper, learned Live, Frozen, training-only) remapped into the modern colour family so the canvas and chrome are one palette
- **Windowed host still**: A gitignored, window-only capture of the editor launched as its own application, used for visual QA

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a side-by-side of the stored before-still and a post-restyle idle editor still, 100% of internal reviewers asked “which looks like a generic developer toolkit?” pick the before-still
- **SC-002**: In a side-by-side of the restyled idle still and a public VS Code still, at least 4 of 5 reviewers agree OpenYourBox is **not** a VS Code clone. Shared dark-modern materials are acceptable; matching activity bar / file tabs / terminal / status-bar layout is a failure
- **SC-003**: A reviewer can identify Live vs Frozen on a mixed canvas in under 1 second without reading labels, in at least 10 of 10 sampled layouts
- **SC-004**: A designer who knows today’s product can complete the first-session path (place a box, wire Audio In → process → Audio Out, open Parameters, move Dry/Wet) without a tutorial. Selecting a box MUST still reveal Parameters in one click. Extra clicks to reach a former tab are allowed only when the new home is obviously better for a graph instrument
- **SC-005**: Illegal-connection refusal remains visible: 100% of testers notice the refused cable without a hint
- **SC-006**: Editor stays subjectively smooth while playing audio; no reviewer reports stuttering chrome during graph pan/zoom at typical plugin window size
- **SC-007**: Every implementation slice that changes appearance leaves at least one new **window-only** still in `.ignore/visual-refs/` captured from the windowed host, including at least one still at typical plugin size
- **SC-008**: A reviewer shown only the restyled idle still does not describe it as “the default developer-toolkit theme” or as “VS Code with a graph,” in at least 5 of 5 sampled reviewers
- **SC-009**: A reviewer shown Library tree, tabs, Dry/Wet, a knob, and XY together agrees they share one instrument control language (not a mix of stock toolkit bars and flat settings widgets), in at least 4 of 5 sampled reviewers
- **SC-010**: At a typical plugin window size, reviewers agree the graph remains the dominant working surface (not a thin centre strip between chrome copied from a desktop IDE), in at least 4 of 5 sampled reviewers

## Assumptions

- OpenYourBox remains a plugin product. The plugin is the customer interface. The existing windowed / Standalone *format* is only a development and visual-QA host (already in the project), not a new shipped application. This respects the constitution’s “plugin is the interface” rule while honouring the request to build that host and take screenshots.
- VS Code (and other public apps) inspire **materials**. Layout defaults that are best for OpenYourBox: graph centre and dominant; library/project immediately reachable (left remains a strong default); Parameters/Info on select without covering the graph (right inspector remains a strong default); Capture/Train/Presets as inspector or compact tool surfaces, not a code-editor activity bar; analysis with the selected box; status visible without cloning a desktop-IDE footer. Planning may diverge from those defaults only to improve graph-instrument use in a plugin window.
- A previous “pale plaza / scientific colour map” direction was explored and reverted. This feature does **not** revive that pale-plaza language. Instrument chrome means crafted dark-mode controls in the modern colour family, including trees and tabs.
- Live = editable/glitchable and Frozen = compiled/stable (with lock) remain constitution-level **roles**. Their hues, the accent, and all canvas family colours remap together into one modern colour family. Current toolkit blue/gold RGB is not locked. Planning SHOULD keep Live cooler and Frozen warmer so the pair still reads instantly.
- Dark mode only. A light, day, cream, or pale-plaza editor is out of scope. Layered dark surfaces may differ slightly in value (sidebar vs canvas) while remaining dark.
- Private files under `.ignore/` stay gitignored and are never required to build or run the plugin.
- Typography is one bundled contemporary UI sans for both chrome and numbers, regular + semibold only. Foundry is chosen at planning if the license can be bundled. System UI fonts, a separate monospace for telemetry, and the Bounded / Computer Says No pair are out.
- Icons are one generic open UI set, bundled, same style on actions. The specific family is chosen at planning. Codicons, emoji-as-icons, and original pictograms are out. Presence of icons does not require an activity bar.

## Out of Scope

- Changing freeze, train, capture, library, or graph **semantics** (behaviour stays; surfaces may move if better for OpenYourBox)
- Adding a command palette, editor file tabs, a minimap, or other code-editor features
- Cloning VS Code’s (or any other product’s) layout, logos, Codicons, or settings widgets
- Shipping a customer-facing application besides the plugin
- Shipping private visual-reference stills
- A light, day, cream, or pale-plaza theme
- Dropping Live vs Frozen as instantly distinct roles, or dropping the Frozen lock affordance
- Leaving stock toolkit trees/tabs/sliders as the visible default
- Reviving the reverted pale-plaza / scientific-map visual language
- Distinctive display faces (Bounded, Computer Says No), a second monospace for telemetry, or OS system UI fonts as the editor face
