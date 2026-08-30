# Feature Specification: RAVE Prior Mix & Insert Catalog

**Feature Branch**: `016-rave-prior-mix`

**Created**: 2026-08-30

**Status**: Clarified (ready for planning)

**Input**: User description: "oyb, externally loaded rave ts or learned oyb rave should have a control to mix between forward and prior modes, in a continuous fashion, but when the control is full prior, the encoder processing should be skipped. latent input pin should be replaced by bias pin and scale pin, like in rave vst. bias should be added to the mean and scale should scale the std or variance (whatever is used). When disconnected, use bias=0 and scale=1. latent output pin should expose the sampled latent values effectively used. also I think the add/insert list is not up to date. It should always show factory elements, and expandable hierarchy of user library"

## Clarifications

### Session 2026-08-30

- Q: How should an intermediate prior-mix value combine the forward (encoder) path and the prior path before decode? → A: Interpolate encoder mean/spread toward prior (0 / 1), then apply bias and scale; bias/scale apply in both forward and prior (and all intermediate mixes)
- Q: Should the prior-mix control be automatable from the DAW host, or only adjustable inside the element’s properties? → A: Same pattern as existing element parameters such as activation gain (and fidelity on RAVE boxes): a continuous box/element property, editable in the element UI and persisted with the box—not a one-off host-parameter or pin-only exception
- Q: Which add/insert surfaces must show the full factory set plus the expandable user-library hierarchy? → A: Right-click / context add-insert menus: they must list all current factory elements (they currently lag) and include an expandable user-library hierarchy; left Library panel is out of scope for this fix
- Q: When an older project still has a cable into the removed latent-input pin on a RAVE-capable box, what should happen on load? → A: Out of scope — older projects with the previous latent-input pin will not be loaded; no migration, remap, or load-time notice is required for that pin change

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Continuous Forward ↔ Prior Mix (Priority: P1)

A performer or sound designer loads a pretrained RAVE TorchScript model or plays a learned OpenYourBox RAVE Gold box. They find a continuous **prior mix** control (0 = full forward / encode-from-audio, 1 = full prior). Sweeping the control morphs smoothly between reconstructing the input and generating from the model’s prior. At full prior, the encoder path is not run, so generation does not waste work encoding audio that is ignored.

**Why this priority**: Prior/posterior morphing is the core creative control of the acids-ircam RAVE VST workflow; without it, loaded and learned RAVE boxes only reconstruct and cannot jam from the prior.

**Independent Test**: Place a RAVE-capable external load or learned Gold RAVE between Audio In and Audio Out; sweep prior mix from 0 to 1 while audio plays; confirm continuous timbre change and that full prior still produces audio without requiring a wired latent drive.

**Acceptance Scenarios**:

1. **Given** a successfully loaded external RAVE-style checkpoint (encode/decode available) or a learned OYB RAVE Gold box, **When** the user views its properties, **Then** a continuous prior-mix property is available spanning full forward to full prior, presented like other continuous element parameters (e.g. activation gain / fidelity)
2. **Given** prior mix at full forward, **When** audio plays through a legal chain, **Then** output follows encode→sample→decode driven by the input audio (subject to fidelity and bias/scale as defined elsewhere)
3. **Given** prior mix at an intermediate value, **When** audio plays, **Then** the box interpolates encoder mean/spread toward prior (0 / 1), applies bias and scale, samples once, and decodes—so output morphs continuously with no discrete jump except at intentional endpoints
4. **Given** prior mix at full prior, **When** audio plays, **Then** the base mean/spread are the prior (0 / 1), bias and scale still apply, the decoder runs from that sampled latent, and encoder processing for that box is skipped
5. **Given** prior mix at full prior, **When** the audio input to the box is silence or disconnected relative to encode, **Then** generation from the prior (plus bias/scale) still proceeds (encoder skip does not silence the box)

---

### User Story 2 - Bias and Scale Pins Replace Latent Input (Priority: P1)

The user wants RAVE-VST-style latent steering: instead of feeding a full external latent tensor into a single latent-in pin, they connect (or leave disconnected) **bias** and **scale** pins. Bias shifts the latent mean; scale multiplies the latent spread used for sampling. Disconnected pins behave as bias = 0 and scale = 1 so the box works out of the box.

**Why this priority**: Replaces the previous latent-input surface with the standard RAVE performance controls; without defaults for disconnected pins, graphs break when users do not wire conditioning.

**Independent Test**: On a RAVE-capable box, confirm latent-in is gone and bias/scale pins exist; leave them disconnected and hear normal reconstruction/prior; connect constant or modulated sources and hear mean shift and spread change.

**Acceptance Scenarios**:

1. **Given** a RAVE-capable external load or learned OYB RAVE Gold box, **When** the user inspects latent-related input pins, **Then** there is no single “latent input” pin for driving decode; instead there are distinct **bias** and **scale** input pins (plus whatever audio/control pins the box already requires)
2. **Given** bias and scale pins are disconnected, **When** audio plays, **Then** the box behaves as bias = 0 and scale = 1 for every latent channel
3. **Given** a connected bias signal at any prior-mix setting (full forward, intermediate, or full prior), **When** audio plays, **Then** bias is added to the post-mix latent mean before sampling (channel-wise when shapes match)
4. **Given** a connected scale signal at any prior-mix setting, **When** audio plays, **Then** scale multiplies the post-mix latent spread quantity that participates in sampling (the same spread path the model already uses for reparameterization), channel-wise when shapes match
5. **Given** illegal bias or scale shapes (channel or time mismatch vs latent width), **When** the user attempts the connection, **Then** the cable is refused with the same class of shape-mismatch feedback used elsewhere

---

### User Story 3 - Latent Output Exposes Sampled Values in Use (Priority: P2)

The user taps the box’s **latent output** pin to visualize or process the latent that actually drives the decoder—including after prior mix, bias, scale, and sampling—not a pre-mix or pre-bias intermediate.

**Why this priority**: Creative “latent jamming” and analysis depend on seeing the effective z; secondary to hearing correct audio from mix/bias/scale.

**Independent Test**: Connect latent out to an analysis view or downstream latent-legal element; change prior mix, bias, and scale; confirm the tapped stream tracks the values used for decode.

**Acceptance Scenarios**:

1. **Given** a RAVE-capable box with a latent output pin, **When** audio plays, **Then** latent out carries the sampled latent values effectively used by the decoder for that buffer
2. **Given** the user changes prior mix, bias, or scale, **When** they observe latent out, **Then** the streamed values reflect those controls (not a stale pre-control encode-only stream)
3. **Given** full prior (encoder skipped), **When** latent out is connected, **Then** it still emits the prior-path sampled latents actually decoded

---

### User Story 4 - Right-Click Add/Insert Shows Factory and User Library (Priority: P2)

When the user **right-clicks** to add/insert an element onto the graph, the menu always shows the complete current set of **factory** elements (including newly shipped types that today are missing from that menu), plus an **expandable hierarchy** of their **user library** so they can insert whole saved boxes or nested subparts without leaving the context menu.

**Why this priority**: Stale right-click insert lists block discovery of new factory types; adding the user library there makes save/reuse as reachable as stock elements. Independent of RAVE prior mix but required for daily authoring.

**Independent Test**: Right-click add/insert with an empty and a populated user library; confirm every current factory type appears; expand user-library folders and insert a nested subpart; ship a new factory type and confirm it appears in the right-click menu without a manual catalog update.

**Acceptance Scenarios**:

1. **Given** the graph editor is open, **When** the user opens a right-click add/insert menu, **Then** all current factory element types are listed and insertable
2. **Given** the user has saved boxes in the user library (including nested groups), **When** they open a right-click add/insert menu, **Then** the user library appears as an expandable hierarchy alongside factory entries
3. **Given** an expandable user-library entry with nested members in that menu, **When** the user expands it, **Then** they can insert the whole saved root or a nested subpart as a new independent instance
4. **Given** the plugin’s factory set has grown (new element types shipped), **When** the user opens a right-click add/insert menu, **Then** those new types appear without requiring the list to be manually curated to an outdated snapshot
5. **Given** the user library is empty, **When** the user opens a right-click add/insert menu, **Then** factory elements still appear in full and the user-library section is empty or clearly vacant—not missing or replacing the factory list

---

### Edge Cases

- Prior mix exactly at full prior: encoder skipped; base mean/spread are 0 / 1; bias/scale still apply before sampling
- Prior mix exactly at full forward: encoder runs; base mean/spread are the encoder’s; bias/scale still apply before sampling
- Intermediate mix with encoder running: mean/spread are interpolated toward 0 / 1, then bias/scale, then one sample; latent out matches what decode uses
- Bias or scale connected with constant 0 / 1 explicitly: same audible result as disconnected defaults
- Scale of 0: sampling collapses toward the (biased) mean; must not crash or produce NaNs in the user-facing path
- Fidelity/compactness active: prior mix, bias, scale, and latent out operate on the effective latent space after compactness rules already defined for RAVE
- External TorchScript load without encode/decode: prior-mix / bias / scale / latent-out RAVE surface does not appear; forward-only behavior from the load-node feature remains
- Learned OYB RAVE after Unfreeze to Blue modular graph: this feature’s unified prior-mix / bias / scale surface applies to the **Gold / loaded RAVE box** forms named in the input; modular Blue reconstruction of the same controls is out of scope unless the live bottleneck already exposes an equivalent surface
- Add/insert while library entry references a removed factory type: insert fails with a clear message; factory list and other library entries remain usable
- Very large user library in the right-click menu: hierarchy remains browsable (expand/collapse); insert still creates an independent instance
- Left Library / side-panel catalog: not changed by this feature’s catalog requirements (scope is right-click add/insert only)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every RAVE-capable box in scope (externally loaded TorchScript with encode/decode, and learned OpenYourBox RAVE Gold) MUST expose a continuous prior-mix **element property** from full forward to full prior, following the same control pattern as existing continuous element parameters such as activation gain (editable on the box, persisted with the element/session/preset)—not a unique host-only or pin-only surface
- **FR-002**: At full prior, the box MUST skip encoder processing for that element while still producing decoded audio from the prior path
- **FR-003**: At intermediate prior-mix values, the box MUST interpolate the encoder mean and spread toward the prior base (mean 0, spread 1) continuously, then apply bias and scale, then sample once—so small control moves produce small audible/latent changes
- **FR-003a**: Processing order MUST be: (1) obtain base mean/spread from the prior-mix rule (encoder at full forward; 0/1 at full prior; interpolated in between), (2) add bias to mean and multiply spread by scale, (3) sample, (4) decode. Bias and scale MUST apply at every prior-mix setting, including full forward and full prior
- **FR-004**: RAVE-capable boxes in scope MUST replace the former latent-input pin with separate **bias** and **scale** input pins (RAVE VST–style), not a single external-z drive pin
- **FR-005**: When bias is disconnected, the box MUST treat bias as 0; when scale is disconnected, the box MUST treat scale as 1
- **FR-006**: Connected bias MUST be added to the post-mix latent mean used for sampling; connected scale MUST multiply the post-mix latent spread quantity used for sampling (the same spread quantity already used by the model’s sampling path)
- **FR-007**: Bias and scale pin shapes MUST be shape-checked against the effective latent width (and time alignment rules consistent with other latent-domain cables); illegal connections MUST be refused with clear feedback
- **FR-008**: The latent output pin MUST expose the sampled latent values effectively used by the decoder after prior mix, bias, scale, and sampling
- **FR-009**: Prior mix, bias, scale, and latent out MUST remain coherent with the existing fidelity/compactness behavior when that control is present
- **FR-010**: Session/preset restore MUST restore prior-mix value and bias/scale connections; disconnected pins restore as defaults 0 and 1
- **FR-011**: Compatibility with older projects that used a latent-input pin on RAVE-capable boxes is **out of scope** (those projects will not be loaded); this feature MUST NOT spend scope on latent-in migration, remapping, or load-time notices for that pin change
- **FR-012**: Every right-click / context **add/insert** menu MUST list the complete current set of factory element types (no stale subset that omits newly shipped factory elements)
- **FR-013**: Those same right-click add/insert menus MUST include the user library as an expandable hierarchy (folders / nested saved groups) from which the user can insert a whole entry or a nested subpart
- **FR-014**: Right-click add/insert menus MUST stay consistent with the live factory registry and user-library contents (not a hard-coded outdated snapshot)
- **FR-015**: This feature’s catalog requirements apply to right-click add/insert menus only; the left Library / side-panel placement UI is out of scope unless it already shares the same menu data source as a consequence of implementation

### Key Entities

- **RAVE-capable box**: A Gold learned OYB RAVE or an externally loaded TorchScript element that exposes encode/decode; owns prior-mix, bias, scale, audio I/O, and latent out
- **Prior mix**: Continuous element property from full forward (encode-driven) to full prior (encoder skipped); same property pattern as activation gain / fidelity
- **Bias pin**: Latent-domain (or compatible) input added to the sampling mean; default 0 when disconnected
- **Scale pin**: Latent-domain (or compatible) input multiplying sampling spread; default 1 when disconnected
- **Effective sampled latent**: The post-mix, post-bias/scale, post-sample tensor that drives decode and is published on latent out
- **Add/insert catalog (right-click)**: Context-menu list used to place factory elements and user-library boxes onto the graph via right-click add/insert

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a guided test, users can morph a RAVE-capable box from full forward to full prior in under 30 seconds and hear a continuous change without restarting the host
- **SC-002**: At full prior, encoder work for that box is observably skipped (no encode-dependent change when only the audio input varies while bias/scale/prior are held fixed at full prior)
- **SC-003**: With bias and scale disconnected, a RAVE-capable box produces the same class of output as “neutral” steering (bias 0, scale 1) on first insert without extra wiring
- **SC-004**: Connecting a non-zero bias and a non-unity scale changes the latent out stream and the decoded audio in a way testers can confirm within one listening pass
- **SC-005**: Latent out matches the latents used for decode across full forward, intermediate mix, and full prior in at least 95% of instrumented buffer checks in QA
- **SC-006**: In usability checks, 100% of current factory element types appear in right-click add/insert, and at least 90% of testers can expand the user library in that menu and insert a nested subpart on the first try
- **SC-007**: After shipping a new factory element type, right-click add/insert shows it without a separate manual catalog refresh step by the user

## Assumptions

- “RAVE-capable” means encode/decode is available (learned OYB RAVE Gold and external TorchScript loads that expose that surface). Forward-only TorchScript loads are unchanged by this feature’s latent controls.
- Prior mix endpoints are full forward (0) and full prior (1). Intermediate values linearly interpolate encoder mean→0 and encoder spread→1, then apply bias/scale, then sample once (not a blend of two separately sampled latent vectors).
- “Spread” scaled by the scale pin is the same quantity the existing RAVE sampling path already uses when drawing latents—not a second parallel definition of variance.
- Prior mix is an ordinary continuous element property (same class as activation gain / RAVE fidelity), not a dedicated DAW host-parameter exception and not a required graph pin. Any future host exposure or macro assignment follows whatever the product already does for that class of element properties.
- Full prior with disconnected bias/scale is neutral prior sampling (base 0/1); creative prior steering is intentionally done via bias/scale on top of that base.
- Bias and scale are per effective latent channel when connected with matching width; broadcasting rules follow existing latent-domain cable conventions where those already exist.
- This feature updates the Gold / loaded RAVE box pin surface; rebuilding the same prior-mix UX on every Blue modular encoder/bottleneck/decoder piece is out of scope.
- Compatibility with older graphs that wired a latent-input pin into RAVE-capable boxes is out of scope; those projects will not be loaded. New graphs use bias/scale only.
- Add/insert catalog scope for this feature is **right-click / context add-insert menus** only (not the left Library side panel, not a marketplace). Factory list is derived from the plugin’s real factory set; user library hierarchy follows existing save/insert semantics (independent instances; group insert collapse rules from prior library specs).
- Fidelity/compactness, when present, continues to apply as already specified for RAVE; this feature does not redefine compactness math.
