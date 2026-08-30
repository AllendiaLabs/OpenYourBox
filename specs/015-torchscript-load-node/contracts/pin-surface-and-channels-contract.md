# Contract: Pin Surface & Channel Overrides

## Purpose

Define how ports and channel counts update after load and how shape integrity uses overrides.

## Pin morph

| Event | Pins |
|-------|------|
| Place / clear / empty | Audio in, audio out only |
| Load with encode/decode | Add latent out + latent in (labels match trained RAVE Gold / `isLatentPin`) |
| Load without encode/decode | No latent pins |
| Load with conditioning | Add Control input (`control` / existing control pin label) |
| Load without conditioning | No Control pin |

Removing a capability on reload must drop optional pins and invalidate illegal cables with Shape Integrity feedback.

## Channel fields (property panel)

| Field | Source |
|-------|--------|
| Inferred in / out / latent | Set on successful load (and encode probe for latent) |
| Override in / out / latent | Editable; empty/sentinel = use inferred |
| Reset overrides | Restore effective = last inferred |

**Effective channels** = override if set, else inferred.

## Shape integrity

- Connection legality uses effective channel counts for the relevant pin.
- Changing an override revalidates cables; illegal results refused or flagged (same class as other graph mismatches).
- If inference failed and required overrides absent → incomplete-shape state; do not treat connections as legal (FR-011).

## Probe policy (implementation guidance)

1. Prefer host channel count or current override as input hint.
2. On failure, retry a small set of common widths before prompting for manual overrides.
3. Output channels from forward probe `size(1)`.
4. Latent channels from encode output when encode/decode present.

## Acceptance

- FR-005, FR-011, FR-012, FR-013, FR-014, FR-015; SC-004.
