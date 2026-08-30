# Contract: Right-Click Insert Catalog

**Feature**: `016-rave-prior-mix`  
**Applies to**: `NodeRenderer::handleContextMenus` Pin **Add** and Link **Insert** menus; shared Factory catalog module

## Scope

| Surface | In scope |
|---------|----------|
| Pin context → Add | Yes |
| Link context → Insert | Yes |
| Left Library Factory / User Library panels | Out of scope for behavior change (may share Factory data source) |
| Empty-canvas / node context without Add today | Out of scope |

## Factory section

1. MUST list every item from the **shared Factory palette catalog** (same set as left Factory), subject to existing filters (`canInsertOnLink`, attach eligibility).
2. MUST NOT use a divergent hard-coded subset that omits types present in the left Factory.
3. Adding a new `PaletteItem` to the shared catalog MUST appear in both left Factory and these menus without a second manual menu list.

## User Library section

1. MUST appear as an expandable hierarchy: folders → entries → nested snapshot members (006/011 semantics).
2. Choosing a root entry inserts via `UserBoxLibrary::insertBox(entryId)` (independent instance; groups collapsed per existing rules).
3. Choosing a nested member inserts via `insertBox(entryId, nestedRootId)`.
4. Empty library: section visible but vacant; Factory section still complete.
5. Failed insert (missing type): clear message; menu remains usable.

## Placement behavior

| Source | Pin Add / Link Insert |
|--------|------------------------|
| Factory item | Existing `attachNodeToPin` / `insertNodeOnLink` |
| Library box | Place using library insert APIs at context position; best-effort wire when a single clear connection exists; otherwise place nearby without failing the insert |

## Non-goals

- Marketplace browser
- Redesigning left Library panel UX
- Auto-including non-palette `NodeType` values
