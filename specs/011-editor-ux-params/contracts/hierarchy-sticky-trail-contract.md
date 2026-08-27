# Contract: Hierarchy Sticky Trail

## Purpose

Keep previously opened child canvases visible and clickable in the top hierarchy trail when navigating up, until the user branches elsewhere (FR-003).

## State

- `focusedGroupId` (existing).
- `stickySpine`: ordered list of group ids forming the last visited path **under** the current focus (descendants that were opened).

## Behavior

| Action | Result |
|--------|--------|
| Open child C from focus F | Focus C; extend spine through C |
| Navigate to ancestor A via trail | Focus A; retain spine entries that are descendants of A |
| Open sibling B (not on retained spine) | Focus B; clear prior sticky descendants; spine = path to B |
| Click sticky child | Focus that group without requiring canvas double-click |
| Delete/ungroup id on spine | Remove that id and its descendants from spine |

## UI

- Render ancestor chain as today (`Graph > …`) **plus** sticky descendants still relevant under current focus.
- Sticky entries are clickable (`SmallButton` or equivalent).

## Persistence

- Session-only in v1 unless viewport serialization already stores focus (then stickySpine MAY join the same store).

## Implementation anchors

- `NodeRenderer::renderScopeBreadcrumb`, `setCanvasFocus`
- `GraphViewport::focusedGroupId`

## Non-goals

- Full history stack / back button across unrelated branches
- Showing all siblings ever visited (only the active spine)
