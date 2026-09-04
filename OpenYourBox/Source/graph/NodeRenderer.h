#pragma once

#include "NodeGraph.h"
#include "../library/TrainingLibrary.h"
#include "../library/UserBoxLibrary.h"
#include "../ui/UserBoxLibraryPanel.h"

#include <imgui.h>
#include <imgui_node_editor.h>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace openyourbox::graph {
/** @brief Message-thread actions emitted by interactive graph controls. */
struct NodeRendererCallbacks {
  /** @brief Invoked after a node property is committed. */
  std::function<void(std::int32_t, const std::string &, int)> propertyChanged;
  /** @brief Invoked after a real property such as Gain is committed. */
  std::function<void(std::int32_t, const std::string &, float)>
      floatPropertyChanged;
  /** @brief Invoked to randomize one weighted live node. */
  std::function<void(std::int32_t, std::int32_t)> randomizeNode;
  /** @brief Invoked to freeze the current connected selection. */
  std::function<void(const std::vector<std::int32_t> &)> freezeSelection;
  /** @brief Invoked to restore one frozen Gold group to Live Blue. */
  std::function<void(std::int32_t)> unfreezeNode;
  /** @brief Invoked when an interaction needs user-facing feedback. */
  std::function<void(const std::string &)> showMessage;
  /** @brief Invoked after the persisted graph document changes.
   *  @param recompile True when the audio graph must be rebuilt.
   *  @param invalidateAnalysis True for patch-affecting edits (record undo and
   *         refresh analysis). False for view-only camera/layout persistence.
   */
  std::function<void(bool recompile, bool invalidateAnalysis)> documentChanged;
  /** @brief Invoked when the user opens analysis on a Blue or Gold node. */
  std::function<void(std::int32_t)> analysisRequested;
  /** @brief Invoked when Knob Input conditioning changes. */
  std::function<void(std::int32_t, float)> knobChanged;
  /** @brief Invoked when XY Trackpad conditioning changes. */
  std::function<void(std::int32_t, float, float)> xyChanged;
  /** @brief Invoked when a trainable node is armed or disarmed for training. */
  std::function<void(std::int32_t, bool)> armChanged;
  /** @brief Invoked when the user browses a weight file for a node. */
  std::function<void(std::int32_t)> browseWeights;
  /**
   * @brief Invoked after a TorchScript Load node is cleared.
   * @param artifactPath Path that is no longer referenced by that node.
   */
  std::function<void(const std::string &)> externalCheckpointCleared;
  /** @brief Invoked after a box is placed from the user box library.
   * @param rootId New node or group identifier.
   */
  std::function<void(std::int32_t)> boxPlaced;
  /**
   * @brief Invoked when a continuous patch gesture starts (knob, XY, layout drag).
   * @param label User-facing history step name.
   */
  std::function<void(const char *)> beginPatchGesture;
  /** @brief Invoked when a continuous patch gesture ends. */
  std::function<void()> endPatchGesture;
};

/**
 * @class NodeRenderer
 * @brief Dear ImGui node editor implementing ML Forge-style interactions.
 */
class NodeRenderer {
public:
  /** @brief Creates an in-memory node-editor context. */
  NodeRenderer();

  /** @brief Releases the node-editor context. */
  ~NodeRenderer();

  /**
   * @brief Draws and edits the graph in the current Dear ImGui window.
   * @param graph Message-thread graph document.
   * @param callbacks Runtime actions owned by the plug-in editor.
   * @param pinchMagnification Relative trackpad pinch scale for this frame.
   * @param boxLibrary User box catalog, or null when the library is unavailable.
   * @param outputRmsByNodeId Latest live output RMS keyed by node id, or null.
   */
  void render(NodeGraph &graph, const NodeRendererCallbacks &callbacks,
              float pinchMagnification = 1.0f,
              openyourbox::library::UserBoxLibrary *boxLibrary = nullptr,
              const std::unordered_map<std::int32_t, float> *outputRmsByNodeId =
                  nullptr);

  /** @brief Returns the first selected node identifier, or zero. */
  [[nodiscard]] std::int32_t getPrimarySelectedNodeId() const noexcept;
  /** @brief Returns the first selected node or group identifier, or zero. */
  [[nodiscard]] std::int32_t getPrimarySelectedBoxId() const noexcept;
  /** @brief Session Parameters-tab binding (live, library inspect, or multi). */
  [[nodiscard]] const SelectionContext &getSelectionContext() const noexcept;
  /**
   * @brief Consumes a one-shot request to activate the Parameters tab.
   * @return True when the editor should switch to Parameters this frame.
   */
  bool consumeForceParametersTab() noexcept;
  /**
   * @brief Draws the right-menu Parameters tab for the current selection.
   * @param graph Message-thread graph document.
   * @param callbacks Runtime actions owned by the plug-in editor.
   * @param boxLibrary User box catalog used for library inspect, or null.
   */
  void renderParametersPanel(
      NodeGraph &graph, const NodeRendererCallbacks &callbacks,
      openyourbox::library::UserBoxLibrary *boxLibrary = nullptr);
  /**
   * @brief Binds Parameters to a read-only user-library snapshot.
   *
   * Clears the canvas selection so a previously selected live box cannot
   * overwrite this inspect binding on the same frame.
   * @param entryId Catalog UUID.
   * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
   */
  void inspectLibraryEntry(const juce::String &entryId,
                           std::int32_t nestedRootId);

  /** @brief True while the Train tab is the active right-menu page. */
  bool trainTabActive = false;
  /** @brief Nodes on the active Data Loader training path. */
  std::unordered_set<std::int32_t> trainOnPathNodeIds;
  /** @brief Armed trainable nodes on that path. */
  std::unordered_set<std::int32_t> trainArmedOnPathNodeIds;
  /** @brief Training Library used by Data Loader binding pickers, or null. */
  openyourbox::library::TrainingLibrary *trainingLibrary = nullptr;

private:
  /**
   * @brief Draws the unified Library palette with Factory and User Library roots.
   * @param graph Message-thread graph document.
   * @param boxLibrary User box catalog, or null when the library is unavailable.
   */
  void renderPalette(NodeGraph &graph,
                     openyourbox::library::UserBoxLibrary *boxLibrary);
  /**
   * @brief Draws the live graph hierarchy under Library for navigate and save.
   *
   * A Main row represents the root canvas; nested groups and elements sit
   * under it. Rows are ordered by longest-path rank along sibling links, then
   * case-insensitive display name, then id. Feedback loops share a rank.
   * Unconnected boxes sit with sources. Group Input/Output hubs are omitted.
   * @param graph Message-thread graph document.
   */
  void renderProjectStructure(NodeGraph &graph);
  /**
   * @brief Draws one Project structure row for a node or nested group.
   * @param graph Message-thread graph document.
   * @param boxId Node or group identifier.
   */
  void renderProjectStructureItem(NodeGraph &graph, std::int32_t boxId);
  /**
   * @brief Draws one node and its inline controls.
   * @param graph Message-thread graph document.
   * @param node Node being rendered.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderNode(NodeGraph &graph, GraphNode &node,
                  const NodeRendererCallbacks &callbacks);
  /**
   * @brief Draws live or read-only property editors for one element.
   * @param graph Message-thread graph document.
   * @param node Node whose properties are shown.
   * @param callbacks Runtime actions owned by the plug-in editor.
   * @param readOnly True to disable commits (library inspect).
   */
  void renderNodeParameterEditors(NodeGraph &graph, GraphNode &node,
                                  const NodeRendererCallbacks &callbacks,
                                  bool readOnly);
  /**
   * @brief Draws live or read-only group name, repeats, and randomize controls.
   * @param graph Message-thread graph document.
   * @param group Group whose properties are shown.
   * @param callbacks Runtime actions owned by the plug-in editor.
   * @param readOnly True to disable commits (library inspect).
   */
  void renderGroupParameterEditors(NodeGraph &graph, GraphGroup &group,
                                   const NodeRendererCallbacks &callbacks,
                                   bool readOnly);
  /**
   * @brief Draws read-only Parameters bound to a library snapshot entry.
   * @param boxLibrary Catalog that owns the inspected snapshot.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderLibraryInspectParameters(
      openyourbox::library::UserBoxLibrary &boxLibrary,
      const NodeRendererCallbacks &callbacks);
  /**
   * @brief Inserts a catalog entry onto the focused canvas at the view centre.
   * @param graph Message-thread graph document.
   * @param entryId Catalog UUID.
   * @param nestedRootId Snapshot node/group id, or 0 for the saved root.
   */
  void placeLibraryEntryOnFocusedCanvas(NodeGraph &graph,
                                        const juce::String &entryId,
                                        std::int32_t nestedRootId);
  /**
   * @brief Selects a live box for Parameters without opening nested canvases.
   * @param graph Message-thread graph document.
   * @param boxId Node or group identifier.
   * @param forceTab True to switch the right menu to Parameters.
   */
  void selectLiveBox(NodeGraph &graph, std::int32_t boxId, bool forceTab);
  /**
   * @brief Focuses the canvas that contains @p boxId and centres on it.
   *
   * For a group, @p openInnerGroup enters that group's inner canvas and frames
   * all of its boxes. Otherwise the parent canvas is focused and the outer box
   * is centred.
   * @param graph Message-thread graph document.
   * @param boxId Node or group identifier.
   * @param openInnerGroup True to enter a group's inner canvas.
   */
  void navigateToBox(NodeGraph &graph, std::int32_t boxId, bool openInnerGroup);
  /**
   * @brief Applies a pending editor selection queued from the structure tree.
   * @param graph Message-thread graph document.
   */
  void applyPendingEditorSelection(NodeGraph &graph);
  /**
   * @brief Publishes document/layout mutations and patch-gesture edges.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void flushDocumentCallbacks(const NodeRendererCallbacks &callbacks);
  /**
   * @brief Accepts live-row, palette, and library drops on a structure target.
   * @param graph Message-thread graph document.
   * @param callbacks Runtime actions owned by the plug-in editor.
   * @param targetParent Destination group, or empty for the project root.
   * @param highlightMin Screen-space top-left of the highlight rectangle.
   * @param highlightMax Screen-space bottom-right of the highlight rectangle.
   */
  void handleStructureDropTarget(
      NodeGraph &graph, const NodeRendererCallbacks &callbacks,
      std::optional<std::int32_t> targetParent, ImVec2 highlightMin,
      ImVec2 highlightMax);
  /**
   * @brief Starts a Project structure drag for a live row.
   * @param graph Message-thread graph document.
   * @param boxId Node or group identifier.
   */
  void beginStructureDrag(NodeGraph &graph, std::int32_t boxId);
  /**
   * @brief Handles Project structure click, double-click, and group open.
   *
   * Single-click selects Parameters without changing the focused canvas.
   * Double-click on a group opens that group's inner canvas. Double-click on a
   * non-group centres the camera without opening groups. Uses a longer click
   * window than Dear ImGui's default so tree drag-source tracking cannot swallow
   * the second click.
   * @param graph Message-thread graph document.
   * @param boxId Node or group identifier.
   */
  void handleStructureRowClick(NodeGraph &graph, std::int32_t boxId);
  /**
   * @brief Selects the inserted or moved box, opens Parameters, and focuses.
   * @param graph Message-thread graph document.
   * @param boxId New or moved node or group identifier.
   * @param destinationParent Destination group, or empty for the root canvas.
   */
  void focusAfterStructureMutation(NodeGraph &graph, std::int32_t boxId,
                                   std::optional<std::int32_t> destinationParent);
  /**
   * @brief Draws a compact group box with mediating I/O pins.
   * @param graph Message-thread graph document.
   * @param group Group being rendered.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderGroup(NodeGraph &graph, GraphGroup &group,
                   const NodeRendererCallbacks &callbacks);
  /**
   * @brief Draws Main > group breadcrumb navigation above the canvas.
   *
   * The trail stays on one line. When it is wider than the canvas it
   * scrolls horizontally, and cropped edges fade out.
   * @param graph Message-thread graph document.
   */
  void renderScopeBreadcrumb(NodeGraph &graph);
  /**
   * @brief Opens a group as the focused canvas, or returns to the graph root.
   * @param graph Message-thread graph document.
   * @param groupId Group to open, or empty for the root canvas.
   */
  void setCanvasFocus(NodeGraph &graph, std::optional<std::int32_t> groupId);
  /**
   * @brief Opens a group's inner canvas and frames every box on it.
   * @param graph Message-thread graph document.
   * @param groupId Group whose interior becomes the focused canvas.
   */
  void openGroupCanvasFitted(NodeGraph &graph, std::int32_t groupId);
  /**
   * @brief Zooms and pans so every box on the focused canvas is visible and centred.
   * @param graph Message-thread graph document.
   * @param canvasSize Screen-space size of the graph editor window.
   */
  void fitCanvasToContents(NodeGraph &graph, ImVec2 canvasSize);
  /**
   * @brief Parents a newly created box to the focused group when legal.
   * @param graph Message-thread graph document.
   * @param boxId New node or group identifier.
   * @param canvasPosition Position in the current canvas coordinates.
   * @param dropGroupId Group box under the pointer, or zero.
   */
  void adoptNewBox(NodeGraph &graph, std::int32_t boxId, ImVec2 canvasPosition,
                   std::int32_t dropGroupId);
  /**
   * @brief Draws Knob Input sliders and numeric readouts for every knob.
   * @param graph Message-thread graph document.
   * @param node Knob Input node.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderKnobControl(NodeGraph &graph, GraphNode &node,
                         const NodeRendererCallbacks &callbacks);
  /**
   * @brief Draws XY Trackpad pad control and X/Y readouts.
   * @param graph Message-thread graph document.
   * @param node XY Trackpad node.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderXyPad(NodeGraph &graph, GraphNode &node,
                   const NodeRendererCallbacks &callbacks);
  /** @brief Validates interactive cable creation. */
  void handleConnections(NodeGraph &graph,
                         const NodeRendererCallbacks &callbacks);
  /** @brief Applies accepted node and link deletion requests. */
  void handleDeletion(NodeGraph &graph);
  /** @brief Draws freeze, unfreeze, insert, add, and delete context actions. */
  void handleContextMenus(NodeGraph &graph,
                          const NodeRendererCallbacks &callbacks);
  /**
   * @brief Draws Factory categories plus an expandable User Library submenu.
   * @param fromPin True for Pin Add; false for Link Insert (applies insert filters).
   * @param targetId Pin or link identifier.
   * @param position Canvas spawn position.
   */
  void renderContextInsertCatalog(bool fromPin, std::int32_t targetId,
                                  juce::Point<float> position);
  /**
   * @brief Renders an active property combobox in screen coordinates.
   * @param graph Message-thread graph document.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void handlePropertyCombo(NodeGraph &graph,
                           const NodeRendererCallbacks &callbacks);
  /**
   * @brief Applies pin/link context-menu actions deferred until after editing.
   * @param graph Message-thread graph document.
   */
  void applyPendingContextActions(NodeGraph &graph);
  /**
   * @brief Draws the canvas overview map in the bottom-right of the editor.
   * @param graph Message-thread graph document.
   * @param canvasOrigin Screen-space origin of the editor canvas.
   * @param canvasSize Screen-space size of the editor canvas.
   */
  void renderMap(NodeGraph &graph, ImVec2 canvasOrigin, ImVec2 canvasSize);
  /** @brief Mirrors node-editor selection into stable graph identifiers. */
  void synchronizeSelection(NodeGraph &graph);
  /**
   * @brief Pans or zooms the editor canvas without moving nodes.
   * @param panDelta Screen-space pan in pixels at the current zoom.
   * @param zoomFactor Multiplicative zoom around the pointer, or 1 to pan only.
   * @param pivotScreen Pointer position in screen space.
   */
  void navigateCanvas(ImVec2 panDelta, float zoomFactor, ImVec2 pivotScreen);
  /**
   * @brief Applies stored graph layout to imgui-node-editor node transforms.
   * @param graph Message-thread graph document.
   */
  void syncEditorTransforms(NodeGraph &graph);
  /**
   * @brief Centres the canvas view on a canvas-space point without changing zoom.
   * @param canvasPoint Target point in editor canvas coordinates.
   */
  void centreViewOnCanvas(ImVec2 canvasPoint);

  /** @brief Owned imgui-node-editor context. */
  ax::NodeEditor::EditorContext *context = nullptr;
  /** @brief Node targeted by the current context menu. */
  std::int32_t contextNodeId = 0;
  /** @brief Link targeted by the current context menu. */
  std::int32_t contextLinkId = 0;
  /** @brief Pin targeted by the current Add context menu. */
  std::int32_t contextPinId = 0;
  /** @brief Stable identifiers selected in the editor. */
  std::vector<std::int32_t> selectedNodeIds;
  /** @brief Selected group identifiers. */
  std::vector<std::int32_t> selectedGroupIds;
  /** @brief Stable selected connection identifiers. */
  std::vector<std::int32_t> selectedLinkIds;
  /**
   * @brief Smoothed RMS fill fraction per visible link, in `[0, 1]`.
   *
   * Fill grows from the source pin toward the destination so the cable
   * reads in the direction of information flow.
   */
  std::unordered_map<std::int32_t, float> displayedLinkFill;
  /** @brief Nodes whose persisted positions were applied to the editor. */
  std::unordered_set<std::int32_t> positionedNodeIds;
  /** @brief Groups whose persisted bounds were applied to the editor. */
  std::unordered_set<std::int32_t> positionedGroupIds;
  /** @brief Group highlighted as a drop target while dragging an element. */
  std::int32_t dropTargetGroupId = 0;
  /** @brief Node currently being dragged, or zero. */
  std::int32_t draggingNodeId = 0;
  /** @brief Group currently being dragged, or zero. */
  std::int32_t draggingGroupId = 0;
  /**
   * @brief Box under an in-progress canvas press (before drag threshold).
   *
   * While this is set, stored graph transforms are not pushed back onto this
   * box (or the rest of @ref layoutGestureBoxIds) so a multi-selection move
   * is not snapped back around the grabbed box.
   */
  std::int32_t canvasPressBoxId = 0;
  /**
   * @brief Boxes that move together for the current canvas press/drag.
   *
   * imgui-node-editor translates every selected node; these ids stay locked
   * to the editor until mouse-up so graph positions cannot overwrite them.
   */
  std::unordered_set<std::int32_t> layoutGestureBoxIds;
  /** @brief Last canvas-clicked box used for double-click detection. */
  std::int32_t canvasLastClickBoxId = 0;
  /** @brief Dear ImGui time of @ref canvasLastClickBoxId. */
  double canvasLastClickTime = -1.0;
  /** @brief Screen-space position of @ref canvasLastClickBoxId. */
  ImVec2 canvasLastClickScreenPos{};
  /** @brief Last Project structure row used for double-click detection. */
  std::int32_t structureLastClickBoxId = 0;
  /** @brief Dear ImGui time of @ref structureLastClickBoxId. */
  double structureLastClickTime = -1.0;
  /** @brief Last Factory palette label used for double-click insert. */
  std::string paletteLastClickLabel;
  /** @brief Dear ImGui time of @ref paletteLastClickLabel. */
  double paletteLastClickTime = -1.0;
  /** @brief Editable group name buffers. */
  std::unordered_map<std::int32_t, std::array<char, 48>> groupNameBuffers;
  /** @brief Group targeted by the current context menu. */
  std::int32_t contextGroupId = 0;
  /** @brief Editable signed seed text retained independently for each node. */
  std::unordered_map<std::int32_t, std::array<char, 16>> seedBuffers;
  /**
   * @brief Editable comma-separated per-repeat property text, keyed by
   *        `nodeId:propertyKey`.
   */
  std::unordered_map<std::string, std::array<char, 512>> propertyListBuffers;
  /** @brief Current transient connection validation message. */
  std::string transientMessage;
  /** @brief Dear ImGui time when transient feedback expires. */
  double transientMessageDeadline = 0.0;
  /** @brief Whether graph topology or parameters changed during this frame. */
  bool mutatedThisFrame = false;
  /** @brief Whether viewport or node layout changed during this frame. */
  bool layoutMutatedThisFrame = false;
  /** @brief Whether this frame's mutation requires an audio recompile. */
  bool recompileThisFrame = false;
  /** @brief True while a coalesced patch gesture is open. */
  bool patchGestureOpen = false;
  /** @brief True when a knob, pad, property drag, or layout drag is held. */
  bool patchGestureHeldThisFrame = false;
  /** @brief History label for the open patch gesture. */
  const char *patchGestureLabel = "Parameter edit";
  /** @brief Whether persisted canvas navigation still needs restoring. */
  bool restoreViewPending = true;
  /** @brief True to keep the focused hierarchy trail entry in view. */
  bool scopeBreadcrumbFollowFocus = true;
  /** @brief Last known canvas-space centre used when spawning from the palette. */
  ImVec2 lastCanvasCentre{250.0f, 140.0f};
  /** @brief True when the overview map captured the pointer on the previous frame. */
  bool mapHoveredLastFrame = false;
  /** @brief User Library tree drawn in the left palette. */
  openyourbox::ui::UserBoxLibraryPanel boxLibraryPanel;
  /** @brief Session width of the left Library palette in pixels. */
  float leftPaletteWidth = 200.0f;
  /** @brief Node or group targeted by the save-to-library dialog. */
  std::int32_t pendingSaveBoxId = 0;
  /** @brief True to open the save-to-library popup this frame. */
  bool requestSaveBoxPopup = false;
  /** @brief Editable name for the save-to-library dialog. */
  std::array<char, 96> saveBoxNameBuffer{};
  /** @brief True when the save dialog is replacing an existing name. */
  bool saveBoxOverwrite = false;
  /** @brief Box library bound for the current render frame, or null. */
  openyourbox::library::UserBoxLibrary *activeBoxLibrary = nullptr;
  /** @brief Editor callbacks bound for the current render frame, or null. */
  const NodeRendererCallbacks *activeCallbacks = nullptr;
  /** @brief Active choice-property combobox rendered outside canvas coordinates. */
  struct ActivePropertyCombo {
    /** @brief Node that owns the edited property. */
    std::int32_t nodeId = 0;
    /** @brief Property key opened by the user. */
    std::string propertyKey;
    /** @brief Screen-space top-left of the property control. */
    ImVec2 anchorMin{};
    /** @brief Whether the combobox should stay active. */
    bool active = false;
    /** @brief Whether the dropdown should open on the next deferred pass. */
    bool requestOpen = false;
  } activePropertyCombo;
  /** @brief Pin attach action queued from the context menu. */
  struct PendingPinAttach {
    /** @brief Target pin identifier. */
    std::int32_t pinId = 0;
    /** @brief Element type to create. */
    NodeType type = NodeType::linear;
    /** @brief Spawn position for the new element. */
    juce::Point<float> position{};
    /** @brief Whether an attach action is queued. */
    bool pending = false;
    /** @brief True when the action inserts a user-library box. */
    bool fromLibrary = false;
    /** @brief Catalog UUID when @ref fromLibrary is true. */
    juce::String libraryEntryId;
    /** @brief Snapshot nested root id; 0 inserts the saved root. */
    std::int32_t nestedRootId = 0;
  } pendingPinAttach;
  /** @brief Link insert action queued from the context menu. */
  struct PendingLinkInsert {
    /** @brief Target link identifier. */
    std::int32_t linkId = 0;
    /** @brief Element type to create. */
    NodeType type = NodeType::linear;
    /** @brief Spawn position for the new element. */
    juce::Point<float> position{};
    /** @brief Whether an insert action is queued. */
    bool pending = false;
    /** @brief True when the action inserts a user-library box. */
    bool fromLibrary = false;
    /** @brief Catalog UUID when @ref fromLibrary is true. */
    juce::String libraryEntryId;
    /** @brief Snapshot nested root id; 0 inserts the saved root. */
    std::int32_t nestedRootId = 0;
  } pendingLinkInsert;
  /** @brief Session Parameters-tab binding. */
  SelectionContext selectionContext;
  /** @brief Signature of the last selection that forced the Parameters tab. */
  std::int32_t lastSelectionSignature = 0;
  /**
   * @brief Editor selection to apply inside the canvas; 0 none, -1 clear.
   */
  std::int32_t pendingEditorSelectId = 0;
  /** @brief True when the canvas should centre on @ref pendingCentrePoint. */
  bool pendingCentreView = false;
  /**
   * @brief True when the next editor frame should frame all boxes on the canvas.
   */
  bool pendingFitCanvas = false;
  /** @brief Canvas-space point to centre after a structure navigation. */
  ImVec2 pendingCentrePoint{};
  /** @brief Live box currently dragged from Project structure, or zero. */
  std::int32_t structureDragSourceId = 0;
  /** @brief Highlighted structure drop parent, empty for root, when valid. */
  std::optional<std::int32_t> structureDropTargetParent;
  /** @brief True when the current structure drop target would accept. */
  bool structureDropValid = false;
  /** @brief True when a structure drop highlight rectangle should be drawn. */
  bool structureDropHighlightActive = false;
  /** @brief Screen-space top-left of the structure drop highlight. */
  ImVec2 structureDropHighlightMin{};
  /** @brief Screen-space bottom-right of the structure drop highlight. */
  ImVec2 structureDropHighlightMax{};
  /** @brief Cached snapshot for the active library inspect binding. */
  juce::ValueTree libraryInspectSnapshot;
  /**
   * @brief True when Project structure/library set the live binding this frame.
   */
  bool preserveLiveSelection = false;
  /**
   * @brief True when User Library inspect must win over leftover canvas selection.
   */
  bool preserveLibraryInspect = false;
};
} // namespace openyourbox::graph
