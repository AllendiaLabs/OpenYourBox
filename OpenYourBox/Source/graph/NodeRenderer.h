#pragma once

#include "NodeGraph.h"

#include <imgui.h>
#include <imgui_node_editor.h>

#include <array>
#include <functional>
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
   *  @param invalidateAnalysis True when open analysis snapshots must refresh.
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
   */
  void render(NodeGraph &graph, const NodeRendererCallbacks &callbacks,
              float pinchMagnification = 1.0f);

  /** @brief Returns the first selected node identifier, or zero. */
  [[nodiscard]] std::int32_t getPrimarySelectedNodeId() const noexcept;

private:
  /** @brief Draws the ML Forge-style element palette. */
  void renderPalette(NodeGraph &graph);
  /** @brief Draws one node and its inline controls. */
  /**
   * @brief Draws one node and its inline controls.
   * @param graph Message-thread graph document.
   * @param node Node being rendered.
   * @param callbacks Runtime actions owned by the plug-in editor.
   */
  void renderNode(NodeGraph &graph, GraphNode &node,
                  const NodeRendererCallbacks &callbacks);
  /**
   * @brief Draws Knob Input rotary control and numeric readout.
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
   * @brief Draws the overview map as an ImGui overlay inside the editor.
   * @param graph Message-thread graph document.
   * @param canvasOrigin Screen-space origin of the editor canvas.
   * @param canvasSize Screen-space size of the editor canvas.
   */
  void renderMap(NodeGraph &graph, ImVec2 canvasOrigin, ImVec2 canvasSize);
  /** @brief Mirrors node-editor selection into stable graph identifiers. */
  void synchronizeSelection();
  /**
   * @brief Pans or zooms the editor canvas without moving nodes.
   * @param panDelta Screen-space pan in pixels at the current zoom.
   * @param zoomFactor Multiplicative zoom around the pointer, or 1 to pan only.
   * @param pivotScreen Pointer position in screen space.
   */
  void navigateCanvas(ImVec2 panDelta, float zoomFactor, ImVec2 pivotScreen);
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
  /** @brief Stable selected connection identifiers. */
  std::vector<std::int32_t> selectedLinkIds;
  /** @brief Nodes whose persisted positions were applied to the editor. */
  std::unordered_set<std::int32_t> positionedNodeIds;
  /** @brief Editable signed seed text retained independently for each node. */
  std::unordered_map<std::int32_t, std::array<char, 16>> seedBuffers;
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
  /** @brief Whether persisted canvas navigation still needs restoring. */
  bool restoreViewPending = true;
  /** @brief Last known canvas-space centre used when spawning from the palette. */
  ImVec2 lastCanvasCentre{250.0f, 140.0f};
  /** @brief True when the overview map captured the pointer on the previous frame. */
  bool mapHoveredLastFrame = false;
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
  } pendingLinkInsert;
};
} // namespace openyourbox::graph
