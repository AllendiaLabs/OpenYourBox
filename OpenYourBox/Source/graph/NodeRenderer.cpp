#include "NodeRenderer.h"
#include "RaveLayouts.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_node_editor_internal.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ed = ax::NodeEditor;

namespace {
struct PaletteItem {
  const char *label;
  openyourbox::graph::NodeType type;
};

/** @brief Factory drag payload for a published RAVE starting graph. */
struct RavePaletteItem {
  /** @brief Row label shown while dragging. */
  const char *label;
  /** @brief Published RAVE lineage. */
  openyourbox::graph::RaveLayoutId layout =
      openyourbox::graph::RaveLayoutId::original;
  /** @brief Channel width: 1 mono or 2 stereo. */
  int channels = 1;
};

constexpr const char *raveLayoutPayloadId = "OPENYOURBOX_RAVE_LAYOUT";

constexpr std::array<PaletteItem, 13> paletteItems{{
    {"Linear", openyourbox::graph::NodeType::linear},
    {"Conv1D", openyourbox::graph::NodeType::convolution},
    {"ConvTranspose1d", openyourbox::graph::NodeType::convTranspose},
    {"BatchNorm1d", openyourbox::graph::NodeType::batchNorm},
    {"Activation", openyourbox::graph::NodeType::activation},
    {"TCN", openyourbox::graph::NodeType::tcn},
    {"Merge", openyourbox::graph::NodeType::merge},
    {"Knob Input", openyourbox::graph::NodeType::knobInput},
    {"XY Trackpad", openyourbox::graph::NodeType::xyTrackpad},
    {"PQMF Analysis", openyourbox::graph::NodeType::pqmfAnalysis},
    {"PQMF Synthesis", openyourbox::graph::NodeType::pqmfSynthesis},
    {"Bottleneck", openyourbox::graph::NodeType::variationalBottleneck},
    {"Noise Synth", openyourbox::graph::NodeType::noiseSynthesizer},
}};

constexpr float nodeBodyWidth = 188.0f;
/** @brief Width reserved for right-aligned property value controls. */
constexpr float propertyValueWidth = 76.0f;
/** @brief Multiplier applied to canvas pan gestures. */
constexpr float canvasPanSpeed = 2.0f;
/** @brief Base scroll-wheel pan distance in screen pixels. */
constexpr float canvasWheelPanStep = 48.0f;
/** @brief Middle-mouse button index used for canvas drag panning. */
constexpr int canvasDragPanButton = 2;

/**
 * @brief Returns the pin drawn on the canvas for one end of a link.
 *
 * Boundary-crossing cables attach to a group box's mediating pin instead of
 * reaching through that box to a member pin.
 * @param graph Graph owning membership and links.
 * @param pinId Real member pin.
 * @param focusedGroupId Group whose interior is the current canvas, or empty.
 */
std::int32_t visualPinForLink(
    const openyourbox::graph::NodeGraph &graph, std::int32_t pinId,
    std::optional<std::int32_t> focusedGroupId) {
  const auto nodeId = graph.findNodeForPin(pinId);
  if (!nodeId.has_value())
    return pinId;
  if (graph.focusedCanvasHostGroup(*nodeId, focusedGroupId).has_value())
    return openyourbox::graph::collapsedGroupPinId(pinId);
  return pinId;
}

/**
 * @brief Returns true when a node is represented on the focused canvas.
 * @param graph Graph owning membership.
 * @param nodeId Candidate node.
 * @param focusedGroupId Focused group, or empty for the graph root.
 */
bool nodeRepresentedOnCanvas(
    const openyourbox::graph::NodeGraph &graph, std::int32_t nodeId,
    std::optional<std::int32_t> focusedGroupId) {
  return graph.isNodeOnFocusedCanvas(nodeId, focusedGroupId) ||
         graph.focusedCanvasHostGroup(nodeId, focusedGroupId).has_value();
}

/**
 * @brief Group box under a canvas point on the focused canvas, or zero.
 * @param graph Graph owning layout.
 * @param canvasPoint Point in the current canvas coordinates.
 * @param focusedGroupId Focused group, or empty for the graph root.
 */
std::int32_t groupBoxAtCanvas(
    const openyourbox::graph::NodeGraph &graph, ImVec2 canvasPoint,
    std::optional<std::int32_t> focusedGroupId) {
  std::int32_t hit = 0;
  for (const auto &group : graph.getGroups()) {
    if (!graph.isGroupOnFocusedCanvas(group.id, focusedGroupId))
      continue;
    const auto size = ImVec2(std::max(8.0f, group.size.x),
                             std::max(8.0f, group.size.y));
    if (canvasPoint.x < group.position.x || canvasPoint.y < group.position.y ||
        canvasPoint.x > group.position.x + size.x ||
        canvasPoint.y > group.position.y + size.y)
      continue;
    hit = group.id;
  }
  return hit;
}

/**
 * @brief Converts a positive graph identifier to imgui-node-editor storage.
 * @param identifier Stable signed graph identifier.
 * @return Width-safe editor identifier payload.
 */
std::uintptr_t editorIdentifier(std::int32_t identifier) noexcept {
  return static_cast<std::uintptr_t>(static_cast<std::uint32_t>(identifier));
}

ImVec4 colourFor(const juce::Colour &colour, float alpha = 1.0f) {
  return {colour.getFloatRed(), colour.getFloatGreen(), colour.getFloatBlue(),
          alpha};
}

/**
 * @brief Returns the concrete imgui-node-editor context.
 * @return Active editor, or null when no editor is bound.
 */
ax::NodeEditor::Detail::EditorContext *detailEditor() {
  return reinterpret_cast<ax::NodeEditor::Detail::EditorContext *>(
      ed::GetCurrentEditor());
}

/**
 * @brief Applies a canvas view rectangle, including zoom, without animation.
 * @param view Target visible region in editor canvas coordinates.
 */
void commitCanvasView(const ImRect &view) {
  auto *editor = detailEditor();
  if (editor == nullptr)
    return;
  editor->NavigateTo(view, false, 0.0f);
}

/**
 * @brief Draws a node-local divider that cannot span the infinite canvas.
 */
void drawNodeDivider() {
  const auto start = ImGui::GetCursorScreenPos();
  ImGui::GetWindowDrawList()->AddLine(
      start, ImVec2(start.x + nodeBodyWidth, start.y),
      ImGui::GetColorU32(ImGuiCol_Separator), 1.0f);
  ImGui::Dummy(ImVec2(nodeBodyWidth, 4.0f));
}

/**
 * @brief Draws a left-right arrow glyph used by property drag handles.
 * @param draw Destination draw list.
 * @param center Glyph centre in screen space.
 * @param color Filled glyph colour.
 */
void drawEastWestArrows(ImDrawList *draw, ImVec2 center, ImU32 color) {
  const float arm = 5.0f;
  const float head = 3.25f;
  draw->AddRectFilled(ImVec2(center.x - arm, center.y - 0.7f),
                      ImVec2(center.x + arm, center.y + 0.7f), color);
  draw->AddTriangleFilled(ImVec2(center.x - arm - 1.0f, center.y),
                          ImVec2(center.x - arm + head, center.y - head),
                          ImVec2(center.x - arm + head, center.y + head),
                          color);
  draw->AddTriangleFilled(ImVec2(center.x + arm + 1.0f, center.y),
                          ImVec2(center.x + arm - head, center.y - head),
                          ImVec2(center.x + arm - head, center.y + head),
                          color);
}

/**
 * @brief Draws a drag handle and returns integer steps from horizontal dragging.
 * @param id ImGui identifier for the handle.
 * @param size Handle size in pixels.
 * @return Signed steps to apply; zero when the handle is idle.
 */
int propertyDragSteps(const char *id, ImVec2 size) {
  ImGui::InvisibleButton(id, size);
  const auto hovered = ImGui::IsItemHovered();
  const auto active = ImGui::IsItemActive();
  if (hovered || active)
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

  const auto min = ImGui::GetItemRectMin();
  const auto max = ImGui::GetItemRectMax();
  const auto color = ImGui::GetColorU32(
      active ? ImGuiCol_SliderGrabActive
             : (hovered ? ImGuiCol_SliderGrab : ImGuiCol_Text));
  drawEastWestArrows(ImGui::GetWindowDrawList(),
                     ImVec2((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f),
                     color);

  auto *storage = ImGui::GetStateStorage();
  const auto storageId = ImGui::GetItemID();
  if (!active) {
    storage->SetFloat(storageId, 0.0f);
    return 0;
  }

  const auto pixelsPerStep = ImGui::GetIO().KeyShift ? 3.0f : 8.0f;
  auto accum = storage->GetFloat(storageId) + ImGui::GetIO().MouseDelta.x;
  const auto steps = static_cast<int>(accum / pixelsPerStep);
  accum -= static_cast<float>(steps) * pixelsPerStep;
  storage->SetFloat(storageId, accum);
  return steps;
}

/**
 * @brief Draws a left-aligned property label and moves the cursor to the value
 *        column on the right edge of the node body.
 * @param label Property label text.
 */
void beginPropertyRow(const char *label) {
  const auto rowOrigin = ImGui::GetCursorScreenPos();
  ImGui::TextUnformatted(label);
  ImGui::SetCursorScreenPos(
      ImVec2(rowOrigin.x + nodeBodyWidth - propertyValueWidth, rowOrigin.y));
  ImGui::SetNextItemWidth(propertyValueWidth);
}

/**
 * @brief Returns true when a palette type can be inserted onto an existing cable.
 * @param type Palette element type.
 * @return False for source-only Knob/XY elements.
 */
bool canInsertOnLink(openyourbox::graph::NodeType type) noexcept {
  return !openyourbox::graph::isConditioningSourceType(type);
}
} // namespace

namespace openyourbox::graph {
NodeRenderer::NodeRenderer() = default;

NodeRenderer::~NodeRenderer() { ed::DestroyEditor(context); }

void NodeRenderer::render(NodeGraph &graph,
                          const NodeRendererCallbacks &callbacks,
                          float pinchMagnification,
                          openyourbox::library::UserBoxLibrary *boxLibrary) {
  mutatedThisFrame = false;
  layoutMutatedThisFrame = false;
  recompileThisFrame = false;
  activeBoxLibrary = boxLibrary;
  if (context == nullptr) {
    ed::Config configuration;
    configuration.SettingsFile = nullptr;
    configuration.DragButtonIndex = 0;
    configuration.SelectButtonIndex = 0;
    configuration.NavigateButtonIndex = -1;
    configuration.ContextMenuButtonIndex = 1;
    configuration.EnableSmoothZoom = false;
    context = ed::CreateEditor(&configuration);
    restoreViewPending = true;
  }

  if (graph.getViewport().focusedGroupId.has_value() &&
      graph.findGroup(*graph.getViewport().focusedGroupId) == nullptr) {
    graph.getViewport().focusedGroupId.reset();
    restoreViewPending = true;
  }

  renderPalette(graph, boxLibrary);
  ImGui::SameLine();
  ImGui::BeginChild("GraphCanvas", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  renderScopeBreadcrumb(graph);
  ImGui::BeginChild("GraphEditor", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  ed::SetCurrentEditor(context);

  const auto canvasOrigin = ImGui::GetWindowPos();
  const auto canvasSize = ImGui::GetWindowSize();
  auto &input = ImGui::GetIO();
  const auto overMap = mapHoveredLastFrame;
  const auto wheelX = input.MouseWheelH;
  const auto wheelY = input.MouseWheel;
  /** @brief Screen-space pointer used for zoom pivots (pre-canvas transform). */
  const auto mouseScreen = input.MousePos;
  /** @brief Screen-space drag delta used for middle-button pan. */
  const auto mouseDeltaScreen = input.MouseDelta;
  input.MouseWheel = 0.0f;
  input.MouseWheelH = 0.0f;
  const auto focusedGroupId = graph.getViewport().focusedGroupId;

  ed::Begin("OpenYourBox Graph");
  lastCanvasCentre = ed::ScreenToCanvas(
      ImVec2(canvasOrigin.x + canvasSize.x * 0.5f,
             canvasOrigin.y + canvasSize.y * 0.5f));

  if (restoreViewPending) {
    auto *editor = detailEditor();
    if (editor != nullptr) {
      juce::Point<float> pan = graph.getViewport().pan;
      float zoom = graph.getViewport().zoom;
      if (focusedGroupId.has_value()) {
        if (const auto *group = graph.findGroup(*focusedGroupId)) {
          pan = group->viewPan;
          zoom = group->viewZoom;
        }
      }
      zoom = std::clamp(zoom, minimumZoom, maximumZoom);
      const auto viewSize =
          ImVec2(canvasSize.x / std::max(0.01f, zoom),
                 canvasSize.y / std::max(0.01f, zoom));
      ImRect view;
      view.Min = ImVec2(pan.x, pan.y);
      view.Max = ImVec2(view.Min.x + viewSize.x, view.Min.y + viewSize.y);
      commitCanvasView(view);
    }
    restoreViewPending = false;
  }

  if ((std::abs(wheelX) > 0.0f || std::abs(wheelY) > 0.0f) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive() && !overMap) {
    if (input.KeyCtrl || input.KeySuper) {
      const auto zoomSteps = wheelY + wheelX;
      if (std::abs(zoomSteps) > 0.0f) {
        const auto zoomFactor = std::pow(1.1f, zoomSteps);
        navigateCanvas(ImVec2(0.0f, 0.0f), zoomFactor, mouseScreen);
      }
    } else {
      navigateCanvas(ImVec2(-wheelX * canvasWheelPanStep * canvasPanSpeed,
                            -wheelY * canvasWheelPanStep * canvasPanSpeed),
                     1.0f, mouseScreen);
    }
  }

  if (ImGui::IsMouseDragging(canvasDragPanButton, 0.0f) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive() && !overMap) {
    navigateCanvas(ImVec2(-mouseDeltaScreen.x * canvasPanSpeed,
                          -mouseDeltaScreen.y * canvasPanSpeed),
                   1.0f, mouseScreen);
  }

  if (std::abs(pinchMagnification - 1.0f) > 0.0001f && !overMap)
    navigateCanvas(ImVec2(0.0f, 0.0f), pinchMagnification, mouseScreen);

  if (const auto doubleClicked = ed::GetDoubleClickedNode()) {
    const auto id = static_cast<std::int32_t>(doubleClicked.Get());
    if (graph.isGroupOnFocusedCanvas(id, focusedGroupId))
      setCanvasFocus(graph, id);
  }

  dropTargetGroupId = 0;
  const auto dragging =
      ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.0f);
  const auto hoveredNode = ed::GetHoveredNode();
  std::int32_t hoveredGroupOnCanvas = 0;
  if (hoveredNode) {
    const auto hoveredId = static_cast<std::int32_t>(hoveredNode.Get());
    if (graph.isGroupOnFocusedCanvas(hoveredId, focusedGroupId))
      hoveredGroupOnCanvas = hoveredId;
  }
  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    const auto commitDrop = [&](std::int32_t boxId) {
      if (hoveredGroupOnCanvas == 0 || hoveredGroupOnCanvas == boxId)
        return;
      const auto result = graph.addToGroup(hoveredGroupOnCanvas, boxId);
      if (result.accepted)
        mutatedThisFrame = true;
      else if (!result.message.empty()) {
        transientMessage = result.message;
        transientMessageDeadline = ImGui::GetTime() + 2.5;
        if (callbacks.showMessage)
          callbacks.showMessage(result.message);
      }
    };
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      if (draggingNodeId != 0) {
        const auto editorPos =
            ed::GetNodePosition(ed::NodeId(editorIdentifier(draggingNodeId)));
        graph.moveNode(draggingNodeId, {editorPos.x, editorPos.y});
        commitDrop(draggingNodeId);
      } else if (draggingGroupId != 0) {
        const auto editorPos =
            ed::GetNodePosition(ed::NodeId(editorIdentifier(draggingGroupId)));
        graph.moveGroup(draggingGroupId, {editorPos.x, editorPos.y});
        commitDrop(draggingGroupId);
      }
    }
    draggingNodeId = 0;
    draggingGroupId = 0;
  } else if (draggingNodeId == 0 && draggingGroupId == 0 && dragging) {
    if (hoveredNode && !ed::GetHoveredPin() && !ed::GetHoveredLink()) {
      const auto id = static_cast<std::int32_t>(hoveredNode.Get());
      if (graph.isGroupOnFocusedCanvas(id, focusedGroupId))
        draggingGroupId = id;
      else if (graph.isNodeOnFocusedCanvas(id, focusedGroupId))
        draggingNodeId = id;
    }
  }

  if (draggingNodeId != 0 || draggingGroupId != 0)
    dropTargetGroupId = hoveredGroupOnCanvas;

  syncEditorTransforms(graph);

  for (auto &group : graph.getGroups()) {
    if (graph.isGroupOnFocusedCanvas(group.id, focusedGroupId))
      renderGroup(graph, group, callbacks);
  }

  for (auto &node : graph.getNodes()) {
    if (graph.isNodeOnFocusedCanvas(node.id, focusedGroupId))
      renderNode(graph, node, callbacks);
  }

  for (const auto &link : graph.getLinks()) {
    const auto sourceNode = graph.findNodeForPin(link.sourcePinId);
    const auto destNode = graph.findNodeForPin(link.destinationPinId);
    if (!sourceNode.has_value() || !destNode.has_value())
      continue;
    if (!nodeRepresentedOnCanvas(graph, *sourceNode, focusedGroupId) ||
        !nodeRepresentedOnCanvas(graph, *destNode, focusedGroupId))
      continue;
    const auto sourceHost =
        graph.focusedCanvasHostGroup(*sourceNode, focusedGroupId);
    const auto destHost =
        graph.focusedCanvasHostGroup(*destNode, focusedGroupId);
    if (sourceHost.has_value() && sourceHost == destHost)
      continue;
    const auto sourcePin =
        visualPinForLink(graph, link.sourcePinId, focusedGroupId);
    const auto destPin =
        visualPinForLink(graph, link.destinationPinId, focusedGroupId);
    ed::Link(ed::LinkId(editorIdentifier(link.id)),
             ed::PinId(editorIdentifier(sourcePin)),
             ed::PinId(editorIdentifier(destPin)),
             ImColor(100, 180, 255, 220), 2.0f);
  }

  handleConnections(graph, callbacks);
  handleDeletion(graph);
  synchronizeSelection(graph);
  handleContextMenus(graph, callbacks);

  if (!ImGui::GetIO().WantTextInput &&
      (ImGui::IsKeyPressed(ImGuiKey_Backspace) ||
       ImGui::IsKeyPressed(ImGuiKey_Delete))) {
    for (const auto linkId : selectedLinkIds)
      ed::DeleteLink(ed::LinkId(editorIdentifier(linkId)));
    for (const auto nodeId : selectedNodeIds) {
      if (!graph.isFixedIoNode(nodeId))
        ed::DeleteNode(ed::NodeId(editorIdentifier(nodeId)));
    }
    for (const auto groupId : selectedGroupIds)
      ed::DeleteNode(ed::NodeId(editorIdentifier(groupId)));
  }

  renderMap(graph, canvasOrigin, canvasSize);

  ed::End();
  applyPendingContextActions(graph);

  if (requestSaveBoxPopup) {
    ImGui::OpenPopup("Save to Box Library");
    requestSaveBoxPopup = false;
  }
  if (ImGui::BeginPopupModal("Save to Box Library", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Name");
    ImGui::SetNextItemWidth(260.0f);
    ImGui::InputText("##saveBoxName", saveBoxNameBuffer.data(),
                     saveBoxNameBuffer.size());
    const auto folder = boxLibraryPanel.getSelectedFolder();
    if (folder.isNotEmpty())
      ImGui::TextDisabled("Folder: User Library/%s", folder.toRawUTF8());
    else
      ImGui::TextDisabled("Folder: User Library");
    if (saveBoxOverwrite)
      ImGui::TextWrapped("A box with this name exists and will be replaced.");
    if (ImGui::Button("Save") && activeBoxLibrary != nullptr) {
      juce::String error;
      const auto name = juce::String(saveBoxNameBuffer.data());
      auto result = activeBoxLibrary->saveBox(graph, pendingSaveBoxId, name,
                                              folder, saveBoxOverwrite, error);
      if (!result.has_value() &&
          error.containsIgnoreCase("already exists") && !saveBoxOverwrite) {
        saveBoxOverwrite = true;
      } else {
        if (result.has_value())
          mutatedThisFrame = true;
        else if (error.isNotEmpty()) {
          transientMessage = error.toStdString();
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(error.toStdString());
        }
        pendingSaveBoxId = 0;
        saveBoxOverwrite = false;
        ImGui::CloseCurrentPopup();
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      pendingSaveBoxId = 0;
      saveBoxOverwrite = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (ImGui::BeginDragDropTarget()) {
    if (const auto *payload =
            ImGui::AcceptDragDropPayload("OPENYOURBOX_NODE_TYPE")) {
      const auto item = *static_cast<const PaletteItem *>(payload->Data);
      const auto canvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
      const auto dropGroup =
          groupBoxAtCanvas(graph, canvasPosition, focusedGroupId);
      if (dropGroup != 0) {
        const auto *target = graph.findGroup(dropGroup);
        const auto local =
            target != nullptr
                ? ImVec2(canvasPosition.x - target->position.x,
                         canvasPosition.y - target->position.y)
                : canvasPosition;
        graph.addNode(item.type, {local.x, local.y}, dropGroup);
      } else {
        graph.addNode(item.type, {canvasPosition.x, canvasPosition.y},
                      focusedGroupId);
      }
      positionedNodeIds.erase(graph.getNodes().back().id);
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
    if (const auto *payload =
            ImGui::AcceptDragDropPayload(raveLayoutPayloadId)) {
      const auto item = *static_cast<const RavePaletteItem *>(payload->Data);
      const auto error = insertRaveLayout(graph, item.layout, item.channels);
      if (!error.empty()) {
        transientMessage = error;
        transientMessageDeadline = ImGui::GetTime() + 3.0;
        if (callbacks.showMessage)
          callbacks.showMessage(error);
      } else {
        mutatedThisFrame = true;
        recompileThisFrame = true;
      }
    }
    if (boxLibrary != nullptr) {
      if (const auto *payload =
              ImGui::AcceptDragDropPayload("OPENYOURBOX_BOX_LIBRARY_ID")) {
        const auto entryId = juce::String::fromUTF8(
            static_cast<const char *>(payload->Data));
        const auto canvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        juce::String error;
        const auto rootId = boxLibrary->insertBox(
            graph, entryId, {canvasPosition.x, canvasPosition.y}, error);
        if (rootId.has_value()) {
          positionedNodeIds.erase(*rootId);
          positionedGroupIds.erase(*rootId);
          adoptNewBox(graph, *rootId, canvasPosition,
                      groupBoxAtCanvas(graph, canvasPosition, focusedGroupId));
          mutatedThisFrame = true;
          recompileThisFrame = true;
          if (callbacks.boxPlaced)
            callbacks.boxPlaced(*rootId);
        } else if (error.isNotEmpty()) {
          transientMessage = error.toStdString();
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(error.toStdString());
        }
      }
    }
    ImGui::EndDragDropTarget();
  }

  for (auto &node : graph.getNodes()) {
    if (positionedNodeIds.count(node.id) == 0)
      continue;
    const auto size = ed::GetNodeSize(ed::NodeId(editorIdentifier(node.id)));
    if (size.x > 0.0f && size.y > 0.0f)
      node.size = {size.x, size.y};
    if (node.id != draggingNodeId)
      continue;
    const auto position =
        ed::GetNodePosition(ed::NodeId(editorIdentifier(node.id)));
    const auto stored = juce::Point<float>(position.x, position.y);
    if (std::abs(node.position.x - stored.x) >
            std::numeric_limits<float>::epsilon() ||
        std::abs(node.position.y - stored.y) >
            std::numeric_limits<float>::epsilon()) {
      graph.moveNode(node.id, stored);
      layoutMutatedThisFrame = true;
    }
  }
  for (auto &group : graph.getGroups()) {
    if (positionedGroupIds.count(group.id) == 0)
      continue;
    const auto size = ed::GetNodeSize(ed::NodeId(editorIdentifier(group.id)));
    if (size.x > 0.0f && size.y > 0.0f)
      group.size = {size.x, size.y};
    if (draggingGroupId != group.id)
      continue;
    const auto position =
        ed::GetNodePosition(ed::NodeId(editorIdentifier(group.id)));
    const auto stored = juce::Point<float>(position.x, position.y);
    if (std::abs(group.position.x - stored.x) >
            std::numeric_limits<float>::epsilon() ||
        std::abs(group.position.y - stored.y) >
            std::numeric_limits<float>::epsilon()) {
      graph.moveGroup(group.id, stored);
      layoutMutatedThisFrame = true;
    }
  }
  const auto zoom =
      std::clamp(detailEditor() != nullptr ? detailEditor()->GetView().Scale
                                           : ed::GetCurrentZoom(),
                 minimumZoom, maximumZoom);
  const auto viewOrigin = ed::ScreenToCanvas(canvasOrigin);
  if (focusedGroupId.has_value()) {
    if (graph.findGroup(*focusedGroupId) != nullptr)
      graph.setGroupView(*focusedGroupId, {viewOrigin.x, viewOrigin.y}, zoom);
  } else {
    graph.getViewport().zoom = zoom;
    graph.getViewport().pan = {viewOrigin.x, viewOrigin.y};
  }

  if (!transientMessage.empty() &&
      ImGui::GetTime() < transientMessageDeadline) {
    const auto available = ImGui::GetContentRegionAvail();
    ImGui::SetCursorPos(ImVec2(12.0f, std::max(12.0f, available.y - 34.0f)));
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       transientMessage.c_str());
  }

  ed::SetCurrentEditor(nullptr);
  ImGui::EndChild();
  ImGui::EndChild();
  if (mutatedThisFrame && callbacks.documentChanged)
    callbacks.documentChanged(recompileThisFrame, true);
  else if (layoutMutatedThisFrame && callbacks.documentChanged)
    callbacks.documentChanged(false, false);
}

std::int32_t NodeRenderer::getPrimarySelectedNodeId() const noexcept {
  return selectedNodeIds.empty() ? 0 : selectedNodeIds.front();
}

void NodeRenderer::renderPalette(NodeGraph &graph,
                                 openyourbox::library::UserBoxLibrary *boxLibrary) {
  ImGui::BeginChild("ElementPalette", ImVec2(200.0f, 0.0f), true);
  ImGui::TextColored(ImVec4(0.39f, 0.70f, 1.0f, 1.0f), "Library");
  ImGui::TextDisabled("Drag onto the graph");
  ImGui::Separator();

  const auto factorySelected =
      boxLibrary == nullptr || boxLibraryPanel.isFactorySelected();
  const auto factoryFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
      ImGuiTreeNodeFlags_SpanAvailWidth |
      (factorySelected ? ImGuiTreeNodeFlags_Selected : 0);
  const auto factoryOpen = ImGui::TreeNodeEx("Factory", factoryFlags);
  if (boxLibrary != nullptr && ImGui::IsItemClicked() &&
      !ImGui::IsItemToggledOpen())
    boxLibraryPanel.selectFactory();
  if (ImGui::BeginPopupContextItem("FactoryRootMenu")) {
    ImGui::TextDisabled("Factory cannot be renamed or deleted");
    ImGui::EndPopup();
  }
  if (factoryOpen) {
    for (const auto &item : paletteItems) {
      ImGui::PushID(item.label);
      ImGui::Selectable(item.label, false,
                        ImGuiSelectableFlags_AllowDoubleClick);
      if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        const auto focus = graph.getViewport().focusedGroupId;
        graph.addNode(item.type, {lastCanvasCentre.x, lastCanvasCentre.y},
                      focus);
        mutatedThisFrame = true;
        recompileThisFrame = true;
      }
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        ImGui::SetDragDropPayload("OPENYOURBOX_NODE_TYPE", &item, sizeof(item));
        ImGui::TextUnformatted(item.label);
        ImGui::EndDragDropSource();
      }
      ImGui::PopID();
    }
    if (ImGui::TreeNodeEx("RAVE layouts",
                          ImGuiTreeNodeFlags_OpenOnArrow |
                              ImGuiTreeNodeFlags_SpanAvailWidth)) {
      const auto renderRaveItem = [&](const RavePaletteItem &item) {
        ImGui::PushID(item.label);
        ImGui::PushID(static_cast<int>(item.layout));
        ImGui::PushID(item.channels);
        ImGui::Selectable(item.label, false,
                          ImGuiSelectableFlags_AllowDoubleClick);
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
          const auto error =
              insertRaveLayout(graph, item.layout, item.channels);
          if (!error.empty()) {
            transientMessage = error;
            transientMessageDeadline = ImGui::GetTime() + 3.0;
          } else {
            mutatedThisFrame = true;
            recompileThisFrame = true;
          }
        }
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
          ImGui::SetDragDropPayload(raveLayoutPayloadId, &item, sizeof(item));
          ImGui::TextUnformatted(item.label);
          ImGui::EndDragDropSource();
        }
        ImGui::PopID();
        ImGui::PopID();
        ImGui::PopID();
      };
      if (ImGui::TreeNodeEx("Original", ImGuiTreeNodeFlags_OpenOnArrow |
                                            ImGuiTreeNodeFlags_SpanAvailWidth)) {
        renderRaveItem({"Mono", RaveLayoutId::original, 1});
        renderRaveItem({"Stereo", RaveLayoutId::original, 2});
        ImGui::TreePop();
      }
      if (ImGui::TreeNodeEx("Latest continuous",
                            ImGuiTreeNodeFlags_OpenOnArrow |
                                ImGuiTreeNodeFlags_SpanAvailWidth)) {
        renderRaveItem({"Mono", RaveLayoutId::latestContinuous, 1});
        renderRaveItem({"Stereo", RaveLayoutId::latestContinuous, 2});
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (boxLibrary != nullptr) {
    openyourbox::ui::UserBoxLibraryPanel::Callbacks boxCallbacks;
    boxCallbacks.showMessage = [this](const std::string &message) {
      transientMessage = message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
    };
    boxLibraryPanel.render(*boxLibrary, boxCallbacks);
  }
  ImGui::TextWrapped(
      "Scroll to pan. Ctrl/Cmd+scroll or pinch to zoom at the pointer. "
      "Double-click a group to open it. Use Graph > group at the top to go back.");
  ImGui::EndChild();
}

void NodeRenderer::renderNode(NodeGraph &graph, GraphNode &node,
                              const NodeRendererCallbacks &callbacks) {
  positionedNodeIds.insert(node.id);
  const auto nodeColour = colourFor(node.colour);
  ed::PushStyleColor(ed::StyleColor_NodeBg,
                     ImVec4(nodeColour.x, nodeColour.y, nodeColour.z, 0.28f));
  ed::PushStyleColor(ed::StyleColor_NodeBorder, nodeColour);
  ed::BeginNode(ed::NodeId(editorIdentifier(node.id)));
  ImGui::PushID(node.id);
  ImGui::Dummy(ImVec2(nodeBodyWidth, 0.0f));
  ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + nodeBodyWidth);

  if (node.state == NodeState::frozenGold)
    ImGui::TextUnformatted("\xF0\x9F\x94\x92");
  if (node.state == NodeState::frozenGold)
    ImGui::SameLine();
  ImGui::TextUnformatted(node.label.c_str());
  drawNodeDivider();

  for (const auto &pin : node.inputs) {
    ed::BeginPin(ed::PinId(editorIdentifier(pin.id)), ed::PinKind::Input);
    const auto shapeLabel = pin.shape.displayLabel();
    if (!shapeLabel.empty())
      ImGui::Text("<- %s (%s)", pin.label.c_str(), shapeLabel.c_str());
    else
      ImGui::Text("<- %s", pin.label.c_str());
    ed::EndPin();
  }

  ImGui::TextDisabled("%s", node.detail.c_str());
  int convStride = 1;
  if (node.type == NodeType::convolution || node.type == NodeType::convTranspose) {
    for (const auto &property : node.properties) {
      if (property.key == "stride")
        convStride = property.value;
    }
  }
  const bool stridedConv =
      (node.type == NodeType::convolution || node.type == NodeType::convTranspose) &&
      convStride > 1;
  if (isRaveProcessingType(node.type) || stridedConv ||
      (node.type == NodeType::blackBox && node.outputs.size() > 1)) {
    const auto delay = raveNodeDelaySamples(node);
    if (delay > 0 || node.type == NodeType::blackBox) {
      const auto samples =
          node.metrics.has_value()
              ? static_cast<double>(
                    std::max<std::uint64_t>(delay, 1))
              : static_cast<double>(delay);
      ImGui::TextDisabled("Delay %.0f smp / %.2f ms @ 48 kHz", samples,
                          samples * 1000.0 / 48000.0);
    }
  }
  const auto frozen = node.state == NodeState::frozenGold;
  if (node.type == NodeType::knobInput)
    renderKnobControl(graph, node, callbacks);
  else if (node.type == NodeType::xyTrackpad)
    renderXyPad(graph, node, callbacks);
  if (frozen)
    ImGui::BeginDisabled();
  for (auto &property : node.properties) {
    const bool liveOnGold = frozen && property.key == "fidelity";
    if (liveOnGold)
      ImGui::EndDisabled();
    ImGui::PushID(property.key.c_str());
    auto value = property.value;
    bool changed = false;
    int dragSteps = 0;
    beginPropertyRow(property.label.c_str());
    if (property.key == "residual") {
      bool residual = property.value != 0;
      if (ImGui::Checkbox("##residual", &residual)) {
        const auto previous = property.value;
        property.setValue(residual ? 1 : 0);
        if (!graph.setProperty(node.id, property.key, property.value))
          property.setValue(previous);
        else {
          mutatedThisFrame = true;
          recompileThisFrame = true;
          if (callbacks.propertyChanged)
            callbacks.propertyChanged(node.id, property.key, property.value);
        }
      }
      ImGui::PopID();
      continue;
    }
    if (property.kind == PropertyKind::choice && !property.choices.empty()) {
      const auto choiceIndex =
          std::clamp(value, 0, static_cast<int>(property.choices.size()) - 1);
      if (ImGui::Button(
              property.choices[static_cast<std::size_t>(choiceIndex)].c_str(),
              ImVec2(propertyValueWidth, 0.0f))) {
        activePropertyCombo.nodeId = node.id;
        activePropertyCombo.propertyKey = property.key;
        activePropertyCombo.active = true;
        activePropertyCombo.requestOpen = true;
        activePropertyCombo.anchorMin =
            ed::CanvasToScreen(ImGui::GetItemRectMin());
      }
    } else if (property.kind == PropertyKind::real) {
      constexpr float handleWidth = 18.0f;
      const auto fieldHeight = ImGui::GetFrameHeight();
      const auto fieldOrigin = ImGui::GetCursorScreenPos();
      ImGui::GetWindowDrawList()->AddRectFilled(
          fieldOrigin,
          ImVec2(fieldOrigin.x + propertyValueWidth,
                 fieldOrigin.y + fieldHeight),
          ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);

      auto floatValue = property.floatValue;
      dragSteps = propertyDragSteps("##drag", ImVec2(handleWidth, fieldHeight));
      if (dragSteps != 0) {
        const auto span = property.floatMaximum - property.floatMinimum;
        const auto stepPerTick =
            ImGui::GetIO().KeyShift ? span / 2000.0f : span / 200.0f;
        floatValue += static_cast<float>(dragSteps) * stepPerTick;
        changed = true;
      }
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
      ImGui::SetNextItemWidth(propertyValueWidth - handleWidth);
      changed = ImGui::InputFloat("##gain", &floatValue, 0.0f, 0.0f, "%.2f") ||
                changed;
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
      if (changed) {
        if (floatValue < property.floatMinimum ||
            floatValue > property.floatMaximum) {
          transientMessage = property.label + " must be between " +
                             std::to_string(property.floatMinimum) + " and " +
                             std::to_string(property.floatMaximum);
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(transientMessage);
        }
        const auto previous = property.floatValue;
        property.setFloatValue(floatValue);
        if (std::abs(property.floatValue - previous) > 1.0e-6f) {
          if (!graph.setFloatProperty(node.id, property.key,
                                      property.floatValue)) {
            property.setFloatValue(previous);
          } else {
            mutatedThisFrame = true;
            if (callbacks.floatPropertyChanged)
              callbacks.floatPropertyChanged(node.id, property.key,
                                             property.floatValue);
          }
        }
      }
    } else {
      constexpr float handleWidth = 18.0f;
      const auto fieldHeight = ImGui::GetFrameHeight();
      const auto fieldOrigin = ImGui::GetCursorScreenPos();
      ImGui::GetWindowDrawList()->AddRectFilled(
          fieldOrigin,
          ImVec2(fieldOrigin.x + propertyValueWidth,
                 fieldOrigin.y + fieldHeight),
          ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);

      dragSteps = propertyDragSteps("##drag", ImVec2(handleWidth, fieldHeight));
      if (dragSteps != 0)
        value += dragSteps;
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
      ImGui::SetNextItemWidth(propertyValueWidth - handleWidth);
      changed = ImGui::InputInt("##value", &value, 0, 0) || dragSteps != 0;
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
    }
    if (changed && property.kind != PropertyKind::real) {
      if (dragSteps == 0 &&
          (value < property.minimum || value > property.maximum)) {
        transientMessage = property.label + " must be between " +
                           std::to_string(property.minimum) + " and " +
                           std::to_string(property.maximum);
        transientMessageDeadline = ImGui::GetTime() + 2.5;
        if (callbacks.showMessage)
          callbacks.showMessage(transientMessage);
      }
      const auto previous = property.value;
      property.setValue(value);
      if (property.value != previous) {
        if (!graph.setProperty(node.id, property.key, property.value)) {
          property.setValue(previous);
          transientMessage =
              "That mode would break downstream channel compatibility";
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(transientMessage);
        } else {
          mutatedThisFrame = true;
          recompileThisFrame = true;
          if (callbacks.propertyChanged)
            callbacks.propertyChanged(node.id, property.key, property.value);
        }
      }
    }
    ImGui::PopID();
    if (liveOnGold)
      ImGui::BeginDisabled();
  }
  if (frozen)
    ImGui::EndDisabled();

  if (node.type == NodeType::convolution || node.type == NodeType::convTranspose) {
    const auto inputRate =
        node.inputs.empty() ? 0 : node.inputs.front().shape.temporalRate;
    const auto upsample = node.type == NodeType::convTranspose;
    const auto notice =
        convolutionRateMessage(convStride, upsample, inputRate);
    if (!notice.empty()) {
      const auto error =
          convolutionRateIsError(convStride, upsample, inputRate);
      const ImVec4 colour = error ? ImVec4(1.0f, 0.38f, 0.38f, 1.0f)
                                  : ImVec4(1.0f, 0.78f, 0.28f, 1.0f);
      ImGui::PushStyleColor(ImGuiCol_Text, colour);
      ImGui::TextWrapped("%s", notice.c_str());
      ImGui::PopStyleColor();
    }
  }

  if (node.hasWeights && node.state == NodeState::liveBlue) {
    const auto isBatchNorm =
        node.type == openyourbox::graph::NodeType::batchNorm;
    if (!isBatchNorm) {
      const auto insertion = seedBuffers.try_emplace(node.id);
      auto &seedBuffer = insertion.first->second;
      if (insertion.second)
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d",
                      node.useExplicitSeed ? node.explicitSeed : node.seed);
      if (!node.useExplicitSeed)
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", node.seed);

      const auto rowOrigin = ImGui::GetCursorScreenPos();
      ImGui::TextUnformatted("Seed");
      constexpr float seedControlWidth = 96.0f;
      ImGui::SetCursorScreenPos(
          ImVec2(rowOrigin.x + nodeBodyWidth - seedControlWidth, rowOrigin.y));
      if (ImGui::Checkbox("##useExplicitSeed", &node.useExplicitSeed)) {
        if (node.useExplicitSeed)
          std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d",
                        node.explicitSeed);
        else
          std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", node.seed);
        mutatedThisFrame = true;
      }
      ImGui::SameLine();
      ImGui::SetNextItemWidth(seedControlWidth - ImGui::GetFrameHeight() - 8.0f);
      if (!node.useExplicitSeed)
        ImGui::BeginDisabled();
      if (ImGui::InputText("##seed", seedBuffer.data(), seedBuffer.size(),
                           ImGuiInputTextFlags_EnterReturnsTrue)) {
        std::int32_t parsed = 0;
        const auto *begin = seedBuffer.data();
        const auto *end = begin + std::strlen(begin);
        const auto result = std::from_chars(begin, end, parsed);
        if (result.ec == std::errc{} && result.ptr == end &&
            parsed >= minimumSeed && parsed <= maximumSeed) {
          graph.setSeed(node.id, parsed);
          node.seed = clampSeed(parsed);
          node.explicitSeed = node.seed;
          mutatedThisFrame = true;
        } else {
          transientMessage = "Seed must be an integer between 0 and 999999";
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(transientMessage);
          std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d",
                        node.useExplicitSeed ? node.explicitSeed : node.seed);
        }
      }
      if (!node.useExplicitSeed)
        ImGui::EndDisabled();
    }

    const char *weightsActionLabel =
        isBatchNorm ? "Reset" : "Randomize Weights";
    if (ImGui::Button(weightsActionLabel, ImVec2(nodeBodyWidth, 0.0f)) &&
        callbacks.randomizeNode) {
      std::int32_t appliedSeed = node.seed;
      if (isBatchNorm) {
        // BatchNorm Reset restores identity affine + stats; seed is unused.
        appliedSeed = 0;
        graph.setSeed(node.id, appliedSeed);
        node.seed = appliedSeed;
      } else if (node.useExplicitSeed) {
        auto &seedBuffer = seedBuffers[node.id];
        std::int32_t parsed = 0;
        const auto *begin = seedBuffer.data();
        const auto *end = begin + std::strlen(begin);
        const auto parsedSeed = std::from_chars(begin, end, parsed);
        if (parsedSeed.ec == std::errc{} && parsedSeed.ptr == end &&
            parsed >= minimumSeed && parsed <= maximumSeed) {
          appliedSeed = clampSeed(parsed);
          graph.setSeed(node.id, appliedSeed);
          node.seed = appliedSeed;
          node.explicitSeed = appliedSeed;
        } else {
          appliedSeed = node.explicitSeed;
        }
      } else {
        auto &seedBuffer = seedBuffers[node.id];
        appliedSeed = juce::Random::getSystemRandom().nextInt(maximumSeed + 1);
        graph.setSeed(node.id, appliedSeed);
        node.seed = appliedSeed;
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", appliedSeed);
      }
      mutatedThisFrame = true;
      // N>1 copies store independent seeds in copySlots; recompile so every
      // materialized copy rebuilds from its own slot instead of only reseeding
      // the visible template node.
      if (graph.effectiveCopyCount(node.id) > 1)
        recompileThisFrame = true;
      callbacks.randomizeNode(node.id, appliedSeed);
    }
  }

  if (node.hasWeights && node.state == NodeState::liveBlue &&
      !isControlSourceType(node.type)) {
    bool armed = node.armedForTraining;
    if (ImGui::Checkbox("Arm for training", &armed)) {
      graph.setArmedForTraining(node.id, armed);
      mutatedThisFrame = true;
      if (callbacks.armChanged)
        callbacks.armChanged(node.id, armed);
    }
  }

  if (node.hasWeights || node.type == NodeType::blackBox) {
    if (node.weightsProvenance == WeightsProvenance::file &&
        !node.weightsPath.empty())
      ImGui::TextWrapped("Weights: %s",
                         juce::File(node.weightsPath).getFileName().toRawUTF8());
    else if (node.type == NodeType::batchNorm)
      ImGui::TextUnformatted("Weights: identity");
    else
      ImGui::Text("Weights: seed %d", node.seed);
    if (ImGui::SmallButton("Browse weights") && callbacks.browseWeights)
      callbacks.browseWeights(node.id);
  }

  if (node.metrics.has_value()) {
    drawNodeDivider();
    ImGui::Text("Compile: %.1f ms", node.metrics->compileTimeMilliseconds);
    ImGui::Text("Inference: %.3f ms", node.metrics->inferenceTimeMilliseconds);
  }

  for (const auto &pin : node.outputs) {
    ed::BeginPin(ed::PinId(editorIdentifier(pin.id)), ed::PinKind::Output);
    const auto shapeLabel = pin.shape.displayLabel();
    if (!shapeLabel.empty())
      ImGui::Text("%s (%s) ->", pin.label.c_str(), shapeLabel.c_str());
    else
      ImGui::Text("%s ->", pin.label.c_str());
    ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ed::EndPin();
  }

  if (ImGui::SmallButton("Analyze") && callbacks.analysisRequested)
    callbacks.analysisRequested(node.id);

  ImGui::PopTextWrapPos();
  ImGui::PopID();
  ed::EndNode();
  ed::PopStyleColor(2);
}

void NodeRenderer::renderKnobControl(NodeGraph &graph, GraphNode &node,
                                     const NodeRendererCallbacks &callbacks) {
  auto value = node.conditioningValue;
  ImGui::SetNextItemWidth(76.0f);
  if (ImGui::VSliderFloat("##knob", ImVec2(28.0f, 72.0f), &value,
                          conditioningMinimum, conditioningMaximum, "")) {
    graph.setConditioningValue(node.id, value);
    if (callbacks.knobChanged)
      callbacks.knobChanged(node.id, node.conditioningValue);
  }
  ImGui::SameLine();
  ImGui::Text("%.2f", static_cast<double>(node.conditioningValue));
}

void NodeRenderer::renderXyPad(NodeGraph &graph, GraphNode &node,
                               const NodeRendererCallbacks &callbacks) {
  const ImVec2 padSize{118.0f, 118.0f};
  const auto origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##xypad", padSize);
  const auto hovered = ImGui::IsItemHovered();
  const auto active = ImGui::IsItemActive();
  auto *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin,
                      ImVec2(origin.x + padSize.x, origin.y + padSize.y),
                      ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
  draw->AddRect(origin, ImVec2(origin.x + padSize.x, origin.y + padSize.y),
                ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
  const auto normalize = [](float value) {
    return (clampConditioning(value) - conditioningMinimum) /
           (conditioningMaximum - conditioningMinimum);
  };
  auto handle = ImVec2(origin.x + normalize(node.conditioningX) * padSize.x,
                       origin.y + (1.0f - normalize(node.conditioningY)) *
                                      padSize.y);
  if ((hovered || active) && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    const auto mouse = ImGui::GetIO().MousePos;
    const auto nx = std::clamp((mouse.x - origin.x) / padSize.x, 0.0f, 1.0f);
    const auto ny =
        std::clamp(1.0f - (mouse.y - origin.y) / padSize.y, 0.0f, 1.0f);
    const auto x =
        conditioningMinimum + nx * (conditioningMaximum - conditioningMinimum);
    const auto y =
        conditioningMinimum + ny * (conditioningMaximum - conditioningMinimum);
    graph.setConditioningPad(node.id, x, y);
    if (callbacks.xyChanged)
      callbacks.xyChanged(node.id, node.conditioningX, node.conditioningY);
    handle = ImVec2(origin.x + nx * padSize.x, origin.y + (1.0f - ny) * padSize.y);
  }
  draw->AddCircleFilled(handle, 5.0f, ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
  ImGui::Text("X %.2f  Y %.2f", static_cast<double>(node.conditioningX),
              static_cast<double>(node.conditioningY));
}

void NodeRenderer::renderGroup(NodeGraph &graph, GraphGroup &group,
                               const NodeRendererCallbacks &callbacks) {
  positionedGroupIds.insert(group.id);
  const auto highlight = dropTargetGroupId == group.id;
  const auto frame = colourFor(highlight ? groupDropHighlightColour
                                         : groupFrameColour);
  ed::PushStyleColor(ed::StyleColor_NodeBg,
                     ImVec4(frame.x, frame.y, frame.z, highlight ? 0.35f : 0.18f));
  ed::PushStyleColor(ed::StyleColor_NodeBorder,
                     ImVec4(frame.x, frame.y, frame.z, highlight ? 1.0f : 0.85f));
  ed::PushStyleColor(ed::StyleColor_GroupBg,
                     ImVec4(frame.x, frame.y, frame.z, highlight ? 0.22f : 0.10f));
  ed::PushStyleColor(ed::StyleColor_GroupBorder,
                     ImVec4(frame.x, frame.y, frame.z, highlight ? 1.0f : 0.55f));
  ed::BeginNode(ed::NodeId(editorIdentifier(group.id)));
  ImGui::PushID(group.id);

  auto &nameBuffer = groupNameBuffers[group.id];
  if (nameBuffer[0] == '\0') {
    const auto copyCount =
        std::min(group.name.size(), nameBuffer.size() - 1);
    std::memcpy(nameBuffer.data(), group.name.data(), copyCount);
    nameBuffer[copyCount] = '\0';
  }
  ImGui::SetNextItemWidth(140.0f);
  if (ImGui::InputText("##groupName", nameBuffer.data(), nameBuffer.size())) {
    if (graph.renameGroup(group.id, nameBuffer.data()))
      mutatedThisFrame = true;
  }
  ImGui::SameLine();
  int copies = group.copies;
  ImGui::SetNextItemWidth(48.0f);
  if (ImGui::InputInt("##copies", &copies, 0, 0)) {
    const auto result = graph.setGroupCopies(group.id, copies);
    if (result.accepted) {
      mutatedThisFrame = true;
      recompileThisFrame = true;
    } else if (!result.message.empty()) {
      transientMessage = result.message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (callbacks.showMessage)
        callbacks.showMessage(result.message);
    }
  }
  ImGui::SameLine();
  ImGui::TextDisabled("copies");
  if (ImGui::Button("Randomize Weights", ImVec2(-FLT_MIN, 0.0f))) {
    if (graph.randomizeGroupWeights(group.id)) {
      for (const auto leaf : graph.collectLeafNodeIds(group.id)) {
        if (const auto *node = graph.findNode(leaf)) {
          auto &seedBuffer = seedBuffers[leaf];
          std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d",
                        node->seed);
        }
      }
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
  }
  ImGui::TextDisabled("Double-click to open");

  std::vector<GroupBoundaryPort> inputs;
  std::vector<GroupBoundaryPort> outputs;
  for (const auto &port : graph.groupInterfacePorts(group.id)) {
    if (port.kind == PinKind::input)
      inputs.push_back(port);
    else
      outputs.push_back(port);
  }
  const auto drawPort = [](const GroupBoundaryPort &port) {
    const auto pinId = collapsedGroupPinId(port.memberPinId);
    ed::BeginPin(ed::PinId(editorIdentifier(pinId)),
                 port.kind == PinKind::input ? ed::PinKind::Input
                                             : ed::PinKind::Output);
    const auto shapeLabel = port.shape.displayLabel();
    if (port.kind == PinKind::input) {
      if (!shapeLabel.empty())
        ImGui::Text("<- %s (%s)", port.label.c_str(), shapeLabel.c_str());
      else
        ImGui::Text("<- %s", port.label.c_str());
    } else {
      if (!shapeLabel.empty())
        ImGui::Text("%s (%s) ->", port.label.c_str(), shapeLabel.c_str());
      else
        ImGui::Text("%s ->", port.label.c_str());
      ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    }
    ed::EndPin();
  };
  for (const auto &port : inputs)
    drawPort(port);
  for (const auto &port : outputs)
    drawPort(port);

  ImGui::PopID();
  ed::EndNode();
  ed::PopStyleColor(4);
}

void NodeRenderer::handleConnections(NodeGraph &graph,
                                     const NodeRendererCallbacks &callbacks) {
  if (!ed::BeginCreate())
    return;
  ed::PinId first;
  ed::PinId second;
  if (ed::QueryNewLink(&first, &second) && first && second) {
    const auto result = graph.connect(static_cast<std::int32_t>(first.Get()),
                                      static_cast<std::int32_t>(second.Get()));
    if (result.accepted) {
      if (!ed::AcceptNewItem())
        graph.removeLink(graph.getLinks().back().id);
      else {
        mutatedThisFrame = true;
        recompileThisFrame = true;
      }
    } else {
      ed::RejectNewItem(ImVec4(1.0f, 0.15f, 0.15f, 1.0f), 3.0f);
      transientMessage = result.message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (callbacks.showMessage)
        callbacks.showMessage(result.message);
    }
  }
  ed::EndCreate();
}

void NodeRenderer::handleDeletion(NodeGraph &graph) {
  if (!ed::BeginDelete())
    return;
  ed::LinkId linkId;
  while (ed::QueryDeletedLink(&linkId)) {
    if (ed::AcceptDeletedItem()) {
      mutatedThisFrame =
          graph.removeLink(static_cast<std::int32_t>(linkId.Get())) ||
          mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
    }
  }
  ed::NodeId nodeId;
  while (ed::QueryDeletedNode(&nodeId)) {
    const auto id = static_cast<std::int32_t>(nodeId.Get());
    if (graph.findGroup(id) != nullptr) {
      if (ed::AcceptDeletedItem()) {
        const auto *group = graph.findGroup(id);
        const auto parent =
            group != nullptr ? group->parentGroupId : std::nullopt;
        const auto focus = graph.getViewport().focusedGroupId;
        bool leaveScope = false;
        if (focus.has_value()) {
          for (const auto ancestor : graph.groupAncestorChain(*focus)) {
            if (ancestor == id) {
              leaveScope = true;
              break;
            }
          }
        }
        mutatedThisFrame = graph.deleteGroup(id).accepted || mutatedThisFrame;
        recompileThisFrame = mutatedThisFrame || recompileThisFrame;
        positionedGroupIds.erase(id);
        if (leaveScope)
          setCanvasFocus(graph, parent);
      }
      continue;
    }
    const auto *node = graph.findNode(id);
    if (node == nullptr || graph.isFixedIoNode(id) ||
        (node->state == NodeState::frozenGold &&
         node->type != NodeType::blackBox)) {
      ed::RejectDeletedItem();
      continue;
    }
    if (ed::AcceptDeletedItem()) {
      mutatedThisFrame = graph.removeNode(id) || mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
      positionedNodeIds.erase(id);
    }
  }
  ed::EndDelete();
}

void NodeRenderer::handleContextMenus(NodeGraph &graph,
                                      const NodeRendererCallbacks &callbacks) {
  ed::Suspend();
  ed::NodeId nodeId;
  ed::LinkId linkId;
  ed::PinId pinId;
  if (ed::ShowPinContextMenu(&pinId)) {
    contextPinId = static_cast<std::int32_t>(pinId.Get());
    contextNodeId = 0;
    contextLinkId = 0;
    if (!graph.isPinConnected(contextPinId))
      ImGui::OpenPopup("Pin Context");
  }
  if (ed::ShowNodeContextMenu(&nodeId)) {
    contextNodeId = static_cast<std::int32_t>(nodeId.Get());
    contextGroupId = graph.findGroup(contextNodeId) != nullptr ? contextNodeId : 0;
    if (contextGroupId != 0)
      contextNodeId = 0;
    contextLinkId = 0;
    contextPinId = 0;
    ImGui::OpenPopup("Node Context");
  }
  if (ed::ShowLinkContextMenu(&linkId)) {
    contextLinkId = static_cast<std::int32_t>(linkId.Get());
    contextNodeId = 0;
    contextGroupId = 0;
    contextPinId = 0;
    ImGui::OpenPopup("Link Context");
  }
  if (ed::ShowBackgroundContextMenu()) {
    contextNodeId = 0;
    contextGroupId = 0;
    contextLinkId = 0;
    contextPinId = 0;
    ImGui::OpenPopup("Node Context");
  }

  if (ImGui::BeginPopup("Node Context")) {
    std::vector<std::int32_t> freezeIds;
    freezeIds.reserve(selectedNodeIds.size() + selectedGroupIds.size() + 2);
    for (const auto id : selectedNodeIds) {
      if (!graph.isFixedIoNode(id))
        freezeIds.push_back(id);
    }
    for (const auto id : selectedGroupIds)
      freezeIds.push_back(id);
    if (contextNodeId != 0 && !graph.isFixedIoNode(contextNodeId) &&
        std::find(freezeIds.begin(), freezeIds.end(), contextNodeId) ==
            freezeIds.end())
      freezeIds.push_back(contextNodeId);
    if (contextGroupId != 0 &&
        std::find(freezeIds.begin(), freezeIds.end(), contextGroupId) ==
            freezeIds.end())
      freezeIds.push_back(contextGroupId);

    const auto *node =
        contextNodeId != 0 ? graph.findNode(contextNodeId) : nullptr;
    const auto *group =
        contextGroupId != 0 ? graph.findGroup(contextGroupId) : nullptr;
    if (node != nullptr && node->state == NodeState::frozenGold) {
      if (ImGui::MenuItem("Unfreeze") && callbacks.unfreezeNode)
        callbacks.unfreezeNode(contextNodeId);
    } else {
      const auto expandedFreeze =
          graph.expandSelectionToFreezableLeaves(freezeIds);
      const auto canFreeze =
          !graph.partitionFreezeChains(expandedFreeze).empty();
      if (ImGui::MenuItem("Freeze Selection", nullptr, false, canFreeze) &&
          callbacks.freezeSelection)
        callbacks.freezeSelection(freezeIds);
      if (node != nullptr && !graph.isFixedIoNode(contextNodeId) &&
          ImGui::MenuItem("Delete")) {
        mutatedThisFrame = graph.removeNode(contextNodeId) || mutatedThisFrame;
        recompileThisFrame = mutatedThisFrame || recompileThisFrame;
        positionedNodeIds.erase(contextNodeId);
      }
    }

    std::vector<std::int32_t> groupMembers = selectedNodeIds;
    for (const auto id : selectedGroupIds) {
      if (id != contextGroupId)
        groupMembers.push_back(id);
    }
    if (contextNodeId != 0 &&
        std::find(groupMembers.begin(), groupMembers.end(), contextNodeId) ==
            groupMembers.end())
      groupMembers.push_back(contextNodeId);
    const auto canGroup = groupMembers.size() >= 2;
    if (ImGui::MenuItem("Group", nullptr, false, canGroup)) {
      const auto result = graph.createGroup(groupMembers);
      if (result.accepted) {
        mutatedThisFrame = true;
        positionedGroupIds.erase(result.groupId);
      } else if (!result.message.empty()) {
        transientMessage = result.message;
        transientMessageDeadline = ImGui::GetTime() + 2.5;
        if (callbacks.showMessage)
          callbacks.showMessage(result.message);
      }
    }
    if (group != nullptr && ImGui::MenuItem("Ungroup")) {
      const auto parent = group->parentGroupId;
      const auto focus = graph.getViewport().focusedGroupId;
      mutatedThisFrame = graph.ungroup(contextGroupId).accepted || mutatedThisFrame;
      positionedGroupIds.erase(contextGroupId);
      if (focus == contextGroupId)
        setCanvasFocus(graph, parent);
    }
    if (group != nullptr && ImGui::MenuItem("Delete")) {
      const auto parent = group->parentGroupId;
      const auto focus = graph.getViewport().focusedGroupId;
      bool leaveScope = false;
      if (focus.has_value()) {
        for (const auto ancestor : graph.groupAncestorChain(*focus)) {
          if (ancestor == contextGroupId) {
            leaveScope = true;
            break;
          }
        }
      }
      mutatedThisFrame =
          graph.deleteGroup(contextGroupId).accepted || mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
      positionedGroupIds.erase(contextGroupId);
      if (leaveScope)
        setCanvasFocus(graph, parent);
    }
    if (group != nullptr && ImGui::MenuItem("Open"))
      setCanvasFocus(graph, contextGroupId);
    if (node != nullptr && node->parentGroupId.has_value() &&
        ImGui::MenuItem("Remove from Group")) {
      mutatedThisFrame =
          graph.removeFromGroup(contextNodeId).accepted || mutatedThisFrame;
    }
    if (group != nullptr && group->parentGroupId.has_value() &&
        ImGui::MenuItem("Remove from Group")) {
      mutatedThisFrame =
          graph.removeFromGroup(contextGroupId).accepted || mutatedThisFrame;
    }
    const auto saveTarget =
        contextGroupId != 0 ? contextGroupId : contextNodeId;
    const auto selectedCount =
        selectedNodeIds.size() + selectedGroupIds.size();
    const bool saveMulti =
        selectedCount > 1 &&
        !((contextGroupId != 0 && selectedGroupIds.size() == 1 &&
           selectedNodeIds.empty() && selectedGroupIds.front() == contextGroupId) ||
          (contextNodeId != 0 && selectedNodeIds.size() == 1 &&
           selectedGroupIds.empty() && selectedNodeIds.front() == contextNodeId));
    const bool saveIo =
        node != nullptr && graph.isFixedIoNode(contextNodeId);
    if (saveTarget != 0 && activeBoxLibrary != nullptr) {
      if (saveMulti)
        ImGui::BeginDisabled();
      else if (saveIo)
        ImGui::BeginDisabled();
      if (ImGui::MenuItem("Save to Box Library...")) {
        pendingSaveBoxId = saveTarget;
        requestSaveBoxPopup = true;
        saveBoxOverwrite = false;
        const auto *saveNode = graph.findNode(saveTarget);
        const auto *saveGroup = graph.findGroup(saveTarget);
        const auto defaultName =
            saveGroup != nullptr
                ? juce::String(saveGroup->name)
                : (saveNode != nullptr ? juce::String(saveNode->label)
                                       : juce::String("Box"));
        const auto utf8 = defaultName.toRawUTF8();
        const auto length =
            std::min(saveBoxNameBuffer.size() - 1, std::strlen(utf8));
        std::memcpy(saveBoxNameBuffer.data(), utf8, length);
        saveBoxNameBuffer[length] = '\0';
      }
      if (saveMulti || saveIo)
        ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (saveMulti)
          ImGui::SetTooltip(
              "Save applies to one box. Group the selection first.");
        else if (saveIo)
          ImGui::SetTooltip(
              "Audio Input and Audio Output cannot be saved to the library.");
      }
    }
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopup("Link Context")) {
    if (ImGui::BeginMenu("Insert")) {
      const auto *link = graph.findLink(contextLinkId);
      for (const auto &item : paletteItems) {
        if (!canInsertOnLink(item.type))
          continue;
        if (ImGui::MenuItem(item.label)) {
          juce::Point<float> position{250.0f, 140.0f};
          if (link != nullptr) {
            const auto sourceNode = graph.findNodeForPin(link->sourcePinId);
            const auto destNode = graph.findNodeForPin(link->destinationPinId);
            const auto *source =
                sourceNode.has_value() ? graph.findNode(*sourceNode) : nullptr;
            const auto *destination =
                destNode.has_value() ? graph.findNode(*destNode) : nullptr;
            if (source != nullptr && destination != nullptr)
              position = {(source->position.x + destination->position.x) *
                              0.5f,
                          (source->position.y + destination->position.y) *
                              0.5f};
          }
          pendingLinkInsert.linkId = contextLinkId;
          pendingLinkInsert.type = item.type;
          pendingLinkInsert.position = position;
          pendingLinkInsert.pending = true;
          ImGui::CloseCurrentPopup();
        }
      }
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Delete")) {
      mutatedThisFrame =
          graph.removeLink(contextLinkId) || mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
    }
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopup("Pin Context")) {
    if (ImGui::BeginMenu("Add")) {
      const auto *pin = graph.findPin(contextPinId);
      const auto ownerId = graph.findNodeForPin(contextPinId);
      const auto *owner =
          ownerId.has_value() ? graph.findNode(*ownerId) : nullptr;
      for (const auto &item : paletteItems) {
        if (ImGui::MenuItem(item.label)) {
          juce::Point<float> position{lastCanvasCentre.x, lastCanvasCentre.y};
          if (!isCollapsedGroupPin(contextPinId) && owner != nullptr &&
              pin != nullptr) {
            const auto offset = pin->kind == PinKind::output ? 210.0f : -210.0f;
            position = {owner->position.x + offset, owner->position.y};
          }
          pendingPinAttach.pinId = contextPinId;
          pendingPinAttach.type = item.type;
          pendingPinAttach.position = position;
          pendingPinAttach.pending = true;
          ImGui::CloseCurrentPopup();
        }
      }
      ImGui::EndMenu();
    }
    ImGui::EndPopup();
  }

  handlePropertyCombo(graph, callbacks);
  ed::Resume();
}

void NodeRenderer::applyPendingContextActions(NodeGraph &graph) {
  if (pendingPinAttach.pending) {
    pendingPinAttach.pending = false;
    if (const auto nodeId = graph.attachNodeToPin(
            pendingPinAttach.pinId, pendingPinAttach.type,
            pendingPinAttach.position);
        nodeId.has_value()) {
      positionedNodeIds.erase(*nodeId);
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
  }

  if (pendingLinkInsert.pending) {
    pendingLinkInsert.pending = false;
    if (const auto nodeId = graph.insertNodeOnLink(
            pendingLinkInsert.linkId, pendingLinkInsert.type,
            pendingLinkInsert.position);
        nodeId.has_value()) {
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
  }
}

void NodeRenderer::handlePropertyCombo(NodeGraph &graph,
                                       const NodeRendererCallbacks &callbacks) {
  if (!activePropertyCombo.active)
    return;

  auto *node = graph.findNode(activePropertyCombo.nodeId);
  if (node == nullptr) {
    activePropertyCombo.active = false;
    activePropertyCombo.requestOpen = false;
    return;
  }

  NodeProperty *property = nullptr;
  for (auto &candidate : node->properties) {
    if (candidate.key == activePropertyCombo.propertyKey &&
        candidate.kind == PropertyKind::choice) {
      property = &candidate;
      break;
    }
  }
  if (property == nullptr || property->choices.empty()) {
    activePropertyCombo.active = false;
    activePropertyCombo.requestOpen = false;
    return;
  }

  constexpr const char *popupId = "OpenYourBoxPropertyCombo";
  if (activePropertyCombo.requestOpen) {
    ImGui::SetNextWindowPos(activePropertyCombo.anchorMin, ImGuiCond_Appearing,
                            ImVec2(0.0f, 1.0f));
    ImGui::OpenPopup(popupId);
    activePropertyCombo.requestOpen = false;
  }

  bool popupVisible = false;
  if (ImGui::BeginPopup(popupId)) {
    popupVisible = true;
    ImGui::PushItemWidth(76.0f);
    const auto rowHeight = ImGui::GetTextLineHeightWithSpacing();
    const auto listHeight = std::min(
        rowHeight * static_cast<float>(property->choices.size()) + 8.0f,
        220.0f);
    ImGui::BeginChild("PropertyComboList", ImVec2(76.0f, listHeight), false);
    for (int index = 0; index < static_cast<int>(property->choices.size());
         ++index) {
      ImGui::PushID(index);
      const auto selected = property->value == index;
      if (ImGui::Selectable(
              property->choices[static_cast<std::size_t>(index)].c_str(),
              selected)) {
        const auto previous = property->value;
        property->setValue(index);
        if (property->value != previous) {
          if (!graph.setProperty(node->id, property->key, property->value)) {
            property->setValue(previous);
            transientMessage =
                "That mode would break downstream channel compatibility";
            transientMessageDeadline = ImGui::GetTime() + 2.5;
            if (callbacks.showMessage)
              callbacks.showMessage(transientMessage);
          } else {
            mutatedThisFrame = true;
            recompileThisFrame = true;
            if (callbacks.propertyChanged)
              callbacks.propertyChanged(node->id, property->key,
                                        property->value);
          }
        }
        ImGui::CloseCurrentPopup();
        activePropertyCombo.active = false;
      }
      ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::EndPopup();
  }

  if (!popupVisible && !ImGui::IsPopupOpen(popupId))
    activePropertyCombo.active = false;
}

void NodeRenderer::renderMap(NodeGraph &graph, ImVec2 canvasOrigin,
                             ImVec2 canvasSize) {
  ed::Suspend();
  ImGui::SetNextWindowPos(
      ImVec2(canvasOrigin.x + canvasSize.x - mapWidth - 12.0f,
             canvasOrigin.y + canvasSize.y - mapHeight - 12.0f));
  ImGui::SetNextWindowSize(ImVec2(mapWidth, mapHeight));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
  ImGui::Begin("GraphMap", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoFocusOnAppearing |
                   ImGuiWindowFlags_NoNav);

  const auto origin = ImGui::GetCursorScreenPos();
  const auto available = ImGui::GetContentRegionAvail();
  ImGui::InvisibleButton("GraphMapHit", available);
  const auto hovered = ImGui::IsItemHovered();
  const auto active = ImGui::IsItemActive();
  mapHoveredLastFrame = hovered || active;

  auto minimum = ImVec2(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
  auto maximum = ImVec2(std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest());
  const auto viewportTopLeft = ed::ScreenToCanvas(canvasOrigin);
  const auto viewportBottomRight = ed::ScreenToCanvas(
      ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y));
  minimum.x = std::min(minimum.x, viewportTopLeft.x);
  minimum.y = std::min(minimum.y, viewportTopLeft.y);
  maximum.x = std::max(maximum.x, viewportBottomRight.x);
  maximum.y = std::max(maximum.y, viewportBottomRight.y);

  struct MapNode {
    std::int32_t id = 0;
    ImVec2 position;
    ImVec2 size;
    juce::Colour colour;
    bool isGroup = false;
  };
  const auto focusedGroupId = graph.getViewport().focusedGroupId;
  std::vector<MapNode> mapNodes;
  mapNodes.reserve(graph.getNodes().size() + graph.getGroups().size());
  const auto pushBox = [&](std::int32_t id, ImVec2 stored, ImVec2 storedSize,
                           juce::Colour colour, bool isGroup) {
    auto position = stored;
    auto size = storedSize;
    if (isGroup ? positionedGroupIds.count(id) != 0
                : positionedNodeIds.count(id) != 0) {
      position = ed::GetNodePosition(ed::NodeId(editorIdentifier(id)));
      const auto editorSize = ed::GetNodeSize(ed::NodeId(editorIdentifier(id)));
      if (editorSize.x > 0.0f && editorSize.y > 0.0f)
        size = editorSize;
    }
    mapNodes.push_back({id, position, size, colour, isGroup});
    minimum.x = std::min(minimum.x, position.x);
    minimum.y = std::min(minimum.y, position.y);
    maximum.x = std::max(maximum.x, position.x + size.x);
    maximum.y = std::max(maximum.y, position.y + size.y);
  };
  for (const auto &node : graph.getNodes()) {
    if (!graph.isNodeOnFocusedCanvas(node.id, focusedGroupId))
      continue;
    pushBox(node.id, ImVec2(node.position.x, node.position.y),
            ImVec2(std::max(8.0f, node.size.x), std::max(8.0f, node.size.y)),
            node.colour, false);
  }
  for (const auto &group : graph.getGroups()) {
    if (!graph.isGroupOnFocusedCanvas(group.id, focusedGroupId))
      continue;
    pushBox(group.id, ImVec2(group.position.x, group.position.y),
            ImVec2(std::max(8.0f, group.size.x), std::max(8.0f, group.size.y)),
            groupFrameColour, true);
  }

  const auto extent = ImVec2(std::max(1.0f, maximum.x - minimum.x),
                             std::max(1.0f, maximum.y - minimum.y));
  const auto scale = std::min((available.x - 10.0f) / extent.x,
                              (available.y - 10.0f) / extent.y);
  auto project = [origin, minimum, scale](ImVec2 canvasPoint) {
    return ImVec2(origin.x + 5.0f + (canvasPoint.x - minimum.x) * scale,
                  origin.y + 5.0f + (canvasPoint.y - minimum.y) * scale);
  };

  auto *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin,
                      ImVec2(origin.x + available.x, origin.y + available.y),
                      ImColor(18, 22, 28, 230), 4.0f);

  std::unordered_map<std::int32_t, MapNode> nodesById;
  nodesById.reserve(mapNodes.size());
  for (const auto &node : mapNodes)
    nodesById.emplace(node.id, node);

  const auto linkEndpoint = [](const MapNode &node, bool isOutput) {
    const auto centerY = node.position.y + node.size.y * 0.5f;
    return isOutput ? ImVec2(node.position.x + node.size.x, centerY)
                    : ImVec2(node.position.x, centerY);
  };

  for (const auto &link : graph.getLinks()) {
    const auto sourceNodeId = graph.findNodeForPin(link.sourcePinId);
    const auto destinationNodeId = graph.findNodeForPin(link.destinationPinId);
    if (!sourceNodeId.has_value() || !destinationNodeId.has_value())
      continue;
    if (!nodeRepresentedOnCanvas(graph, *sourceNodeId, focusedGroupId) ||
        !nodeRepresentedOnCanvas(graph, *destinationNodeId, focusedGroupId))
      continue;
    const auto sourceHost =
        graph.focusedCanvasHostGroup(*sourceNodeId, focusedGroupId);
    const auto destHost =
        graph.focusedCanvasHostGroup(*destinationNodeId, focusedGroupId);
    if (sourceHost.has_value() && sourceHost == destHost)
      continue;
    const auto sourceId = sourceHost.value_or(*sourceNodeId);
    const auto destId = destHost.value_or(*destinationNodeId);
    const auto source = nodesById.find(sourceId);
    const auto destination = nodesById.find(destId);
    if (source == nodesById.end() || destination == nodesById.end())
      continue;
    const auto from = project(linkEndpoint(source->second, true));
    const auto to = project(linkEndpoint(destination->second, false));
    draw->AddLine(from, to, ImColor(100, 180, 255, 170), 1.0f);
  }

  for (const auto &node : mapNodes) {
    const auto topLeft = project(node.position);
    const auto bottomRight =
        ImVec2(topLeft.x + std::max(4.0f, node.size.x * scale),
               topLeft.y + std::max(3.0f, node.size.y * scale));
    draw->AddRectFilled(topLeft, bottomRight,
                        ImColor(node.colour.getRed(), node.colour.getGreen(),
                                node.colour.getBlue(), 210),
                        2.0f);
  }
  draw->AddRect(project(viewportTopLeft), project(viewportBottomRight),
                ImColor(255, 255, 255, 220), 0.0f, 0, 1.5f);

  if (active && scale > 0.0f) {
    const auto mouse = ImGui::GetMousePos();
    centreViewOnCanvas(
        ImVec2(minimum.x + (mouse.x - origin.x - 5.0f) / scale,
               minimum.y + (mouse.y - origin.y - 5.0f) / scale));
  }

  ImGui::End();
  ImGui::PopStyleVar();
  ed::Resume();
}

void NodeRenderer::synchronizeSelection(NodeGraph &graph) {
  const auto objectCount = ed::GetSelectedObjectCount();
  std::vector<ed::NodeId> selectedNodes(static_cast<std::size_t>(objectCount));
  const auto selectedCount =
      ed::GetSelectedNodes(selectedNodes.data(), objectCount);
  selectedNodeIds.clear();
  selectedGroupIds.clear();
  for (int index = 0; index < selectedCount; ++index) {
    const auto id = static_cast<std::int32_t>(
        selectedNodes[static_cast<std::size_t>(index)].Get());
    if (graph.findGroup(id) != nullptr)
      selectedGroupIds.push_back(id);
    else if (graph.findNode(id) != nullptr)
      selectedNodeIds.push_back(id);
  }

  std::vector<ed::LinkId> selectedLinks(static_cast<std::size_t>(objectCount));
  const auto selectedLinkCount =
      ed::GetSelectedLinks(selectedLinks.data(), objectCount);
  selectedLinkIds.clear();
  selectedLinkIds.reserve(static_cast<std::size_t>(selectedLinkCount));
  for (int index = 0; index < selectedLinkCount; ++index)
    selectedLinkIds.push_back(static_cast<std::int32_t>(
        selectedLinks[static_cast<std::size_t>(index)].Get()));
}

void NodeRenderer::navigateCanvas(ImVec2 panDelta, float zoomFactor,
                                  ImVec2 pivotScreen) {
  auto *editor = detailEditor();
  if (editor == nullptr)
    return;
  auto view = editor->GetViewRect();
  const auto currentScale =
      std::clamp(editor->GetView().Scale, minimumZoom, maximumZoom);
  auto appliedScale = currentScale;
  if (std::abs(zoomFactor - 1.0f) > 0.0001f) {
    const auto pivot = ed::ScreenToCanvas(pivotScreen);
    const auto targetScale =
        std::clamp(currentScale * zoomFactor, minimumZoom, maximumZoom);
    const auto applied =
        currentScale > 0.0f ? targetScale / currentScale : 1.0f;
    view.Min.x = pivot.x + (view.Min.x - pivot.x) / applied;
    view.Min.y = pivot.y + (view.Min.y - pivot.y) / applied;
    view.Max.x = pivot.x + (view.Max.x - pivot.x) / applied;
    view.Max.y = pivot.y + (view.Max.y - pivot.y) / applied;
    appliedScale = targetScale;
  }
  const auto invScale = appliedScale > 0.0f ? 1.0f / appliedScale : 1.0f;
  view.Translate(ImVec2(panDelta.x * invScale, panDelta.y * invScale));
  commitCanvasView(view);
  layoutMutatedThisFrame = true;
}

void NodeRenderer::syncEditorTransforms(NodeGraph &graph) {
  const auto focusedGroupId = graph.getViewport().focusedGroupId;
  for (auto &group : graph.getGroups()) {
    if (!graph.isGroupOnFocusedCanvas(group.id, focusedGroupId))
      continue;
    positionedGroupIds.insert(group.id);
    if (group.id != draggingGroupId) {
      ed::SetNodePosition(ed::NodeId(editorIdentifier(group.id)),
                          ImVec2(group.position.x, group.position.y));
    }
  }
  for (auto &node : graph.getNodes()) {
    if (!graph.isNodeOnFocusedCanvas(node.id, focusedGroupId) ||
        node.id == draggingNodeId)
      continue;
    positionedNodeIds.insert(node.id);
    ed::SetNodePosition(ed::NodeId(editorIdentifier(node.id)),
                        ImVec2(node.position.x, node.position.y));
  }
}

void NodeRenderer::renderScopeBreadcrumb(NodeGraph &graph) {
  const auto focused = graph.getViewport().focusedGroupId;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 4.0f));
  const auto current = focused.has_value();
  if (!current) {
    ImGui::TextColored(ImVec4(0.75f, 0.88f, 1.0f, 1.0f), "Graph");
  } else if (ImGui::SmallButton("Graph")) {
    setCanvasFocus(graph, std::nullopt);
  }
  if (focused.has_value()) {
    for (const auto id : graph.groupAncestorChain(*focused)) {
      const auto *group = graph.findGroup(id);
      if (group == nullptr)
        continue;
      ImGui::SameLine();
      ImGui::TextDisabled(">");
      ImGui::SameLine();
      ImGui::PushID(id);
      if (id == *focused) {
        ImGui::TextColored(ImVec4(0.75f, 0.88f, 1.0f, 1.0f), "%s",
                           group->name.c_str());
      } else if (ImGui::SmallButton(group->name.c_str())) {
        setCanvasFocus(graph, id);
      }
      ImGui::PopID();
    }
  }
  ImGui::PopStyleVar();
}

void NodeRenderer::setCanvasFocus(NodeGraph &graph,
                                  std::optional<std::int32_t> groupId) {
  if (groupId.has_value() && graph.findGroup(*groupId) == nullptr)
    groupId.reset();
  auto &viewport = graph.getViewport();
  if (viewport.focusedGroupId == groupId)
    return;
  viewport.focusedGroupId = groupId;
  restoreViewPending = true;
  layoutMutatedThisFrame = true;
  mutatedThisFrame = true;
}

void NodeRenderer::adoptNewBox(NodeGraph &graph, std::int32_t boxId,
                               ImVec2 canvasPosition, std::int32_t dropGroupId) {
  const auto focused = graph.getViewport().focusedGroupId;
  auto targetParent = focused;
  ImVec2 localPosition = canvasPosition;
  if (dropGroupId != 0 && dropGroupId != boxId &&
      graph.isGroupOnFocusedCanvas(dropGroupId, focused)) {
    targetParent = dropGroupId;
    if (const auto *target = graph.findGroup(dropGroupId)) {
      localPosition = ImVec2(canvasPosition.x - target->position.x,
                             canvasPosition.y - target->position.y);
    }
  }
  if (!targetParent.has_value())
    return;
  if (graph.findNode(boxId) != nullptr)
    graph.moveNode(boxId, {localPosition.x, localPosition.y});
  else if (graph.findGroup(boxId) != nullptr)
    graph.moveGroup(boxId, {localPosition.x, localPosition.y});
  const auto result = graph.addToGroup(*targetParent, boxId, true);
  if (!result.accepted && !result.message.empty()) {
    transientMessage = result.message;
    transientMessageDeadline = ImGui::GetTime() + 2.5;
  }
}

void NodeRenderer::centreViewOnCanvas(ImVec2 canvasPoint) {
  auto *editor = detailEditor();
  if (editor == nullptr)
    return;
  const auto view = editor->GetViewRect();
  const auto size = view.GetSize();
  ImRect focused;
  focused.Min =
      ImVec2(canvasPoint.x - size.x * 0.5f, canvasPoint.y - size.y * 0.5f);
  focused.Max =
      ImVec2(canvasPoint.x + size.x * 0.5f, canvasPoint.y + size.y * 0.5f);
  commitCanvasView(focused);
  layoutMutatedThisFrame = true;
}
} // namespace openyourbox::graph
