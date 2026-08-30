#include "NodeRenderer.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_node_editor_internal.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
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
  /** @brief True when this item places a TorchScript Load Gold node. */
  bool externalLoad = false;
};

struct PaletteCategory {
  const char *label;
  const PaletteItem *items;
  std::size_t count;
};

constexpr PaletteItem effectsPaletteItems[] = {
    {"ExpDecayReverb", openyourbox::graph::NodeType::expDecayReverb},
    {"FilteredNoiseReverb", openyourbox::graph::NodeType::filteredNoiseReverb},
    {"FIRFilter", openyourbox::graph::NodeType::firFilter},
    {"ModDelay", openyourbox::graph::NodeType::modDelay},
    {"Reverb", openyourbox::graph::NodeType::reverb},
};

constexpr PaletteItem neuralPaletteItems[] = {
    {"LSTM", openyourbox::graph::NodeType::lstm},
    {"RNN", openyourbox::graph::NodeType::rnn},
    {"TorchScript Load", openyourbox::graph::NodeType::blackBox, true},
};

constexpr PaletteItem layerPaletteItems[] = {
    {"Activation", openyourbox::graph::NodeType::activation},
    {"BatchNorm1d", openyourbox::graph::NodeType::batchNorm},
    {"Bottleneck", openyourbox::graph::NodeType::variationalBottleneck},
    {"Conv1D", openyourbox::graph::NodeType::convolution},
    {"ConvTranspose1d", openyourbox::graph::NodeType::convTranspose},
    {"Linear", openyourbox::graph::NodeType::linear},
    {"Math Expression", openyourbox::graph::NodeType::mathExpression},
    {"Noise Synth", openyourbox::graph::NodeType::noiseSynthesizer},
    {"PQMF Analysis", openyourbox::graph::NodeType::pqmfAnalysis},
    {"PQMF Synthesis", openyourbox::graph::NodeType::pqmfSynthesis},
    {"TCN", openyourbox::graph::NodeType::tcn},
    {"Utility", openyourbox::graph::NodeType::merge},
};

constexpr PaletteItem sourcePaletteItems[] = {
    {"Knob Input", openyourbox::graph::NodeType::knobInput},
    {"XY Trackpad", openyourbox::graph::NodeType::xyTrackpad},
};

constexpr PaletteCategory paletteCategories[] = {
    {"Effects", effectsPaletteItems, std::size(effectsPaletteItems)},
    {"Neural / Sequence", neuralPaletteItems, std::size(neuralPaletteItems)},
    {"Layers", layerPaletteItems, std::size(layerPaletteItems)},
    {"Sources", sourcePaletteItems, std::size(sourcePaletteItems)},
};

/**
 * @brief Invokes @p visitor for every Factory palette item.
 * @tparam Visitor Callable `(const PaletteItem &)`.
 */
template <typename Visitor> void forEachPaletteItem(Visitor &&visitor) {
  for (const auto &category : paletteCategories) {
    for (std::size_t index = 0; index < category.count; ++index)
      visitor(category.items[index]);
  }
}

constexpr float nodeBodyWidth = 188.0f;
/** @brief Width reserved for right-aligned property value controls. */
constexpr float propertyValueWidth = 76.0f;
/** @brief Body width used by property-row helpers (panel or slim node). */
float propertyLayoutWidth = nodeBodyWidth;
/** @brief Value-column width used by property-row helpers. */
float propertyFieldWidth = propertyValueWidth;
/** @brief Drag-drop payload id for Project structure live rows. */
constexpr const char *structureBoxPayloadId = "OPENYOURBOX_STRUCTURE_BOX";
/** @brief Multiplier applied to canvas pan gestures. */
constexpr float canvasPanSpeed = 2.0f;
/** @brief Base scroll-wheel pan distance in screen pixels. */
constexpr float canvasWheelPanStep = 48.0f;
/** @brief Pixels per mouse-wheel tick when scrolling the hierarchy trail. */
constexpr float breadcrumbWheelStep = 48.0f;
/** @brief Width of the fade drawn when the hierarchy trail is cropped. */
constexpr float breadcrumbCropFadeWidth = 28.0f;
/** @brief Middle-mouse button index used for canvas drag panning. */
constexpr int canvasDragPanButton = 2;
/** @brief Time window for canvas and Project structure double-clicks. */
constexpr double boxDoubleClickSeconds = 0.50;
/** @brief Max pointer travel that still counts as a double-click. */
constexpr float boxDoubleClickMaxPixels = 10.0f;
/** @brief Pixels of movement before a canvas press becomes a box move. */
constexpr float boxDragStartPixels = 3.0f;
/** @brief Canvas-space padding when framing every box on a group canvas. */
constexpr float canvasFitPadding = 48.0f;
/** @brief Display name of the root canvas in Project structure and the trail. */
constexpr const char *rootCanvasLabel = "Main";
/** @brief Sentinel id for the Project structure Main row double-click. */
constexpr std::int32_t mainStructureRowId = -1;
/** @brief Idle cable colour before RMS fill is applied. */
constexpr ImU32 idleLinkColour = IM_COL32(100, 180, 255, 220);
/** @brief RMS fill colour drawn from source toward destination. */
constexpr ImU32 levelLinkColour = IM_COL32(255, 176, 64, 255);
/** @brief Thickness of the idle cable in canvas units. */
constexpr float idleLinkThickness = 2.0f;
/** @brief Thickness of the RMS fill overlay in canvas units. */
constexpr float levelLinkThickness = 3.0f;
/** @brief Floor of the cable meter in dBFS; below this the fill is empty. */
constexpr float rmsMeterFloorDb = -60.0f;
/** @brief Smoothing time constant for cable fill animation, in seconds. */
constexpr float linkFillSmoothSeconds = 0.08f;
/**
 * @brief imgui-node-editor `c_LinkChannel_Flow` (above cables, below nodes).
 * @see imgui_node_editor.cpp channel layout
 */
constexpr int imguiLinkFlowChannel = 8;
/** @brief imgui-node-editor `c_UserChannel_Content` restored after fill draw. */
constexpr int imguiUserContentChannel = 1;

/**
 * @brief Draws node/group chrome that must not steal canvas clicks or drags.
 * @param text Label contents.
 * @param disabledLook True to use the disabled text colour.
 */
void drawPassiveBoxText(const char *text, bool disabledLook = false) {
  ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  if (disabledLook)
    ImGui::TextDisabled("%s", text);
  else
    ImGui::TextUnformatted(text);
  ImGui::PopItemFlag();
}

/**
 * @brief Reserves slim-box width without creating a click-stealing hit target.
 * @param width Reserved layout width in pixels.
 */
void reservePassiveBoxWidth(float width) {
  ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  ImGui::Dummy(ImVec2(width, 0.0f));
  ImGui::PopItemFlag();
}

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
 * @brief Boxes removed by context-menu Delete, expanding to the selection.
 *
 * When the right-clicked box is selected, every selected node and group is
 * included so a multi-selection is one cut. Host I/O and group hubs stay
 * out of the list.
 * @param graph Graph owning undeletable checks.
 * @param contextId Right-clicked node or group identifier.
 * @param selectedNodeIds Current node selection.
 * @param selectedGroupIds Current group selection.
 */
std::vector<std::int32_t> contextDeleteBoxIds(
    const openyourbox::graph::NodeGraph &graph, std::int32_t contextId,
    const std::vector<std::int32_t> &selectedNodeIds,
    const std::vector<std::int32_t> &selectedGroupIds) {
  const auto selected =
      std::find(selectedNodeIds.begin(), selectedNodeIds.end(), contextId) !=
          selectedNodeIds.end() ||
      std::find(selectedGroupIds.begin(), selectedGroupIds.end(), contextId) !=
          selectedGroupIds.end();
  std::vector<std::int32_t> ids;
  if (selected) {
    for (const auto id : selectedNodeIds) {
      if (!graph.isFixedIoNode(id) && !graph.isGroupBoundaryNode(id))
        ids.push_back(id);
    }
    ids.insert(ids.end(), selectedGroupIds.begin(), selectedGroupIds.end());
  } else if (contextId != 0 && !graph.isFixedIoNode(contextId) &&
             !graph.isGroupBoundaryNode(contextId)) {
    ids.push_back(contextId);
  }
  return ids;
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
 * @brief Maps a linear RMS amplitude to a `[0, 1]` cable-fill fraction.
 * @param rms Collapsed linear RMS, typically in `[0, 1]`.
 * @return Fill amount from the source pin toward the destination.
 */
float rmsToLinkFill(float rms) noexcept {
  if (!(rms > 1.0e-8f))
    return 0.0f;
  const float db = 20.0f * std::log10(std::min(rms, 1.0f));
  return std::clamp((db - rmsMeterFloorDb) / -rmsMeterFloorDb, 0.0f, 1.0f);
}

/**
 * @brief Strokes the source-to-destination portion of a live editor link.
 * @param linkId Stable graph link identifier.
 * @param fill Fraction of the cubic in `[0, 1]`.
 */
void drawDirectedLinkFill(std::int32_t linkId, float fill) {
  auto *editor = detailEditor();
  if (editor == nullptr || fill <= 0.001f)
    return;
  auto *link = editor->FindLink(ed::LinkId(editorIdentifier(linkId)));
  if (link == nullptr || !link->m_IsLive)
    return;
  auto *drawList = editor->GetDrawList();
  if (drawList == nullptr)
    return;
  drawList->ChannelsSetCurrent(imguiLinkFlowChannel);
  const auto curve = link->GetCurve();
  const auto t = std::clamp(fill, 0.0f, 1.0f);
  if (t >= 0.999f) {
    drawList->AddBezierCubic(curve.P0, curve.P1, curve.P2, curve.P3,
                             levelLinkColour, levelLinkThickness);
    return;
  }
  const auto split = ImCubicBezierSplit(curve, t);
  drawList->AddBezierCubic(split.Left.P0, split.Left.P1, split.Left.P2,
                           split.Left.P3, levelLinkColour, levelLinkThickness);
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
 * @brief Reads an integer node property or a fallback when missing.
 * @param node Graph node.
 * @param key Property key.
 * @param fallback Value when the key is absent.
 * @return Property value or fallback.
 */
int readIntProperty(const openyourbox::graph::GraphNode &node, const char *key,
                    int fallback) {
  for (const auto &property : node.properties) {
    if (property.key == key)
      return property.value;
  }
  return fallback;
}

/**
 * @brief Returns the causal delay of a rate-changing node in samples.
 * @param node Graph node.
 * @return Delay at host rate, or 0 when the type does not add history.
 */
std::uint64_t nodeDelaySamples(const openyourbox::graph::GraphNode &node) {
  using openyourbox::graph::NodeType;
  using openyourbox::graph::defaultBottleneckKernelSize;
  using openyourbox::graph::defaultPqmfBands;
  switch (node.type) {
  case NodeType::pqmfAnalysis:
  case NodeType::pqmfSynthesis: {
    const auto nBand =
        std::max(2, readIntProperty(node, "n_band", defaultPqmfBands));
    return static_cast<std::uint64_t>(4 * nBand);
  }
  case NodeType::convolution:
  case NodeType::rateConv:
  case NodeType::convTranspose: {
    const auto kernel = std::max(1, readIntProperty(node, "kernel_size", 3));
    const auto dilation = std::max(1, readIntProperty(node, "dilation", 1));
    const auto stride = std::max(1, readIntProperty(node, "stride", 1));
    return static_cast<std::uint64_t>(kernel - 1) *
           static_cast<std::uint64_t>(dilation) *
           static_cast<std::uint64_t>(stride);
  }
  case NodeType::variationalBottleneck: {
    const auto kernel = std::max(
        1, readIntProperty(node, "kernel_size", defaultBottleneckKernelSize));
    return static_cast<std::uint64_t>(kernel - 1);
  }
  default:
    return 0;
  }
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
   *
   * The click frame and sub-step pointer jitter are ignored so a press-and-hold
   * does not emit a value change.
   * @param id ImGui identifier for the handle.
   * @param size Handle size in pixels.
   * @return Signed steps to apply; zero when the handle is idle or still held.
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
  const auto accumId = ImGui::GetItemID();
  const auto phaseId = ImGui::GetID("##dragPhase");
  if (!active) {
    storage->SetFloat(accumId, 0.0f);
    storage->SetInt(phaseId, 0);
    return 0;
  }

  const auto pixelsPerStep = ImGui::GetIO().KeyShift ? 3.0f : 8.0f;
  auto phase = storage->GetInt(phaseId);
  if (phase == 0) {
    storage->SetInt(phaseId, 1);
    storage->SetFloat(accumId, 0.0f);
    return 0;
  }

  auto accum = storage->GetFloat(accumId) + ImGui::GetIO().MouseDelta.x;
  if (phase == 1) {
    if (std::abs(accum) < pixelsPerStep) {
      storage->SetFloat(accumId, accum);
      return 0;
    }
    storage->SetInt(phaseId, 2);
  }
  const auto steps = static_cast<int>(accum / pixelsPerStep);
  accum -= static_cast<float>(steps) * pixelsPerStep;
  storage->SetFloat(accumId, accum);
  return steps;
}

/** @brief Info-box text queued until after pin/node builders finish. */
std::string pendingInfoTooltip;

/**
 * @brief Opens the circled-i tooltip in screen space.
 * @param text Tooltip body.
 */
void showInfoTooltip(const std::string &text) {
  ImGui::BeginTooltip();
  ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
  ImGui::TextUnformatted(text.c_str());
  ImGui::PopTextWrapPos();
  ImGui::EndTooltip();
}

/**
 * @brief Draws a circled i; queues the info box while the pointer is over it.
 *
 * Tooltips cannot be opened inside @c BeginPin / @c BeginNode: those builders
 * own the editor draw-list splitter, and @c ed::Suspend() would swap it out
 * from under them. Outside the canvas editor (Parameters tab) the tooltip is
 * shown immediately.
 * @param id Unique ImGui id for the hover target.
 * @param text Tooltip body shown inside the info box.
 */
void drawInfoHover(const char *id, const std::string &text) {
  if (text.empty())
    return;
  const float height = ImGui::GetTextLineHeight();
  const auto origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton(id, ImVec2(height, height));
  const bool hovered = ImGui::IsItemHovered();
  auto *draw = ImGui::GetWindowDrawList();
  const ImVec2 centre(origin.x + height * 0.5f, origin.y + height * 0.5f);
  const float radius = height * 0.40f;
  const auto colour =
      ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
  draw->AddCircle(centre, radius, colour, 12, 1.15f);
  const auto letterSize = ImGui::CalcTextSize("i");
  draw->AddText(ImVec2(centre.x - letterSize.x * 0.5f, origin.y), colour, "i");
  if (!hovered)
    return;
  if (ed::GetCurrentEditor() != nullptr)
    pendingInfoTooltip = text;
  else
    showInfoTooltip(text);
}

/**
 * @brief Shows the queued info box in screen space after nodes have closed.
 */
void flushPendingInfoTooltip() {
  if (pendingInfoTooltip.empty())
    return;
  if (ed::GetCurrentEditor() != nullptr)
    ed::Suspend();
  showInfoTooltip(pendingInfoTooltip);
  if (ed::GetCurrentEditor() != nullptr)
    ed::Resume();
  pendingInfoTooltip.clear();
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
      ImVec2(rowOrigin.x + propertyLayoutWidth - propertyFieldWidth, rowOrigin.y));
  ImGui::SetNextItemWidth(propertyFieldWidth);
}

/**
 * @brief Draws a property label, optional expanded-value info icon, and a
 *        full-width value row beneath.
 * @param label Property label text.
 * @param expandedInfo Hierarchical expanded-repeat preview, or empty.
 */
void beginRepeatListPropertyRow(const char *label,
                              const std::string &expandedInfo) {
  ImGui::TextUnformatted(label);
  if (!expandedInfo.empty()) {
    ImGui::SameLine(0.0f, 4.0f);
    drawInfoHover("##expandedInfo", expandedInfo);
  }
  ImGui::SetNextItemWidth(propertyLayoutWidth);
}

/**
 * @brief Draws a pin caption, hiding the shape behind a hover info icon.
 *
 * The caption never inlines the shape, including for a single repeat and for
 * pins that are not inside a group. The info icon is omitted only when no
 * concrete shape is known yet.
 * @param kind Input or output direction (affects arrows).
 * @param label Pin name.
 * @param inlineShape Shape shown in the info tooltip when there is no repeat list.
 * @param expandedShapes Nested repeat-shape preview, or empty to use @p inlineShape.
 */
void drawPinCaption(openyourbox::graph::PinKind kind, const char *label,
                    const openyourbox::graph::ShapeSignature &inlineShape,
                    const std::string &expandedShapes) {
  const bool isInput = kind == openyourbox::graph::PinKind::input;
  auto shapeText = expandedShapes;
  if (shapeText.empty())
    shapeText = inlineShape.displayLabel();

  if (isInput)
    ImGui::Text("<- %s", label);
  else
    ImGui::TextUnformatted(label);
  if (!shapeText.empty()) {
    ImGui::SameLine(0.0f, 4.0f);
    drawInfoHover("##shapeInfo", shapeText);
  }
  if (!isInput) {
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::TextUnformatted("->");
  }
}

/**
 * @brief Shape tooltip text for @p pin, including a single repeat or no group.
 * @param pin Endpoint whose @c repeatShapes may be populated.
 * @param repeatCounts Outer→inner counts used to nest a multi-repeat list.
 */
std::string pinExpandedShapeInfo(const openyourbox::graph::Pin &pin,
                                 const std::vector<int> &repeatCounts) {
  return openyourbox::graph::formatShapeRepeatList(pin.repeatShapes, pin.shape,
                                                 repeatCounts);
}

/**
 * @brief Returns true when a palette type can be inserted onto an existing cable.
 * @param type Palette element type.
 * @return False for source-only Knob/XY elements.
 */
bool canInsertOnLink(openyourbox::graph::NodeType type) noexcept {
  return !openyourbox::graph::isConditioningSourceType(type) &&
         type != openyourbox::graph::NodeType::blackBox;
}

/**
 * @brief Places a Factory palette item onto the focused canvas or group.
 * @param graph Editable graph document.
 * @param item Palette payload.
 * @param position Destination canvas coordinates.
 * @param parentGroupId Destination group, or empty for the root.
 * @return Stable identifier of the new node.
 */
std::int32_t placePaletteItem(openyourbox::graph::NodeGraph &graph,
                              const PaletteItem &item,
                              juce::Point<float> position,
                              std::optional<std::int32_t> parentGroupId) {
  if (item.externalLoad)
    return graph.addExternalTorchScriptLoadNode(position, parentGroupId);
  return graph.addNode(item.type, position, parentGroupId);
}

/**
 * @brief Draws a fade on cropped edges of the hierarchy trail.
 * @param min Screen-space top-left of the trail viewport.
 * @param max Screen-space bottom-right of the trail viewport.
 * @param croppedLeft True when content is clipped on the left.
 * @param croppedRight True when content is clipped on the right.
 */
void drawBreadcrumbCropFades(ImVec2 min, ImVec2 max, bool croppedLeft,
                             bool croppedRight) {
  if (!croppedLeft && !croppedRight)
    return;
  auto *draw = ImGui::GetWindowDrawList();
  const ImU32 opaque = ImGui::GetColorU32(ImGuiCol_WindowBg);
  const ImU32 clear = opaque & ~IM_COL32_A_MASK;
  const float fade = breadcrumbCropFadeWidth;
  if (croppedLeft)
    draw->AddRectFilledMultiColor(min, ImVec2(min.x + fade, max.y), opaque,
                                  clear, clear, opaque);
  if (croppedRight)
    draw->AddRectFilledMultiColor(ImVec2(max.x - fade, min.y), max, clear,
                                  opaque, opaque, clear);
}
} // namespace

namespace openyourbox::graph {
NodeRenderer::NodeRenderer() = default;

NodeRenderer::~NodeRenderer() { ed::DestroyEditor(context); }

void NodeRenderer::render(NodeGraph &graph,
                          const NodeRendererCallbacks &callbacks,
                          float pinchMagnification,
                          openyourbox::library::UserBoxLibrary *boxLibrary,
                          const std::unordered_map<std::int32_t, float>
                              *outputRmsByNodeId) {
  mutatedThisFrame = false;
  layoutMutatedThisFrame = false;
  recompileThisFrame = false;
  patchGestureHeldThisFrame = false;
  activeBoxLibrary = boxLibrary;
  activeCallbacks = &callbacks;
  structureDropHighlightActive = false;
  structureDropValid = false;
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
  applyPendingEditorSelection(graph);
  lastCanvasCentre = ed::ScreenToCanvas(
      ImVec2(canvasOrigin.x + canvasSize.x * 0.5f,
             canvasOrigin.y + canvasSize.y * 0.5f));

  if (restoreViewPending && !pendingFitCanvas) {
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
  if (pendingFitCanvas) {
    fitCanvasToContents(graph, canvasSize);
    pendingFitCanvas = false;
    restoreViewPending = false;
  }
  if (pendingCentreView) {
    centreViewOnCanvas(pendingCentrePoint);
    pendingCentreView = false;
  }

  dropTargetGroupId = 0;
  const auto hoveredNode = ed::GetHoveredNode();
  const auto boxBodyHovered =
      hoveredNode && !ed::GetHoveredPin() && !ed::GetHoveredLink() && !overMap;
  std::int32_t hoveredGroupOnCanvas = 0;
  if (hoveredNode) {
    const auto hoveredId = static_cast<std::int32_t>(hoveredNode.Get());
    if (graph.isGroupOnFocusedCanvas(hoveredId, focusedGroupId))
      hoveredGroupOnCanvas = hoveredId;
  }

  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && boxBodyHovered &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    const auto id = static_cast<std::int32_t>(hoveredNode.Get());
    const auto isGroup = graph.isGroupOnFocusedCanvas(id, focusedGroupId);
    const auto isNode = graph.isNodeOnFocusedCanvas(id, focusedGroupId);
    if (isGroup || isNode) {
      const auto now = ImGui::GetTime();
      const auto dx = mouseScreen.x - canvasLastClickScreenPos.x;
      const auto dy = mouseScreen.y - canvasLastClickScreenPos.y;
      const auto isDouble =
          canvasLastClickBoxId == id &&
          (now - canvasLastClickTime) <= boxDoubleClickSeconds &&
          (dx * dx + dy * dy) <=
              boxDoubleClickMaxPixels * boxDoubleClickMaxPixels;
      canvasLastClickBoxId = id;
      canvasLastClickTime = now;
      canvasLastClickScreenPos = mouseScreen;
      if (isDouble && isGroup) {
        canvasLastClickBoxId = 0;
        canvasPressBoxId = 0;
        openGroupCanvasFitted(graph, id);
      } else {
        canvasPressBoxId = id;
        ed::SelectNode(ed::NodeId(editorIdentifier(id)), input.KeyShift);
      }
    }
  }

  const auto boxMoveHeld = canvasPressBoxId != 0 || draggingNodeId != 0 ||
                           draggingGroupId != 0;
  if ((std::abs(wheelX) > 0.0f || std::abs(wheelY) > 0.0f) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive() && !overMap && !boxMoveHeld) {
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
      !ImGui::IsAnyItemActive() && !overMap && !boxMoveHeld) {
    navigateCanvas(ImVec2(-mouseDeltaScreen.x * canvasPanSpeed,
                          -mouseDeltaScreen.y * canvasPanSpeed),
                   1.0f, mouseScreen);
  }

  const auto wheelUsed =
      std::abs(wheelX) > 0.0f || std::abs(wheelY) > 0.0f;
  if (std::abs(pinchMagnification - 1.0f) > 0.0001f && !overMap &&
      !boxMoveHeld && !wheelUsed)
    navigateCanvas(ImVec2(0.0f, 0.0f), pinchMagnification, mouseScreen);

  const auto hadLayoutDrag = draggingNodeId != 0 || draggingGroupId != 0;
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
    canvasPressBoxId = 0;
    if (hadLayoutDrag) {
      patchGestureHeldThisFrame = true;
      patchGestureLabel = "Move box";
    }
  } else if (canvasPressBoxId != 0 &&
             ImGui::IsMouseDragging(ImGuiMouseButton_Left, boxDragStartPixels)) {
    canvasLastClickBoxId = 0;
    if (graph.isGroupOnFocusedCanvas(canvasPressBoxId, focusedGroupId))
      draggingGroupId = canvasPressBoxId;
    else if (graph.isNodeOnFocusedCanvas(canvasPressBoxId, focusedGroupId))
      draggingNodeId = canvasPressBoxId;
  }

  if (draggingNodeId != 0 || draggingGroupId != 0) {
    dropTargetGroupId = hoveredGroupOnCanvas;
    patchGestureHeldThisFrame = true;
    patchGestureLabel = "Move box";
  }

  syncEditorTransforms(graph);
  pendingInfoTooltip.clear();

  for (auto &group : graph.getGroups()) {
    if (graph.isGroupOnFocusedCanvas(group.id, focusedGroupId))
      renderGroup(graph, group, callbacks);
  }

  for (auto &node : graph.getNodes()) {
    if (graph.isNodeOnFocusedCanvas(node.id, focusedGroupId))
      renderNode(graph, node, callbacks);
  }

  std::unordered_set<std::int32_t> liveLinkIds;
  liveLinkIds.reserve(graph.getLinks().size());
  const float fillSmooth =
      1.0f - std::exp(-ImGui::GetIO().DeltaTime / linkFillSmoothSeconds);
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
             ImColor(idleLinkColour), idleLinkThickness);
    liveLinkIds.insert(link.id);
    float targetFill = 0.0f;
    if (outputRmsByNodeId != nullptr) {
      const auto found = outputRmsByNodeId->find(*sourceNode);
      if (found != outputRmsByNodeId->end())
        targetFill = rmsToLinkFill(found->second);
    }
    auto &displayed = displayedLinkFill[link.id];
    displayed += (targetFill - displayed) * fillSmooth;
    drawDirectedLinkFill(link.id, displayed);
  }
  for (auto it = displayedLinkFill.begin(); it != displayedLinkFill.end();) {
    if (liveLinkIds.count(it->first) == 0)
      it = displayedLinkFill.erase(it);
    else
      ++it;
  }
  if (auto *editor = detailEditor();
      editor != nullptr && editor->GetDrawList() != nullptr)
    editor->GetDrawList()->ChannelsSetCurrent(imguiUserContentChannel);

  flushPendingInfoTooltip();
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
      if (!graph.isFixedIoNode(nodeId) &&
          !graph.isGroupBoundaryNode(nodeId))
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
        const auto id =
            placePaletteItem(graph, item, {local.x, local.y}, dropGroup);
        (void)id;
      } else {
        placePaletteItem(graph, item,
                         {canvasPosition.x, canvasPosition.y}, focusedGroupId);
      }
      positionedNodeIds.erase(graph.getNodes().back().id);
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
    if (boxLibrary != nullptr) {
      if (const auto *payload =
              ImGui::AcceptDragDropPayload(
                  openyourbox::ui::boxLibraryPayloadId)) {
        const auto *drop =
            static_cast<const openyourbox::ui::BoxLibraryDropPayload *>(
                payload->Data);
        const auto entryId = juce::String::fromUTF8(drop->entryId);
        const auto canvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        juce::String error;
        const auto rootId = boxLibrary->insertBox(
            graph, entryId, {canvasPosition.x, canvasPosition.y}, error,
            drop->nestedRootId);
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
    if (size.x > 0.0f && size.y > 0.0f && node.id != draggingNodeId &&
        node.id != canvasPressBoxId)
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
    if (size.x > 0.0f && size.y > 0.0f && group.id != draggingGroupId &&
        group.id != canvasPressBoxId)
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
  flushDocumentCallbacks(callbacks);
}

std::int32_t NodeRenderer::getPrimarySelectedNodeId() const noexcept {
  return selectedNodeIds.empty() ? 0 : selectedNodeIds.front();
}

void NodeRenderer::renderPalette(NodeGraph &graph,
                                 openyourbox::library::UserBoxLibrary *boxLibrary) {
  constexpr float splitterWidth = 6.0f;
  constexpr float minPalette = 140.0f;
  constexpr float maxPalette = 420.0f;
  constexpr float minCanvas = 220.0f;
  const auto avail = ImGui::GetContentRegionAvail();
  auto width = std::clamp(leftPaletteWidth, minPalette, maxPalette);
  width = std::min(width, std::max(minPalette, avail.x - minCanvas - splitterWidth));
  leftPaletteWidth = width;
  ImGui::BeginChild("ElementPalette", ImVec2(width, 0.0f), true);
  ImGui::TextColored(ImVec4(0.39f, 0.70f, 1.0f, 1.0f), "Library");
  ImGui::TextDisabled("Drag onto the graph");
  ImGui::Separator();

  const auto factorySelected =
      boxLibrary == nullptr || boxLibraryPanel.isFactorySelected();
  const auto factoryFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
      ImGuiTreeNodeFlags_SpanAvailWidth |
      (factorySelected ? ImGuiTreeNodeFlags_Selected : 0);
  openyourbox::ui::pushPaletteRootHeaderStyle();
  const auto factoryOpen =
      ImGui::TreeNodeEx("FACTORY###Factory", factoryFlags);
  openyourbox::ui::popPaletteRootHeaderStyle();
  if (boxLibrary != nullptr && ImGui::IsItemClicked() &&
      !ImGui::IsItemToggledOpen())
    boxLibraryPanel.selectFactory();
  if (ImGui::BeginPopupContextItem("FactoryRootMenu")) {
    ImGui::TextDisabled("Factory cannot be renamed or deleted");
    ImGui::EndPopup();
  }
  if (factoryOpen) {
    const auto renderPaletteSelectable = [&](const PaletteItem &item) {
      ImGui::PushID(item.label);
      ImGui::Selectable(item.label, false);
      if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        const auto now = ImGui::GetTime();
        const auto isDouble =
            paletteLastClickLabel == item.label &&
            (now - paletteLastClickTime) <= boxDoubleClickSeconds;
        paletteLastClickLabel = item.label;
        paletteLastClickTime = now;
        if (isDouble) {
          paletteLastClickLabel.clear();
          const auto focus = graph.getViewport().focusedGroupId;
          placePaletteItem(graph, item,
                           {lastCanvasCentre.x, lastCanvasCentre.y}, focus);
          mutatedThisFrame = true;
          recompileThisFrame = true;
        }
      }
      if (ImGui::IsItemActive() &&
          ImGui::IsMouseDragging(ImGuiMouseButton_Left,
                                 ImGui::GetIO().MouseDragThreshold)) {
        paletteLastClickLabel.clear();
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
          ImGui::SetDragDropPayload("OPENYOURBOX_NODE_TYPE", &item,
                                    sizeof(item));
          ImGui::TextUnformatted(item.label);
          ImGui::EndDragDropSource();
        }
      }
      ImGui::PopID();
    };
    for (const auto &category : paletteCategories) {
      if (ImGui::TreeNodeEx(category.label, ImGuiTreeNodeFlags_DefaultOpen)) {
        for (std::size_t index = 0; index < category.count; ++index)
          renderPaletteSelectable(category.items[index]);
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
  if (boxLibrary != nullptr) {
    openyourbox::ui::UserBoxLibraryPanel::Callbacks boxCallbacks;
    boxCallbacks.showMessage = [this](const std::string &message) {
      transientMessage = message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
    };
    boxCallbacks.inspectRequested = [this](const juce::String &entryId,
                                           std::int32_t nestedRootId) {
      inspectLibraryEntry(entryId, nestedRootId);
    };
    boxCallbacks.placeRequested = [this, &graph](const juce::String &entryId,
                                                 std::int32_t nestedRootId) {
      placeLibraryEntryOnFocusedCanvas(graph, entryId, nestedRootId);
    };
    boxLibraryPanel.render(*boxLibrary, boxCallbacks,
                           [boxLibrary](const juce::String &id,
                                        juce::String &error) {
                             return boxLibrary->loadEntrySnapshot(id, error);
                           });
  }
  openyourbox::ui::pushPaletteRootHeaderStyle();
  const auto structureOpen = ImGui::TreeNodeEx(
      "PROJECT STRUCTURE###Project structure",
      ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth);
  openyourbox::ui::popPaletteRootHeaderStyle();
  if (structureOpen) {
    renderProjectStructure(graph);
    ImGui::TreePop();
  }
  if (structureDropHighlightActive) {
    auto *draw = ImGui::GetWindowDrawList();
    draw->AddRect(structureDropHighlightMin, structureDropHighlightMax,
                  ImGui::GetColorU32(ImVec4(0.47f, 0.82f, 1.0f, 0.95f)), 3.0f,
                  0, 2.0f);
  }
  ImGui::TextWrapped(
      "Scroll to pan. Ctrl/Cmd+scroll or pinch to zoom at the pointer. "
      "Double-click a group to open it. Use Main > group at the top to go back.");
  ImGui::EndChild();
  ImGui::SameLine(0.0f, 0.0f);
  ImGui::InvisibleButton("PaletteSplitter",
                         ImVec2(splitterWidth, avail.y > 0.0f ? avail.y : 1.0f));
  if (ImGui::IsItemActive()) {
    leftPaletteWidth =
        std::clamp(leftPaletteWidth + ImGui::GetIO().MouseDelta.x, minPalette,
                   std::min(maxPalette, avail.x - minCanvas - splitterWidth));
    layoutMutatedThisFrame = true;
  }
  if (ImGui::IsItemHovered() || ImGui::IsItemActive())
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
}

void NodeRenderer::renderNode(NodeGraph &graph, GraphNode &node,
                              const NodeRendererCallbacks &callbacks) {
  (void)callbacks;
  positionedNodeIds.insert(node.id);
  const auto nodeColour = colourFor(node.colour);
  ed::PushStyleColor(ed::StyleColor_NodeBg,
                     ImVec4(nodeColour.x, nodeColour.y, nodeColour.z, 0.28f));
  ed::PushStyleColor(ed::StyleColor_NodeBorder, nodeColour);
  ed::BeginNode(ed::NodeId(editorIdentifier(node.id)));
  ImGui::PushID(node.id);
  reservePassiveBoxWidth(nodeBodyWidth);
  ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + nodeBodyWidth);

  if (node.state == NodeState::frozenGold)
    drawPassiveBoxText("\xF0\x9F\x94\x92");
  if (node.state == NodeState::frozenGold)
    ImGui::SameLine();
  drawPassiveBoxText(node.label.c_str());

  if (node.type != NodeType::groupInput) {
    const auto pinRepeatCounts = graph.ancestorRuntimeRepeatCounts(node.id);
    for (const auto &pin : node.inputs) {
    ed::BeginPin(ed::PinId(editorIdentifier(pin.id)), ed::PinKind::Input);
    ImGui::PushID(pin.id);
    drawPinCaption(PinKind::input, pin.label.c_str(), pin.shape,
                   pinExpandedShapeInfo(pin, pinRepeatCounts));
    ImGui::PopID();
    ed::EndPin();
    }
  }

  if (node.type != NodeType::groupOutput) {
    const auto pinRepeatCounts = graph.ancestorRuntimeRepeatCounts(node.id);
    for (const auto &pin : node.outputs) {
    ed::BeginPin(ed::PinId(editorIdentifier(pin.id)), ed::PinKind::Output);
    ImGui::PushID(pin.id);
    drawPinCaption(PinKind::output, pin.label.c_str(), pin.shape,
                   pinExpandedShapeInfo(pin, pinRepeatCounts));
    ImGui::PopID();
    ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ed::EndPin();
    }
  }

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
  if (ImGui::IsItemActive()) {
    patchGestureHeldThisFrame = true;
    patchGestureLabel = "Parameter edit";
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
  if (active) {
    patchGestureHeldThisFrame = true;
    patchGestureLabel = "Parameter edit";
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
  (void)callbacks;

  reservePassiveBoxWidth(nodeBodyWidth);
  const auto repeatStatus = graph.groupRepeatStatus(group.id);
  const auto groupLabel = groupBoxDisplayLabel(group, repeatStatus);
  if (!repeatStatus.active && repeatStatus.requestedRepeats > 1) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.18f, 1.0f));
    drawPassiveBoxText(groupLabel.c_str());
    ImGui::PopStyleColor();
  } else {
    drawPassiveBoxText(groupLabel.c_str());
  }
  drawPassiveBoxText("Double-click to open", true);

  std::vector<GroupBoundaryPort> inputs;
  std::vector<GroupBoundaryPort> outputs;
  for (const auto &port : graph.groupInterfacePorts(group.id)) {
    if (port.kind == PinKind::input)
      inputs.push_back(port);
    else
      outputs.push_back(port);
  }
  const auto drawPort = [&graph](const GroupBoundaryPort &port) {
    const auto pinId = collapsedGroupPinId(port.memberPinId);
    ed::BeginPin(ed::PinId(editorIdentifier(pinId)),
                 port.kind == PinKind::input ? ed::PinKind::Input
                                             : ed::PinKind::Output);
    ImGui::PushID(pinId);
    // Collapsed boxes fold this group's own serial repeats and any nested
    // descendant repeat axes to first-in / last-out; ancestor repeat axes still
    // nest as a list.
    ShapeSignature displayShape = port.shape;
    std::string expandedShapes;
    if (const auto *memberPin = graph.findPin(port.memberPinId)) {
      const auto repeatCounts =
          graph.ancestorRuntimeRepeatCounts(port.memberNodeId);
      const bool takeLast = port.kind == PinKind::output;
      if (takeLast && !memberPin->repeatShapes.empty())
        displayShape = memberPin->repeatShapes.back();
      expandedShapes = formatCollapsedGroupPinShapes(
          memberPin->repeatShapes, displayShape, repeatCounts, takeLast);
    }
    drawPinCaption(port.kind, port.label.c_str(), displayShape, expandedShapes);
    if (port.kind == PinKind::output)
      ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ImGui::PopID();
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
  std::vector<std::int32_t> deletedLinkIds;
  std::vector<std::int32_t> deletedBoxIds;
  const auto focus = graph.getViewport().focusedGroupId;
  ed::LinkId linkId;
  while (ed::QueryDeletedLink(&linkId)) {
    if (ed::AcceptDeletedItem())
      deletedLinkIds.push_back(static_cast<std::int32_t>(linkId.Get()));
  }
  ed::NodeId nodeId;
  while (ed::QueryDeletedNode(&nodeId)) {
    const auto id = static_cast<std::int32_t>(nodeId.Get());
    if (graph.findGroup(id) != nullptr) {
      if (ed::AcceptDeletedItem()) {
        deletedBoxIds.push_back(id);
        positionedGroupIds.erase(id);
      }
      continue;
    }
    const auto *node = graph.findNode(id);
    if (node == nullptr || graph.isFixedIoNode(id) ||
        graph.isGroupBoundaryNode(id)) {
      ed::RejectDeletedItem();
      continue;
    }
    if (ed::AcceptDeletedItem()) {
      deletedBoxIds.push_back(id);
      positionedNodeIds.erase(id);
    }
  }
  ed::EndDelete();

  std::optional<std::int32_t> focusAfterDelete;
  bool retargetFocus = false;
  if (focus.has_value()) {
    for (const auto ancestor : graph.groupAncestorChain(*focus)) {
      if (std::find(deletedBoxIds.begin(), deletedBoxIds.end(), ancestor) !=
          deletedBoxIds.end()) {
        const auto *group = graph.findGroup(ancestor);
        focusAfterDelete =
            group != nullptr ? group->parentGroupId : std::nullopt;
        retargetFocus = true;
        break;
      }
    }
  }

  if (!deletedBoxIds.empty()) {
    mutatedThisFrame = graph.removeBoxes(deletedBoxIds) || mutatedThisFrame;
    recompileThisFrame = mutatedThisFrame || recompileThisFrame;
    if (retargetFocus)
      setCanvasFocus(graph, focusAfterDelete);
  }
  for (const auto id : deletedLinkIds) {
    mutatedThisFrame = graph.removeLink(id) || mutatedThisFrame;
    recompileThisFrame = mutatedThisFrame || recompileThisFrame;
  }
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
      if (!graph.isFixedIoNode(id) && !graph.isGroupBoundaryNode(id))
        freezeIds.push_back(id);
    }
    for (const auto id : selectedGroupIds)
      freezeIds.push_back(id);
    if (contextNodeId != 0 && !graph.isFixedIoNode(contextNodeId) &&
        !graph.isGroupBoundaryNode(contextNodeId) &&
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
      const auto allowUnfreeze = graph.canUnfreeze(contextNodeId);
      if (ImGui::MenuItem("Unfreeze", nullptr, false, allowUnfreeze) &&
          callbacks.unfreezeNode)
        callbacks.unfreezeNode(contextNodeId);
      if (!allowUnfreeze && isExternalLoadNode(*node)) {
        ImGui::TextDisabled("External checkpoints cannot Unfreeze");
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip(
              "TorchScript Load has no OpenYourBox source graph to restore.");
      }
    } else {
      const auto expandedFreeze =
          graph.expandSelectionToFreezableLeaves(freezeIds);
      const auto canFreeze =
          !graph.partitionFreezeChains(expandedFreeze).empty();
      if (ImGui::MenuItem("Freeze Selection", nullptr, false, canFreeze) &&
          callbacks.freezeSelection)
        callbacks.freezeSelection(freezeIds);
    }
    if (node != nullptr && !graph.isFixedIoNode(contextNodeId) &&
        !graph.isGroupBoundaryNode(contextNodeId) &&
        ImGui::MenuItem("Delete")) {
      const auto ids = contextDeleteBoxIds(graph, contextNodeId,
                                           selectedNodeIds, selectedGroupIds);
      mutatedThisFrame = graph.removeBoxes(ids) || mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
      for (const auto id : ids) {
        positionedNodeIds.erase(id);
        positionedGroupIds.erase(id);
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
      const auto ids = contextDeleteBoxIds(graph, contextGroupId,
                                           selectedNodeIds, selectedGroupIds);
      mutatedThisFrame = graph.removeBoxes(ids) || mutatedThisFrame;
      recompileThisFrame = mutatedThisFrame || recompileThisFrame;
      for (const auto id : ids)
        positionedGroupIds.erase(id);
      positionedGroupIds.erase(contextGroupId);
      if (leaveScope)
        setCanvasFocus(graph, parent);
    }
    if (group != nullptr && ImGui::MenuItem("Open"))
      setCanvasFocus(graph, contextGroupId);
    if (node != nullptr && !graph.isGroupBoundaryNode(contextNodeId) &&
        node->parentGroupId.has_value() &&
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
        node != nullptr &&
        (graph.isFixedIoNode(contextNodeId) ||
         graph.isGroupBoundaryNode(contextNodeId));
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
      forEachPaletteItem([&](const PaletteItem &item) {
        if (!canInsertOnLink(item.type))
          return;
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
      });
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
      forEachPaletteItem([&](const PaletteItem &item) {
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
      });
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

  const auto totalBoxes = selectedNodeIds.size() + selectedGroupIds.size();
  const auto holdLibraryInspect = preserveLibraryInspect;
  preserveLibraryInspect = false;
  const auto userPickedCanvasBox =
      !holdLibraryInspect && ed::HasSelectionChanged() && totalBoxes >= 1;
  if (selectionContext.kind == SelectionContextKind::LibraryInspect &&
      !userPickedCanvasBox) {
    preserveLiveSelection = false;
    if (!holdLibraryInspect && ed::IsBackgroundClicked()) {
      selectionContext.kind = SelectionContextKind::None;
      selectionContext.liveBoxId.reset();
      selectionContext.libraryEntryId.clear();
      selectionContext.libraryNestedRootId = 0;
    }
  } else if (totalBoxes > 1) {
    selectionContext.kind = SelectionContextKind::Multi;
    selectionContext.liveBoxId.reset();
    selectionContext.libraryEntryId.clear();
    selectionContext.libraryNestedRootId = 0;
    preserveLiveSelection = false;
  } else if (totalBoxes == 1) {
    const auto boxId =
        !selectedGroupIds.empty() ? selectedGroupIds.front() : selectedNodeIds.front();
    selectionContext.kind = SelectionContextKind::Live;
    selectionContext.liveBoxId = boxId;
    selectionContext.libraryEntryId.clear();
    selectionContext.libraryNestedRootId = 0;
    preserveLiveSelection = false;
  } else if (ed::IsBackgroundClicked()) {
    selectionContext.kind = SelectionContextKind::None;
    selectionContext.liveBoxId.reset();
    selectionContext.libraryEntryId.clear();
    selectionContext.libraryNestedRootId = 0;
    preserveLiveSelection = false;
  } else if (preserveLiveSelection) {
    preserveLiveSelection = false;
  } else if (selectionContext.kind == SelectionContextKind::LibraryInspect) {
    // Keep catalog inspect until a live box is selected.
  } else if (selectionContext.kind == SelectionContextKind::Live &&
             selectionContext.liveBoxId.has_value() &&
             (graph.findNode(*selectionContext.liveBoxId) != nullptr ||
              graph.findGroup(*selectionContext.liveBoxId) != nullptr)) {
    // Keep Project structure selection when the box is off the focused canvas.
  } else {
    selectionContext.kind = SelectionContextKind::None;
    selectionContext.liveBoxId.reset();
  }

  std::int32_t signature = 0;
  if (selectionContext.kind == SelectionContextKind::Live &&
      selectionContext.liveBoxId.has_value())
    signature = *selectionContext.liveBoxId;
  else if (selectionContext.kind == SelectionContextKind::Multi)
    signature = -1;
  else if (selectionContext.kind == SelectionContextKind::LibraryInspect) {
    signature = -2;
    for (char character : selectionContext.libraryEntryId)
      signature = signature * 33 - static_cast<int>(character);
  }
  if (signature != lastSelectionSignature) {
    lastSelectionSignature = signature;
    if (selectionContext.kind != SelectionContextKind::None)
      selectionContext.forceParametersTab = true;
  }
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
    if (group.id != draggingGroupId && group.id != canvasPressBoxId) {
      ed::SetNodePosition(ed::NodeId(editorIdentifier(group.id)),
                          ImVec2(group.position.x, group.position.y));
    }
  }
  for (auto &node : graph.getNodes()) {
    if (!graph.isNodeOnFocusedCanvas(node.id, focusedGroupId) ||
        node.id == draggingNodeId || node.id == canvasPressBoxId)
      continue;
    positionedNodeIds.insert(node.id);
    ed::SetNodePosition(ed::NodeId(editorIdentifier(node.id)),
                        ImVec2(node.position.x, node.position.y));
  }
}

void NodeRenderer::renderScopeBreadcrumb(NodeGraph &graph) {
  auto &viewport = graph.getViewport();
  viewport.stickySpine.erase(
      std::remove_if(viewport.stickySpine.begin(), viewport.stickySpine.end(),
                     [&graph](std::int32_t id) {
                       return graph.findGroup(id) == nullptr;
                     }),
      viewport.stickySpine.end());
  const auto focused = viewport.focusedGroupId;
  const auto followFocus = scopeBreadcrumbFollowFocus;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::BeginChild("ScopeBreadcrumb",
                    ImVec2(0.0f, ImGui::GetFrameHeightWithSpacing()), false,
                    ImGuiWindowFlags_HorizontalScrollbar |
                        ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PopStyleVar();
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 4.0f));
  if (!focused.has_value()) {
    ImGui::TextColored(ImVec4(0.75f, 0.88f, 1.0f, 1.0f), "%s", rootCanvasLabel);
    if (followFocus)
      ImGui::SetScrollHereX(0.0f);
  } else if (ImGui::SmallButton(rootCanvasLabel)) {
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
        if (followFocus)
          ImGui::SetScrollHereX(0.75f);
      } else if (ImGui::SmallButton(group->name.c_str())) {
        setCanvasFocus(graph, id);
      }
      ImGui::PopID();
    }
  }
  for (const auto id : viewport.stickySpine) {
    const auto *group = graph.findGroup(id);
    if (group == nullptr)
      continue;
    ImGui::SameLine();
    ImGui::TextDisabled(">");
    ImGui::SameLine();
    ImGui::PushID(static_cast<int>(id) ^ 0x5a5a5a5a);
    if (ImGui::SmallButton(group->name.c_str()))
      setCanvasFocus(graph, id);
    ImGui::PopID();
  }
  ImGui::PopStyleVar();

  auto &io = ImGui::GetIO();
  if (ImGui::IsWindowHovered()) {
    const auto wheel =
        io.MouseWheelH != 0.0f ? io.MouseWheelH : io.MouseWheel;
    if (wheel != 0.0f) {
      ImGui::SetScrollX(ImGui::GetScrollX() - wheel * breadcrumbWheelStep);
      io.MouseWheel = 0.0f;
      io.MouseWheelH = 0.0f;
    }
    if (!ImGui::IsAnyItemActive() &&
        (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f) ||
         ImGui::IsMouseDragging(canvasDragPanButton, 0.0f))) {
      ImGui::SetScrollX(ImGui::GetScrollX() - io.MouseDelta.x);
    }
  }
  const auto croppedLeft = ImGui::GetScrollX() > 1.0f;
  const auto croppedRight =
      ImGui::GetScrollX() < ImGui::GetScrollMaxX() - 1.0f;
  ImGui::EndChild();
  if (followFocus)
    scopeBreadcrumbFollowFocus = false;
  drawBreadcrumbCropFades(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                          croppedLeft, croppedRight);
}

void NodeRenderer::setCanvasFocus(NodeGraph &graph,
                                  std::optional<std::int32_t> groupId) {
  if (groupId.has_value() && graph.findGroup(*groupId) == nullptr)
    groupId.reset();
  auto &viewport = graph.getViewport();
  if (viewport.focusedGroupId == groupId)
    return;
  const auto previousChain =
      viewport.focusedGroupId.has_value()
          ? graph.groupAncestorChain(*viewport.focusedGroupId)
          : std::vector<std::int32_t>{};
  const auto nextChain =
      groupId.has_value() ? graph.groupAncestorChain(*groupId)
                          : std::vector<std::int32_t>{};
  updateHierarchyStickySpine(viewport.stickySpine, previousChain, nextChain);
  viewport.focusedGroupId = groupId;
  restoreViewPending = true;
  scopeBreadcrumbFollowFocus = true;
  layoutMutatedThisFrame = true;
}

void NodeRenderer::openGroupCanvasFitted(NodeGraph &graph,
                                         std::int32_t groupId) {
  setCanvasFocus(graph, groupId);
  pendingFitCanvas = true;
}

void NodeRenderer::fitCanvasToContents(NodeGraph &graph, ImVec2 canvasSize) {
  if (canvasSize.x < 1.0f || canvasSize.y < 1.0f)
    return;
  const auto focused = graph.getViewport().focusedGroupId;
  ImVec2 min{std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};
  ImVec2 max{std::numeric_limits<float>::lowest(),
             std::numeric_limits<float>::lowest()};
  auto any = false;
  const auto include = [&](juce::Point<float> position, juce::Point<float> size) {
    any = true;
    min.x = std::min(min.x, position.x);
    min.y = std::min(min.y, position.y);
    max.x = std::max(max.x, position.x + std::max(8.0f, size.x));
    max.y = std::max(max.y, position.y + std::max(8.0f, size.y));
  };
  for (const auto &node : graph.getNodes()) {
    if (graph.isNodeOnFocusedCanvas(node.id, focused))
      include(node.position, node.size);
  }
  for (const auto &group : graph.getGroups()) {
    if (graph.isGroupOnFocusedCanvas(group.id, focused))
      include(group.position, group.size);
  }
  ImVec2 centre{defaultNewBoxPosition.x, defaultNewBoxPosition.y};
  auto zoom = 1.0f;
  if (any) {
    centre = ImVec2((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f);
    const auto padded =
        ImVec2(std::max(1.0f, max.x - min.x + canvasFitPadding * 2.0f),
               std::max(1.0f, max.y - min.y + canvasFitPadding * 2.0f));
    zoom = std::clamp(std::min(canvasSize.x / padded.x, canvasSize.y / padded.y),
                      minimumZoom, maximumZoom);
  }
  const auto viewSize =
      ImVec2(canvasSize.x / std::max(0.01f, zoom),
             canvasSize.y / std::max(0.01f, zoom));
  ImRect view;
  view.Min = ImVec2(centre.x - viewSize.x * 0.5f, centre.y - viewSize.y * 0.5f);
  view.Max = ImVec2(centre.x + viewSize.x * 0.5f, centre.y + viewSize.y * 0.5f);
  commitCanvasView(view);
  layoutMutatedThisFrame = true;
}

void NodeRenderer::renderProjectStructure(NodeGraph &graph) {
  std::vector<std::int32_t> roots;
  for (const auto &group : graph.getGroups()) {
    if (!group.parentGroupId.has_value())
      roots.push_back(group.id);
  }
  for (const auto &node : graph.getNodes()) {
    if (!node.parentGroupId.has_value() && !isFixedIoType(node.type) &&
        !isGroupBoundaryType(node.type))
      roots.push_back(node.id);
  }
  roots = graph.orderSiblingBoxesByFlow(std::nullopt, std::move(roots));

  ImGui::PushID("projectMain");
  const auto flags = ImGuiTreeNodeFlags_DefaultOpen |
                     ImGuiTreeNodeFlags_OpenOnArrow |
                     ImGuiTreeNodeFlags_SpanAvailWidth;
  const auto open = ImGui::TreeNodeEx(rootCanvasLabel, flags);
  const auto headerMin = ImGui::GetItemRectMin();
  const auto headerMax = ImGui::GetItemRectMax();
  if (!ImGui::IsItemToggledOpen() &&
      ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
    const auto now = ImGui::GetTime();
    const auto isDouble = structureLastClickBoxId == mainStructureRowId &&
                          (now - structureLastClickTime) <= boxDoubleClickSeconds;
    structureLastClickBoxId = mainStructureRowId;
    structureLastClickTime = now;
    if (isDouble) {
      structureLastClickBoxId = 0;
      setCanvasFocus(graph, std::nullopt);
      pendingFitCanvas = true;
    }
  }
  if (activeCallbacks != nullptr)
    handleStructureDropTarget(graph, *activeCallbacks, std::nullopt, headerMin,
                              headerMax);
  if (open) {
    if (roots.empty())
      ImGui::TextDisabled("Empty graph");
    else {
      for (const auto id : roots)
        renderProjectStructureItem(graph, id);
    }
    ImGui::TreePop();
  }
  if (structureDropHighlightActive && !structureDropTargetParent.has_value()) {
    structureDropHighlightMin = headerMin;
    structureDropHighlightMax = ImVec2(
        std::max(headerMax.x, ImGui::GetWindowPos().x + ImGui::GetWindowWidth() -
                                  8.0f),
        std::max(headerMax.y, ImGui::GetCursorScreenPos().y));
  }
  ImGui::PopID();
}

void NodeRenderer::renderProjectStructureItem(NodeGraph &graph,
                                              std::int32_t boxId) {
  if (const auto *boundary = graph.findNode(boxId);
      boundary != nullptr && isGroupBoundaryType(boundary->type))
    return;

  ImGui::PushID(boxId);
  if (const auto *group = graph.findGroup(boxId)) {
    const auto flags =
        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
        (selectionContext.kind == SelectionContextKind::Live &&
                 selectionContext.liveBoxId == boxId
             ? ImGuiTreeNodeFlags_Selected
             : 0);
    const auto open =
        ImGui::TreeNodeEx("##projectGroup", flags, "%s", group->name.c_str());
    const auto headerMin = ImGui::GetItemRectMin();
    const auto headerMax = ImGui::GetItemRectMax();
    handleStructureRowClick(graph, boxId);
    if (ImGui::IsItemActive() &&
        ImGui::IsMouseDragging(ImGuiMouseButton_Left,
                               ImGui::GetIO().MouseDragThreshold)) {
      structureLastClickBoxId = 0;
      beginStructureDrag(graph, boxId);
    }
    if (activeCallbacks != nullptr)
      handleStructureDropTarget(graph, *activeCallbacks, boxId, headerMin,
                                headerMax);
    if (ImGui::BeginPopupContextItem("ProjectStructureGroup")) {
      if (ImGui::MenuItem("Save to Box Library...")) {
        pendingSaveBoxId = boxId;
        requestSaveBoxPopup = true;
        saveBoxOverwrite = false;
        const auto utf8 = juce::String(group->name).toRawUTF8();
        const auto length =
            std::min(saveBoxNameBuffer.size() - 1, std::strlen(utf8));
        std::memcpy(saveBoxNameBuffer.data(), utf8, length);
        saveBoxNameBuffer[length] = '\0';
      }
      ImGui::EndPopup();
    }
    if (open) {
      const auto members = graph.orderSiblingBoxesByFlow(boxId, group->memberIds);
      for (const auto member : members)
        renderProjectStructureItem(graph, member);
      ImGui::TreePop();
    }
    if (structureDropHighlightActive && structureDropTargetParent == boxId) {
      structureDropHighlightMin = headerMin;
      structureDropHighlightMax =
          ImVec2(std::max(headerMax.x, ImGui::GetWindowPos().x +
                                           ImGui::GetWindowWidth() - 8.0f),
                 std::max(headerMax.y, ImGui::GetCursorScreenPos().y));
    }
  } else if (const auto *node = graph.findNode(boxId)) {
    const auto selected = selectionContext.kind == SelectionContextKind::Live &&
                          selectionContext.liveBoxId == boxId;
    ImGui::TreeNodeEx(node->label.c_str(),
                      ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                          ImGuiTreeNodeFlags_SpanAvailWidth |
                          (selected ? ImGuiTreeNodeFlags_Selected : 0));
    const auto headerMin = ImGui::GetItemRectMin();
    const auto headerMax = ImGui::GetItemRectMax();
    handleStructureRowClick(graph, boxId);
    if (ImGui::IsItemActive() &&
        ImGui::IsMouseDragging(ImGuiMouseButton_Left,
                               ImGui::GetIO().MouseDragThreshold)) {
      structureLastClickBoxId = 0;
      beginStructureDrag(graph, boxId);
    }
    if (activeCallbacks != nullptr)
      handleStructureDropTarget(graph, *activeCallbacks, node->parentGroupId,
                                headerMin, headerMax);
    if (!isFixedIoType(node->type) && !isGroupBoundaryType(node->type) &&
        ImGui::BeginPopupContextItem("ProjectStructureNode")) {
      if (ImGui::MenuItem("Save to Box Library...")) {
        pendingSaveBoxId = boxId;
        requestSaveBoxPopup = true;
        saveBoxOverwrite = false;
        const auto utf8 = juce::String(node->label).toRawUTF8();
        const auto length =
            std::min(saveBoxNameBuffer.size() - 1, std::strlen(utf8));
        std::memcpy(saveBoxNameBuffer.data(), utf8, length);
        saveBoxNameBuffer[length] = '\0';
      }
      ImGui::EndPopup();
    }
  }
  ImGui::PopID();
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

std::int32_t NodeRenderer::getPrimarySelectedBoxId() const noexcept {
  if (!selectedGroupIds.empty())
    return selectedGroupIds.front();
  if (!selectedNodeIds.empty())
    return selectedNodeIds.front();
  if (selectionContext.kind == SelectionContextKind::Live &&
      selectionContext.liveBoxId.has_value())
    return *selectionContext.liveBoxId;
  return 0;
}

const SelectionContext &NodeRenderer::getSelectionContext() const noexcept {
  return selectionContext;
}

bool NodeRenderer::consumeForceParametersTab() noexcept {
  const auto requested = selectionContext.forceParametersTab;
  selectionContext.forceParametersTab = false;
  return requested;
}

void NodeRenderer::inspectLibraryEntry(const juce::String &entryId,
                                       std::int32_t nestedRootId) {
  selectionContext.kind = SelectionContextKind::LibraryInspect;
  selectionContext.liveBoxId.reset();
  selectionContext.libraryEntryId = entryId.toStdString();
  selectionContext.libraryNestedRootId = nestedRootId;
  selectionContext.forceParametersTab = true;
  libraryInspectSnapshot = {};
  lastSelectionSignature = 0;
  preserveLiveSelection = false;
  preserveLibraryInspect = true;
  pendingEditorSelectId = -1;
}

void NodeRenderer::placeLibraryEntryOnFocusedCanvas(
    NodeGraph &graph, const juce::String &entryId,
    std::int32_t nestedRootId) {
  if (activeBoxLibrary == nullptr)
    return;
  juce::String error;
  const auto canvasPosition = lastCanvasCentre;
  const auto rootId = activeBoxLibrary->insertBox(
      graph, entryId, {canvasPosition.x, canvasPosition.y}, error,
      nestedRootId);
  if (!rootId.has_value()) {
    if (error.isNotEmpty()) {
      transientMessage = error.toStdString();
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (activeCallbacks != nullptr && activeCallbacks->showMessage)
        activeCallbacks->showMessage(error.toStdString());
    }
    return;
  }
  positionedNodeIds.erase(*rootId);
  positionedGroupIds.erase(*rootId);
  adoptNewBox(graph, *rootId, canvasPosition, 0);
  mutatedThisFrame = true;
  recompileThisFrame = true;
  if (activeCallbacks != nullptr && activeCallbacks->boxPlaced)
    activeCallbacks->boxPlaced(*rootId);
}

void NodeRenderer::selectLiveBox(NodeGraph &graph, std::int32_t boxId,
                                 bool forceTab) {
  if (graph.findNode(boxId) == nullptr && graph.findGroup(boxId) == nullptr)
    return;
  selectionContext.kind = SelectionContextKind::Live;
  selectionContext.liveBoxId = boxId;
  selectionContext.libraryEntryId.clear();
  selectionContext.libraryNestedRootId = 0;
  if (forceTab)
    selectionContext.forceParametersTab = true;
  preserveLiveSelection = true;
  selectedNodeIds.clear();
  selectedGroupIds.clear();
  if (graph.findGroup(boxId) != nullptr)
    selectedGroupIds.push_back(boxId);
  else
    selectedNodeIds.push_back(boxId);
  const auto focused = graph.getViewport().focusedGroupId;
  const auto onCanvas =
      graph.findGroup(boxId) != nullptr
          ? graph.isGroupOnFocusedCanvas(boxId, focused)
          : graph.isNodeOnFocusedCanvas(boxId, focused);
  pendingEditorSelectId = onCanvas ? boxId : -1;
}

void NodeRenderer::navigateToBox(NodeGraph &graph, std::int32_t boxId,
                                 bool openInnerGroup) {
  if (openInnerGroup && graph.findGroup(boxId) != nullptr) {
    openGroupCanvasFitted(graph, boxId);
    return;
  }
  std::optional<std::int32_t> containing;
  ImVec2 centre{defaultNewBoxPosition.x, defaultNewBoxPosition.y};
  if (const auto *group = graph.findGroup(boxId)) {
    containing = group->parentGroupId;
    centre = ImVec2(group->position.x + group->size.x * 0.5f,
                    group->position.y + group->size.y * 0.5f);
  } else if (const auto *node = graph.findNode(boxId)) {
    containing = node->parentGroupId;
    centre = ImVec2(node->position.x + node->size.x * 0.5f,
                    node->position.y + node->size.y * 0.5f);
  } else
    return;
  setCanvasFocus(graph, containing);
  pendingCentreView = true;
  pendingCentrePoint = centre;
}

void NodeRenderer::applyPendingEditorSelection(NodeGraph &graph) {
  (void)graph;
  if (pendingEditorSelectId == 0)
    return;
  ed::ClearSelection();
  if (pendingEditorSelectId > 0)
    ed::SelectNode(ed::NodeId(editorIdentifier(pendingEditorSelectId)), false);
  pendingEditorSelectId = 0;
}

void NodeRenderer::flushDocumentCallbacks(
    const NodeRendererCallbacks &callbacks) {
  if (patchGestureHeldThisFrame && !patchGestureOpen) {
    patchGestureOpen = true;
    if (callbacks.beginPatchGesture)
      callbacks.beginPatchGesture(patchGestureLabel);
  }
  if (mutatedThisFrame && callbacks.documentChanged)
    callbacks.documentChanged(recompileThisFrame, true);
  else if (layoutMutatedThisFrame && callbacks.documentChanged)
    callbacks.documentChanged(false, false);
  if (patchGestureOpen && !patchGestureHeldThisFrame) {
    patchGestureOpen = false;
    if (callbacks.endPatchGesture)
      callbacks.endPatchGesture();
  }
  mutatedThisFrame = false;
  layoutMutatedThisFrame = false;
  recompileThisFrame = false;
}

void NodeRenderer::beginStructureDrag(NodeGraph &graph, std::int32_t boxId) {
  if (!ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
    return;
  ImGui::SetDragDropPayload(structureBoxPayloadId, &boxId, sizeof(boxId));
  const char *label = "Box";
  if (const auto *group = graph.findGroup(boxId))
    label = group->name.c_str();
  else if (const auto *node = graph.findNode(boxId))
    label = node->label.c_str();
  ImGui::TextUnformatted(label);
  structureDragSourceId = boxId;
  ImGui::EndDragDropSource();
}

void NodeRenderer::handleStructureRowClick(NodeGraph &graph,
                                           std::int32_t boxId) {
  if (ImGui::IsItemToggledOpen())
    return;
  if (!ImGui::IsItemClicked(ImGuiMouseButton_Left))
    return;
  const auto now = ImGui::GetTime();
  const auto isDouble = structureLastClickBoxId == boxId &&
                        (now - structureLastClickTime) <= boxDoubleClickSeconds;
  structureLastClickBoxId = boxId;
  structureLastClickTime = now;
  if (isDouble) {
    structureLastClickBoxId = 0;
    navigateToBox(graph, boxId, graph.findGroup(boxId) != nullptr);
    return;
  }
  selectLiveBox(graph, boxId, true);
}

void NodeRenderer::focusAfterStructureMutation(
    NodeGraph &graph, std::int32_t boxId,
    std::optional<std::int32_t> destinationParent) {
  selectLiveBox(graph, boxId, true);
  setCanvasFocus(graph, destinationParent);
  ImVec2 centre{defaultNewBoxPosition.x, defaultNewBoxPosition.y};
  if (const auto *node = graph.findNode(boxId))
    centre = ImVec2(node->position.x, node->position.y);
  else if (const auto *group = graph.findGroup(boxId))
    centre = ImVec2(group->position.x, group->position.y);
  pendingCentreView = true;
  pendingCentrePoint = centre;
}

void NodeRenderer::handleStructureDropTarget(
    NodeGraph &graph, const NodeRendererCallbacks &callbacks,
    std::optional<std::int32_t> targetParent, ImVec2 highlightMin,
    ImVec2 highlightMax) {
  if (!ImGui::BeginDragDropTarget())
    return;
  const auto peekFlags = ImGuiDragDropFlags_AcceptBeforeDelivery |
                         ImGuiDragDropFlags_AcceptNoDrawDefaultRect;
  const auto accept = [&](const char *type) {
    return ImGui::AcceptDragDropPayload(type, peekFlags);
  };

  const auto *structure = accept(structureBoxPayloadId);
  const auto *palette = accept("OPENYOURBOX_NODE_TYPE");
  const auto *library = accept(openyourbox::ui::boxLibraryPayloadId);
  if (structure == nullptr && palette == nullptr && library == nullptr) {
    ImGui::EndDragDropTarget();
    return;
  }

  bool valid = true;
  std::int32_t sourceId = 0;
  if (structure != nullptr) {
    sourceId = *static_cast<const std::int32_t *>(structure->Data);
    if (sourceId == 0)
      valid = false;
    if (targetParent.has_value() && *targetParent == sourceId)
      valid = false;
    if (targetParent.has_value() && graph.findGroup(sourceId) != nullptr &&
        graph.findGroup(*targetParent) != nullptr) {
      std::function<bool(std::int32_t, std::int32_t)> owns =
          [&](std::int32_t groupId, std::int32_t id) -> bool {
        const auto *group = graph.findGroup(groupId);
        if (group == nullptr)
          return false;
        for (const auto member : group->memberIds) {
          if (member == id)
            return true;
          if (graph.findGroup(member) != nullptr && owns(member, id))
            return true;
        }
        return false;
      };
      if (owns(sourceId, *targetParent))
        valid = false;
    }
    if (targetParent.has_value() && graph.findGroup(*targetParent) == nullptr)
      valid = false;
  } else if (targetParent.has_value() &&
             graph.findGroup(*targetParent) == nullptr)
    valid = false;

  structureDropTargetParent = targetParent;
  structureDropValid = valid;
  structureDropHighlightActive = true;
  structureDropHighlightMin = highlightMin;
  structureDropHighlightMax = highlightMax;

  const auto delivered = (structure != nullptr && structure->IsDelivery()) ||
                         (palette != nullptr && palette->IsDelivery()) ||
                         (library != nullptr && library->IsDelivery());
  if (delivered && !valid) {
    structureDropHighlightActive = false;
    ImGui::EndDragDropTarget();
    return;
  }
  if (!delivered) {
    ImGui::EndDragDropTarget();
    return;
  }

  if (structure != nullptr) {
    std::optional<std::int32_t> currentParent;
    if (const auto *node = graph.findNode(sourceId))
      currentParent = node->parentGroupId;
    else if (const auto *group = graph.findGroup(sourceId))
      currentParent = group->parentGroupId;
    if (currentParent == targetParent) {
      structureDropHighlightActive = false;
      ImGui::EndDragDropTarget();
      return;
    }
    const auto result = graph.reparentBoxLikeInsert(sourceId, targetParent);
    if (result.accepted) {
      mutatedThisFrame = true;
      recompileThisFrame = true;
      focusAfterStructureMutation(graph, sourceId, targetParent);
    } else if (!result.message.empty()) {
      transientMessage = result.message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (callbacks.showMessage)
        callbacks.showMessage(result.message);
    }
  } else if (palette != nullptr) {
    const auto item = *static_cast<const PaletteItem *>(palette->Data);
    const auto id =
        placePaletteItem(graph, item, defaultNewBoxPosition, targetParent);
    positionedNodeIds.erase(id);
    mutatedThisFrame = true;
    recompileThisFrame = true;
    focusAfterStructureMutation(graph, id, targetParent);
  } else if (library != nullptr && activeBoxLibrary != nullptr) {
    const auto *drop =
        static_cast<const openyourbox::ui::BoxLibraryDropPayload *>(
            library->Data);
    juce::String error;
    const auto rootId = activeBoxLibrary->insertBox(
        graph, juce::String::fromUTF8(drop->entryId), defaultNewBoxPosition,
        error, drop->nestedRootId);
    if (rootId.has_value()) {
      positionedNodeIds.erase(*rootId);
      positionedGroupIds.erase(*rootId);
      if (targetParent.has_value()) {
        const auto parented = graph.addToGroup(*targetParent, *rootId, true);
        if (!parented.accepted && !parented.message.empty()) {
          transientMessage = parented.message;
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(parented.message);
        }
      }
      mutatedThisFrame = true;
      recompileThisFrame = true;
      focusAfterStructureMutation(graph, *rootId, targetParent);
    } else if (error.isNotEmpty()) {
      transientMessage = error.toStdString();
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (callbacks.showMessage)
        callbacks.showMessage(error.toStdString());
    }
  }
  structureDropHighlightActive = false;
  ImGui::EndDragDropTarget();
}

void NodeRenderer::renderParametersPanel(
    NodeGraph &graph, const NodeRendererCallbacks &callbacks,
    openyourbox::library::UserBoxLibrary *boxLibrary) {
  activeCallbacks = &callbacks;
  activeBoxLibrary = boxLibrary;
  propertyLayoutWidth =
      std::max(180.0f, ImGui::GetContentRegionAvail().x);
  propertyFieldWidth = std::clamp(propertyLayoutWidth * 0.42f, 72.0f, 160.0f);

  if (selectionContext.kind == SelectionContextKind::None) {
    ImGui::TextDisabled("Select a box to edit parameters.");
    flushDocumentCallbacks(callbacks);
    return;
  }
  if (selectionContext.kind == SelectionContextKind::Multi) {
    ImGui::TextWrapped(
        "Multiple boxes selected. Select a single box to edit parameters.");
    flushDocumentCallbacks(callbacks);
    return;
  }
  if (selectionContext.kind == SelectionContextKind::LibraryInspect) {
    if (boxLibrary != nullptr)
      renderLibraryInspectParameters(*boxLibrary, callbacks);
    else
      ImGui::TextDisabled("Library catalog is unavailable.");
    flushDocumentCallbacks(callbacks);
    return;
  }

  if (!selectionContext.liveBoxId.has_value()) {
    ImGui::TextDisabled("Select a box to edit parameters.");
    flushDocumentCallbacks(callbacks);
    return;
  }
  const auto boxId = *selectionContext.liveBoxId;
  if (auto *node = graph.findNode(boxId)) {
    ImGui::TextUnformatted(node->label.c_str());
    if (!node->detail.empty())
      ImGui::TextDisabled("%s", node->detail.c_str());
    ImGui::Separator();
    renderNodeParameterEditors(graph, *node, callbacks, false);
  } else if (auto *group = graph.findGroup(boxId)) {
    ImGui::TextUnformatted(group->name.c_str());
    ImGui::Separator();
    renderGroupParameterEditors(graph, *group, callbacks, false);
  } else {
    selectionContext.kind = SelectionContextKind::None;
    selectionContext.liveBoxId.reset();
    ImGui::TextDisabled("Select a box to edit parameters.");
  }
  flushDocumentCallbacks(callbacks);
}

void NodeRenderer::renderLibraryInspectParameters(
    openyourbox::library::UserBoxLibrary &boxLibrary,
    const NodeRendererCallbacks &callbacks) {
  (void)callbacks;
  ImGui::TextUnformatted("Library preview");
  ImGui::TextDisabled("Read-only — catalog files are not modified.");
  ImGui::Separator();
  juce::String error;
  const auto entryId = juce::String(selectionContext.libraryEntryId);
  if (!libraryInspectSnapshot.isValid() ||
      static_cast<std::int32_t>(libraryInspectSnapshot.getProperty(
          "inspectedNested", -1)) != selectionContext.libraryNestedRootId ||
      libraryInspectSnapshot.getProperty("inspectedEntry", "").toString() !=
          entryId) {
    libraryInspectSnapshot = boxLibrary.loadEntrySnapshot(entryId, error);
    if (libraryInspectSnapshot.isValid()) {
      libraryInspectSnapshot.setProperty("inspectedEntry", entryId, nullptr);
      libraryInspectSnapshot.setProperty(
          "inspectedNested", selectionContext.libraryNestedRootId, nullptr);
    }
  }
  if (!libraryInspectSnapshot.isValid()) {
    ImGui::TextWrapped("%s",
                       error.isNotEmpty() ? error.toRawUTF8()
                                          : "Unable to load library snapshot.");
    return;
  }
  const auto nested = selectionContext.libraryNestedRootId;
  const auto rootId =
      nested != 0 ? nested
                  : static_cast<std::int32_t>(libraryInspectSnapshot["rootId"]);
  juce::ValueTree target;
  for (const auto child : libraryInspectSnapshot) {
    if ((child.hasType("Node") || child.hasType("Group")) &&
        static_cast<std::int32_t>(child["id"]) == rootId) {
      target = child;
      break;
    }
  }
  if (!target.isValid()) {
    ImGui::TextDisabled("Snapshot member is unavailable.");
    return;
  }
  ImGui::BeginDisabled();
  if (target.hasType("Group")) {
    ImGui::TextUnformatted(
        target.getProperty("name", "Group").toString().toRawUTF8());
    ImGui::Text("Repeats: %d",
                static_cast<int>(target.getProperty("repeats", 1)));
  } else {
    ImGui::TextUnformatted(target.getProperty("label", "Element")
                               .toString()
                               .toRawUTF8());
    for (const auto child : target) {
      if (!child.hasType("Property"))
        continue;
      ImGui::PushID(child["key"].toString().toRawUTF8());
      beginPropertyRow(child.getProperty("label", child["key"])
                           .toString()
                           .toRawUTF8());
      const auto kind = static_cast<int>(child.getProperty("kind", 0));
      if (kind == static_cast<int>(PropertyKind::real)) {
        auto value = static_cast<float>(child.getProperty("floatValue", 0.0f));
        ImGui::InputFloat("##inspectFloat", &value, 0.0f, 0.0f, "%.2f");
      } else if (kind == static_cast<int>(PropertyKind::string)) {
        auto text = child.getProperty("stringValue", "").toString();
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%s", text.toRawUTF8());
        ImGui::InputText("##inspectString", buffer, sizeof(buffer));
      } else {
        auto value = static_cast<int>(child.getProperty("value", 0));
        ImGui::InputInt("##inspectInt", &value, 0, 0);
      }
      ImGui::PopID();
    }
  }
  ImGui::EndDisabled();
}

void NodeRenderer::renderGroupParameterEditors(
    NodeGraph &graph, GraphGroup &group,
    const NodeRendererCallbacks &callbacks, bool readOnly) {
  if (readOnly)
    ImGui::BeginDisabled();
  auto &nameBuffer = groupNameBuffers[group.id];
  if (nameBuffer[0] == '\0') {
    const auto nameByteCount =
        std::min(group.name.size(), nameBuffer.size() - 1);
    std::memcpy(nameBuffer.data(), group.name.data(), nameByteCount);
    nameBuffer[nameByteCount] = '\0';
  }
  beginPropertyRow("Name");
  if (ImGui::InputText("##groupName", nameBuffer.data(), nameBuffer.size())) {
    if (!readOnly && graph.renameGroup(group.id, nameBuffer.data()))
      mutatedThisFrame = true;
  }
  int repeats = group.repeats;
  beginPropertyRow("Repeats");
  if (ImGui::InputInt("##repeats", &repeats, 0, 0) && !readOnly) {
    const auto result = graph.setGroupRepeats(group.id, repeats);
    if (result.accepted) {
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
    if (!result.message.empty()) {
      transientMessage = result.message;
      transientMessageDeadline = ImGui::GetTime() + 2.5;
      if (callbacks.showMessage)
        callbacks.showMessage(result.message);
    }
  }
  const auto repeatStatus = graph.groupRepeatStatus(group.id);
  if (!repeatStatus.active) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.18f, 1.0f));
    ImGui::TextWrapped("Requested %d; effective 1",
                       repeatStatus.requestedRepeats);
    ImGui::TextWrapped("%s", repeatStatus.message.c_str());
    ImGui::PopStyleColor();
  }
  if (ImGui::Button("Randomize Weights", ImVec2(-FLT_MIN, 0.0f)) && !readOnly) {
    if (graph.randomizeGroupWeights(group.id)) {
      for (const auto leaf : graph.collectLeafNodeIds(group.id)) {
        if (const auto *node = graph.findNode(leaf)) {
          auto &seedBuffer = seedBuffers[leaf];
          std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", node->seed);
        }
      }
      mutatedThisFrame = true;
      recompileThisFrame = true;
    }
  }
  if (readOnly)
    ImGui::EndDisabled();
}

void NodeRenderer::renderNodeParameterEditors(
    NodeGraph &graph, GraphNode &node, const NodeRendererCallbacks &callbacks,
    bool readOnly) {
  if (readOnly)
    ImGui::BeginDisabled();
  if (node.type == NodeType::knobInput)
    renderKnobControl(graph, node, callbacks);
  else if (node.type == NodeType::xyTrackpad)
    renderXyPad(graph, node, callbacks);

  int convStride = 1;
  if (node.type == NodeType::convolution ||
      node.type == NodeType::convTranspose) {
    for (const auto &property : node.properties) {
      if (property.key == "stride")
        convStride = property.value;
    }
  }
  const bool stridedConv =
      (node.type == NodeType::convolution ||
       node.type == NodeType::convTranspose) &&
      convStride > 1;
  if (isRaveProcessingType(node.type) || stridedConv ||
      (node.type == NodeType::blackBox && node.outputs.size() > 1)) {
    const auto delay = nodeDelaySamples(node);
    if (delay > 0 || node.type == NodeType::blackBox) {
      const auto samples =
          node.metrics.has_value()
              ? static_cast<double>(std::max<std::uint64_t>(delay, 1))
              : static_cast<double>(delay);
      ImGui::TextDisabled("Delay %.0f smp / %.2f ms @ 48 kHz", samples,
                          samples * 1000.0 / 48000.0);
    }
  }

  if (isExternalLoadNode(node)) {
    const char *statusLabel = "Empty";
    switch (node.externalLoadStatus) {
    case ExternalLoadStatus::loading:
      statusLabel = "Loading";
      break;
    case ExternalLoadStatus::ready:
      statusLabel = "Ready";
      break;
    case ExternalLoadStatus::error:
      statusLabel = "Error";
      break;
    case ExternalLoadStatus::empty:
      break;
    }
    ImGui::TextWrapped("Status: %s", statusLabel);
    if (node.externalLoadStatus == ExternalLoadStatus::empty)
      ImGui::TextDisabled("Choose a TorchScript checkpoint to process audio.");
    if (!node.artifactPath.empty())
      ImGui::TextWrapped("Path: %s",
                         juce::File(node.artifactPath).getFullPathName().toRawUTF8());
    else
      ImGui::TextDisabled("Path: (none)");
    if (!node.externalLoadErrorMessage.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.35f, 1.0f));
      ImGui::TextWrapped("%s", node.externalLoadErrorMessage.c_str());
      ImGui::PopStyleColor();
    }
    if (!node.sampleRateWarning.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.72f, 0.18f, 1.0f));
      ImGui::TextWrapped("%s", node.sampleRateWarning.c_str());
      ImGui::PopStyleColor();
    }
    if (ImGui::SmallButton("Browse") && callbacks.browseWeights && !readOnly)
      callbacks.browseWeights(node.id);
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear") && !readOnly) {
      const auto clearedPath = node.artifactPath.empty() ? node.runtimeArtifactPath
                                                         : node.artifactPath;
      graph.clearExternalCheckpoint(node.id);
      mutatedThisFrame = true;
      recompileThisFrame = true;
      if (callbacks.externalCheckpointCleared)
        callbacks.externalCheckpointCleared(clearedPath);
    }
    if (node.externalLoadStatus == ExternalLoadStatus::ready ||
        node.inferredInputChannels > 0 || node.overrideInputChannels > 0) {
      ImGui::Separator();
      ImGui::TextDisabled("Inferred in/out/latent: %d / %d / %d",
                          node.inferredInputChannels, node.inferredOutputChannels,
                          node.inferredLatentChannels);
      auto overrideRow = [&](const char *label, const char *key, int current) {
        int value = current > 0 ? current : 0;
        ImGui::SetNextItemWidth(80.0f);
        if (ImGui::InputInt(label, &value) && !readOnly) {
          if (!graph.setExternalChannelOverride(node.id, key, value)) {
            transientMessage = graph.lastPropertyMessage();
            transientMessageDeadline = ImGui::GetTime() + 2.5;
            if (callbacks.showMessage && !transientMessage.empty())
              callbacks.showMessage(transientMessage);
          } else {
            mutatedThisFrame = true;
            recompileThisFrame = true;
          }
        }
      };
      overrideRow("Input ch##extIn", "input",
                  effectiveInputChannels(node));
      overrideRow("Output ch##extOut", "output",
                  effectiveOutputChannels(node));
      if (node.externalHasEncodeDecode)
        overrideRow("Latent ch##extLatent", "latent",
                    effectiveLatentChannels(node));
      if (ImGui::SmallButton("Reset to inferred") && !readOnly) {
        graph.resetExternalChannelOverrides(node.id);
        mutatedThisFrame = true;
        recompileThisFrame = true;
      }
    }
    if (node.externalHasEncodeDecode && !node.compactnessReady)
      ImGui::TextDisabled("Fidelity needs compactness buffers on this checkpoint.");
    if (node.externalShapeIncomplete)
      ImGui::TextDisabled("Enter channel overrides before connections are legal.");
  }

  const auto frozen = node.state == NodeState::frozenGold;
  if (frozen)
    ImGui::BeginDisabled();
  for (auto &property : node.properties) {
    const bool fidelityLocked =
        property.key == "fidelity" && !node.compactnessReady &&
        (node.type == NodeType::variationalBottleneck ||
         node.type == NodeType::blackBox);
    const bool liveOnGold =
        frozen && property.key == "fidelity" && !fidelityLocked;
    if (liveOnGold)
      ImGui::EndDisabled();
    if (!frozen && fidelityLocked)
      ImGui::BeginDisabled();
    const bool hideNegativeSlope = property.key == "negative_slope" && [&node]() {
      for (const auto &candidate : node.properties) {
        if (candidate.key == "activation")
          return candidate.value != leakyReluActivationIndex;
      }
      return true;
    }();
    if (hideNegativeSlope) {
      if (!frozen && fidelityLocked)
        ImGui::EndDisabled();
      if (liveOnGold)
        ImGui::BeginDisabled();
      continue;
    }
    if (property.key == "reverb_length" && isDdspEffectType(node.type)) {
      const auto milliseconds =
          static_cast<double>(property.value) * 1000.0 / 48000.0;
      if (milliseconds > liveSafeIrLengthMilliseconds) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.2f, 1.0f));
        ImGui::TextWrapped("Live-safe warning: this IR length exceeds one "
                           "second at 48 kHz.");
        ImGui::PopStyleColor();
      }
    }
    ImGui::PushID(property.key.c_str());
    if (const auto hint = graph.groupRepeatPropertyHint(node.id, property.key);
        hint.has_value()) {
      ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.18f, 1.0f), "! Repeat shape");
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(hint->c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
      }
    }
    auto value = property.value;
    bool changed = false;
    int dragSteps = 0;
    if (isBooleanPropertyKey(property.key)) {
      beginPropertyRow(property.label.c_str());
      bool enabled = property.value != 0;
      if (ImGui::Checkbox("##flag", &enabled) && !readOnly) {
        const auto previous = property.value;
        property.setValue(enabled ? 1 : 0);
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
    if (property.kind == PropertyKind::string) {
      beginRepeatListPropertyRow(property.label.c_str(), {});
      const auto bufferKey = std::to_string(node.id) + ":" + property.key;
      auto &buffer = propertyListBuffers[bufferKey];
      const auto widgetId = ImGui::GetID("##string");
      if (ImGui::GetActiveID() != widgetId) {
        std::snprintf(buffer.data(), buffer.size(), "%s",
                      property.stringValue.c_str());
      }
      ImGui::SetNextItemWidth(propertyLayoutWidth);
      ImGui::InputTextWithHint("##string", mathExpressionPlaceholder,
                               buffer.data(), buffer.size());
      if (ImGui::IsItemDeactivatedAfterEdit() && !readOnly) {
        if (!graph.setStringProperty(node.id, property.key, buffer.data())) {
          transientMessage = graph.lastPropertyMessage().empty()
                                 ? (property.label + " expression is invalid")
                                 : graph.lastPropertyMessage();
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(transientMessage);
          std::snprintf(buffer.data(), buffer.size(), "%s",
                        property.stringValue.c_str());
        } else {
          mutatedThisFrame = true;
          recompileThisFrame = true;
        }
      }
      ImGui::PopID();
      if (fidelityLocked)
        ImGui::TextDisabled("Compactness not ready");
      if (!frozen && fidelityLocked)
        ImGui::EndDisabled();
      if (liveOnGold)
        ImGui::BeginDisabled();
      continue;
    }
    const auto repeatCount = graph.effectiveRepeatCount(node.id);
    const auto ancestorCounts = graph.ancestorRepeatCounts(node.id);
    const auto listEdit = propertySupportsRepeatValueList(property);
    if (!listEdit)
      beginPropertyRow(property.label.c_str());
    if (listEdit) {
      constexpr float handleWidth = 18.0f;
      const auto expandedInfo =
          repeatCount > 1 ? formatExpandedPropertyRepeatList(
                                property, repeatCount, ancestorCounts)
                          : std::string{};
      beginRepeatListPropertyRow(property.label.c_str(), expandedInfo);
      const auto fieldHeight = ImGui::GetFrameHeight();
      const auto fieldOrigin = ImGui::GetCursorScreenPos();
      ImGui::GetWindowDrawList()->AddRectFilled(
          fieldOrigin,
          ImVec2(fieldOrigin.x + propertyLayoutWidth,
                 fieldOrigin.y + fieldHeight),
          ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);
      dragSteps = propertyDragSteps("##drag", ImVec2(handleWidth, fieldHeight));
      if (ImGui::IsItemActive()) {
        patchGestureHeldThisFrame = true;
        patchGestureLabel = "Parameter edit";
      }
      if (dragSteps != 0 && !readOnly) {
        if (property.kind == PropertyKind::real) {
          const auto span = property.floatMaximum - property.floatMinimum;
          const auto stepPerTick =
              ImGui::GetIO().KeyShift ? span / 2000.0f : span / 200.0f;
          auto values = property.repeatFloatValues;
          if (values.empty())
            values.push_back(property.floatValue);
          for (auto &item : values)
            item += static_cast<float>(dragSteps) * stepPerTick;
          if (graph.setFloatPropertyRepeatValues(node.id, property.key,
                                                 values)) {
            mutatedThisFrame = true;
            recompileThisFrame = true;
            if (callbacks.floatPropertyChanged)
              callbacks.floatPropertyChanged(node.id, property.key,
                                             property.floatValue);
          }
        } else if (!property.preserveInBound) {
          auto values = property.repeatIntValues;
          if (values.empty())
            values.push_back(property.value);
          for (auto &item : values)
            item += dragSteps;
          if (graph.setPropertyRepeatValues(node.id, property.key, values)) {
            mutatedThisFrame = true;
            recompileThisFrame = true;
            if (callbacks.propertyChanged)
              callbacks.propertyChanged(node.id, property.key, property.value);
          }
        }
      }
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
      ImGui::SetNextItemWidth(propertyLayoutWidth - handleWidth);
      const auto bufferKey = std::to_string(node.id) + ":" + property.key;
      auto &buffer = propertyListBuffers[bufferKey];
      const auto widgetId = ImGui::GetID("##repeatList");
      if (ImGui::GetActiveID() != widgetId) {
        const auto formatted = formatAuthoredPropertyRepeatList(property);
        std::snprintf(buffer.data(), buffer.size(), "%s", formatted.c_str());
      }
      ImGui::InputTextWithHint("##repeatList", parameterExpressionPlaceholder,
                               buffer.data(), buffer.size());
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
      if (ImGui::IsItemDeactivatedAfterEdit() && !readOnly) {
        const auto parsed = parsePropertyRepeatList(
            property, ancestorCounts, buffer.data(),
            propertySupportsPreserveIn(property));
        if (!parsed.accepted) {
          transientMessage = parsed.message;
          transientMessageDeadline = ImGui::GetTime() + 2.5;
          if (callbacks.showMessage)
            callbacks.showMessage(transientMessage);
        } else if (parsed.preserveIn) {
          if (graph.setPropertyPreserveIn(
                  node.id, property.key,
                  static_cast<int>(parsed.intValues.size()))) {
            mutatedThisFrame = true;
            recompileThisFrame = true;
            if (callbacks.propertyChanged)
              callbacks.propertyChanged(node.id, property.key, property.value);
          }
        } else if (property.kind == PropertyKind::real) {
          if (graph.setFloatPropertyRepeatValues(node.id, property.key,
                                                 parsed.floatValues,
                                                 parsed.authoredTokens)) {
            mutatedThisFrame = true;
            recompileThisFrame = true;
            if (callbacks.floatPropertyChanged)
              callbacks.floatPropertyChanged(node.id, property.key,
                                             property.floatValue);
          }
        } else if (graph.setPropertyRepeatValues(node.id, property.key,
                                                 parsed.intValues,
                                                 parsed.authoredTokens)) {
          mutatedThisFrame = true;
          recompileThisFrame = true;
          if (callbacks.propertyChanged)
            callbacks.propertyChanged(node.id, property.key, property.value);
        }
      }
      if (property.repeatListInvalid &&
          !property.repeatListInvalidMessage.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.38f, 0.38f, 1.0f));
        ImGui::TextWrapped("%s", property.repeatListInvalidMessage.c_str());
        ImGui::PopStyleColor();
      }
      ImGui::PopID();
      if (fidelityLocked)
        ImGui::TextDisabled("Compactness not ready");
      if (!frozen && fidelityLocked)
        ImGui::EndDisabled();
      if (liveOnGold)
        ImGui::BeginDisabled();
      continue;
    }
    if (property.kind == PropertyKind::choice && !property.choices.empty()) {
      const auto choiceIndex = std::clamp(
          value, 0, static_cast<int>(property.choices.size()) - 1);
      const auto *preview =
          property.choices[static_cast<std::size_t>(choiceIndex)].c_str();
      if (ImGui::BeginCombo("##choice", preview)) {
        for (int index = 0; index < static_cast<int>(property.choices.size());
             ++index) {
          const auto selected = property.value == index;
          if (ImGui::Selectable(
                  property.choices[static_cast<std::size_t>(index)].c_str(),
                  selected) &&
              !readOnly) {
            const auto previous = property.value;
            property.setValue(index);
            if (!graph.setProperty(node.id, property.key, property.value))
              property.setValue(previous);
            else {
              mutatedThisFrame = true;
              recompileThisFrame = true;
              if (callbacks.propertyChanged)
                callbacks.propertyChanged(node.id, property.key,
                                          property.value);
            }
          }
        }
        ImGui::EndCombo();
      }
    } else if (property.kind == PropertyKind::real) {
      constexpr float handleWidth = 18.0f;
      const auto fieldHeight = ImGui::GetFrameHeight();
      const auto fieldOrigin = ImGui::GetCursorScreenPos();
      ImGui::GetWindowDrawList()->AddRectFilled(
          fieldOrigin,
          ImVec2(fieldOrigin.x + propertyFieldWidth,
                 fieldOrigin.y + fieldHeight),
          ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);
      auto floatValue = property.floatValue;
      dragSteps = propertyDragSteps("##drag", ImVec2(handleWidth, fieldHeight));
      if (ImGui::IsItemActive()) {
        patchGestureHeldThisFrame = true;
        patchGestureLabel = "Parameter edit";
      }
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
      ImGui::SetNextItemWidth(propertyFieldWidth - handleWidth);
      changed =
          ImGui::InputFloat("##gain", &floatValue, 0.0f, 0.0f, "%.2f") ||
          changed;
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
      if (changed && !readOnly) {
        const auto previous = property.floatValue;
        property.setFloatValue(floatValue);
        if (std::abs(property.floatValue - previous) > 1.0e-6f) {
          if (!graph.setFloatProperty(node.id, property.key,
                                      property.floatValue))
            property.setFloatValue(previous);
          else {
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
          ImVec2(fieldOrigin.x + propertyFieldWidth,
                 fieldOrigin.y + fieldHeight),
          ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);
      dragSteps = propertyDragSteps("##drag", ImVec2(handleWidth, fieldHeight));
      if (ImGui::IsItemActive()) {
        patchGestureHeldThisFrame = true;
        patchGestureLabel = "Parameter edit";
      }
      if (dragSteps != 0)
        value += dragSteps;
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
      ImGui::SetNextItemWidth(propertyFieldWidth - handleWidth);
      changed = ImGui::InputInt("##value", &value, 0, 0) || dragSteps != 0;
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
    }
    if (changed && property.kind != PropertyKind::real && !readOnly) {
      const auto previous = property.value;
      property.setValue(value);
      if (property.value != previous) {
        if (!graph.setProperty(node.id, property.key, property.value)) {
          property.setValue(previous);
          transientMessage = graph.lastPropertyMessage().empty()
                                 ? "That mode would break downstream channel compatibility"
                                 : graph.lastPropertyMessage();
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
    if (fidelityLocked)
      ImGui::TextDisabled("Compactness not ready");
    if (!frozen && fidelityLocked)
      ImGui::EndDisabled();
    if (liveOnGold)
      ImGui::BeginDisabled();
  }
  if (frozen)
    ImGui::EndDisabled();

  if (node.type == NodeType::convolution ||
      node.type == NodeType::convTranspose) {
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
    const auto isBatchNorm = node.type == NodeType::batchNorm;
    if (!isBatchNorm) {
      const auto insertion = seedBuffers.try_emplace(node.id);
      auto &seedBuffer = insertion.first->second;
      if (insertion.second)
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d",
                      node.useExplicitSeed ? node.explicitSeed : node.seed);
      if (!node.useExplicitSeed)
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", node.seed);
      beginPropertyRow("Seed");
      if (ImGui::Checkbox("##useExplicitSeed", &node.useExplicitSeed) &&
          !readOnly)
        mutatedThisFrame = true;
      ImGui::SameLine();
      ImGui::SetNextItemWidth(propertyFieldWidth - ImGui::GetFrameHeight() -
                              8.0f);
      if (!node.useExplicitSeed)
        ImGui::BeginDisabled();
      if (ImGui::InputText("##seed", seedBuffer.data(), seedBuffer.size(),
                           ImGuiInputTextFlags_EnterReturnsTrue) &&
          !readOnly) {
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
        }
      }
      if (!node.useExplicitSeed)
        ImGui::EndDisabled();
    }
    const char *weightsActionLabel = isBatchNorm ? "Reset" : "Randomize Weights";
    if (ImGui::Button(weightsActionLabel, ImVec2(-FLT_MIN, 0.0f)) &&
        callbacks.randomizeNode && !readOnly) {
      std::int32_t appliedSeed = node.seed;
      if (isBatchNorm) {
        appliedSeed = 0;
        graph.setSeed(node.id, appliedSeed);
        node.seed = appliedSeed;
      } else if (node.useExplicitSeed)
        appliedSeed = node.explicitSeed;
      else {
        auto &seedBuffer = seedBuffers[node.id];
        appliedSeed = juce::Random::getSystemRandom().nextInt(maximumSeed + 1);
        graph.setSeed(node.id, appliedSeed);
        node.seed = appliedSeed;
        std::snprintf(seedBuffer.data(), seedBuffer.size(), "%d", appliedSeed);
      }
      mutatedThisFrame = true;
      if (graph.effectiveRepeatCount(node.id) > 1)
        recompileThisFrame = true;
      callbacks.randomizeNode(node.id, appliedSeed);
    }
  }

  if (node.hasWeights && node.state == NodeState::liveBlue &&
      !isControlSourceType(node.type)) {
    bool armed = node.armedForTraining;
    if (ImGui::Checkbox("Arm for training", &armed) && !readOnly) {
      graph.setArmedForTraining(node.id, armed);
      mutatedThisFrame = true;
      if (callbacks.armChanged)
        callbacks.armChanged(node.id, armed);
    }
  }
  if ((node.hasWeights || node.type == NodeType::blackBox) &&
      !isExternalLoadNode(node)) {
    if (node.weightsProvenance == WeightsProvenance::file &&
        !node.weightsPath.empty())
      ImGui::TextWrapped("Weights: %s",
                         juce::File(node.weightsPath).getFileName().toRawUTF8());
    else if (node.type == NodeType::batchNorm)
      ImGui::TextUnformatted("Weights: identity");
    else
      ImGui::Text("Weights: seed %d", node.seed);
    if (ImGui::SmallButton("Browse weights") && callbacks.browseWeights &&
        !readOnly)
      callbacks.browseWeights(node.id);
  }
  if (node.metrics.has_value()) {
    ImGui::Separator();
    ImGui::Text("Compile: %.1f ms", node.metrics->compileTimeMilliseconds);
    ImGui::Text("Inference: %.3f ms", node.metrics->inferenceTimeMilliseconds);
  }
  if (!isGroupBoundaryType(node.type) && ImGui::SmallButton("Analyze") &&
      callbacks.analysisRequested && !readOnly)
    callbacks.analysisRequested(node.id);
  if (readOnly)
    ImGui::EndDisabled();
}
} // namespace openyourbox::graph
