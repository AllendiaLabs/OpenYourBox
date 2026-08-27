#pragma once

#include "../dsp/TCNModel.h"
#include "GraphTypes.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace openyourbox::graph {
/**
 * @class NodeGraph
 * @brief Editable, serializable graph document owned by the message thread.
 *
 * Stable node, pin, and link identifiers keep the Dear ImGui editor and saved
 * plug-in state synchronized without exposing mutable graph data to the audio
 * thread.
 */
class NodeGraph {
public:
  /**
   * @brief Replaces the graph with a linear TCN signal-flow projection.
   * @param configuration Architecture represented by the graph.
   */
  void rebuildFromModel(const dsp::TCNConfiguration &configuration);

  /**
   * @brief Adds one element with its default ports and properties.
   * @param type Element type to create.
   * @param position Initial position in the destination canvas coordinates.
   * @param parentGroupId Group that owns the new node, or empty for the root canvas.
   * @return Stable identifier of the new node.
   */
  std::int32_t addNode(NodeType type, juce::Point<float> position,
                       std::optional<std::int32_t> parentGroupId = std::nullopt);

  /**
   * @brief Ensures the graph has exactly one undeletable host input and output.
   *
   * Channel mode is Mono, Mirrored, or Stereo from each node's Mode property
   * (default Stereo). Existing modes are preserved.
   */
  void ensureFixedHostIo();

  /**
   * @brief Removes a node and every connection attached to it.
   * @param nodeId Stable node identifier.
   * @return True when a deletable node was removed.
   */
  bool removeNode(std::int32_t nodeId);

  /**
   * @brief Splits a connection and inserts a processing element on that cable.
   * @param linkId Existing connection to replace.
   * @param type Processing element to insert.
   * @param position Canvas position for the new element.
   * @return Identifier of the inserted node, or no value when insertion failed.
   */
  std::optional<std::int32_t> insertNodeOnLink(std::int32_t linkId, NodeType type,
                                               juce::Point<float> position);

  /**
   * @brief Adds a processing element onto an unconnected input or output pin.
   * @param pinId Unconnected pin that receives the new cable.
   * @param type Processing element to create.
   * @param position Canvas position for the new element.
   * @return Identifier of the added node, or no value when attachment failed.
   */
  std::optional<std::int32_t> attachNodeToPin(std::int32_t pinId, NodeType type,
                                              juce::Point<float> position);

  /**
   * @brief Stores a node position reported by the editor.
   * @param nodeId Stable node identifier.
   * @param position New canvas position.
   */
  void moveNode(std::int32_t nodeId, juce::Point<float> position);

  /**
   * @brief Creates a named group from selected nodes and/or nested groups.
   * @param memberIds Node ids and nested group ids to wrap.
   * @return Accepted group id, or a refusal reason.
   */
  GroupActionResult createGroup(const std::vector<std::int32_t> &memberIds);

  /**
   * @brief Lifts members to the parent scope and removes the group container.
   * @param groupId Group to dissolve.
   */
  GroupActionResult ungroup(std::int32_t groupId);

  /**
   * @brief Deletes a group and every descendant node and nested group.
   * @param groupId Group to remove.
   */
  GroupActionResult deleteGroup(std::int32_t groupId);

  /**
   * @brief Renames an existing group.
   * @param groupId Target group.
   * @param name New display name.
   */
  bool renameGroup(std::int32_t groupId, const std::string &name);

  /**
   * @brief Adds a node or nested group to @p groupId when the move is legal.
   * @param groupId Destination group.
   * @param memberId Node id or nested group id.
   * @param preserveStoredPosition True when @p memberId is already in the
   *        destination group's local coordinates.
   */
  GroupActionResult addToGroup(std::int32_t groupId, std::int32_t memberId,
                               bool preserveStoredPosition = false);

  /**
   * @brief Removes a node or nested group from its parent group.
   * @param memberId Node id or nested group id.
   */
  GroupActionResult removeFromGroup(std::int32_t memberId);

  /**
   * @brief Sets presentation-only collapse state.
   * @param groupId Target group.
   * @param collapsed True to hide interiors.
   */
  bool setGroupCollapsed(std::int32_t groupId, bool collapsed);

  /**
   * @brief Toggles presentation-only collapse state.
   * @param groupId Target group.
   */
  bool toggleGroupCollapsed(std::int32_t groupId);

  /**
   * @brief Sets independent serial copy count without drawing extra UI nodes.
   * @param groupId Target group.
   * @param copies Requested N ≥ 1.
   */
  GroupActionResult setGroupCopies(std::int32_t groupId, int copies);

  /**
   * @brief Randomizes live weighted leaves of a group, resetting BatchNorm.
   *
   * Members with the seed checkbox enabled keep `explicitSeed` as the base.
   * Other weighted members draw a new base seed. Extra copy slots use
   * `seed + i`. BatchNorm members are restored to identity affine parameters.
   * Frozen Gold members are left unchanged. These seeds drive live audition and
   * freeze only; training uses PyTorch default initialization.
   * @param groupId Target group.
   * @return True when at least one live weighted member was updated.
   */
  bool randomizeGroupWeights(std::int32_t groupId);

  /**
   * @brief Stores expanded-frame bounds reported by the editor.
   * @param groupId Target group.
   * @param position New origin.
   * @param size New size.
   */
  void setGroupBounds(std::int32_t groupId, juce::Point<float> position,
                      juce::Point<float> size);

  /**
   * @brief Moves a group's frame in its parent space.
   *
   * Member coordinates stay group-local; their canvas positions follow the
   * group's transform.
   * @param groupId Target group.
   * @param position New origin in parent-canvas or parent-group space.
   */
  void moveGroup(std::int32_t groupId, juce::Point<float> position);

  /**
   * @brief Sizes the group frame so every current member is visible.
   * @param groupId Target group.
   */
  void fitGroupToMembers(std::int32_t groupId);

  /**
   * @brief Sets a group's inner-canvas camera.
   * @param groupId Target group.
   * @param pan Camera origin in group-local content space.
   * @param zoom Camera zoom, clamped to the editor zoom range.
   */
  void setGroupView(std::int32_t groupId, juce::Point<float> pan, float zoom);

  /**
   * @brief Canvas-space origin of a group's frame.
   * @param groupId Target group.
   */
  [[nodiscard]] juce::Point<float>
  worldPositionOfGroup(std::int32_t groupId) const;

  /**
   * @brief Canvas-space origin of a node, applying ancestor group cameras.
   * @param nodeId Target node.
   */
  [[nodiscard]] juce::Point<float>
  worldPositionOfNode(std::int32_t nodeId) const;

  /**
   * @brief Converts a canvas point into group-local content coordinates.
   * @param groupId Target group.
   * @param worldPoint Point in editor canvas space.
   */
  [[nodiscard]] juce::Point<float>
  worldToGroupLocal(std::int32_t groupId, juce::Point<float> worldPoint) const;

  /**
   * @brief Converts a group-local content point into canvas space.
   * @param groupId Target group.
   * @param localPoint Point in the group's inner canvas.
   */
  [[nodiscard]] juce::Point<float>
  groupLocalToWorld(std::int32_t groupId, juce::Point<float> localPoint) const;

  /**
   * @brief Scale from one group's content-local units to editor canvas units.
   * @param groupId Target group.
   */
  [[nodiscard]] float groupContentToCanvasScale(std::int32_t groupId) const;

  /**
   * @brief Expands groups in a selection to freezable leaf node ids.
   * @param selectedIds Mixed node and group identifiers.
   */
  [[nodiscard]] std::vector<std::int32_t> expandSelectionToFreezableLeaves(
      const std::vector<std::int32_t> &selectedIds) const;

  /**
   * @brief Returns a compile-only graph with invisible copies unrolled in series.
   */
  [[nodiscard]] NodeGraph withInvisibleCopiesMaterialized() const;

  /**
   * @brief Same as @ref withInvisibleCopiesMaterialized, optionally recording
   *        which expanded node came from which original slot.
   * @param provenance Expanded node id → (original node id, absolute slot).
   */
  [[nodiscard]] NodeGraph withInvisibleCopiesMaterialized(
      std::unordered_map<std::int32_t, std::pair<std::int32_t, int>> *provenance)
      const;

  /** @brief Returns groups in stable insertion order. */
  [[nodiscard]] const std::vector<GraphGroup> &getGroups() const noexcept;

  /** @brief Returns mutable groups for message-thread rendering. */
  [[nodiscard]] std::vector<GraphGroup> &getGroups() noexcept;

  /** @brief Finds a group by stable identifier. */
  [[nodiscard]] GraphGroup *findGroup(std::int32_t groupId) noexcept;

  /** @brief Finds a group by stable identifier. */
  [[nodiscard]] const GraphGroup *findGroup(std::int32_t groupId) const noexcept;

  /**
   * @brief Returns true when a node is hidden by a collapsed ancestor.
   * @param nodeId Candidate node.
   */
  [[nodiscard]] bool isNodeHiddenByCollapse(std::int32_t nodeId) const;

  /**
   * @brief Returns true when a nested group is hidden by a collapsed ancestor.
   * @param groupId Candidate group.
   */
  [[nodiscard]] bool isGroupHiddenByCollapse(std::int32_t groupId) const;

  /**
   * @brief Returns true when a visible expanded group clips @p nodeId.
   * @param nodeId Candidate node.
   */
  [[nodiscard]] bool isNodeClippedByGroup(std::int32_t nodeId) const;

  /**
   * @brief Innermost expanded group whose frame contains @p canvasPoint.
   * @param canvasPoint Position in editor canvas space.
   */
  [[nodiscard]] std::optional<std::int32_t>
  findExpandedGroupAt(juce::Point<float> canvasPoint) const;

  /**
   * @brief Returns true when @p nodeId is a direct child of the focused canvas.
   * @param nodeId Candidate node.
   * @param focusedGroupId Focused group, or empty for the graph root.
   */
  [[nodiscard]] bool isNodeOnFocusedCanvas(
      std::int32_t nodeId,
      std::optional<std::int32_t> focusedGroupId) const;

  /**
   * @brief Returns true when @p groupId is a group box on the focused canvas.
   * @param groupId Candidate group.
   * @param focusedGroupId Focused group, or empty for the graph root.
   */
  [[nodiscard]] bool isGroupOnFocusedCanvas(
      std::int32_t groupId,
      std::optional<std::int32_t> focusedGroupId) const;

  /**
   * @brief Nested group box on the focused canvas that contains @p nodeId.
   * @param nodeId Candidate node.
   * @param focusedGroupId Focused group, or empty for the graph root.
   * @return Host group id, or empty when the node itself is on the canvas.
   */
  [[nodiscard]] std::optional<std::int32_t> focusedCanvasHostGroup(
      std::int32_t nodeId, std::optional<std::int32_t> focusedGroupId) const;

  /**
   * @brief Ancestor chain from the graph root to @p groupId, inclusive.
   * @param groupId Target group.
   */
  [[nodiscard]] std::vector<std::int32_t>
  groupAncestorChain(std::int32_t groupId) const;

  /**
   * @brief Currently attached cables that cross a group's membership boundary.
   * @param groupId Target group.
   */
  [[nodiscard]] std::vector<GroupBoundaryPort>
  groupBoundaryPorts(std::int32_t groupId) const;

  /**
   * @brief Explicit Group Input/Output pins that form the external interface.
   *
   * Legacy groups without hubs temporarily fall back to inferred unconnected
   * leaf pins while they are migrated during restore.
   * @param groupId Target group.
   */
  [[nodiscard]] std::vector<GroupBoundaryPort>
  groupInterfacePorts(std::int32_t groupId) const;

  /**
   * @brief Innermost group that is currently drawn and contains @p nodeId.
   * @param nodeId Candidate node.
   */
  [[nodiscard]] std::optional<std::int32_t>
  innermostVisibleGroupOf(std::int32_t nodeId) const;

  /**
   * @brief Leaf processing nodes owned by a group, including nested groups.
   *
   * Includes nodes listed in @c memberIds and nodes whose @c parentGroupId is
   * this group or a nested group, so materialized copy clones stay visible.
   * @param groupId Target group.
   */
  [[nodiscard]] std::vector<std::int32_t>
  collectLeafNodeIds(std::int32_t groupId) const;

  /**
   * @brief Returns @p boxIds sorted by information-flow rank, then name.
   *
   * Boxes are treated as siblings under @p scopeGroupId, or the root canvas
   * when @p scopeGroupId is empty. Rank is the longest path from sources in the
   * sibling-link DAG; feedback loops share a rank. Within a rank, order is
   * case-insensitive name, then id. Unconnected boxes sit with sources (rank
   * 0). Group Input/Output hubs and fixed I/O are omitted.
   * @param scopeGroupId Parent group, or empty for the graph root.
   * @param boxIds Candidate sibling node and group identifiers.
   */
  [[nodiscard]] std::vector<std::int32_t> orderSiblingBoxesByFlow(
      std::optional<std::int32_t> scopeGroupId,
      std::vector<std::int32_t> boxIds) const;

  /**
   * @brief Product of copies on groups that contain @p nodeId.
   * @param nodeId Candidate node.
   */
  [[nodiscard]] int effectiveCopyCount(std::int32_t nodeId) const;

  /**
   * @brief Product of currently chainable ancestor copy counts.
   * @param nodeId Candidate node.
   * @return Runtime copy product, using one for every inactive request.
   */
  [[nodiscard]] int effectiveRuntimeCopyCount(std::int32_t nodeId) const;

  /**
   * @brief Evaluates a group's requested serial copies without mutating it.
   * @param groupId Target group.
   * @return Effective count and actionable diagnostics.
   */
  [[nodiscard]] GroupCopyStatus groupCopyStatus(std::int32_t groupId) const;

  /**
   * @brief Returns the inactive-copy hint affecting one property, if any.
   * @param nodeId Element containing the property.
   * @param propertyKey Canonical property key.
   */
  [[nodiscard]] std::optional<std::string>
  groupCopyPropertyHint(std::int32_t nodeId,
                        const std::string &propertyKey) const;

  /**
   * @brief Ancestor copy counts from outermost group to the node's parent.
   * @param nodeId Candidate node.
   * @return Outer→inner vector C; empty when the node has no parent group.
   */
  [[nodiscard]] std::vector<int> ancestorCopyCounts(std::int32_t nodeId) const;

  /**
   * @brief Re-tiles or flags authored copy lists after a nest change.
   * @param nodeId Leaf node whose ancestor copies may have changed.
   */
  void validateAuthoredCopyLists(std::int32_t nodeId);

  /**
   * @brief Validates and commits a directed connection.
   * @param firstPinId First endpoint selected by the user.
   * @param secondPinId Second endpoint selected by the user.
   * @return Acceptance state and user-facing rejection reason.
   */
  ConnectionResult connect(std::int32_t firstPinId, std::int32_t secondPinId);

  /**
   * @brief Removes a stable graph link.
   * @param linkId Link identifier.
   * @return True when a link was removed.
   */
  bool removeLink(std::int32_t linkId);

  /**
   * @brief Updates one validated inline property.
   * @param nodeId Target node identifier.
   * @param key Canonical property key.
   * @param value Proposed integer value.
   * @return True when the property exists and was updated.
   */
  bool setProperty(std::int32_t nodeId, const std::string &key, int value);

  /**
   * @brief Updates one validated real inline property such as Gain.
   * @param nodeId Target node identifier.
   * @param key Canonical property key.
   * @param value Proposed real value.
   * @return True when the property exists and was updated.
   */
  bool setFloatProperty(std::int32_t nodeId, const std::string &key,
                        float value);

  /**
   * @brief Writes an authored integer list (length in the nest dividing set).
   * @param nodeId Target node identifier.
   * @param key Canonical property key.
   * @param values Authored integers of legal length L.
   * @return True when the property exists, is list-capable, and was updated.
   */
  bool setPropertyCopyValues(std::int32_t nodeId, const std::string &key,
                             const std::vector<int> &values);

  /**
   * @brief Binds an integer dim/channels/features field to the `in` keyword.
   * @param nodeId Target node identifier.
   * @param key Canonical property key.
   * @param authoredLength Number of `in` tokens (must be in D(C)).
   * @return True when the field supports binding and the length is legal.
   */
  bool setPropertyPreserveIn(std::int32_t nodeId, const std::string &key,
                             int authoredLength);

  /**
   * @brief Writes an authored real list (length in the nest dividing set).
   * @param nodeId Target node identifier.
   * @param key Canonical property key.
   * @param values Authored reals of legal length L.
   * @return True when the property exists, is list-capable, and was updated.
   */
  bool setFloatPropertyCopyValues(std::int32_t nodeId, const std::string &key,
                                  const std::vector<float> &values);

  /**
   * @brief Stores a Knob Input conditioning scalar.
   * @param nodeId Target Knob Input node.
   * @param value Proposed conditioning value.
   * @return True when the node is a Knob Input.
   */
  bool setConditioningValue(std::int32_t nodeId, float value);

  /**
   * @brief Stores XY Trackpad conditioning scalars.
   * @param nodeId Target XY Trackpad node.
   * @param x Proposed X conditioning value.
   * @param y Proposed Y conditioning value.
   * @return True when the node is an XY Trackpad.
   */
  bool setConditioningPad(std::int32_t nodeId, float x, float y);

  /**
   * @brief Stores the persisted analysis view preference for one node.
   * @param nodeId Target node identifier.
   * @param view Selected analysis view.
   * @return True when the node exists.
   */
  bool setSelectedAnalysisView(std::int32_t nodeId, AnalysisView view);

  /**
   * @brief Stores a deterministic signed seed on a weighted live node.
   * @param nodeId Target node identifier.
   * @param seed Seed clamped to `[minimumSeed, maximumSeed]`.
   * @return True when the target accepts randomization.
   */
  bool setSeed(std::int32_t nodeId, std::int32_t seed);

  /**
   * @brief Marks a connected live selection as frozen Gold without regrouping.
   * @param selectedNodeIds Stable identifiers selected for freezing.
   * @param result Successful worker result containing artifact metadata.
   * @return Identifier of one frozen node, or no value for an invalid selection.
   */
  std::optional<std::int32_t>
  freezeSelection(const std::vector<std::int32_t> &selectedNodeIds,
                  const FreezeSelectionResult &result);

  /**
   * @brief Restores frozen Gold elements to Live Blue, keeping cables.
   * @param nodeId Any frozen node in the group, or a train-origin BlackBox.
   * @return True when restoration succeeded.
   *
   * Train-origin BlackBoxes restore their source subgraph **and** the
   * boundary cables that were connected when training finished. Restored
   * weighted nodes keep the trained artifact path so audio still uses the
   * trained weights.
   */
  bool unfreeze(std::int32_t nodeId);

  /**
   * @brief Builds a worker request from a connected live selection.
   * @param selectedNodeIds Stable identifiers selected by the user.
   * @return Request DTO, or no value when selection is invalid.
   */
  [[nodiscard]] std::optional<FreezeSelectionRequest>
  createFreezeRequest(const std::vector<std::int32_t> &selectedNodeIds) const;

  /**
   * @brief Builds a train-worker JSON snapshot of armed trainable nodes.
   * @return Request DTO, or no value when fewer than one armed trainable node.
   */
  [[nodiscard]] std::optional<TrainJobRequest> createTrainRequest() const;

  /**
   * @brief Returns true when an armed path can run reconstruction Train.
   *
   * Requires a variational bottleneck on a live path that can decode back
   * to audio (optional matching PQMF synthesis).
   */
  [[nodiscard]] bool hasReconstructionTrainPath() const;

  /**
   * @brief User-facing reason when reconstruction Train is blocked.
   * @return Empty when `hasReconstructionTrainPath()` is true.
   */
  [[nodiscard]] std::string reconstructionGateMessage() const;

  /**
   * @brief Returns identifiers of currently armed trainable processing nodes.
   */
  [[nodiscard]] std::vector<std::int32_t> getArmedTrainableNodeIds() const;

  /**
   * @brief Sets the training-arm flag on a trainable node.
   * @param nodeId Target node identifier.
   * @param armed Whether the node is included in the next train snapshot.
   * @return True when the node accepts arm state.
   */
  bool setArmedForTraining(std::int32_t nodeId, bool armed);

  /**
   * @brief Stores file-backed Weights provenance on a weight-bearing node.
   * @param nodeId Target node identifier.
   * @param path Absolute or app-relative weight file path.
   * @return True when the node owns weights.
   */
  bool setWeightsPath(std::int32_t nodeId, const std::string &path);

  /**
   * @brief Restores seed provenance after randomization.
   * @param nodeId Target node identifier.
   * @param seed New randomization seed for the visible element (slot 0).
   * @return True when the node owns weights.
   *
   * Ensures copy-slot count matches enclosing group N. Non-BatchNorm members
   * derive `seed + i` on every slot so N&gt;1 copies stay distinct for live
   * audition/freeze. Training does not use these seeds for initialization.
   */
  bool clearWeightsToSeed(std::int32_t nodeId, std::int32_t seed);

  /**
   * @brief Replaces the armed trainable chain with a train-origin Gold BlackBox.
   * @param result Successful train artifact metadata.
   * @return Identifier of the new BlackBox, or no value on failure.
   */
  std::optional<std::int32_t>
  absorbArmedChain(const TrainJobResult &result);

  /**
   * @brief Splits a freeze selection into independent source-to-sink chains.
   * @param selectedNodeIds Stable identifiers selected by the user.
   * @return One node-id list per chain, or empty when any component is invalid.
   *
   * Each chain has a single source and a single sink. Disconnected selections
   * become multiple chains so each sink can show its own compile and inference
   * times.
   */
  [[nodiscard]] std::vector<std::vector<std::int32_t>> partitionFreezeChains(
      const std::vector<std::int32_t> &selectedNodeIds) const;

  /** @brief Returns true when a pin already has a cable. */
  [[nodiscard]] bool isPinConnected(std::int32_t pinId) const noexcept;

  /** @brief Returns graph nodes in stable insertion order. */
  [[nodiscard]] const std::vector<GraphNode> &getNodes() const noexcept;

  /** @brief Returns mutable graph nodes for message-thread rendering. */
  [[nodiscard]] std::vector<GraphNode> &getNodes() noexcept;

  /** @brief Returns directed links in stable insertion order. */
  [[nodiscard]] const std::vector<GraphLink> &getLinks() const noexcept;

  /** @brief Returns mutable links for message-thread rendering. */
  [[nodiscard]] std::vector<GraphLink> &getLinks() noexcept;

  /** @brief Finds a node by stable identifier. */
  [[nodiscard]] GraphNode *findNode(std::int32_t nodeId) noexcept;

  /** @brief Finds a node by stable identifier. */
  [[nodiscard]] const GraphNode *findNode(std::int32_t nodeId) const noexcept;

  /** @brief Finds a pin by stable identifier. */
  [[nodiscard]] const Pin *findPin(std::int32_t pinId) const noexcept;

  /** @brief Finds a link by stable identifier. */
  [[nodiscard]] const GraphLink *findLink(std::int32_t linkId) const noexcept;

  /** @brief Returns true for the undeletable host audio boundary nodes. */
  [[nodiscard]] bool isFixedIoNode(std::int32_t nodeId) const noexcept;

  /** @brief Returns true for a non-removable Group Input/Output hub. */
  [[nodiscard]] bool isGroupBoundaryNode(std::int32_t nodeId) const noexcept;

  /** @brief Returns the node that owns a pin, or no value. */
  [[nodiscard]] std::optional<std::int32_t>
  findNodeForPin(std::int32_t pinId) const noexcept;

  /** @brief Returns mutable viewport persistence state. */
  [[nodiscard]] ViewportState &getViewport() noexcept;

  /** @brief Returns immutable viewport persistence state. */
  [[nodiscard]] const ViewportState &getViewport() const noexcept;

  /** @brief Serializes the complete graph into a JUCE value tree. */
  [[nodiscard]] juce::ValueTree toValueTree() const;

  /**
   * @brief Returns true when @p tree is a GraphDocument this build can restore.
   *
   * Rejects unknown element types so apply can fail closed without a partial
   * graph. Groups, layout positions, and Gold artifact paths are accepted when
   * well-formed.
   *
   * @param tree Candidate graph document.
   * @param error Receives a user-facing refusal.
   */
  static bool documentIsRestorable(const juce::ValueTree &tree,
                                   juce::String &error);

  /**
   * @brief Restores a graph from a JUCE value tree.
   * @param tree Persisted graph state.
   * @return True when the tree was recognized and restored.
   */
  bool restoreFromValueTree(const juce::ValueTree &tree);

  /**
   * @brief Serializes one element or one group tree (internal links only).
   * @param boxId Node id or group id to snapshot.
   * @param error Receives a user-facing refusal when empty.
   * @return Snapshot tree, or invalid when refused.
   */
  [[nodiscard]] juce::ValueTree exportBox(std::int32_t boxId,
                                          juce::String &error) const;

  /**
   * @brief Clones a box snapshot into this graph at @p position with new IDs.
   * @param snapshot Tree produced by @ref exportBox.
   * @param position Canvas origin for the placed root.
   * @param collapseGroups True to force every imported group collapsed.
   * @param error Receives a user-facing failure.
   * @param nestedRootId Snapshot node/group id to insert instead of the saved
   *        root; 0 inserts the snapshot root.
   * @return New root node or group id, or no value on failure.
   */
  std::optional<std::int32_t> importBox(const juce::ValueTree &snapshot,
                                        juce::Point<float> position,
                                        bool collapseGroups,
                                        juce::String &error,
                                        std::int32_t nestedRootId = 0);

  /** @brief Serializes the graph into compact JSON for worker IPC. */
  [[nodiscard]] std::string toJson() const;

private:
  /** @brief Creates a fully initialized node without inserting it. */
  GraphNode makeNode(NodeType type, juce::Point<float> position);
  /**
   * @brief Rebuilds Utility input ports to match the Inputs property.
   * @param node Utility node to update.
   * @param inputCount Requested input port count (minimum 1).
   */
  void setMixerInputCount(GraphNode &node, int inputCount);
  /**
   * @brief Resizes paired lanes on a Group Input/Output hub.
   * @param node Boundary hub to update.
   * @param portCount Requested lane count (minimum one).
   * @return False when shrinking would remove a connected lane.
   */
  bool setGroupBoundaryPortCount(GraphNode &node, int portCount);
  /**
   * @brief Adds fixed boundary hubs and routes a new or legacy group through them.
   * @param groupId Group requiring an explicit interface.
   * @param preserveInferredPorts True to preserve every legacy inferred port;
   *        false to declare only currently crossing cables.
   */
  void ensureGroupBoundaryNodes(std::int32_t groupId,
                                bool preserveInferredPorts);
  /**
   * @brief Splices paired lanes through one boundary hub and removes it.
   * @param nodeId Group Input/Output node to dissolve.
   */
  void spliceGroupBoundaryNode(std::int32_t nodeId);
  /** @brief Removes editor-only hubs by splicing every paired lane. */
  void flattenGroupBoundaryNodes();
  /** @brief Tests whether a candidate edge would introduce a directed cycle. */
  bool wouldCreateCycle(std::int32_t sourceNodeId,
                        std::int32_t destinationNodeId) const;
  /** @brief Tests whether selected live nodes form one connected component. */
  bool
  selectionIsConnected(const std::vector<std::int32_t> &selectedNodeIds) const;

  /** @brief Stable insertion-ordered node collection. */
  std::vector<GraphNode> nodes;
  /** @brief Stable insertion-ordered group collection. */
  std::vector<GraphGroup> groups;
  /** @brief Stable insertion-ordered link collection. */
  std::vector<GraphLink> links;
  /** @brief Persisted graph navigation state. */
  ViewportState viewport;
  /** @brief Next unused node identifier. */
  std::int32_t nextNodeId = 1;
  /** @brief Next unused pin identifier. */
  std::int32_t nextPinId = 1001;
  /** @brief Next unused link identifier. */
  std::int32_t nextLinkId = 2001;
};
} // namespace openyourbox::graph
