#include "graph/NodeGraph.h"

#include <JuceHeader.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using openyourbox::graph::GraphNode;
using openyourbox::graph::NodeGraph;
using openyourbox::graph::NodeState;
using openyourbox::graph::NodeType;
using openyourbox::graph::isControlInputPin;

/**
 * @brief Reports a failed invariant.
 * @param condition Invariant result.
 * @param message Human-readable failure.
 * @return The supplied condition.
 */
bool expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

/**
 * @brief Returns true when a cable exists between two pins.
 * @param graph Graph owning links.
 * @param sourcePin Output pin identifier.
 * @param destPin Input pin identifier.
 */
bool hasLink(const NodeGraph &graph, std::int32_t sourcePin,
             std::int32_t destPin) {
  for (const auto &link : graph.getLinks()) {
    if (link.sourcePinId == sourcePin && link.destinationPinId == destPin)
      return true;
  }
  return false;
}

/**
 * @brief Counts cables that land on an input pin.
 * @param graph Graph owning links.
 * @param destPin Input pin identifier.
 */
int incomingCount(const NodeGraph &graph, std::int32_t destPin) {
  int count = 0;
  for (const auto &link : graph.getLinks()) {
    if (link.destinationPinId == destPin)
      ++count;
  }
  return count;
}

/**
 * @brief First non-control input pin of @p node, or 0.
 * @param node Element to inspect.
 */
std::int32_t signalIn(const GraphNode &node) {
  for (const auto &pin : node.inputs) {
    if (!isControlInputPin(pin))
      return pin.id;
  }
  return 0;
}

/**
 * @brief First output pin of @p node, or 0.
 * @param node Element to inspect.
 */
std::int32_t signalOut(const GraphNode &node) {
  return node.outputs.empty() ? 0 : node.outputs.front().id;
}

/**
 * @brief Control input pin of @p node, or 0.
 * @param node Element to inspect.
 */
std::int32_t controlIn(const GraphNode &node) {
  for (const auto &pin : node.inputs) {
    if (isControlInputPin(pin))
      return pin.id;
  }
  return 0;
}

/**
 * @brief Resolves a node pointer or fails the test.
 * @param graph Graph owning nodes.
 * @param nodeId Candidate identifier.
 */
const GraphNode *requireNode(const NodeGraph &graph, std::int32_t nodeId) {
  return graph.findNode(nodeId);
}
} // namespace

/**
 * @brief Runs bypass-on-delete topology checks.
 * @return Zero when every invariant passes.
 */
int main() {
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto mid = graph.addNode(NodeType::convolution, {120.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {240.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *midNode = requireNode(graph, mid);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && midNode != nullptr &&
                         outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*midNode))
                             .accepted &&
                         graph.connect(signalOut(*midNode), signalIn(*outNode))
                             .accepted,
                     "AudioIn→Conv→AudioOut must commit");
    const auto inOut = signalOut(*inNode);
    const auto outIn = signalIn(*outNode);
    passed &= expect(graph.removeNode(mid), "middle conv must delete");
    passed &= expect(graph.findNode(mid) == nullptr &&
                         hasLink(graph, inOut, outIn),
                     "deleting the only box between Audio In and Out must "
                     "connect them");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto a = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto b = graph.addNode(NodeType::convolution, {240.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && nodeA != nullptr && nodeB != nullptr &&
                         outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*nodeA))
                             .accepted &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*outNode))
                             .accepted,
                     "AudioIn→A→B→AudioOut must commit");
    const auto inOut = signalOut(*inNode);
    const auto outIn = signalIn(*outNode);
    passed &= expect(graph.removeBoxes({a, b}),
                     "every box between Audio In and Out must delete");
    passed &= expect(graph.findNode(a) == nullptr && graph.findNode(b) == nullptr &&
                         hasLink(graph, inOut, outIn),
                     "deleting everything between Audio In and Out must "
                     "connect them");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted,
                     "A→B→C must commit");
    passed &= expect(graph.removeNode(b), "B must be deletable");
    nodeA = requireNode(graph, a);
    nodeC = requireNode(graph, c);
    passed &= expect(graph.findNode(b) == nullptr, "B must be gone");
    passed &= expect(nodeA != nullptr && nodeC != nullptr &&
                         hasLink(graph, signalOut(*nodeA), signalIn(*nodeC)),
                     "deleting B in A→B→C must yield A→C");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto d = graph.addNode(NodeType::activation, {360.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    const auto *nodeD = requireNode(graph, d);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         nodeD != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted &&
                         graph.connect(signalOut(*nodeC), signalIn(*nodeD))
                             .accepted,
                     "A→B→C→D must commit");
    passed &= expect(graph.removeBoxes({b, c}), "B and C must delete together");
    nodeA = requireNode(graph, a);
    nodeD = requireNode(graph, d);
    passed &= expect(graph.findNode(b) == nullptr && graph.findNode(c) == nullptr,
                     "selection B,C must be gone");
    passed &= expect(nodeA != nullptr && nodeD != nullptr &&
                         hasLink(graph, signalOut(*nodeA), signalIn(*nodeD)),
                     "deleting B and C in A→B→C→D must yield A→D");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto d = graph.addNode(NodeType::activation, {360.0f, 0.0f});
    const auto e = graph.addNode(NodeType::activation, {480.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    const auto *nodeD = requireNode(graph, d);
    const auto *nodeE = requireNode(graph, e);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         nodeD != nullptr && nodeE != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted &&
                         graph.connect(signalOut(*nodeC), signalIn(*nodeD))
                             .accepted &&
                         graph.connect(signalOut(*nodeD), signalIn(*nodeE))
                             .accepted,
                     "A→B→C→D→E must commit");
    passed &= expect(graph.removeBoxes({b, d}),
                     "non-contiguous B and D must delete");
    nodeA = requireNode(graph, a);
    nodeC = requireNode(graph, c);
    nodeE = requireNode(graph, e);
    passed &= expect(nodeA != nullptr && nodeC != nullptr && nodeE != nullptr &&
                         hasLink(graph, signalOut(*nodeA), signalIn(*nodeC)) &&
                         hasLink(graph, signalOut(*nodeC), signalIn(*nodeE)),
                     "deleting B and D must yield A→C→E");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto d = graph.addNode(NodeType::activation, {240.0f, 80.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    const auto *nodeD = requireNode(graph, d);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         nodeD != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeD))
                             .accepted,
                     "A→B fan-out to C and D must commit");
    passed &= expect(graph.removeNode(b), "fan-out B must delete");
    nodeA = requireNode(graph, a);
    nodeC = requireNode(graph, c);
    nodeD = requireNode(graph, d);
    passed &= expect(nodeA != nullptr && nodeC != nullptr && nodeD != nullptr &&
                         hasLink(graph, signalOut(*nodeA), signalIn(*nodeC)) &&
                         hasLink(graph, signalOut(*nodeA), signalIn(*nodeD)),
                     "deleting fan-out B must reconnect A to C and D");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto extra = graph.addNode(NodeType::activation, {0.0f, 80.0f});
    const auto mix = graph.addNode(NodeType::merge, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    graph.setProperty(mix, "inputs", 2);
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeExtra = requireNode(graph, extra);
    const auto *nodeMix = requireNode(graph, mix);
    const auto *nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeExtra != nullptr &&
                         nodeMix != nullptr && nodeC != nullptr &&
                         nodeMix->inputs.size() >= 2 &&
                         graph.connect(signalOut(*nodeA), nodeMix->inputs[0].id)
                             .accepted &&
                         graph
                             .connect(signalOut(*nodeExtra),
                                      nodeMix->inputs[1].id)
                             .accepted &&
                         graph.connect(signalOut(*nodeMix), signalIn(*nodeC))
                             .accepted,
                     "two sources into a merge must commit");
    passed &= expect(graph.removeNode(mix), "merge must delete");
    nodeA = requireNode(graph, a);
    nodeExtra = requireNode(graph, extra);
    nodeC = requireNode(graph, c);
    const auto aLinked =
        nodeA != nullptr && nodeC != nullptr &&
        hasLink(graph, signalOut(*nodeA), signalIn(*nodeC));
    const auto extraLinked =
        nodeExtra != nullptr && nodeC != nullptr &&
        hasLink(graph, signalOut(*nodeExtra), signalIn(*nodeC));
    passed &= expect(nodeC != nullptr && incomingCount(graph, signalIn(*nodeC)) == 1 &&
                         (aLinked != extraLinked),
                     "merge fan-in bypass keeps exactly one legal source");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    passed &= expect(nodeA != nullptr && nodeB != nullptr,
                     "duplicate-bypass fixture nodes exist");
    const auto aOut = signalOut(*nodeA);
    const auto bIn = signalIn(*nodeB);
    const auto bOut = signalOut(*nodeB);
    const auto extra = graph.addNode(NodeType::merge, {240.0f, 80.0f});
    graph.setProperty(extra, "inputs", 2);
    const auto *nodeExtra = requireNode(graph, extra);
    passed &= expect(nodeExtra != nullptr && nodeExtra->inputs.size() >= 2 &&
                         graph.connect(aOut, bIn).accepted &&
                         graph.connect(bOut, nodeExtra->inputs[0].id).accepted &&
                         graph.connect(aOut, nodeExtra->inputs[1].id).accepted,
                     "A already feeds a second dest besides B");
    passed &= expect(graph.removeNode(b), "B beside an existing A cable deletes");
    nodeA = requireNode(graph, a);
    nodeExtra = requireNode(graph, extra);
    passed &= expect(nodeA != nullptr && nodeExtra != nullptr &&
                         hasLink(graph, signalOut(*nodeA),
                                 nodeExtra->inputs[1].id) &&
                         hasLink(graph, signalOut(*nodeA),
                                 nodeExtra->inputs[0].id),
                     "existing A cable survives and B's dest is also healed");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        graph.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto synth =
        graph.addNode(NodeType::pqmfSynthesis, {240.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *anNode = requireNode(graph, analysis);
    const auto *syNode = requireNode(graph, synth);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && anNode != nullptr &&
                         syNode != nullptr && outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*anNode))
                             .accepted &&
                         graph.connect(signalOut(*anNode), signalIn(*syNode))
                             .accepted &&
                         graph.connect(signalOut(*syNode), signalIn(*outNode))
                             .accepted,
                     "Audio→PQMF→Synth→Output must commit");
    passed &= expect(graph.removeNode(analysis), "PQMF analysis must delete");
    inNode = requireNode(graph, input);
    syNode = requireNode(graph, synth);
    outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && syNode != nullptr &&
                         outNode != nullptr &&
                         !hasLink(graph, signalOut(*inNode), signalIn(*syNode)) &&
                         hasLink(graph, signalOut(*syNode), signalIn(*outNode)),
                     "incompatible Audio→Synth bypass must be skipped");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        graph.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto synth =
        graph.addNode(NodeType::pqmfSynthesis, {240.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *anNode = requireNode(graph, analysis);
    const auto *syNode = requireNode(graph, synth);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && anNode != nullptr &&
                         syNode != nullptr && outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*anNode))
                             .accepted &&
                         graph.connect(signalOut(*anNode), signalIn(*syNode))
                             .accepted &&
                         graph.connect(signalOut(*syNode), signalIn(*outNode))
                             .accepted,
                     "batch PQMF fixture must commit");
    passed &= expect(graph.removeBoxes({analysis, synth}),
                     "PQMF pair must delete as one cut");
    inNode = requireNode(graph, input);
    outNode = requireNode(graph, output);
    passed &= expect(graph.findNode(analysis) == nullptr &&
                         graph.findNode(synth) == nullptr &&
                         inNode != nullptr && outNode != nullptr &&
                         hasLink(graph, signalOut(*inNode), signalIn(*outNode)),
                     "deleting analysis+synth must heal Audio→Output");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto tcn = graph.addNode(NodeType::tcn, {120.0f, 0.0f});
    const auto knob = graph.addNode(NodeType::knobInput, {120.0f, 80.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {240.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *tcnNode = requireNode(graph, tcn);
    const auto *knobNode = requireNode(graph, knob);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && tcnNode != nullptr &&
                         knobNode != nullptr && outNode != nullptr &&
                         controlIn(*tcnNode) != 0 &&
                         graph.connect(signalOut(*inNode), signalIn(*tcnNode))
                             .accepted &&
                         graph
                             .connect(signalOut(*knobNode), controlIn(*tcnNode))
                             .accepted &&
                         graph.connect(signalOut(*tcnNode), signalIn(*outNode))
                             .accepted,
                     "audio+control into TCN must commit");
    passed &= expect(graph.removeNode(tcn), "TCN must delete");
    inNode = requireNode(graph, input);
    knobNode = requireNode(graph, knob);
    outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && outNode != nullptr && knobNode != nullptr &&
                         hasLink(graph, signalOut(*inNode), signalIn(*outNode)),
                     "TCN audio path must heal to Audio→Output");
    passed &= expect(incomingCount(graph, signalIn(*outNode)) == 1,
                     "TCN control must not be spliced onto the audio output");
    bool knobStillConnected = false;
    for (const auto &link : graph.getLinks()) {
      if (link.sourcePinId == signalOut(*knobNode))
        knobStillConnected = true;
    }
    passed &= expect(!knobStillConnected,
                     "TCN control source must drop when the TCN is deleted");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto analysis =
        graph.addNode(NodeType::pqmfAnalysis, {120.0f, 0.0f});
    const auto synth =
        graph.addNode(NodeType::pqmfSynthesis, {240.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *anNode = requireNode(graph, analysis);
    const auto *syNode = requireNode(graph, synth);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && anNode != nullptr &&
                         syNode != nullptr && outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*anNode))
                             .accepted &&
                         graph.connect(signalOut(*anNode), signalIn(*syNode))
                             .accepted &&
                         graph.connect(signalOut(*syNode), signalIn(*outNode))
                             .accepted,
                     "sequential PQMF fixture must commit");
    passed &= expect(graph.removeNode(analysis),
                     "first sequential PQMF delete must succeed");
    passed &= expect(graph.removeNode(synth),
                     "second sequential PQMF delete must succeed");
    inNode = requireNode(graph, input);
    outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && outNode != nullptr &&
                         hasLink(graph, signalOut(*inNode), signalIn(*outNode)),
                     "deleting PQMF then synth one-by-one must still heal "
                     "Audio In→Out");
  }

  {
    NodeGraph graph;
    const auto input = graph.addNode(NodeType::audioInput, {0.0f, 0.0f});
    const auto a = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto output = graph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
    const auto *inNode = requireNode(graph, input);
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *outNode = requireNode(graph, output);
    passed &= expect(inNode != nullptr && nodeA != nullptr && nodeB != nullptr &&
                         outNode != nullptr &&
                         graph.connect(signalOut(*inNode), signalIn(*nodeA))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*outNode))
                             .accepted,
                     "disconnected In→A and B→Out must commit");
    passed &= expect(graph.removeBoxes({a, b}),
                     "disconnected interiors must delete as one cut");
    inNode = requireNode(graph, input);
    outNode = requireNode(graph, output);
    passed &= expect(graph.findNode(a) == nullptr && graph.findNode(b) == nullptr &&
                         inNode != nullptr && outNode != nullptr &&
                         hasLink(graph, signalOut(*inNode), signalIn(*outNode)),
                     "deleting everything between Audio In and Out must "
                     "connect them even when interiors were not chained");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto midA = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto midB = graph.addNode(NodeType::activation, {120.0f, 80.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto d = graph.addNode(NodeType::activation, {240.0f, 80.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeMidA = requireNode(graph, midA);
    const auto *nodeMidB = requireNode(graph, midB);
    const auto *nodeC = requireNode(graph, c);
    const auto *nodeD = requireNode(graph, d);
    passed &= expect(nodeA != nullptr && nodeMidA != nullptr &&
                         nodeMidB != nullptr && nodeC != nullptr &&
                         nodeD != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeMidA))
                             .accepted &&
                         graph.connect(signalOut(*nodeMidA), signalIn(*nodeC))
                             .accepted &&
                         graph
                             .connect(signalOut(*nodeMidB), signalIn(*nodeD))
                             .accepted,
                     "two-lane pre-group wiring must commit");
    const auto extra = graph.addNode(NodeType::activation, {0.0f, 80.0f});
    const auto *nodeExtra = requireNode(graph, extra);
    passed &= expect(nodeExtra != nullptr &&
                         graph
                             .connect(signalOut(*nodeExtra), signalIn(*nodeMidB))
                             .accepted,
                     "second lane source must commit");
    const auto grouped = graph.createGroup({midA, midB});
    passed &= expect(grouped.accepted, "two-lane members must group");
    const auto aOut = signalOut(*requireNode(graph, a));
    const auto extraOut = signalOut(*requireNode(graph, extra));
    const auto cIn = signalIn(*requireNode(graph, c));
    const auto dIn = signalIn(*requireNode(graph, d));
    passed &= expect(graph.deleteGroup(grouped.groupId).accepted,
                     "grouped two-lane block must delete");
    passed &= expect(graph.findNode(midA) == nullptr &&
                         graph.findNode(midB) == nullptr &&
                         hasLink(graph, aOut, cIn) &&
                         hasLink(graph, extraOut, dIn),
                     "deleting a two-lane group must heal each lane");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted,
                     "group-chain fixture must commit");
    const auto side = graph.addNode(NodeType::activation, {120.0f, 80.0f});
    const auto grouped = graph.createGroup({b, side});
    passed &= expect(grouped.accepted, "B plus a sibling must group");
    graph.setGroupCollapsed(grouped.groupId, true);
    const auto aOut = signalOut(*requireNode(graph, a));
    const auto cIn = signalIn(*requireNode(graph, c));
    passed &= expect(graph.deleteGroup(grouped.groupId).accepted,
                     "collapsed group must delete");
    passed &= expect(hasLink(graph, aOut, cIn),
                     "deleting a collapsed group must heal A→C");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto side = graph.addNode(NodeType::activation, {120.0f, 80.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted,
                     "interior-delete fixture must commit");
    const auto grouped = graph.createGroup({b, side});
    passed &= expect(grouped.accepted, "interior member group must create");
    passed &= expect(graph.removeNode(b), "interior member B must delete");
    passed &= expect(graph.findGroup(grouped.groupId) != nullptr,
                     "group container survives interior delete");
    nodeA = requireNode(graph, a);
    nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeC != nullptr &&
                         !hasLink(graph, signalOut(*nodeA), signalIn(*nodeC)),
                     "A and C stay attached to hubs, not to each other");
    bool aFeedsHub = false;
    bool hubFeedsC = false;
    for (const auto &link : graph.getLinks()) {
      const auto source = graph.findNodeForPin(link.sourcePinId);
      const auto dest = graph.findNodeForPin(link.destinationPinId);
      if (source == a && dest.has_value() && graph.isGroupBoundaryNode(*dest))
        aFeedsHub = true;
      if (dest == c && source.has_value() && graph.isGroupBoundaryNode(*source))
        hubFeedsC = true;
    }
    passed &= expect(aFeedsHub && hubFeedsC,
                     "deleting interior B must splice through the group hubs");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto side = graph.addNode(NodeType::activation, {120.0f, 80.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto *nodeA = requireNode(graph, a);
    const auto *nodeB = requireNode(graph, b);
    const auto *nodeC = requireNode(graph, c);
    passed &= expect(nodeA != nullptr && nodeB != nullptr && nodeC != nullptr &&
                         graph.connect(signalOut(*nodeA), signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB), signalIn(*nodeC))
                             .accepted,
                     "group-plus-member fixture must commit");
    const auto grouped = graph.createGroup({b, side});
    passed &= expect(grouped.accepted, "overlapping-id group must create");
    const auto aOut = signalOut(*requireNode(graph, a));
    const auto cIn = signalIn(*requireNode(graph, c));
    passed &= expect(graph.removeBoxes({grouped.groupId, b}),
                     "group plus a member id must delete once");
    passed &= expect(graph.findGroup(grouped.groupId) == nullptr &&
                         graph.findNode(b) == nullptr &&
                         hasLink(graph, aOut, cIn),
                     "overlapping group+member delete still heals A→C");
  }

  {
    NodeGraph graph;
    const auto a = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    const auto b = graph.addNode(NodeType::activation, {120.0f, 0.0f});
    const auto c = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    const auto *nodeB = requireNode(graph, b);
    passed &= expect(nodeB != nullptr &&
                         graph.connect(signalOut(*requireNode(graph, a)),
                                       signalIn(*nodeB))
                             .accepted &&
                         graph.connect(signalOut(*nodeB),
                                       signalIn(*requireNode(graph, c)))
                             .accepted,
                     "frozen-gold fixture must commit");
    auto *mutableB = graph.findNode(b);
    passed &= expect(mutableB != nullptr, "B must be mutable");
    if (mutableB != nullptr)
      mutableB->state = NodeState::frozenGold;
    const auto aOut = signalOut(*requireNode(graph, a));
    const auto cIn = signalIn(*requireNode(graph, c));
    passed &= expect(graph.removeNode(b), "frozen Gold must delete");
    passed &= expect(graph.findNode(b) == nullptr && hasLink(graph, aOut, cIn),
                     "deleting frozen Gold B must still heal A→C");
  }

  {
    NodeGraph graph;
    const auto lonely = graph.addNode(NodeType::activation, {0.0f, 0.0f});
    passed &= expect(graph.removeNode(lonely) && graph.findNode(lonely) == nullptr,
                     "unconnected box deletes without inventing cables");
    passed &= expect(graph.getLinks().empty(),
                     "lonely delete leaves no leftover links");
    passed &= expect(!graph.removeBoxes({}) && !graph.removeBoxes({99999}),
                     "empty or unknown ids must not report a mutation");
  }

  {
    NodeGraph graph;
    const auto knob = graph.addNode(NodeType::knobInput, {0.0f, 0.0f});
    const auto mix = graph.addNode(NodeType::merge, {120.0f, 0.0f});
    const auto tail = graph.addNode(NodeType::activation, {240.0f, 0.0f});
    graph.setProperty(mix, "inputs", 2);
    const auto *knobNode = requireNode(graph, knob);
    const auto *mixNode = requireNode(graph, mix);
    const auto *tailNode = requireNode(graph, tail);
    passed &= expect(knobNode != nullptr && mixNode != nullptr &&
                         tailNode != nullptr && mixNode->inputs.size() >= 1 &&
                         graph.connect(signalOut(*knobNode), mixNode->inputs[0].id)
                             .accepted &&
                         graph.connect(signalOut(*mixNode), signalIn(*tailNode))
                             .accepted,
                     "knob into merge must commit");
    passed &= expect(graph.removeNode(knob), "source-only knob must delete");
    mixNode = requireNode(graph, mix);
    tailNode = requireNode(graph, tail);
    passed &= expect(mixNode != nullptr && tailNode != nullptr &&
                         incomingCount(graph, mixNode->inputs[0].id) == 0 &&
                         hasLink(graph, signalOut(*mixNode), signalIn(*tailNode)),
                     "deleting a source-only box must not invent a bypass");
  }

  {
    NodeGraph graph;
    const auto knob = graph.addNode(NodeType::knobInput, {0.0f, 0.0f});
    const auto *node = requireNode(graph, knob);
    passed &= expect(node != nullptr && node->outputs.size() == 2 &&
                         node->outputs.front().label == "c" &&
                         node->outputs.back().label == "concat" &&
                         node->outputs.back().shape.channels == 1,
                     "Knob Input defaults to one c pin plus concat");
    passed &= expect(graph.setProperty(knob, "knobs", 3),
                     "Knobs property may grow the element");
    node = requireNode(graph, knob);
    passed &= expect(node != nullptr && node->outputs.size() == 4 &&
                         node->outputs[0].label == "c1" &&
                         node->outputs[2].label == "c3" &&
                         node->outputs.back().label == "concat" &&
                         node->outputs.back().shape.channels == 3 &&
                         node->detail == "3D conditioning",
                     "three knobs expose c1–c3 plus concat");
    const auto firstPin = node->outputs.front().id;
    passed &=
        expect(graph.setConditioningValue(knob, 0, 1.25f) &&
                   graph.setConditioningValue(knob, 2, -2.5f) &&
                   !graph.setConditioningValue(knob, 3, 0.0f),
               "per-knob values stay in range of the current count");
    const auto mix = graph.addNode(NodeType::merge, {120.0f, 0.0f});
    graph.setProperty(mix, "inputs", 2);
    node = requireNode(graph, knob);
    const auto *mixNode = requireNode(graph, mix);
    passed &= expect(node != nullptr && mixNode != nullptr &&
                         node->outputs.size() >= 2 &&
                         graph.connect(node->outputs[1].id, mixNode->inputs[0].id)
                             .accepted,
                     "second knob pin may connect");
    passed &= expect(graph.setProperty(knob, "knobs", 1),
                     "Knobs property may shrink the element");
    node = requireNode(graph, knob);
    mixNode = requireNode(graph, mix);
    passed &= expect(node != nullptr && mixNode != nullptr &&
                         node->outputs.size() == 2 &&
                         node->outputs.front().id == firstPin &&
                         node->outputs.front().label == "c" &&
                         node->outputs.back().label == "concat" &&
                         incomingCount(graph, mixNode->inputs[0].id) == 0,
                     "shrinking knobs keeps the first pin, concat, and drops extra cables");
    passed &= expect(graph.setProperty(knob, "knobs", 3) &&
                         graph.setConditioningValue(knob, 1, 4.5f),
                     "values restore after growing again");
    auto tree = graph.toValueTree();
    NodeGraph restored;
    passed &= expect(restored.restoreFromValueTree(tree),
                     "multi-knob document must restore");
    const auto *restoredNode = restored.findNode(knob);
    passed &= expect(
        restoredNode != nullptr && restoredNode->outputs.size() == 4 &&
            restoredNode->outputs.back().label == "concat" &&
            restoredNode->outputs.back().shape.channels == 3 &&
            restoredNode->conditioningValues.size() == 3 &&
            std::abs(restoredNode->conditioningValues[1] - 4.5f) < 1.0e-4f,
        "knob count, concat pin, and values survive ValueTree recall");
  }

  {
    NodeGraph legacySource;
    const auto knob = legacySource.addNode(NodeType::knobInput, {0.0f, 0.0f});
    passed &= expect(legacySource.setConditioningValue(knob, 0.75f),
                     "legacy fixture knob value");
    auto tree = legacySource.toValueTree();
    for (int index = 0; index < tree.getNumChildren(); ++index) {
      auto child = tree.getChild(index);
      if (!child.hasType("Node") || child["type"].toString() != "knob_input")
        continue;
      for (int propertyIndex = child.getNumChildren() - 1; propertyIndex >= 0;
           --propertyIndex) {
        auto nested = child.getChild(propertyIndex);
        if (nested.hasType("Property") && nested["key"].toString() == "knobs")
          child.removeChild(propertyIndex, nullptr);
      }
      child.removeProperty("conditioningValues", nullptr);
    }
    NodeGraph legacy;
    passed &= expect(legacy.restoreFromValueTree(tree),
                     "legacy single-knob documents must load");
    const auto *legacyNode = legacy.findNode(knob);
    int knobsValue = 0;
    if (legacyNode != nullptr) {
      for (const auto &property : legacyNode->properties) {
        if (property.key == "knobs")
          knobsValue = property.value;
      }
    }
    passed &= expect(legacyNode != nullptr &&
                         openyourbox::graph::individualConditioningOutputCount(
                             *legacyNode) == 1 &&
                         legacyNode->outputs.size() == 2 &&
                         legacyNode->outputs.back().label == "concat" &&
                         knobsValue == 1 &&
                         std::abs(legacyNode->conditioningValue - 0.75f) <
                             1.0e-4f,
                     "legacy Knob Input gains a Knobs parameter of 1 and concat");
  }

  {
    NodeGraph graph;
    const auto pad = graph.addNode(NodeType::xyTrackpad, {0.0f, 0.0f});
    const auto *node = requireNode(graph, pad);
    passed &= expect(node != nullptr && node->outputs.size() == 3 &&
                         node->outputs[0].label == "x" &&
                         node->outputs[1].label == "y" &&
                         node->outputs[2].label == "concat" &&
                         node->outputs[2].shape.channels == 2,
                     "XY Trackpad exposes x, y, and concat");
    auto tree = graph.toValueTree();
    for (int index = 0; index < tree.getNumChildren(); ++index) {
      auto child = tree.getChild(index);
      if (!child.hasType("Node") || child["type"].toString() != "xy_trackpad")
        continue;
      for (int pinIndex = child.getNumChildren() - 1; pinIndex >= 0;
           --pinIndex) {
        auto nested = child.getChild(pinIndex);
        if (nested.hasType("Pin") && nested["kind"].toString() == "output" &&
            nested["label"].toString() == "concat")
          child.removeChild(pinIndex, nullptr);
      }
    }
    NodeGraph restored;
    passed &= expect(restored.restoreFromValueTree(tree),
                     "legacy XY documents must load");
    const auto *restoredPad = restored.findNode(pad);
    passed &= expect(restoredPad != nullptr && restoredPad->outputs.size() == 3 &&
                         restoredPad->outputs.back().label == "concat" &&
                         restoredPad->outputs.back().shape.channels == 2,
                     "legacy XY Trackpad gains a concat pin");
  }

  if (!passed)
    return 1;
  std::cout << "GraphBypassDeleteTests passed\n";
  return 0;
}
