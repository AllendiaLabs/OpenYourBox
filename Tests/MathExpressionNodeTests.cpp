#include "dsp/LiveGraphEngine.h"
#include "graph/GraphTypes.h"
#include "graph/NodeGraph.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <string>

namespace {
/**
 * @brief Reports a failed Math Expression invariant.
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
 * @brief Looks up a string property on @p node.
 * @param node Graph element.
 * @param key Canonical property key.
 * @return Pointer to the property, or nullptr.
 */
const openyourbox::graph::NodeProperty *
findProperty(const openyourbox::graph::GraphNode &node, const std::string &key) {
  for (const auto &property : node.properties) {
    if (property.key == key)
      return &property;
  }
  return nullptr;
}
} // namespace

/**
 * @brief Runs Math Expression pin, refuse, broadcast, and mod_sigmoid coverage.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::dsp::LiveGraphCompileError;
  using openyourbox::dsp::LiveGraphCompileOptions;
  using openyourbox::dsp::LiveGraphEngine;
  using openyourbox::graph::NodeGraph;
  using openyourbox::graph::NodeType;
  juce::ScopedJuceInitialiser_GUI juceInitialiser;
  bool passed = true;

  NodeGraph graph;
  const auto math = graph.addNode(NodeType::mathExpression, {180.0f, 0.0f});
  const auto *mathNode = graph.findNode(math);
  passed &= expect(mathNode != nullptr && mathNode->inputs.size() == 1 &&
                       mathNode->inputs.front().label == "x1",
                   "Math Expression defaults to one pin labelled x1");
  const auto *expression = mathNode != nullptr
                               ? findProperty(*mathNode, "expression")
                               : nullptr;
  passed &= expect(expression != nullptr && expression->stringValue == "x1",
                   "Math Expression defaults to identity x1");

  passed &= expect(!graph.setStringProperty(math, "expression", "sin(x1)"),
                   "unknown functions are refused");
  mathNode = graph.findNode(math);
  expression = mathNode != nullptr ? findProperty(*mathNode, "expression") : nullptr;
  passed &= expect(expression != nullptr && expression->stringValue == "x1",
                   "invalid expression is not stored");
  passed &= expect(!graph.lastPropertyMessage().empty(),
                   "refuse reports a user-facing message");

  passed &=
      expect(graph.setStringProperty(math, "expression", "2 * x1^2.3 + 1e-7"),
             "mod_sigmoid partner expression commits");
  graph.setProperty(math, "inputs", 0);
  mathNode = graph.findNode(math);
  passed &= expect(mathNode != nullptr && mathNode->inputs.size() == 1,
                   "Inputs stays at least 1");
  passed &= expect(graph.setProperty(math, "inputs", 2),
                   "Inputs can grow to 2");
  mathNode = graph.findNode(math);
  passed &= expect(mathNode != nullptr && mathNode->inputs.size() == 2 &&
                       mathNode->inputs[0].label == "x1" &&
                       mathNode->inputs[1].label == "x2",
                   "Inputs=2 rebuilds pins x1 and x2");
  passed &= expect(graph.setProperty(math, "inputs", 1),
                   "Inputs may shrink when the expression only uses x1");
  passed &= expect(graph.setProperty(math, "inputs", 2),
                   "Inputs restored to 2 before a two-input expression");
  passed &= expect(graph.setStringProperty(math, "expression", "x1 + x2"),
                   "two-input expression commits");
  passed &= expect(!graph.setProperty(math, "inputs", 1),
                   "Inputs reduction is refused while expression references x2");
  mathNode = graph.findNode(math);
  passed &= expect(mathNode != nullptr && mathNode->inputs.size() == 2,
                   "refused Inputs reduction keeps both pins");

  LiveGraphCompileOptions options;
  options.hostInputChannels = 2;
  options.hostOutputChannels = 2;
  options.maximumBlockSize = 256;

  NodeGraph unusedPinGraph;
  const auto unusedIn =
      unusedPinGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto unusedMath =
      unusedPinGraph.addNode(NodeType::mathExpression, {180.0f, 0.0f});
  const auto unusedOut =
      unusedPinGraph.addNode(NodeType::audioOutput, {360.0f, 0.0f});
  passed &= expect(unusedPinGraph.setProperty(unusedMath, "inputs", 2),
                   "unused-pin fixture Inputs=2");
  passed &= expect(unusedPinGraph.setStringProperty(unusedMath, "expression", "x1"),
                   "expression may ignore x2");
  unusedPinGraph.connect(
      unusedPinGraph.findNode(unusedIn)->outputs.front().id,
      unusedPinGraph.findNode(unusedMath)->inputs.front().id);
  unusedPinGraph.connect(
      unusedPinGraph.findNode(unusedMath)->outputs.front().id,
      unusedPinGraph.findNode(unusedOut)->inputs.front().id);
  const auto unusedCompiled = LiveGraphEngine::compile(unusedPinGraph, options);
  passed &= expect(unusedCompiled.succeeded(),
                   "unconnected unused x2 is valid when the expression only uses x1");

  NodeGraph broadcastGraph;
  const auto broadcastIn =
      broadcastGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto scalar =
      broadcastGraph.addNode(NodeType::linear, {120.0f, 80.0f});
  const auto broadcastMath =
      broadcastGraph.addNode(NodeType::mathExpression, {240.0f, 0.0f});
  const auto broadcastOut =
      broadcastGraph.addNode(NodeType::audioOutput, {400.0f, 0.0f});
  passed &= expect(broadcastGraph.setProperty(scalar, "features", 1),
                   "scalar path is 1 channel");
  passed &= expect(broadcastGraph.setProperty(broadcastMath, "inputs", 2),
                   "broadcast Math Inputs=2");
  passed &= expect(
      broadcastGraph.setStringProperty(broadcastMath, "expression", "x1 * x2"),
      "x1 * x2 commits");
  const auto *broadcastMathNode = broadcastGraph.findNode(broadcastMath);
  const auto *broadcastInNode = broadcastGraph.findNode(broadcastIn);
  const auto *scalarNode = broadcastGraph.findNode(scalar);
  passed &= expect(broadcastGraph
                       .connect(broadcastInNode->outputs.front().id,
                                broadcastMathNode->inputs[0].id)
                       .accepted,
                   "stereo x1 connects");
  passed &= expect(broadcastGraph
                       .connect(broadcastInNode->outputs.front().id,
                                scalarNode->inputs.front().id)
                       .accepted,
                   "scalar linear takes host audio");
  passed &= expect(broadcastGraph
                       .connect(scalarNode->outputs.front().id,
                                broadcastMathNode->inputs[1].id)
                       .accepted,
                   "1-channel x2 broadcasts against stereo x1");
  passed &= expect(broadcastGraph
                       .connect(broadcastMathNode->outputs.front().id,
                                broadcastGraph.findNode(broadcastOut)
                                    ->inputs.front()
                                    .id)
                       .accepted,
                   "broadcast math feeds Audio Out");
  const auto broadcastCompiled =
      LiveGraphEngine::compile(broadcastGraph, options);
  passed &= expect(broadcastCompiled.succeeded(),
                   "Utility-style 2ch × 1ch Math Expression compiles");

  NodeGraph mismatchGraph;
  const auto mismatchIn =
      mismatchGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto left = mismatchGraph.addNode(NodeType::linear, {120.0f, -40.0f});
  const auto right = mismatchGraph.addNode(NodeType::linear, {120.0f, 80.0f});
  const auto mismatchMath =
      mismatchGraph.addNode(NodeType::mathExpression, {260.0f, 20.0f});
  passed &= expect(mismatchGraph.setProperty(left, "features", 2),
                   "left path is 2ch");
  passed &= expect(mismatchGraph.setProperty(right, "features", 3),
                   "right path is 3ch");
  passed &= expect(mismatchGraph.setProperty(mismatchMath, "inputs", 2),
                   "mismatch Math Inputs=2");
  passed &= expect(
      mismatchGraph.setStringProperty(mismatchMath, "expression", "x1 * x2"),
      "mismatch expression commits");
  mismatchGraph.connect(mismatchGraph.findNode(mismatchIn)->outputs.front().id,
                        mismatchGraph.findNode(left)->inputs.front().id);
  mismatchGraph.connect(mismatchGraph.findNode(mismatchIn)->outputs.front().id,
                        mismatchGraph.findNode(right)->inputs.front().id);
  mismatchGraph.connect(mismatchGraph.findNode(left)->outputs.front().id,
                        mismatchGraph.findNode(mismatchMath)->inputs[0].id);
  const auto mismatchConnect =
      mismatchGraph.connect(mismatchGraph.findNode(right)->outputs.front().id,
                            mismatchGraph.findNode(mismatchMath)->inputs[1].id);
  passed &= expect(!mismatchConnect.accepted,
                   "incompatible multi-channel widths are refused");

  NodeGraph sigmoidGraph;
  const auto sigmoidIn =
      sigmoidGraph.addNode(NodeType::audioInput, {0.0f, 0.0f});
  const auto activation =
      sigmoidGraph.addNode(NodeType::activation, {160.0f, 0.0f});
  const auto sigmoidMath =
      sigmoidGraph.addNode(NodeType::mathExpression, {320.0f, 0.0f});
  const auto sigmoidOut =
      sigmoidGraph.addNode(NodeType::audioOutput, {480.0f, 0.0f});
  passed &= expect(sigmoidGraph.setProperty(activation, "activation", 1),
                   "Activation Function = Sigmoid");
  passed &= expect(sigmoidGraph.setStringProperty(sigmoidMath, "expression",
                                                  "2 * x1^2.3 + 1e-7"),
                   "mod_sigmoid expression commits");
  sigmoidGraph.connect(sigmoidGraph.findNode(sigmoidIn)->outputs.front().id,
                       sigmoidGraph.findNode(activation)->inputs.front().id);
  sigmoidGraph.connect(sigmoidGraph.findNode(activation)->outputs.front().id,
                       sigmoidGraph.findNode(sigmoidMath)->inputs.front().id);
  sigmoidGraph.connect(sigmoidGraph.findNode(sigmoidMath)->outputs.front().id,
                       sigmoidGraph.findNode(sigmoidOut)->inputs.front().id);
  const auto sigmoidCompiled = LiveGraphEngine::compile(sigmoidGraph, options);
  passed &= expect(sigmoidCompiled.succeeded(),
                   "Sigmoid → Math Expression compiles");
  LiveGraphCompileError error;
  const auto sigmoidRuntime =
      LiveGraphEngine::prepare(sigmoidCompiled.snapshot, error);
  passed &= expect(sigmoidRuntime != nullptr, "mod_sigmoid cascade prepares");
  if (sigmoidRuntime != nullptr) {
    const auto ones = torch::ones({1, 2, 8}, torch::kFloat32);
    const auto shaped = sigmoidRuntime->processTensor(ones);
    const auto sigmoidOne = 1.0 / (1.0 + std::exp(-1.0));
    const auto expected = 2.0 * std::pow(sigmoidOne, 2.3) + 1.0e-7;
    passed &= expect(shaped.defined() &&
                         std::abs(shaped[0][0][0].item<float>() -
                                  static_cast<float>(expected)) < 1.0e-5f,
                     "Sigmoid → 2 * x1^2.3 + 1e-7 matches 2 * sigmoid(x)^2.3 + 1e-7");
  }

  if (!passed)
    return 1;
  std::cout << "MathExpressionNodeTests passed\n";
  return 0;
}
