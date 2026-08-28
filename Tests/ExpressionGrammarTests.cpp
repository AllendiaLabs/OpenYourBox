#include "graph/ExpressionParser.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
/**
 * @brief Reports a failed grammar invariant.
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
 * @brief Evaluates a Math Expression string with optional `xK` scalars.
 * @param text Expression source.
 * @param maxInput Highest legal `xK`.
 * @param xs Values for `x1`… in order.
 * @return Finite value, or NaN when parsing/eval failed.
 */
double evalMath(const char *text, int maxInput,
                const std::vector<double> &xs = {}) {
  using openyourbox::graph::ExpressionIdentContext;
  using openyourbox::graph::evaluateExpression;
  using openyourbox::graph::parseExpression;
  const auto parsed =
      parseExpression(text, ExpressionIdentContext::mathInputs, maxInput);
  if (!parsed.accepted)
    return std::numeric_limits<double>::quiet_NaN();
  const auto evaluated = evaluateExpression(parsed.ast, 0.0, xs);
  return evaluated.ok ? evaluated.value
                      : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Evaluates a parameter expression at index @p i.
 * @param text Expression source.
 * @param i Copy index.
 * @return Finite value, or NaN when parsing/eval failed.
 */
double evalParam(const char *text, double i) {
  using openyourbox::graph::evaluateParameterToken;
  const auto evaluated = evaluateParameterToken(text, i);
  return evaluated.ok ? evaluated.value
                      : std::numeric_limits<double>::quiet_NaN();
}
} // namespace

/**
 * @brief Runs shared-grammar parse/eval coverage for Math Expression and `i`.
 * @return Zero when every invariant passes.
 */
int main() {
  using openyourbox::graph::ExpressionIdentContext;
  using openyourbox::graph::expressionValueIsInteger;
  using openyourbox::graph::maxReferencedMathInput;
  using openyourbox::graph::parseExpression;
  bool passed = true;

  passed &= expect(std::abs(evalMath("2 * x1^2.3 + 1e-7", 1, {0.5}) -
                            (2.0 * std::pow(0.5, 2.3) + 1e-7)) < 1.0e-12,
                   "mod_sigmoid partner formula evaluates");
  passed &= expect(std::abs(evalMath("2^3^2", 1) - 512.0) < 1.0e-12,
                   "^ is right-associative: 2^3^2 = 512");
  passed &= expect(std::abs(evalMath("2**3", 1) - 8.0) < 1.0e-12,
                   "** is an ASCII synonym for power");
  passed &= expect(std::abs(evalMath("2**3**2", 1) - 512.0) < 1.0e-12,
                   "** is right-associative like ^");
  passed &= expect(std::abs(evalMath("2 * x1**2.3 + 1e-7", 1, {0.5}) -
                            (2.0 * std::pow(0.5, 2.3) + 1e-7)) < 1.0e-12,
                   "mod_sigmoid formula works with **");
  passed &= expect(std::abs(evalParam("2**3", 7.0) - 8.0) < 1.0e-12,
                   "parameter ** power matches ^");
  passed &= expect(std::abs(evalMath("2+3*4", 1) - 14.0) < 1.0e-12,
                   "* binds tighter than +");
  passed &= expect(std::abs(evalMath("8/2/2", 1) - 2.0) < 1.0e-12,
                   "/ is left-associative: 8/2/2 = 2");
  passed &= expect(std::abs(evalMath("2 * x1 / 4", 1, {6.0}) - 3.0) < 1.0e-12,
                   "* and / share precedence, left to right");
  passed &= expect(std::abs(evalMath("exp(0)", 1) - 1.0) < 1.0e-12,
                   "exp(0) is 1");
  passed &= expect(std::abs(evalMath("exp(x1)", 1, {0.0}) - 1.0) < 1.0e-12,
                   "exp(x1) evaluates the natural exponential");
  passed &= expect(std::abs(evalParam("exp(0)", 3.0) - 1.0) < 1.0e-12,
                   "exp() is legal in parameter expressions");
  passed &= expect(std::abs(evalMath("(2+3)*4", 1) - 20.0) < 1.0e-12,
                   "parentheses override precedence");
  passed &= expect(std::abs(evalMath("-3 + 5", 1) - 2.0) < 1.0e-12,
                   "unary minus is allowed");
  passed &= expect(std::abs(evalMath("2*-3", 1) - -6.0) < 1.0e-12,
                   "unary minus after * is allowed");
  passed &= expect(std::abs(evalMath("-(1+2)", 1) - -3.0) < 1.0e-12,
                   "unary minus of a group is allowed");
  passed &= expect(std::abs(evalMath("1.5e2", 1) - 150.0) < 1.0e-12,
                   "scientific notation is accepted");
  passed &= expect(std::abs(evalMath("x1 * x2", 2, {3.0, 4.0}) - 12.0) < 1.0e-12,
                   "x1 and x2 bind to configured inputs");

  const auto empty =
      parseExpression("   ", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(!empty.accepted, "whitespace-only expressions are refused");
  const auto div =
      parseExpression("2 * x1 / 3", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(div.accepted, "division is allowed");
  passed &= expect(std::abs(evalMath("2 * x1 / 3", 1, {3.0}) - 2.0) < 1.0e-12,
                   "2 * x1 / 3 with x1=3 is 2");
  const auto unknown =
      parseExpression("sin(x1)", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(!unknown.accepted, "unknown function calls are refused");
  const auto expBare =
      parseExpression("exp", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(!expBare.accepted, "exp without parentheses is refused");
  const auto oneOverZero =
      openyourbox::graph::evaluateParameterToken("1/0", 0.0);
  passed &= expect(!oneOverZero.ok && !oneOverZero.message.empty(),
                   "division by zero is refused as non-finite");
  const auto missingX =
      parseExpression("x2", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(!missingX.accepted, "x2 is refused when Inputs is 1");
  const auto juxtaposition =
      parseExpression("2i", ExpressionIdentContext::parameterIndex, 0);
  passed &= expect(!juxtaposition.accepted,
                   "implicit multiplication 2i is refused");
  const auto xInParam =
      parseExpression("x1", ExpressionIdentContext::parameterIndex, 0);
  passed &= expect(!xInParam.accepted, "x1 is not legal in parameter fields");
  const auto iInMath =
      parseExpression("i", ExpressionIdentContext::mathInputs, 1);
  passed &= expect(!iInMath.accepted, "i is not legal in Math Expression");

  passed &= expect(std::abs(evalParam("2*i+1", 0.0) - 1.0) < 1.0e-12,
                   "i=0 yields 1 for 2*i+1");
  passed &= expect(std::abs(evalParam("2*i+1", 3.0) - 7.0) < 1.0e-12,
                   "i=3 yields 7 for 2*i+1");
  passed &= expect(std::abs(evalParam("2^3", 7.0) - 8.0) < 1.0e-12,
                   "constant 2^3 is 8 regardless of i");
  passed &= expect(expressionValueIsInteger(8.0) &&
                       !expressionValueIsInteger(std::sqrt(2.0)),
                   "integer check accepts 8 and refuses sqrt(2)");

  const auto parsedRefs =
      parseExpression("x1 + x3", ExpressionIdentContext::mathInputs, 4);
  passed &= expect(parsedRefs.accepted &&
                       maxReferencedMathInput(parsedRefs.ast) == 3,
                   "max referenced input is x3");

  const auto zeroPow =
      openyourbox::graph::evaluateParameterToken("0^0", 0.0);
  if (zeroPow.ok)
    passed &= expect(std::isfinite(zeroPow.value),
                     "0^0 is accepted only when finite");
  else
    passed &= expect(!zeroPow.message.empty(),
                     "non-finite 0^0 is refused with a message");

  if (!passed)
    return 1;
  std::cout << "ExpressionGrammarTests passed\n";
  return 0;
}
