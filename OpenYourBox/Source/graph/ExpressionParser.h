#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace openyourbox::graph {
/**
 * @brief Identifier binding used when parsing an expression.
 *
 * Math Expression nodes bind `x1`…`xN`. Copy-expanded numeric parameters bind
 * `i` only.
 */
enum class ExpressionIdentContext {
  /** @brief Allow `x1`…`xN` where N is the configured input count. */
  mathInputs,
  /** @brief Allow only the copy-index identifier `i`. */
  parameterIndex
};

/**
 * @brief One postfix instruction in a compiled expression.
 *
 * The program is built on the GUI/document thread and later evaluated with
 * either scalar bindings (parameters) or prepared tensors (live/freeze).
 */
struct ExpressionInstruction {
  /** @brief Postfix opcode. */
  enum class Op {
    /** @brief Push @ref literal onto the stack. */
    pushLiteral,
    /** @brief Push the bound identifier (@ref identIndex). */
    pushIdent,
    /** @brief Pop one value and push its negation. */
    negate,
    /** @brief Pop b, pop a, push a+b. */
    add,
    /** @brief Pop b, pop a, push a-b. */
    subtract,
    /** @brief Pop b, pop a, push a*b. */
    multiply,
    /** @brief Pop b, pop a, push a/b (same precedence as multiply). */
    divide,
    /** @brief Pop b, pop a, push a^b (right-associative at parse time). */
    power,
    /** @brief Pop one value and push exp(a). */
    exp
  };

  /** @brief Opcode for this step. */
  Op op = Op::pushLiteral;
  /** @brief Literal payload when @ref op is @ref Op::pushLiteral. */
  double literal = 0.0;
  /**
   * @brief Identifier payload when @ref op is @ref Op::pushIdent.
   *
   * Parameter context uses 0 for `i`. Math context uses 1…N for `x1`…`xN`.
   */
  int identIndex = 0;
};

/**
 * @brief Compiled infix expression (source is persisted; this is a cache).
 */
struct ExpressionAst {
  /** @brief Postfix program. Empty when parsing failed. */
  std::vector<ExpressionInstruction> instructions;
  /**
   * @brief 1-based Math Expression pin indices referenced by the program.
   *
   * Empty when the expression has no `xK` identifiers.
   */
  std::vector<int> referencedInputs;
  /** @brief True when the program reads parameter index `i`. */
  bool usesIndexI = false;
};

/**
 * @brief Outcome of parsing an expression string.
 */
struct ExpressionParseResult {
  /** @brief True when the text is a legal expression for the context. */
  bool accepted = false;
  /** @brief User-facing reason when @ref accepted is false. */
  std::string message;
  /** @brief Compiled program when accepted. */
  ExpressionAst ast;
};

/**
 * @brief Outcome of a scalar evaluation.
 */
struct ExpressionEvalResult {
  /** @brief True when the result is finite (and integer, when required). */
  bool ok = false;
  /** @brief Evaluated value when @ref ok is true. */
  double value = 0.0;
  /** @brief User-facing reason when @ref ok is false. */
  std::string message;
};

/**
 * @brief Absolute tolerance used to decide whether a scalar is a whole number.
 *
 * Values whose distance to the nearest integer is at most this tolerance are
 * treated as integers; others are refused for integer-typed parameters.
 */
inline constexpr double expressionIntegerTolerance = 1.0e-9;

/**
 * @brief Placeholder shown on the Math Expression field (FR-014).
 */
inline constexpr const char *mathExpressionPlaceholder = "2 * x1**2.3 + 1e-7";

/**
 * @brief Placeholder shown on numeric parameter fields (FR-014).
 */
inline constexpr const char *parameterExpressionPlaceholder = "2*i+1";

/**
 * @brief Tooltip documenting the shared grammar (FR-014).
 */
inline constexpr const char *expressionGrammarTooltip =
    "Operators: ( ) + - * / ^ or **  (^ / ** is power, right-associative). "
    "Prefer ** if your keyboard uses ^ as an accent dead key (e.g. AZERTY). "
    "exp(x) is the natural exponential. "
    "Math Expression uses x1, x2, … for inputs. "
    "Parameters use i as the 0-based copy index (i is 0 when ungrouped). "
    "Write 2*i, not 2i. Scientific literals such as 1e-7 are allowed.";

/**
 * @brief Returns true when @p value is a finite whole number.
 * @param value Candidate scalar.
 */
inline bool expressionValueIsInteger(double value) noexcept {
  if (!std::isfinite(value))
    return false;
  const auto nearest = std::nearbyint(value);
  return std::abs(value - nearest) <= expressionIntegerTolerance;
}

/**
 * @brief Parses @p text under @p context.
 * @param text User-authored expression (whitespace ignored between tokens).
 * @param context Identifier allow-list.
 * @param maxInputIndex Highest legal `xK` index (Math context); ignored for `i`.
 * @return Accepted AST or a refuse message. Invalid strings are never stored
 *         by callers.
 */
ExpressionParseResult parseExpression(std::string_view text,
                                      ExpressionIdentContext context,
                                      int maxInputIndex = 0);

/**
 * @brief Evaluates @p ast with scalar identifier bindings.
 * @param ast Compiled program from @ref parseExpression.
 * @param indexI Value of `i` (parameter context).
 * @param mathValues Values of `x1`…`xN` at indices 0…N-1 (Math context).
 * @return Finite scalar, or an error when the stack is invalid or non-finite.
 */
ExpressionEvalResult evaluateExpression(
    const ExpressionAst &ast, double indexI = 0.0,
    const std::vector<double> &mathValues = {});

/**
 * @brief Parses and evaluates a parameter-field token at copy index @p indexI.
 * @param token One list entry or a whole-field expression.
 * @param indexI Expanded slot index (`i`).
 * @return Finite scalar, or a refuse result.
 */
ExpressionEvalResult evaluateParameterToken(std::string_view token,
                                            double indexI);

/**
 * @brief Highest 1-based `xK` referenced by @p ast, or 0 when none.
 * @param ast Compiled Math Expression program.
 */
int maxReferencedMathInput(const ExpressionAst &ast) noexcept;

/**
 * @brief Returns true when @p ast references Math pin @p oneBasedIndex.
 * @param ast Compiled program.
 * @param oneBasedIndex Pin index starting at 1 (`x1`).
 */
bool mathExpressionReferencesInput(const ExpressionAst &ast,
                                   int oneBasedIndex) noexcept;
} // namespace openyourbox::graph
