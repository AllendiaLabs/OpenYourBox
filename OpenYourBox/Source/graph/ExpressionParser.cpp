#include "ExpressionParser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace openyourbox::graph {
namespace {
/** @brief Token kinds produced by the expression lexer. */
enum class TokenKind {
  number,
  ident,
  plus,
  minus,
  star,
  slash,
  caret,
  lparen,
  rparen,
  end,
  invalid
};

/** @brief One lexical token. */
struct Token {
  TokenKind kind = TokenKind::end;
  std::string text;
  double number = 0.0;
};

/**
 * @brief Reads the next token from @p text starting at @p offset.
 * @param text Full source string.
 * @param offset In/out byte offset.
 * @return The next token (invalid on characters outside the grammar).
 */
Token nextToken(std::string_view text, std::size_t &offset) {
  while (offset < text.size() &&
         std::isspace(static_cast<unsigned char>(text[offset])))
    ++offset;
  Token token;
  if (offset >= text.size()) {
    token.kind = TokenKind::end;
    return token;
  }
  const auto ch = text[offset];
  if (ch == '+') {
    ++offset;
    token.kind = TokenKind::plus;
    token.text = "+";
    return token;
  }
  if (ch == '-') {
    ++offset;
    token.kind = TokenKind::minus;
    token.text = "-";
    return token;
  }
  if (ch == '*') {
    ++offset;
    // `**` is an ASCII-friendly synonym for power (AZERTY `^` is a dead key).
    if (offset < text.size() && text[offset] == '*') {
      ++offset;
      token.kind = TokenKind::caret;
      token.text = "**";
      return token;
    }
    token.kind = TokenKind::star;
    token.text = "*";
    return token;
  }
  if (ch == '/') {
    ++offset;
    token.kind = TokenKind::slash;
    token.text = "/";
    return token;
  }
  if (ch == '^') {
    ++offset;
    token.kind = TokenKind::caret;
    token.text = "^";
    return token;
  }
  if (ch == '(') {
    ++offset;
    token.kind = TokenKind::lparen;
    token.text = "(";
    return token;
  }
  if (ch == ')') {
    ++offset;
    token.kind = TokenKind::rparen;
    token.text = ")";
    return token;
  }
  if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
    const auto begin = offset;
    ++offset;
    while (offset < text.size()) {
      const auto next = text[offset];
      if (!std::isalnum(static_cast<unsigned char>(next)) && next != '_')
        break;
      ++offset;
    }
    token.kind = TokenKind::ident;
    token.text = std::string(text.substr(begin, offset - begin));
    return token;
  }
  if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
    const auto begin = offset;
    auto sawDigit = false;
    auto sawDot = false;
    auto sawExp = false;
    if (ch == '.') {
      sawDot = true;
      ++offset;
    }
    while (offset < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[offset]))) {
      sawDigit = true;
      ++offset;
    }
    if (!sawDot && offset < text.size() && text[offset] == '.') {
      sawDot = true;
      ++offset;
      while (offset < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[offset]))) {
        sawDigit = true;
        ++offset;
      }
    }
    if (offset < text.size() &&
        (text[offset] == 'e' || text[offset] == 'E')) {
      const auto expPos = offset;
      ++offset;
      if (offset < text.size() &&
          (text[offset] == '+' || text[offset] == '-'))
        ++offset;
      auto expDigit = false;
      while (offset < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[offset]))) {
        expDigit = true;
        ++offset;
      }
      if (expDigit)
        sawExp = true;
      else
        offset = expPos;
    }
    if (!sawDigit) {
      token.kind = TokenKind::invalid;
      token.text = std::string(text.substr(begin, offset - begin));
      return token;
    }
    const auto slice = text.substr(begin, offset - begin);
    double value = 0.0;
    try {
      value = std::stod(std::string(slice));
    } catch (...) {
      token.kind = TokenKind::invalid;
      token.text = std::string(slice);
      return token;
    }
    (void)sawExp;
    token.kind = TokenKind::number;
    token.text = std::string(slice);
    token.number = value;
    return token;
  }
  ++offset;
  token.kind = TokenKind::invalid;
  token.text = std::string(1, ch);
  return token;
}

/**
 * @brief Recursive-descent parser for the shared infix grammar.
 */
class Parser {
public:
  /**
   * @brief Prepares a parser over @p source.
   * @param source Expression text.
   * @param context Identifier allow-list.
   * @param maxInput Highest legal `xK` (Math context).
   */
  Parser(std::string_view source, ExpressionIdentContext context, int maxInput)
      : text(source), identContext(context), maxInputIndex(std::max(0, maxInput)) {
    advance();
  }

  /**
   * @brief Parses a complete expression.
   * @return Result with AST or a refuse message.
   */
  ExpressionParseResult parse() {
    ExpressionParseResult result;
    if (current.kind == TokenKind::end) {
      result.message = "Expression is empty";
      return result;
    }
    auto ast = parseAdd();
    if (!ok) {
      result.message = error;
      return result;
    }
    if (current.kind == TokenKind::invalid) {
      result.message = "Invalid token '" + current.text + "'";
      return result;
    }
    if (current.kind != TokenKind::end) {
      result.message = "Unexpected '" + current.text + "' in expression";
      return result;
    }
    if (ast.instructions.empty()) {
      result.message = "Expression is empty";
      return result;
    }
    result.accepted = true;
    result.ast = std::move(ast);
    return result;
  }

private:
  void advance() { current = nextToken(text, offset); }

  void fail(std::string message) {
    if (ok) {
      ok = false;
      error = std::move(message);
    }
  }

  ExpressionAst parseAdd() {
    auto left = parseMul();
    while (ok && (current.kind == TokenKind::plus ||
                  current.kind == TokenKind::minus)) {
      const auto op = current.kind == TokenKind::plus
                          ? ExpressionInstruction::Op::add
                          : ExpressionInstruction::Op::subtract;
      advance();
      auto right = parseMul();
      if (!ok)
        return {};
      left = concat(std::move(left), std::move(right), op);
    }
    return left;
  }

  ExpressionAst parseMul() {
    auto left = parsePow();
    while (ok && (current.kind == TokenKind::star ||
                  current.kind == TokenKind::slash)) {
      const auto op = current.kind == TokenKind::star
                          ? ExpressionInstruction::Op::multiply
                          : ExpressionInstruction::Op::divide;
      advance();
      auto right = parsePow();
      if (!ok)
        return {};
      left = concat(std::move(left), std::move(right), op);
    }
    return left;
  }

  ExpressionAst parsePow() {
    auto base = parseUnary();
    if (!ok)
      return {};
    if (current.kind != TokenKind::caret)
      return base;
    advance();
    auto exponent = parsePow();
    if (!ok)
      return {};
    return concat(std::move(base), std::move(exponent),
                  ExpressionInstruction::Op::power);
  }

  ExpressionAst parseUnary() {
    if (current.kind == TokenKind::minus) {
      advance();
      auto inner = parseUnary();
      if (!ok)
        return {};
      ExpressionInstruction neg;
      neg.op = ExpressionInstruction::Op::negate;
      inner.instructions.push_back(neg);
      return inner;
    }
    return parsePrimary();
  }

  ExpressionAst parsePrimary() {
    if (current.kind == TokenKind::number) {
      ExpressionAst ast;
      ExpressionInstruction push;
      push.op = ExpressionInstruction::Op::pushLiteral;
      push.literal = current.number;
      ast.instructions.push_back(push);
      advance();
      return ast;
    }
    if (current.kind == TokenKind::ident) {
      if (current.text == "exp") {
        advance();
        if (current.kind != TokenKind::lparen) {
          fail("exp requires parentheses: exp(...)");
          return {};
        }
        advance();
        auto inner = parseAdd();
        if (!ok)
          return {};
        if (current.kind != TokenKind::rparen) {
          fail("Missing closing parenthesis");
          return {};
        }
        advance();
        ExpressionInstruction exp;
        exp.op = ExpressionInstruction::Op::exp;
        inner.instructions.push_back(exp);
        return inner;
      }
      auto ast = bindIdent(current.text);
      advance();
      return ast;
    }
    if (current.kind == TokenKind::lparen) {
      advance();
      auto inner = parseAdd();
      if (!ok)
        return {};
      if (current.kind != TokenKind::rparen) {
        fail("Missing closing parenthesis");
        return {};
      }
      advance();
      return inner;
    }
    if (current.kind == TokenKind::end)
      fail("Expression is incomplete");
    else
      fail("Expected a number, identifier, exp(...), or '('");
    return {};
  }

  ExpressionAst bindIdent(const std::string &name) {
    ExpressionAst ast;
    ExpressionInstruction push;
    push.op = ExpressionInstruction::Op::pushIdent;
    if (identContext == ExpressionIdentContext::parameterIndex) {
      if (name != "i") {
        fail("Unknown symbol '" + name + "'; parameter expressions may use i");
        return {};
      }
      push.identIndex = 0;
      ast.usesIndexI = true;
      ast.instructions.push_back(push);
      return ast;
    }
    if (name.size() < 2 || name[0] != 'x') {
      fail("Unknown symbol '" + name + "'; use x1, x2, … for inputs");
      return {};
    }
    int index = 0;
    for (std::size_t pos = 1; pos < name.size(); ++pos) {
      const auto digit = name[pos];
      if (digit < '0' || digit > '9') {
        fail("Unknown symbol '" + name + "'; use x1, x2, … for inputs");
        return {};
      }
      index = index * 10 + (digit - '0');
    }
    if (index < 1) {
      fail("Unknown symbol '" + name + "'; use x1, x2, … for inputs");
      return {};
    }
    if (index > maxInputIndex) {
      fail("'" + name + "' is not a configured input (Inputs = " +
           std::to_string(maxInputIndex) + ")");
      return {};
    }
    push.identIndex = index;
    ast.referencedInputs.push_back(index);
    ast.instructions.push_back(push);
    return ast;
  }

  static ExpressionAst concat(ExpressionAst left, ExpressionAst right,
                              ExpressionInstruction::Op op) {
    left.instructions.insert(left.instructions.end(),
                             right.instructions.begin(),
                             right.instructions.end());
    ExpressionInstruction instruction;
    instruction.op = op;
    left.instructions.push_back(instruction);
    left.usesIndexI = left.usesIndexI || right.usesIndexI;
    left.referencedInputs.insert(left.referencedInputs.end(),
                                 right.referencedInputs.begin(),
                                 right.referencedInputs.end());
    return left;
  }

  std::string_view text;
  ExpressionIdentContext identContext = ExpressionIdentContext::mathInputs;
  int maxInputIndex = 0;
  std::size_t offset = 0;
  Token current;
  bool ok = true;
  std::string error;
};

/**
 * @brief Deduplicates and sorts referenced Math pin indices.
 * @param ast Program whose @ref ExpressionAst::referencedInputs is compacted.
 */
void normalizeReferencedInputs(ExpressionAst &ast) {
  std::sort(ast.referencedInputs.begin(), ast.referencedInputs.end());
  ast.referencedInputs.erase(
      std::unique(ast.referencedInputs.begin(), ast.referencedInputs.end()),
      ast.referencedInputs.end());
}
} // namespace

ExpressionParseResult parseExpression(std::string_view text,
                                      ExpressionIdentContext context,
                                      int maxInputIndex) {
  Parser parser(text, context, maxInputIndex);
  auto result = parser.parse();
  if (result.accepted)
    normalizeReferencedInputs(result.ast);
  return result;
}

ExpressionEvalResult evaluateExpression(const ExpressionAst &ast, double indexI,
                                        const std::vector<double> &mathValues) {
  ExpressionEvalResult result;
  if (ast.instructions.empty()) {
    result.message = "Expression is empty";
    return result;
  }
  std::vector<double> stack;
  stack.reserve(ast.instructions.size());
  const auto binary = [&](auto op, const char *name) -> bool {
    if (stack.size() < 2) {
      result.message = std::string("Invalid ") + name + " expression";
      return false;
    }
    const auto b = stack.back();
    stack.pop_back();
    const auto a = stack.back();
    stack.pop_back();
    stack.push_back(op(a, b));
    return true;
  };
  for (const auto &instruction : ast.instructions) {
    switch (instruction.op) {
    case ExpressionInstruction::Op::pushLiteral:
      stack.push_back(instruction.literal);
      break;
    case ExpressionInstruction::Op::pushIdent:
      if (instruction.identIndex == 0) {
        stack.push_back(indexI);
      } else {
        const auto slot =
            static_cast<std::size_t>(instruction.identIndex - 1);
        if (slot >= mathValues.size()) {
          result.message = "Missing value for x" +
                           std::to_string(instruction.identIndex);
          return result;
        }
        stack.push_back(mathValues[slot]);
      }
      break;
    case ExpressionInstruction::Op::negate:
      if (stack.empty()) {
        result.message = "Invalid negation";
        return result;
      }
      stack.back() = -stack.back();
      break;
    case ExpressionInstruction::Op::add:
      if (!binary([](double a, double b) { return a + b; }, "add"))
        return result;
      break;
    case ExpressionInstruction::Op::subtract:
      if (!binary([](double a, double b) { return a - b; }, "subtract"))
        return result;
      break;
    case ExpressionInstruction::Op::multiply:
      if (!binary([](double a, double b) { return a * b; }, "multiply"))
        return result;
      break;
    case ExpressionInstruction::Op::divide:
      if (!binary([](double a, double b) { return a / b; }, "divide"))
        return result;
      break;
    case ExpressionInstruction::Op::power:
      if (!binary([](double a, double b) { return std::pow(a, b); }, "power"))
        return result;
      break;
    case ExpressionInstruction::Op::exp:
      if (stack.empty()) {
        result.message = "Invalid exp()";
        return result;
      }
      stack.back() = std::exp(stack.back());
      break;
    }
  }
  if (stack.size() != 1) {
    result.message = "Invalid expression";
    return result;
  }
  if (!std::isfinite(stack.front())) {
    result.message = "Expression result is not a finite number";
    return result;
  }
  result.ok = true;
  result.value = stack.front();
  return result;
}

ExpressionEvalResult evaluateParameterToken(std::string_view token,
                                            double indexI) {
  const auto parsed =
      parseExpression(token, ExpressionIdentContext::parameterIndex, 0);
  if (!parsed.accepted) {
    ExpressionEvalResult result;
    result.message = parsed.message;
    return result;
  }
  return evaluateExpression(parsed.ast, indexI, {});
}

int maxReferencedMathInput(const ExpressionAst &ast) noexcept {
  int maxIndex = 0;
  for (const auto index : ast.referencedInputs)
    maxIndex = std::max(maxIndex, index);
  return maxIndex;
}

bool mathExpressionReferencesInput(const ExpressionAst &ast,
                                   int oneBasedIndex) noexcept {
  for (const auto index : ast.referencedInputs) {
    if (index == oneBasedIndex)
      return true;
  }
  return false;
}
} // namespace openyourbox::graph
