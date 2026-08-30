# compiler/parser.py

from .lexer     import NUMBER, IDENT, PLUS, MINUS, STAR, SLASH, LPAREN, RPAREN, EOF
from .ast_nodes import Number, Var, BinOp

class Parser:
    def __init__(self, tokens):
        self.tokens  = tokens
        self.pos     = 0
        self.current = tokens[0]

    # ── public API ───────────────────────────────────────────────────────────
    @classmethod
    def parse(cls, tokens):
        """Parse a token list and return the AST root node."""
        return cls(tokens)._run()

    # ── internal helpers ─────────────────────────────────────────────────────
    def _run(self):
        node = self._expr()
        self._eat(EOF)
        return node

    def _advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]

    def _eat(self, type):
        if self.current.type == type:
            tok = self.current
            self._advance()
            return tok
        raise SyntaxError(
            f"Expected {type}, got {self.current.type!r} ({self.current.value!r})"
        )

    # ── precedence level 3 (highest): numbers, vars, parenthesised expressions
    def _factor(self):
        tok = self.current

        if tok.type == NUMBER:
            self._advance()
            return Number(tok.value)

        if tok.type == IDENT:
            self._advance()
            return Var(tok.value)

        if tok.type == LPAREN:
            self._advance()
            node = self._expr()
            self._eat(RPAREN)
            return node

        raise SyntaxError(f"Unexpected token: {tok!r}")

    # ── precedence level 2: * and /
    def _term(self):
        node = self._factor()

        while self.current.type in (STAR, SLASH):
            op = self.current.value
            self._advance()
            node = BinOp(op, node, self._factor())

        return node

    # ── precedence level 1 (lowest): + and -
    def _expr(self):
        node = self._term()

        while self.current.type in (PLUS, MINUS):
            op = self.current.value
            self._advance()
            node = BinOp(op, node, self._term())

        return node