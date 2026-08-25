# compiler/parser.py

from .lexer      import NUMBER, IDENT, PLUS, MINUS, STAR, SLASH, LPAREN, RPAREN, EOF
from .ast_nodes  import Number, Var, BinOp

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0
        self.current = tokens[0]

    def advance(self):
        """Move to next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]

    def eat(self, type):
        """Consume current token if it matches expected type, else error."""
        if self.current.type == type:
            tok = self.current
            self.advance()
            return tok
        raise SyntaxError(
            f"Expected {type}, got {self.current.type!r} ({self.current.value!r})"
        )

    # ── precedence level 3 (highest): numbers, vars, parenthesised expressions
    def factor(self):
        tok = self.current

        if tok.type == NUMBER:
            self.advance()
            return Number(tok.value)

        if tok.type == IDENT:
            self.advance()
            return Var(tok.value)

        if tok.type == LPAREN:
            self.advance()        # consume '('
            node = self.expr()    # parse inner expression
            self.eat(RPAREN)      # consume ')'
            return node

        raise SyntaxError(f"Unexpected token: {tok!r}")

    # ── precedence level 2: * and /
    def term(self):
        node = self.factor()     # left side

        while self.current.type in (STAR, SLASH):
            op  = self.current.value
            self.advance()
            node = BinOp(op, node, self.factor())   # fold into tree

        return node

    # ── precedence level 1 (lowest): + and -
    def expr(self):
        node = self.term()       # left side

        while self.current.type in (PLUS, MINUS):
            op  = self.current.value
            self.advance()
            node = BinOp(op, node, self.term())     # fold into tree

        return node

    def parse(self):
        node = self.expr()
        self.eat(EOF)       # make sure we consumed everything
        return node