NUMBER = "NUMBER"
IDENT  = "IDENT"
PLUS   = "PLUS"
MINUS  = "MINUS"
STAR   = "STAR"
SLASH  = "SLASH"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
EOF    = "EOF"

class Token:
    def __init__(self, type, value):
        self.type  = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    KEYWORDS = {"kernel", "return"}

    def __init__(self, text):
        self.text    = text
        self.pos     = 0
        self.current = text[0] if text else None

    # ── public API ──────────────────────────────────────────────────────────
    @classmethod
    def tokenize(cls, text):
        """Lex *text* and return a list of Tokens (including the EOF token)."""
        return cls(text)._run()

    # ── internal helpers ─────────────────────────────────────────────────────
    def _run(self):
        tokens = []
        while True:
            tok = self._next_token()
            tokens.append(tok)
            if tok.type == EOF:
                break
        return tokens

    def _advance(self):
        self.pos += 1
        self.current = self.text[self.pos] if self.pos < len(self.text) else None

    def _skip_whitespace(self):
        while self.current and self.current.isspace():
            self._advance()

    def _read_number(self):
        result = ""
        while self.current and (self.current.isdigit() or self.current == "."):
            result += self.current
            self._advance()
        return Token(NUMBER, float(result) if "." in result else int(result))

    def _read_ident(self):
        result = ""
        while self.current and (self.current.isalnum() or self.current == "_"):
            result += self.current
            self._advance()
        tok_type = result.upper() if result in self.KEYWORDS else IDENT
        return Token(tok_type, result)

    _OPS = {"+": PLUS, "-": MINUS, "*": STAR, "/": SLASH,
            "(": LPAREN, ")": RPAREN}

    def _next_token(self):
        while self.current:
            if self.current.isspace():
                self._skip_whitespace()
                continue
            if self.current.isdigit():
                return self._read_number()
            if self.current.isalpha() or self.current == "_":
                return self._read_ident()
            if self.current in self._OPS:
                tok = Token(self._OPS[self.current], self.current)
                self._advance()
                return tok
            raise SyntaxError(f"Unknown character: {self.current!r}")
        return Token(EOF, None)


# ── test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "a + b * 2",
        "(a + b) * (c - 3)",
        "10.5 / x + y * 2",
        "kernel",
    ]
    for src in tests:
        print(f"\nInput:  {src!r}")
        for tok in Lexer.tokenize(src):      # ← clean API
            print(f"  {tok}")