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

    def advance(self):
        self.pos += 1
        self.current = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current and self.current.isspace():
            self.advance()

    def read_number(self):
        result = ""
        while self.current and (self.current.isdigit() or self.current == "."):
            result += self.current
            self.advance()
        return Token(NUMBER, float(result) if "." in result else int(result))

    def read_ident(self):
        result = ""
        while self.current and (self.current.isalnum() or self.current == "_"):
            result += self.current
            self.advance()
        type = result.upper() if result in self.KEYWORDS else IDENT
        return Token(type, result)

    def next_token(self):
        while self.current:
            if self.current.isspace():
                self.skip_whitespace(); continue
            if self.current.isdigit():
                return self.read_number()
            if self.current.isalpha() or self.current == "_":
                return self.read_ident()
            ops = {"+": PLUS, "-": MINUS, "*": STAR, "/": SLASH,
                   "(": LPAREN, ")": RPAREN}
            if self.current in ops:
                tok = Token(ops[self.current], self.current)
                self.advance()
                return tok
            raise SyntaxError(f"Unknown character: {self.current!r}")
        return Token(EOF, None)

    def tokenize(self):
        tokens = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == EOF:
                break
        return tokens


# --- test it ---
if __name__ == "__main__":
    tests = [
        "a + b * 2",
        "(a + b) * (c - 3)",
        "10.5 / x + y * 2",
        "kernel",         # keyword test
    ]
    for src in tests:
        print(f"\nInput:  {src!r}")
        for tok in Lexer(src).tokenize():
            print(f"  {tok}")