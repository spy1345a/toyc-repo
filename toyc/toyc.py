# compiler/toyc.py

import json
from .lexer     import Lexer
from .parser    import Parser
from .ast_nodes import node_from_dict

__version__ = "0.1.0"

def compile_to_file(source: str, out_path: str):
    """Lex + parse source, serialize AST to .toyc file."""
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()

    payload = {
        "version": __version__,
        "source":  source,
        "ast":     ast.to_dict(),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Compiled → {out_path}")


def load_from_file(path: str):
    """Read a .toyc file and reconstruct the AST."""
    with open(path) as f:
        payload = json.load(f)

    version = payload["version"]
    source  = payload["source"]
    ast     = node_from_dict(payload["ast"])

    print(f"Loaded {path!r}  (compiled with v{version})")
    print(f"Source: {source!r}")
    return ast