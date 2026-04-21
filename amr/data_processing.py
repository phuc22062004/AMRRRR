"""Parse raw AMR text files into pandas DataFrames."""
import contextlib
import io
import re

import pandas as pd
import penman
from penman.models.noop import NoOpModel


def penman_to_one_line(penman_str: str) -> str:
    lines = penman_str.strip().split('\n')
    one_line = ' '.join(line.strip() for line in lines)
    return re.sub(r'\s+', ' ', one_line)


def fix_missing_closing_brackets(graph_str: str) -> str:
    missing = graph_str.count('(') - graph_str.count(')')
    if missing > 0:
        graph_str += ')' * missing
    return graph_str


def fix_multiword_nodes(graph_str: str) -> str:
    def repl(match):
        return '/ ' + match.group(1).replace(' ', '_')
    return re.sub(r'/ ([^\(\):]+)', repl, graph_str)


def decode_with_warnings(graph_str: str, sent: str):
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        try:
            graph = penman.decode(graph_str, model=NoOpModel())
            warnings = f.getvalue()
            if warnings.strip():
                print(f"Warning(s) during decoding sentence: {sent}")
                print(warnings)
            return graph, None
        except Exception as e:
            return None, e


_META_SNT_RE = re.compile(r'^#\s*::snt\s+(.*)$')
_META_TOK_RE = re.compile(r'^#\s*::tok\s+(.*)$')
_META_ANY_RE = re.compile(r'^#\s*::')


def read_amr_direct(
    filename: str,
    one_line: bool = True,
    prefer_tok: bool = True,
) -> pd.DataFrame:
    """Read an AMR file into a DataFrame with 'query' and 'amr' columns.

    Handles standard AMR/ViAMR metadata lines such as ``# ::snt``,
    ``# ::tok`` and ``# ::id``. When ``prefer_tok`` is True, the already
    word-segmented ``# ::tok`` line is used as the query (matching how
    AMR concepts are tokenized, e.g. ``tình_thương``).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    queries, amr_list = [], []
    current_sent: str | None = None
    current_tok: str | None = None
    current_graph_lines: list[str] = []

    def flush():
        nonlocal current_sent, current_tok, current_graph_lines
        if current_sent is None or not current_graph_lines:
            current_sent, current_tok, current_graph_lines = None, None, []
            return
        graph_str = "\n".join(current_graph_lines).strip()
        graph_str = fix_missing_closing_brackets(graph_str)
        graph_str = fix_multiword_nodes(graph_str)
        graph, error = decode_with_warnings(graph_str, current_sent)
        if not error:
            amr_str = penman.encode(graph, model=NoOpModel())
            if one_line:
                amr_str = penman_to_one_line(amr_str)
            query = current_tok if (prefer_tok and current_tok) else current_sent
            query = re.sub(r'\s+', ' ', query).strip()
            queries.append(query)
            amr_list.append(amr_str)
        current_sent, current_tok, current_graph_lines = None, None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current_graph_lines:
                flush()
            continue
        m_snt = _META_SNT_RE.match(line)
        if m_snt:
            if current_graph_lines:
                flush()
            current_sent = m_snt.group(1).strip()
            continue
        m_tok = _META_TOK_RE.match(line)
        if m_tok:
            current_tok = m_tok.group(1).strip()
            continue
        if _META_ANY_RE.match(line):
            continue
        current_graph_lines.append(line)
    flush()

    return pd.DataFrame({"query": queries, "amr": amr_list})
