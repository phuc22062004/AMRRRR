"""AMR/PENMAN string sanitization utilities."""
import re


def join_concepts_underscores(text: str) -> str:
    """Replace spaces in multi-word concepts (following '/') with underscores."""
    rx = re.compile(r'/\s+([^\n\r():/]+?)(?=\s*[:/)]|\s*$)', re.UNICODE)

    def _repl(m):
        return '/ ' + re.sub(r'\s+', '_', m.group(1).strip())

    return rx.sub(_repl, text)


def fix_amr_vars(amr_str: str) -> str:
    """Remove redundant concept definitions in the head of a node."""
    pat = re.compile(r'(\(\s*\w+\s*/\s*[^()\s:]+)\s*/\s*[^()\s:]+', re.UNICODE)
    fixed = amr_str
    while True:
        new_fixed = pat.sub(r'\1', fixed)
        if new_fixed == fixed:
            break
        fixed = new_fixed
    return fixed


def normalize_roles_spacing(s: str) -> str:
    """Normalize whitespace around AMR roles."""
    s = re.sub(r'(?<!:)([^\s():]+):([A-Za-z][\w-]*)', r'\1 :\2', s)
    s = re.sub(r'(:[\w-]+)\(', r'\1 (', s)
    s = re.sub(r'(:[\w-]+)_([^\s():]+)', r'\1 \2', s)
    return s


def strip_orphan_slashes(s: str) -> str:
    """Remove stray '/' that are not followed by a valid concept."""
    return re.sub(r'/\s*(?=\)|:[\w-]|\n|$)', '', s)


def balance_parens(s: str) -> str:
    """Remove unmatched ')' and append missing ')'."""
    out, depth = [], 0
    for ch in s:
        if ch == '(':
            depth += 1
            out.append(ch)
        elif ch == ')':
            if depth > 0:
                depth -= 1
                out.append(ch)
        else:
            out.append(ch)
    if depth > 0:
        out.append(')' * depth)
    return ''.join(out)


def dedup_selected_roles(amr: str, roles=()):
    """Remove duplicate occurrences of ':role var' for the specified roles."""
    if not roles:
        return amr
    role_pat = "|".join(re.escape(r[1:] if r.startswith(":") else r) for r in roles)
    rx = re.compile(rf'(\s:(?P<role>{role_pat})\s+(?P<var>[A-Za-z][A-Za-z0-9_-]*))(?!\s*/)')
    seen = set()

    def _sub(m):
        key = (m.group("role"), m.group("var"))
        if key in seen:
            return ""
        seen.add(key)
        return m.group(1)

    return rx.sub(_sub, amr)


def penman_safe_minimal(amr: str, roles_to_dedup=()) -> str:
    """Minimal sanitization pipeline for AMR/PENMAN strings."""
    s = amr
    s = normalize_roles_spacing(s)
    s = join_concepts_underscores(s)
    s = fix_amr_vars(s)
    s = strip_orphan_slashes(s)
    s = balance_parens(s)
    if roles_to_dedup:
        s = dedup_selected_roles(s, roles=roles_to_dedup)
    s = re.sub(r'[ \t]+', ' ', s).strip()
    return s


def has_duplicate_nodes(amr_str: str) -> bool:
    """Return True if an AMR string contains duplicate variable names."""
    pattern = re.compile(r'\(\s*([^\s/()]+)\s*/')
    nodes = pattern.findall(amr_str)
    seen = set()
    for var in nodes:
        if var in seen:
            print(f"Duplicate node found: {var}")
            return True
        seen.add(var)
    return False
