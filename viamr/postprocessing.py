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


def dedup_vars(amr: str) -> str:
    """Rename duplicate variable names so each variable is unique.

    When the model produces AMR with two nodes sharing the same variable
    (e.g. two ``(n / ...)``) the smatch scorer cannot parse the graph and
    returns 0.  This function renames the *second* (and subsequent)
    occurrences by appending an incrementing digit, both at the definition
    site ``(var / concept)`` and at every reference site.
    """
    # 1. Find all definition sites and identify duplicates.
    def_pat = re.compile(r'\(\s*([A-Za-z]\w*)\s*/')
    definitions = def_pat.findall(amr)
    if len(definitions) == len(set(definitions)):
        return amr  # no duplicates

    # 2. Build a rename map: for each duplicate occurrence (2nd, 3rd, …)
    #    assign a new unique name.
    all_vars = set(definitions)
    seen: dict[str, int] = {}
    rename_map: list[tuple[int, str, str]] = []  # (occurrence_index, old, new)
    occurrence_index = 0
    for var in definitions:
        count = seen.get(var, 0)
        seen[var] = count + 1
        if count > 0:
            # Find a name that doesn't collide.
            suffix = count
            new_var = f"{var}{suffix}"
            while new_var in all_vars:
                suffix += 1
                new_var = f"{var}{suffix}"
            all_vars.add(new_var)
            rename_map.append((occurrence_index, var, new_var))
        occurrence_index += 1

    # 3. Walk definition sites in order and apply renames.
    #    For each renamed definition we also rename all *references* to that
    #    variable that appear between this definition and the next definition
    #    of the same original variable (i.e. within the subtree scope).
    # Strategy: process the string by splitting at definition sites.
    parts: list[str] = []
    last_end = 0
    def_positions: list[tuple[int, int, str]] = []  # (start, end, var)
    for m in def_pat.finditer(amr):
        def_positions.append((m.start(), m.end(), m.group(1)))

    rename_by_idx = {r[0]: r for r in rename_map}

    for idx, (start, end, var) in enumerate(def_positions):
        # Text before this definition (or between previous def and this one)
        between = amr[last_end:start]

        # Apply pending reference renames to the "between" text.
        # References to renamed vars that were defined *before* this point.
        for prev_idx, old, new in rename_map:
            if prev_idx < idx:
                # Replace references: word-boundary match, but NOT at def sites
                between = re.sub(rf'(?<!\()\b{re.escape(old)}\b(?!\s*/)', new, between)

        parts.append(between)

        # The definition site itself
        if idx in rename_by_idx:
            _, old, new = rename_by_idx[idx]
            parts.append(amr[start:end].replace(old, new, 1))
        else:
            parts.append(amr[start:end])

        last_end = end

    # Remaining text after last definition
    tail = amr[last_end:]
    for prev_idx, old, new in rename_map:
        tail = re.sub(rf'(?<!\()\b{re.escape(old)}\b(?!\s*/)', new, tail)
    parts.append(tail)

    return ''.join(parts)


def penman_safe_minimal(amr: str, roles_to_dedup=()) -> str:
    """Minimal sanitization pipeline for AMR/PENMAN strings."""
    s = amr
    s = normalize_roles_spacing(s)
    s = join_concepts_underscores(s)
    s = fix_amr_vars(s)
    s = strip_orphan_slashes(s)
    s = dedup_vars(s)
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
