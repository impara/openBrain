from __future__ import annotations

import re


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "your",
    "you",
    "does",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "into",
    "about",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "is",
    "to",
    "of",
    "in",
    "on",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9@._:-]{1,}")
_CITATION_RE = re.compile(r"\b\d{1,3}:\d{1,3}\b")
_CLI_FLAG_RE = re.compile(r"(?:^|\s)--?[A-Za-z0-9][\w-]*")
_ENV_ASSIGN_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```|~~~")
_IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+\S+\s+import\s+\S+|import\s+\S+)\s*$", re.MULTILINE)
_DEF_LINE_RE = re.compile(r"^\s*(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*", re.MULTILINE)
_SQL_LINE_RE = re.compile(r"^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b", re.MULTILINE)
_SHELL_COMMAND_PREFIXES = {
    "apt",
    "aws",
    "bash",
    "brew",
    "cargo",
    "chmod",
    "chown",
    "composer",
    "cp",
    "curl",
    "docker",
    "find",
    "git",
    "go",
    "grep",
    "helm",
    "jq",
    "kubectl",
    "ls",
    "make",
    "mkdir",
    "mv",
    "nix",
    "npm",
    "npx",
    "pip",
    "pnpm",
    "poetry",
    "psql",
    "python",
    "python3",
    "rm",
    "rsync",
    "scp",
    "sed",
    "ssh",
    "systemctl",
    "terraform",
    "uv",
    "wget",
    "yarn",
}


def looks_like_natural_language(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 8:
        return False
    stopword_hits = sum(1 for word in words if word.lower() in _STOPWORDS)
    sentence_like = len(re.findall(r"[.!?]", text))
    return stopword_hits >= 4 and sentence_like >= 1


def detect_auto_ingest_strategy(text: str, source: str = "import") -> str:
    del source
    clean = (text or "").strip()
    if not clean:
        return "personal"

    lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
    first_line = lines[0].strip() if lines else clean
    first_token = first_line.lstrip("$># ").split(maxsplit=1)[0].lower() if first_line else ""
    symbol_ratio = (
        sum(1 for ch in clean if not ch.isalnum() and not ch.isspace()) / max(len(clean), 1)
    )
    sentence_count = len(re.findall(r"[.!?]", clean))
    word_count = len(re.findall(r"\b\w+\b", clean))

    score = 0
    if _CODE_FENCE_RE.search(clean):
        score += 5
    if first_token in _SHELL_COMMAND_PREFIXES:
        score += 3
    if first_line.startswith(("$ ", "# ", "> ")) and first_token:
        score += 2
    if _CLI_FLAG_RE.search(clean):
        score += 1
    if _ENV_ASSIGN_RE.search(clean):
        score += 2
    if _IMPORT_LINE_RE.search(clean) or _DEF_LINE_RE.search(clean) or _SQL_LINE_RE.search(clean):
        score += 3
    if len(lines) >= 2 and symbol_ratio >= 0.14:
        score += 2
    if any(char in clean for char in ("{", "}", "[", "]", "=>", "::", "</", "/>")):
        score += 1
    if len(lines) >= 3 and any(line.startswith(("  ", "\t")) for line in lines):
        score += 2

    if looks_like_natural_language(clean):
        score -= 2
    if word_count >= 80 and sentence_count >= 3:
        score -= 2

    return "snippet" if score >= 3 else "personal"


def chunk_text_sentences(text: str, max_chunk_chars: int = 900, max_chunks: int = 6) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    if not sentences:
        fallback = [s.strip() for s in re.split(r"[\n;]+", text or "") if s.strip()]
        sentences = fallback if fallback else [text.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        proposed = sentence if not current else f"{current} {sentence}"
        if len(proposed) > max_chunk_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = proposed
        if len(chunks) >= max_chunks:
            break

    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def chunk_text_passages(text: str, max_chunk_chars: int = 420, max_chunks: int = 12) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    blocks = [" ".join(block.split()) for block in re.split(r"\n{2,}", raw) if block.strip()]
    if not blocks:
        return chunk_text_sentences(raw, max_chunk_chars=max_chunk_chars, max_chunks=max_chunks)

    chunks: list[str] = []
    for block in blocks:
        if len(chunks) >= max_chunks:
            break
        if len(block) <= max_chunk_chars:
            chunks.append(block)
            continue
        for sentence_chunk in chunk_text_sentences(
            block,
            max_chunk_chars=max_chunk_chars,
            max_chunks=max_chunks - len(chunks),
        ):
            chunks.append(sentence_chunk)
            if len(chunks) >= max_chunks:
                break

    if not chunks:
        return chunk_text_sentences(raw, max_chunk_chars=max_chunk_chars, max_chunks=max_chunks)

    # Preserve citation-bearing blocks and drop only exact duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        clean = " ".join(chunk.split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped[:max_chunks]


def heuristic_bullets(chunk: str, max_bullets: int = 3) -> list[str]:
    sentences = [s.strip("•*- ").strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    prioritized: list[str] = []
    keywords = ("channel", "frequency", "standard", "limit", "mode", "uses", "allowed", "reserved")
    for sentence in sentences:
        lower = sentence.lower()
        if len(sentence) < 25 or len(sentence) > 240:
            continue
        if any(keyword in lower for keyword in keywords) or re.search(r"\d", sentence):
            prioritized.append(sentence.rstrip("."))
        if len(prioritized) >= max_bullets:
            break
    if prioritized:
        return prioritized
    return [sentence.rstrip(".") for sentence in sentences[:max_bullets] if len(sentence) > 20]


def tokenize(text: str) -> list[str]:
    return sorted(
        {
            token
            for token in _TOKEN_RE.findall((text or "").lower())
            if len(token) > 1 and token not in _STOPWORDS
        }
    )


def normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def topic_key(text: str, *, max_parts: int = 8, max_len: int = 80) -> str:
    tokens = tokenize(text)[:max_parts]
    if not tokens:
        return ""
    return "-".join(tokens)[:max_len].strip("-")
