"""Passages: a long message split into parts of at most ``PASSAGE_WORDS``
words, which evidence and turn search search, judge and show in place of
the whole message (``MemoryStore.evidence``, ``MemoryStore.turn_search``).

A message is shown whole or not at all, and the shares of a context are a
few hundred tokens, so a long reply could never be shown. Search scores
long documents by their passages for the same reason (Dai and Callan 2019,
a window of 150 words). A message of ``PASSAGE_WORDS`` words or fewer is
one passage, its text unchanged, so chats of short messages are read as
before.
"""

from __future__ import annotations

import re

from ..models import Episode, Passage

#: The most words of one passage: Dai and Callan's window, about 200 tokens,
#: below the smallest share a context gives turns (a quarter of the
#: default 1,200 tokens), so any passage fits any share on its own.
PASSAGE_WORDS = 150

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pieces(paragraph: str, words: int) -> list[tuple[str, str]]:
    """A paragraph as pieces of at most ``words`` words, each with the
    separator that joins it to the piece before: the paragraph whole; else
    its sentences, or its lines when it has one sentence (code, a list);
    a piece still longer cut every ``words`` words."""
    if len(paragraph.split()) <= words:
        return [(paragraph, "")]
    sentences = _SENTENCE.split(paragraph)
    parts, joiner = (sentences, " ") if len(sentences) > 1 else (paragraph.split("\n"), "\n")
    pieces: list[tuple[str, str]] = []
    for part in parts:
        tokens = part.split()
        if not tokens:
            continue
        if len(tokens) <= words:
            pieces.append((part.rstrip(), joiner))
        else:
            pieces.extend((" ".join(tokens[i:i + words]), joiner)
                          for i in range(0, len(tokens), words))
    return pieces


def split_passages(text: str, words: int = PASSAGE_WORDS) -> list[str]:
    """``text`` as passages of at most ``words`` words, in order: whole
    paragraphs packed while they fit, a longer paragraph by its pieces
    (``_pieces``). A text of ``words`` words or fewer is returned whole."""
    if len(text.split()) <= words:
        return [text]
    units: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH.split(text.strip()):
        if not paragraph.strip():
            continue
        for k, (piece, joiner) in enumerate(_pieces(paragraph.strip("\n"), words)):
            units.append((piece, "\n\n" if k == 0 else joiner))
    passages: list[str] = []
    current, count = "", 0
    for piece, joiner in units:
        n = len(piece.split())
        if current and count + n > words:
            passages.append(current)
            current, count = "", 0
        current = piece if not current else current + joiner + piece
        count += n
    if current:
        passages.append(current)
    return passages


def passages_of(episode: Episode, words: int = PASSAGE_WORDS) -> list[Passage]:
    """The passages of one episode, in order (``split_passages``)."""
    texts = split_passages(episode.content, words)
    return [Passage(id=passage_id(episode.id, i), episode_id=episode.id, position=i,
                    count=len(texts), content=text) for i, text in enumerate(texts)]


def passage_id(episode_id: str, position: int) -> str:
    """A passage's id: its episode's id and its place, so the passages of an
    episode written twice are the same rows."""
    return f"{episode_id}:{position}"
