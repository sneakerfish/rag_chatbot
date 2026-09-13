#!/usr/bin/env python3
"""
Text chunking strategies for the RAG ingest pipeline.

Two strategies are provided:

* ``fixed_chunk_text``    - the original sliding window (N characters with overlap).
* ``semantic_chunk_text`` - semantic breakpoint chunking.

Semantic breakpoint chunking works like this:

1. Split the text into sentences.
2. Embed each sentence together with its neighbours (a small "buffer" window
   smooths out noise from very short sentences).
3. Compute the cosine distance between each pair of adjacent embeddings.
4. Wherever the distance jumps above a percentile threshold, the topic has
   shifted, so a chunk boundary is placed there.
5. Chunks that are too large are split again at their weakest internal
   boundary; chunks that are too small are merged with a neighbour.

The result is a set of chunks whose boundaries fall between concepts rather
than in the middle of an explanation, which is what a fixed-size window does.
"""

import re
from typing import Callable, List, Sequence

import numpy as np

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------------
# Text normalisation and sentence splitting
# ---------------------------------------------------------------------------

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“‘])")
# A numbered section heading on its own line, e.g. "2.4 Logical vectors" or
# "3 Objects, their modes and attributes". Short, and not ending like a sentence.
_HEADING_LINE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,90}$")
# A sentence that begins with a numbered section heading (after normalisation
# the heading and the first sentence of its section share a line).
_HEADING_START = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")
# A line holding nothing but a page number.
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
_PARAGRAPH_TOKEN = "\u2029"  # Unicode paragraph separator, used as a marker


def normalize_pdf_text(text: str) -> str:
    """Tidy up text extracted from a PDF.

    PDF extraction hard-wraps every line and hyphenates words across lines.
    This joins hyphenated words, collapses line-wrapping whitespace, keeps
    blank lines as paragraph breaks, drops bare page-number lines, and treats
    numbered section headings as the start of a new paragraph so a heading
    stays with the section it introduces rather than the paragraph before it.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _PAGE_NUMBER_LINE.match(stripped):
            lines.append("")
        elif _HEADING_LINE.match(stripped):
            lines.extend(["", stripped])
        else:
            lines.append(line)
    text = "\n".join(lines)

    # Preserve paragraph breaks before collapsing whitespace.
    text = _PARAGRAPH_BREAK.sub(_PARAGRAPH_TOKEN, text)
    text = re.sub(r"[ \t\n\f\v]+", " ", text)
    text = re.sub(f"[ ]*{_PARAGRAPH_TOKEN}+[ ]*", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Paragraph breaks are always sentence boundaries. Within a paragraph a
    sentence ends at ., ! or ? followed by whitespace and an upper-case
    letter, digit, or opening bracket/quote. This deliberately does not split
    on things like ``e.g. foo`` or ``x.y`` inside code.
    """
    sentences: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


# ---------------------------------------------------------------------------
# Fixed-size chunking (original behaviour)
# ---------------------------------------------------------------------------

def fixed_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping fixed-size chunks, preferring sentence ends."""
    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to break at a sentence boundary
        # within the last 100 characters of the window.
        if end < len(text):
            for i in range(end - 1, max(start + chunk_size - 100, start), -1):
                if text[i] in ".!?":
                    end = i + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Semantic breakpoint chunking
# ---------------------------------------------------------------------------

def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between each embedding and the next one."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    similarity = np.sum(unit[:-1] * unit[1:], axis=1)
    return 1.0 - similarity


def _hard_split(text: str, max_size: int) -> List[str]:
    """Split a single over-long sentence on whitespace so each piece fits."""
    pieces = []
    words = text.split(" ")
    current = ""
    for word in words:
        if len(word) > max_size:
            # A single token longer than max_size: cut it by characters.
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i:i + max_size] for i in range(0, len(word), max_size))
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _group_length(sentences: Sequence[str], start: int, end: int) -> int:
    """Character length of sentences[start:end] once joined with spaces."""
    return sum(len(s) for s in sentences[start:end]) + max(0, end - start - 1)


def _split_to_fit(sentences: Sequence[str], distances: np.ndarray,
                  start: int, end: int, max_size: int) -> List[List[str]]:
    """Recursively split sentences[start:end] at its weakest boundary until it fits."""
    if _group_length(sentences, start, end) <= max_size:
        return [list(sentences[start:end])]
    if end - start == 1:
        return [[piece] for piece in _hard_split(sentences[start], max_size)]

    # distances[i] is the distance between sentence i and i+1. Candidate cut
    # points are between start and end-1 inclusive.
    candidate = distances[start:end - 1]
    cut = start + int(np.argmax(candidate)) + 1
    return (_split_to_fit(sentences, distances, start, cut, max_size)
            + _split_to_fit(sentences, distances, cut, end, max_size))


def _merge_small(groups: List[List[str]], min_size: int, max_size: int) -> List[List[str]]:
    """Merge groups shorter than min_size into a neighbour when that still fits.

    A short group is folded into the group before it if the result fits within
    max_size; otherwise it is kept and the following group is folded into it
    instead. Fragments wedged between two full-size neighbours stay as they are.
    """
    merged: List[List[str]] = []
    for group in groups:
        if merged:
            previous = merged[-1]
            previous_small = _group_length(previous, 0, len(previous)) < min_size
            current_small = _group_length(group, 0, len(group)) < min_size
            if previous_small or current_small:
                combined = previous + group
                if _group_length(combined, 0, len(combined)) <= max_size:
                    merged[-1] = combined
                    continue
        merged.append(group)
    return merged


def _embed_in_batches(embed: EmbedFn, texts: List[str], batch_size: int) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embed(texts[i:i + batch_size]))
    return np.asarray(vectors, dtype=np.float32)


def semantic_chunk_text(
    text: str,
    embed: EmbedFn,
    breakpoint_percentile: float = 90.0,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    buffer_size: int = 1,
    batch_size: int = 64,
    break_at_headings: bool = True,
) -> List[str]:
    """Split text into chunks at semantic breakpoints.

    Args:
        text: Raw text (PDF extraction output is fine, it is normalised here).
        embed: Callable mapping a list of strings to a list of embedding vectors.
        breakpoint_percentile: Adjacent-sentence distances above this percentile
            become chunk boundaries. Lower values give more, smaller chunks.
        max_chunk_size: Chunks longer than this (in characters) are split again
            at their weakest internal boundary.
        min_chunk_size: Chunks shorter than this are merged into a neighbour.
        buffer_size: Number of neighbouring sentences on each side to include
            when embedding a sentence. Smooths out very short sentences.
        batch_size: How many sentences to embed per call.
        break_at_headings: Always start a new chunk at a numbered section
            heading such as "2.4 Logical vectors", whatever the embeddings say.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if min_chunk_size > max_chunk_size:
        raise ValueError("min_chunk_size must not exceed max_chunk_size")
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("breakpoint_percentile must be between 0 and 100")

    sentences = split_sentences(normalize_pdf_text(text))
    if not sentences:
        return []

    if len(sentences) == 1:
        groups = [[piece] for piece in _hard_split(sentences[0], max_chunk_size)]
        return [" ".join(g) for g in groups]

    # Embed each sentence with a little surrounding context.
    contexts = [
        " ".join(sentences[max(0, i - buffer_size): i + buffer_size + 1])
        for i in range(len(sentences))
    ]
    embeddings = _embed_in_batches(embed, contexts, batch_size)
    distances = _cosine_distances(embeddings)

    threshold = float(np.percentile(distances, breakpoint_percentile))
    breakpoints = set(int(i) for i in np.nonzero(distances > threshold)[0])
    if break_at_headings:
        # A break after sentence i-1 puts the heading at the start of a chunk.
        breakpoints.update(
            i - 1 for i in range(1, len(sentences)) if _HEADING_START.match(sentences[i])
        )

    # First pass: cut at semantic breakpoints.
    groups: List[List[str]] = []
    start = 0
    for i in range(len(sentences)):
        if i in breakpoints:
            groups.extend(_split_to_fit(sentences, distances, start, i + 1, max_chunk_size))
            start = i + 1
    groups.extend(_split_to_fit(sentences, distances, start, len(sentences), max_chunk_size))

    # Second pass: fold fragments into a neighbour.
    groups = _merge_small(groups, min_chunk_size, max_chunk_size)

    return [" ".join(group) for group in groups]


def default_embedding_function() -> EmbedFn:
    """The embedding function ChromaDB uses when none is specified.

    Using the same model for chunking and retrieval means breakpoints are
    chosen by the same notion of "similar" that queries will use later.
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()
