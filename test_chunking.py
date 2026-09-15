#!/usr/bin/env python3
"""
Unit tests for chunking.py. Run with: python -m pytest test_chunking.py

These use a fake embedding function so they run offline without downloading
an embedding model.
"""

import numpy as np
import pytest

from chunking import (
    fixed_chunk_text,
    normalize_pdf_text,
    semantic_chunk_text,
    split_sentences,
)


def topic_embedder(topics):
    """Build an embedder that maps each sentence to a one-hot vector for its topic.

    ``topics`` maps a keyword to a topic index. A sentence mentioning a keyword
    is embedded as that topic's basis vector, so sentences on the same topic have
    distance 0 and sentences on different topics have distance 1.
    """
    dim = max(topics.values()) + 2

    def embed(texts):
        vectors = []
        for text in texts:
            vector = np.zeros(dim)
            for keyword, index in topics.items():
                if keyword in text:
                    vector[index] += 1.0
            if not vector.any():
                vector[-1] = 1.0
            vectors.append(vector.tolist())
        return vectors

    return embed


CATS = "Cats are small carnivorous mammals. Cats have retractable claws. Cats purr when content."
CARS = "Cars are wheeled motor vehicles. Cars run on petrol or electricity. Cars need regular servicing."
CODE = "Code is written in R. Code can be vectorised. Code should be tested."


class TestNormalizePdfText:
    def test_joins_hyphenated_line_breaks(self):
        assert normalize_pdf_text("an ex-\npression") == "an expression"

    def test_collapses_line_wrapping_but_keeps_paragraphs(self):
        text = "first line\nsecond line\n\nnew paragraph"
        assert normalize_pdf_text(text) == "first line second line\n\nnew paragraph"

    def test_numbered_heading_starts_new_paragraph(self):
        text = "end of previous section.\n2.4 Logical vectors\nAs well as numeric vectors."
        assert normalize_pdf_text(text) == (
            "end of previous section.\n\n2.4 Logical vectors As well as numeric vectors."
        )

    def test_page_number_lines_are_dropped(self):
        assert normalize_pdf_text("some text\n17\nmore text") == "some text\n\nmore text"


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]

    def test_does_not_split_inside_code_or_abbreviations(self):
        assert split_sentences("Use x.y or e.g. foo here.") == ["Use x.y or e.g. foo here."]

    def test_paragraph_break_is_a_boundary(self):
        assert split_sentences("no punctuation here\n\nnext paragraph") == [
            "no punctuation here",
            "next paragraph",
        ]

    def test_empty(self):
        assert split_sentences("   \n\n ") == []


class TestFixedChunkText:
    def test_empty(self):
        assert fixed_chunk_text("") == []

    def test_short_text_is_one_chunk(self):
        assert fixed_chunk_text("hello world") == ["hello world"]

    def test_chunks_overlap_and_cover_text(self):
        text = " ".join(f"Sentence number {i}." for i in range(200))
        chunks = fixed_chunk_text(text, chunk_size=300, overlap=50)
        assert all(len(c) <= 300 for c in chunks)
        assert chunks[0].startswith("Sentence number 0.")
        assert chunks[-1].endswith("Sentence number 199.")
        # Consecutive chunks share text.
        assert chunks[1][:20] in chunks[0]

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            fixed_chunk_text("text", chunk_size=10, overlap=10)


class TestSemanticChunkText:
    embed = staticmethod(topic_embedder({"Cats": 0, "Cars": 1, "Code": 2}))

    def test_empty(self):
        assert semantic_chunk_text("", self.embed) == []

    def test_breaks_where_topic_changes(self):
        text = f"{CATS} {CARS} {CODE}"
        chunks = semantic_chunk_text(
            text, self.embed, breakpoint_percentile=50, min_chunk_size=0, buffer_size=0
        )
        assert chunks == [CATS, CARS, CODE]

    def test_fixed_window_splits_topic_but_semantic_does_not(self):
        # The point of the change: a fixed window cuts through the middle of
        # a concept, a semantic breakpoint does not.
        text = f"{CATS} {CARS}"
        fixed = fixed_chunk_text(text, chunk_size=len(CATS) + 40, overlap=0)
        assert any("Cats" in c and "Cars" in c for c in fixed)

        semantic = semantic_chunk_text(
            text, self.embed, breakpoint_percentile=50, min_chunk_size=0, buffer_size=0
        )
        assert not any("Cats" in c and "Cars" in c for c in semantic)

    def test_respects_max_chunk_size(self):
        text = " ".join(f"Cats sentence {i} is about cats." for i in range(100))
        chunks = semantic_chunk_text(text, self.embed, max_chunk_size=200, min_chunk_size=0)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)
        assert " ".join(chunks) == text

    def test_single_sentence_longer_than_max_is_hard_split(self):
        text = "Cats " * 200
        chunks = semantic_chunk_text(text.strip(), self.embed, max_chunk_size=100, min_chunk_size=0)
        assert all(len(c) <= 100 for c in chunks)
        assert " ".join(chunks) == text.strip()

    def test_small_fragments_are_merged(self):
        text = f"{CATS} Cars. {CODE}"
        chunks = semantic_chunk_text(
            text, self.embed, breakpoint_percentile=50, min_chunk_size=20, buffer_size=0
        )
        assert all(len(c) >= 20 for c in chunks)
        assert " ".join(chunks) == text

    def test_no_text_is_lost(self):
        text = f"{CATS} {CARS} {CODE} " * 5
        chunks = semantic_chunk_text(text.strip(), self.embed)
        assert " ".join(chunks) == text.strip()

    def test_single_sentence(self):
        assert semantic_chunk_text("Just one sentence.", self.embed) == ["Just one sentence."]

    def test_numbered_heading_starts_a_chunk(self):
        # All one topic, so the embeddings alone would never break here.
        text = "Cats are mammals.\n2.4 More cats\nCats have claws. Cats purr."
        chunks = semantic_chunk_text(
            text, self.embed, breakpoint_percentile=100, min_chunk_size=0, buffer_size=0
        )
        assert chunks == ["Cats are mammals.", "2.4 More cats Cats have claws. Cats purr."]

        chunks = semantic_chunk_text(
            text, self.embed, breakpoint_percentile=100, min_chunk_size=0, buffer_size=0,
            break_at_headings=False,
        )
        assert chunks == ["Cats are mammals. 2.4 More cats Cats have claws. Cats purr."]

    def test_embedder_is_called_in_batches(self):
        calls = []

        def counting_embed(texts):
            calls.append(len(texts))
            return self.embed(texts)

        text = " ".join(f"Cats sentence {i}." for i in range(10))
        semantic_chunk_text(text, counting_embed, batch_size=4)
        assert calls == [4, 4, 2]

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            semantic_chunk_text("a. B.", self.embed, breakpoint_percentile=120)
        with pytest.raises(ValueError):
            semantic_chunk_text("a. B.", self.embed, max_chunk_size=0)
        with pytest.raises(ValueError):
            semantic_chunk_text("a. B.", self.embed, min_chunk_size=50, max_chunk_size=10)
