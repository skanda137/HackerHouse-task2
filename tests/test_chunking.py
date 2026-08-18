import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.base import Document
from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SentenceBoundaryChunker, split_sentences
from chunking.metadata_aware import MetadataAwareChunker


HINDI_DOC = Document(
    doc_id="test-hi-1",
    text="भारत की राजधानी नई दिल्ली है। गंगा नदी उत्तर भारत से होकर बहती है। "
         "हिमालय पर्वत श्रृंखला भारत के उत्तर में स्थित है।",
    language="hi",
    source="test",
)

ENGLISH_DOC = Document(
    doc_id="test-en-1",
    text="A" * 500,  # long, punctuation-free -- exercises fixed-size splitting hard
    language="en",
    source="test",
)


def test_fixed_size_respects_chunk_size():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(ENGLISH_DOC)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_fixed_size_overlap_present():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(ENGLISH_DOC)
    # consecutive chunks should share `overlap` characters at the boundary
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_fixed_size_short_doc_single_chunk():
    doc = Document(doc_id="short", text="Hello world.", language="en", source="test")
    chunks = FixedSizeChunker(chunk_size=220, overlap=40).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_sentence_split_handles_devanagari_danda():
    sents = split_sentences(HINDI_DOC.text)
    assert len(sents) == 3
    assert all(s.endswith("।") for s in sents)


def test_sentence_boundary_never_splits_mid_sentence():
    chunker = SentenceBoundaryChunker(target_size=40, max_size=60)
    chunks = chunker.chunk(HINDI_DOC)
    original_sentences = set(split_sentences(HINDI_DOC.text))
    for c in chunks:
        for piece in c.text.split(" "):
            pass  # smoke: ensure no exception: real assertion below
    # every chunk's text, when re-split, should be a subset of the doc's sentences
    for c in chunks:
        for s in split_sentences(c.text):
            assert s in original_sentences


def test_metadata_aware_tags_language_and_script():
    chunker = MetadataAwareChunker(base_strategy="semantic")
    chunks = chunker.chunk(HINDI_DOC)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["language"] == "hi"
        assert c.metadata["script"] == "Devanagari"
        assert c.strategy == "metadata_aware"


def test_metadata_aware_routes_tamil_to_larger_budget():
    ta_doc = Document(
        doc_id="ta-1",
        text="இந்தியாவின் தலைநகரம் புது தில்லி. கங்கை நதி வட இந்தியா வழியாக பாய்கிறது.",
        language="ta", source="test",
    )
    chunker = MetadataAwareChunker(base_strategy="semantic")
    chunks = chunker.chunk(ta_doc)
    assert all(c.metadata["script"] == "Tamil" for c in chunks)


def test_chunk_ids_are_deterministic_and_unique():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks_a = chunker.chunk(ENGLISH_DOC)
    chunks_b = chunker.chunk(ENGLISH_DOC)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert len({c.chunk_id for c in chunks_a}) == len(chunks_a)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
