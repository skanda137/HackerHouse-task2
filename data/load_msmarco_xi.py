"""
Loader for ai4bharat/MSMARCO-XI.

Verified against the actual dataset on the HF Hub (2026-08-22), which does
NOT match what this file originally assumed:

- There is a single HF config ("default"), not one config per language.
- Language is instead split across per-language parquet FILES, one per
  split, named with 3-letter codes: train/{hintrain,tamtrain,bentrain,...}.parquet,
  validation/{hinval,tamval,telval,benval,...}.parquet. Telugu has NO train
  file at all -- validation only.
- Each row is a QA example, not a bare passage: `passages.Translated_passages`
  is a list of ~10 candidate passages per query, with a parallel
  `passages.is_selected` 0/1 list marking the gold-relevant one(s). A
  realistic retrieval corpus is built from all of them (selected +
  distractors), not just the gold passage.
- `train/*.parquet` files are ~3.7GB each (~80GB across all languages);
  `validation/*.parquet` files are ~460-490MB each. `load_real()` defaults
  to `validation` so a from-scratch run is tractable, and because it's the
  only split where all four supported languages exist.

`load_synthetic()` is an offline stand-in with the *same shape* (doc_id,
text, language, source) used for local dev, tests, and this repo's own
benchmark, on machines/CI runners without HuggingFace Hub access. Swapping
from synthetic to real data is a one-line change in pipeline.py -- nothing
downstream (chunking/embedding/index/retrieval) knows or cares which one
fed it.
"""
from __future__ import annotations

import random
from typing import List

from chunking.base import Document

SUPPORTED_LANGUAGES = ["hi", "ta", "te", "bn"]

# Our 2-letter codes -> the dataset's 3-letter file-name prefixes.
_LANG_FILE_PREFIX = {"hi": "hin", "ta": "tam", "te": "tel", "bn": "ben"}


def load_real(languages: List[str] = None, split: str = "validation",
              max_docs_per_language: int = 2000) -> List[Document]:
    """
    Real loader. Requires: pip install datasets, and network access to
    huggingface.co (files are cached locally after first download).

    Fetches each language's parquet file directly by URL via the generic
    "parquet" builder rather than `load_dataset("ai4bharat/MSMARCO-XI", ...)`,
    which would otherwise pull every language's file to resolve the dataset
    (there is no per-language HF config to select with).
    """
    from datasets import load_dataset  # local import: only required for this path

    languages = languages or SUPPORTED_LANGUAGES
    file_suffix = "train" if split == "train" else "val"
    docs: List[Document] = []

    for lang in languages:
        prefix = _LANG_FILE_PREFIX.get(lang)
        if prefix is None:
            raise ValueError(f"no known ai4bharat/MSMARCO-XI file for language '{lang}'")

        url = (f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/"
               f"{split}/{prefix}{file_suffix}.parquet")
        ds = load_dataset("parquet", data_files=url, split="train")

        count = 0
        for row in ds:
            passages = row.get("passages") or {}
            translated = passages.get("Translated_passages") or []
            selected = passages.get("is_selected") or []
            qid = row.get("query_id")

            for i, text in enumerate(translated):
                if not text or not text.strip():
                    continue
                docs.append(Document(
                    doc_id=f"{lang}-{split}-{qid}-{i}",
                    text=text,
                    language=lang,
                    source=f"msmarco-xi/{lang}/{split}",
                    extra={
                        "query_id": qid,
                        "is_selected": bool(selected[i]) if i < len(selected) else False,
                        "query": row.get("query"),
                    },
                ))
                count += 1
                if count >= max_docs_per_language:
                    break
            if count >= max_docs_per_language:
                break

    return docs


# ---------------------------------------------------------------------------
# Offline synthetic fallback -- same shape as the real dataset, no network.
# ---------------------------------------------------------------------------

_TOPIC_SENTENCES = {
    "hi": [
        "भारत की राजधानी नई दिल्ली है और यह देश का प्रमुख राजनीतिक केंद्र है।",
        "गंगा नदी उत्तर भारत से होकर बहती है और लाखों लोगों के लिए जल का स्रोत है।",
        "महात्मा गांधी ने भारत की स्वतंत्रता के लिए अहिंसात्मक आंदोलन चलाया।",
        "भारतीय अर्थव्यवस्था कृषि, उद्योग और सेवा क्षेत्र पर निर्भर करती है।",
        "हिमालय पर्वत श्रृंखला भारत के उत्तर में स्थित है और इसे विश्व की सबसे ऊँची पर्वत श्रृंखला माना जाता है।",
        "भारत में सैकड़ों भाषाएँ बोली जाती हैं, जिनमें हिंदी सबसे व्यापक रूप से बोली जाने वाली भाषा है।",
        "क्रिकेट भारत में सबसे लोकप्रिय खेल है और करोड़ों लोग इसे देखते हैं।",
        "ताजमहल आगरा शहर में स्थित एक प्रसिद्ध ऐतिहासिक स्मारक है।",
    ],
    "bn": [
        "ভারতের রাজধানী নয়াদিল্লি এবং এটি দেশের প্রধান রাজনৈতিক কেন্দ্র।",
        "গঙ্গা নদী উত্তর ভারত দিয়ে প্রবাহিত হয় এবং লক্ষ লক্ষ মানুষের জলের উৎস।",
        "মহাত্মা গান্ধী ভারতের স্বাধীনতার জন্য অহিংস আন্দোলন পরিচালনা করেছিলেন।",
        "ভারতীয় অর্থনীতি কৃষি, শিল্প এবং পরিষেবা খাতের উপর নির্ভরশীল।",
        "হিমালয় পর্বতমালা ভারতের উত্তরে অবস্থিত এবং বিশ্বের সর্বোচ্চ পর্বতমালা হিসেবে বিবেচিত।",
        "ভারতে শত শত ভাষা বলা হয়, যার মধ্যে হিন্দি সবচেয়ে বেশি প্রচলিত।",
        "ক্রিকেট ভারতের সবচেয়ে জনপ্রিয় খেলা এবং কোটি কোটি মানুষ এটি দেখে।",
        "তাজমহল আগ্রা শহরে অবস্থিত একটি বিখ্যাত ঐতিহাসিক স্মৃতিস্তম্ভ।",
    ],
    "ta": [
        "இந்தியாவின் தலைநகரம் புது தில்லி மற்றும் இது நாட்டின் முக்கிய அரசியல் மையமாகும்.",
        "கங்கை நதி வட இந்தியா வழியாக பாய்கிறது மற்றும் மில்லியன் கணக்கான மக்களுக்கு நீர் ஆதாரமாக உள்ளது.",
        "மகாத்மா காந்தி இந்தியாவின் சுதந்திரத்திற்காக அகிம்சை இயக்கத்தை நடத்தினார்.",
        "இந்திய பொருளாதாரம் விவசாயம், தொழில் மற்றும் சேவைத் துறையை சார்ந்துள்ளது.",
        "இமயமலைத் தொடர் இந்தியாவின் வடக்கில் அமைந்துள்ளது மற்றும் உலகின் மிக உயரமான மலைத்தொடராக கருதப்படுகிறது.",
        "இந்தியாவில் நூற்றுக்கணக்கான மொழிகள் பேசப்படுகின்றன, அவற்றில் இந்தி மிகவும் பரவலாக பேசப்படுகிறது.",
        "கிரிக்கெட் இந்தியாவின் மிகவும் பிரபலமான விளையாட்டு மற்றும் கோடிக்கணக்கான மக்கள் இதைப் பார்க்கிறார்கள்.",
        "தாஜ்மஹால் ஆக்ரா நகரில் அமைந்துள்ள ஒரு புகழ்பெற்ற வரலாற்று நினைவுச்சின்னமாகும்.",
    ],
    "te": [
        "భారతదేశ రాజధాని న్యూఢిల్లీ మరియు ఇది దేశానికి ప్రధాన రాజకీయ కేంద్రం.",
        "గంగా నది ఉత్తర భారతదేశం గుండా ప్రవహిస్తుంది మరియు లక్షలాది మందికి నీటి వనరుగా ఉంది.",
        "మహాత్మా గాంధీ భారతదేశ స్వాతంత్ర్యం కోసం అహింసా ఉద్యమాన్ని నడిపించారు.",
        "భారత ఆర్థిక వ్యవస్థ వ్యవసాయం, పరిశ్రమలు మరియు సేవా రంగంపై ఆధారపడి ఉంటుంది.",
        "హిమాలయ పర్వత శ్రేణి భారతదేశ ఉత్తర భాగంలో ఉంది మరియు ఇది ప్రపంచంలోనే అత్యంత ఎత్తైన పర్వత శ్రేణిగా పరిగణించబడుతుంది.",
        "భారతదేశంలో వందలాది భాషలు మాట్లాడతారు, వాటిలో హిందీ అత్యంత విస్తృతంగా మాట్లాడే భాష.",
        "క్రికెట్ భారతదేశంలో అత్యంత ప్రజాదరణ పొందిన ఆట మరియు కోట్లాది మంది దీనిని చూస్తారు.",
        "తాజ్ మహల్ ఆగ్రా నగరంలో ఉన్న ఒక ప్రసిద్ధ చారిత్రక కట్టడం.",
    ],
}


def load_synthetic(languages: List[str] = None, docs_per_language: int = 150,
                    seed: int = 42) -> List[Document]:
    """
    Deterministic, offline, dependency-free dataset with the same
    (doc_id, text, language, source) shape as `load_real()`. Documents are
    built by combining 2-5 topic sentences per language so we get realistic
    variation in length -- long enough to meaningfully exercise all three
    chunking strategies, short enough to build/index/benchmark in seconds.
    """
    languages = languages or SUPPORTED_LANGUAGES
    rng = random.Random(seed)
    docs: List[Document] = []

    for lang in languages:
        pool = _TOPIC_SENTENCES[lang]
        for i in range(docs_per_language):
            n_sents = rng.randint(2, 6)
            sents = [rng.choice(pool) for _ in range(n_sents)]
            text = " ".join(sents)
            docs.append(Document(
                doc_id=f"{lang}-synthetic-{i}",
                text=text,
                language=lang,
                source=f"msmarco-xi-synthetic/{lang}",
                extra={"synthetic": True},
            ))
    return docs
