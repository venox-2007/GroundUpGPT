"""Comprehensive deterministic evaluation script for Step 44 user knowledge retrieval.

Executes 50 deterministic test cases validating:
1. Exact substring chunk offset invariant: source.content[start:end] == chunk.text
2. Storage, session limits, and security sanitization
3. BM25 retrieval discrimination across distinct entities
4. Classification accuracy: relevant_passage, ambiguous_evidence, insufficient_evidence, no_match
5. Zero-model-call evidence routing and attribution integrity
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.user_knowledge import (
    KnowledgeStore,
    chunk_document_bpe,
    sanitize_display_name,
    MAX_SOURCE_BYTES,
    MAX_SOURCES_PER_SESSION,
    MAX_SESSION_BYTES,
)
from src.user_knowledge_retrieval import UserKnowledgeRetriever
from backend.assistant_router import AssistantRouter, ORIGIN_LOCAL_FACT, MODE_ASSISTED

logging.basicConfig(level=logging.WARNING)


def run_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("GroundUpGPT Step 44: User Knowledge Retrieval Comprehensive Evaluation")
    print("=" * 70)

    store = KnowledgeStore()
    results: List[Dict[str, Any]] = []
    start_total_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Group 1: Exact Substring Preservation Invariant (10 Tests)
    # -------------------------------------------------------------------------
    docs_to_test = [
        ("Short single sentence", "GroundUpGPT is trained completely from scratch without external weights."),
        ("Multi-paragraph", "Paragraph one introduces Transformer layers.\n\nParagraph two covers attention heads and embeddings.\n\nParagraph three describes AdamW training dynamics."),
        ("Markdown headers and lists", "# Project Overview\n\n- Phase 1: Pretraining\n- Phase 2: Anchor SFT\n- Phase 3: Knowledge retrieval."),
        ("Code snippet", "def calculate_loss(logits, targets):\n    return F.cross_entropy(logits.view(-1, 4096), targets.view(-1))\n\n# Verified PyTorch implementation"),
        ("Quotes and symbols", 'Dr. Carl Sagan once stated: "Somewhere, something incredible is waiting to be known." It cost $100 & 50% effort!'),
        ("Repeated phrases", "Repeat word word word word word word word word word word word word word word word word word word word word word word."),
        ("Unicode characters", "GroundUpGPT supports UTF-8 characters: café, na\u00efve, \u03c0 = 3.14159, \u2248 approximately, \u00a9 2026."),
        ("Numbers and dates", "Historical event on July 20, 1969 at 20:17:40 UTC. Apollo 11 Lunar Module Eagle touched down on the Moon."),
        ("Whitespace padding", "   \n\n   This text has leading and trailing indentation and extra space.   \n\n  "),
        ("Structured fact table", "Entity: Jupiter | Moons: 95 | Type: Gas Giant\nEntity: Saturn | Moons: 146 | Type: Gas Giant with Rings"),
    ]

    group1_passed = 0
    for idx, (label, content) in enumerate(docs_to_test):
        test_id = f"g1_substring_{idx+1}"
        chunks = chunk_document_bpe(content, source_id=f"src_{idx}", tokenizer=store.tokenizer)
        passed = True
        err = None
        if not chunks and content.strip():
            passed = False
            err = "No chunks produced for non-empty content"
        for c in chunks:
            expected = content[c.start_char : c.end_char]
            if expected != c.text:
                passed = False
                err = f"Mismatch at [{c.start_char}:{c.end_char}]: expected '{expected}', got '{c.text}'"
                break
        if passed:
            group1_passed += 1
        results.append({
            "test_id": test_id,
            "group": "substring_invariant",
            "name": label,
            "passed": passed,
            "details": {"chunks_count": len(chunks), "error": err},
        })

    # -------------------------------------------------------------------------
    # Group 2: Storage Limits, Security & Sanitization (10 Tests)
    # -------------------------------------------------------------------------
    sid = "eval_group2_session"
    limits_tests = [
        ("path_traversal_1", "../../secret.txt", "secret.txt"),
        ("path_traversal_2", "C:\\Windows\\System32\\calc.exe", "calc.exe"),
        ("control_chars", "doc\x00\x01\x1f.md", "doc.md"),
        ("empty_name", "   ", "Untitled Document"),
        ("max_name_length", "a" * 150, "a" * 100),
    ]
    group2_passed = 0
    for test_name, raw_name, expected_name in limits_tests:
        cleaned = sanitize_display_name(raw_name)
        ok = cleaned == expected_name
        if ok:
            group2_passed += 1
        results.append({
            "test_id": f"g2_sanitize_{test_name}",
            "group": "sanitization",
            "name": f"Sanitize {test_name}",
            "passed": ok,
            "details": {"raw": raw_name, "got": cleaned, "expected": expected_name},
        })

    # Limits validations
    store.clear_session(sid)
    # Binary null byte rejection
    try:
        store.add_source(sid, "binary.txt", "hello\x00world")
        b_pass = False
    except ValueError:
        b_pass = True
    if b_pass: group2_passed += 1
    results.append({"test_id": "g2_binary_rejection", "group": "limits", "name": "Binary null byte rejection", "passed": b_pass})

    # Empty content rejection
    try:
        store.add_source(sid, "empty.txt", "   \t\n  ")
        e_pass = False
    except ValueError:
        e_pass = True
    if e_pass: group2_passed += 1
    results.append({"test_id": "g2_empty_rejection", "group": "limits", "name": "Empty content rejection", "passed": e_pass})

    # Document size boundary (16 KB)
    try:
        store.add_source(sid, "oversized.txt", "a" * (MAX_SOURCE_BYTES + 50))
        o_pass = False
    except ValueError:
        o_pass = True
    if o_pass: group2_passed += 1
    results.append({"test_id": "g2_doc_size_limit", "group": "limits", "name": "16 KB document limit enforcement", "passed": o_pass})

    # Max sources per session (10 sources)
    for i in range(MAX_SOURCES_PER_SESSION):
        store.add_source(sid, f"doc_{i}.txt", f"Content for source number {i} with sufficient text tokens.")
    try:
        store.add_source(sid, "doc_11.txt", "Eleventh source.")
        max_s_pass = False
    except ValueError:
        max_s_pass = True
    if max_s_pass: group2_passed += 1
    results.append({"test_id": "g2_max_sources_limit", "group": "limits", "name": "10 sources per session limit", "passed": max_s_pass})

    # Session cleanup
    cleared = store.clear_session(sid)
    c_pass = cleared and len(store.list_sources(sid)) == 0
    if c_pass: group2_passed += 1
    results.append({"test_id": "g2_session_clear", "group": "limits", "name": "Session clearing verification", "passed": c_pass})

    # -------------------------------------------------------------------------
    # Group 3: BM25 Retrieval Discrimination (15 Tests)
    # -------------------------------------------------------------------------
    sid_ret = "eval_bm25_session"
    store.clear_session(sid_ret)
    knowledge_corpus = [
        ("apollo_11.txt", "The Apollo 11 mission landed Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969. Michael Collins piloted the command module Columbia."),
        ("artemis_3.txt", "Project Artemis aims to land the first woman and next man on the lunar south pole by late 2026 using the Orion spacecraft and Starship HLS."),
        ("voyager_1.txt", "Voyager 1 was launched on September 5, 1977. In 2012, it became the first human spacecraft to enter interstellar space."),
        ("hubble.txt", "The Hubble Space Telescope was deployed by space shuttle Discovery in 1990 into low Earth orbit and observed distant galaxies."),
        ("james_webb.txt", "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point, observing infrared astronomy and ancient stars."),
    ]
    for name, text in knowledge_corpus:
        store.add_source(sid_ret, name, text)

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid_ret))

    bm25_queries = [
        ("When did Apollo 11 land on the Moon?", "apollo_11.txt", "1969"),
        ("Who piloted the command module Columbia?", "apollo_11.txt", "Michael Collins"),
        ("What is the primary target of Project Artemis?", "artemis_3.txt", "lunar south pole"),
        ("Which spacecraft does Artemis use?", "artemis_3.txt", "Orion"),
        ("In what year was Voyager 1 launched?", "voyager_1.txt", "1977"),
        ("When did Voyager 1 enter interstellar space?", "voyager_1.txt", "2012"),
        ("Which space shuttle deployed Hubble?", "hubble.txt", "Discovery"),
        ("Where was Hubble deployed?", "hubble.txt", "low Earth orbit"),
        ("Where does the James Webb Space Telescope operate?", "james_webb.txt", "L2"),
        ("What kind of astronomy does James Webb observe?", "james_webb.txt", "infrared"),
        ("Neil Armstrong Moon mission", "apollo_11.txt", "1969"),
        ("interstellar probe launched in 1977", "voyager_1.txt", "Voyager 1"),
        ("telescope observing distant galaxies from orbit", "hubble.txt", "Hubble"),
        ("spacecraft and Starship HLS mission", "artemis_3.txt", "Artemis"),
        ("infrared telescope at Sun-Earth Lagrange point", "james_webb.txt", "James Webb"),
    ]

    group3_passed = 0
    retrieval_latencies: List[float] = []
    for idx, (q, expected_src, expected_kw) in enumerate(bm25_queries):
        t0 = time.perf_counter()
        ret = retriever.retrieve(q)
        dt = (time.perf_counter() - t0) * 1000.0
        retrieval_latencies.append(dt)

        ok = (
            ret.status == "relevant_passage"
            and ret.source_display_name == expected_src
            and ret.chunk is not None
            and expected_kw.lower() in ret.chunk.text.lower()
        )
        if ok:
            group3_passed += 1
        results.append({
            "test_id": f"g3_bm25_{idx+1}",
            "group": "bm25_discrimination",
            "query": q,
            "expected_source": expected_src,
            "actual_source": ret.source_display_name,
            "score": ret.score,
            "latency_ms": round(dt, 3),
            "passed": ok,
        })

    # -------------------------------------------------------------------------
    # Group 4: Retrieval Status Classification (10 Tests)
    # -------------------------------------------------------------------------
    sid_status = "eval_status_session"
    store.clear_session(sid_status)
    store.add_source(sid_status, "physics_notes.txt", "The speed of light in vacuum is approximately 299,792 kilometers per second.")
    store.add_source(sid_status, "project_alpha.txt", "Project Aurora was designed by Team Alpha in 2023 for quantum simulation.")
    store.add_source(sid_status, "project_beta.txt", "Project Aurora was designed by Team Beta in 2023 for quantum computing.")

    retriever_status = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid_status))

    status_cases = [
        # relevant_passage
        ("What is the speed of light in vacuum?", "relevant_passage", False),
        ("vacuum speed in kilometers per second", "relevant_passage", False),
        # ambiguous_evidence (competing teams for Project Aurora)
        ("Who designed Project Aurora in 2023?", "ambiguous_evidence", True),
        ("Project Aurora quantum 2023", "ambiguous_evidence", True),
        # no_match
        ("What is the capital of France?", "no_match", False),
        ("recipe for chocolate cake", "no_match", False),
        ("who won the world cup in 1994", "no_match", False),
        ("", "no_match", False),
        # insufficient_evidence
        ("speed something unknown query with low overlap", "insufficient_evidence", False),
        ("designed something completely unrelated", "insufficient_evidence", False),
    ]

    group4_passed = 0
    for idx, (q, expected_status, expected_ambiguous) in enumerate(status_cases):
        ret = retriever_status.retrieve(q)
        ok = (ret.status == expected_status) and (ret.is_ambiguous == expected_ambiguous)
        if ok:
            group4_passed += 1
        results.append({
            "test_id": f"g4_status_{idx+1}",
            "group": "status_classification",
            "query": q,
            "expected_status": expected_status,
            "actual_status": ret.status,
            "expected_ambiguous": expected_ambiguous,
            "actual_ambiguous": ret.is_ambiguous,
            "score": ret.score,
            "passed": ok,
        })

    # -------------------------------------------------------------------------
    # Group 5: Routing Bug Fix & Evidence Mode Validation (5 Tests)
    # -------------------------------------------------------------------------
    router = AssistantRouter()
    group5_passed = 0

    # 5.1 Multi-country capital query regression 1
    dec_multi1 = router.classify_and_route("What are the capitals of France and Norway?", mode=MODE_ASSISTED)
    ok_m1 = dec_multi1.origin == ORIGIN_LOCAL_FACT and dec_multi1.error_code == "ambiguous_query" and "Paris" not in (dec_multi1.answer_text or "")
    if ok_m1: group5_passed += 1
    results.append({"test_id": "g5_capital_multi_1", "group": "router_fix", "name": "Capitals of France and Norway rejection", "passed": ok_m1})

    # 5.2 Multi-country capital query regression 2
    dec_multi2 = router.classify_and_route("What are the capitals of Japan and Atlantis?", mode=MODE_ASSISTED)
    ok_m2 = dec_multi2.origin == ORIGIN_LOCAL_FACT and dec_multi2.error_code == "ambiguous_query" and "Tokyo" not in (dec_multi2.answer_text or "")
    if ok_m2: group5_passed += 1
    results.append({"test_id": "g5_capital_multi_2", "group": "router_fix", "name": "Capitals of Japan and Atlantis rejection", "passed": ok_m2})

    # 5.3 Valid single-country capital query preserved
    dec_single = router.classify_and_route("What is the capital of Japan?", mode=MODE_ASSISTED)
    ok_s = dec_single.origin == ORIGIN_LOCAL_FACT and dec_single.answer_text == "Tokyo"
    if ok_s: group5_passed += 1
    results.append({"test_id": "g5_capital_single", "group": "router_fix", "name": "Single capital query (Tokyo) preserved", "passed": ok_s})

    # 5.4 Arithmetic preserved
    dec_math = router.classify_and_route("What is 25 * 4?", mode=MODE_ASSISTED)
    ok_math = dec_math.origin == "safe_calculator" and dec_math.answer_text == "100"
    if ok_math: group5_passed += 1
    results.append({"test_id": "g5_arithmetic_preserved", "group": "router_fix", "name": "SafeCalculator 25 * 4 preserved", "passed": ok_math})

    # 5.5 Fast retrieval latency guarantee
    avg_latency = sum(retrieval_latencies) / max(len(retrieval_latencies), 1)
    ok_lat = avg_latency < 5.0  # < 5ms retrieval requirement
    if ok_lat: group5_passed += 1
    results.append({"test_id": "g5_latency_benchmark", "group": "performance", "name": "Sub-5ms BM25 retrieval latency", "passed": ok_lat, "details": {"avg_latency_ms": round(avg_latency, 3)}})

    # -------------------------------------------------------------------------
    # Summary & Output Generation
    # -------------------------------------------------------------------------
    total_tests = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_time = time.perf_counter() - start_total_time

    print(f"\nGroup 1 (Substring Invariant): {group1_passed}/10 passed")
    print(f"Group 2 (Limits & Security)  : {group2_passed}/10 passed")
    print(f"Group 3 (BM25 Discrimination) : {group3_passed}/15 passed")
    print(f"Group 4 (Status Classification): {group4_passed}/10 passed")
    print(f"Group 5 (Router Fix & Latency): {group5_passed}/5 passed")
    print("-" * 70)
    print(f"Total Step 44 Test Results: {total_passed}/{total_tests} PASSED ({total_passed/total_tests*100:.1f}%) in {total_time:.3f}s")
    print(f"Average BM25 Retrieval Latency: {avg_latency:.3f} ms")
    print("=" * 70)

    report = {
        "step": 44,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "pass_rate": round(total_passed / total_tests, 4),
        "total_elapsed_sec": round(total_time, 3),
        "avg_retrieval_latency_ms": round(avg_latency, 3),
        "group_summary": {
            "group1_substring_invariant": f"{group1_passed}/10",
            "group2_limits_security": f"{group2_passed}/10",
            "group3_bm25_discrimination": f"{group3_passed}/15",
            "group4_status_classification": f"{group4_passed}/10",
            "group5_router_fix_latency": f"{group5_passed}/5",
        },
        "results": results,
    }

    out_dir = REPO_ROOT / "logs" / "autonomous_steps_44_46"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step44_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to: {out_path}")

    return report


if __name__ == "__main__":
    report = run_evaluation()
    if report["total_passed"] != report["total_tests"]:
        sys.exit(1)
