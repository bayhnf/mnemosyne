"""
Self-Harmonizing Memory Reasoning (SHMR)
========================================
Built on ECHO-OR research (AxDSan/ECHO-OR) but fully rearchitected for
continuous local memory orchestration inside Mnemosyne's BEAM architecture.

Core idea: related memories "echo" each other in the background, negotiating
contradictions, surfacing hidden patterns, and converging into stable beliefs.

This is Mnemosyne's signature reasoning layer -- no Honcho dreams, no Hindsight
reflections, no Mem0 static graphs. Memories actively resonate and self-correct.
"""

import os
import time
import logging
import json
from typing import List, Dict, Optional

import numpy as np

from mnemosyne.core import embeddings as _embeddings

logger = logging.getLogger("mnemosyne.shmr")

# --- Config ---
SHMR_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_BATCH_SIZE", "50"))
SHMR_MAX_ITERATIONS = int(os.environ.get("MNEMOSYNE_SHMR_MAX_ITERATIONS", "3"))
SHMR_SIMILARITY_THRESHOLD = float(
    os.environ.get("MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD", "0.70")
)
SHMR_HARMONY_THRESHOLD = float(
    os.environ.get("MNEMOSYNE_SHMR_HARMONY_THRESHOLD", "0.60")
)
SHMR_MODEL = os.environ.get("MNEMOSYNE_SHMR_MODEL", "")
SHMR_MIN_CLUSTER_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE", "2"))
SHMR_TEMPERATURE = float(os.environ.get("MNEMOSYNE_SHMR_TEMPERATURE", "0.2"))

EMBEDDING_DIM = _embeddings.EMBEDDING_DIM  # derived from configured model

# --- SQL Schema ---
FACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS harmonic_beliefs (
    belief_id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    provenance TEXT,   -- JSON array of source fact_ids or memory_ids
    cluster_id TEXT,
    iteration INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_resonance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    cluster_count INTEGER,
    beliefs_generated INTEGER,
    contradictions_resolved INTEGER,
    harmony_score_avg REAL,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON harmonic_beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON harmonic_beliefs(predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON harmonic_beliefs(confidence);
"""


# --- Proposal schema (Task 4: Dream proposal foundation) ---
# shmr_proposals is the ONLY table SHMR writes during propose_harmony().
# It records read-only synthesized candidates for Dream to plan against.
# Source tables (facts / working_memory / episodic_memory) are never mutated.
PROPOSAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shmr_proposals (
    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    cited_source_ids TEXT NOT NULL,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence REAL,
    action TEXT,
    target_source_id TEXT,
    rationale TEXT,
    status TEXT DEFAULT 'proposed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_shmr_proposals_run ON shmr_proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_shmr_proposals_session ON shmr_proposals(session_id);
CREATE INDEX IF NOT EXISTS idx_shmr_proposals_status ON shmr_proposals(status);
"""


def _init_proposal_schema(conn):
    """Ensure the proposal/audit table exists. Source tables are untouched."""
    conn.executescript(PROPOSAL_SCHEMA_SQL)
    conn.commit()


def _init_schema(conn):
    """Ensure SHMR tables exist."""
    conn.executescript(FACTS_SCHEMA_SQL)
    conn.commit()


def _embed(text: str) -> np.ndarray:
    """Embed text using Mnemosyne's embedding pipeline (BAAI/bge-small)."""
    emb = _embeddings.embed(text)
    if emb.ndim > 1:
        emb = emb.flatten()
    return emb.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def _cluster_by_similarity(
    items: List[Dict],
    threshold: float,
) -> List[List[Dict]]:
    """Greedy connected-components clustering by cosine similarity.

    Each item must have an 'embedding' key with a numpy array.
    Returns list of clusters (each cluster is a list of items).
    """
    if not items:
        return []

    n = len(items)
    # Build adjacency: items are connected if sim >= threshold
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(items[i]["embedding"], items[j]["embedding"])
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components (BFS)
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        cluster = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(items[node])
            stack.extend(adj[node] - visited)
        clusters.append(cluster)

    return clusters


def _format_cluster_for_llm(cluster: List[Dict]) -> str:
    """Format a memory cluster as a prompt for the LLM harmonizer."""
    lines = ["=== MEMORY CLUSTER ==="]
    for i, item in enumerate(cluster):
        subject = item.get("subject", "unknown")
        predicate = item.get("predicate", "stated")
        obj = item.get("object", item.get("content", ""))
        confidence = item.get("confidence", 0.5)
        source = item.get("source", "fact")
        lines.append(
            f"[{i}] ({source}, conf={confidence:.2f}) {subject} | {predicate} | {obj}"
        )
    return "\n".join(lines)


HARMONY_PROMPT = """You are the Self-Harmonizing Memory Reasoner for Mnemosyne.
These memories belong to the same semantic cluster -- they all relate to the
same entities, topics, or events. Your job is to harmonize them:

1. **Resolve contradictions**: If two memories conflict, determine which is more
   likely true based on recency, specificity, and internal consistency. Flag the
   weaker one as dampened, not deleted.
2. **Extract higher-order beliefs**: Find patterns that span multiple memories.
   What does this cluster as a whole tell us? What's the stable truth?
3. **Dampen noise, amplify signal**: Low-confidence or stale memories get lower
   weight. Corroborated facts get reinforced.
4. **Output only stable beliefs**: Return NEW or UPDATED facts with confidence
   scores. Don't regurgitate every input fact -- synthesize.

Output as JSON array of belief objects:
[{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0,
  "action": "create"|"update"|"dampen", "target_fact_id": null|"fact_id",
  "rationale": "one sentence explaining why"}]

RULES:
- Confidence 0.9+ = highly corroborated (multiple sources agree)
- Confidence 0.5-0.8 = reasonable inference from the cluster
- Confidence <0.4 = speculative, mark as such
- Use "dampen" to reduce confidence of contradicted facts (never delete)
- Use "update" to modify an existing fact with new information
- Output 1-5 beliefs per cluster (don't over-generate)"""


def _call_llm(prompt: str, system: str = "") -> str:
    """Call the configured LLM for harmonization.

    Uses the same LLM chain as mnemosyne_sleep's summarization:
    local_llm first, fallback to cloud extraction client.
    """
    # Try local LLM first
    try:
        from mnemosyne.core.local_llm import _call_local_llm

        result = _call_local_llm(prompt, system=system, temperature=SHMR_TEMPERATURE)
        if result and len(result.strip()) > 10:
            return result
    except Exception:
        pass

    # Fallback to the cloud extraction client.
    #
    # This path was dead. It imported ExtractionConfig and ExtractionClient
    # from mnemosyne.core.extraction, which exports neither: they live in
    # the mnemosyne.extraction package. The ImportError was swallowed by the
    # bare except below, so the fallback silently never ran and only the
    # local GGUF path could ever produce a belief. It also passed a config
    # object positionally to ExtractionClient(model=...), which would have
    # been wrong even with the right import.
    #
    # MNEMOSYNE_SHMR_MODEL, when set, overrides the model for harmonization
    # only. Harmonization is a reasoning task and may warrant a stronger
    # model than extraction. That variable was previously read at import and
    # then never used.
    try:
        from mnemosyne.extraction import ExtractionClient

        client = ExtractionClient(model=SHMR_MODEL or None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = client.chat(messages, temperature=SHMR_TEMPERATURE)
        if result:
            return result
    except Exception:
        logger.debug("SHMR cloud fallback failed", exc_info=True)

    return ""


def _compute_harmony_score(
    beliefs: List[Dict],
    cluster: List[Dict],
) -> float:
    """Score how well the harmonized beliefs represent the cluster.

    Uses cosine similarity between belief embeddings and cluster centroid,
    plus a consistency bonus for beliefs that don't contradict each other.
    """
    if not beliefs or not cluster:
        return 0.0

    # Compute cluster centroid
    cluster_embs = np.array(
        [item.get("embedding", np.zeros(EMBEDDING_DIM)) for item in cluster]
    )
    centroid = cluster_embs.mean(axis=0)

    # Score each belief against centroid
    belief_scores = []
    for belief in beliefs:
        belief_text = f"{belief.get('predicate', '')} {belief.get('object', '')}"
        try:
            belief_emb = _embed(belief_text)
            sim = _cosine_similarity(belief_emb, centroid)
            belief_scores.append(sim * belief.get("confidence", 0.5))
        except Exception:
            belief_scores.append(0.3)

    # Consistency bonus: penalize if beliefs contradict each other
    consistency_bonus = 1.0
    if len(beliefs) > 1:
        belief_embs = []
        for b in beliefs:
            try:
                belief_embs.append(
                    _embed(f"{b.get('predicate', '')} {b.get('object', '')}")
                )
            except Exception:
                belief_embs.append(np.zeros(EMBEDDING_DIM))
        belief_embs = np.array(belief_embs)

        # Check pairwise similarity of beliefs (if they're too different,
        # that suggests the LLM produced contradictory beliefs)
        pairwise_sims = []
        for i in range(len(belief_embs)):
            for j in range(i + 1, len(belief_embs)):
                pairwise_sims.append(_cosine_similarity(belief_embs[i], belief_embs[j]))
        if pairwise_sims:
            avg_pairwise = np.mean(pairwise_sims)
            # Lower pairwise similarity = potential contradiction = penalty
            consistency_bonus = min(1.0, avg_pairwise + 0.3)

    avg_belief_score = np.mean(belief_scores) if belief_scores else 0.0
    return float(avg_belief_score * consistency_bonus)


def _extract_json_from_llm_output(text: str) -> List[Dict]:
    """Robust JSON extraction from LLM output (handles markdown wrappers)."""
    import re

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "beliefs" in parsed:
            return parsed["beliefs"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` block
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try extracting bare array
    array_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse line by line for { ... } objects
    objects = re.findall(r"\{[^{}]*\}", text)
    results = []
    for obj_str in objects:
        try:
            results.append(json.loads(obj_str))
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def _apply_beliefs(conn, beliefs, cluster, cluster_id):
    """Deprecated no-op (Task 4 hardening).

    The legacy implementation UPDATEd ``facts`` and wrote
    ``harmonic_beliefs``. SHMR is now proposal-only: nothing in this
    module mutates source rows. This function is retained as an import-
    stable symbol but does nothing and writes nothing, so any stray
    reference cannot reintroduce source mutation.
    """
    logger.debug(
        "_apply_beliefs is a deprecated no-op; SHMR is proposal-only. "
        "Use propose_harmony() for Dream-candidate persistence."
    )
    return None


def harmonize(
    beam,
    batch_size: int = None,
    max_iterations: int = None,
    similarity_threshold: float = None,
) -> Dict:
    """Proposal-only SHMR cycle (Task 4 hardening).

    Historically this was the source-mutating entry point: it UPDATEd
    facts, wrote harmonic_beliefs, and queried a ``facts.status`` column
    that does not exist in the standard Mnemosyne schema. As of Task 4 it
    is a thin wrapper around :func:`propose_harmony` and is strictly
    read-only on sources. It persists only to ``shmr_proposals`` and never
    applies, dampens, or deletes source rows.

    The signature is preserved for compatibility. ``max_iterations`` is
    accepted but ignored (propose_harmony is single-pass); it is retained
    only so existing call sites do not break.

    Args:
        beam: BeamMemory instance.
        batch_size, similarity_threshold: forwarded to propose_harmony.
        max_iterations: ignored (kept for signature compatibility).

    Returns:
        Dict from propose_harmony, with legacy keys (beliefs_generated,
        contradictions_resolved, harmony_score_avg) mirrored from the
        proposal-only counters so old consumers do not KeyError.
    """
    result = propose_harmony(
        beam,
        llm_call=_call_llm,
        batch_size=batch_size,
        similarity_threshold=similarity_threshold,
    )
    # Mirror legacy keys for any consumer that still reads them.
    result.setdefault("beliefs_generated", result.get("proposals_persisted", 0))
    result.setdefault("contradictions_resolved", result.get("proposals_rejected", 0))
    result.setdefault("harmony_score_avg", 0.0)
    return result


def recall_beliefs(beam, query: str, top_k: int = 10) -> List[Dict]:
    """Search harmonic beliefs for a given query.

    Used by recall() when harmonic=True flag is set.
    """
    cursor = beam.conn.cursor()
    _init_schema(beam.conn)

    try:
        query_emb = _embed(query)

        # Search by embedding on object text
        results = []
        rows = cursor.execute(
            """
            SELECT belief_id, subject, predicate, object, confidence,
                   provenance, created_at
            FROM harmonic_beliefs
            ORDER BY confidence DESC
            LIMIT ?
        """,
            (top_k * 2,),
        ).fetchall()

        # Score by embedding similarity
        scored = []
        for row in rows:
            try:
                belief_emb = _embed(row["object"])
                sim = _cosine_similarity(query_emb, belief_emb)
                scored.append((sim * row["confidence"], row))
            except Exception:
                scored.append((row["confidence"] * 0.3, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        for score, row in scored[:top_k]:
            results.append(
                {
                    "content": row["object"],
                    "score": round(score, 4),
                    "belief_id": row["belief_id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "provenance": row["provenance"],
                    "source": "harmonic_belief",
                }
            )

        return results
    except Exception:
        return []


# ============================================================
#  Phase 3A: Reflective Recall (single-pass fact synthesis)
# ============================================================

REFLECTION_PROMPT = """You are a memory reasoning assistant. You have retrieved facts
from a conversation database and need to synthesize a coherent answer.

QUESTION: {question}

RETRIEVED FACTS:
{fact_context}

Based on these facts, provide a concise synthesis (2-4 sentences) that:
1. Answers the question directly if the facts are sufficient
2. Identifies any contradictions or gaps in the facts
3. Notes temporal context (dates, order of events) if present
4. If facts are insufficient, states what's missing clearly

SYNTHESIS:"""


def reflect(
    beam, question: str, facts: List[Dict] = None, top_k: int = 10
) -> Optional[str]:
    """Single-pass reflective synthesis over retrieved facts.

    Takes a question and a list of fact dicts (from fact_recall()), sends them
    to an LLM, and returns a coherent synthesis paragraph. This synthesis is
    then injected as additional context for the final answering LLM.

    This is Phase 3A: lightweight, works with any LLM, no iteration needed.
    Phase 3B (SHMR harmonize()) replaces this with multi-iteration harmony loop.

    Args:
        beam: BeamMemory instance (for fact_recall if facts not provided)
        question: The question to synthesize for
        facts: Pre-retrieved facts (if None, calls fact_recall automatically)
        top_k: Max facts to include in the reflection

    Returns:
        Synthesis string, or None if no facts available.
    """
    # Get facts if not provided
    if facts is None and beam is not None:
        try:
            facts = beam.fact_recall(question, top_k=top_k)
        except Exception:
            return None

    if not facts:
        return None

    # Build fact context (limit to top_k, sort by score)
    sorted_facts = sorted(facts, key=lambda f: f.get("score", 0), reverse=True)[:top_k]
    fact_lines = []
    for i, f in enumerate(sorted_facts):
        content = f.get("content", "")
        score = f.get("score", 0.5)
        source = f.get("source", "fact")
        fact_lines.append(f"[{i}] ({source}, conf={score:.2f}) {content}")

    fact_context = "\n".join(fact_lines)
    prompt = REFLECTION_PROMPT.format(question=question, fact_context=fact_context)

    synthesis = _call_llm(prompt)
    if synthesis and len(synthesis.strip()) > 10:
        return synthesis.strip()
    return None


def get_resonance_log(beam, limit: int = 10) -> List[Dict]:
    """Get recent harmonization run logs."""
    cursor = beam.conn.cursor()
    _init_schema(beam.conn)
    try:
        rows = cursor.execute(
            """
            SELECT * FROM memory_resonance_log
            ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ============================================================
#  Task 4: Safe SHMR -> Dream proposal foundation
# ============================================================
#
# propose_harmony() is the proposal-only successor to the legacy harmonize().
# Differences that matter to Dream (Task 5) and to safety:
#
#   * Read-only on sources. It never UPDATEs/DELETEs facts, working_memory,
#     episodic_memory, or canonical state. The only table it writes is
#     shmr_proposals. Legacy harmonize() mutated facts in place; that path is
#     retained for compatibility but has no caller in the shipped code.
#   * Fresh-schema safe. It queries only columns that exist in the standard
#     Mnemosyne facts table (fact_id, session_id, subject, predicate, object,
#     confidence, timestamp). It does NOT reference a legacy facts.status.
#   * Stable source IDs in every prompt. Each candidate sent to the LLM is
#     labelled with its source id, and the prompt instructs the model to cite
#     those ids back. Without this Dream cannot build a manifest.
#   * Out-of-cluster target rejection. An LLM-returned target_source_id that is
#     not one of the current cluster's source ids is dropped and counted as
#     rejected, never persisted. This blocks hallucinated targets.
#   * Explicit scope. Every persisted proposal records session_id plus any
#     available producer/actor fields off the source rows, so Task 5 can route
#     a proposal without re-reading sources.
#   * Transactional persistence. Proposal writes happen inside a savepoint; a
#     failure rolls the proposals back and leaves sources untouched. The
#     function returns status="rolled_back" rather than raising.
#   * No network by default. The LLM is injected via the llm_call seam, so
#     tests and Dream can drive it deterministically. The legacy _call_llm
#     network path is not used here.

import hashlib as _hashlib


PROPOSAL_PROMPT_TEMPLATE = """You are the Mnemosyne Self-Harmonizing Memory Reasoner.
You are given one semantic cluster of source memories that already relate to the
same entities or topics. Synthesize candidate beliefs that Dream will later
decide whether to apply. You do NOT apply anything yourself.

Each source memory is labelled with a stable SOURCE ID in square brackets.
When you propose a belief, you MUST set "target_source_id" to one of the SOURCE
IDs from this cluster (or null for a brand-new belief). Do not invent ids.

=== SOURCE CLUSTER (session={session}) ===
{cluster_block}

Return ONLY a JSON array. Each object has keys:
  subject, predicate, object, confidence (0.0-1.0),
  action ("create"|"update"|"dampen"),
  target_source_id (one of the cluster's source ids, or null),
  rationale (one short sentence).

Rules:
- target_source_id must be null for "create", and must be a cluster source id
  for "update"/"dampen". Any other value will be rejected.
- Confidence 0.9+ = corroborated by multiple sources; 0.5-0.8 = reasonable
  inference; <0.4 = speculative.
- Output 0-5 beliefs. Return [] if nothing is stable enough.
"""


_DEFAULT_EMBED_FN = None


def _embedding_fn():
    """Return the active embedding callable, or None to use the lexical fallback.

    Production resolves to ``mnemosyne.core.embeddings.embed`` when a backend
    is configured. Tests monkeypatch this (or ``_DEFAULT_EMBED_FN``) to force
    the deterministic offline path and prove SHMR never reaches the network
    even when fastembed is installed with an uncached model (Finding 3).
    """
    if _DEFAULT_EMBED_FN is not None:
        return _DEFAULT_EMBED_FN
    if _embeddings._is_disabled():
        return None
    return _embeddings.embed


def _coerce_confidence(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce an LLM-supplied confidence value to a finite float in [0,1].

    Returns None when the value is non-numeric, non-finite, or a bool. The
    caller treats None as untrusted/malformed output and counts the proposal
    as rejected rather than persisting it or raising (Finding 4). bool is
    rejected before float(): it subclasses int, so float(True) would silently
    read a JSON ``true`` as full confidence. NaN must never survive: it
    compares False against every threshold so downstream gates pass it.
    """
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if f != f:  # NaN
        return None
    import math

    if not math.isfinite(f):
        return None
    return max(0.0, min(1.0, f))


def _scope_key(row: Dict) -> tuple:
    """Deterministic provenance key for partitioning candidates (Finding 5).

    Two candidates with different session, producer (author_type), actor
    (author_id), or project (channel_id) must NOT be merged into one cluster.
    Returns a tuple of the explicit scope fields so clustering can partition
    by scope before running similarity. We do not infer missing fields: a
    None field is its own bucket so a candidate never silently inherits
    another candidate's producer/actor/project.
    """
    return (
        row.get("session_id"),
        row.get("author_type"),
        row.get("author_id"),
        row.get("channel_id"),
    )


def _scope_from_row(row: Dict) -> Dict:
    """Build a complete, deterministic scope record from a single source row.

    Every key that has a non-empty value is recorded; nothing is inferred
    from cluster[0] as authority (Finding 5)."""
    scope = {}
    for key in ("session_id", "author_id", "author_type", "channel_id"):
        val = row.get(key)
        if val not in (None, ""):
            scope[key] = val
    return scope


def _row_scope(row: Dict, session_id: str) -> Dict:
    """Pull explicit scope off a source row for the Dream manifest.

    We do not infer scope; we only copy fields that are actually present on
    the row. session_id always carries; author_id / channel_id are included
    when the column exists and is non-null.
    """
    scope = {"session_id": session_id}
    for key in ("author_id", "author_type", "channel_id"):
        val = row.get(key)
        if val not in (None, ""):
            scope[key] = val
    return scope


def _gather_candidates(
    beam, batch_size: int, degraded_reasons: Optional[List[str]] = None
) -> List[Dict]:
    """Read echo candidates from the standard schema. Read-only.

    Pulls active facts and recent episodic memories for the beam's session.
    Embeddings are computed lazily and only when an embedding backend is
    reachable; if it isn't, we fall back to lexical clustering on subject so
    SHMR still produces a result instead of crashing. The legacy harmonize()
    called _embed unconditionally and crashed in any env without a model.
    """
    cursor = beam.conn.cursor()
    candidates: List[Dict] = []

    rows = cursor.execute(
        "SELECT fact_id, session_id, subject, predicate, object, confidence, "
        "timestamp FROM facts WHERE session_id = ? OR session_id IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (beam.session_id, batch_size),
    ).fetchall()
    for row in rows:
        candidates.append(
            {
                "source_id": row["fact_id"],
                "source_table": "facts",
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "confidence": row["confidence"]
                if row["confidence"] is not None
                else 0.5,
                "timestamp": row["timestamp"],
                "session_id": row["session_id"] or beam.session_id,
                "author_id": None,
                "author_type": None,
                "channel_id": None,
            }
        )

    try:
        ep_rows = cursor.execute(
            "SELECT id, content, importance, session_id, author_id, "
            "author_type, channel_id FROM episodic_memory "
            "WHERE session_id = ? OR session_id IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (beam.session_id, max(1, batch_size // 2)),
        ).fetchall()
    except Exception:
        if degraded_reasons is not None:
            degraded_reasons.append("episodic_fetch_failed")
        logger.debug(
            "SHMR episodic fetch failed; continuing without episodic candidates"
        )
        ep_rows = []
    for row in ep_rows:
        content = row["content"] or ""
        if len(content) <= 10:
            continue
        candidates.append(
            {
                "source_id": row["id"],
                "source_table": "episodic_memory",
                "subject": "memory",
                "predicate": "contains",
                "object": content[:300],
                "confidence": row["importance"]
                if row["importance"] is not None
                else 0.5,
                "timestamp": None,
                "session_id": row["session_id"] or beam.session_id,
                "author_id": row["author_id"] if "author_id" in row.keys() else None,
                "author_type": row["author_type"]
                if "author_type" in row.keys()
                else None,
                "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
            }
        )

    # Embed when a backend is reachable; otherwise fall back to the
    # deterministic lexical vector so clustering still works without a model.
    # The embed function is swappable (_embedding_fn) so tests can force the
    # offline path and prove no network access occurs (Finding 3).
    embed_fn = _embedding_fn()
    dense_used = False
    if embed_fn is not None:
        try:
            texts = [c["object"] for c in candidates]
            embs = embed_fn(texts)
            if (
                embs is not None
                and hasattr(embs, "shape")
                and embs.shape[0] == len(candidates)
            ):
                for i, c in enumerate(candidates):
                    c["embedding"] = embs[i].astype(np.float32).flatten()
                dense_used = True
        except Exception:
            if degraded_reasons is not None:
                degraded_reasons.append("embedding_lexical_fallback")
            logger.debug("SHMR dense embedding failed; using lexical fallback")
            dense_used = False
    if not dense_used:
        for c in candidates:
            c["embedding"] = _lexical_vector(c["object"], c["subject"])
    return candidates


def _normalize_token(tok: str) -> str:
    tok = tok.lower()
    tok = "".join(ch for ch in tok if ch.isalnum())
    if len(tok) > 4:
        for suf in ("ing", "ed", "es", "s", "ly"):
            if tok.endswith(suf):
                tok = tok[: -len(suf)]
                break
    return tok


def _lexical_vector(text: str, subject: str) -> np.ndarray:
    """Deterministic fallback embedding when no model is loaded.

    Bags normalized+lightly-stemmed tokens of subject+object into a fixed-size
    vector via hashing. Good enough to put near-duplicate paraphrases in the
    same cluster so propose_harmony is testable without a network/model. Real
    deployments with a configured embedding backend get dense vectors instead.

    ponytail: ceiling is lexical overlap -- true semantic paraphrase clusters
    still need a real embedding model. Upgrade path: configure
    MNEMOSYNE_EMBEDDING_MODEL and the dense path in _gather_candidates wins.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for raw in (str(subject) + " " + str(text)).split():
        nt = _normalize_token(raw)
        if not nt:
            continue
        h = int(_hashlib.md5(nt.encode()).hexdigest(), 16)
        vec[h % EMBEDDING_DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _format_cluster_block(cluster: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(cluster):
        sid = item.get("source_id", "")
        subject = item.get("subject", "unknown")
        predicate = item.get("predicate", "stated")
        obj = item.get("object", "")
        conf = item.get("confidence", 0.5)
        lines.append(
            f"[{i}] source_id={sid} ({subject} | {predicate} | {obj}) conf={conf:.2f}"
        )
    return "\n".join(lines)


def _partition_by_scope(candidates: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Group candidates by their explicit provenance key (Finding 5).

    Returns a dict mapping scope-key -> candidates with that exact scope.
    Clustering then runs WITHIN each partition so two memories with the same
    text but different producer/actor/project are never silently merged.
    """
    partitions: Dict[tuple, List[Dict]] = {}
    for c in candidates:
        key = _scope_key(c)
        partitions.setdefault(key, []).append(c)
    return partitions


def propose_harmony(
    beam,
    *,
    llm_call,
    batch_size: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
) -> Dict:
    """Run one read-only SHMR cycle and persist Dream proposals.

    Args:
        beam: BeamMemory instance with a fresh standard schema.
        llm_call: callable(prompt, system="") -> str. Injected so tests and
            Dream can drive SHMR deterministically; no network is used.
        batch_size, similarity_threshold, min_cluster_size: optional overrides.

    Returns:
        Dict with clusters_found, proposals_persisted, proposals_rejected,
        status in {"proposed", "no_convergence", "insufficient_candidates",
        "rolled_back"}, and on rollback a ``failure_reason`` string.

    Safety contract:
        - Never mutates facts / working_memory / episodic_memory / canonical.
        - Every persisted proposal records the exact source ids its LLM entry
          cited (validated as a subset of the cluster) plus an explicit,
          complete scope built from the candidate rows -- never cluster[0]
          as a silent authority.
        - Candidates are partitioned by provenance (session + producer/actor/
          project) before clustering, so mixed scope is never merged.
        - Rejects LLM target_source_id values and cited ids that are not
          cluster source ids.
        - The ENTIRE run is wrapped in one transaction; a failure on any
          cluster rolls back ALL proposal writes for the run and returns
          status="rolled_back" without raising.
    """
    t0 = time.perf_counter()
    batch_size = batch_size or SHMR_BATCH_SIZE
    similarity_threshold = (
        SHMR_SIMILARITY_THRESHOLD
        if similarity_threshold is None
        else similarity_threshold
    )
    min_cluster_size = (
        SHMR_MIN_CLUSTER_SIZE if min_cluster_size is None else min_cluster_size
    )

    _init_proposal_schema(beam.conn)
    degraded_reasons: List[str] = []
    candidates = _gather_candidates(beam, batch_size, degraded_reasons)

    if len(candidates) < min_cluster_size:
        return {
            "clusters_found": 0,
            "proposals_persisted": 0,
            "proposals_rejected": 0,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "status": "insufficient_candidates",
            "degraded_reasons": degraded_reasons,
        }

    # Finding 5: partition by provenance BEFORE clustering so mixed scope is
    # never silently merged into one cluster.
    all_clusters: List[List[Dict]] = []
    for part in _partition_by_scope(candidates).values():
        all_clusters.extend(
            c
            for c in _cluster_by_similarity(part, similarity_threshold)
            if len(c) >= min_cluster_size
        )

    run_id = f"shmr_{int(time.time() * 1000)}"
    total_persisted = 0
    total_rejected = 0
    any_cluster_proposed = False
    # Materialize per-cluster insert plans BEFORE opening the transaction, so
    # the transaction body only does INSERTs and a failure rolls the whole run
    # back atomically (Finding 2).
    plans: List[Dict] = []
    for cluster_idx, cluster in enumerate(all_clusters):
        cluster_id = f"{run_id}_c{cluster_idx}"
        cluster_source_ids = {c.get("source_id") for c in cluster if c.get("source_id")}
        # Finding 5: scope is built from the cluster's rows, but every row in
        # a scope-partitioned cluster shares the same scope key, so this is
        # complete and deterministic rather than cluster[0]-as-authority.
        scope = _scope_from_row(cluster[0])

        prompt = PROPOSAL_PROMPT_TEMPLATE.format(
            session=beam.session_id,
            cluster_block=_format_cluster_block(cluster),
        )
        try:
            raw = llm_call(prompt)
        except Exception:
            degraded_reasons.append("llm_call_failed")
            logger.debug("SHMR LLM call failed for cluster %s", cluster_id)
            raw = ""
        beliefs = _extract_json_from_llm_output(raw) if raw else []

        rows_to_insert: List[Dict] = []
        for b in beliefs:
            if not isinstance(b, dict):
                continue
            action = str(b.get("action") or "create").strip().lower()
            if action not in ("create", "update", "dampen"):
                total_rejected += 1
                continue
            target = b.get("target_source_id") or b.get("target_fact_id")
            target = str(target).strip() if target else None
            if action in ("update", "dampen"):
                if target is None or target not in cluster_source_ids:
                    total_rejected += 1
                    continue
            else:
                target = None

            # Finding 4: malformed confidence is rejected, not raised.
            confidence = _coerce_confidence(b.get("confidence", 0.5))
            if confidence is None:
                total_rejected += 1
                continue

            # Finding 6: validate per-proposal cited ids against the cluster.
            # If the model cites an id that is NOT in this cluster, that is
            # untrusted output: reject the proposal and count it. We do not
            # silently filter the bad id out (that would be inferring a
            # citation the model did not make). When the model provides no
            # explicit citation, we record full-cluster provenance, documented
            # as the broad-but-truthful default.
            raw_cited = b.get("cited_source_ids")
            if raw_cited is None:
                raw_cited = b.get("evidence_ids")
            if raw_cited is None:
                cited = sorted(cid for cid in cluster_source_ids if cid)
            else:
                cited_list = raw_cited if isinstance(raw_cited, list) else [raw_cited]
                cited_set = {str(x).strip() for x in cited_list if str(x).strip()}
                if not cited_set:
                    cited = sorted(cid for cid in cluster_source_ids if cid)
                elif not cited_set <= cluster_source_ids:
                    # At least one cited id is out-of-cluster: reject, do not
                    # silently drop the bad citation.
                    total_rejected += 1
                    continue
                else:
                    cited = sorted(cited_set)

            rows_to_insert.append(
                {
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "session_id": beam.session_id,
                    "scope_json": json.dumps(scope, sort_keys=True),
                    "cited_source_ids": json.dumps(cited),
                    "subject": b.get("subject"),
                    "predicate": b.get("predicate"),
                    "object": b.get("object"),
                    "confidence": confidence,
                    "action": action,
                    "target_source_id": target,
                    "rationale": b.get("rationale"),
                }
            )

        if rows_to_insert:
            any_cluster_proposed = True
            plans.extend(rows_to_insert)

    # Finding 2 + Round 2: one run-level transaction. Every INSERT for the
    # whole run happens inside a single SAVEPOINT. On ANY failure (including
    # a RELEASE failure) the savepoint is still live, so ROLLBACK TO always
    # works and no rows survive.
    #
    # Root cause fixed here: the old code RELEASEd the savepoint (which for
    # the outermost SQLite savepoint IS the commit), then called
    # beam.conn.commit() inside the same try. If that redundant commit raised,
    # the except path tried ROLLBACK TO SAVEPOINT shmr_run — which no longer
    # existed after RELEASE — swallowed the "no such savepoint" error, and
    # returned status="rolled_back" with the rows still live in the
    # connection state (persisted by the next commit on that connection).
    #
    # Fix: RELEASE on the outermost savepoint is the atomic durability point;
    # no separate commit() is needed or safe afterward. If any statement
    # raises before RELEASE, the savepoint is still live and ROLLBACK TO
    # undoes all INSERTs for this run. If RELEASE itself raises, the
    # savepoint is still live for the same reason. We never swallow the
    # rollback's own failure: if ROLLBACK TO also fails, we surface that as a
    # distinct failure reason rather than silently masking it.
    if plans:
        try:
            beam.conn.execute("SAVEPOINT shmr_run")
            for row in plans:
                beam.conn.execute(
                    "INSERT INTO shmr_proposals "
                    "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
                    "subject, predicate, object, confidence, action, "
                    "target_source_id, rationale) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["run_id"],
                        row["cluster_id"],
                        row["session_id"],
                        row["scope_json"],
                        row["cited_source_ids"],
                        row["subject"],
                        row["predicate"],
                        row["object"],
                        row["confidence"],
                        row["action"],
                        row["target_source_id"],
                        row["rationale"],
                    ),
                )
            beam.conn.execute("RELEASE SAVEPOINT shmr_run")
            total_persisted = len(plans)
        except Exception as exc:
            logger.warning("SHMR run %s rolled back: %s", run_id, exc, exc_info=False)
            rollback_reason = str(exc)
            try:
                beam.conn.execute("ROLLBACK TO SAVEPOINT shmr_run")
                beam.conn.execute("RELEASE SAVEPOINT shmr_run")
            except Exception as rb_exc:
                # Do NOT swallow a rollback failure. Surface it as a distinct
                # reason so the caller knows the transaction state may be
                # dirty, rather than silently claiming rolled_back.
                rollback_reason = f"{exc} (rollback also failed: {rb_exc})"
            return {
                "clusters_found": len(all_clusters),
                "proposals_persisted": 0,
                "proposals_rejected": total_rejected,
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "status": "rolled_back",
                "failure_reason": rollback_reason,
                "degraded_reasons": degraded_reasons,
            }

    return {
        "clusters_found": len(all_clusters),
        "proposals_persisted": total_persisted,
        "proposals_rejected": total_rejected,
        "duration_ms": int((time.perf_counter() - t0) * 1000),
        "status": "proposed" if any_cluster_proposed else "no_convergence",
        "degraded_reasons": degraded_reasons,
    }
