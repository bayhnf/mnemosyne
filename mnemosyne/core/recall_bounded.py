"""Task 3 — Authoritative bounded recall.

Additive post-hydration gate that returns a :class:`RecallEnvelope` with
hard result/token caps, strict isolation, and deterministic fallback,
while leaving legacy :meth:`BeamMemory.recall` untouched.

Every retrieval mode (linear, enhanced, associative, polyphonic, entity,
fact, MEMORIA) is hydrated into a flat candidate list and then passed
through one gate::

    hydrate → strict predicate → lifecycle check → deduplicate
            → rank → hard top_k → rendered-token budget

The bounded path is read-only: it never bumps ``recall_count``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Sources treated as pending proposals / Dream output — always excluded
# from bounded recall before any apply step. Proposals carry
# ``source='sleep_model_refresh_proposal'`` with ``metadata.status='pending'``
# (see mnemosyne/core/model_refresh.py). We exclude by source string so the
# gate does not need to parse metadata JSON on every row.
_PROPOSAL_SOURCES = frozenset({"sleep_model_refresh_proposal"})


@dataclass(frozen=True)
class RecallPolicy:
    """Authoritative filter + bound specification for bounded recall.

    ``only_active`` defaults True so expired (``valid_until`` in the past)
    and superseded (``superseded_by`` set) rows are excluded — this is the
    ``active`` lifecycle. Pending proposals and Dream output are always
    excluded regardless of this flag.
    """

    top_k: int = 20
    max_tokens: Optional[int] = None
    max_item_tokens: Optional[int] = None
    # Identity allowlists. None = unconstrained for that axis (but still
    # subject to session isolation unless include_shared/include_legacy_shared).
    producer_ids: Optional[Sequence[str]] = None
    actor_ids: Optional[Sequence[str]] = None
    project_ids: Optional[Sequence[str]] = None
    session_ids: Optional[Sequence[str]] = None
    producer_types: Optional[Sequence[str]] = None
    include_shared: bool = False
    include_legacy_shared: bool = False
    memory_types: Optional[Sequence[str]] = None
    veracity: Optional[Sequence[str]] = None
    source: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    only_active: bool = True
    require_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive int, got {self.top_k!r}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens!r}")
        if self.max_item_tokens is not None and self.max_item_tokens <= 0:
            raise ValueError(
                f"max_item_tokens must be positive, got {self.max_item_tokens!r}"
            )


@dataclass
class RecallEnvelope:
    """Bounded result envelope. Diagnostics are non-sensitive."""

    results: List[Dict[str, Any]] = field(default_factory=list)
    rendered_context: str = ""
    token_count: int = 0
    retrieval_mode: str = "recent_fallback"
    applied_filters: Dict[str, Any] = field(default_factory=dict)
    degradation_reasons: List[str] = field(default_factory=list)
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            # Stable, non-sensitive id: hash of timestamp + uuid. Never
            # embeds query content or user data.
            material = f"{datetime.now(timezone.utc).isoformat()}|{uuid.uuid4().hex}"
            self.trace_id = "rb_" + hashlib.sha256(material.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    from mnemosyne.core.token_counter import estimate_tokens

    return estimate_tokens(text)


def _render_row(row: Dict[str, Any], max_item_tokens: Optional[int]) -> str:
    """Render one candidate as a single context line, truncating to the
    per-item token cap when set."""
    content = (row.get("content") or "").strip()
    if max_item_tokens is not None:
        words = content.split()
        if len(words) > max_item_tokens:
            content = " ".join(words[:max_item_tokens])
    ts = (row.get("timestamp") or "")[:10] or "?"
    return f"- {content} ({ts})"


def _row_is_proposal(row: Dict[str, Any]) -> bool:
    return (row.get("source") or "") in _PROPOSAL_SOURCES


def _passes_policy(
    row: Dict[str, Any],
    policy: RecallPolicy,
    *,
    calling_session_id: str,
    now_iso: str,
) -> bool:
    """Strict predicate: identity allowlists, lifecycle, shared scope.

    This is the authoritative native policy — it does not rely on any
    environment scoping side effect.
    """

    # --- Always-on: proposals / Dream output ---
    if _row_is_proposal(row):
        return False

    # --- Lifecycle: active only ---
    if policy.only_active:
        valid_until = row.get("valid_until")
        if valid_until and valid_until <= now_iso:
            return False
        if row.get("superseded_by"):
            return False

    # --- Session isolation (native, not env-driven) ---
    row_scope = row.get("scope") or "session"
    row_session = row.get("session_id")
    if row_scope == "global":
        # global rows: gated by include_shared / include_legacy_shared.
        if not policy.include_shared and not policy.include_legacy_shared:
            return False
        if policy.include_legacy_shared and not policy.include_shared:
            # legacy-only: author_type NULL or 'legacy' (do not infer).
            at = row.get("author_type")
            if at is not None and at != "legacy":
                return False
    else:
        # session-scoped: must match the calling session or an allowlist.
        allowed_sessions = (
            None
            if policy.session_ids is None
            else set(policy.session_ids)
        )
        if allowed_sessions is None:
            if row_session is not None and row_session != calling_session_id:
                return False
        else:
            if row_session not in allowed_sessions:
                return False

    # --- Identity allowlists ---
    if policy.actor_ids is not None:
        if row.get("author_id") not in set(policy.actor_ids):
            return False
    if policy.producer_ids is not None:
        if row.get("author_id") not in set(policy.producer_ids):
            return False
    if policy.producer_types is not None:
        if row.get("author_type") not in set(policy.producer_types):
            return False
    if policy.project_ids is not None:
        if row.get("channel_id") not in set(policy.project_ids):
            return False

    # --- Content filters ---
    if policy.memory_types is not None:
        if (row.get("memory_type") or "unknown") not in set(policy.memory_types):
            return False
    if policy.veracity is not None:
        if (row.get("veracity") or "unknown") not in set(policy.veracity):
            return False
    if policy.source is not None:
        if row.get("source") != policy.source:
            return False
    ts = row.get("timestamp") or ""
    if policy.from_date and ts < policy.from_date:
        return False
    if policy.to_date and ts > f"{policy.to_date}T23:59:59":
        return False
    return True


def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate ids, keeping the first (highest-ranked) occurrence."""
    seen = set()
    out = []
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def _rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable sort by score descending (ties keep insertion order)."""
    return sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)


def _apply_token_budget(
    rows: List[Dict[str, Any]],
    policy: RecallPolicy,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """Apply hard top_k then rendered-token budget. Returns
    (kept_rows, rendered_context, token_count).

    When ``max_item_tokens`` is set, each kept row's ``content`` is
    truncated in-place so downstream consumers see the bounded form,
    not just the rendered context."""
    rows = rows[: policy.top_k]
    lines: List[str] = []
    kept: List[Dict[str, Any]] = []
    for r in rows:
        if policy.max_item_tokens is not None:
            words = (r.get("content") or "").split()
            if len(words) > policy.max_item_tokens:
                r = dict(r)
                r["content"] = " ".join(words[: policy.max_item_tokens])
        line = _render_row(r, policy.max_item_tokens)
        if policy.max_tokens is not None:
            projected = _estimate_tokens("\n".join(lines + [line]))
            if projected > policy.max_tokens and lines:
                break
        lines.append(line)
        kept.append(r)
    context = "\n".join(lines)
    return kept, context, _estimate_tokens(context)


# ---------------------------------------------------------------------------
# Candidate hydration (read-only; no recall_count mutation)
# ---------------------------------------------------------------------------


def _hydrate_candidates(
    beam,
    query: str,
    policy: RecallPolicy,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Hydrate candidates from the available retrieval paths without
    side effects. Returns (rows, retrieval_mode, degradation_reasons).

    Degradation is deterministic: vector → FTS → bounded recent fallback.
    """
    from mnemosyne.core import beam as _beam_mod

    degradation: List[str] = []
    conn = beam.conn
    now_iso = _now_iso()
    query_lower = query.lower()
    query_words = _beam_mod._recall_tokens(query_lower)

    # Build identity/session SQL fragments from the policy (native).
    where_parts: List[str] = [
        "(valid_until IS NULL OR valid_until > ?)",
        "superseded_by IS NULL",
    ]
    params: List[Any] = [now_iso]
    # session scope: native isolation unless include_shared opens globals.
    if policy.include_shared or policy.include_legacy_shared:
        where_parts.append("(1=1)")
    else:
        where_parts.append("(session_id = ? OR scope = 'global')")
        params.append(beam.session_id)
    if policy.session_ids is not None:
        ph = ",".join("?" * len(policy.session_ids))
        where_parts.append(f"session_id IN ({ph})")
        params.extend(policy.session_ids)
    if policy.actor_ids is not None:
        ph = ",".join("?" * len(policy.actor_ids))
        where_parts.append(f"author_id IN ({ph})")
        params.extend(policy.actor_ids)
    if policy.producer_types is not None:
        ph = ",".join("?" * len(policy.producer_types))
        where_parts.append(f"author_type IN ({ph})")
        params.extend(policy.producer_types)
    if policy.project_ids is not None:
        ph = ",".join("?" * len(policy.project_ids))
        where_parts.append(f"channel_id IN ({ph})")
        params.extend(policy.project_ids)
    if policy.memory_types is not None:
        ph = ",".join("?" * len(policy.memory_types))
        where_parts.append(f"memory_type IN ({ph})")
        params.extend(policy.memory_types)
    if policy.veracity is not None:
        ph = ",".join("?" * len(policy.veracity))
        where_parts.append(f"veracity IN ({ph})")
        params.extend(policy.veracity)
    if policy.source is not None:
        where_parts.append("source = ?")
        params.append(policy.source)
    if policy.from_date:
        where_parts.append("timestamp >= ?")
        params.append(f"{policy.from_date}T00:00:00")
    if policy.to_date:
        where_parts.append("timestamp <= ?")
        params.append(f"{policy.to_date}T23:59:59")
    # Exclude proposals at SQL level too (defence in depth).
    where_parts.append("source NOT IN ('sleep_model_refresh_proposal')")
    where_sql = " AND ".join(where_parts)

    wm_cols = (
        "id, content, source, timestamp, session_id, importance, "
        "recall_count, last_recalled, valid_until, superseded_by, scope, "
        "author_id, author_type, channel_id, veracity, memory_type"
    )

    candidates: Dict[str, Dict[str, Any]] = {}
    mode = "recent_fallback"
    had_vector = False
    had_fts = False

    # --- Vector path (working + episodic) ---
    embeddings_available = _beam_mod._embeddings.available()
    query_embedding = None
    if embeddings_available:
        try:
            query_embedding = _beam_mod._embeddings.embed_query(query)
        except Exception:
            logger.info("bounded: query embedding failed", exc_info=True)
            query_embedding = None

    if query_embedding is not None:
        # Working memory vector search.
        try:
            wm_vec = _beam_mod._wm_vec_search(
                conn, query_embedding, k=max(policy.top_k * 3, 50),
                where_sql="wm.superseded_by IS NULL AND (wm.valid_until IS NULL OR wm.valid_until > ?)",
                where_params=(now_iso,),
            )
            for vr in wm_vec:
                had_vector = True
                candidates.setdefault(vr["id"], {"id": vr["id"], "_vec_sim": vr["sim"]})
        except Exception:
            logger.info("bounded: wm vec search failed", exc_info=True)
        # Episodic vector search.
        try:
            if _beam_mod._vec_available(conn):
                vec_rows = _beam_mod._vec_search(
                    conn, query_embedding.tolist(), k=max(policy.top_k * 3, 20),
                )
            else:
                vec_rows = _beam_mod._in_memory_vec_search(
                    conn, query_embedding, k=max(policy.top_k * 3, 20),
                )
            if vec_rows:
                max_distance = max(vr["distance"] for vr in vec_rows)
                for vr in vec_rows:
                    had_vector = True
                    sim = (
                        max(0.0, 1.0 - (vr["distance"] / max_distance))
                        if max_distance > 0
                        else 1.0
                    )
                    candidates.setdefault(vr["rowid"], {"id": None, "_rowid": vr["rowid"], "_vec_sim": sim})
        except Exception:
            logger.info("bounded: episodic vec search failed", exc_info=True)

    # --- FTS path (working + episodic) ---
    try:
        wm_fts = _beam_mod._fts_search_working(conn, query, k=max(policy.top_k * 3, 50))
    except Exception:
        wm_fts = []
    for fr in wm_fts:
        had_fts = True
        candidates.setdefault(fr["id"], {"id": fr["id"], "_fts_rank": fr["rank"]})
    try:
        em_fts = _beam_mod._fts_search(conn, query, k=max(policy.top_k * 3, 20))
    except Exception:
        em_fts = []
    for fr in em_fts:
        had_fts = True
        candidates.setdefault(("__rowid__", fr["rowid"]), {"id": None, "_rowid": fr["rowid"], "_fts_rank": fr["rank"]})

    # --- Resolve candidate ids → full rows ---
    wm_ids_to_fetch = [v["id"] for v in candidates.values() if v.get("id") and not v.get("_rowid")]
    em_rowids_to_fetch = [v["_rowid"] for v in candidates.values() if v.get("_rowid")]

    resolved: Dict[Any, Dict[str, Any]] = {}

    if wm_ids_to_fetch:
        ph = ",".join("?" * len(wm_ids_to_fetch))
        rows = conn.execute(
            f"SELECT {wm_cols} FROM working_memory WHERE id IN ({ph}) AND {where_sql}",
            (*wm_ids_to_fetch, *params),
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["_tier"] = "working"
            resolved[d["id"]] = d

    if em_rowids_to_fetch:
        ph = ",".join("?" * len(em_rowids_to_fetch))
        rows = conn.execute(
            f"SELECT rowid, {wm_cols} FROM episodic_memory WHERE rowid IN ({ph}) AND {where_sql}",
            (*em_rowids_to_fetch, *params),
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["_tier"] = "episodic"
            resolved[("__rowid__", d["rowid"])] = d

    # --- Determine retrieval mode ---
    if had_vector and had_fts:
        mode = "hybrid"
    elif had_vector:
        mode = "vector"
    elif had_fts:
        mode = "fts"
    else:
        mode = "recent_fallback"
        degradation.append("no_vector_or_fts_match")

    # --- Assemble scored candidate list ---
    scored: List[Dict[str, Any]] = []
    min_relevance = _beam_mod._minimum_recall_relevance(query_words)

    for key, cand in candidates.items():
        row = resolved.get(key)
        if row is None:
            continue
        vec_sim = cand.get("_vec_sim", 0.0)
        fts_rank = cand.get("_fts_rank")
        lexical = _beam_mod._lexical_relevance(query_words, row.get("content", ""), query_lower)
        decay = _beam_mod._recency_decay(row.get("timestamp", ""))
        importance = row.get("importance") or 0.5
        score = max(
            vec_sim * 0.5 + (lexical * 0.3 if fts_rank is not None else 0.0) + importance * 0.2,
            lexical * 0.8,
        ) * (0.7 + 0.3 * decay)
        if fts_rank is None and vec_sim == 0.0 and lexical < min_relevance:
            continue
        row["score"] = round(score, 4)
        row["dense_score"] = round(vec_sim, 4)
        row["fts_score"] = round((1.0 - min(1.0, abs(fts_rank))) if fts_rank is not None else 0.0, 4)
        scored.append(row)

    # --- Bounded recent fallback: if nothing matched, pull recent active rows ---
    if not scored and policy.require_fallback:
        fallback_rows = conn.execute(
            f"SELECT {wm_cols} FROM working_memory WHERE {where_sql} "
            f"ORDER BY timestamp DESC LIMIT {min(int(policy.top_k * 3), 200)}",
            tuple(params),
        ).fetchall()
        for row in fallback_rows:
            d = dict(row)
            d["_tier"] = "working"
            lexical = _beam_mod._lexical_relevance(query_words, d.get("content", ""), query_lower)
            decay = _beam_mod._recency_decay(d.get("timestamp", ""))
            score = lexical * 0.6 * (0.7 + 0.3 * decay)
            d["score"] = round(score, 4)
            d["dense_score"] = 0.0
            d["fts_score"] = 0.0
            scored.append(d)

    return scored, mode, degradation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def recall_bounded(beam, query: str, policy: Optional[RecallPolicy] = None) -> RecallEnvelope:
    """Run an authoritative bounded recall against ``beam``.

    Read-only: never mutates ``recall_count`` / ``last_recalled``.
    """
    if policy is None:
        policy = RecallPolicy()
    now_iso = _now_iso()

    # --- Polyphonic path (gated by env) ---
    if os.environ.get("MNEMOSYNE_POLYPHONIC_RECALL", "0") == "1":
        poly_rows, poly_mode, poly_degradation = _hydrate_polyphonic(beam, query, policy)
        if poly_rows:
            # Run polyphonic candidates through the same gate.
            gated = [
                r for r in poly_rows
                if _passes_policy(r, policy, calling_session_id=beam.session_id, now_iso=now_iso)
            ]
            gated = _dedupe(gated)
            gated = _rank(gated)
            kept, context, tokens = _apply_token_budget(gated, policy)
            return RecallEnvelope(
                results=kept,
                rendered_context=context,
                token_count=tokens,
                retrieval_mode="hybrid" if poly_mode == "polyphonic" else poly_mode,
                applied_filters=_applied_filters(policy, beam.session_id),
                degradation_reasons=poly_degradation,
            )
        # Empty / failed polyphonic → bounded linear fallback.
        linear_rows, mode, degradation = _hydrate_candidates(beam, query, policy)
        degradation.append("polyphonic_empty_fallback_linear")
    else:
        linear_rows, mode, degradation = _hydrate_candidates(beam, query, policy)

    # --- One gate for every mode ---
    gated = [
        r for r in linear_rows
        if _passes_policy(r, policy, calling_session_id=beam.session_id, now_iso=now_iso)
    ]
    gated = _dedupe(gated)
    gated = _rank(gated)
    kept, context, tokens = _apply_token_budget(gated, policy)

    return RecallEnvelope(
        results=kept,
        rendered_context=context,
        token_count=tokens,
        retrieval_mode=mode,
        applied_filters=_applied_filters(policy, beam.session_id),
        degradation_reasons=degradation,
    )


def _applied_filters(policy: RecallPolicy, session_id: str) -> Dict[str, Any]:
    """Non-sensitive diagnostics summary of what the policy enforced."""
    return {
        "top_k": policy.top_k,
        "max_tokens": policy.max_tokens,
        "max_item_tokens": policy.max_item_tokens,
        "include_shared": policy.include_shared,
        "include_legacy_shared": policy.include_legacy_shared,
        "only_active": policy.only_active,
        "has_actor_filter": policy.actor_ids is not None,
        "has_project_filter": policy.project_ids is not None,
        "has_session_filter": policy.session_ids is not None,
        "has_producer_type_filter": policy.producer_types is not None,
        "has_memory_type_filter": policy.memory_types is not None,
        "has_veracity_filter": policy.veracity is not None,
        "calling_session": True,  # boolean, not the id itself
    }


def _hydrate_polyphonic(beam, query: str, policy: RecallPolicy):
    """Hydrate candidates from the polyphonic engine, read-only.

    Returns (rows, mode, degradation_reasons). On engine failure returns
    ([], 'recent_fallback', ['polyphonic_engine_failed']).
    """
    degradation: List[str] = []
    try:
        from mnemosyne.core import beam as _beam_mod

        engine = beam._get_polyphonic_engine()
        query_embedding = None
        if _beam_mod._embeddings.available():
            try:
                vecs = _beam_mod._embeddings.embed([query])
                if vecs is not None and len(vecs) > 0:
                    query_embedding = vecs[0]
            except Exception:
                query_embedding = None
        poly_results = engine.recall(query=query, query_embedding=query_embedding, top_k=policy.top_k * 2)
    except Exception as exc:
        logger.info("bounded: polyphonic engine failed: %s", exc)
        return [], "recent_fallback", ["polyphonic_engine_failed"]

    if not poly_results:
        degradation.append("polyphonic_empty_result")
        return [], "recent_fallback", degradation

    out: List[Dict[str, Any]] = []
    cursor = beam.conn.cursor()
    for r in poly_results:
        memory_id = r.memory_id
        if memory_id.startswith("cf_"):
            continue
        row_dict = beam._fetch_polyphonic_row(cursor, memory_id)
        if row_dict is None:
            continue
        row_dict["score"] = r.combined_score
        row_dict["voice_scores"] = dict(r.voice_scores)
        out.append(row_dict)
    return out, "hybrid", degradation
