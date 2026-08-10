"""Task 3 — Authoritative bounded recall.

Additive post-hydration gate returning a :class:`RecallEnvelope` with
hard result/token caps, strict isolation, and deterministic fallback,
while leaving legacy :meth:`BeamMemory.recall` untouched.

Every retrieval mode — linear, enhanced, associative, polyphonic,
entity, fact, MEMORIA, and episodic supplements — hydrates into one flat
candidate list and passes through a single gate::

    hydrate → strict predicate → lifecycle → deduplicate
            → rank → hard top_k → rendered-token budget

The bounded path is **read-only**: it never bumps ``recall_count`` /
``last_recalled``. It calls only the read-only local helpers
(``_fts_search``, ``_wm_vec_search``, ``_find_memories_by_entity``,
``_find_memories_by_fact``, ``memoria_retrieve``, ``fact_recall``,
``_fetch_polyphonic_row``), never the side-effecting legacy
``recall()`` / ``recall_enhanced()`` / ``_recall_polyphonic()``.
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

# Sources treated as pending proposals / Dream output — always excluded.
_PROPOSAL_SOURCES = frozenset({"sleep_model_refresh_proposal"})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


def _is_strict_int(value: Any) -> bool:
    """Return True only for real ints (not bool, not float)."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class RecallPolicy:
    """Authoritative filter + bound specification for bounded recall.

    Producer corresponds to ``author_type`` (who/what produced the
    memory: human, agent, system, legacy). Actor corresponds to
    ``author_id`` (the identity of the producer instance).
    """

    top_k: int = 20
    max_tokens: Optional[int] = None
    max_item_tokens: Optional[int] = None
    # Identity allowlists. None = unconstrained for that axis.
    producer_ids: Optional[Sequence[str]] = None  # → author_type
    actor_ids: Optional[Sequence[str]] = None  # → author_id
    project_ids: Optional[Sequence[str]] = None  # → channel_id
    session_ids: Optional[Sequence[str]] = None  # → session_id
    producer_types: Optional[Sequence[str]] = None  # → author_type (alias)
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
        if not _is_strict_int(self.top_k) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive int, got {self.top_k!r}")
        if self.max_tokens is not None and (not _is_strict_int(self.max_tokens) or self.max_tokens <= 0):
            raise ValueError(f"max_tokens must be a positive int, got {self.max_tokens!r}")
        if self.max_item_tokens is not None and (not _is_strict_int(self.max_item_tokens) or self.max_item_tokens <= 0):
            raise ValueError(
                f"max_item_tokens must be a positive int, got {self.max_item_tokens!r}"
            )
        if not isinstance(self.include_shared, bool):
            raise ValueError(f"include_shared must be bool, got {type(self.include_shared)}")
        if not isinstance(self.include_legacy_shared, bool):
            raise ValueError(f"include_legacy_shared must be bool, got {type(self.include_legacy_shared)}")
        if not isinstance(self.only_active, bool):
            raise ValueError(f"only_active must be bool, got {type(self.only_active)}")
        if not isinstance(self.require_fallback, bool):
            raise ValueError(f"require_fallback must be bool, got {type(self.require_fallback)}")
        # I-6: every allowlist must be a sequence of strings (or None).
        # A bare string is rejected so it cannot be char-split into a
        # set of characters (e.g. "ab" -> {'a','b'}) by _passes_policy
        # or _build_where. Empty allowlists are allowed and fail closed
        # (match nothing) in both code paths.
        for _field in (
            "session_ids", "actor_ids", "producer_ids", "project_ids",
            "producer_types", "memory_types", "veracity",
        ):
            _val = getattr(self, _field)
            if _val is None:
                continue
            # str is a Sequence[str] by accident (iterates chars); reject it.
            if isinstance(_val, str) or not isinstance(_val, Sequence):
                raise ValueError(
                    f"{_field} must be a sequence of strings, got "
                    f"{type(_val).__name__}: {_val!r}"
                )
            for _el in _val:
                if not isinstance(_el, str):
                    raise ValueError(
                        f"{_field} must contain only strings, got "
                        f"element of type {type(_el).__name__}: {_el!r}"
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


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` estimated tokens.

    Uses binary search on the word-prefix to find the largest prefix whose
    estimated token count fits, since token estimation is not linearly
    proportional to word count (tiktoken merges subwords).
    """
    if max_tokens <= 0 or not text:
        return ""
    if _estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    if not words:
        return ""
    # Binary search for the largest word-prefix that fits.
    lo, hi = 0, len(words)
    best = ""
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[: mid + 1])
        if _estimate_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid
    return best


def _row_is_proposal(row: Dict[str, Any]) -> bool:
    return (row.get("source") or "") in _PROPOSAL_SOURCES


def _passes_policy(
    row: Dict[str, Any],
    policy: RecallPolicy,
    *,
    calling_session_id: str,
    now_iso: str,
) -> bool:
    """Strict predicate: identity allowlists, lifecycle, shared scope."""

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
        if not policy.include_shared and not policy.include_legacy_shared:
            return False
        if policy.include_legacy_shared and not policy.include_shared:
            at = row.get("author_type")
            if at is not None and at != "legacy":
                return False
    else:
        allowed_sessions = (
            None if policy.session_ids is None else set(policy.session_ids)
        )
        if allowed_sessions is None:
            if row_session is not None and row_session != calling_session_id:
                return False
        else:
            if row_session not in allowed_sessions:
                return False

    # --- Identity allowlists ---
    # producer → author_type; actor → author_id.
    if policy.producer_ids is not None:
        if (row.get("author_type") or "") not in set(policy.producer_ids):
            return False
    if policy.producer_types is not None:
        if (row.get("author_type") or "") not in set(policy.producer_types):
            return False
    if policy.actor_ids is not None:
        if row.get("author_id") not in set(policy.actor_ids):
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
    if policy.from_date and ts < f"{policy.from_date}T00:00:00":
        return False
    if policy.to_date and ts > f"{policy.to_date}T23:59:59":
        return False
    return True


def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate ids, keeping the **highest-scored** occurrence."""
    best: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for r in rows:
        rid = r.get("id")
        if rid not in best:
            best[rid] = r
            order.append(rid)
        elif r.get("score", 0.0) > best[rid].get("score", 0.0):
            best[rid] = r
    return [best[rid] for rid in order]


def _rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable sort by score descending (ties keep insertion order)."""
    return sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)


def _apply_token_budget(
    rows: List[Dict[str, Any]],
    policy: RecallPolicy,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """Apply hard top_k then rendered-token budget.

    Both ``max_tokens`` and ``max_item_tokens`` are **estimated-token**
    hard limits (via :func:`estimate_tokens`), not whitespace-word counts.
    An oversized first candidate that exceeds ``max_tokens`` is skipped —
    never returned above budget.
    """
    rows = rows[: policy.top_k]
    lines: List[str] = []
    kept: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)  # never mutate caller's dict
        r.pop("_bounded_fts_match", None)
        r.pop("_bounded_memoria_source", None)
        # Per-item token truncation on the content field.
        if policy.max_item_tokens is not None:
            r["content"] = _truncate_to_tokens(
                r.get("content") or "", policy.max_item_tokens
            )
        line = _render_row(r)
        # Drop candidates whose content is empty after truncation.
        if not (r.get("content") or "").strip():
            continue
        if policy.max_tokens is not None:
            line_tokens = _estimate_tokens(line)
            projected = _estimate_tokens("\n".join(lines + [line])) if lines else line_tokens
            if projected > policy.max_tokens:
                continue  # skip this row; hard budget
        lines.append(line)
        kept.append(r)
    context = "\n".join(lines)
    tokens = _estimate_tokens(context)
    # Final safety clamp: never return above budget.
    if policy.max_tokens is not None and tokens > policy.max_tokens:
        context = _truncate_to_tokens(context, policy.max_tokens)
        tokens = _estimate_tokens(context)
    return kept, context, tokens


def _render_row(row: Dict[str, Any]) -> str:
    """Render one candidate as a single context line."""
    content = (row.get("content") or "").strip()
    ts = (row.get("timestamp") or "")[:10] or "?"
    return f"- {content} ({ts})"


# ---------------------------------------------------------------------------
# Candidate hydration (read-only; no recall_count mutation)
# ---------------------------------------------------------------------------


def _build_where(
    beam, policy: RecallPolicy, now_iso: str, *, table_prefix: str = "",
) -> Tuple[str, List[Any]]:
    """Build native identity/session/lifecycle SQL from the policy.

    ``table_prefix`` (e.g. ``"wm."``) is applied to every column so the
    same predicate can be pushed into a JOIN'd vector-search query
    without duplicating the scope/identity/lifecycle logic.
    """
    p = table_prefix
    where_parts: List[str] = []
    params: List[Any] = []
    if policy.only_active:
        # Fail-closed lifecycle pre-filter. When only_active=False the
        # gate (_passes_policy) still authoritatively decides, so we do
        # NOT pre-filter here — expired/superseded rows must be able to
        # reach the gate subject to the remaining scope/policy filters.
        where_parts.append(f"({p}valid_until IS NULL OR {p}valid_until > ?)")
        params.append(now_iso)
        where_parts.append(f"{p}superseded_by IS NULL")
    if policy.include_shared or policy.include_legacy_shared:
        where_parts.append("(1=1)")
    else:
        where_parts.append(f"({p}session_id = ? OR {p}scope = 'global')")
        params.append(beam.session_id)
    # I-6: empty allowlists fail closed (match nothing) via a
    # guaranteed-false predicate; non-empty use parameterized IN (...).
    for _field, _col in (
        ("session_ids", "session_id"),
        ("actor_ids", "author_id"),
        ("producer_ids", "author_type"),
        ("producer_types", "author_type"),
        ("project_ids", "channel_id"),
        ("memory_types", "memory_type"),
        ("veracity", "veracity"),
    ):
        _vals = getattr(policy, _field)
        if _vals is None:
            continue
        if len(_vals) == 0:
            # Empty allowlist = allow nothing. 1=0 is always false and
            # avoids invalid `IN ()` SQL. _passes_policy also enforces
            # this in Python as a defense-in-depth check.
            where_parts.append("(1=0)")
        else:
            ph = ",".join("?" * len(_vals))
            where_parts.append(f"{p}{_col} IN ({ph})")
            params.extend(_vals)
    if policy.source is not None:
        where_parts.append(f"{p}source = ?")
        params.append(policy.source)
    if policy.from_date:
        where_parts.append(f"{p}timestamp >= ?")
        params.append(f"{policy.from_date}T00:00:00")
    if policy.to_date:
        where_parts.append(f"{p}timestamp <= ?")
        params.append(f"{policy.to_date}T23:59:59")
    where_parts.append(f"{p}source NOT IN ('sleep_model_refresh_proposal')")
    return " AND ".join(where_parts), params


_WM_COLS = (
    "id, content, source, timestamp, session_id, importance, "
    "recall_count, last_recalled, valid_until, superseded_by, scope, "
    "author_id, author_type, channel_id, veracity, memory_type, "
    "metadata_json"
)


def _hydrate_candidates(
    beam,
    query: str,
    policy: RecallPolicy,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Hydrate candidates from all read-only retrieval paths.

    Covers: vector, FTS, entity, fact, MEMORIA, episodic supplements.
    Degradation is deterministic: vector → FTS → bounded recent fallback.
    """
    from mnemosyne.core import beam as _beam_mod

    degradation: List[str] = []
    conn = beam.conn
    now_iso = _now_iso()
    query_lower = query.lower()
    query_words = _beam_mod._recall_tokens(query_lower)
    where_sql, params = _build_where(beam, policy, now_iso)

    candidates: Dict[Any, Dict[str, Any]] = {}
    had_vector = False
    had_fts = False

    # --- Vector path (working + episodic) ---
    embeddings_available = _beam_mod._embeddings.available()
    query_embedding = None
    if embeddings_available:
        try:
            query_embedding = _beam_mod._embeddings.embed_query(query)
        except Exception:
            degradation.append("query_embedding_failed")
            logger.info("bounded: query embedding failed")
            query_embedding = None

    if query_embedding is not None:
        try:
            # Build the working-vector where clause from the same policy
            # that drives every other path — scope/identity/lifecycle
            # filters are identical, just prefixed with ``wm.`` for the
            # JOIN'd vector query. When only_active=False the lifecycle
            # clauses are omitted so expired/superseded rows can reach
            # the gate (_passes_policy), matching the FTS/fallback paths.
            wm_where, wm_params = _build_where(
                beam, policy, now_iso, table_prefix="wm.",
            )
            wm_vec = _beam_mod._wm_vec_search(
                conn, query_embedding, k=max(policy.top_k * 3, 50),
                where_sql=wm_where,
                where_params=tuple(wm_params),
            )
            for vr in wm_vec:
                had_vector = True
                candidates.setdefault(vr["id"], {"id": vr["id"], "_vec_sim": vr["sim"]})
        except Exception:
            degradation.append("vec_working_failed")
            logger.info("bounded: wm vec search failed")
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
                max_distance = max(vr["distance"] for vr in vec_rows) or 1.0
                for vr in vec_rows:
                    had_vector = True
                    sim = max(0.0, 1.0 - (vr["distance"] / max_distance))
                    candidates.setdefault(
                        ("__rowid__", vr["rowid"]),
                        {"id": None, "_rowid": vr["rowid"], "_vec_sim": sim},
                    )
        except Exception:
            degradation.append("vec_episodic_failed")
            logger.info("bounded: episodic vec search failed")

    # --- FTS path (working + episodic) ---
    try:
        wm_fts = _beam_mod._fts_search_working(conn, query, k=max(policy.top_k * 3, 50))
    except Exception:
        # I-3: structured degradation signal + content-free log. Never
        # echo the raw query or exception detail into log messages.
        wm_fts = []
        degradation.append("fts_working_failed")
        logger.info("bounded: working fts search failed")
    for fr in wm_fts:
        had_fts = True
        candidates.setdefault(fr["id"], {"id": fr["id"], "_fts_rank": fr["rank"]})
    try:
        em_fts = _beam_mod._fts_search(conn, query, k=max(policy.top_k * 3, 20))
    except Exception:
        # I-3: structured degradation signal + content-free log.
        em_fts = []
        degradation.append("fts_episodic_failed")
        logger.info("bounded: episodic fts search failed")
    for fr in em_fts:
        had_fts = True
        candidates.setdefault(
            ("__rowid__", fr["rowid"]),
            {"id": None, "_rowid": fr["rowid"], "_fts_rank": fr["rank"]},
        )

    # --- Entity supplement ---
    try:
        entity_ids = _beam_mod._find_memories_by_entity(beam, query)
        for eid in entity_ids:
            candidates.setdefault(eid, {"id": eid, "_entity_match": True})
    except Exception:
        degradation.append("entity_lookup_failed")
        logger.info("bounded: entity lookup failed")

    # --- Fact supplement ---
    try:
        fact_ids = _beam_mod._find_memories_by_fact(beam, query)
        for fid in fact_ids:
            candidates.setdefault(fid, {"id": fid, "_fact_match": True})
    except Exception:
        degradation.append("fact_lookup_failed")
        logger.info("bounded: fact lookup failed")

    # --- MEMORIA supplement ---
    try:
        memoria = beam.memoria_retrieve(query, top_k=max(policy.top_k, 3))
        if memoria and memoria.get("source") != "fallback":
            ctx = memoria.get("context", "")
            source_memory_ids = [
                sid for sid in (memoria.get("source_memory_ids") or []) if sid
            ]
            # Determine safe scope/session from cited source rows so the
            # MEMORIA synthetic candidate inherits real provenance rather
            # than being hard-coded global (which default policy rejects).
            memoria_scope = "session"
            memoria_session = beam.session_id
            if source_memory_ids:
                ph = ",".join("?" * len(source_memory_ids))
                src_rows = conn.execute(
                    f"SELECT session_id, scope FROM working_memory WHERE id IN ({ph})",
                    tuple(source_memory_ids),
                ).fetchall()
                if src_rows:
                    # Inherit from the first cited source row.
                    memoria_session = src_rows[0]["session_id"] or beam.session_id
                    memoria_scope = src_rows[0]["scope"] or "session"
            if ctx:
                mid = f"memoria_{memoria.get('source', 'unknown')}"
                candidates.setdefault(mid, {
                    "id": mid,
                    "_memoria": True,
                    "_content_override": f"[MEMORIA {memoria.get('source', '')}] {ctx}",
                    "_memoria_scope": memoria_scope,
                    "_memoria_session": memoria_session,
                })
            # Also pull source memory ids that MEMORIA references.
            for sid in source_memory_ids:
                candidates.setdefault(sid, {"id": sid, "_memoria_source": True})
    except Exception:
        # I-3: structured degradation signal + content-free log. Never
        # echo the raw query or memory content.
        degradation.append("memoria_failed")
        logger.info("bounded: memoria lookup failed")

    # --- Resolve candidate ids → full rows ---
    # First pass: resolve string ids against working_memory.
    wm_ids_to_fetch = [v["id"] for v in candidates.values() if v.get("id") and not v.get("_rowid")]
    em_rowids_to_fetch = [v["_rowid"] for v in candidates.values() if v.get("_rowid")]

    resolved: Dict[Any, Dict[str, Any]] = {}
    resolved_wm_ids: set = set()

    if wm_ids_to_fetch:
        ph = ",".join("?" * len(wm_ids_to_fetch))
        rows = conn.execute(
            f"SELECT {_WM_COLS} FROM working_memory WHERE id IN ({ph}) AND {where_sql}",
            (*wm_ids_to_fetch, *params),
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["_tier"] = "working"
            resolved[d["id"]] = d
            resolved_wm_ids.add(d["id"])

    # Second pass: entity/fact/memoria-source IDs that were NOT found in
    # working_memory may exist in episodic_memory. Resolve them by id.
    unresolved_em_ids = [
        mid for mid in wm_ids_to_fetch if mid not in resolved_wm_ids
    ]
    if unresolved_em_ids:
        ph = ",".join("?" * len(unresolved_em_ids))
        rows = conn.execute(
            f"SELECT {_WM_COLS} FROM episodic_memory WHERE id IN ({ph}) AND {where_sql}",
            (*unresolved_em_ids, *params),
        ).fetchall()
        for row in rows:
            d = dict(row)
            d["_tier"] = "episodic"
            resolved[d["id"]] = d

    if em_rowids_to_fetch:
        ph = ",".join("?" * len(em_rowids_to_fetch))
        rows = conn.execute(
            f"SELECT rowid, {_WM_COLS} FROM episodic_memory WHERE rowid IN ({ph}) AND {where_sql}",
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
        # MEMORIA synthetic rows have no DB row; materialize from override.
        if cand.get("_memoria"):
            content = cand.get("_content_override", "")
            lexical = _beam_mod._lexical_relevance(query_words, content, query_lower)
            row = {
                "id": cand["id"],
                "content": content,
                "source": "memoria",
                "timestamp": "",
                "session_id": cand.get("_memoria_session", beam.session_id),
                "scope": cand.get("_memoria_scope", "session"),
                "importance": 0.5,
                "score": round(min(0.6, lexical * 0.6), 4),
                "_tier": "memoria",
            }
            scored.append(row)
            continue

        row = resolved.get(key)
        if row is None:
            continue
        vec_sim = cand.get("_vec_sim", 0.0)
        fts_rank = cand.get("_fts_rank")
        is_entity = cand.get("_entity_match", False)
        is_fact = cand.get("_fact_match", False)
        is_memoria_src = cand.get("_memoria_source", False)
        lexical = _beam_mod._lexical_relevance(query_words, row.get("content", ""), query_lower)
        decay = _beam_mod._recency_decay(row.get("timestamp", ""))
        importance = row.get("importance") or 0.5

        # Base hybrid score.
        score = max(
            vec_sim * 0.5
            + (lexical * 0.3 if fts_rank is not None else 0.0)
            + importance * 0.2,
            lexical * 0.8,
        ) * (0.7 + 0.3 * decay)

        # Entity/fact/MEMORIA-source boosts (mirror legacy recall bonuses).
        if is_entity:
            score = min(score * 1.3, 1.0)
        if is_fact:
            score = min(score * 1.2, 1.0)
        if is_memoria_src:
            score = min(score * 1.1, 1.0)

        # Relevance gate for non-vec, non-FTS rows.
        if fts_rank is None and vec_sim == 0.0 and not is_entity and not is_fact and not is_memoria_src:
            if lexical < min_relevance:
                continue

        row["score"] = round(score, 4)
        row["dense_score"] = round(vec_sim, 4)
        row["fts_score"] = round(
            (1.0 - min(1.0, abs(fts_rank))) if fts_rank is not None else 0.0, 4
        )
        if is_entity:
            row["entity_match"] = True
        if is_fact:
            row["fact_match"] = True
        row["_bounded_fts_match"] = fts_rank is not None
        row["_bounded_memoria_source"] = is_memoria_src
        scored.append(row)

    # --- Associative supplement (graph traversal, depth=1) ---
    # Resolve each related memory_id to its REAL working/episodic row so
    # strict producer/actor/project/session/lifecycle filtering applies.
    # Attach relationship metadata without overwriting the real ``source``
    # field. Read-only: no recall_count mutation.
    if beam.episodic_graph is not None and scored:
        try:
            existing_ids = {r["id"] for r in scored}
            assoc_added: Dict[str, Dict[str, Any]] = {}
            # Traverse from the top-scored seeds (legacy uses top 5).
            for seed in sorted(scored, key=lambda r: r.get("score", 0.0), reverse=True)[:5]:
                related = beam.episodic_graph.find_related_memories(
                    seed["id"], depth=1,
                )
                for rel in related:
                    mid = rel["memory_id"]
                    if mid in existing_ids or mid in assoc_added:
                        continue
                    # Resolve the real row — try episodic then working.
                    cursor = beam.conn.cursor()
                    real_row = beam._fetch_polyphonic_row(cursor, mid)
                    if real_row is None:
                        continue
                    # Attach relationship metadata without overwriting source.
                    real_row["score"] = round(
                        min(rel.get("weight", 0.3) * 0.8, 1.0), 4
                    )
                    real_row["associative"] = True
                    real_row["connecting_edge"] = rel.get("edge_type", "related")
                    real_row["assoc_depth"] = rel.get("depth", 1)
                    assoc_added[mid] = real_row
            scored.extend(assoc_added.values())
        except Exception:
            logger.info("bounded: associative hydration failed")

    return scored, mode, degradation


def _enhanced_query_and_weights(query: str) -> Tuple[str, Tuple[float, float, float]]:
    """Resolve the existing pure enhanced query/rank helpers."""
    from mnemosyne.core import beam as _beam_mod

    expanded_query = (
        _beam_mod.expand_query(query)
        if _beam_mod.expand_query is not None
        else query
    )
    weights = _beam_mod._resolve_recall_weights(None, None, None)
    if _beam_mod.classify_intent is not None and _beam_mod.adjust_weights is not None:
        intent = _beam_mod.classify_intent(query)
        if intent.category != "general":
            weights = _beam_mod._normalize_recall_weight_values(
                *_beam_mod.adjust_weights(
                    base_vec=weights.vec,
                    base_fts=weights.fts,
                    base_importance=weights.importance,
                    intent=intent,
                )
            )
    return expanded_query, weights.as_tuple()


def _rank_enhanced(
    rows: List[Dict[str, Any]],
    query: str,
    weights: Tuple[float, float, float],
) -> List[Dict[str, Any]]:
    """Apply intent weights, Weibull scoring, and MMR after filtering."""
    from mnemosyne.core import beam as _beam_mod

    query_lower = query.lower()
    query_words = _beam_mod._recall_tokens(query_lower)
    vec_weight, fts_weight, importance_weight = weights
    for row in rows:
        fts_match = row.pop("_bounded_fts_match", None)
        is_memoria_source = row.pop("_bounded_memoria_source", False)
        if fts_match is None:
            continue
        lexical = _beam_mod._lexical_relevance(
            query_words,
            row.get("content", ""),
            query_lower,
        )
        decay = _beam_mod._recency_decay(row.get("timestamp", ""))
        score = max(
            row.get("dense_score", 0.0) * vec_weight
            + (lexical * fts_weight if fts_match else 0.0)
            + (row.get("importance") or 0.5) * importance_weight,
            lexical * 0.8,
        ) * (0.7 + 0.3 * decay)
        if row.get("entity_match"):
            score = min(score * 1.3, 1.0)
        if row.get("fact_match"):
            score = min(score * 1.2, 1.0)
        if is_memoria_source:
            score = min(score * 1.1, 1.0)
        row["score"] = round(score, 4)

    if _beam_mod.weibull_boost is not None:
        now = datetime.now(timezone.utc)
        for row in rows:
            memory_type = row.get("memory_type") or "general"
            if memory_type == "unknown":
                memory_type = "general"
            boost = _beam_mod.weibull_boost(
                row.get("timestamp"),
                now,
                memory_type=memory_type,
            )
            row["score"] = round(row.get("score", 0.0) * 0.7 + boost * 0.3, 4)
            row["weibull_boost"] = round(boost, 4)
            row["memory_type"] = memory_type

    if _beam_mod.mmr_rerank is not None and len(rows) > 1:
        return _beam_mod.mmr_rerank(rows, lambda_param=0.7, top_k=len(rows))
    return _rank(rows)


def _recent_fallback_rows(
    beam, policy: RecallPolicy, where_sql: str, params: List[Any],
) -> List[Dict[str, Any]]:
    """Bounded recent fallback: pull recent active rows from working_memory."""

    conn = beam.conn

    fallback_rows = conn.execute(
        f"SELECT {_WM_COLS} FROM working_memory WHERE {where_sql} "
        f"ORDER BY timestamp DESC LIMIT {min(int(policy.top_k * 3), 200)}",
        tuple(params),
    ).fetchall()
    out = []
    for row in fallback_rows:
        d = dict(row)
        d["_tier"] = "working"
        d["score"] = round(d.get("importance", 0.5) * 0.3, 4)
        d["dense_score"] = 0.0
        d["fts_score"] = 0.0
        out.append(d)
    return out


def _hydrate_polyphonic(beam, query: str, policy: RecallPolicy):
    """Hydrate candidates from the polyphonic engine, read-only."""
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
        poly_results = engine.recall(
            query=query, query_embedding=query_embedding, top_k=policy.top_k * 2,
        )
    except Exception:
        logger.info("bounded: polyphonic engine failed")
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


def _applied_filters(policy: RecallPolicy) -> Dict[str, Any]:
    """Non-sensitive diagnostics summary of what the policy enforced."""
    return {
        "top_k": policy.top_k,
        "max_tokens": policy.max_tokens,
        "max_item_tokens": policy.max_item_tokens,
        "include_shared": policy.include_shared,
        "include_legacy_shared": policy.include_legacy_shared,
        "only_active": policy.only_active,
        "has_actor_filter": policy.actor_ids is not None,
        "has_producer_filter": policy.producer_ids is not None,
        "has_project_filter": policy.project_ids is not None,
        "has_session_filter": policy.session_ids is not None,
        "has_producer_type_filter": policy.producer_types is not None,
        "has_memory_type_filter": policy.memory_types is not None,
        "has_veracity_filter": policy.veracity is not None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _public_metadata_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal metadata_json and expose parsed ``metadata: dict``.

    Runs after policy/fallback selection and before dedupe/rank/token
    rendering. Rows that already carry a parsed ``metadata`` dict
    (polyphonic hydration via beam._fetch_polyphonic_row) keep it;
    MEMORIA/fact/associative synthetic rows receive ``{}``. The raw
    ``metadata_json`` storage field is never surfaced.
    """
    from mnemosyne.core import beam as _beam_mod

    public = dict(row)
    if not isinstance(public.get("metadata"), dict):
        raw = public.pop("metadata_json", None)
        public["metadata"] = _beam_mod._parse_recall_metadata(raw)
    else:
        public.pop("metadata_json", None)
    return public


def _run_gate(
    rows: List[Dict[str, Any]],
    beam,
    query: str,
    policy: RecallPolicy,
    mode: str,
    degradation: List[str],
    *,
    used_polyphonic: bool = False,
    enhanced_weights: Optional[Tuple[float, float, float]] = None,
) -> RecallEnvelope:
    """One authoritative gate for every mode.

    If all rows are removed by the strict predicate and fallback is
    required, pull deterministic bounded recent rows before applying the
    gate's final stages.
    """
    now_iso = _now_iso()
    gated = [
        r for r in rows
        if _passes_policy(r, policy, calling_session_id=beam.session_id, now_iso=now_iso)
    ]

    # --- Post-filter fallback: if everything was removed. ---
    if not gated and policy.require_fallback:
        if used_polyphonic:
            # Polyphonic all-filtered: fall back to bounded LINEAR hydration
            # first (not recent fallback), with truthful mode reporting.
            degradation.append("polyphonic_empty_fallback_linear")
            linear_rows, lin_mode, lin_deg = _hydrate_candidates(beam, query, policy)
            degradation.extend(lin_deg)
            gated = [
                r for r in linear_rows
                if _passes_policy(r, policy, calling_session_id=beam.session_id, now_iso=now_iso)
            ]
            if gated:
                mode = lin_mode
        if not gated:
            # Last resort: deterministic bounded recent fallback.
            where_sql, params = _build_where(beam, policy, now_iso)
            fb_rows = _recent_fallback_rows(beam, policy, where_sql, params)
            gated = [
                r for r in fb_rows
                if _passes_policy(r, policy, calling_session_id=beam.session_id, now_iso=now_iso)
            ]
            if gated:
                mode = "recent_fallback"
                degradation.append("recent_fallback_after_filter")

    # Normalize every public row: strip internal metadata_json and
    # expose parsed ``metadata: dict``. Synthetic rows (MEMORIA, fact,
    # associative) receive ``{}``; polyphonic rows keep their already-
    # parsed dict. Runs after policy/fallback, before dedupe/rank/token
    # rendering, so metadata never affects ranking or the token budget.
    gated = [_public_metadata_row(r) for r in gated]

    gated = _dedupe(gated)
    gated = (
        _rank_enhanced(gated, query, enhanced_weights)
        if enhanced_weights is not None
        else _rank(gated)
    )
    kept, context, tokens = _apply_token_budget(gated, policy)

    return RecallEnvelope(
        results=kept,
        rendered_context=context,
        token_count=tokens,
        retrieval_mode=mode,
        applied_filters=_applied_filters(policy),
        degradation_reasons=degradation,
    )


def recall_bounded(beam, query: str, policy: Optional[RecallPolicy] = None) -> RecallEnvelope:
    """Module-level adapter — delegates to ``BeamMemory.recall_bounded``.

    Kept for backward compatibility with callers that imported the free
    function. The canonical public API is
    :meth:`BeamMemory.recall_bounded`.
    """
    if policy is None:
        policy = RecallPolicy()
    return beam.recall_bounded(query, policy)


# ---------------------------------------------------------------------------
# BeamMemory method (installed via beam.py import)
# ---------------------------------------------------------------------------


def _beam_recall_bounded(self, query: str, policy: Optional[RecallPolicy] = None) -> RecallEnvelope:
    """Canonical native bounded recall on :class:`BeamMemory`.

    Read-only: never mutates ``recall_count`` / ``last_recalled``.
    """
    if policy is None:
        policy = RecallPolicy()

    degradation: List[str] = []

    # --- Polyphonic path (gated by env) ---
    if os.environ.get("MNEMOSYNE_POLYPHONIC_RECALL", "0") == "1":
        poly_rows, poly_mode, poly_deg = _hydrate_polyphonic(self, query, policy)
        degradation.extend(poly_deg)
        if poly_rows:
            return _run_gate(
                poly_rows, self, query, policy,
                mode="hybrid", degradation=degradation,
                used_polyphonic=True,
            )
        # Empty / failed polyphonic → bounded linear fallback.
        degradation.append("polyphonic_empty_fallback_linear")

    if os.environ.get("MNEMOSYNE_ENHANCED_RECALL", "0") == "1":
        expanded_query, score_weights = _enhanced_query_and_weights(query)
        enhanced_rows, mode, enhanced_deg = _hydrate_candidates(
            self,
            expanded_query,
            policy,
        )
        degradation.extend(enhanced_deg)
        return _run_gate(
            enhanced_rows,
            self,
            expanded_query,
            policy,
            mode=mode,
            degradation=degradation,
            used_polyphonic=False,
            enhanced_weights=score_weights,
        )

    # --- Linear + supplements path ---
    linear_rows, mode, lin_deg = _hydrate_candidates(self, query, policy)
    degradation.extend(lin_deg)
    return _run_gate(
        linear_rows, self, query, policy,
        mode=mode, degradation=degradation,
        used_polyphonic=False,
    )
