"""
Mnemosyne MCP Server — Model Context Protocol for cross-agent sharing.

This module provides MCP tool definitions and handlers for Mnemosyne,
enabling any MCP-compatible client (Claude Desktop, etc.) to interact
with the memory system.

Usage:
    from mnemosyne.mcp_tools import TOOLS, handle_tool_call

All imports are guarded — this module loads safely even if mcp is not installed.
"""

from typing import TYPE_CHECKING, Dict, Any, List, TypeAlias
import json
import math  # noqa: F401
import os
import sqlite3
from pathlib import Path

# Guarded import — MCP is optional
try:
    from mcp.types import Tool, TextContent, CallToolResult, ErrorData
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    Tool = None
    TextContent = None
    CallToolResult = None
    ErrorData = None

from mnemosyne.core.beam import BeamMemory, _guarded_transaction

from mnemosyne import tool_schemas
from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
from mnemosyne.batch_tool import (
    BatchValidationError,
    apply_beam_batch,
    batch_validation_error_payload,
    dry_run_batch,
    validate_batch_operations,
)

if TYPE_CHECKING:
    from mnemosyne.core.memory import Mnemosyne

    MnemosyneInstance: TypeAlias = Mnemosyne
else:
    # Runtime type-hint resolution must not import core.memory.
    MnemosyneInstance: TypeAlias = Any


def __getattr__(name: str) -> Any:
    """Lazily retain the historical public ``Mnemosyne`` export.

    Importing ``mnemosyne.core.memory`` initializes legacy default-memory
    state, so only an explicit compatibility access may trigger that import.
    """
    if name == "Mnemosyne":
        from mnemosyne.core.memory import Mnemosyne

        return Mnemosyne
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = []
for _s in ALL_TOOL_SCHEMAS:
    _t = dict(_s)
    # ``mcp`` SDK 2.x renamed the wire field on ``Tool`` from the camelCase
    # ``inputSchema`` (1.x) to the snake_case ``input_schema``. The
    # canonical Python-side key matches the model field. Pydantic still
    # accepts the camelCase alias on construction, but internal code that
    # reads ``tool["input_schema"]`` is the only fully-supported path.
    if "inputSchema" in _t:
        _t["input_schema"] = _t.pop("inputSchema")
    elif "parameters" in _t:
        _t["input_schema"] = _t.pop("parameters")
    # Apply readOnly metadata (Task 6B) at consumption time so the canonical
    # schema dicts in tool_schemas.py stay byte-equal to the provider copies
    # (test_hermes_provider_parity). Hand-set values on the schema dict win.
    if "readOnly" not in _t:
        _t["readOnly"] = _t["name"] in tool_schemas.READ_ONLY_TOOLS
    TOOLS.append(_t)

# ---------------------------------------------------------------------------
# Individual tool schemas (lazy - computed on first access)
# ---------------------------------------------------------------------------

def _get_schema(name: str) -> Dict[str, Any]:
    """Extract input_schema from TOOLS by tool name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["input_schema"]
    raise KeyError(f"Tool not found: {name}")

class _SchemaProxy:
    """Lazy proxy to access tool schemas after TOOLS is populated."""
    def __init__(self, name: str):
        self._name = name
        self._schema = None
    
    def __getattr__(self, attr):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return getattr(self._schema, attr)
    
    def __getitem__(self, key):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return self._schema[key]
    
    def __contains__(self, key):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return key in self._schema
    
    def get(self, key, default=None):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return self._schema.get(key, default)
    
    def __iter__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return iter(self._schema)
    
    def __len__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return len(self._schema)
    
    def __repr__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return repr(self._schema)

_REMEMBER_SCHEMA = _SchemaProxy("mnemosyne_remember")
_RECALL_SCHEMA = _SchemaProxy("mnemosyne_recall")
_SLEEP_SCHEMA = _SchemaProxy("mnemosyne_sleep")
_SCRATCHPAD_READ_SCHEMA = _SchemaProxy("mnemosyne_scratchpad_read")
_SCRATCHPAD_WRITE_SCHEMA = _SchemaProxy("mnemosyne_scratchpad_write")
_GET_STATS_SCHEMA = _SchemaProxy("mnemosyne_stats")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
_MNEMOSYNE_HOME = os.environ.get("MNEMOSYNE_HOME", str(Path(_HERMES_HOME) / "mnemosyne"))


def _shared_db_path() -> Path:
    """Return the shared surface DB path."""
    return Path(os.environ.get("MNEMOSYNE_SHARED_DB_PATH", str(Path(_MNEMOSYNE_HOME) / "data" / "shared" / "mnemosyne.db")))


def _create_instance(session_id: str = None, author_id: str = None,
                     author_type: str = None, channel_id: str = None,
                     bank: str = "default") -> MnemosyneInstance:
    """Create a fresh Mnemosyne instance for each MCP connection.

    Identity is resolved from:
    1. Explicit args (from tool call or constructor)
    2. Environment variables (MNEMOSYNE_AUTHOR_ID, etc.)
    3. None (backward compatible, no identity tracking)

    ``MnemosyneInstance`` resolves to ``Any`` at runtime so type-hint
    resolution remains side-effect-free, while static checkers use the
    TYPE_CHECKING Mnemosyne alias above.
    """
    # Importing core.memory initializes the legacy default database, so keep it
    # on this mutation-capable construction path rather than at MCP import time.
    from mnemosyne.core.memory import Mnemosyne

    auth = author_id or os.environ.get("MNEMOSYNE_AUTHOR_ID")
    auth_type = author_type or os.environ.get("MNEMOSYNE_AUTHOR_TYPE")
    chan = channel_id or os.environ.get("MNEMOSYNE_CHANNEL_ID") or session_id or "default"
    sess = session_id or f"mcp_{bank}"

    return Mnemosyne(
        session_id=sess,
        author_id=auth,
        author_type=auth_type,
        channel_id=chan,
        bank=bank
    )


def _create_surface_instance() -> BeamMemory:
    """Create a BeamMemory instance for the shared surface DB."""
    shared_path = _shared_db_path()
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    return BeamMemory(session_id="mcp_shared_surface", db_path=shared_path)


def _resolve_bank(arguments: Dict[str, Any]) -> str:
    """Resolve per-call bank, falling back to MCP server default bank."""
    return arguments.get("bank") or os.environ.get("MNEMOSYNE_MCP_BANK") or "default"


def _resolve_default_scope() -> str:
    """Resolve default scope for remember() calls.
    
    Precedence: MNEMOSYNE_DEFAULT_SCOPE env var, falling back to 'session'.
    Only 'session' and 'global' are accepted; unrecognized values fall through
    to the hardcoded default."""
    raw = os.environ.get("MNEMOSYNE_DEFAULT_SCOPE", "").strip().lower()
    if raw in ("session", "global"):
        return raw
    return "session"


def _serialize(obj):
    """Recursively convert non-serializable objects (datetime, etc.) to strings."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

class _WrapperBatchAdapter:
    """Expose Mnemosyne wrapper mutations through the Beam batch interface."""

    def __init__(self, mem: MnemosyneInstance):
        """Adapt a Mnemosyne instance without resolving its import at runtime."""
        self._mem = mem
        self.conn = mem.beam.conn
        self.wrapper_events = []

    def remember(self, **kwargs):
        return self._call_wrapper("remember", **kwargs)

    def update_working(self, memory_id: str, *, content=None, importance=None):
        wrapper_ok = self._call_wrapper("update", memory_id, content=content, importance=importance)
        if wrapper_ok:
            return True
        return self._mem.beam.update_working(memory_id, content=content, importance=importance)

    def forget_working(self, memory_id: str):
        return self._call_wrapper("forget", memory_id)

    def invalidate(self, memory_id: str, *, replacement_id=None):
        return self._call_wrapper("invalidate", memory_id, replacement_id=replacement_id)

    def replay_wrapper_events(self) -> None:
        for event_type, memory_id, kwargs in self.wrapper_events:
            self._mem._emit_wrapper(event_type, memory_id, **kwargs)

    def _call_wrapper(self, method_name: str, *args, **kwargs):
        original_conn = self._mem.conn
        original_emit = self._mem._emit_wrapper
        self._mem.conn = self.conn
        self._mem._emit_wrapper = lambda event_type, memory_id, **event_kwargs: self.wrapper_events.append(
            (event_type, memory_id, event_kwargs)
        )
        try:
            return getattr(self._mem, method_name)(*args, **kwargs)
        finally:
            self._mem.conn = original_conn
            self._mem._emit_wrapper = original_emit


def _handle_remember(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_remember tool call."""
    content = arguments["content"]
    source = arguments.get("source", "mcp")
    importance = arguments.get("importance", 0.5)
    metadata = arguments.get("metadata", {})
    extract_entities = arguments.get("extract_entities", False)
    extract = arguments.get("extract", False)
    scope = arguments.get("scope", _resolve_default_scope())
    valid_until = arguments.get("valid_until") or None
    veracity = arguments.get("veracity", "unknown")
    bank = _resolve_bank(arguments)

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    memory_id = mem.remember(
        content=content,
        source=source,
        importance=importance,
        metadata=metadata,
        extract_entities=extract_entities,
        extract=extract,
        scope=scope,
        valid_until=valid_until,
        veracity=veracity,
    )

    return {
        "status": "stored",
        "memory_id": memory_id,
        "content_preview": content[:100],
        "bank": bank
    }


def _handle_batch(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_batch tool call."""
    try:
        normalized = validate_batch_operations(arguments.get("operations"))
    except BatchValidationError as exc:
        return batch_validation_error_payload(exc)

    if bool(arguments.get("dry_run", False)):
        return dry_run_batch(normalized)

    bank = _resolve_bank(arguments)
    mem = _create_instance(
        author_id=arguments.get("author_id"),
        author_type=arguments.get("author_type"),
        channel_id=arguments.get("channel_id"),
        bank=bank,
    )
    audit_events = []
    adapter = _WrapperBatchAdapter(mem)
    result = apply_beam_batch(
        adapter,
        normalized,
        default_scope=_resolve_default_scope(),
        remember_source_default="mcp",
        audit_event=lambda name, **kwargs: audit_events.append({"event": name, **kwargs}),
    )
    if result.get("status") == "ok":
        adapter.replay_wrapper_events()
    result["bank"] = bank
    if audit_events:
        result["audit_events"] = audit_events
    return result


def _handle_recall(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_recall tool call.

    Without bounded options: legacy plain-list path (unchanged). With any
    bounded option (producer/actor/project/session/max_tokens/max_item_tokens/
    top_k_bounded/include_shared): returns a RecallEnvelope via recall_bounded.
    Mirrors mnemosyne.cli.cmd_recall's two-path dispatch.
    """
    query = arguments["query"]
    if any(arguments.get(k) is not None for k in _BOUNDED_RECALL_KEYS):
        return _recall_bounded_envelope(arguments, query, _resolve_bank(arguments))
    top_k = int(arguments.get("limit", arguments.get("top_k", 5)))
    bank = _resolve_bank(arguments)
    temporal_weight = arguments.get("temporal_weight", 0.0)
    query_time = arguments.get("query_time")
    # "" (and whitespace) means "omitted" for callers built against the older
    # schema (#555). Deliberately narrower than `or None`, which both misses
    # whitespace-only strings (truthy) and swallows falsey non-strings such as
    # 0/False/[] that must still reach _parse_query_time's TypeError.
    if isinstance(query_time, str) and not query_time.strip():
        query_time = None
    temporal_halflife = arguments.get("temporal_halflife", 24)
    vec_weight = arguments.get("vec_weight")
    fts_weight = arguments.get("fts_weight")
    importance_weight = arguments.get("importance_weight")
    explain = bool(arguments.get("explain", False))

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    recall_payload = mem.recall(
        query=query,
        top_k=top_k,
        temporal_weight=temporal_weight,
        query_time=query_time,
        temporal_halflife=temporal_halflife,
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        importance_weight=importance_weight,
        explain=explain,
    )
    if explain:
        results = recall_payload.get("results", [])
        explain_payload = recall_payload.get("explain", {})
    else:
        results = recall_payload
        explain_payload = None

    serializable = []
    for r in results:
        item = dict(r) if hasattr(r, "keys") else r
        for key in ["timestamp", "created_at", "valid_until", "last_recalled"]:
            if key in item and item[key] is not None:
                if hasattr(item[key], "isoformat"):
                    item[key] = item[key].isoformat()
        serializable.append(item)

    response = {
        "status": "ok",
        "count": len(serializable),
        "results": serializable,
        "bank": bank
    }
    if explain_payload is not None:
        response.update({"query": query, "top_k": top_k, "explain": explain_payload})
    return response


# Bounded-field keys that force the RecallEnvelope path (Task 6B).
_BOUNDED_RECALL_KEYS = (
    "producer", "actor", "project", "session",
    "max_tokens", "max_item_tokens", "top_k_bounded", "include_shared",
)


def _recall_bounded_envelope(arguments: Dict[str, Any], query: str, bank: str) -> Dict[str, Any]:
    """Take the bounded recall path when any bounded control is supplied.

    Mirrors mnemosyne.cli.cmd_recall's bounded branch: builds a RecallPolicy
    from the optional fields, calls mem.recall_bounded, and projects the
    RecallEnvelope as JSON-serializable structured data. Legacy args (limit,
    temporal_*, weights) are ignored on this path, exactly as the CLI does.
    """
    from mnemosyne.core.recall_bounded import RecallPolicy

    policy_kwargs: Dict[str, Any] = {}
    top_k_bounded = arguments.get("top_k_bounded")
    if top_k_bounded is not None:
        policy_kwargs["top_k"] = int(top_k_bounded)
    max_tokens = arguments.get("max_tokens")
    if max_tokens is not None:
        policy_kwargs["max_tokens"] = int(max_tokens)
    max_item_tokens = arguments.get("max_item_tokens")
    if max_item_tokens is not None:
        policy_kwargs["max_item_tokens"] = int(max_item_tokens)
    producer = arguments.get("producer")
    if producer:
        policy_kwargs["producer_ids"] = [producer]
    actor = arguments.get("actor")
    if actor:
        policy_kwargs["actor_ids"] = [actor]
    project = arguments.get("project")
    if project:
        policy_kwargs["project_ids"] = [project]
    session = arguments.get("session")
    if session:
        policy_kwargs["session_ids"] = [session]
    policy_kwargs["include_shared"] = bool(arguments.get("include_shared", False))

    try:
        policy = RecallPolicy(**policy_kwargs)
    except (ValueError, TypeError) as exc:
        return {"error": f"invalid recall policy: {exc}"}

    mem = _create_instance(
        author_id=arguments.get("author_id"),
        author_type=arguments.get("author_type"),
        channel_id=arguments.get("channel_id"),
        bank=bank,
    )
    env = mem.recall_bounded(query, policy)
    return {
        "status": "ok",
        "results": _serialize(env.results),
        "rendered_context": env.rendered_context,
        "token_count": env.token_count,
        "retrieval_mode": env.retrieval_mode,
        "applied_filters": _serialize(env.applied_filters),
        "degradation_reasons": list(env.degradation_reasons),
        "trace_id": env.trace_id,
        "bank": bank,
    }


def _handle_shared_remember(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_remember tool call."""
    content = (arguments.get("content") or "").strip()
    if not content:
        return {"error": "content is required"}
    if content.startswith("[USER]") or content.startswith("[ASSISTANT]"):
        return {"error": "raw conversation content is not allowed in shared memory"}
    kind = (arguments.get("kind") or "meta").strip().lower()
    if kind not in {"meta", "preference", "correction", "identity"}:
        return {"error": "kind must be one of: meta, preference, correction, identity"}
    importance = max(0.0, min(float(arguments.get("importance", 0.8)), 1.0))
    metadata = arguments.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {"error": "metadata must be an object"}

    surface_beam = _create_surface_instance()
    import hashlib
    normalized = " ".join(str(content).lower().split())
    content_hash = hashlib.sha256(f"surface:v1:{normalized}".encode("utf-8")).hexdigest()[:24]
    prefixes = ("surface meta:", "surface preference:", "surface correction:", "surface identity:", "surface fact:")
    if content.lower().startswith(prefixes):
        surface_content = content
    else:
        label_map = {"meta": "Surface meta", "preference": "Surface preference",
                     "correction": "Surface correction", "identity": "Surface identity"}
        surface_content = f"{label_map.get(kind, 'Surface meta')}: {content}"
    stable_id = "sf_" + content_hash
    meta = dict(metadata)
    meta.update({"shared_memory": True, "surface_kind": kind, "write_path": "mcp_tool"})
    memory_id = surface_beam.remember(
        content=surface_content,
        source="surface_manual",
        importance=importance,
        metadata=meta,
        scope="global",
        memory_id=stable_id,
    )

    return {
        "status": "stored_shared",
        "memory_id": memory_id,
        "content_preview": surface_content[:120],
        "shared_db": str(_shared_db_path()),
        "kind": kind,
    }


def _handle_shared_recall(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_recall tool call."""
    query = arguments.get("query", "")
    if not query:
        return {"error": "query is required"}
    top_k = int(arguments.get("limit", 5))
    surface_beam = _create_surface_instance()
    results = []
    for r in surface_beam.recall(query, top_k=top_k):
        r = dict(r)
        r["shared_surface"] = True
        results.append(r)
    return {"query": query, "count": len(results), "shared_db": str(_shared_db_path()), "results": results}


def _handle_shared_forget(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_forget tool call."""
    memory_id = (arguments.get("memory_id") or "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    surface_beam = _create_surface_instance()
    ok = surface_beam.forget_working(memory_id)
    return {"status": "deleted" if ok else "not_found", "memory_id": memory_id, "shared_db": str(_shared_db_path())}


def _handle_shared_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_stats tool call."""
    surface_beam = _create_surface_instance()
    return {
        "provider": "mnemosyne_shared",
        "shared_db": str(_shared_db_path()),
        "working": _serialize(surface_beam.get_working_stats()),
        "episodic": _serialize(surface_beam.get_episodic_stats()),
    }


def _handle_sleep(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_sleep tool call."""
    dry_run = arguments.get("dry_run", False)
    force = arguments.get("force", False)
    all_sessions = arguments.get("all_sessions", False)
    bank = _resolve_bank(arguments)

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if all_sessions and hasattr(mem, "sleep_all_sessions"):
        result = mem.sleep_all_sessions(dry_run=dry_run, force=force)
    else:
        result = mem.sleep(dry_run=dry_run, force=force)

    working = _serialize(mem.beam.get_working_stats()) if hasattr(mem, "beam") else {}
    episodic = _serialize(mem.beam.get_episodic_stats()) if hasattr(mem, "beam") else {}

    return {
        "status": result.get("status", "consolidated"),
        "result": result,
        "working": working,
        "episodic": episodic,
        "bank": bank
    }


def _handle_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_stats tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    stats = mem.get_stats()
    return {"provider": "mnemosyne", "session_id": mem._session_id if hasattr(mem, "_session_id") else None, "stats": _serialize(stats)}


def _handle_invalidate(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_invalidate tool call."""
    memory_id = arguments.get("memory_id", "")
    replacement_id = arguments.get("replacement_id") or None
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if not mem.invalidate(memory_id, replacement_id=replacement_id):
        return {"status": "memory_not_found", "memory_id": memory_id}
    return {"status": "invalidated", "memory_id": memory_id}


def _handle_validate(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_validate tool call."""
    memory_id = arguments.get("memory_id", "")
    action = arguments.get("action", "")
    bank = arguments.get("bank", "private")
    validator = arguments.get("validator") or os.environ.get("MNEMOSYNE_AUTHOR_ID") or "mcp"
    new_content = arguments.get("new_content", "")
    note = arguments.get("note", "")

    if not memory_id:
        return {"error": "memory_id is required"}
    if action not in ("attest", "update", "invalidate", "delete"):
        return {"error": f"unknown action: {action}"}
    if bank not in ("private", "surface"):
        return {"error": f"unknown bank: {bank}"}
    if action == "update" and not new_content:
        return {"error": "new_content is required for action='update'"}

    if bank == "surface":
        target_beam = _create_surface_instance()
    else:
        mem = _create_instance()
        target_beam = mem.beam

    conn = target_beam.conn
    existing = conn.execute(
        "SELECT id, author_id, content FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if not existing:
        return {"error": "memory_not_found", "memory_id": memory_id, "bank": bank}

    author_id = existing[1]
    prev_content = existing[2]

    try:
        with _guarded_transaction(conn):
            if action == "delete":
                # Cascade delete child rows before removing parent.
                conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
                conn.execute("DELETE FROM annotations WHERE memory_id = ?", (memory_id,))
                # vec_working is optional (sqlite-vec may be unavailable).
                row = conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()
                if row is not None:
                    try:
                        conn.execute("DELETE FROM vec_working WHERE rowid = ?", (row["rowid"],))
                    except sqlite3.OperationalError as vec_err:
                        if "no such table" not in str(vec_err).lower():
                            raise
                conn.execute("DELETE FROM working_memory WHERE id = ?", (memory_id,))
            elif action == "update":
                conn.execute(
                    "UPDATE working_memory SET content = ?, validator = ?, "
                    "validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (new_content, validator, memory_id),
                )
            elif action == "invalidate":
                conn.execute(
                    "UPDATE working_memory SET valid_until = CURRENT_TIMESTAMP, "
                    "validator = ?, validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (validator, memory_id),
                )
            else:
                conn.execute(
                    "UPDATE working_memory SET validator = ?, "
                    "validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (validator, memory_id),
                )
            conn.execute(
                "INSERT INTO memory_validations "
                "(memory_id, validator, action, new_content, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, validator, action,
                 new_content if action == "update" else None,
                 note or None),
            )
    except Exception as exc:
        return {"error": "validation_failed", "reason": str(exc), "memory_id": memory_id}

    return {
        "status": f"validation_{action}",
        "memory_id": memory_id,
        "bank": bank,
        "validator": validator,
        "author_id": author_id,
        "previous_content": prev_content[:200] if prev_content else None,
    }


def _handle_get(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_get tool call."""
    memory_id = arguments.get("memory_id", "")
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    result = mem.get(memory_id)
    if result is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "ok", "memory": _serialize(result)}


def _handle_triple_add(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_triple_add tool call.

    Routes annotation-flavored predicates (mentions, fact, occurred_on,
    has_source) to AnnotationStore; everything else to TripleStore.
    For occurred_on, valid_from is forwarded to AnnotationStore (issue #111).
    """
    import logging
    _log = logging.getLogger("mnemosyne.mcp.triple_add")

    from mnemosyne.core.annotations import ANNOTATION_KINDS, AnnotationStore
    from mnemosyne.core.triples import TripleStore

    predicate = arguments["predicate"]

    if isinstance(predicate, str) and predicate in ANNOTATION_KINDS:
        bank = _resolve_bank(arguments)
        mem = _create_instance(bank=bank)
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = getattr(mem.beam, "annotations", None)
        if store is None:
            store = AnnotationStore(db_path=db_path, conn=mem.beam.conn)
        valid_from = arguments.get("valid_from")
        if predicate == "occurred_on" and valid_from:
            row_id = store.add(
                memory_id=arguments["subject"],
                kind=predicate,
                value=arguments["object"],
                source=arguments.get("source", "conversation"),
                confidence=arguments.get("confidence", 1.0),
                valid_from=valid_from,
            )
        else:
            if valid_from:
                _log.warning(
                    "mnemosyne_triple_add: valid_from=%r provided with "
                    "predicate=%r (not occurred_on); valid_from discarded.",
                    valid_from, predicate,
                )
            row_id = store.add(
                memory_id=arguments["subject"],
                kind=predicate,
                value=arguments["object"],
                source=arguments.get("source", "conversation"),
                confidence=arguments.get("confidence", 1.0),
            )
        return {"status": "added", "annotation_id": row_id, "store": "annotations"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
    kg = TripleStore(db_path=db_path)
    triple_id = kg.add(
        subject=arguments["subject"],
        predicate=predicate,
        object=arguments["object"],
        valid_from=arguments.get("valid_from"),
        source=arguments.get("source", "conversation"),
        confidence=arguments.get("confidence", 1.0),
    )
    return {"status": "added", "triple_id": triple_id, "store": "triples"}


def _handle_triple_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_triple_query tool call.

    Mirrors the write-side routing: annotation predicates query
    AnnotationStore; others query TripleStore.
    """
    from mnemosyne.core.annotations import ANNOTATION_KINDS, AnnotationStore
    from mnemosyne.core.triples import TripleStore

    predicate = arguments.get("predicate")
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path

    if isinstance(predicate, str) and predicate in ANNOTATION_KINDS:
        store = getattr(mem.beam, "annotations", None)
        if store is None:
            store = AnnotationStore(db_path=db_path, conn=mem.beam.conn)
        results = store.query_by_kind(
            kind=predicate,
            value=arguments.get("object"),
            memory_id=arguments.get("subject"),
        )
        return {"results_count": len(results), "results": results, "store": "annotations"}

    kg = TripleStore(db_path=db_path)
    results = kg.query(
        subject=arguments.get("subject"),
        predicate=predicate,
        object=arguments.get("object"),
        as_of=arguments.get("as_of"),
    )
    return {"results_count": len(results), "results": results, "store": "triples"}


def _canonical_owner(arguments: Dict[str, Any]) -> str:
    """Owner id for canonical ops over the MCP surface.

    The shared tool schema does not expose owner_id (so an LLM can't target
    another owner's bank); over MCP the owner is a deployment-level setting via
    MNEMOSYNE_DEFAULT_OWNER, defaulting to "default" for single-owner use."""
    return (os.environ.get("MNEMOSYNE_DEFAULT_OWNER") or "default").strip() or "default"


def _handle_remember_canonical(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_remember_canonical tool call."""
    from mnemosyne.core.canonical import CanonicalStore

    category = (arguments.get("category") or "").strip()
    name = (arguments.get("name") or "").strip()
    body = (arguments.get("body") or "").strip()
    if not category or not name:
        return {"error": "category and name are required"}
    if not body:
        return {"error": "body is required"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)

    owner_id = _canonical_owner(arguments)
    row = store.remember(
        owner_id, category, name, body,
        source=arguments.get("source", "canonical_tool"),
        confidence=arguments.get("confidence", 1.0),
    )
    status = row.pop("status", "stored")
    return {"status": status, "owner_id": owner_id, "category": category,
            "name": name, "version": row.get("version"), "store": "canonical"}


def _handle_recall_canonical(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_recall_canonical tool call."""
    from mnemosyne.core.canonical import CanonicalStore

    category = (arguments.get("category") or "").strip()
    name = (arguments.get("name") or "").strip()
    query = (arguments.get("query") or "").strip()
    include_history = bool(arguments.get("include_history", False))
    try:
        limit = int(arguments.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)
    owner_id = _canonical_owner(arguments)

    if query:
        results = store.search(owner_id, query, limit=limit)
        return {"mode": "search", "owner_id": owner_id, "query": query,
                "results_count": len(results), "results": results, "store": "canonical"}
    if category and name:
        if include_history:
            results = store.history(owner_id, category, name)
            return {"mode": "history", "owner_id": owner_id, "category": category,
                    "name": name, "results_count": len(results),
                    "results": results, "store": "canonical"}
        row = store.recall(owner_id, category, name)
        result = {"mode": "recall", "owner_id": owner_id, "category": category,
                "name": name, "found": row is not None, "result": row,
                "store": "canonical"}
        if row is None:
            # Diagnostic: check if the row exists under a different owner_id
            try:
                conn = store.conn if hasattr(store, "conn") else None
                if conn is not None:
                    cur = conn.execute(
                        "SELECT owner_id FROM canonical_facts "
                        "WHERE category=? AND name=? AND valid_until IS NULL LIMIT 1",
                        (category, name),
                    )
                    alt = cur.fetchone()
                    if alt:
                        result["hint"] = (
                            f"Row exists under owner_id '{alt[0]}' but you queried "
                            f"with '{owner_id}'. Set MNEMOSYNE_DEFAULT_OWNER={alt[0]} "
                            f"or check your profile/provider config."
                        )
            except Exception:
                pass
        return result
    results = store.list(owner_id, category=category or None)
    return {"mode": "list", "owner_id": owner_id, "category": category or None,
            "results_count": len(results), "results": results, "store": "canonical"}


def _handle_forget_canonical(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_forget_canonical tool call.

    Retires the current canonical fact for (category, name) by stamping
    valid_until; nothing is deleted. Mirrors the Hermes provider path and
    reuses CanonicalStore.forget directly.
    """
    from mnemosyne.core.canonical import CanonicalStore

    category = (arguments.get("category") or "").strip()
    name = (arguments.get("name") or "").strip()
    if not category or not name:
        return {"error": "category and name are required"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)

    owner_id = _canonical_owner(arguments)
    retired = store.forget(owner_id, category, name)
    return {"retired": retired, "owner_id": owner_id,
            "category": category, "name": name, "store": "canonical"}


def _handle_scratchpad_write(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_write tool call."""
    content = arguments.get("content", "").strip()
    if not content:
        return {"error": "Content is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    entry_id = mem.scratchpad_write(content)
    return {"status": "written", "id": entry_id}


def _handle_scratchpad_read(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_read tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    entries = mem.scratchpad_read()
    return {"entries_count": len(entries), "entries": entries}


def _handle_scratchpad_clear(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_clear tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    mem.scratchpad_clear()
    return {"status": "cleared"}


def _handle_export(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_export tool call."""
    output_path = arguments.get("output_path", "").strip()
    if not output_path:
        return {"error": "output_path is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    result = mem.export_to_file(output_path)
    return _serialize(result)


def _handle_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_update tool call."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    content = arguments.get("content")
    importance = arguments.get("importance")
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    ok = mem.update(memory_id, content=content, importance=importance)
    return {"status": "updated" if ok else "not_found", "memory_id": memory_id}


def _handle_forget(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_forget tool call."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    ok = mem.forget(memory_id)
    return {"status": "deleted" if ok else "not_found", "memory_id": memory_id}


def _handle_import(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_import tool call."""
    provider = (arguments.get("provider") or "").strip().lower()
    input_path = arguments.get("input_path", "").strip()
    dry_run = bool(arguments.get("dry_run", False))
    force = bool(arguments.get("force", False))
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)

    if provider:
        api_key = arguments.get("api_key", "").strip()
        user_id = arguments.get("user_id", "").strip() or None
        agent_id = arguments.get("agent_id", "").strip() or None
        base_url = arguments.get("base_url", "").strip() or None
        channel_id = arguments.get("channel_id")
        if not api_key:
            env_key = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_key, "")
        if not api_key:
            return {"error": f"api_key required for {provider} import. Set {provider.upper()}_API_KEY env var or pass api_key parameter."}
        from mnemosyne.core.importers import import_from_provider
        result = import_from_provider(
            provider, mem,
            api_key=api_key,
            user_id=user_id,
            agent_id=agent_id,
            base_url=base_url,
            dry_run=dry_run,
            channel_id=channel_id,
        )
        return _serialize(result.to_dict() if hasattr(result, "to_dict") else result)

    if not input_path:
        return {"error": "Either input_path (for file import) or provider (for cross-provider import) is required"}
    stats = mem.import_from_file(input_path, force=force, dry_run=dry_run)
    return {"status": "dry_run" if dry_run else "imported", "stats": stats, "dry_run": dry_run}


def _handle_diagnose(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_diagnose tool call.

    MCP diagnose is strictly read-only: it MUST NOT write a JSONL log, create
    a default database, run repair, or mutate SQLite. It calls
    ``run_diagnostics(read_only=True)`` unconditionally. A requested repair
    (``repair_vec_working=True``) through this surface returns a structured
    rejection instead of mutating. There is NO legacy fallback: if the safe
    read-only path is unavailable, the handler fails closed with a structured
    error rather than silently calling the writable ``dry_run`` path.
    """
    from mnemosyne.diagnose import run_diagnostics

    if arguments.get("repair_vec_working"):
        # Repair over MCP is never permitted. Fail-closed: return a structured
        # rejection the client can act on.
        return {
            "status": "read_only",
            "repair_rejected": True,
            "error": "repair_not_permitted_over_mcp",
            "detail": (
                "The MCP diagnose surface is read-only and never performs "
                "repair. Use the CLI (mnemosyne repair) to mutate."
            ),
        }

    # Do NOT call _create_instance() here: constructing a Mnemosyne
    # materializes a default DB, which violates the read-only contract.
    # run_diagnostics already includes any db_path it resolved internally.
    # Fail closed/structured if the read-only path is not accepted rather than
    # silently routing to the legacy writable signature.
    try:
        result = run_diagnostics(read_only=True)
    except TypeError:
        return {
            "status": "read_only",
            "error": "read_only_unavailable",
            "detail": (
                "The read-only diagnostics path is unavailable in this "
                "runtime; MCP diagnose refuses to fall back to a writable path."
            ),
        }
    return _serialize(result)


def _handle_graph_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_graph_query tool call."""
    seed_id = arguments.get("seed_memory_id", "").strip()
    if not seed_id:
        return {"error": "seed_memory_id is required"}
    depth = int(arguments.get("max_hops", 2))
    if depth < 1:
        return {"error": "max_hops must be greater than 0"}
    edge_type = arguments.get("edge_type", "") or ""
    min_weight = float(arguments.get("min_weight", 0.0))
    if not (0.0 <= min_weight <= 1.0):
        return {"error": "min_weight must be between 0.0 and 1.0"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if mem.beam.episodic_graph is None:
        return {"error": "Episodic graph not available"}
    related = mem.beam.episodic_graph.find_related_memories(
        seed_id, depth=depth, edge_type=edge_type, min_weight=min_weight
    )
    return {
        "seed_memory_id": seed_id,
        "max_hops": depth,
        "edge_type": edge_type or "all",
        "min_weight": min_weight,
        "count": len(related),
        "results": related,
    }


def _handle_graph_link(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_graph_link tool call."""
    source_id = arguments.get("source_id", "").strip()
    target_id = arguments.get("target_id", "").strip()
    relationship = arguments.get("relationship", "").strip()
    weight = float(arguments.get("weight", 0.5))
    if not (0.0 <= weight <= 1.0):
        return {"error": "weight must be between 0.0 and 1.0"}
    if not all([source_id, target_id, relationship]):
        return {"error": "source_id, target_id, and relationship are required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if mem.beam.episodic_graph is None:
        return {"error": "Episodic graph not available"}
    from mnemosyne.core.episodic_graph import GraphEdge
    from datetime import datetime
    edge = GraphEdge(
        source=source_id,
        target=target_id,
        edge_type=relationship,
        weight=weight,
        timestamp=datetime.now().isoformat(),
    )
    mem.beam.episodic_graph.add_edge(edge)
    return {"status": "linked", "source": source_id, "target": target_id, "relationship": relationship}


# ---------------------------------------------------------------------------
# Hygiene handlers (issue #428)
# ---------------------------------------------------------------------------

def _handle_hygiene_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_hygiene_audit tool call."""
    from mnemosyne.core.banks import get_bank_db_path_read_only
    from mnemosyne.core.hygiene import audit_noise
    from mnemosyne.doctor import open_readonly_doctor_db

    bank = _resolve_bank(arguments)
    db_path = get_bank_db_path_read_only(bank)

    limit = arguments.get("limit", 200)
    min_score = arguments.get("min_score", 0.3)
    tables = arguments.get("tables") or None
    offset = arguments.get("offset", 0)
    scan_all = arguments.get("scan_all", False)
    batch_size = arguments.get("batch_size", 1000)

    conn = open_readonly_doctor_db(db_path)
    try:
        report = audit_noise(
            db_path=db_path,
            limit=limit,
            tables=tables,
            min_score=min_score,
            offset=offset,
            scan_all=scan_all,
            batch_size=batch_size,
            conn=conn,
        )
    finally:
        conn.close()
    return {
        "status": "audited",
        "report": report.to_dict(),
        "bank": bank,
    }


def _handle_hygiene_clean(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_hygiene_clean tool call."""
    from mnemosyne.core.hygiene import NoiseCandidate, clean_noise, validate_hygiene_candidate

    candidates_json = arguments.get("candidates_json", "[]")
    try:
        raw_candidates = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
    except json.JSONDecodeError:
        return {"error": "candidates_json is not valid JSON"}

    if not isinstance(raw_candidates, list):
        return {"error": "candidates_json must be a list of valid hygiene candidates"}

    candidates = []
    for candidate_data in raw_candidates:
        try:
            validate_hygiene_candidate(candidate_data)
        except ValueError:
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        candidates.append(
            NoiseCandidate(
                memory_id=candidate_data["memory_id"],
                table_name=candidate_data["table_name"],
                content_preview=candidate_data.get("content_preview", ""),
                noise_score=candidate_data.get("noise_score", 0.0),
                noise_reasons=candidate_data.get("noise_reasons", []),
                secret_flags=candidate_data.get("secret_flags", []),
                importance=candidate_data.get("importance", 0.5),
                source=candidate_data.get("source", ""),
                timestamp=candidate_data.get("timestamp", ""),
                suggested_action=candidate_data.get("suggested_action", "keep"),
                content_length=candidate_data.get("content_length", 0),
            )
        )

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path

    action = arguments.get("action", "keep")
    confirm = arguments.get("confirm", False)
    dry_run = not confirm

    result = clean_noise(
        db_path=db_path,
        candidates=candidates,
        action=action,
        confirm=confirm,
        dry_run=dry_run,
    )
    return {
        "status": "dry_run" if dry_run else "applied",
        "result": result.to_dict(),
        "bank": bank,
    }


# ---------------------------------------------------------------------------
# Task 6B — Native Inhale / Dream / Reclaim / Persona / Sync handlers
# ---------------------------------------------------------------------------

def _dream_projection(run) -> Dict[str, Any]:
    """Content-free curated JSON projection of a DreamRun.

    Mirrors mnemosyne.cli._dream_run_projection exactly: NEVER includes scope,
    raw manifest, actions, before/after images, content, config audit, or
    unbounded failure text. Only durable identifiers, state, manifest hash,
    checkpoint, error code, safe timestamps, and receipt role/status counts.
    """
    receipt_counts: Dict[str, int] = {}
    raw_receipts = getattr(run, "receipts", None) or []
    if isinstance(raw_receipts, list):
        for r in raw_receipts:
            if not isinstance(r, dict):
                continue
            role = r.get("role", "unknown")
            verdict = r.get("verdict", "unknown")
            key = f"{role}:{verdict}"
            receipt_counts[key] = receipt_counts.get(key, 0) + 1
    action_count = 0
    raw_actions = getattr(run, "actions", None)
    if isinstance(raw_actions, list):
        action_count = len(raw_actions)
    # Content-free idempotency signal for dream_undo: core surfaces a second
    # undo via failure_reason="already_undone" (a fixed enum), which we expose
    # as an explicit boolean so first vs second undo are distinguishable
    # WITHOUT echoing the unbounded failure_reason field.
    already_undone = getattr(run, "failure_reason", None) == "already_undone"
    return {
        "run_id": run.run_id,
        "state": run.state,
        "manifest_hash": getattr(run, "manifest_hash", "") or "",
        "checkpoint": getattr(run, "checkpoint", "") or "",
        "error_code": getattr(run, "error_code", None),
        "created_at": getattr(run, "created_at", "") or "",
        "updated_at": getattr(run, "updated_at", "") or "",
        "request_id": getattr(run, "request_id", None),
        "action_count": action_count,
        "receipt_counts": receipt_counts,
        "already_undone": already_undone,
    }


def _receipt_projection(receipt) -> Dict[str, Any]:
    """Content-free projection of an IngestReceipt.

    Strips content, content_hash, and payload; keeps only durable identifiers,
    status enums, attempts, and structured error fields.
    """
    return {
        "event_id": getattr(receipt, "event_id", ""),
        "status": getattr(receipt, "status", ""),
        "index_status": getattr(receipt, "index_status", ""),
        "attempts": getattr(receipt, "attempts", 0),
        "last_error_code": getattr(receipt, "last_error_code", None),
        "last_error_at": getattr(receipt, "last_error_at", None),
        "created_at": getattr(receipt, "created_at", ""),
        "updated_at": getattr(receipt, "updated_at", ""),
    }


def _handle_ingest(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_ingest tool call — durable receipt-backed ingest."""
    import hashlib as _hashlib
    from mnemosyne.core.inhale import IngestEvent

    required = ("event_id", "producer", "actor_id", "project_id",
                "session_id", "turn_id", "role", "content", "occurred_at")
    missing = [f for f in required if not arguments.get(f)]
    if missing:
        return {"error": f"missing required fields: {', '.join(missing)}"}

    content = arguments["content"]
    metadata = arguments.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return {"error": "metadata must be an object"}

    event = IngestEvent(
        event_id=arguments["event_id"],
        producer=arguments["producer"],
        actor_id=arguments["actor_id"],
        project_id=arguments["project_id"],
        session_id=arguments["session_id"],
        turn_id=arguments["turn_id"],
        role=arguments["role"],
        content=content,
        content_hash=_hashlib.sha256(content.encode("utf-8")).hexdigest(),
        occurred_at=arguments["occurred_at"],
        metadata=metadata,
    )

    bank = _resolve_bank(arguments)
    mem = _create_instance(
        author_id=arguments.get("author_id"),
        author_type=arguments.get("author_type"),
        channel_id=arguments.get("channel_id"),
        bank=bank,
    )
    receipt = mem.remember_event(event)
    # NEVER echo content / content_hash in the result.
    result = _receipt_projection(receipt)
    result["bank"] = bank
    return result


def _read_only_beam(bank: str):
    """Return a lightweight beam-like exposing a read-only SQLite connection.

    Resolves the bank DB path WITHOUT materializing state (no Mnemosyne/beam
    construction, no config.yaml seed, no init_db) and opens it with
    ``mode=ro`` + ``query_only``. Used by read-only MCP handlers so a fresh
    data dir is never mutated by a readOnly-flagged call. Raises
    ``FileNotFoundError`` if the database does not exist yet.
    """
    import types
    from mnemosyne.core.banks import get_bank_db_path_read_only
    from mnemosyne.doctor import open_readonly_doctor_db
    db_path = get_bank_db_path_read_only(bank)
    try:
        conn = open_readonly_doctor_db(db_path)
    except sqlite3.OperationalError as exc:
        # mode=ro refuses to create a missing DB; surface as FileNotFoundError
        # so callers can return an empty structured result instead of mutating.
        if "unable to open" in str(exc).lower():
            raise FileNotFoundError(f"database for bank '{bank}' does not exist")
        raise
    return types.SimpleNamespace(conn=conn, db_path=db_path)


def _handle_ingest_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_ingest_status tool call — content-free receipts.

    Routed through a read-only DB connection: does NOT construct a Mnemosyne
    instance and therefore never materializes a default DB on a fresh data dir.
    """
    event_id = arguments.get("event_id") or None
    try:
        limit = int(arguments.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    if limit < 1:
        return {"error": "limit must be a positive integer"}

    bank = _resolve_bank(arguments)
    from mnemosyne.core.inhale import ingest_status as _ingest_status
    try:
        beam = _read_only_beam(bank)
    except (FileNotFoundError, ValueError):
        # No database exists yet for this bank: no receipts to report. Do NOT
        # materialize one; return an empty structured result.
        return {"status": "ok", "count": 0, "receipts": [], "bank": bank}
    try:
        rows = _ingest_status(beam, event_id=event_id, limit=limit)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    finally:
        beam.conn.close()
    return {
        "status": "ok",
        "count": len(rows),
        "receipts": [_receipt_projection(r) for r in rows],
        "bank": bank,
    }


def _handle_ingest_retry(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_ingest_retry tool call — re-index non-ready receipts."""
    try:
        limit = int(arguments.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    if limit < 1:
        return {"error": "limit must be a positive integer"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    report = mem.retry_pending_ingest(limit=limit)
    return {
        "status": "ok",
        "attempted": report.attempted,
        "succeeded": report.succeeded,
        "degraded": report.degraded,
        "failed_retryable": report.failed_retryable,
        "failed_terminal": report.failed_terminal,
        "bank": bank,
    }


def _handle_dream_plan(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_plan tool call."""
    session_id = arguments.get("session_id")
    if not session_id:
        return {"error": "session_id is required for dream plan"}

    scope: Dict[str, Any] = {"session_id": session_id}
    for opt in ("actor_id", "producer", "project_id"):
        v = arguments.get(opt)
        if v:
            scope[opt] = v

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    run = mem.dream_plan(
        scope=scope,
        limits=None,
        request_id=arguments.get("request_id"),
    )
    return _dream_projection(run)


def _handle_dream_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_status tool call."""
    run_id = arguments.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    run = mem.dream_status(run_id)
    return _dream_projection(run)


def _dream_submit_receipt_handler(arguments: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Shared handler for dream_review (role='reviewer') and dream_verify
    (role='verifier'). Role is fixed by the caller, never taken from args."""
    run_id = arguments.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    actor_id = arguments.get("actor_id")
    if not actor_id:
        return {"error": "actor_id is required"}
    verdict = arguments.get("verdict")
    if verdict not in ("PASS", "FAIL"):
        return {"error": "verdict must be PASS or FAIL"}

    from datetime import datetime, timezone
    manifest_hash = arguments.get("manifest_hash") or ""
    receipt = {
        "role": role,
        "actor_id": actor_id,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": verdict,
        "reason_code": arguments.get("reason_code") or "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    # Bind manifest_hash from the durable run if not supplied.
    if not receipt["manifest_hash"]:
        existing = mem.dream_status(run_id)
        receipt["manifest_hash"] = existing.manifest_hash or ""
    run = mem.dream_submit_receipt(run_id, receipt)
    return _dream_projection(run)


def _handle_dream_review(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_review — fixed reviewer role."""
    return _dream_submit_receipt_handler(arguments, "reviewer")


def _handle_dream_verify(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_verify — fixed verifier role."""
    return _dream_submit_receipt_handler(arguments, "verifier")


def _handle_dream_resume(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_resume tool call."""
    run_id = arguments.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    run = mem.dream_resume(run_id)
    return _dream_projection(run)


def _handle_dream_apply(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_apply tool call."""
    run_id = arguments.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    run = mem.dream_apply(run_id)
    return _dream_projection(run)


def _handle_dream_undo(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_dream_undo tool call."""
    run_id = arguments.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    run = mem.dream_undo(run_id)
    return _dream_projection(run)


def _handle_reclaim_orphans(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_reclaim_orphans — dry-run by default."""
    dry_run = arguments.get("apply") is not True
    try:
        stale_after_seconds = int(arguments.get("stale_after_seconds", 3600))
    except (TypeError, ValueError):
        stale_after_seconds = 3600
    if stale_after_seconds < 0:
        return {"error": "stale_after_seconds must be non-negative"}
    try:
        limit = int(arguments.get("limit", 1000))
    except (TypeError, ValueError):
        limit = 1000
    if limit < 0:
        return {"error": "limit must be non-negative"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    result = mem.reclaim_orphans(
        dry_run=dry_run,
        stale_after_seconds=stale_after_seconds,
        limit=limit,
    )
    # Strip any row-level content; expose counts only.
    return {
        "status": "ok",
        "dry_run": dry_run,
        "reclaimed": result.get("reclaimed", 0),
        "candidates": result.get("candidates", 0),
        "bank": bank,
    }


# ---------------------------------------------------------------------------
# triple_end gap closure (binding behavior #1)
# ---------------------------------------------------------------------------

def _handle_triple_end(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_triple_end tool call.

    Mirrors hermes_memory_provider._handle_triple_end: expires open triples
    for subject+predicate (or only the matching object when given).
    """
    subject = arguments.get("subject")
    predicate = arguments.get("predicate")
    if not subject or not predicate:
        return {"error": "subject and predicate are required"}
    obj = arguments.get("object") or None
    valid_until = arguments.get("valid_until") or None

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
    from mnemosyne.core.triples import end_triple
    n = end_triple(subject, predicate, object=obj, valid_until=valid_until,
                   db_path=db_path)
    return {"status": "ended", "count": n}


# ---------------------------------------------------------------------------
# Persona handlers (binding behavior #1) — thin adapter over PersonaAdapter
# ---------------------------------------------------------------------------

def _persona_adapter(mem):
    """Lazily build a PersonaAdapter bound to the given Mnemosyne's beam."""
    from hermes_memory_provider.persona_adapter import PersonaAdapter
    return PersonaAdapter(beam_instance=mem.beam)


def _persona_result(raw: str) -> Dict[str, Any]:
    """PersonaAdapter returns JSON strings; parse into a dict for MCP."""
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {"status": "error", "error": "internal: malformed persona response"}


def _handle_persona_promote(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_persona_promote tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    adapter = _persona_adapter(mem)
    raw = adapter.handle_tool_call("mnemosyne_persona_promote", {
        "memory_id": arguments.get("memory_id", ""),
        "tier": arguments.get("tier", "long_term"),
        "reason": arguments.get("reason", ""),
    })
    result = _persona_result(raw)
    result["bank"] = bank
    return result


def _handle_persona_demote(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_persona_demote tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    adapter = _persona_adapter(mem)
    raw = adapter.handle_tool_call("mnemosyne_persona_demote", {
        "persona_id": arguments.get("persona_id", 0),
        "reason": arguments.get("reason", ""),
    })
    result = _persona_result(raw)
    result["bank"] = bank
    return result


def _handle_persona_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_persona_list tool call.

    Routed through a read-only DB connection so the call never materializes a
    default DB or seeds config.yaml on a fresh data dir. PersonaAdapter._list
    only executes SELECTs, so a read-only conn is sufficient and truthful.
    """
    bank = _resolve_bank(arguments)
    try:
        beam = _read_only_beam(bank)
    except (FileNotFoundError, ValueError):
        return {"status": "ok", "count": 0, "personas": [], "bank": bank}
    try:
        from hermes_memory_provider.persona_adapter import PersonaAdapter
        adapter = PersonaAdapter(beam_instance=beam)
        raw = adapter.handle_tool_call("mnemosyne_persona_list", {
            "tier": arguments.get("tier"),
            "topic": arguments.get("topic"),
        })
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    finally:
        beam.conn.close()
    result = _persona_result(raw)
    result["bank"] = bank
    return result


def _handle_persona_reinforce(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_persona_reinforce tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    adapter = _persona_adapter(mem)
    raw = adapter.handle_tool_call("mnemosyne_persona_reinforce", {
        "persona_id": arguments.get("persona_id", 0),
    })
    result = _persona_result(raw)
    result["bank"] = bank
    return result


# ---------------------------------------------------------------------------
# Sync handlers (binding behavior #1, #6) — safe-default, no network widening
# ---------------------------------------------------------------------------

def _sync_resolve_remote() -> str:
    """Resolve the configured sync remote URL.

    Honors the SAME resolution as ``SyncAdapter._resolve_remote`` and the
    CLI/config layer: config.yaml ``sync_remote`` → ``MNEMOSYNE_SYNC_REMOTE``
    env → ``MNEMOSYNE_SYNC_HOST`` + ``MNEMOSYNE_SYNC_PORT`` env. A deployment
    configured through any of these must not be rejected as 'unconfigured'.

    An MCP handler never widens remote/network authority: this only resolves
    the configured value; it never accepts an inline URL from the call.
    """
    from mnemosyne.core.config import get_config
    remote = str(get_config().get("sync_remote", "") or "").strip()
    if remote:
        return remote
    remote = (os.environ.get("MNEMOSYNE_SYNC_REMOTE") or "").strip()
    if remote:
        return remote
    host = (os.environ.get("MNEMOSYNE_SYNC_HOST") or "").strip()
    port = (os.environ.get("MNEMOSYNE_SYNC_PORT") or "").strip()
    if host and port:
        return f"https://{host}:{port}"
    return ""


def _sync_unconfigured_result() -> Dict[str, Any]:
    return {
        "status": "unconfigured",
        "remote": "(unconfigured)",
        "error": "No remote configured. Set MNEMOSYNE_SYNC_REMOTE (or sync_remote in config.yaml, or MNEMOSYNE_SYNC_HOST+PORT), or use the CLI (mnemosyne sync --remote URL --db-path PATH).",
    }


def _handle_sync_push(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_sync_push tool call.

    Over MCP, sync push is gated on an explicitly configured remote. Without
    one we return a structured 'unconfigured' rejection rather than
    attempting any network call. This intentionally does NOT accept an inline
    remote URL: the CLI is the trust boundary for binding a remote.
    """
    if not _sync_resolve_remote():
        return _sync_unconfigured_result()
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    try:
        from hermes_memory_provider.sync_adapter import SyncAdapter
        adapter = SyncAdapter(mem.beam, config={})
        raw = adapter.handle_tool_call("mnemosyne_sync_push", {})
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        result = {"status": "error", "error": "sync_push_failed"}
    result["bank"] = bank
    return result


def _handle_sync_pull(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_sync_pull tool call."""
    if not _sync_resolve_remote():
        return _sync_unconfigured_result()
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    try:
        from hermes_memory_provider.sync_adapter import SyncAdapter
        adapter = SyncAdapter(mem.beam, config={})
        raw = adapter.handle_tool_call("mnemosyne_sync_pull", {})
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        result = {"status": "error", "error": "sync_pull_failed"}
    result["bank"] = bank
    return result


def _handle_sync_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_sync_status tool call.

    Local status (device id, event count, encryption state) is returned even
    when no remote is configured: ``SyncAdapter._handle_status`` produces this
    without any network contact. Only push/pull require a configured remote.
    """
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    try:
        from hermes_memory_provider.sync_adapter import SyncAdapter
        adapter = SyncAdapter(mem.beam, config={})
        raw = adapter.handle_tool_call("mnemosyne_sync_status", {})
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        result = {"status": "error", "error": "sync_status_failed"}
    result["bank"] = bank
    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "mnemosyne_remember": _handle_remember,
    "mnemosyne_batch": _handle_batch,
    "mnemosyne_recall": _handle_recall,
    "mnemosyne_shared_remember": _handle_shared_remember,
    "mnemosyne_shared_recall": _handle_shared_recall,
    "mnemosyne_shared_forget": _handle_shared_forget,
    "mnemosyne_shared_stats": _handle_shared_stats,
    "mnemosyne_sleep": _handle_sleep,
    "mnemosyne_stats": _handle_stats,
    "mnemosyne_invalidate": _handle_invalidate,
    "mnemosyne_validate": _handle_validate,
    "mnemosyne_get": _handle_get,
    "mnemosyne_triple_add": _handle_triple_add,
    "mnemosyne_triple_query": _handle_triple_query,
    "mnemosyne_triple_end": _handle_triple_end,
    "mnemosyne_remember_canonical": _handle_remember_canonical,
    "mnemosyne_recall_canonical": _handle_recall_canonical,
    "mnemosyne_forget_canonical": _handle_forget_canonical,
    "mnemosyne_scratchpad_write": _handle_scratchpad_write,
    "mnemosyne_scratchpad_read": _handle_scratchpad_read,
    "mnemosyne_scratchpad_clear": _handle_scratchpad_clear,
    "mnemosyne_export": _handle_export,
    "mnemosyne_update": _handle_update,
    "mnemosyne_forget": _handle_forget,
    "mnemosyne_import": _handle_import,
    "mnemosyne_diagnose": _handle_diagnose,
    "mnemosyne_graph_query": _handle_graph_query,
    "mnemosyne_graph_link": _handle_graph_link,
    "mnemosyne_hygiene_audit": _handle_hygiene_audit,
    "mnemosyne_hygiene_clean": _handle_hygiene_clean,
    # Task 6B — native Inhale / Dream / Reclaim endpoints
    "mnemosyne_ingest": _handle_ingest,
    "mnemosyne_ingest_status": _handle_ingest_status,
    "mnemosyne_ingest_retry": _handle_ingest_retry,
    "mnemosyne_dream_plan": _handle_dream_plan,
    "mnemosyne_dream_status": _handle_dream_status,
    "mnemosyne_dream_review": _handle_dream_review,
    "mnemosyne_dream_verify": _handle_dream_verify,
    "mnemosyne_dream_resume": _handle_dream_resume,
    "mnemosyne_dream_apply": _handle_dream_apply,
    "mnemosyne_dream_undo": _handle_dream_undo,
    "mnemosyne_reclaim_orphans": _handle_reclaim_orphans,
    # Task 6B — persona gap closures (thin adapter over PersonaAdapter)
    "mnemosyne_persona_promote": _handle_persona_promote,
    "mnemosyne_persona_demote": _handle_persona_demote,
    "mnemosyne_persona_list": _handle_persona_list,
    "mnemosyne_persona_reinforce": _handle_persona_reinforce,
    # Task 6B — sync gap closures (safe-default, no network widening)
    "mnemosyne_sync_push": _handle_sync_push,
    "mnemosyne_sync_pull": _handle_sync_pull,
    "mnemosyne_sync_status": _handle_sync_status,
}


def handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch an MCP tool call to the correct handler.

    Args:
        name: Tool name (e.g., "mnemosyne_remember")
        arguments: Parsed JSON arguments

    Returns:
        JSON-serializable result dict

    Raises:
        ValueError: If tool name is unknown
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}. Available: {list(_TOOL_HANDLERS.keys())}")

    return handler(arguments)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return all tool definitions for MCP server registration."""
    return TOOLS


# Keep the exact pre-lazy-import star-import surface.  ``Mnemosyne`` remains
# lazy through ``__getattr__`` above, but star imports historically resolved it
# because this module had no ``__all__`` and held the class in its globals.
# New typing-only implementation details deliberately stay private to stars.
__all__ = [
    "ALL_TOOL_SCHEMAS",
    "TOOLS",
    "Any",
    "BatchValidationError",
    "BeamMemory",
    "CallToolResult",
    "Dict",
    "ErrorData",
    "List",
    "Mnemosyne",
    "Path",
    "TextContent",
    "Tool",
    "apply_beam_batch",
    "batch_validation_error_payload",
    "dry_run_batch",
    "get_tool_definitions",
    "handle_tool_call",
    "json",
    "math",
    "os",
    "sqlite3",
    "validate_batch_operations",
]
