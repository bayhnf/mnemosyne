"""Regression tests for base installs without optional embedding dependencies."""

import os
import subprocess
import sys
import textwrap


_BLOCK_OPTIONAL_DEPS = r"""
import importlib.abc
import sys

class BlockOptionalEmbeddingDeps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        if fullname == "fastembed" or fullname.startswith("fastembed."):
            raise ModuleNotFoundError("No module named 'fastembed'")
        return None

sys.meta_path.insert(0, BlockOptionalEmbeddingDeps())
"""


def _run_with_optional_embedding_deps_blocked(code: str, tmp_path):
    env = os.environ.copy()
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    env["HOME"] = str(tmp_path / "home")
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_OPTIONAL_DEPS + "\n" + code],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_embeddings_module_imports_without_numpy_or_fastembed(tmp_path):
    result = _run_with_optional_embedding_deps_blocked(
        textwrap.dedent(
            """
            from mnemosyne.core import embeddings

            assert embeddings.available() is False
            assert embeddings.embed_query("hello") is None
            assert embeddings.embed(["hello"]) is None
            """
        ),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr


def test_cli_stats_works_without_optional_embedding_dependencies(tmp_path):
    result = _run_with_optional_embedding_deps_blocked(
        textwrap.dedent(
            """
            import sys
            from mnemosyne.cli import run_cli

            sys.argv = ["mnemosyne", "stats"]
            run_cli()
            """
        ),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "Mnemosyne Stats" in result.stdout
    assert "Traceback" not in result.stderr


def test_shmr_propose_harmony_offline_without_numpy(tmp_path):
    """Task 25: propose_harmony must run on a base install with no NumPy.

    The documented SHMR lexical/offline fallback path (_lexical_vector and
    _cosine_similarity) is reached when embeddings are off, regardless of
    whether NumPy is importable. A naive ``try/except import numpy`` is not
    enough: the offline path itself used to call ``np.zeros`` /
    ``np.linalg.norm`` / ``np.dot``. This test forces both numpy and
    fastembed to be unimportable in a subprocess and drives the complete
    propose_harmony() offline path end-to-end (seed -> gather -> lexical
    vector -> cosine cluster -> LLM call -> no-proposal outcome).
    """
    result = _run_with_optional_embedding_deps_blocked(
        textwrap.dedent(
            """
            from mnemosyne.core import shmr
            from mnemosyne.core.beam import BeamMemory

            # NumPy must be unavailable in this subprocess.
            assert shmr.np is None, (
                "Task 25 requires shmr.np to be None when numpy is unimportable"
            )

            # Force the offline lexical path even if an embedding backend
            # somehow looks reachable.
            shmr._embedding_fn = lambda: None

            beam = BeamMemory(session_id="s1", db_path=r"DBPATH")
            beam.conn.executemany(
                "INSERT INTO facts "
                "(fact_id, session_id, subject, predicate, object, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("f1", "s1", "alice", "likes",
                     "rust programming language", 0.9),
                    ("f2", "s1", "alice", "likes",
                     "the rust language for systems work", 0.9),
                ],
            )
            beam.conn.commit()

            out = shmr.propose_harmony(
                beam, llm_call=lambda prompt, system="": "[]",
                similarity_threshold=0.4,
            )

            assert out["status"] in ("no_convergence", "proposed"), out
            assert out["clusters_found"] >= 1, out
            """
        ).replace("DBPATH", str(tmp_path / "shmr_offline.db")),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
