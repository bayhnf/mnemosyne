"""Real thread/process regression for issue #833; temporary databases only."""
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from mnemosyne.core import beam


def initialize(path, barrier, legacy=False):
    barrier.wait(timeout=30)
    if legacy:
        from mnemosyne.core.memory import init_db
        init_db(Path(path))
    else:
        beam.init_beam(Path(path))


def check(path):
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        columns = [r[1] for r in conn.execute('PRAGMA table_info(working_memory)')]
        assert columns.count('recall_count') == 1
    beam.init_beam(path)


@pytest.mark.parametrize('legacy', [False, True])
def test_threads(tmp_path, legacy):
    import threading
    path = tmp_path / 'threads.db'
    for _ in range(3):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(initialize, str(path), barrier, legacy) for _ in range(2)]
            for future in futures:
                future.result(timeout=60)
        check(path)


@pytest.mark.parametrize('legacy', [False, True])
def test_processes(tmp_path, legacy):
    ctx = multiprocessing.get_context('spawn')
    path = tmp_path / 'processes.db'
    barrier = ctx.Barrier(2)
    children = [ctx.Process(target=initialize, args=(str(path), barrier, legacy)) for _ in range(2)]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(60)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
    check(path)


def test_alias(tmp_path):
    path = tmp_path / 'real.db'
    beam.init_beam(path)
    alias = tmp_path / 'alias.db'
    alias.symlink_to(path)
    beam.init_beam(alias)
    assert beam._get_connection(alias) is beam._get_connection(path)
    assert not Path(str(alias) + '.init.lock').exists()


@pytest.mark.parametrize('message', ['disk I/O error', 'database is locked', 'attempt to write a readonly database'])
def test_errors_propagate(tmp_path, monkeypatch, message):
    def fail(path):
        raise sqlite3.OperationalError(message)
    monkeypatch.setattr(beam, '_get_connection', fail)
    with pytest.raises(sqlite3.OperationalError, match=message):
        beam.init_beam(tmp_path / 'failure.db')


def test_sidecar_error(tmp_path):
    path = tmp_path / 'failure.db'
    Path(str(path) + '.init.lock').mkdir()
    with pytest.raises(IsADirectoryError):
        beam.init_beam(path)
