import sys

import pytest


_HERMES_MODULE_PREFIXES = ("hermes_memory_provider", "mnemosyne_hermes")


@pytest.fixture(scope="module", autouse=True)
def _restore_hermes_module_cache():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _HERMES_MODULE_PREFIXES
        )
    }
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _HERMES_MODULE_PREFIXES
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        for name, module in saved.items():
            parent_name, _, child_name = name.rpartition(".")
            if parent_name:
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, child_name, module)
