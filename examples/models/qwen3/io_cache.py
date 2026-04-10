"""I/O tensor caching plugin for pypto-lib examples.

Provides utilities to cache input tensors and golden outputs to disk,
so that re-runs skip the expensive golden computation during development.

Usage (add to any example file)::

    from io_cache import make_cache_dir, wrap_tensor_specs, wrap_golden
    from io_cache import add_cache_args, setup_cache_dir

    # 1. Define DEFAULT_CACHE_DIR with hyperparameter hash:
    DEFAULT_CACHE_DIR = make_cache_dir("prefill_scope3",
        BATCH=BATCH, MAX_SEQ=MAX_SEQ, HIDDEN=HIDDEN, INTERMEDIATE=INTERMEDIATE)

    # 2. In compile_and_run, wrap specs and golden:
    tensor_specs = wrap_tensor_specs(build_tensor_specs(...), cache_dir)
    golden = wrap_golden(original_golden, "out", cache_dir)
    result = run(..., tensor_specs=tensor_specs, golden=golden, ...)

    # 3. In __main__, add CLI args:
    add_cache_args(parser, DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    cache_dir = setup_cache_dir(args)
    compile_and_run(..., cache_dir=cache_dir)
"""
from __future__ import annotations

import hashlib


def make_cache_dir(scope_name: str, **hyperparams) -> str:
    """Generate a cache directory path with a deterministic hyperparameter hash.

    Changing any hyperparameter produces a different directory, preventing
    shape mismatches when re-using cached tensors.
    """
    items = sorted(hyperparams.items())
    key = "|".join(f"{k}={v}" for k, v in items)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"build_output/{scope_name}_io_cache_{h}"


def wrap_tensor_specs(specs, cache_dir: str | None):
    """Replace callable init_values in tensor specs with cached versions.

    On first call (no cache), generates tensors via original init_value and
    saves them to disk.  On subsequent calls, loads pre-computed tensors.
    Factory functions (torch.randn, etc.) and non-callable init_values are
    left unchanged.

    A helper ``_loaders.py`` module is written to *cache_dir* so that each
    tensor gets a loader function with a unique name — this avoids
    golden_writer name-collision issues.
    """
    if cache_dir is None:
        return specs
    import importlib.util
    import os

    import torch

    os.makedirs(cache_dir, exist_ok=True)
    _skip = {torch.randn, torch.rand, torch.zeros, torch.ones}

    # Phase 1 — generate & cache tensors that need caching.
    cached_entries: list[tuple[str, str]] = []  # (spec.name, cache_path)
    for spec in specs:
        init_fn = getattr(spec, "init_value", None)
        if spec.is_output or not callable(init_fn) or init_fn in _skip:
            continue
        cache_path = os.path.join(cache_dir, f"{spec.name}.pt")
        if not os.path.exists(cache_path):
            t = init_fn()
            torch.save(t, cache_path)
        cached_entries.append((spec.name, cache_path))

    if not cached_entries:
        return specs

    # Phase 2 — write a loader module with one uniquely-named function per
    # tensor so that golden_writer emits separate preamble entries.
    loader_path = os.path.join(cache_dir, "_loaders.py")
    lines = ["import torch\n\n"]
    for name, path in cached_entries:
        safe = name.replace("-", "_").replace(".", "_")
        lines.append(f"_path_{safe} = {path!r}\n\n\n")
        lines.append(f"def _load_{safe}():\n")
        lines.append(f"    return torch.load(_path_{safe})\n\n\n")
    with open(loader_path, "w") as f:
        f.writelines(lines)

    # Phase 3 — import the loader module and assign functions.
    mod_spec = importlib.util.spec_from_file_location(
        "_io_cache_loaders", loader_path,
    )
    loaders = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(loaders)

    for spec in specs:
        init_fn = getattr(spec, "init_value", None)
        if spec.is_output or not callable(init_fn) or init_fn in _skip:
            continue
        safe = spec.name.replace("-", "_").replace(".", "_")
        loader_fn = getattr(loaders, f"_load_{safe}", None)
        if loader_fn is not None:
            spec.init_value = loader_fn

    return specs


def wrap_golden(golden_fn, output_name: str, cache_dir: str | None,
                tensor_specs=None):
    """Wrap a golden function with cache load/save.

    Always returns a golden_writer-compatible loader.  If no cache exists,
    pre-computes the golden output using *tensor_specs* before returning
    the loader.

    Args:
        golden_fn: Original golden function ``(tensors, params) -> None``.
        output_name: Name of the output tensor (e.g. ``"out"``, ``"attn_out"``).
        cache_dir: Cache directory path, or ``None`` to disable caching.
        tensor_specs: List of ``TensorSpec`` — required on first run so
            that golden can be pre-computed outside of ``run()``.
    """
    if cache_dir is None:
        return golden_fn
    import os

    golden_path = os.path.join(cache_dir, f"golden_{output_name}.pt")

    if not os.path.exists(golden_path):
        import time

        import torch

        if tensor_specs is None:
            raise ValueError(
                "tensor_specs is required for first-run golden pre-computation. "
                "Pass the (wrapped) tensor_specs from wrap_tensor_specs()."
            )
        # Build tensors dict matching what the runtime feeds to golden.
        tensors = {}
        for spec in tensor_specs:
            if spec.is_output:
                tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
            else:
                t = spec.init_value()
                tensors[spec.name] = t.to(spec.dtype) if hasattr(t, "to") else t
        print("Pre-computing golden for caching...")
        t0 = time.time()
        golden_fn(tensors, {})
        elapsed = time.time() - t0
        print(f"Golden pre-computation took {elapsed:.3f}s")
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(tensors[output_name].clone(), golden_path)

    # Always return a simple loader — only string closures, so
    # golden_writer serialisation works correctly.
    def _load_golden(tensors, params):
        import torch

        tensors[output_name][:] = torch.load(golden_path)

    return _load_golden


def add_cache_args(parser, default_cache_dir: str | None = None):
    """Add ``--cache-dir`` and ``--clear-cache`` to an argparse parser."""
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=default_cache_dir,
        help="directory for cached I/O tensors (default: %(default)s)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="delete cached tensors before running",
    )


def setup_cache_dir(args) -> str | None:
    """Process cache CLI args.  Returns absolute cache_dir path or ``None``."""
    import os
    import shutil

    if args.cache_dir is None:
        return None
    cache_dir = os.path.abspath(args.cache_dir)
    if args.clear_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")
    if os.path.isdir(cache_dir):
        print(f"Using cached I/O from: {cache_dir}")
    else:
        print(f"No cache found, will generate and save to: {cache_dir}")
    return cache_dir
