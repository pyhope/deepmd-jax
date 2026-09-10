"""Content-bound, process-independent initialization snapshots for training.

These trusted local pickle artifacts are accelerators, not replacements for
training checkpoints. Never load artifacts from an untrusted source.
"""
import hashlib
import io
import importlib.metadata
import os
from pathlib import Path
import pickle
import tempfile
from .utils import _write_sha256_sidecar, _verify_sha256_sidecar


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(paths):
    """Hash contents, including the set inventory; never trust mtimes alone."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    result = []
    for path in paths:
        if isinstance(path, (tuple, list)):
            result.extend(dataset_manifest(path))
            continue
        p = Path(path).absolute()
        if p.is_file():
            files = [p]
        else:
            files = sorted(p.glob("*.raw")) + sorted(p.glob("set.*/*.npy"))
            if not files or not (p / "type.raw").is_file():
                raise ValueError("Invalid dataset for resume cache: %s" % p)
        result.append((str(p), [(str(f.relative_to(p)) if p.is_dir() else f.name,
                                file_sha256(f)) for f in files]))
    return result


def implementation_signature():
    root = Path(__file__).parent
    return {
        "source": {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("jax", "jaxlib", "numpy", "flax", "optax", "ase")},
    }


def write_cache(path, payload):
    """Write once; checkpoints reference an immutable cache by its SHA256."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite initialization cache: %s" % path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        digest = file_sha256(tmp)
        # Atomic no-clobber publication, including concurrent writers.
        os.link(tmp, path)
        _write_sha256_sidecar(str(path), digest)
    finally:
        os.unlink(tmp)
    return {"path": str(path), "sha256": digest, "version": 1}


def load_cache(reference, request, manifests):
    path = reference["path"]
    if reference.get("version") != 1:
        raise ValueError("Unsupported initialization cache version")
    digest = _verify_sha256_sidecar(path, required=True)
    if digest != reference["sha256"]:
        raise ValueError("Initialization cache does not match checkpoint")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if cache["implementation"] != implementation_signature():
        raise ValueError("Initialization cache code/environment mismatch")
    if cache["request"] != request or cache["manifests"] != manifests:
        raise ValueError("Initialization cache input/data mismatch")
    return cache


def executable_signature_digest(signature):
    """Hash the complete acyclic signature without Python object-alias topology.

    Pickle memoization can encode equal shape/static metadata differently after
    a dataset snapshot reload. All types and values are retained; memo references
    are disabled for this acyclic metadata only, never for stored checkpoints.
    """
    buffer = io.BytesIO()
    writer = pickle.Pickler(buffer, protocol=5)
    writer.fast = True
    writer.dump(signature)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class ExecutableCache:
    """Optional same-runtime binary reuse, bypassing tracing on a cache hit.

    The caller must bind every closed-over constant in ``contract_sha256``.
    Each shape/static-argument variant is immutable and independently checked.
    GPU placement and runtime changes use a new namespace rather than forcing
    an incompatible executable to load.
    """
    def __init__(self, directory, contract_sha256, fast_lookup=True):
        import jax
        self.directory = Path(directory).absolute()
        self.memory = {}
        self.fast_lookup = fast_lookup
        self.routes = {}
        self.lookup_stats = dict(full_signatures=0, route_hits=0)

        self.identity = {
            'implementation': implementation_signature(),
            'contract': contract_sha256,
            'devices': [(d.platform, d.device_kind) for d in jax.devices()],
            'platform_version': jax.devices()[0].client.platform_version,
            'xla_flags': os.environ.get('XLA_FLAGS', ''),
            'x64': jax.config.jax_enable_x64,
            'matmul_precision': jax.config.jax_default_matmul_precision,
        }

    def call(self, name, function, dynamic_args, static_args):
        import jax
        import time
        from jax.experimental import serialize_executable
        # Model/optimizer/state structure is fixed by the training contract.
        # Route on the changing batch's abstract type and immutable static args;
        # the loaded Compiled callable still validates ALL runtime arguments.
        # No tensor values are hashed, copied, or inspected on this fast path.
        route = None
        if self.fast_lookup:
            batch_leaves, batch_tree = jax.tree_util.tree_flatten(dynamic_args[0])
            batch_types = []
            for leaf in batch_leaves:
                aval = jax.core.get_aval(leaf)
                batch_types.append((tuple(aval.shape), aval.dtype, bool(aval.weak_type)))
            route = (name, function, batch_tree, tuple(batch_types), static_args)
            compiled = self.routes.get(route)
            if compiled is not None:
                self.lookup_stats['route_hits'] += 1
                return compiled(*dynamic_args)
        self.lookup_stats['full_signatures'] += 1
        leaves, tree = jax.tree_util.tree_flatten(dynamic_args)
        abstract = []
        for leaf in leaves:
            aval = jax.core.get_aval(leaf)
            abstract.append((tuple(aval.shape), str(aval.dtype), bool(aval.weak_type)))
        signature = (self.identity, name, str(tree), abstract, static_args)
        key = executable_signature_digest(signature)
        if key not in self.memory:
            start = time.monotonic()
            path = self.directory / (name + '-' + key + '.pkl')
            if path.exists():
                _verify_sha256_sidecar(str(path), required=True)
                with path.open('rb') as f:
                    artifact = pickle.load(f)
                if artifact['key'] != key:
                    raise ValueError('Executable cache signature mismatch')
                binary, in_tree, out_tree = artifact['executable']
                compiled = serialize_executable.deserialize_and_load(
                    binary, in_tree, out_tree, execution_devices=jax.devices())
                mode = 'HIT'
            else:
                compiled = function.lower(*dynamic_args, static_args).compile()
                executable = serialize_executable.serialize(compiled)
                write_cache(path, {'key': key, 'executable': executable})
                mode = 'BUILD'
            self.memory[key] = compiled
            print('# Executable cache %s %s %s %.6f s' %
                  (mode, name, key, time.monotonic() - start), flush=True)
        compiled = self.memory[key]
        if route is not None:
            self.routes[route] = compiled
        return compiled(*dynamic_args)
