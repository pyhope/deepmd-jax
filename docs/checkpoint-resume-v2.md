# Fast checkpoint resume (experimental)

Based on `pyhope/checkpoint-resume-v1`. The numerical model, loss, gradient,
Adam update and sampling algorithms are unchanged. This first implementation
accelerates non-hybrid `energy` training only; other models retain the v1 path.

`fast_resume=True` (default when checkpointing) saves an immutable initialization
snapshot once. It contains prepared host datasets (including shifted coordinates,
ordered types and lattice/neighbor bounds), exact serialized model parameters,
source/environment identity and a full source-data SHA256 manifest. Checkpoints
retain all original model/Adam/iteration/loss/history/sampler/RNG fields and add
only a content-addressed snapshot reference. Resume validates source data contents
and set inventory, loads the snapshot, and bypasses dataset constructors, neighbor
statistics, energy fitting, random model initialization and Adam initialization.
Sampler state is always restored from the current checkpoint, not the snapshot.

Legacy checkpoints remain readable. Their first v2 resume performs legacy
initialization and writes the snapshot; the saving cost must be reported separately
from subsequent fast resumes. `fast_resume=False` uses the legacy path. Corrupt or
incompatible referenced snapshots fail explicitly; they are never silently ignored.
Use `resume_cache_path` to relocate the snapshot while retaining its bound hash.
Snapshot files contain trusted pickle objects, like the existing checkpoints.

Optional `executable_cache_dir` stores serialized *complete* training updates and
validation executables using JAX's experimental serialization API. On a hit the
shape-specialized callable is loaded directly, bypassing tracing and lowering.
Keys bind source/packages, the scientific model contract, loss smoothing, input
pytree/shape/dtype/weak type, static arguments, device kinds, runtime platform,
XLA flags and precision settings. Incompatible environments use different keys;
corrupt matching artifacts fail. This option must be qualified for the installed
JAX/runtime. Leave it unset for ordinary persistent JAX compilation caching.
Preserve the complete initialization snapshot plus its SHA256 sidecar and the
executable-cache directory, in addition to the ordinary checkpoint and sidecar.

This is not a CUDA-process snapshot. A new process still initializes CUDA, reads
and checks files, allocates/transfers arrays and loads binaries. Newly encountered
shapes must compile once. The initialization cache is immutable and written once,
while shape-specific executable files can be added atomically. Keep the existing
persistent JAX cache enabled as a fallback when a new executable key is compiled.

Qualification must compare v1/v2 first resumed update, complete optimizer/sampler
state and a multi-shape trajectory, with exact integer/RNG/order states and a
preregistered floating tolerance; report process startup, first interval, later
intervals, steady throughput, memory, cache hits and one-time preparation costs.

## In-memory lookup optimization

Validated batch shape/dtype/weak-type + immutable static arguments now route
straight to a loaded executable. Full weight/Adam/state pytree flattening,
stringification, pickling and SHA256 computation happen only for a new route.
The compiled callable retains JAX's runtime pytree/shape/dtype validation for all
arguments. Model and optimizer structures are fixed for a training invocation;
an incompatible state on an existing route is rejected by the compiled callable,
not silently recompiled. Batch shapes and static arguments remain part of the
route and trigger a full signature on first encounter. No tensor values are
hashed. Disk cache identity, file checksums, data/initialization guards and the
scientific update are unchanged. `lookup_stats` counts full signatures and fast
route hits. `executable_cache_fast_lookup=False` enables the old lookup path for
paired performance checks; it does not change the disk-cache keys or the math.

### Restored device state and executable cache keys

Checkpoint arrays are stored on the host. Resume now places `variables`,
`opt_state`, and `state` on the execution device after contract validation and
before the first update. This preserves numerical values while matching the JAX
abstract-value representation used by ongoing training. Without this placement,
Python reference sharing in the pickle-encoded signature can give equal shapes
and dtypes different executable keys on the first resumed call.

The checkpoint format, sampler/RNG restoration, learning-rate schedule and cache
compatibility checks are unchanged. This addresses a reproducible false miss; it
does not eliminate legitimate compilation for new shapes, hardware or software.
Upgrading source changes the implementation signature, so existing initialization
snapshots and executable caches must still pass the normal compatibility checks.
Do not overwrite a frozen running campaign's source to apply this change.

`tests/test_checkpoint_device_state.py` checks bitwise state preservation and a
warm executable HIT after a fresh checkpoint deserialization. The numerical
continuous-versus-segmented regression remains in `tests/test_training_resume.py`.
The production-sized GPU A/B qualification is separate and was still pending
when this change was prepared; no GPU speedup is asserted here.
