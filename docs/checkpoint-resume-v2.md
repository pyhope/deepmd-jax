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
