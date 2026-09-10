"""Resume placement preserves values and reuses a warm serialized executable."""
import contextlib
import io
import pickle
from functools import partial

import jax
import numpy as np

from deepmd_jax.resume import ExecutableCache
from deepmd_jax.train import _restore_training_state


def test_restored_state_reuses_warm_executable(tmp_path):
    checkpoint = {
        'variables': {'a': np.ones((3, 8), np.float32),
                      'b': np.full((3, 8), 2, np.float32)},
        'opt_state': {'count': np.asarray(3, np.int32)},
        'state': {'iteration': np.asarray(3, np.int32)},
        'train_sampler_state': {'rng_state': {'value': 42}},
    }
    before = pickle.dumps(checkpoint, protocol=5)
    restored = _restore_training_state(checkpoint)
    host = tuple(checkpoint[k] for k in ('variables', 'opt_state', 'state'))
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(host)
    for a, b in zip(jax.tree_util.tree_leaves(host),
                    jax.tree_util.tree_leaves(restored)):
        assert isinstance(b, jax.Array)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == np.asarray(b).tobytes()
    assert pickle.dumps(checkpoint, protocol=5) == before

    @partial(jax.jit, static_argnums=(4,))
    def update(batch, variables, opt_state, state, static_args):
        return batch + variables['a'] + variables['b']

    batch = np.ones((3, 8), np.float32)
    def call(state):
        cache = ExecutableCache(str(tmp_path), 'checkpoint-placement', fast_lookup=True)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            result = cache.call('train', update, (batch,) + state, ())
        return np.asarray(result), log.getvalue()

    # Emulate a warm training call, then a separate checkpoint load/cache object.
    expected, warm_log = call(jax.device_put(host))
    actual, resume_log = call(_restore_training_state(pickle.loads(before)))
    assert 'BUILD' in warm_log
    assert 'HIT' in resume_log and 'BUILD' not in resume_log
    np.testing.assert_array_equal(expected, actual)
