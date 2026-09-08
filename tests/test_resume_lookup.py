"""Allocated-GPU qualification for the executable lookup fast path."""
import tempfile
import unittest
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from deepmd_jax.resume import ExecutableCache


class ResumeLookupTest(unittest.TestCase):
    def test_routes_preserve_runtime_guards(self):
        self.assertEqual(jax.default_backend(), 'gpu')
        @partial(jax.jit, static_argnames=('static_args',))
        def update(batch, params, static_args):
            return batch['x'] * params['w'] + static_args['bias']
        with tempfile.TemporaryDirectory() as directory:
            c = ExecutableCache(directory, 'lookup-qualification')
            batch = {'x': jnp.arange(4, dtype=jnp.float32)}
            params = {'w': jnp.ones(4, dtype=jnp.float32)}
            static = FrozenDict(bias=2.)
            expected = np.arange(4, dtype=np.float32) + 2
            for _ in range(3):
                np.testing.assert_array_equal(c.call('update', update, (batch, params), static), expected)
            self.assertEqual(c.lookup_stats, dict(full_signatures=1, route_hits=2))
            for invalid in ({'w': jnp.ones(3, dtype=jnp.float32)},
                            {'w': jnp.ones(4, dtype=jnp.float16)},
                            {'w': params['w'], 'extra': jnp.array(0.)}):
                with self.assertRaises((TypeError, ValueError)):
                    c.call('update', update, (batch, invalid), static)
            self.assertEqual(c.lookup_stats['full_signatures'], 1)
            # New batch abstract type and static values produce new checked routes.
            batch8 = {'x': jnp.arange(8, dtype=jnp.float32)}
            params8 = {'w': jnp.ones(8, dtype=jnp.float32)}
            c.call('update', update, (batch8, params8), static).block_until_ready()
            c.call('update', update, (batch, params), FrozenDict(bias=3.)).block_until_ready()
            self.assertEqual(c.lookup_stats['full_signatures'], 3)
            # Cold in-process route map still loads the same trusted disk artifact.
            loaded = ExecutableCache(directory, 'lookup-qualification')
            np.testing.assert_array_equal(loaded.call('update', update, (batch, params), static), expected)
            legacy = ExecutableCache(directory, 'lookup-qualification', fast_lookup=False)
            for _ in range(3):
                np.testing.assert_array_equal(legacy.call('update', update, (batch, params), static), expected)
            self.assertEqual(legacy.lookup_stats['full_signatures'], 3)


if __name__ == '__main__':
    unittest.main()
