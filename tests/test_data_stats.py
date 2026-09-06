import numpy as np

from deepmd_jax.data import Dataset


def test_shared_stats_jit_matches_previous_result(tmp_path):
    set_dir = tmp_path / 'set.000'
    set_dir.mkdir()
    nframes = 4
    coords = np.zeros((nframes, 3, 3), dtype=np.float32)
    coords[:, :, 0] = [
        [0.0, 1.3, 2.7],
        [0.0, 1.4, 2.8],
        [0.0, 1.5, 2.9],
        [0.0, 1.6, 3.0],
    ]
    boxes = np.repeat(
        (np.eye(3, dtype=np.float32) * 6.0)[None], nframes, axis=0)
    np.savetxt(tmp_path / 'type.raw', [0, 0, 0], fmt='%d')
    np.save(set_dir / 'coord.npy', coords.reshape(nframes, -1))
    np.save(set_dir / 'box.npy', boxes.reshape(nframes, -1))
    np.save(set_dir / 'energy.npy', np.arange(nframes, dtype=np.float32))
    np.save(set_dir / 'force.npy', np.zeros((nframes, 9), dtype=np.float32))

    dataset = Dataset(
        str(tmp_path), ['coord', 'box', 'energy', 'force'],
        rng=np.random.default_rng(7))
    dataset.compute_lattice_candidate(2.5, False)
    stats = dataset.get_stats(2.5, 2)
    np.testing.assert_allclose(stats['sr_mean'], [0.29563713], rtol=0, atol=1e-8)
    np.testing.assert_allclose(stats['sr_std'], [0.044928342], rtol=0, atol=1e-8)
    np.testing.assert_allclose(stats['Nnbrs'], 2.3333335, rtol=0, atol=1e-7)
