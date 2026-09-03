import json
import hashlib
import pickle
import shutil

import jax
import numpy as np
import pytest

from deepmd_jax.train import test as evaluate_model
from deepmd_jax.train import train
from deepmd_jax.utils import load_model


def _write_dataset(path):
    set_dir = path / 'set.000'
    set_dir.mkdir(parents=True)
    nframes = 8
    coords = np.zeros((nframes, 3, 3), dtype=np.float32)
    forces = np.zeros_like(coords)
    energies = np.zeros(nframes, dtype=np.float32)
    for frame in range(nframes):
        x1 = 1.25 + 0.025 * frame
        x2 = 2.70 - 0.015 * frame
        coords[frame, :, 0] = [0.0, x1, x2]
        d01 = x1 - 1.4
        d12 = (x2 - x1) - 1.4
        energies[frame] = 0.5 * (d01**2 + d12**2)
        forces[frame, 0, 0] = d01
        forces[frame, 1, 0] = -d01 + d12
        forces[frame, 2, 0] = -d12
    boxes = np.repeat(np.eye(3, dtype=np.float32)[None] * 6.0,
                      nframes, axis=0)
    np.savetxt(path / 'type.raw', np.zeros(3, dtype=int), fmt='%d')
    np.save(set_dir / 'coord.npy', coords.reshape(nframes, -1))
    np.save(set_dir / 'box.npy', boxes.reshape(nframes, -1))
    np.save(set_dir / 'energy.npy', energies)
    np.save(set_dir / 'force.npy', forces.reshape(nframes, -1))


def _train_kwargs(dataset, save_path, checkpoint_path, history_path):
    return dict(
        model_type='energy',
        rcut=2.5,
        train_data_path=str(dataset),
        val_data_path=str(dataset),
        save_path=str(save_path),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        checkpoint_every=1,
        step=6,
        seed=20260901,
        mp=False,
        embed_widths=[4, 4, 8],
        fit_widths=[8, 8],
        axis_neurons=2,
        batch_size=2,
        val_batch_size_ratio=1,
        print_every=2,
        getstat_bs=2,
        compress=False,
        loss='l2',
    )


def _assert_trees_identical(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_segment_resume_matches_continuous_training(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)

    continuous = _train_kwargs(
        dataset, tmp_path / 'continuous.pkl', tmp_path / 'continuous.train.pkl',
        tmp_path / 'continuous.history.json')
    continuous_result = train(**continuous)
    assert continuous_result['completed']
    continuous_model, _ = load_model(continuous['save_path'], replicate=False)
    portable_params = tmp_path / 'portable-model-params.pkl'
    portable_raw = pickle.dumps(
        jax.tree_util.tree_map(
            lambda value: np.asarray(value) if hasattr(value, 'shape') else value,
            continuous_model.params),
        protocol=pickle.HIGHEST_PROTOCOL)
    portable_params.write_bytes(portable_raw)
    portable_params.with_name(portable_params.name + '.sha256').write_text(
        hashlib.sha256(portable_raw).hexdigest() + '  portable-model-params.pkl\n')

    segmented = _train_kwargs(
        dataset, tmp_path / 'segmented.pkl', tmp_path / 'segmented.train.pkl',
        tmp_path / 'segmented.history.json')
    first_result = train(**segmented, max_updates_per_run=3)
    assert first_result == {
        'completed': False,
        'completed_updates': 3,
        'target_updates': 6,
        'checkpoint_path': str(tmp_path / 'segmented.train.pkl'),
    }
    assert not (tmp_path / 'segmented.pkl').exists()
    resumed_result = train(
        **segmented, resume=True, model_params_path=str(portable_params))
    assert resumed_result['completed']

    continuous_model, continuous_variables = load_model(
        continuous['save_path'], replicate=False)
    segmented_model, segmented_variables = load_model(
        segmented['save_path'], replicate=False)
    assert continuous_model.params == segmented_model.params
    _assert_trees_identical(continuous_variables, segmented_variables)
    continuous_history = json.loads(
        (tmp_path / 'continuous.history.json').read_text())
    segmented_history = json.loads(
        (tmp_path / 'segmented.history.json').read_text())
    assert continuous_history == segmented_history
    assert [record['update'] for record in continuous_history] == [2, 4, 6]


def test_checkpoint_hash_and_contract_fail_closed(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)
    kwargs = _train_kwargs(
        dataset, tmp_path / 'model.pkl', tmp_path / 'model.train.pkl',
        tmp_path / 'history.json')
    result = train(**kwargs, max_updates_per_run=2)
    assert not result['completed']

    clean_checkpoint = tmp_path / 'clean.train.pkl'
    clean_sidecar = tmp_path / 'clean.train.pkl.sha256'
    shutil.copy2(kwargs['checkpoint_path'], clean_checkpoint)
    shutil.copy2(kwargs['checkpoint_path'] + '.sha256', clean_sidecar)

    with open(kwargs['checkpoint_path'], 'ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        train(**kwargs, resume=True)

    contract_kwargs = dict(kwargs)
    contract_kwargs['checkpoint_path'] = str(clean_checkpoint)
    contract_kwargs['lr'] = 0.003
    with pytest.raises(ValueError, match='differing_fields=lr'):
        train(**contract_kwargs, resume=True)


def test_exact_signature_prewarm_preserves_training_trajectory(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)
    seed_kwargs = _train_kwargs(
        dataset, tmp_path / 'seed.pkl', tmp_path / 'seed.train.pkl',
        tmp_path / 'seed.history.json')
    seed_result = train(**seed_kwargs, max_updates_per_run=2)
    assert seed_result['completed_updates'] == 2

    baseline_checkpoint = tmp_path / 'baseline.train.pkl'
    baseline_sidecar = tmp_path / 'baseline.train.pkl.sha256'
    prewarm_checkpoint = tmp_path / 'prewarm.train.pkl'
    prewarm_sidecar = tmp_path / 'prewarm.train.pkl.sha256'
    for destination, sidecar in (
            (baseline_checkpoint, baseline_sidecar),
            (prewarm_checkpoint, prewarm_sidecar)):
        shutil.copy2(seed_kwargs['checkpoint_path'], destination)
        shutil.copy2(seed_kwargs['checkpoint_path'] + '.sha256', sidecar)

    baseline_kwargs = _train_kwargs(
        dataset, tmp_path / 'baseline.pkl', baseline_checkpoint,
        tmp_path / 'baseline.history.json')
    baseline_result = train(**baseline_kwargs, resume=True)
    assert baseline_result['completed']

    prewarm_kwargs = _train_kwargs(
        dataset, tmp_path / 'prewarm.pkl', prewarm_checkpoint,
        tmp_path / 'prewarm.history.json')
    checkpoint_bytes = prewarm_checkpoint.read_bytes()
    sidecar_bytes = prewarm_sidecar.read_bytes()
    prewarm_result = train(
        **prewarm_kwargs, resume=True, prewarm_updates=4, prewarm_only=True)
    assert prewarm_result['prewarm_only']
    assert prewarm_result['completed_updates'] == 2
    assert prewarm_result['prewarm_planned_updates'] == 4
    assert prewarm_result['prewarm_train_signatures'] == 1
    assert prewarm_result['prewarm_validation_signatures'] == 1
    assert prewarm_checkpoint.read_bytes() == checkpoint_bytes
    assert prewarm_sidecar.read_bytes() == sidecar_bytes

    resumed_result = train(**prewarm_kwargs, resume=True)
    assert resumed_result['completed']
    baseline_model, baseline_variables = load_model(
        baseline_kwargs['save_path'], replicate=False)
    prewarm_model, prewarm_variables = load_model(
        prewarm_kwargs['save_path'], replicate=False)
    assert baseline_model.params == prewarm_model.params
    _assert_trees_identical(baseline_variables, prewarm_variables)
    assert json.loads((tmp_path / 'baseline.history.json').read_text()) == json.loads(
        (tmp_path / 'prewarm.history.json').read_text())


def test_dpmp_train_save_reload_and_test(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)
    model_path = tmp_path / 'dpmp.pkl'
    result = train(
        model_type='energy',
        rcut=2.5,
        train_data_path=str(dataset),
        save_path=str(model_path),
        step=2,
        seed=20260902,
        mp=True,
        embed_widths=[4, 4, 8],
        embed_mp_widths=[8, 8, 8],
        fit_widths=[8, 8],
        axis_neurons=2,
        batch_size=2,
        print_every=1,
        getstat_bs=2,
        compress=False,
        loss='l2',
    )
    assert result['completed']
    assert model_path.is_file()
    assert (tmp_path / 'dpmp.pkl.sha256').is_file()
    metrics, predictions = evaluate_model(str(model_path), str(dataset), batch_size=2)
    assert len(predictions) == 8
    assert np.isfinite(metrics['rmse']['energy'])
    assert np.isfinite(metrics['rmse']['force'])
    with open(model_path, 'ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_model(str(model_path), replicate=False)
