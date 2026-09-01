import numpy as np

from deepmd_jax.data import compute_lattice_candidate


def test_lattice_candidate_matches_previous_orthogonal_results():
    boxes = np.array([
        np.diag([12.0, 11.0, 10.0]),
        np.diag([11.5, 10.5, 9.5]),
    ])
    result = compute_lattice_candidate(boxes, 6.0, print_info=False)
    assert result == {
        'lattice_cand': (
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ),
        'lattice_max': 2,
        'ortho': True,
    }


def test_lattice_candidate_matches_previous_triclinic_results():
    boxes = np.array([[[10.0, 0.0, 0.0],
                       [1.2, 9.0, 0.0],
                       [0.5, 0.8, 8.0]]])
    result = compute_lattice_candidate(boxes, 6.0, print_info=False)
    assert result == {
        'lattice_cand': (
            (-1, 0, 0),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (1, 0, 0),
        ),
        'lattice_max': 2,
        'ortho': False,
    }
    disabled = compute_lattice_candidate(
        np.array([np.diag([12.0, 11.0, 10.0])]), 2.5,
        print_info=False, disable_ortho=True)
    assert disabled['ortho'] is False
