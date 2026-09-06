import jax.numpy as jnp
import numpy as np
from ase import Atoms

from deepmd_jax.ase_calc import DPJaxCalculator


def test_ase_calculator_skips_stress_when_only_forces_are_requested():
    calc = object.__new__(DPJaxCalculator)
    calc.atoms = None
    calc.results = {}
    calc._dtype = jnp.float32
    calc._static_args = "unchanged"
    calc._get_static_args = lambda position: "unchanged"
    calls = {"force": 0, "stress": 0}

    def energy_forces(coords, box, static_args):
        calls["force"] += 1
        return jnp.array(1.25), jnp.ones_like(coords)

    def energy_forces_stress(coords, box, static_args):
        calls["stress"] += 1
        return jnp.array(1.25), jnp.ones_like(coords), jnp.arange(6.0)

    calc._energy_and_forces_fn = energy_forces
    calc._energy_forces_and_stress_fn = energy_forces_stress
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[8, 8, 8], pbc=True)

    calc.calculate(atoms, properties=["forces"])

    assert calls == {"force": 1, "stress": 0}
    assert calc.results["energy"] == 1.25
    np.testing.assert_allclose(calc.results["forces"], np.ones((2, 3)))
    assert "stress" not in calc.results

    calc.calculate(atoms, properties=["stress"])

    assert calls == {"force": 1, "stress": 1}
    np.testing.assert_allclose(calc.results["stress"], np.arange(6.0))
