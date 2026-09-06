# Fe/site scalar transfer learning

This branch starts at `pyhope/portable-training-v1` (`bc4fb35`). Existing energy
training, portable parameters and full-state restart behavior are retained.

`DPModel(...)(..., descriptor_only=True)` returns its invariant descriptor before
any energy fitting network. Default energy behavior is unchanged. Descriptor rows
follow the model's species-sorted order for one visible device.

`deepmd_jax.moment.transfer_from_energy(...)` initializes a `MomentModel` with a
new selected-type scalar head and copies every backbone leaf from an uncompressed
energy model. It checks names, tree structures, shapes and dtypes, and permits only
energy fitting networks to be excluded. It keeps all pretrained normalization,
cutoff and MP settings. It does not implement energy-to-property migration through
a naive change of `atomic=True`, which could renumber Flax compact modules.

The head sees only the selected species. Same-species sorting is stable, so outputs
follow their input order. All species still contribute to the DP-MP environment.
All backbone/head leaves are trainable; parameters whose features do not influence
selected outputs naturally have zero gradients. The inherited energy model object
and arrays remain immutable. Current support requires one visible GPU/device per
process; independent replicas may run on different devices.

The scalar head uses tanh hidden layers and a linear final output. There is no
forced absolute value or SIGMA input. Label convention is chosen by the campaign:
the accompanying full-trajectory comparison uses **absolute** Fe PAW moments to
match the historical CHGNet training labels. This is a property predictor, not an
energy/force model or a spin-dependent potential.

`moment_loss` computes within-frame Fe-site Huber plus unique Fe-pair-difference
Huber (delta 0.05). Average the per-frame terms for frame-balanced minibatches.
Singleton-Fe frames contribute zero pair term, matching the reference CHGNet
runner's per-frame averaging convention.

GPU regression tests are in `tests/test_moment_transfer.py`; they cover exact
pretrained transfer with/without DP-MP, Fe order, finite and nonzero backbone
gradients, backbone updates, preserved original energy behavior, and scalar loss.
The campaign adds real pretrained-model reconstruction, spatial symmetry,
coordinate-gradient, exact serialization/resume and full training-data checks.
