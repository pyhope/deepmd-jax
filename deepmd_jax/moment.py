"""Selected-site scalar transfer learning from an uncompressed energy DPModel.

The backbone retains its original species-complete descriptor/MP computation.
Only the new head and output are restricted to selected sites. No energy head,
force derivative, frame parameter, absolute-value activation, or new MP is used.
"""
import copy
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from .dpmodel import DPModel


class MomentModel(nn.Module):
    backbone_params: dict
    selected_type: int
    head_widths: tuple = (64, 64, 64)
    output_bias: float = 3.7

    @nn.compact
    def __call__(self, coord, box, static_args):
        if jax.device_count() != 1:
            raise ValueError('MomentModel currently requires one visible device.')
        types = np.asarray(static_args['type_idx'])
        if not np.any(types == self.selected_type):
            raise ValueError('No selected sites in this structure.')
        features = DPModel(self.backbone_params, name='backbone')(
            coord, box, static_args, descriptor_only=True)
        selected = np.flatnonzero(types[np.argsort(types, kind='stable')] == self.selected_type)
        x = features[selected]
        for i, width in enumerate(self.head_widths):
            x = nn.tanh(nn.Dense(width, name=f'moment_hidden_{i}')(x))
        return nn.Dense(1, name='moment_output',
                        bias_init=nn.initializers.constant(self.output_bias))(x)[:, 0]


def transfer_from_energy(energy_model, energy_variables, key, coord, box, static_args,
                         selected_type, head_widths=(64, 64, 64), output_bias=3.7):
    """Copy ALL descriptor leaves exactly; initialize only a new scalar head."""
    p = copy.deepcopy(energy_model.params)
    if p.get('atomic') or p.get('is_compressed') or p.get('hybrid') or p['type'] != 'energy':
        raise ValueError('Transfer requires an uncompressed, non-hybrid energy model.')
    if selected_type not in p['valid_types']:
        raise ValueError('Selected type absent from pretrained type map.')
    model = MomentModel(p, selected_type, tuple(head_widths), float(output_bias))
    variables = unfreeze(model.init(key, coord, box, static_args))
    source = unfreeze(energy_variables)['params']
    target = variables['params']['backbone']
    excluded = set(source) - set(target)
    if not excluded or any(not name.startswith('fitting_net_') for name in excluded):
        raise ValueError(f'Unexpected unused pretrained parameters: {excluded}')
    for name, value in target.items():
        if name not in source:
            raise ValueError(f'Missing descriptor module {name}')
        a, sa = jax.tree_util.tree_flatten(value)
        b, sb = jax.tree_util.tree_flatten(source[name])
        if sa != sb or any(np.shape(x) != np.shape(y) or np.asarray(x).dtype != np.asarray(y).dtype
                          for x, y in zip(a, b)):
            raise ValueError(f'Descriptor shape/dtype mismatch: {name}')
        target[name] = source[name]
    variables = freeze(variables)
    return model, variables


def huber(error, delta=0.05):
    absolute = jnp.abs(error)
    return jnp.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))


def moment_loss(prediction, target, delta=0.05):
    """One-frame site/pair terms, ready for frame-balanced gradient accumulation."""
    error = prediction - target
    site = jnp.mean(huber(error, delta))
    i, j = np.triu_indices(error.shape[0], 1)
    pair = jnp.mean(huber(error[i] - error[j], delta)) if len(i) else jnp.zeros_like(site)
    return site + pair, (site, pair)
