"""Transfer an MP energy model to native selected atomic scalars.

Flax auto-numbered modules depend on the selected central types. Map by
semantic loop position, never by matching names alone. Retain the native
scalar normalization and newly initialized final scalar readout.
"""
import numpy as np
import jax
from flax.core import unfreeze

def inherit_energy(params, variables, source_model, source_variables):
    old = source_model.params
    assert old['type'] == 'energy' and params['type'] == 'atomic_scalar'
    assert params['use_mp'] and old['use_mp']
    for key in ('rcut', 'embed_widths', 'embedMP_widths',
                'axis', 'use_2nd', 'valid_types', 'ntypes'):
        assert np.array_equal(params[key], old[key]), key
    types = list(params['valid_types']); n = len(types)
    selected = [types.index(t) for t in params['nsel']]; m = len(selected)
    dst = unfreeze(variables); src = unfreeze(source_variables)
    mapping = {'Tbias':'Tbias'}
    # First selected-pair embedding; two all-pair feature embeddings;
    # then selected-pair message-passing embedding.
    for a,i in enumerate(selected):
        for j in range(n):
            mapping[f'embedding_net_{a*n+j}'] = f'embedding_net_{i*n+j}'
            mapping[f'embedding_net_{m*n+2*n*n+a*n+j}'] = f'embedding_net_{3*n*n+i*n+j}'
            for k in range(4):
                mapping[f'linear_norm_{4*(a*n+j)+k}'] = f'linear_norm_{4*(i*n+j)+k}'
    for k in range(2*n*n):
        mapping[f'embedding_net_{m*n+k}'] = f'embedding_net_{n*n+k}'
    inherit_fit = np.array_equal(params['fit_widths'], old['fit_widths'])
    if inherit_fit:
        for a,i in enumerate(selected): mapping[f'fitting_net_{a}'] = f'fitting_net_{i}'
    else:
        print('# Fresh magnetic fitting network:', params['fit_widths'])
    copied = 0
    for target,origin in mapping.items():
        a,b=dst['params'][target],src['params'][origin]
        assert jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b),(target,origin)
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            assert x.shape==y.shape,(target,origin,x.shape,y.shape)
        if target.startswith('fitting_net_'):
            # Output carries energy units; use the seeded native magnetic head.
            final = 'Dense_'+str(len(params['fit_widths']))
            b = dict(b); b[final] = a[final]
        dst['params'][target] = b
        copied += sum(x.size for x in jax.tree_util.tree_leaves(b))
    new=set(dst['params'])-set(mapping)
    expected = {f'layer_norm_{i}_{j}' for i in selected for j in range(n)}
    if not inherit_fit:
        expected |= {f'fitting_net_{a}' for a in range(m)}
    assert new == expected, new
    print('# Finetune semantic module map:', mapping)
    print('# Inherited modules:',len(mapping),'parameters including fresh head:',copied)
    return dst
