"""Transfer identity, selected-site order, and differentiable scalar loss."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.core import freeze
from deepmd_jax.dpmodel import DPModel
from deepmd_jax.moment import transfer_from_energy, moment_loss


@pytest.mark.parametrize('mp',[False,True])
def test_transfer_and_full_gradient(mp):
 params=dict(type='energy',atomic=False,hybrid=False,ntypes=2,valid_types=np.array([0,1]),rcut=3.,axis=2,use_2nd=True,use_mp=mp,embed_widths=(4,8),embedMP_widths=(8,8),fit_widths=(8,8),sr_mean=np.array([.1,.1],dtype=np.float32),sr_std=np.array([.2,.2],dtype=np.float32),Nnbrs=4.,Ebias=np.zeros(2,dtype=np.float32),out_norm=1.)
 st=freeze(dict(type_idx=(1,0,1,0),use_neighborlist=False,max_nbrs=None,lattice=dict(lattice_cand=((0,0,0),),lattice_max=1,ortho=True)))
 x=jnp.array([[0.,0.,0.],[1.,.2,0.],[2.,0.,.1],[0.,1.5,.2]],dtype=jnp.float32);b=jnp.eye(3)*12
 energy=DPModel(params);ev=energy.init(jax.random.PRNGKey(1),x,b,st)
 model,v=transfer_from_energy(energy,ev,jax.random.PRNGKey(2),x,b,st,1,(8,8),3.7)
 for key in v['params']['backbone']:
  for u,w in zip(jax.tree_util.tree_leaves(v['params']['backbone'][key]),jax.tree_util.tree_leaves(ev['params'][key])):np.testing.assert_array_equal(u,w)
 old=float(energy.apply(ev,x,b,st)[0]);prediction=model.apply(v,x,b,st)
 assert prediction.shape==(2,)
 gradient=jax.grad(lambda vv:moment_loss(model.apply(vv,x,b,st),jnp.array([3.9,3.5]))[0])(v)
 assert float(optax.global_norm(gradient['params']['backbone']))>0
 assert all(np.isfinite(z).all() for z in jax.tree_util.tree_leaves(gradient))
 updated=jax.tree_util.tree_map(lambda a,g:a-.001*g,v,gradient)
 assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(v['params']['backbone']),jax.tree_util.tree_leaves(updated['params']['backbone'])))
 assert float(energy.apply(ev,x,b,st)[0])==old
 np.testing.assert_allclose(model.apply(v,x[jnp.array([2,1,0,3])],b,st),prediction[::-1],atol=1e-6)


def test_loss_convention():
 pred=jnp.array([3.8,3.6,3.9]);target=jnp.array([3.7,3.65,3.8]);e=np.asarray(pred-target)
 def h(z):return np.where(abs(z)<=.05,.5*z*z,.05*(abs(z)-.025))
 i,j=np.triu_indices(3,1)
 np.testing.assert_allclose(moment_loss(pred,target)[0],h(e).mean()+h(e[i]-e[j]).mean(),rtol=1e-6)
 assert float(moment_loss(jnp.array([3.7]),jnp.array([3.7]))[0])==0
 assert float(moment_loss(jnp.array([3.8]),jnp.array([3.7]))[1][1])==0
