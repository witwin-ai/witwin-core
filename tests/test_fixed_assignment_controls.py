"""Core discrete assignment controls preserve real geometry/material gradients."""
import pytest
import torch
import witwin.core as mw
from witwin.core import PhysicalMaterial as Material


@pytest.mark.parametrize('kind',('priority','enabled','nested_enabled'))
def test_received_trainable_structure_assignment_control_is_not_silently_replaced(kind):
    theta=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    kwargs={'enabled':(theta,)} if kind=='nested_enabled' else {kind:theta}
    with pytest.raises(NotImplementedError,match='fixed assignment control'):
        mw.Structure(mw.Box(size=(.02,.02,.02)),Material(eps_r=theta.exp()),**kwargs)


def test_fixed_tensor_controls_and_actual_box_material_gradients_remain_supported():
    theta=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    geometry=mw.Box(position=(theta*.001,0.,0.),size=(.02,.02,.02))
    material=Material(eps_r=theta.exp())
    value=mw.Structure(geometry,material,enabled=torch.tensor(1.),priority=torch.tensor(2.))
    assert value.enabled and value.priority==2
    gradient=torch.autograd.grad(value.geometry.position.sum()+value.material.eps_r,theta)[0]
    torch.testing.assert_close(gradient,theta.exp()+.001)


def test_object_array_nested_assignment_control_cannot_hide_tensor_ancestry():
    import numpy as np
    value=torch.tensor(1.,requires_grad=True)
    nested=np.empty(1,dtype=object);nested[0]=value
    with pytest.raises(NotImplementedError,match='fixed assignment control'):
        mw.Structure(mw.Box(size=(.02,.02,.02)),Material(),enabled=nested)
