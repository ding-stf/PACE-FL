import torch
from spectral.temporal_spectral import dct,idct
from spectral.spectral_groups import transform_update,inverse_groups
def test_dct_round_trip():
    x=torch.randn(4,7); assert torch.allclose(idct(dct(x)),x,atol=1e-5)
def test_group_round_trip():
    u={"graph_conv.theta":torch.randn(3,2,2),"temporal_in.weight":torch.randn(4,1,3,1),"bias":torch.randn(2)}; assert all(torch.allclose(u[k],inverse_groups(transform_update(u),u)[k],atol=1e-5) for k in u)

