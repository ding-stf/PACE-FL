import torch
from models.stgcn import STGCN
def test_stgcn_shape_and_named_parameters():
    model=STGCN(1,8,12,3); x=torch.randn(2,12,5,1); y=model(x,torch.eye(5)); y_dynamic=model(x,torch.eye(5).repeat(2,1,1)); assert y.shape==y_dynamic.shape==(2,12,5); assert model.graph_conv.theta.shape[0]==3; assert "temporal_in.weight" in dict(model.named_parameters())
