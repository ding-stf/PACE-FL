from __future__ import annotations
import math
import numpy as np

class RDPAccountant:
    """Conservative full-participation Gaussian RDP accountant.

    Each released group is sequentially composed because all groups depend on the
    same client update. Client subsampling amplification is deliberately omitted.
    """
    def __init__(self,orders: np.ndarray|None=None)->None:
        self.orders=np.asarray(orders if orders is not None else list(range(2,65))+[128,256],dtype=float); self.rdp=np.zeros_like(self.orders); self.events=[]
    def step(self,noise_multiplier: float,groups: int=1,steps: int=1)->None:
        if noise_multiplier<=0: raise ValueError("noise_multiplier must be positive")
        self.rdp+=steps*groups*self.orders/(2*noise_multiplier**2); self.events.append((noise_multiplier,groups,steps))
    def epsilon(self,delta: float)->float:
        if not 0<delta<1: raise ValueError("delta must be in (0,1)")
        return float(np.min(self.rdp+math.log(1/delta)/(self.orders-1)))
    @classmethod
    def calibrate_noise(cls,target_epsilon: float,delta: float,rounds: int,groups: int=1)->float:
        if target_epsilon<=0: raise ValueError("target epsilon must be positive")
        low,high=1e-3,1e4
        for _ in range(100):
            mid=(low+high)/2; a=cls(); a.step(mid,groups,rounds)
            if a.epsilon(delta)>target_epsilon: low=mid
            else: high=mid
        return high

