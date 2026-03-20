import numpy as np

gamma=1.4
r=0.01
rho=1.0


# compute initial pressure inside radius r
V=np.pi*r**2
eps=0.311357
print("2d - initial pressure p={}".format((gamma-1)*eps/V))

V=4/3*np.pi*r**3
eps=0.851072
print("3d - initial pressure p={}".format((gamma-1)*eps/V))
