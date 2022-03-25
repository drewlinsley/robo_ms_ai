import numpy as np
from matplotlib import pyplot as plt


non_nls = np.load("experiments/2022-03-22/20-02-34/attributions.npz", allow_pickle=True, encoding="latin1")
nls = np.load("experiments/2022-03-22/19-57-37/attributions.npz", allow_pickle=True, encoding="latin1")

non_nls_imgs = non_nls["images"]
nls_imgs = nls["images"]

non_nls_attr = non_nls["viz"]
nls_attr = nls["viz"]

thresh = 0
comb_fun = lambda x: np.max(np.abs(x), -1)



