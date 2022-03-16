import os
import numpy as np
from glob import glob
import pandas as pd


# Split GEDI up into train/val/test sets
db_folder = "/home/drew/dropbox_symlink/GEDInls"
folders = ["GC150-Dead", "GC150-Live", "GC150nls-Dead", "GC150nls-Live"]
non_nls = folders[:2]
nls = folders[2:]
files = {}
min_files = 10000000
for f in folders:
    files[f] = glob(os.path.join(db_folder, f, "*.tif"))
    min_files = min(min_files, len(files[f]))

print("Minimum files: {}".format(min_files))

# Now trim to min_files # files in each folder
for f in folders:
    files[f] = files[f][:min_files]

# Now split each into train/val/test sets
val_test_split = 0.2
train, val, test = {}, {}, {}
for f in folders:
    it_files = files[f]
    train_cutoff = int(len(it_files) * val_test_split)
    train[f] = it_files[:train_cutoff]
    val_test = it_files[train_cutoff:]
    val_test_split = train_cutoff // 2
    val[f] = val_test[:val_test_split]
    test[f] = val_test[val_test_split:]

# Package non-nls together
non_nls_train, non_nls_val, non_nls_test = [], [], []
for f in non_nls:
    non_nls_train = non_nls_train + train[f]
    non_nls_val = non_nls_val + val[f]
    non_nls_test = non_nls_test + test[f]

# Package nls together
nls_train, nls_val, nls_test = [], [], []
for f in non_nls:
    nls_train = nls_train + train[f]
    nls_val = nls_val + val[f]
    nls_test = nls_test + test[f]

# Save csv and npz of the files
np.savez(
    os.path.join("data", "non_nls_files"),
    train=non_nls_train,
    val=non_nls_val,
    test=non_nls_test)
np.savez(
    os.path.join("data", "nls_files"),
    train=nls_train,
    val=nls_val,
    test=nls_test)

non_nls_data = non_nls_train + non_nls_val + non_nls_test
non_nls_splits = ["train"] * len(non_nls_train) + ["val"] * len(non_nls_val)  + ["test"] * len(non_nls_test)
non_nls_df = pd.DataFrame(np.stack((non_nls_data, non_nls_splits), 1), columns=["file", "split"])
non_nls_df.to_csv(os.path.join("data", "non_nls_splits.csv"))

nls_data = nls_train + nls_val + nls_test
nls_splits = ["train"] * len(nls_train) + ["val"] * len(nls_val)  + ["test"] * len(nls_test)
nls_df = pd.DataFrame(np.stack((nls_data, nls_splits), 1), columns=["file", "split"])
nls_df.to_csv(os.path.join("data", "nls_splits.csv"))
