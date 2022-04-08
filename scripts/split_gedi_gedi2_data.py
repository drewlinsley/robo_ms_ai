import os
import numpy as np
from glob import glob
import pandas as pd


# Split GEDI up into train/val/test sets
db_folder = "/home/drew/dropbox_symlink/GEDInls"
folders = ["GEDI2-Dead", "GEDI2-Live", "GC150nls-Dead", "GC150nls-Live"]
non_nls = folders[:2]
nls = folders[2:]
files = {}
min_files = 10000000
for f in folders:
    files[f] = glob(os.path.join(db_folder, f, "*.tif"))
    print(len(files[f]))
    min_files = min(min_files, len(files[f]))
print("Minimum files: {}".format(min_files))

# Now trim to min_files # files in each folder
for f in folders:
    files[f] = files[f][:min_files]

# Now split each into train/val/test sets
val_test_split = 0.8
train, val, test = {}, {}, {}
for f in folders:
    it_files = files[f]
    train_cutoff = int(len(it_files) * val_test_split)
    train[f] = it_files[:train_cutoff]
    val_test = it_files[train_cutoff:]
    it_val_test_split = len(val_test) // 2
    val[f] = val_test[:it_val_test_split]
    test[f] = val_test[it_val_test_split:]

# Package non-nls together
non_nls_train, non_nls_val, non_nls_test = [], [], []
non_nls_train_labels, non_nls_val_labels, non_nls_test_labels = [], [], []
for f in non_nls:
    non_nls_train = non_nls_train + train[f]
    non_nls_val = non_nls_val + val[f]
    non_nls_test = non_nls_test + test[f]
    label = 1 if "Live" in f else 0
    non_nls_train_labels = non_nls_train_labels + [label] * len(train[f])
    non_nls_val_labels = non_nls_val_labels + [label] * len(val[f])
    non_nls_test_labels = non_nls_test_labels + [label] * len(test[f])

# Package nls together
nls_train, nls_val, nls_test = [], [], []
nls_train_labels, nls_val_labels, nls_test_labels = [], [], []
for f in nls:
    nls_train = nls_train + train[f]
    nls_val = nls_val + val[f]
    nls_test = nls_test + test[f]
    label = 1 if "Live" in f else 0
    nls_train_labels = nls_train_labels + [label] * len(train[f])
    nls_val_labels = nls_val_labels + [label] * len(val[f])
    nls_test_labels = nls_test_labels + [label] * len(test[f])

# Save csv and npzs of the files
np.savez(
    os.path.join("data", "gedi2_files_train"),
    files=non_nls_train,
    labels=non_nls_train_labels)
np.savez(
    os.path.join("data", "gedi2_files_val"),
    files=non_nls_val,
    labels=non_nls_val_labels)
np.savez(
    os.path.join("data", "gedi2_files_test"),
    files=non_nls_test,
    labels=non_nls_test_labels)
np.savez(
    os.path.join("data", "gedi2_nls_files_train"),
    files=nls_train,
    labels=nls_train_labels)
np.savez(
    os.path.join("data", "gedi2_nls_files_val"),
    files=nls_val,
    labels=nls_val_labels)
np.savez(
    os.path.join("data", "gedi2_nls_files_test"),
    files=nls_test,
    labels=nls_test_labels)

non_nls_data = non_nls_train + non_nls_val + non_nls_test
non_nls_labels = non_nls_train_labels + non_nls_val_labels + non_nls_test_labels
non_nls_splits = ["train"] * len(non_nls_train) + ["val"] * len(non_nls_val)  + ["test"] * len(non_nls_test)
non_nls_df = pd.DataFrame(np.stack((non_nls_data, non_nls_splits, non_nls_labels), 1), columns=["file", "split", "label"])
non_nls_df.to_csv(os.path.join("data", "gedi2_splits.csv"))

nls_data = nls_train + nls_val + nls_test
non_nls_labels = nls_train_labels + nls_val_labels + nls_test_labels
nls_splits = ["train"] * len(nls_train) + ["val"] * len(nls_val)  + ["test"] * len(nls_test)
nls_df = pd.DataFrame(np.stack((nls_data, nls_splits, non_nls_labels), 1), columns=["file", "split", "label"])
nls_df.to_csv(os.path.join("data", "gedi2_nls_splits.csv"))
