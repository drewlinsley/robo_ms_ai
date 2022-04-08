import numpy as np
from glob2 import glob
import os


cytosol = glob(os.path.join("/home/drew/dropbox_symlink/GEDInlsCurationExamples/Dead-with-Cytosol/", "*.tif"))
nipple = glob(os.path.join("/home/drew/dropbox_symlink/GEDInlsCurationExamples/Dead-with-nipple/", "*.tif"))

files = np.asarray(cytosol + nipple)
labels = np.ones(files.shape, dtype=int)

np.savez(
    os.path.join("data", "nls_viz_files_train"),
    files=files,
    labels=labels)

np.savez(
    os.path.join("data", "nls_viz_files_val"),
    files=files,
    labels=labels)

np.savez(
    os.path.join("data", "nls_viz_files_test"),
    files=files,
    labels=labels)


