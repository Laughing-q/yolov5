import os
import numpy as np
import glob

label_paths = 'data/license_plates/labels/val_/*'
label_paths_n = 'data/license_plates/labels/val'
os.makedirs(label_paths_n, exist_ok=True)


for p in glob.glob(label_paths):
    with open(p, 'r') as f:
        label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
    label = np.delete(label, [1, 2, 3, 4], axis=1)
    np.savetxt(p.replace('val_', 'val'), label)

