import numpy as np

from data_loading import Data
from analysis import get_design_matrix

subject = "01"  # subjects 01 and 02
run = "1"
task = "bird"
#space = "T1w"
space = "MNI152NLin2009cAsym"
tr = 1.97
project = "TC2See_prdgm"

run = 1  # 8 runs in [1, 8]
data = Data(project, subject, task, 1, space)

# the tr (=1.97)
print(data.tr)

# access the scan with  (89, 105, 89, 201)
# data is in a nilearn format
# to get the numpy array call data.fmri_img.get_fdata()
print(data.fmri_img.shape)

# or the anatomical data  (193, 229, 193)
print(data.anat.shape)

# or the mask
print(data.mask)

# calculate the design_matrix
design_matrix = get_design_matrix(frame_times=np.arange(data.fmri_img.shape[3])*data.tr, events=data.events, keep_order=False)

print(design_matrix)

