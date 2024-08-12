""" Preprocessing functions for generator.py """
import numpy as np

LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4]
    + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)

def merge_homologs(label):#, device):
    max_value = 31
    # idx = torch.arange(max_value + 1, dtype=torch.long).to(device)
    idx = np.arange(max_value+1)
    idx[31] = 17
    idx[30] = 16
    idx[29] = 15
    idx[28] = 14
    idx[27] = 10
    idx[26] = 9
    idx[25] = 8
    idx[24] = 7
    idx[23] = 6
    idx[22] = 5
    idx[21] = 4
    idx[20] = 3
    idx[19] = 2
    idx[18] = 1
    # return the corresponding values from idx
    return idx[label]

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab].astype(np.uint8)

def preprocess_image_min_max(img: np.ndarray):
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    img = preprocess_image_min_max(img) * 255
    img = img.astype(np.uint8)
    lab = preprocess_label(lab)
    lab = merge_homologs(lab)
    lab = lab.astype(np.uint8)    
    return (img, lab) 
