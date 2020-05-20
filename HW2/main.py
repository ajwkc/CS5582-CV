import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.feature import hog, daisy


def getFeatures(image, opt):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if opt == "sift":
        sift = cv2.xfeatures2d_SIFT.create()
        _, d = sift.detectAndCompute(image, None)
        return d

    elif opt == "hog":
        d, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=True, multichannel=False)
        return d

    elif opt == "daisy":
        d, _ = daisy(image, step=5, radius=24, rings=2, histograms=6,
                     orientations=8, visualize=True)
        return d


def get_dist(img1, img2):
    Edist = np.linalg.norm(img1 - img2)
    return Edist


mpPaths = glob.glob('CDVS/mp*.jpg')
mp = []
for image in mpPaths:
    f = getFeatures(image, 'hog')
    mp.append(f)
mpArray = np.stack(mp)

nmp = []
nmpPaths = glob.glob('CDVS/nmp*.jpg')
for image in nmpPaths:
    f = getFeatures(image, 'hog')
    nmp.append(f)
nmpArray = np.stack(nmp)
