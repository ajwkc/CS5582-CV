import cv2
import glob

owList = glob.glob('OwenWilson/*.jp*g')
new = []
for image in owList:
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (20, 20))
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new.append(im)

