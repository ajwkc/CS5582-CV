import csv
import cv2
from scipy import spatial, special
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt

# Constants
n_samples = 100
channels = [0, 1, 2]
hBins = 8
sBins = 4
vBins = 2
hRange = [0, 180]
sRange = [0, 256]
vRange = [0, 256]
label_path = 'cifar-10/trainLabels.csv'
image_path = 'cifar-10/train/'
x_axis = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8',
          'S1', 'S2', 'S3', 'S4', 'V1', 'V2']

# List of CIFAR-10 image labels
labels = ['airplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']


def get_hist(im_number):
    image = cv2.imread('cifar-10/train/' + str(im_number) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hHist = cv2.calcHist([image], [0], None, [hBins], [0, 180])
    sHist = cv2.calcHist([image], [1], None, [sBins], [0, 256])
    vHist = cv2.calcHist([image], [2], None, [vBins], [0, 256])
    total_hist = np.concatenate([hHist, sHist, vHist])
    total_hist = np.ndarray.flatten(total_hist).tolist()
    return total_hist


def plot_hist(im_number):
    hist = get_hist(im_number)
    plt.bar(x_axis, hist)
    plt.show()


def get_dist(img1, img2):
    hist1 = np.array(get_hist(img1))
    hist2 = np.array(get_hist(img2))
    Edist = np.linalg.norm(hist1 - hist2)
    KLdist = special.kl_div(hist1, hist2)
    print('Euclidean distance between %d.png and %d.png: %f' % (img1, img2, Edist))
    print('Kullback-Leibler divergence: ' + str(KLdist))
    return Edist, KLdist


def nearest_match(im_number):
    query = ave_vector_tree.query(get_hist(im_number))
    print('Image ' + str(im_number) + '.png predicted label: ' + labels[query[1]])
    for v in vector_list:
        if v[1] == get_hist(im_number):
            print('Actual label: ' + str(labels[v[0]]))


def ten_matches(im_number):
    im = get_hist(im_number)
    _, topMatch = vector_tree.query(im, k=10)
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 5),
                     axes_pad=0.2)
    for ax, i in zip(grid, topMatch):
        im = cv2.imread('cifar-10/train/' + str(vector_list[i][0]) + '.png')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im)
    plt.show()


# Open the csv
file = open(label_path, 'r')
reader = csv.reader(file)

# Index the images by label
img_list = []
for i in range(10):
    file.seek(0)  # Back to top of csv
    next(reader, None)  # Skip the header
    j = 0
    for row in reader:
        if row[1] == labels[i]:
            img_list.append([i, int(row[0])])
            j += 1
            if j == n_samples:
                break
file.close()

vector_list = []
for img in img_list:
    hist = get_hist(img[1])
    vector_list.append([img[1], labels[img[0]], hist])
vector_tree = spatial.KDTree([i[2] for i in vector_list])

vector_averages = []
for i in range(10):
    temp = []
    for _, label, vector in vector_list:
        if label == labels[i]:
            temp.append(vector)
    vector_averages.append([i, np.mean(temp, axis=0)])
ave_vector_tree = spatial.KDTree([i[1] for i in vector_averages])
