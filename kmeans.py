import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
from scipy.ndimage import filters
import sys
from scipy.misc import imread
import os
import sklearn.preprocessing

from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import sys
import math

filename = "./DATA/official_ts_dataset_dft.csv"

def parseData(filename, datapts, losses, ids):
    f = open(filename)
    for line in f:
        if "2014" in line: #ID
            ids.append(line.strip())
            continue

        breakpt = line.find(":")

        loss = line[breakpt+1:].strip()
        losses.append(loss)

        ts = line[:breakpt]
        datapts.append([int(num) for num in ts.strip().split()])
    f.close()
    datapts = np.array(datapts)

datapts = []
losses = []
ids = []
parseData(filename, datapts, losses, ids)

'''
def getFilenames(filepath):
    #input: the filepath of the folder of interest
    #output: list of all the iflenames in the path of interest
    f = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        f.extend(filenames)
        break
    return f

def plotFourierRow(row):
    #num observations i think
    N = len(row)

    # sample spacing
    T = 93.0/129

    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = row
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    plt.grid()
    plt.show()

n = 129 #num samples in DTFT

#Ouput time series datapoints as ts_dataset.csv
output_file = open("official_ts_dataset.csv", 'w')
output_file.write("\"id\",\"loss\",{0}\n".format(",".join(["\"X" + str(x) + "\"" for x in range(1,130)])))
for i, ID in enumerate(ids):
  output_file.write("\"{0}\",{1},{2},{3}\n".format(ID, losses[i], ",".join([str(x) for x in list(datapts[i])])))

output_file.close()
sys.exit(1)

fTransformed = abs(np.array([np.fft.fft(row, n) for row in datapts]))
fTransformed = fTransformed[:,1:n//2]
print(fTransformed.shape)

# plt.plot(fTransformed[0], range(1,n//2))
# plt.show()

# print data.shape, fTransformed.shape
# plotFourierRow(fTransformed[200])
# sys.exit(1)

# Input: X Matrix (each row is a training point)
# Output: A vector of labels, corresponding to row indices

np.random.seed(341)

data = scale(fTransformed) # fTransformed = DTFT

n_samples, n_features = data.shape
print(n_samples, n_features)

k = 3 #expected clusters
sample_size = n_samples
labels = ids #If we want to manually label?

print("clusters: %d, \t n_samples %d, \t n_features %d"
      % (k, n_samples, n_features))

print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

def bench_k_means(estimator, name, data, dist_metric='euclidean'):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric=dist_metric,
                                      sample_size=sample_size)))


bench_k_means(KMeans(init='k-means++', n_clusters=k, n_init=10),
              name="k-means++", data=data, dist_metric='cosine')

bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
              name="random", data=data, dist_metric='cosine')

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=k).fit(data)
#print(pca.components_)
#print(len(pca.components_[1]))

bench_k_means(KMeans(init=pca.components_, n_clusters=k, n_init=1),
              name="PCA-based",
              data=data, dist_metric='cosine')
print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
#print(reduced_data)
kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_

# Go through dataset and associate labels to clusters
mapping = {0: [], 1: [], 2: []}
for i, point in enumerate(reduced_data):
  closest_centroid = 0
  min_dist = np.linalg.norm(point - centroids[0], 2)

  for centroid_index, centroid in enumerate(centroids):
    dist = np.linalg.norm(point - centroid, 2)
    if (dist < min_dist):
        min_dist = dist
        closest_centroid = centroid_index
  mapping[closest_centroid].append(labels[i])

#print(mapping)
print(len(mapping[0]), len(mapping[1]), len(mapping[2]))
sys.exit(1)

for label in mapping[2]:
    filename = "SquiggleExamples/" + label
    img = Image.open(filename)
    img.show(100)
    img.close()

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color=['w','b','r'], zorder=10)

plt.title('K-means clustering on the scaled DTFT dataset (PCA-reduced data)\n')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
'''
