import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
from scipy.ndimage import filters
import sys
import os

output_file = 'ts_data.txt'
master_matrix = []

#TODO: account for issue of wrong starting point? start at any point --> move forward

# Put a dot on original image, where we selected the timepoint

# Questions:
# 1) How do we distinguish between intensities when points are either black/white?
# 2) How do we use a median filter to get rid of noisy sections?

# Takes in two "points": point1 = column index, point2 = (column index, intensity)
def loss(point1, point2, alpha, beta, intensities, row, col):
    col_index1 = point1
    (row_index2, col_index2, intensity2) = point2

    neighbor_intensity = 0
    if (row_index2 != 0 and row_index2 != row-1 and col_index2 != 0 and col_index2 != col-1):
        for row_index in range(row_index2-1, row_index2+2):
            for col_index in range(int(col_index2)-1, int(col_index2)+2):
                if row_index != col_index:
                    neighbor_intensity += intensities[row_index, col_index]

    return ( -1.0*alpha*intensity2 + -1.0*beta*neighbor_intensity + (1.0-alpha-beta)*math.pow(col_index1-col_index2, 2) )

def dynamic_algo(row, col, intensities, img, filename):
    # k = tuning parameters; number of top intensity points to consider per time interval
    k = 40
    # alpha = weight on intensity vs. deviation in loss function
    alpha = 0.1
    # beta = weight on neighbor square intensities
    beta = 0.9

    #initialize to set of potential points per time interval based on highest intensity
    optimal_points = np.zeros((row,k))
    for row_index in range(row):
        optimal_points[row_index,:] = np.argsort(intensities[row_index,:])[k*-1:]

    ###########

    # Dynamic Programming Algorithm: start with one of k initial points, deterministically select next point based on minimizing loss function
    # Choose optimal of k paths based on lowest aggregate loss

    optimal_paths = np.zeros((row,k))
    aggregate_loss = np.zeros(k)

    # Iterate through potential initial points
    for initial_index, initial_point in enumerate(optimal_points[0,:]):
        optimal_paths[0, initial_index] = initial_point

        # Iterate through each row_index corresponding to a time interval slice
        for row_index in range(1,row):

            # Initialize to first point in list of potential optimal points
            nextpt = optimal_points[row_index,0]

            # point1 = previous optimal point, point2 = current potentially optimal point in question
            point1 = optimal_paths[row_index-1, initial_index] #column index
            point2 = (row_index, nextpt, intensities[row_index,nextpt]) # (row index, column index, intensity)

            nextloss = loss(point1, point2, alpha, beta, intensities, row, col)

            # Iterate through potential next points
            for potential_nextpt in optimal_points[row_index,:]:

                point2 = (row_index, potential_nextpt, intensities[row_index, potential_nextpt]) # (row index, column index, intensity)
                curloss = loss(point1, point2, alpha, beta, intensities, row, col)

                if curloss < nextloss: #lower loss
                    nextpt = potential_nextpt
                    nextloss = curloss

            optimal_paths[row_index, initial_index] = nextpt
            aggregate_loss[initial_index] += nextloss

    # Select optimal path based on path with lowest aggregate loss
    optimal_index = np.argmin(aggregate_loss)
    optimal_path = optimal_paths[:,optimal_index]

    master_matrix.append(optimal_path)

    # Overlay path on image

    # im = plt.imread(filename)
    # implot = plt.imshow(im)
    # plt.scatter(x=optimal_path, y=range(row), c='r', s=40)
    # plt.show()

    # sys.exit(1)

    # plt.plot(optimal_path, range(row), 'o')
    # #plt.plot(range(row), optimal_path) #Turn 90 degrees, swap axes
    # plt.show()

    # img.show()

def naive_algo(row, col, intensities, img, filename):
    optimal_path = np.zeros(row)

    # Naive Algorithm; choosest highest intensity (most white) pixel per row/time interval
    for row_index in range(row):
        optimal_path[row_index] = np.argmax(intensities[row_index,:])

    # Overlay path on image
    im = plt.imread(filename)
    implot = plt.imshow(im)
    plt.scatter(x=optimal_path, y=range(row), c='r', s=40)
    plt.show()

    sys.exit(1)

    plt.plot(optimal_path, range(row), 'o')
    plt.show()

    img.show()

    # 0 = black, 255 = white

def main():

    count = 1
    for fn in os.listdir("./SquiggleExamples/"):
        if "png" in fn:
            print(count)
            count += 1

            filename = "SquiggleExamples/" + fn
            img = Image.open(filename)

            # Convert to grayscale if necessary
            img = img.convert("L")

            col,row =  img.size

            intensities = np.zeros((row, col))
            pixels = img.load()

            # print(row, col)
            # Dimensions: 768 by 129
            for i in range(row):
                for j in range(col):
                    #print(i, j)
                    intensity =  pixels[j,i]
                    intensities[i,j] = intensity

            '''
            plt.imshow(intensities)
            plt.show()
            # Median filtering of intensities
            intensities = filters.gaussian_filter(intensities, 0.01)
            plt.imshow(intensities)
            plt.show()
            sys.exit(1)
            '''

            #naive_algo(row, col, intensities, img, filename)
            dynamic_algo(row, col, intensities, img, filename)

    np.savetxt(output_file, master_matrix)

if __name__ == "__main__":
    main()

