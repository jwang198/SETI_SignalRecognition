import matplotlib.pyplot as py  
from numpy import *  
import numpy as np
import csv

def hurst(p):  
    tau = []; lagvec = []  
    #  Step through the different lags  
    for lag in range(2,20):  
        #  produce price difference with lag 
        pp = subtract(p[lag:],p[:-lag])  
        #  Write the different lags into a vector  
        lagvec.append(lag)  
        #  Calculate the variance of the differnce vector  
        tau.append(sqrt(std(pp)))  
    #  linear fit to double-log graph (gives power)  
    m = polyfit(log10(lagvec),log10(tau),1)  
    # calculate hurst  
    hurst = m[0]*2  
    # plot lag vs variance  
    #py.plot(lagvec,tau,'o'); show()  
    return hurst  

with open('./DATA/Squiggles(833)/COMPLETE_squiggle_dft.csv', 'rU') as csvinput:
    with open('./DATA/Squiggles(833)/COMPLETE_squiggle_dft_hurst.csv', 'w') as csvoutput:
            reader = csv.DictReader(csvinput)
            fieldnames = reader.fieldnames + ['Hurst']

            writer = csv.DictWriter(csvoutput, fieldnames = fieldnames)
            writer.writeheader()

            rows = []
            for row in reader:
                x = []
                #print row
                for i in range(1, 64):
                    x.append(row['X' + str(i)])
                x = np.array([float(i) for i in x])
                row['Hurst'] = hurst(x)
                rows.append(row)

            writer.writerows(rows)

with open('./DATA/Squiggles(833)/COMPLETE_squiggle_raw.csv', 'rU') as csvinput:
    with open('./DATA/Squiggles(833)/COMPLETE_squiggle_raw_hurst.csv', 'w') as csvoutput:
            reader = csv.DictReader(csvinput)
            fieldnames = reader.fieldnames + ['Hurst']

            writer = csv.DictWriter(csvoutput, fieldnames = fieldnames)
            writer.writeheader()

            rows = []
            for row in reader:
                x = []
                for i in range(1, 130):
                    x.append(row['X' + str(i)])
                x = np.array([float(i) for i in x])
                row['Hurst'] = hurst(x)
                rows.append(row)

            writer.writerows(rows)

with open('./DATA/Squiggles(833)/COMPLETE_unknown_raw.csv', 'rU') as csvinput:
    with open('./DATA/Squiggles(833)/COMPLETE_unknown_raw_hurst.csv', 'w') as csvoutput:
            reader = csv.DictReader(csvinput)
            fieldnames = reader.fieldnames + ['Hurst']

            writer = csv.DictWriter(csvoutput, fieldnames = fieldnames)
            writer.writeheader()

            rows = []
            for row in reader:
                x = []
                for i in range(1, 130):
                    x.append(row['X' + str(i)])
                x = np.array([float(i) for i in x])
                row['Hurst'] = hurst(x)
                rows.append(row)

            writer.writerows(rows)