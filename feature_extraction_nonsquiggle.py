import numpy as np
import sys

data1 = "/Users/Jason/Desktop/SETI_TimeSeries/DATA/nonsquiggle_ts(unknown).txt"
#data2 = "/Users/Jason/Desktop/SETI_TimeSeries/DATA/ts_params_nonsquiggle.txt"

def parseData(filename, datapts, losses, ids):
    f = open(filename)
    for line in f:
        if "wat" in line or len(line.split())==1: #ID
            ids.append(line.strip())
            continue

        line = line.strip()
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
parseData(data1, datapts, losses, ids)

# Normalize ts to mean 0: subtract mean of each row
datapts = np.array(datapts).astype(float)
mean = np.mean(datapts,axis=1) #row means
for i in range(len(datapts)):
    datapts[i,:] = datapts[i,:].astype(float) - mean[i]

#raw_mean = []
raw_variance = []

temp = np.array(datapts)
for i, ts in enumerate(temp):
    #raw_mean.append(np.mean(ts))
    raw_variance.append(np.var(ts))

'''
arima = []
# Extract arima(1,1,1) params
f = open(data2)
f.readline()
for line in f:
    line = line.strip().split(",")
    arima.append([float(line[1]) if line[1] != 'NA' else 0.0, float(line[2]) if line[2] != 'NA' else 0.0])
'''

#Ouput features as ts_dataset.csv
output_file = open("official_ts_dataset_nonsquiggle.csv", 'w')
output_file.write("\"id\",\"loss\",\"raw_variance\",{0}\n".format(",".join(["\"X" + str(x) + "\"" for x in range(1,130)])))
for i, ID in enumerate(ids):
  output_file.write("\"{0}\",{1},{2},{3}\n".format(ID, losses[i], raw_variance[i],",".join([str(x) for x in list(datapts[i])])))

output_file.close()

#Ouput features as ts_dataset_dft.csv
n=129

fTransformed = abs(np.array([np.fft.fft(row, n) for row in datapts]))
fTransformed = fTransformed[:,1:n//2]

#Ouput features as ts_dataset.csv
output_file = open("official_ts_dataset_nonsquiggle_dft.csv", 'w')
output_file.write("\"id\",\"loss\",\"raw_variance\",{0}\n".format(",".join(["\"X" + str(x) + "\"" for x in range(1,n//2)])))
for i, ID in enumerate(ids):
  output_file.write("\"{0}\",{1},{2},{3}\n".format(ID, losses[i], raw_variance[i],",".join([str(x) for x in list(fTransformed[i])])))

output_file.close()
