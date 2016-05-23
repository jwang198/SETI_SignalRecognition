import numpy as np
import sys

data1 = "/Users/Jason/Desktop/SETI_TimeSeries/DATA/nonsquiggle_ts.txt"
#data2 = "/Users/Jason/Desktop/SETI_TimeSeries/DATA/ts_params.txt"

def parseData(filename, datapts, losses, ids):
    f = open(filename)
    for line in f:
        if "2014" in line or "2013" in line: #ID
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
parseData(data1, datapts, losses, ids)

#Ouput features as ts_dataset.csv
output_file = open("official_ts_dataset_nonsquiggle.csv", 'w')
output_file.write("\"id\",\"loss\",{0}\n".format(",".join(["\"X" + str(x) + "\"" for x in range(1,130)])))
for i, ID in enumerate(ids):
  output_file.write("\"{0}\",{1},{2}\n".format(ID, losses[i],",".join([str(x) for x in list(datapts[i])])))

output_file.close()

#Ouput features as ts_dataset_dft.csv
n=129

fTransformed = abs(np.array([np.fft.fft(row, n) for row in datapts]))
fTransformed = fTransformed[:,1:n//2]

#Ouput features as ts_dataset.csv
output_file = open("official_ts_dataset_nonsquiggle_dft.csv", 'w')
output_file.write("\"id\",\"loss\",{0}\n".format(",".join(["\"X" + str(x) + "\"" for x in range(1,n//2)])))
for i, ID in enumerate(ids):
  output_file.write("\"{0}\",{1},{2}\n".format(ID, losses[i],",".join([str(x) for x in list(fTransformed[i])])))

output_file.close()


#print(datapts)
#print(losses)
#print(ids)


