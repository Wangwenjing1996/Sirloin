# Sirloin
This is the source code of Sirloin.

Our paper, "Streaming Time Series Subsequence Anomaly Detection: A Glance and Focus Approach", is submitted to VLDB2025.

## Directory description
  * data: The dataset required for Sirloin has the following format: the first row indicates the length of the time series, the first column contains the data points, and the second column is the labels (where 0 represents normal and 1 represents an anomaly).
  * faiss-main: 

## Linux build
We implement Sirloin in GCC 8.3.1, and all experiments are run on a Linux machine with an Intel Xeon Gold 622R @ 2.90GHz processor and 92GB memory. 

### Build steps
1. Download this repository and change to the Sirloin folder.

2. Create a "build" directory inside it: `mkdir build`

3. Change to the "build" directory: `cd build`

4. And run: `cmake ..`

5. `make`

6. `./main` or `cd ..` then `bash run.sh`
