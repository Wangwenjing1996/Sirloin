# Sirloin
This is the source code of Sirloin.

Our paper, "Streaming Time Series Subsequence Anomaly Detection: A Glance and Focus Approach", is submitted to VLDB2025.

## Directory description
  * data: The dataset required for Sirloin has the following format: the first row indicates the length of the time series, the first column contains the data points, and the second column is the labels (where 0 represents normal and 1 represents an anomaly).
  * faiss-main and include: The source code for Sirloin.

## Linux build
We implement Sirloin in GCC 8.3.1, and all experiments are run on a Linux machine with an Intel Xeon Gold 622R @ 2.90GHz processor and 92GB memory. 

### Prerequisites
 * Ubuntu 22.04.02 LTS
 * GCC 8.3.1
 * CMake 3.24.0
 * BLAS
   
   `$ sudo apt-get update`
   
   `$ sudo apt-get install libblas-dev liblapack-dev`

### Build and run steps
 * Download this repository and change to the Sirloin folder.

   `$ cd Sirloin`
 * Create a "build" directory inside it.

   `$ mkdir build`
 * Change to the "build" directory.

   `$ cd build`
 * Run.

   `$ cmake ..`
   
   `$ make`
   
   `$ ./main` or `$ cd ..` then `$ bash run.sh`
