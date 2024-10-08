#include <iostream>
#include <random>
#include "example.h"
#include "dataio.h"
#include "offline_ivfpq.h"
#include "offline_ivfpq_mogai.h"
#include "online_ivfpq_baoli.h"
#include "online_ivfpq_baseline.h"
#include "online_ivfpq_tmp.h"
#include "online_ivfpq_optimize.h"
#include "evaluate.h"


int main( int argc, char *argv[]) {
    
    const char *path = "../data/Dodger/101-freeway-traffic.test.out";


    float *ts;
    short *labels;
    int len_ts;
    int dim = 64;
    float *x_data;
    int nlist = 100;
    int nprobe = 10;
    int k = 25;
    int M = 8;
    int b = 5000;
    if( argc == 3) {
        path = argv[1];
        dim = std::atoi(argv[2]);
        std::cout << "data: " << path << " d: " << dim << std::endl;
    }
    else if( argc != 1) {
        std::cout << "wrong args" << std::endl;
    }
    dataloader( path, ts, labels, len_ts);
    std::cout << "read series of length: " << len_ts << std::endl;
    int n_data = len_ts - dim + 1;
    preprocess( x_data, ts, dim, n_data);
    std::cout << "test" << std::endl;
    float *scores;
    int H = n_data / 2;
    online_ivfpq_baseline( dim, n_data, x_data, nlist, nprobe, k, M, scores, b, H);
    // online_ivfpq_optimize( dim, n_data, x_data, nlist, nprobe, k, M, scores, b, H);


    float recall = get_precision( scores, labels, n_data, dim);
    std::cout << "recall: " << recall << std::endl;

    delete [] ts;
    delete [] labels;
    delete [] x_data;
    delete [] scores;

    return 0;
}