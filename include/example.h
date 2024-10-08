#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <random>
#include <iostream>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

void example() {
    int d = 64;      // dimension
    int nb = 10000; // database size
    int nq = 1000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;
    int k = 4;
    int m = 8;                       // bytes per vector
    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);

    index.train(nb, xb);
    index.add(nb, xb);
    
    std::cout << "added!" << std::endl;

    { // search xq
        faiss::idx_t* I = new faiss::idx_t[k * nq];
        float* D = new float[k * nq];

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;
}


#endif