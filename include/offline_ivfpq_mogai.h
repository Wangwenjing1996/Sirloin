#ifndef OFFLINE_IVFPQ_MOGAI_H
#define OFFLINE_IVFPQ_MOGAI_H


#include "myIndexFlatL2.h"
#include "myIndexIVFPQ.h"
#include <iostream>


void offline_ivfpq_mogai( int dim, int nb, int nq, float *xb, float *xq, 
                        int nlist, int nprobe, int k, int m, float *&scores) {
    
    faiss::myIndexFlatL2 quantizer(dim); // the other index
    faiss::myIndexIVFPQ index(&quantizer, dim, nlist, m, 8);

    quantizer.verbose = true;
    index.verbose = true;
    // index.by_residual = false;
    index.by_residual = true;
    index.train( nb, xb);
    index.add( nb, xb);

    scores = new float [nq];

    { // search xq
        faiss::idx_t* I = new faiss::idx_t[k * nq];
        float* D = new float[k * nq];

        // index.nprobe = nprobe;
        index.search(nq, xq, k, D, I);

        for( int i = 0; i < nq; i++) {
            scores[i] = D[k*(i+1)-1];
        }
        delete[] I;
        delete[] D;
    }
}


#endif