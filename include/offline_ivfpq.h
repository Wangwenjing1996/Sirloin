#ifndef OFFLINE_IVFPQ_H
#define OFFLINE_IVFPQ_H


#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"


void offline_ivfpq( int dim, int nb, int nq, float *xb, float *xq, 
                    int nlist, int nprobe, int k, int m, float *&scores) {
    
    faiss::IndexFlatL2 quantizer(dim); // the other index
    faiss::IndexIVFPQ index(&quantizer, dim, nlist, m, 8);

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

        index.nprobe = nprobe;
        index.search(nq, xq, k, D, I);

        // printf("I=\n");  
        // for (int i = nq - 5; i < nq; i++) {
        //     for (int j = 0; j < k; j++)
        //         printf("%5zd ", I[i * k + j]);
        //     printf("\n");
        // }

        // printf("D=\n");
        // for (int i = nq - 5; i < nq; i++) {
        //     for (int j = 0; j < k; j++)
        //         printf("%5f ", D[i * k + j]);
        //     printf("\n");
        // }

        for( int i = 0; i < nq; i++) {
            scores[i] = D[k*(i+1)-1];
        }
        delete[] I;
        delete[] D;
    }
}


#endif