#ifndef ONLINE_IVFPQ_BAOLI_H
#define ONLINE_IVFPQ_BAOLI_H


#include "myIndexFlatL2.h"
#include "myIndexIVFPQ.h"
#include <iostream>


void online_ivfpq_baoli( int dim, int nb, float *xb, int nlist, int nprobe, int k, int m, 
                        float *&scores, int batch_size, int horizon_size) {

    // faiss::myIndexFlatL2 quantizer(dim); // the other index
    // faiss::myIndexIVFPQ index(&quantizer, dim, nlist, m, 8);
    scores = new float [nb];

    int cnt_ba = 0;
    for( int i = 0; i < nb; i += batch_size) {
        int rb = std::min( i + batch_size, nb);
        int lb = std::max( 0, rb - horizon_size);
        std::cout << "the " << cnt_ba++ << "-th batch from "
         << i << " to " << rb << " window from " << lb << std::endl;
        int nb_ho = rb - lb;
        float *xb_ho = xb + lb * dim;
        // int nb_ho = rb - i - dim + 1;
        // float *xb_ho = xb + i * dim;
        int nq_ba = rb - i;
        float *xq_ba = xb + i * dim;
        if( i == 0) {
            faiss::IndexFlatL2 quantizer(dim); // the other index
            faiss::IndexIVFPQ index(&quantizer, dim, nlist, m, 8);
            index.train( nb_ho, xb_ho);
            index.add( nb_ho, xb_ho);

            faiss::idx_t *I = new faiss::idx_t[k*nq_ba];
            float *D = new float [k*nq_ba];
            index.nprobe = nprobe;
            // index.nprobe = nlist;
            index.search( nq_ba, xq_ba, k, D, I);
            for( int j = 0; j < nq_ba; j++) {
                scores[i+j] = D[k*(j+1)-1];
            }
            delete [] I;
            delete [] D;
        }
        else {
            faiss::IndexFlatL2 quantizer(dim); // the other index
            faiss::IndexIVFPQ index(&quantizer, dim, nlist, m, 8);
            index.train( nb_ho, xb_ho);
            index.add( nb_ho, xb_ho);

            faiss::idx_t *I = new faiss::idx_t[k*nq_ba];
            float *D = new float [k*nq_ba];
            index.nprobe = nprobe;
            // index.nprobe = nlist;
            index.search( nq_ba, xq_ba, k, D, I);
            for( int j = 0; j < nq_ba; j++) {
                scores[i+j] = D[k*(j+1)-1];
            }
            delete [] I;
            delete [] D;
        }
    }
}


#endif