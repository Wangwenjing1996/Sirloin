#ifndef ONLINE_IVFPQ_TMP_H
#define ONLINE_IVFPQ_TMP_H


#include "myIndexFlatL2.h"
#include "myIndexIVFPQ.h"
#include <iostream>


void online_ivfpq_tmp( int dim, int nb, float *xb, int nlist, int nprobe, int k, int m, 
                        float *&scores, int batch_size, int horizon_size) {

    scores = new float [nb];
    faiss::idx_t *ids = new faiss::idx_t [nb];
    for( int i = 0; i < nb; i++) {
        ids[i] = i;
    }

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
            faiss::myIndexFlatL2 quantizer(dim); // the other index
            faiss::myIndexIVFPQ index(&quantizer, dim, nlist, m, 8);
            index.by_residual = false;

            faiss::idx_t *I = new faiss::idx_t[k*nq_ba];
            float *D = new float [k*nq_ba];
            
            index.initial_batch( nb_ho, xb_ho, ids+i);
            index.nprobe = nprobe;
            index.my_search( nb_ho, xb_ho, k, D, I, ids+i, scores+i);
            // for( int j = 0; j < nq_ba; j++) {
            //     float max_v = 0;
            //     for( int t = 0; t < k; t++) {
            //         max_v = D[k*j+t] > max_v ? D[k*j+t] : max_v;
            //     }
            //     scores[i+j] = max_v;
            // }
            delete [] I;
            delete [] D;
        }
        else {
            faiss::myIndexFlatL2 quantizer(dim); // the other index
            faiss::myIndexIVFPQ index(&quantizer, dim, nlist, m, 8);
            index.by_residual = false;

            faiss::idx_t *I = new faiss::idx_t[k*nq_ba];
            float *D = new float [k*nq_ba];
            
            index.initial_batch( nb_ho, xb_ho, ids+i);
            index.nprobe = nprobe;
            index.my_search( nq_ba, xq_ba, k, D, I, ids+i, scores+i);
            // for( int j = 0; j < nq_ba; j++) {
            //     float max_v = 0;
            //     for( int t = 0; t < k; t++) {
            //         max_v = D[k*j+t] > max_v ? D[k*j+t] : max_v;
            //     }
            //     scores[i+j] = max_v;
            // }
            delete [] I;
            delete [] D;
        }
    }
}


#endif