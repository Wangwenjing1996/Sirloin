#ifndef ONLINE_IVFPQ_BASELINE_H
#define ONLINE_IVFPQ_BASELINE_H


#include "myIndexFlatL2.h"
#include "myIndexIVFPQ.h"
#include <iostream>
#include <assert.h>
#include <chrono>


void online_ivfpq_baseline( int dim, int nb, float *xb, int nlist, int nprobe, int k, int m, 
                            float *&scores, int batch_size, int horizon_size) {

    faiss::myIndexFlatL2 quantizer(dim); // the other index
    faiss::myIndexIVFPQ index(&quantizer, dim, nlist, m, 8);
    scores = new float [nb];
    faiss::idx_t *ids = new faiss::idx_t [nb];
    for( int i = 0; i < nb; i++) {
        ids[i] = i;
    }
    index.by_residual = false;

    auto start = std::chrono::high_resolution_clock::now();
    int cnt_ba = 0;
    for( int i = 0; i < nb; i += batch_size) {
        int rb = std::min( i + batch_size, nb);
        int lb = std::max( 0, rb - horizon_size);
        if( i == 0) {
            rb = horizon_size;
        }
        std::cout << "the " << cnt_ba++ << "-th batch from "
         << i << " to " << rb << " window from " << lb << std::endl;
        int nb_ba = rb - i;
        float *xb_ba = xb + i * dim;
        if( i == 0) {
            index.initial_batch( nb_ba, xb_ba, ids);

            faiss::idx_t *I = new faiss::idx_t[k*nb_ba];
            float *D = new float [k*nb_ba];
            index.nprobe = nprobe;
            // index.nprobe = nlist / 10;
            // index.nprobe = index.nlist;
            // index.search( nb_ba, xb_ba, k, D, I);
            index.my_search( nb_ba, xb_ba, k, D, I, ids, scores);
            index.delete_expired( lb);
            // for( int j = 0; j < k; j++) {
            //     std::cout << I[j] << std::endl;
            // }
            for( int j = 0; j < nb_ba; j++) {
                // scores[i+j] = D[k*(j+1)-1];
                for( int t = 0; t < k; t++) {
                    assert( I[j*k+t] < ids[j] - dim || I[j*k+t] > ids[j] + dim);
                }
            }
            delete [] I;
            delete [] D;
            // exit(-1);
            i = horizon_size - batch_size;
        }
        else {
            auto update_start = std::chrono::high_resolution_clock::now();
            index.update_batch( nb_ba, xb_ba, ids + i, nlist);
            auto update_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> update_duration = update_end - update_start;
            std::cout << "Update time: " << update_duration.count() << " seconds" << std::endl;

            faiss::idx_t *I = new faiss::idx_t[k*nb_ba];
            float *D = new float [k*nb_ba];
            index.nprobe = nprobe;
            // index.nprobe = nlist / 10;
            // index.nprobe = index.nlist;
            // index.search( nb_ba, xb_ba, k, D, I);
            auto search_start = std::chrono::high_resolution_clock::now();
            index.my_search( nb_ba, xb_ba, k, D, I, ids + i, scores + i);
            auto search_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> search_duration = search_end - search_start;
            std::cout << "Search time: " << search_duration.count() << " seconds" << std::endl;
            auto delete_start = std::chrono::high_resolution_clock::now();
            index.delete_expired( lb);
            auto delete_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> delete_duration = delete_end - delete_start;
            std::cout << "Delete time: " << delete_duration.count() << " seconds" << std::endl;
            for( int j = 0; j < nb_ba; j++) {
                // scores[i+j] = D[k*(j+1)-1];
                for( int t = 0; t < k; t++) {
                    assert( I[j*k+t] < ids[i+j] - dim || I[j*k+t] > ids[i+j] + dim);
                }
            }
            delete [] I;
            delete [] D;
            std::cout << "\tTotal time: " << delete_duration.count() + search_duration.count() + update_duration.count() << " seconds" << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}


#endif