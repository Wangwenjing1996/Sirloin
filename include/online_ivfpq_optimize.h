#ifndef ONLINE_IVFPQ_OPTIMIZE_H
#define ONLINE_IVFPQ_OPTIMIZE_H


#include "myIndexFlatL2.h"
#include "myIndexIVFPQ.h"
#include <iostream>
#include <assert.h>
#include <chrono>


void online_ivfpq_optimize( int dim, int nb, float *xb, int nlist, int nprobe, int k, int m, 
                            float *&scores, int batch_size, int horizon_size) {
    std::cout << "optimize version" << std::endl;
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
            faiss::idx_t *I = new faiss::idx_t[k*nb_ba];
            float *D = new float [k*nb_ba];
            auto _start = std::chrono::high_resolution_clock::now();
            // encode the batch by old_index
            index.flat_codes.clear();
            index.flat_codes.resize( nb_ba * index.code_size);
            index.encode_vectors( nb_ba, xb_ba, nullptr, index.flat_codes.data());
            // update count
            index.update_count( nb_ba);
            // update codebook
            index.update_codebook( nb_ba, xb_ba);
            // declare an myIVFPQ new_index
            faiss::myIndexFlatL2 batch_quantizer(dim); // the other index
            faiss::myIndexIVFPQ batch_index(&batch_quantizer, dim, nlist, m, 8);
            batch_index.by_residual = false;
            batch_index.nprobe = nprobe;
            // cluster the batch
            faiss::Clustering clus( dim, nlist, index.cp);
            clus.train( nb_ba, xb_ba, batch_quantizer);
            batch_index.is_trained = true;
            std::vector<float> new_distances(nb_ba);
            std::vector<faiss::idx_t> new_coarse_idx(nb_ba);
            auto repeat_start = std::chrono::high_resolution_clock::now();
            batch_quantizer.search( nb_ba, xb_ba, 1, new_distances.data(), new_coarse_idx.data());
            auto repeat_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> repeat_duration = repeat_end - repeat_start;
            // copy the codebook to new_index
            memcpy( batch_index.pq.centroids.data(), index.pq.centroids.data(), sizeof(float) * index.pq.ksub * dim);
            // add codes to new_index without encode!!!
            batch_index.my_add_core_wo_encode( nb_ba, xb_ba, ids + i, new_coarse_idx.data(), new_distances.data(), index.flat_codes);
            // search new_index return distances!!!
            // batch_index.my_search(nb_ba, xb_ba, k, D, I, ids + i, scores + i);
            // search old_index
            // index.my_search(nb_ba, xb_ba, k, D, I, ids + i, scores + i);

            index.my_search_optimize( nb_ba, xb_ba, k, D, I, ids + i, scores + i, batch_index);

            auto _end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> _duration = _end - _start;
            std::cout << "Optimize time: " << _duration.count() - repeat_duration.count() << " seconds" << std::endl;
            // compute scores
            // ========== latency ==========
            // merge clusters and add codes to old_index without encode
            index.my_update_q1_optimize( nb_ba, xb_ba, ids + i, nlist, new_distances.data(), new_coarse_idx.data(), clus.centroids.data());
            // delete
            // index.my_search(nb_ba, xb_ba, k, D, I, ids + i, scores + i);
            index.delete_expired( lb);
            
            delete [] I;
            delete [] D;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}


#endif