#ifndef MY_INDEX_IVF_H
#define MY_INDEX_IVF_H


#include <omp.h>
#include <inttypes.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/CodePacker.h>
#include "myIndexFlatL2.h"
#include "faiss/IndexIVF.h"
#include "faiss/Clustering.h"
#include "faiss/IndexIVFFlat.h"
#include <vector>
#include <float.h>
#include <faiss/utils/utils.h>
#include <mutex>
#include <faiss/impl/AuxIndexStructures.h>
#include "myHeap.h"


namespace faiss {


struct myIndexIVF : virtual IndexIVF {

    std::vector<float> c_radius;
    std::vector<size_t> c_size;
    std::vector<uint8_t> flat_codes;

    explicit myIndexIVF( myIndexFlatL2 *quantizer, size_t d, size_t nlist, size_t code_size)
            : IndexIVF( quantizer, d, nlist, code_size) {
        c_radius.resize( nlist, 0);
        c_size.resize( nlist, 0);
    }

    void my_add_init_q1( idx_t n, const float *x, const idx_t *xids);

    void my_update_q1( idx_t n, const float *x, const idx_t *xids, int nlist_ba);

    void my_update_q1_optimize( idx_t n, const float *x, const idx_t *xids, int nlist_ba, 
                                float *new_distances, idx_t *new_coarse_idx, float *new_centroids);

    void my_add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        float* coarse_dists,
        std::vector<uint8_t> &flat_codes,
        void* inverted_list_context = nullptr);

    void my_add_core_wo_encode(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        float* coarse_dists,
        std::vector<uint8_t> &flat_codes,
        void* inverted_list_context = nullptr);

    void my_search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            idx_t* self_idx,
            float* scores,
            const SearchParameters* params = nullptr);

    void my_search_optimize(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            idx_t* self_idx,
            float* scores,
            myIndexIVF &batch_index,
            const SearchParameters* params = nullptr);
    
    void my_search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            float* dists_q2c,
            float* dists_o2c,
            idx_t* selfid,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr);
    
    void my_search_preassigned_optimize(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            const idx_t* assign_new,
            const float* centroid_dis_new,
            float* distances,
            idx_t* labels,
            float* dists_q2c,
            float* dists_o2c,
            idx_t* selfid,
            bool store_pairs,
            myIndexIVF &batch_index,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr);
};


void myIndexIVF::my_add_init_q1( idx_t n, const float *x, const idx_t *xids) {
    // compute distances and assign
    std::vector<float> distances(n);
    std::vector<idx_t> coarse_idx(n);
    quantizer->search( n, x, 1, distances.data(), coarse_idx.data());
    // add data to inverted list
    flat_codes.clear();
    flat_codes.resize(n * code_size);
    my_add_core(n, x, xids, coarse_idx.data(), distances.data(), flat_codes);
    // compute cluster radius and size
    for( int i = 0; i < n; i++) {
        if( c_radius[coarse_idx[i]] < sqrt(distances[i])) {
            c_radius[coarse_idx[i]] = sqrt(distances[i]);
        }
    }
    for( size_t i = 0; i < nlist; i++) {
        c_size[i] = invlists->list_size(i);
    }
}


void myIndexIVF:: my_update_q1_optimize( idx_t n, const float *x, const idx_t *xids, int nlist_ba,
                                     float *new_distances, idx_t *new_coarse_idx, float *new_centroids) {
    std::vector<float> new_radius(nlist_ba, 0.0f);
    std::vector<int> new_size(nlist_ba, 0);
    for( int i = 0; i < n; i++) {
        if( new_radius[new_coarse_idx[i]] < sqrt(new_distances[i])) {
            new_radius[new_coarse_idx[i]] = sqrt(new_distances[i]);
        }
        new_size[new_coarse_idx[i]]++;
    }

    int cnt_cents = quantizer->ntotal;
    std::cout << cnt_cents << " " << quantizer->ntotal << " " << nlist << std::endl;
    std::vector<float> ori_centroids( cnt_cents * d);
    for( int i = 0; i < cnt_cents; i++) {
        quantizer->reconstruct( i, ori_centroids.data() + i*d);
    }

    int increase = 0;
    std::vector<float> new2old_dists(nlist_ba);
    std::vector<idx_t> new2old_idx(nlist_ba);
    std::vector<idx_t> new2merge_idx(nlist_ba);
    quantizer->search( nlist_ba, new_centroids, 1, new2old_dists.data(), new2old_idx.data());
    std::vector<std::vector<int>> to_add(cnt_cents);
    std::vector<int> new_c;
    for( int i = 0; i < nlist_ba; i++) {
        size_t list_no = new2old_idx[i];
        if( sqrt(new2old_dists[i]) >= c_radius[list_no]) {
            new_c.push_back(i);
            new2merge_idx[i] = cnt_cents + increase;
            increase++;
        }
        else {
            to_add[list_no].push_back(i);
            new2merge_idx[i] = list_no;
        }
    }
    std::cout << increase << std::endl;

    std::vector<float> merge_centroids( (cnt_cents+increase) * d);
    for( int i = 0; i < cnt_cents; i++) {
        int offset = i * d;
        memcpy(merge_centroids.data()+offset, ori_centroids.data()+offset, sizeof(float) * d);
        if( to_add[i].size() > 0) {
            float w_sum = 1.0f;
            c_size[i] = invlists->list_size(i);
            size_t tmp_size = c_size[i];
            for( int j = 0; j < to_add[i].size(); j++) {
                int cid_in_new = to_add[i][j];
                float tmp_w = 1.0f * new_size[cid_in_new] / c_size[i];
                tmp_size += new_size[cid_in_new];
                c_radius[i] += tmp_w * new_radius[cid_in_new];
                for( int t = 0; t < d; t++) {
                    merge_centroids[offset+t] += tmp_w * new_centroids[cid_in_new*d+t];
                }
                w_sum += tmp_w;
            }
            c_size[i] = tmp_size;
            c_radius[i] /= w_sum;
            for( int t = offset; t < offset + d; t++) {
                merge_centroids[t] /= w_sum;
            }
        }
    }

    c_size.resize(cnt_cents+increase);
    c_radius.resize(cnt_cents+increase);
    for( int i = 0; i < increase; i++) {
        int cid_in_new = new_c[i];
        memcpy(merge_centroids.data()+(cnt_cents+i)*d, new_centroids+cid_in_new*d, sizeof(float) * d);
        c_size[cnt_cents+i] = new_size[cid_in_new];
        c_radius[cnt_cents+i] = new_radius[cid_in_new];
    }
    cnt_cents += increase;
    quantizer->reset();
    quantizer->add( cnt_cents, merge_centroids.data());

    invlists->expand( cnt_cents);
    nlist = cnt_cents;

    std::vector<idx_t> merge_coarse_idx(n);
    for( int i = 0; i < n; i++) {
        merge_coarse_idx[i] = new2merge_idx[new_coarse_idx[i]];
    }

    my_add_core_wo_encode( n, x, xids, merge_coarse_idx.data(), new_distances, flat_codes);
}


void myIndexIVF::my_update_q1( idx_t n, const float *x, const idx_t *xids, int nlist_ba) {

    // cluster a batch
    auto clus_start = std::chrono::high_resolution_clock::now();
    size_t d = quantizer->d;
    IndexFlatL2 batch_quantizer( d);
    Clustering clus( d, nlist_ba, cp);
    clus.train( n, x, batch_quantizer);
    auto clus_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> clus_duration = clus_end - clus_start;
    std::cout << "\t\tCluster batch time: " << clus_duration.count() << " seconds" << std::endl;

    // compute distances and assign
    auto check_start = std::chrono::high_resolution_clock::now();
    std::vector<float> new_distances(n);
    std::vector<idx_t> new_coarse_idx(n);
    batch_quantizer.search( n, x, 1, new_distances.data(), new_coarse_idx.data());
    // compute radius and size
    std::vector<float> new_radius(nlist_ba, 0.0f);
    std::vector<int> new_size(nlist_ba, 0);
    for( int i = 0; i < n; i++) {
        if( new_radius[new_coarse_idx[i]] < sqrt(new_distances[i])) {
            new_radius[new_coarse_idx[i]] = sqrt(new_distances[i]);
        }
        new_size[new_coarse_idx[i]]++;
    }
    auto check_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> check_duration = check_end - check_start;
    std::cout << "\t\tCompute clusters dist time: " << check_duration.count() << " seconds" << std::endl;

    auto other_start = std::chrono::high_resolution_clock::now();
    // merge
    int cnt_cents = quantizer->ntotal;
    std::cout << cnt_cents << " " << quantizer->ntotal << " " << nlist << std::endl;
    float *new_centroids = clus.centroids.data();
    std::vector<float> ori_centroids( cnt_cents * d);
    for( int i = 0; i < cnt_cents; i++) {
        quantizer->reconstruct( i, ori_centroids.data() + i*d);
    }
    
    int increase = 0;
    std::vector<float> new2old_dists(nlist_ba);
    std::vector<idx_t> new2old_idx(nlist_ba);
    std::vector<idx_t> new2merge_idx(nlist_ba);
    quantizer->search( nlist_ba, new_centroids, 1, new2old_dists.data(), new2old_idx.data());
    std::vector<std::vector<int>> to_add(cnt_cents);
    std::vector<int> new_c;
    for( int i = 0; i < nlist_ba; i++) {
        size_t list_no = new2old_idx[i];
        if( sqrt(new2old_dists[i]) >= c_radius[list_no]) {
            new_c.push_back(i);
            new2merge_idx[i] = cnt_cents + increase;
            increase++;
        }
        else {
            to_add[list_no].push_back(i);
            new2merge_idx[i] = list_no;
        }
    }
    std::cout << increase << std::endl;

    // update centroids
    // original
    std::vector<float> merge_centroids( (cnt_cents+increase) * d);
    for( int i = 0; i < cnt_cents; i++) {
        int offset = i * d;
        memcpy(merge_centroids.data()+offset, ori_centroids.data()+offset, sizeof(float) * d);
        if( to_add[i].size() > 0) {
            float w_sum = 1.0f;
            c_size[i] = invlists->list_size(i);
            size_t tmp_size = c_size[i];
            for( int j = 0; j < to_add[i].size(); j++) {
                int cid_in_new = to_add[i][j];
                float tmp_w = 1.0f * new_size[cid_in_new] / c_size[i];
                tmp_size += new_size[cid_in_new];
                c_radius[i] += tmp_w * new_radius[cid_in_new];
                for( int t = 0; t < d; t++) {
                    merge_centroids[offset+t] += tmp_w * new_centroids[cid_in_new*d+t];
                }
                w_sum += tmp_w;
            }
            c_size[i] = tmp_size;
            c_radius[i] /= w_sum;
            for( int t = offset; t < offset + d; t++) {
                merge_centroids[t] /= w_sum;
            }
        }
    }
    int ttttcnt = 0;
    for( int i = 0; i < cnt_cents; i++) {
        if( invlists->list_size(i) == 0) ttttcnt++;
    }
    std::cout << ttttcnt << std::endl;
    // increase
    c_size.resize(cnt_cents+increase);
    c_radius.resize(cnt_cents+increase);
    for( int i = 0; i < increase; i++) {
        int cid_in_new = new_c[i];
        memcpy(merge_centroids.data()+(cnt_cents+i)*d, new_centroids+cid_in_new*d, sizeof(float) * d);
        c_size[cnt_cents+i] = new_size[cid_in_new];
        c_radius[cnt_cents+i] = new_radius[cid_in_new];
    }
    cnt_cents += increase;
    quantizer->reset();
    quantizer->add( cnt_cents, merge_centroids.data());

    // update invlists
    invlists->expand( cnt_cents);
    nlist = cnt_cents;

    std::vector<idx_t> merge_coarse_idx(n);
    for( int i = 0; i < n; i++) {
        merge_coarse_idx[i] = new2merge_idx[new_coarse_idx[i]];
    }
    // std::vector<uint8_t> flat_codes(n * code_size);
    flat_codes.clear();
    flat_codes.resize(n * code_size);
    my_add_core( n, x, xids, merge_coarse_idx.data(), new_distances.data(), flat_codes);
    auto other_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> other_duration = other_end - other_start;
    std::cout << "\t\tOther time: " << other_duration.count() << " seconds" << std::endl;
}


void myIndexIVF::my_add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        float* coarse_dists,
        std::vector<uint8_t> &flat_codes,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0)
            nminus1++;
    }

    // std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.data());

    DirectMapAdd dm_adder(direct_map, n, xids);

    // omp_set_num_threads(1);
#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        flat_codes.data() + i * code_size,
                        &coarse_dists[i]);

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
               nadd,
               n,
               nminus1);
    }

    ntotal += n;
}


void myIndexIVF::my_add_core_wo_encode(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        float* coarse_dists,
        std::vector<uint8_t> &flat_codes,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0)
            nminus1++;
    }

    DirectMapAdd dm_adder(direct_map, n, xids);

    // omp_set_num_threads(1);
#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        flat_codes.data() + i * code_size,
                        &coarse_dists[i]);

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
               nadd,
               n,
               nminus1);
    }

    ntotal += n;
}


void myIndexIVF::my_search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        idx_t* self_idx,
        float* scores,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<float []> dists_q2c( new float [n*k]);
    std::unique_ptr<float []> dists_o2c( new float [n*k]);
    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   float* dists_q2c,
                                   float* dists_o2c,
                                   idx_t* selfid,
                                   float* scores,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        my_search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                dists_q2c,
                dists_o2c,
                selfid,
                false,
                params,
                ivf_stats);
        // score here
        float tmp_inner, sum1, sum2;
        for( int i = 0; i < n; i++) {
            sum1 = 0.0f;
            sum2 = 0.0f;
            for( int j = 0; j < k; j++) {
                tmp_inner = distances[i*k+j] + dists_q2c[i*k+j] - dists_o2c[i*k+j];
                sum1 += tmp_inner;
                sum2 += pow( tmp_inner, 2);
            }
            scores[i] = ( sum2 - pow( sum1, 2) / k) / k;
        }
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            dists_q2c.get() + i0 * k,
                            dists_o2c.get() + i0 * k,
                            self_idx + i0,
                            scores + i0,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        // sub_search_func(n, x, distances, labels, self_idx, &indexIVF_stats);
    }
}


void myIndexIVF::my_search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        float* dists_q2c,
        float* dists_o2c,
        idx_t* selfid,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        // std::unique_ptr<InvertedListScanner> scanner(
        //         get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi, float* q2ci, float* o2ci) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                my_heap_heapify<HeapForIP>(k, simi, idxi, q2ci, o2ci);
            } else {
                my_heap_heapify<HeapForL2>(k, simi, idxi, q2ci, o2ci);
            }
        };

        // auto add_local_results = [&](const float* local_dis,
        //                              const idx_t* local_idx,
        //                              float* simi,
        //                              idx_t* idxi) {
        //     if (metric_type == METRIC_INNER_PRODUCT) {
        //         heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        //     } else {
        //         heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        //     }
        // };

        // auto reorder_result = [&](float* simi, idx_t* idxi) {
        //     if (!do_heap_init)
        //         return;
        //     if (metric_type == METRIC_INNER_PRODUCT) {
        //         heap_reorder<HeapForIP>(k, simi, idxi);
        //     } else {
        //         heap_reorder<HeapForL2>(k, simi, idxi);
        //     }
        // };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 float* q2ci,
                                 float* o2ci,
                                 idx_t list_size_max,
                                 std::unique_ptr<InvertedListScanner> &scanner) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    // size_t list_size = 0;

                    // std::unique_ptr<InvertedListsIterator> it(
                    //         invlists->get_iterator(key, inverted_list_context));

                    // nheap += scanner->iterate_codes(
                    //         it.get(), simi, idxi, k, list_size);

                    // return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    const float* tmp = invlists->get_o2c( key);

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->my_scan_codes(
                            list_size, codes, ids, coarse_dis_i, tmp, simi, idxi, q2ci, o2ci, k);
                    // nheap += scanner->scan_codes(
                    //         list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
// #pragma omp for
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();
            IDSelectorRange *my_sel_sm_yes = new IDSelectorRange( 0, 0, false);
            IDSelectorNot *my_sel_sm_no = new IDSelectorNot( my_sel_sm_yes);
            std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, my_sel_sm_no));
            // for (idx_t i = 0; i < n; i++) {
            for( idx_t i = rank; i < n; i += nt) {
                // if( i % nt != rank) continue;
                if (interrupt) {
                    continue;
                }

                // loop over queries
                idx_t lb = selfid[i] - d;
                idx_t rb = selfid[i] + d + 1;
                my_sel_sm_yes->imin = lb;
                my_sel_sm_yes->imax = rb;
                // IDSelectorRange *my_sel_sm_yes = new IDSelectorRange( lb, rb, false);
                // IDSelectorNot *my_sel_sm_no = new IDSelectorNot( my_sel_sm_yes);
                // std::unique_ptr<InvertedListScanner> scanner(
                //     get_InvertedListScanner(store_pairs, my_sel_sm_no));
                
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
                float* q2ci = dists_q2c + i * k;
                float* o2ci = dists_o2c + i * k;

                init_result(simi, idxi, q2ci, o2ci);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            q2ci,
                            o2ci,
                            max_codes - nscan,
                            scanner);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                // reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } 
//         else if (pmode == 1) {
//             std::vector<idx_t> local_idx(k);
//             std::vector<float> local_dis(k);

//             for (size_t i = 0; i < n; i++) {
//                 scanner->set_query(x + i * d);
//                 init_result(local_dis.data(), local_idx.data());

// #pragma omp for schedule(dynamic)
//                 for (idx_t ik = 0; ik < nprobe; ik++) {
//                     ndis += scan_one_list(
//                             keys[i * nprobe + ik],
//                             coarse_dis[i * nprobe + ik],
//                             local_dis.data(),
//                             local_idx.data(),
//                             unlimited_list_size);

//                     // can't do the test on max_codes
//                 }
//                 // merge thread-local results

//                 float* simi = distances + i * k;
//                 idx_t* idxi = labels + i * k;
// #pragma omp single
//                 init_result(simi, idxi);

// #pragma omp barrier
// #pragma omp critical
//                 {
//                     add_local_results(
//                             local_dis.data(), local_idx.data(), simi, idxi);
//                 }
// #pragma omp barrier
// #pragma omp single
//                 reorder_result(simi, idxi);
//             }
//         } else if (pmode == 2) {
//             std::vector<idx_t> local_idx(k);
//             std::vector<float> local_dis(k);

// #pragma omp single
//             for (int64_t i = 0; i < n; i++) {
//                 init_result(distances + i * k, labels + i * k);
//             }

// #pragma omp for schedule(dynamic)
//             for (int64_t ij = 0; ij < n * nprobe; ij++) {
//                 size_t i = ij / nprobe;

//                 scanner->set_query(x + i * d);
//                 init_result(local_dis.data(), local_idx.data());
//                 ndis += scan_one_list(
//                         keys[ij],
//                         coarse_dis[ij],
//                         local_dis.data(),
//                         local_idx.data(),
//                         unlimited_list_size);
// #pragma omp critical
//                 {
//                     add_local_results(
//                             local_dis.data(),
//                             local_idx.data(),
//                             distances + i * k,
//                             labels + i * k);
//                 }
//             }
// #pragma omp single
//             for (int64_t i = 0; i < n; i++) {
//                 reorder_result(distances + i * k, labels + i * k);
//             }
//         }
         else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}


void myIndexIVF::my_search_optimize(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            idx_t* self_idx,
            float* scores,
            myIndexIVF &batch_index,
            const SearchParameters* params_in) {

    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<float []> dists_q2c( new float [n*k]);
    std::unique_ptr<float []> dists_o2c( new float [n*k]);

    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   float* dists_q2c,
                                   float* dists_o2c,
                                   idx_t* selfid,
                                   float* scores,
                                   IndexIVFStats* ivf_stats,
                                   myIndexIVF &batch_index) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        std::unique_ptr<idx_t[]> idx_new(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis_new(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        batch_index.quantizer->search( n, x, nprobe, coarse_dis_new.get(), idx_new.get(), 
                                        params ? params->quantizer_params : nullptr);

        my_search_preassigned_optimize(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                idx_new.get(),
                coarse_dis_new.get(),
                distances,
                labels,
                dists_q2c,
                dists_o2c,
                selfid,
                false,
                batch_index,
                params,
                ivf_stats);

        // score here
        float tmp_inner, sum1, sum2;
        for( int i = 0; i < n; i++) {
            sum1 = 0.0f;
            sum2 = 0.0f;
            for( int j = 0; j < k; j++) {
                tmp_inner = distances[i*k+j] + dists_q2c[i*k+j] - dists_o2c[i*k+j];
                sum1 += tmp_inner;
                sum2 += pow( tmp_inner, 2);
            }
            scores[i] = ( sum2 - pow( sum1, 2) / k) / k;
        }
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            dists_q2c.get() + i0 * k,
                            dists_o2c.get() + i0 * k,
                            self_idx + i0,
                            scores + i0,
                            &stats[slice],
                            batch_index);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    }
}


void myIndexIVF::my_search_preassigned_optimize(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        const idx_t* keys_new,
        const float* coarse_dis_new,
        float* distances,
        idx_t* labels,
        float* dists_q2c,
        float* dists_o2c,
        idx_t* selfid,
        bool store_pairs,
        myIndexIVF &batch_index,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) {

    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;
    

    #pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        // std::unique_ptr<InvertedListScanner> scanner(
        //         get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi, float* q2ci, float* o2ci) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                my_heap_heapify<HeapForIP>(k, simi, idxi, q2ci, o2ci);
            } else {
                my_heap_heapify<HeapForL2>(k, simi, idxi, q2ci, o2ci);
            }
        };

        // auto add_local_results = [&](const float* local_dis,
        //                              const idx_t* local_idx,
        //                              float* simi,
        //                              idx_t* idxi) {
        //     if (metric_type == METRIC_INNER_PRODUCT) {
        //         heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        //     } else {
        //         heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        //     }
        // };

        // auto reorder_result = [&](float* simi, idx_t* idxi) {
        //     if (!do_heap_init)
        //         return;
        //     if (metric_type == METRIC_INNER_PRODUCT) {
        //         heap_reorder<HeapForIP>(k, simi, idxi);
        //     } else {
        //         heap_reorder<HeapForL2>(k, simi, idxi);
        //     }
        // };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 float* q2ci,
                                 float* o2ci,
                                 idx_t list_size_max,
                                 std::unique_ptr<InvertedListScanner> &scanner) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    // size_t list_size = 0;

                    // std::unique_ptr<InvertedListsIterator> it(
                    //         invlists->get_iterator(key, inverted_list_context));

                    // nheap += scanner->iterate_codes(
                    //         it.get(), simi, idxi, k, list_size);

                    // return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    const float* tmp = invlists->get_o2c( key);

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    // nheap += scanner->my_scan_codes(
                            // list_size, codes, ids, coarse_dis_i, tmp, simi, idxi, q2ci, o2ci, k);

                    nheap += scanner->my_scan_codes_pruning(
                            list_size, codes, ids, coarse_dis_i, tmp, simi, idxi, q2ci, o2ci, k);
                    // nheap += scanner->scan_codes(
                    //         list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        auto scan_one_list_externel = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 float* q2ci,
                                 float* o2ci,
                                 idx_t list_size_max,
                                 std::unique_ptr<InvertedListScanner> &scanner,
                                 myIndexIVF &batch_index) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (batch_index.invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (batch_index.invlists->use_iterator) {
                    // size_t list_size = 0;

                    // std::unique_ptr<InvertedListsIterator> it(
                    //         invlists->get_iterator(key, inverted_list_context));

                    // nheap += scanner->iterate_codes(
                    //         it.get(), simi, idxi, k, list_size);

                    // return list_size;
                } else {
                    size_t list_size = batch_index.invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(batch_index.invlists, key);
                    const uint8_t* codes = scodes.get();

                    const float* tmp = batch_index.invlists->get_o2c( key);

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                batch_index.invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->my_scan_codes(
                            list_size, codes, ids, coarse_dis_i, tmp, simi, idxi, q2ci, o2ci, k);
                    // nheap += scanner->scan_codes(
                    //         list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
// #pragma omp for
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();
            IDSelectorRange *my_sel_sm_yes = new IDSelectorRange( 0, 0, false);
            IDSelectorNot *my_sel_sm_no = new IDSelectorNot( my_sel_sm_yes);
            std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, my_sel_sm_no));
            // for (idx_t i = 0; i < n; i++) {
            for( idx_t i = rank; i < n; i += nt) {
                // if( i % nt != rank) continue;
                if (interrupt) {
                    continue;
                }

                // loop over queries
                idx_t lb = selfid[i] - d;
                idx_t rb = selfid[i] + d + 1;
                my_sel_sm_yes->imin = lb;
                my_sel_sm_yes->imax = rb;
                // IDSelectorRange *my_sel_sm_yes = new IDSelectorRange( lb, rb, false);
                // IDSelectorNot *my_sel_sm_no = new IDSelectorNot( my_sel_sm_yes);
                // std::unique_ptr<InvertedListScanner> scanner(
                //     get_InvertedListScanner(store_pairs, my_sel_sm_no));
                
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
                float* q2ci = dists_q2c + i * k;
                float* o2ci = dists_o2c + i * k;

                init_result(simi, idxi, q2ci, o2ci);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list_externel(
                            keys_new[i * nprobe + ik],
                            coarse_dis_new[i * nprobe + ik],
                            simi,
                            idxi,
                            q2ci,
                            o2ci,
                            max_codes - nscan,
                            scanner,
                            batch_index
                    );
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            q2ci,
                            o2ci,
                            max_codes - nscan,
                            scanner);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                // external index here

                ndis += nscan;
                // reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } 
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}


}


#endif