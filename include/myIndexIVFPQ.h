#ifndef MY_IVFPQ_H
#define MY_IVFPQ_H


#include "myIndexFlatL2.h"
#include "faiss/IndexIVFPQ.h"
#include "myIndexIVF.h"
#include "faiss/utils/distances.h"
#include "faiss/impl/code_distance/code_distance.h"
#include <faiss/utils/hamming.h>
#include "myInvertedLists.h"


namespace faiss {


struct myIndexIVFPQ : IndexIVFPQ, myIndexIVF {

    std::vector<float> cnt_codes;

    explicit myIndexIVFPQ( myIndexFlatL2 *quantizer, size_t d, size_t nlist,
                            size_t M, size_t nbits_per_idx)
            : IndexIVF( quantizer, d, nlist, 0),
              IndexIVFPQ( quantizer, d, nlist, M, nbits_per_idx), 
              myIndexIVF( quantizer, d, nlist, 0) {
        cnt_codes.resize( pq.M * pq.ksub, 0);
        delete invlists;
        invlists = new myInvertedLists( nlist, nbits_per_idx);
    }

    void initial_batch( idx_t n, const float *x, const idx_t *xids);

    void update_batch( idx_t n, const float *x, const idx_t *xids, int nlist_ba);

    void update_count( idx_t n);

    void update_codebook( idx_t n, const float *x);

    void delete_expired( idx_t horizon_bound);

    InvertedListScanner* get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel) const override;
};


void myIndexIVFPQ::update_codebook( idx_t n, const float *x) {
    std::vector<float> distortions( pq.M * pq.ksub * pq.dsub, 0);
    // distortion
    for( int i = 0; i < n; i++) {
        const float *cur_vec = x + i*quantizer->d;
        for( int j = 0; j < pq.M; j++) {
            uint8_t code = flat_codes[i*pq.M + j];
            const float *cur_subvec = cur_vec + pq.dsub * j;
            float *sub_center = pq.get_centroids( j, code);
            float *cur_distortion = distortions.data() + ( j*pq.ksub + code) * pq.dsub;
            for( int t = 0; t < pq.dsub; t++) {
                cur_distortion[t] += cur_subvec[t] - sub_center[t];
            }
        }
    }
    for( int m = 0; m < pq.M; m++) {
        for( int i = 0; i < pq.ksub; i++) {
            float *sub_center = pq.get_centroids( m, i);
            for( int t = 0; t < pq.dsub; t++) {
                sub_center[t] += distortions[(m*pq.ksub+i)*pq.dsub] / cnt_codes[m*pq.ksub+i];
            }
        }
    }
}


void myIndexIVFPQ::initial_batch( idx_t n, const float *x, const idx_t *xids) {
    train( n, x);
    my_add_init_q1( n, x, xids);
    update_count( n);
}


void myIndexIVFPQ::delete_expired( idx_t horizon_bound) {
    int cnt_cents = quantizer->ntotal;
    std::vector<float> ori_centroids( cnt_cents * d);
    for( int i = 0; i < cnt_cents; i++) {
        quantizer->reconstruct( i, ori_centroids.data() + i*d);
    }
    quantizer->reset();
    std::vector<float> cnts(cnt_codes.size(), 0);
    for( int i = nlist - 1; i >= 0; i--) {
        // if( invlists->list_size(i) > 0 && invlists->get_ids(i)[0] < horizon_bound) {
        bool is_empty = invlists->delete_oldest( i, horizon_bound, cnts.data(), pq.M, pq.ksub);
        if( is_empty) {
            c_size.erase( c_size.begin() + i);
            c_radius.erase( c_radius.begin() + i);
            ori_centroids.erase( ori_centroids.begin() + i * d, ori_centroids.begin() + (i + 1) * d);
        }
        // }
    }
    nlist = invlists->nlist;
    quantizer->add( nlist, ori_centroids.data());
    for( int i = 0; i < cnt_codes.size(); i++) {
      cnt_codes[i] -= cnts[i];
    }
}

void myIndexIVFPQ::update_batch( idx_t n, const float *x, const idx_t *xids, int nlist_ba) {
    auto q1_start = std::chrono::high_resolution_clock::now();
    my_update_q1( n, x, xids, nlist_ba);
    auto q1_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> q1_duration = q1_end - q1_start;
    std::cout << "\tUpdate q1 time: " << q1_duration.count() << " seconds" << std::endl;
    auto cnt_start = std::chrono::high_resolution_clock::now();
    update_count( n);
    auto cnt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cnt_duration = cnt_end - cnt_start;
    std::cout << "\tUpdate count time: " << cnt_duration.count() << " seconds" << std::endl;
    auto cd_start = std::chrono::high_resolution_clock::now();
    update_codebook( n, x);
    auto cd_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cd_duration = cd_end - cd_start;
    std::cout << "\tUpdate codebook time: " << cd_duration.count() << " seconds" << std::endl;
}


void myIndexIVFPQ::update_count( idx_t n) {
    for( int i = 0; i < n; i++) {
        for( int j = 0; j < pq.M; j++) {
            uint8_t code = flat_codes[i*pq.M + j];
            cnt_codes[j*pq.ksub + code]++;
        }
    }
}


namespace {

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

/** QueryTables manages the various ways of searching an
 * IndexIVFPQ. The code contains a lot of branches, depending on:
 * - metric_type: are we computing L2 or Inner product similarity?
 * - by_residual: do we encode raw vectors or residuals?
 * - use_precomputed_table: are x_R|x_C tables precomputed?
 * - polysemous_ht: are we filtering with polysemous codes?
 */
struct myQueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const myIndexIVFPQ& ivfpq;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQ for easier access
    int d;
    const ProductQuantizer& pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;
    int polysemous_ht;

    // pre-allocated data buffers
    float *sim_table, *sim_table_2;
    float *residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float*> sim_table_ptrs;

    explicit myQueryTables(
            const myIndexIVFPQ& ivfpq,
            const IVFSearchParameters* params)
            : ivfpq(ivfpq),
              d(ivfpq.d),
              pq(ivfpq.pq),
              metric_type(ivfpq.metric_type),
              by_residual(ivfpq.by_residual),
              use_precomputed_table(ivfpq.use_precomputed_table) {
        mem.resize(pq.ksub * pq.M * 2 + d * 2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        polysemous_ht = ivfpq.polysemous_ht;
        if (auto ivfpq_params =
                    dynamic_cast<const IVFPQSearchParameters*>(params)) {
            polysemous_ht = ivfpq_params->polysemous_ht;
        }
        if (polysemous_ht != 0) {
            q_code.resize(pq.code_size);
        }
        init_list_cycles = 0;
        sim_table_ptrs.resize(pq.M);
    }

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float* qi;

    // query-specific initialization
    void init_query(const float* qi) {
        this->qi = qi;
        if (metric_type == METRIC_INNER_PRODUCT)
            init_query_IP();
        else
            init_query_L2();
        if (!by_residual && polysemous_ht != 0)
            pq.compute_code(qi, q_code.data());
    }

    void init_query_IP() {
        // precompute some tables specific to the query qi
        pq.compute_inner_prod_table(qi, sim_table);
    }

    void init_query_L2() {
        if (!by_residual) {
            pq.compute_distance_table(qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table(qi, sim_table_2);
        }
    }

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    idx_t key;
    float coarse_dis;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                dis0 = precompute_list_tables_IP();
            else
                dis0 = precompute_list_tables_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    float precompute_list_table_pointers() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                FAISS_THROW_MSG("not implemented");
            else
                dis0 = precompute_list_table_pointers_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    /*****************************************************
     * compute tables for inner prod
     *****************************************************/

    float precompute_list_tables_IP() {
        // prepare the sim_table that will be used for accumulation
        // and dis0, the initial value
        ivfpq.quantizer->reconstruct(key, decoded_vec);
        // decoded_vec = centroid
        float dis0 = fvec_inner_product(qi, decoded_vec, d);

        if (polysemous_ht) {
            for (int i = 0; i < d; i++) {
                residual_vec[i] = qi[i] - decoded_vec[i];
            }
            pq.compute_code(residual_vec, q_code.data());
        }
        return dis0;
    }

    /*****************************************************
     * compute tables for L2 distance
     *****************************************************/

    float precompute_list_tables_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 0 || use_precomputed_table == -1) {
            ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);

            if (polysemous_ht != 0) {
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd(
                    pq.M * pq.ksub,
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M,
                    -2.0,
                    sim_table_2,
                    sim_table);

            if (polysemous_ht != 0) {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            const float* qtab = sim_table_2; // query-specific table
            float* ltab = sim_table;         // (output) list-specific table

            long k = key;
            for (int cm = 0; cm < cpq.M; cm++) {
                // compute PQ index
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                // get corresponding table
                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                if (polysemous_ht == 0) {
                    // sum up with query-specific table
                    fvec_madd(Mf * pq.ksub, pc, -2.0, qtab, ltab);
                    ltab += Mf * pq.ksub;
                    qtab += Mf * pq.ksub;
                } else {
                    for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
                        q_code[m] = fvec_madd_and_argmin(
                                pq.ksub, pc, -2, qtab, ltab);
                        pc += pq.ksub;
                        ltab += pq.ksub;
                        qtab += pq.ksub;
                    }
                }
            }
        }

        return dis0;
    }

    float precompute_list_table_pointers_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            const float* s =
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs[m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                for (int m = m0; m < m0 + Mf; m++) {
                    sim_table_ptrs[m] = pc;
                    pc += pq.ksub;
                }
                m0 += Mf;
            }
        } else {
            FAISS_THROW_MSG("need precomputed tables");
        }

        if (polysemous_ht) {
            FAISS_THROW_MSG("not implemented");
            // Not clear that it makes sense to implemente this,
            // because it costs M * ksub, which is what we wanted to
            // avoid with the tables pointers.
        }

        return dis0;
    }
};

// This way of handling the selector is not optimal since all distances
// are computed even if the id would filter it out.
template <class C, bool use_sel>
struct KnnSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // heap params
    size_t k;
    float* heap_sim;
    idx_t* heap_ids;

    size_t nup;

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        if (C::cmp(heap_sim[0], dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            heap_replace_top<C>(k, heap_sim, heap_ids, dis, id);
            nup++;
        }
    }
};

template <class C, bool use_sel>
struct myKnnSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // heap params
    size_t k;
    float* heap_sim;
    idx_t* heap_ids;
    float* heap_q2c;
    float* heap_o2c;

    size_t nup;

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis, float q2c, float o2c) {
        if (C::cmp(heap_sim[0], dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            my_heap_replace_top<C>(k, heap_sim, heap_ids, heap_q2c, heap_o2c, dis, id, q2c, o2c);
            nup++;
        }
    }
};

// template <class C, bool use_sel>
// struct RangeSearchResults {
//     idx_t key;
//     const idx_t* ids;
//     const IDSelector* sel;

//     // wrapped result structure
//     float radius;
//     RangeQueryResult& rres;

//     inline bool skip_entry(idx_t j) {
//         return use_sel && !sel->is_member(ids[j]);
//     }

//     inline void add(idx_t j, float dis) {
//         if (C::cmp(radius, dis)) {
//             idx_t id = ids ? ids[j] : lo_build(key, j);
//             rres.add(dis, id);
//         }
//     }
// };

/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/
template <typename IDType, MetricType METRIC_TYPE, class PQDecoder>
struct myIVFPQScannerT : myQueryTables {
    const uint8_t* list_codes;
    const IDType* list_ids;
    size_t list_size;

    myIVFPQScannerT(const myIndexIVFPQ& ivfpq, const IVFSearchParameters* params)
            : myQueryTables(ivfpq, params) {
        assert(METRIC_TYPE == metric_type);
    }

    float dis0;

    void init_list(idx_t list_no, float coarse_dis, int mode) {
        this->key = list_no;
        this->coarse_dis = coarse_dis;

        if (mode == 2) {
            dis0 = precompute_list_tables();
        } else if (mode == 1) {
            dis0 = precompute_list_table_pointers();
        }
    }

    /*****************************************************
     * Scaning the codes: simple PQ scan.
     *****************************************************/

    // This is the baseline version of scan_list_with_tables().
    // It demonstrates what this function actually does.
    //
    // /// version of the scan where we use precomputed tables.
    // template <class SearchResultType>
    // void scan_list_with_table(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         SearchResultType& res) const {
    //
    //     for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
    //         if (res.skip_entry(j)) {
    //             continue;
    //         }
    //         float dis = dis0 + distance_single_code<PQDecoder>(
    //             pq, sim_table, codes);
    //         res.add(j, dis);
    //     }
    // }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    /// version of the scan where we use precomputed tables.
    template <class SearchResultType>
    void scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis);
        }
    }

    template <class SearchResultType>
    void my_scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            float q2c,
            const float* o2c,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0, q2c, o2c[saved_j[0]]);
                res.add(saved_j[1], dis0 + distance_1, q2c, o2c[saved_j[1]]);
                res.add(saved_j[2], dis0 + distance_2, q2c, o2c[saved_j[2]]);
                res.add(saved_j[3], dis0 + distance_3, q2c, o2c[saved_j[3]]);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis, q2c, o2c[saved_j[0]]);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis, q2c, o2c[saved_j[1]]);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis, q2c, o2c[saved_j[2]]);
        }
    }

    template <class SearchResultType>
    void my_scan_list_with_table_pruning(
            size_t ncode,
            const uint8_t* codes,
            float q2c,
            const float* o2c,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            // saved_j[0] = (counter == 0) ? j : saved_j[0];
            // saved_j[1] = (counter == 1) ? j : saved_j[1];
            // saved_j[2] = (counter == 2) ? j : saved_j[2];
            // saved_j[3] = (counter == 3) ? j : saved_j[3];
            if( res.heap_sim[0] <= fabs(q2c-o2c[j])) {
                saved_j[0] = (counter == 0) ? j : saved_j[0];
                saved_j[1] = (counter == 1) ? j : saved_j[1];
                saved_j[2] = (counter == 2) ? j : saved_j[2];
                saved_j[3] = (counter == 3) ? j : saved_j[3];
                counter += 1;
            }

            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0, q2c, o2c[saved_j[0]]);
                res.add(saved_j[1], dis0 + distance_1, q2c, o2c[saved_j[1]]);
                res.add(saved_j[2], dis0 + distance_2, q2c, o2c[saved_j[2]]);
                res.add(saved_j[3], dis0 + distance_3, q2c, o2c[saved_j[3]]);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis, q2c, o2c[saved_j[0]]);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis, q2c, o2c[saved_j[1]]);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis, q2c, o2c[saved_j[2]]);
        }
    }

    /// tables are not precomputed, but pointers are provided to the
    /// relevant X_c|x_r tables
    template <class SearchResultType>
    void scan_list_with_pointer(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            PQDecoder decoder(codes, pq.nbits);
            float dis = dis0;
            const float* tab = sim_table_2;

            for (size_t m = 0; m < pq.M; m++) {
                int ci = decoder.decode();
                dis += sim_table_ptrs[m][ci] - 2 * tab[ci];
                tab += pq.ksub;
            }
            res.add(j, dis);
        }
    }

    /// nothing is precomputed: access residuals on-the-fly
    template <class SearchResultType>
    void scan_on_the_fly_dist(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        const float* dvec;
        float dis0 = 0;
        if (by_residual) {
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                ivfpq.quantizer->reconstruct(key, residual_vec);
                dis0 = fvec_inner_product(residual_vec, qi, d);
            } else {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            }
            dvec = residual_vec;
        } else {
            dvec = qi;
            dis0 = 0;
        }

        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            pq.decode(codes, decoded_vec);

            float dis;
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                dis = dis0 + fvec_inner_product(decoded_vec, qi, d);
            } else {
                dis = fvec_L2sqr(decoded_vec, dvec, d);
            }
            res.add(j, dis);
        }
    }

    /*****************************************************
     * Scanning codes with polysemous filtering
     *****************************************************/

    // This is the baseline version of scan_list_polysemous_hc().
    // It demonstrates what this function actually does.

    //     template <class HammingComputer, class SearchResultType>
    //     void scan_list_polysemous_hc(
    //             size_t ncode,
    //             const uint8_t* codes,
    //             SearchResultType& res) const {
    //         int ht = ivfpq.polysemous_ht;
    //         size_t n_hamming_pass = 0, nup = 0;
    //
    //         int code_size = pq.code_size;
    //
    //         HammingComputer hc(q_code.data(), code_size);
    //
    //         for (size_t j = 0; j < ncode; j++, codes += code_size) {
    //             if (res.skip_entry(j)) {
    //                 continue;
    //             }
    //             const uint8_t* b_code = codes;
    //             int hd = hc.hamming(b_code);
    //             if (hd < ht) {
    //                 n_hamming_pass++;
    //
    //                 float dis =
    //                         dis0 +
    //                         distance_single_code<PQDecoder>(
    //                             pq, sim_table, codes);
    //
    //                 res.add(j, dis);
    //             }
    //         }
    // #pragma omp critical
    //         { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    //     }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    template <class HammingComputer, class SearchResultType>
    void scan_list_polysemous_hc(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int ht = ivfpq.polysemous_ht;
        size_t n_hamming_pass = 0;

        int code_size = pq.code_size;

        size_t saved_j[8];
        int counter = 0;

        HammingComputer hc(q_code.data(), code_size);

        for (size_t j = 0; j < (ncode / 4) * 4; j += 4) {
            const uint8_t* b_code = codes + j * code_size;

            // Unrolling is a key. Basically, doing multiple popcount
            // operations one after another speeds things up.

            // 9999999 is just an arbitrary large number
            int hd0 = (res.skip_entry(j + 0))
                    ? 99999999
                    : hc.hamming(b_code + 0 * code_size);
            int hd1 = (res.skip_entry(j + 1))
                    ? 99999999
                    : hc.hamming(b_code + 1 * code_size);
            int hd2 = (res.skip_entry(j + 2))
                    ? 99999999
                    : hc.hamming(b_code + 2 * code_size);
            int hd3 = (res.skip_entry(j + 3))
                    ? 99999999
                    : hc.hamming(b_code + 3 * code_size);

            saved_j[counter] = j + 0;
            counter = (hd0 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 1;
            counter = (hd1 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 2;
            counter = (hd2 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 3;
            counter = (hd3 < ht) ? (counter + 1) : counter;

            if (counter >= 4) {
                // process four codes at the same time
                n_hamming_pass += 4;

                float distance_0 = dis0;
                float distance_1 = dis0;
                float distance_2 = dis0;
                float distance_3 = dis0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);

                //
                counter -= 4;
                saved_j[0] = saved_j[4];
                saved_j[1] = saved_j[5];
                saved_j[2] = saved_j[6];
                saved_j[3] = saved_j[7];
            }
        }

        for (size_t kk = 0; kk < counter; kk++) {
            n_hamming_pass++;

            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[kk] * pq.code_size);

            res.add(saved_j[kk], dis);
        }

        // process leftovers
        for (size_t j = (ncode / 4) * 4; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }
            const uint8_t* b_code = codes + j * code_size;
            int hd = hc.hamming(b_code);
            if (hd < ht) {
                n_hamming_pass++;

                float dis = dis0 +
                        distance_single_code<PQDecoder>(
                                    pq.M,
                                    pq.nbits,
                                    sim_table,
                                    codes + j * code_size);

                res.add(j, dis);
            }
        }

#pragma omp critical
        { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    }

    template <class SearchResultType>
    struct Run_scan_list_polysemous_hc {
        using T = void;
        template <class HammingComputer, class... Types>
        void f(const myIVFPQScannerT* scanner, Types... args) {
            scanner->scan_list_polysemous_hc<HammingComputer, SearchResultType>(
                    args...);
        }
    };

    template <class SearchResultType>
    void scan_list_polysemous(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        Run_scan_list_polysemous_hc<SearchResultType> r;
        dispatch_HammingComputer(pq.code_size, r, this, ncode, codes, res);
    }
};

/* We put as many parameters as possible in template. Hopefully the
 * gain in runtime is worth the code bloat.
 *
 * C is the comparator < or >, it is directly related to METRIC_TYPE.
 *
 * precompute_mode is how much we precompute (2 = precompute distance tables,
 * 1 = precompute pointers to distances, 0 = compute distances one by one).
 * Currently only 2 is supported
 *
 * use_sel: store or ignore the IDSelector
 */
template <MetricType METRIC_TYPE, class C, class PQDecoder, bool use_sel>
struct myIVFPQScanner : myIVFPQScannerT<idx_t, METRIC_TYPE, PQDecoder>,
                      InvertedListScanner {
    int precompute_mode;
    const IDSelector* sel;

    myIVFPQScanner(
            const myIndexIVFPQ& ivfpq,
            bool store_pairs,
            int precompute_mode,
            const IDSelector* sel)
            : myIVFPQScannerT<idx_t, METRIC_TYPE, PQDecoder>(ivfpq, nullptr),
              precompute_mode(precompute_mode),
              sel(sel) {
        this->store_pairs = store_pairs;
        this->keep_max = is_similarity_metric(METRIC_TYPE);
    }

    size_t my_scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float q2c,
            const float* o2c,
            float* heap_sim,
            idx_t* heap_ids,
            float* heap_q2c,
            float* heap_o2c,
            size_t k) const override {
        myKnnSearchResults<C, use_sel> res = {
            /* key */ this->key,
            /* ids */ this->store_pairs ? nullptr : ids,
            /* sel */ this->sel,
            /* k */ k,
            /* heap_sim */ heap_sim,
            /* heap_ids */ heap_ids,
            /* heap_q2c */ heap_q2c,
            /* heap_o2c */ heap_o2c,
            /* nup */ 0};
        assert( precompute_mode == 2);
        this->my_scan_list_with_table(ncode, codes, q2c, o2c, res);
    }

    size_t my_scan_codes_pruning(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float q2c,
            const float* o2c,
            float* heap_sim,
            idx_t* heap_ids,
            float* heap_q2c,
            float* heap_o2c,
            size_t k) const override {
        myKnnSearchResults<C, use_sel> res = {
            /* key */ this->key,
            /* ids */ this->store_pairs ? nullptr : ids,
            /* sel */ this->sel,
            /* k */ k,
            /* heap_sim */ heap_sim,
            /* heap_ids */ heap_ids,
            /* heap_q2c */ heap_q2c,
            /* heap_o2c */ heap_o2c,
            /* nup */ 0};
        assert( precompute_mode == 2);
        this->my_scan_list_with_table_pruning(ncode, codes, q2c, o2c, res);
    }

    void set_query(const float* query) override {
        this->init_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->init_list(list_no, coarse_dis, precompute_mode);
    }

    float distance_to_code(const uint8_t* code) const override {
        assert(precompute_mode == 2);
        float dis = this->dis0 +
                distance_single_code<PQDecoder>(
                            this->pq.M, this->pq.nbits, this->sim_table, code);
        return dis;
    }

    size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* heap_sim,
            idx_t* heap_ids,
            size_t k) const override {
        KnnSearchResults<C, use_sel> res = {
                /* key */ this->key,
                /* ids */ this->store_pairs ? nullptr : ids,
                /* sel */ this->sel,
                /* k */ k,
                /* heap_sim */ heap_sim,
                /* heap_ids */ heap_ids,
                /* nup */ 0};

        if (this->polysemous_ht > 0) {
            assert(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            this->scan_list_with_table(ncode, codes, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }
        return res.nup;
    }

    // void scan_codes_range(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         const idx_t* ids,
    //         float radius,
    //         RangeQueryResult& rres) const override {
    //     RangeSearchResults<C, use_sel> res = {
    //             /* key */ this->key,
    //             /* ids */ this->store_pairs ? nullptr : ids,
    //             /* sel */ this->sel,
    //             /* radius */ radius,
    //             /* rres */ rres};

    //     if (this->polysemous_ht > 0) {
    //         assert(precompute_mode == 2);
    //         this->scan_list_polysemous(ncode, codes, res);
    //     } else if (precompute_mode == 2) {
    //         this->scan_list_with_table(ncode, codes, res);
    //     } else if (precompute_mode == 1) {
    //         this->scan_list_with_pointer(ncode, codes, res);
    //     } else if (precompute_mode == 0) {
    //         this->scan_on_the_fly_dist(ncode, codes, res);
    //     } else {
    //         FAISS_THROW_MSG("bad precomp mode");
    //     }
    // }
};

template <class PQDecoder, bool use_sel>
InvertedListScanner* my_get_InvertedListScanner1(
        const myIndexIVFPQ& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.metric_type == METRIC_INNER_PRODUCT) {
        return new myIVFPQScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    } else if (index.metric_type == METRIC_L2) {
        return new myIVFPQScanner<
                METRIC_L2,
                CMax<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    }
    return nullptr;
}

template <bool use_sel>
InvertedListScanner* my_get_InvertedListScanner2(
        const myIndexIVFPQ& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.pq.nbits == 8) {
        return my_get_InvertedListScanner1<PQDecoder8, use_sel>(
                index, store_pairs, sel);
    } else if (index.pq.nbits == 16) {
        return my_get_InvertedListScanner1<PQDecoder16, use_sel>(
                index, store_pairs, sel);
    } else {
        return my_get_InvertedListScanner1<PQDecoderGeneric, use_sel>(
                index, store_pairs, sel);
    }
}

} // anonymous namespace


InvertedListScanner* myIndexIVFPQ::get_InvertedListScanner(
    bool store_pairs,
    const IDSelector* sel) const {
    if (sel) {
        return my_get_InvertedListScanner2<true>(*this, store_pairs, sel);
    } else {
        return my_get_InvertedListScanner2<false>(*this, store_pairs, sel);
    }
    return nullptr;
}


}


#endif