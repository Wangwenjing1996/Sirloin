#ifndef MY_INVERTED_LISTS_H
#define MY_INVERTED_LISTS_H


#include <faiss/invlists/InvertedLists.h>
#include <assert.h>
#include <string.h>


namespace faiss {


struct myInvertedLists : ArrayInvertedLists {

    std::vector<std::vector<float>> dists_o2c;

    myInvertedLists( size_t nlist, size_t code_size)
        : ArrayInvertedLists( nlist, code_size) {
        dists_o2c.resize( nlist);
    }

    const float* get_o2c( size_t list_no) const override {
        assert(list_no < nlist);
        return dists_o2c[list_no].data();
    }

    void resize( size_t list_no, size_t new_size) override {
        ids[list_no].resize(new_size);
        codes[list_no].resize(new_size * code_size);
        dists_o2c[list_no].resize(new_size);
    }

    void expand( size_t new_nlist) override {
        nlist = new_nlist;
        ids.resize(new_nlist);
        codes.resize(new_nlist);
        dists_o2c.resize(new_nlist);
    }

    size_t add_entry(
        size_t list_no,
        idx_t theid,
        const uint8_t* code,
        void* dists) override {
        assert(list_no < nlist);
        size_t o = ids[list_no].size();
        resize( list_no, o+1);
        memcpy(&ids[list_no][o], &theid, sizeof(theid));
        memcpy(&codes[list_no][o * code_size], code, code_size);
        memcpy(&dists_o2c[list_no][o], dists, sizeof(float));
    }

    bool delete_oldest( size_t list_no, size_t bound, float *cnts, size_t M, size_t K) override {
        if( ids[list_no][0] >= bound) {
            return false;
        }
        auto vec_iter = std::lower_bound( ids[list_no].begin(), ids[list_no].end(), bound);
        // assert( vec_iter != ids[list_no].begin());
        if( vec_iter == ids[list_no].end()) {
            nlist--;
            for( int i = 0; i < ids[list_no].size(); i++) {
                for( int j = 0; j < M; j++) {
                    uint8_t code = codes[list_no][i*M+j];
                    cnts[j*K+code]++;
                }
            }
            ids.erase( ids.begin() + list_no);
            codes.erase( codes.begin() + list_no);
            dists_o2c.erase( dists_o2c.begin() + list_no);
            return true;
        }
        // for( auto it = ids[list_no].begin(); it < vec_iter; it++) {
        //     assert( *it < bound);
        // }
        ids[list_no].erase( ids[list_no].begin(), vec_iter);
        int num_del = vec_iter - ids[list_no].begin();
        for( int i = 0; i < num_del; i++) {
            for( int j = 0; j < M; j++) {
                uint8_t code = codes[list_no][i*M+j];
                cnts[j*K+code]++;
            }
        }
        codes[list_no].erase( codes[list_no].begin(), codes[list_no].begin() + num_del * code_size);
        dists_o2c[list_no].erase( dists_o2c[list_no].begin(), dists_o2c[list_no].begin() + num_del);
        return false;
    }
};


}


#endif