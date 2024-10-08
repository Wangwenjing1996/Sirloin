#ifndef MY_INDEX_IVF_FLAT_H
#define MY_INDEX_IVF_FLAT_H


#include "faiss/IndexFlat.h"
#include "faiss/Clustering.h"


namespace faiss {


struct myIndexFlatL2 : IndexFlatL2 {
    
    explicit myIndexFlatL2( idx_t d) : IndexFlatL2( d) {}
    // myIndexFlatL2() {}

    // void update_q1( size_t n, const float *x);
};


}


#endif