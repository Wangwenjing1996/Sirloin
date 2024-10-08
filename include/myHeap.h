#include <faiss/utils/ordered_key_value.h>
#include "faiss/utils/Heap.h"


template <class C>
inline void my_heap_push(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_q2c,
        typename C::T* bh_o2c,
        typename C::T val,
        typename C::TI id,
        typename C::T q2c,
        typename C::T o2c) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    bh_q2c--;
    bh_o2c--;
    size_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(val, bh_val[i_father], id, bh_ids[i_father])) {
            /* the heap structure is ok */
            break;
        }
        bh_val[i] = bh_val[i_father];
        bh_ids[i] = bh_ids[i_father];
        bh_q2c[i] = bh_q2c[i_father];
        bh_o2c[i] = bh_o2c[i_father];
        i = i_father;
    }
    bh_val[i] = val;
    bh_ids[i] = id;
    bh_q2c[i] = q2c;
    bh_o2c[i] = o2c;
}


template <class C>
inline void my_heap_heapify(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_q2c,
        typename C::T* bh_o2c,
        const typename C::T* x = nullptr,
        const typename C::TI* ids = nullptr,
        const typename C::T* q2c = nullptr,
        const typename C::T* o2c = nullptr,
        size_t k0 = 0) {
    if (k0 > 0)
        assert(x);

    if (ids) {
        for (size_t i = 0; i < k0; i++)
            my_heap_push<C>(i + 1, bh_val, bh_ids, bh_q2c, bh_o2c, x[i], ids[i], q2c[i], o2c[i]);
    } else {
        for (size_t i = 0; i < k0; i++)
            my_heap_push<C>(i + 1, bh_val, bh_ids, bh_q2c, bh_o2c, x[i], i, q2c[i], o2c[i]);
    }

    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
        bh_q2c[i] = C::neutral();
        bh_o2c[i] = C::neutral();
    }
}


template <class C>
inline void my_heap_replace_top(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_q2c,
        typename C::T* bh_o2c,
        typename C::T val,
        typename C::TI id,
        typename C::T q2c,
        typename C::T o2c) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    bh_q2c--;
    bh_o2c--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k) {
            break;
        }

        // Note that C::cmp2() is a bool function answering
        // `(a1 > b1) || ((a1 == b1) && (a2 > b2))` for max
        // heap and same with the `<` sign for min heap.
        if ((i2 == k + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            bh_q2c[i] = bh_q2c[i1];
            bh_o2c[i] = bh_o2c[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            bh_q2c[i] = bh_q2c[i2];
            bh_o2c[i] = bh_o2c[i2];
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
    bh_q2c[i] = q2c;
    bh_o2c[i] = o2c;
}