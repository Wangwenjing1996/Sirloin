#ifndef EVALUATE_H
#define EVALUATE_H


#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>


bool cmp_score(std::pair<float, int> a, std::pair<float, int> b) {
    return a.first > b.first;
}


float get_precision( float *score, short *label, int n_data, int ts_length) {
    int *cnt_point2event = new int [n_data]();
    bool in_event = false;
    int n_anomaly = 0;
    std::vector<int> begins;
    std::vector<int> ends;
    for( int i = 0; i < n_data; i++) {
        if( !in_event && label[i] == 0) {
            continue;
        }
        else if( in_event && label[i] == 1) {
            cnt_point2event[i]++;
            continue;
        }
        else if( in_event && label[i] == 0) {
            in_event = false;
            ends.push_back( i-1);
        }
        else if( !in_event && label[i] == 1) {
            n_anomaly++;
            in_event = true;
            int j;
            for( j = i; j > i - ts_length && j >= 0; j--) {
                cnt_point2event[j]++;
            }
            begins.push_back( j+1);
        }
    }
    if( in_event) {
        ends.push_back( n_data-1);
    }
    // for( int i = 0; i < n_anomaly; i++) {
    //     printf( "(%d, %d) ", begins[i], ends[i]);
    // }
    // printf("\n");

    std::pair<float, int> *tmp = new std::pair<float, int> [n_data];
    for( int i = 0; i < n_data; i++) {
        tmp[i] = std::make_pair( score[i], i);
    }
    std::sort( tmp, tmp + n_data, cmp_score);

    int cnt_total = 0;
    int cnt_hit = 0;
    short *vis_point = new short [n_data]();
    short *vis_event = new short [n_anomaly]();
    for( int i = 0; i < n_data; i++) {
        if( cnt_total == n_anomaly) {
            break;
        }
        int idx = tmp[i].second;
        if( vis_point[idx] == 1) {
            continue;
        }
        if( cnt_point2event[idx] > 0) {
            for( int j = 0; j < n_anomaly; j++) {
                if( vis_event[j] == 1) {
                    continue;
                }
                if( idx < begins[j] || idx > ends[j]) {
                    continue;
                }
                vis_event[j] = 1;
                cnt_total++;
                cnt_hit++;
                for( int k = begins[j]; k <= ends[j]; k++) {
                    if( cnt_point2event[k] == 1) {
                        vis_point[k] = 1;
                    }
                    cnt_point2event[k]--;
                }
            }
        }
        else {
            int lb = std::max( idx - (int) ts_length, 0);
            int ub = std::min( idx + (int) ts_length, (int) n_data - 1);
            for( int j = lb; j <= ub; j++) {
                vis_point[j] = 1;
            }
            cnt_total++;
        }
    }

    delete [] cnt_point2event;
    delete [] tmp;
    delete [] vis_point;
    delete [] vis_event;

    return 1.0 * cnt_hit / cnt_total;
}

#endif