#ifndef DATA_H
#define DATA_H


#include <cstdio>
#include <iostream>


bool dataloader( const char *path, float *& ts, short *& label, int &n) {
    FILE *fp;
    fp = fopen( path, "r");
    if( fp == NULL) {
        std::cerr << "cannot open " << path << std::endl;
        return false;
    }
    
    fscanf(fp, "%d", &n);
    ts = new float [n];
    label = new short [n];

    for( int i = 0; i < n; i++) {
        fscanf( fp, "%f,%hd", &ts[i], &label[i]);
    }

    fclose( fp);

    return true;
}


void preprocess( float *& x_data, float *ts, int dim, int n_data) {
    x_data = new float [n_data * dim];
    for( int i = 0; i < n_data; i++) {
        for( int j = 0; j < dim; j++) {
            x_data[i*dim + j] = ts[i+j];
        }
    }
}


// void slidingWindow( float *ts,  float **ts_list, unsigned int sub_len, unsigned int n_data) {
//     for( int i = 0; i < n_data; i++) {
//         for( int j = 0; j < sub_len; j++) {
//             ts_list[i][j] = ts[i + j];
//         }
//     }
//     return;
// }


// void save_binary( const char *path, float **ts_list, int dim, int size) {
//     FILE *fp;
//     fp = fopen( path, "wb");
//     if( fp == NULL) {
//         std::cerr << "cannot open " << path << std::endl;
//         return;
//     }

//     for( int i = 0; i < size; i++) {
//         fwrite( ts_list[i], sizeof(float), dim, fp);
//     }
//     fclose( fp);
//     return;
// }


#endif