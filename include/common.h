#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>

#define cudaCheck(err, msg)   \
    if (err != cudaSuccess) { \
        printf("error code : %d", err);\
        perror(msg);          \
        exit(-1);             \
    }



int cmpint(const void *a, const void *b);

int cmpdouble(const void *a, const void *b);

#endif
