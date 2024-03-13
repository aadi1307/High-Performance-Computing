#include "optimize.h"

void optimize1(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = v->data;
    data_t temp = IDENT;
    for (int i = 0; i < length; i++) {
        temp = temp OP d[i];
    }
    *dest = temp;
}

void optimize2(vec *v, data_t *dest) {
    int length = v->len;
    int limit = length - 1;
    data_t *d = v->data;
    data_t x = IDENT;
    int i;
    for (i = 0; i < limit; i += 2) {
        x = (x OP d[i]) OP d[i + 1];
    }
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize3(vec *v, data_t *dest) {
    int length = v->len;
    int limit = length - 1;
    data_t *d = v->data;
    data_t x = IDENT;
    int i;
    for (i = 0; i < limit; i += 2) {
        x = x OP (d[i] OP d[i + 1]);
    }
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize4(vec *v, data_t *dest) {
    long length = v->len;
    long limit = length - 1;
    data_t *d = v->data;
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    long i;
    for (i = 0; i < limit; i += 2) {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    for (; i < length; i++) {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}

void optimize5(vec *v, data_t *dest) {
    constexpr int K = 3;
    constexpr int L = 3;  // Unrolling factor

    int length = v->len;
    int limit = length - (length % L);  // Ensuring we do not go past the array bounds
    data_t *d = v->data;
    data_t x[K] = {IDENT, IDENT, IDENT};  // Initialize all accumulators to IDENT

    int i = 0;
    for (; i < limit; i += L) {
        for (int j = 0; j < L; ++j) {
            x[j % K] = x[j % K] OP d[i + j];  // Modulo for wrapping around the K accumulators
        }
    }

    // Handle the tail case
    for (; i < length; ++i) {
        x[0] = x[0] OP d[i];  // Accumulate remaining elements in x[0]
    }

    // Combine the results from the separate accumulators
    data_t result = IDENT;
    for (int j = 0; j < K; ++j) {
        result = result OP x[j];
    }

    *dest = result;
}
