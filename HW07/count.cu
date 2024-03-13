#include "count.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts) {
    // 1: Creating a mutable copy of the input as the original is read-only.
    thrust::device_vector<int> data(d_in);

    // 2: Arranging the elements of the 'data' vector in a non-descending order.
    thrust::sort(data.begin(), data.end());

    // 3: Adjusting the size of 'values' to potentially accommodate all unique elements.
    values.resize(data.size());

    // 4: Transferring unique elements from 'data' to 'values', avoiding duplicates.
    auto end_unique = thrust::unique_copy(data.begin(), data.end(), values.begin());

    // 5: Modifying 'values' size to match the actual count of unique elements post-copy.
    values.resize(thrust::distance(values.begin(), end_unique));

    // 6: Setting up 'counts' and calculating how frequently each unique element appears.
    counts.resize(values.size());
    thrust::reduce_by_key(thrust::device, data.begin(), data.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(),
                          counts.begin());
}
