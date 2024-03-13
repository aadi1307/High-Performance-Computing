#include "msort.h"
#include <algorithm>  // Used for the std::sort function
#include <vector>

// Function declarations for internal use within merge sort
void parallel_mergesort(int* arr, int* temp, const std::size_t left, const std::size_t right, const std::size_t threshold);
void merge(int* arr, int* temp, const std::size_t left, const std::size_t mid, const std::size_t right);

// Conducts an in-place merge sort on an integer array
void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    // Reserve space for a support array used in merging
    std::vector<int> temp(n);
    // Initiate the recursive sorting process
    parallel_mergesort(arr, temp.data(), 0, n, threshold);
}

// Recursively sorts the array segments in parallel where permissible
void parallel_mergesort(int* arr, int* temp, const std::size_t left, const std::size_t right, const std::size_t threshold) {
    if (right - left <= threshold) {
        // Use a non-parallel sort for small segments
        std::sort(arr + left, arr + right);
        return;
    }

    // Determine the middle point to divide the array segment
    const std::size_t mid = left + (right - left) / 2;

    // Create tasks for sorting the subarrays in parallel if above the size threshold
    #pragma omp task shared(arr, temp) if(right - left > threshold)
    parallel_mergesort(arr, temp, left, mid, threshold);

    #pragma omp task shared(arr, temp) if(right - left > threshold)
    parallel_mergesort(arr, temp, mid, right, threshold);

    // Ensure tasks are completed before merging
    #pragma omp taskwait

    // Combine the sorted subarrays
    merge(arr, temp, left, mid, right);
}

// Combines two sorted subarrays into one sorted array
void merge(int* arr, int* temp, const std::size_t left, const std::size_t mid, const std::size_t right) {
    std::size_t i = left, j = mid, k = left;

    // Sequentially merge elements into the support array
    while (i < mid && j < right) {
        if (arr[i] < arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    // Transfer any remaining elements of the subarrays
    while (i < mid) {
        temp[k++] = arr[i++];
    }
    while (j < right) {
        temp[k++] = arr[j++];
    }

    // Move the sorted elements back to the original array
    for (std::size_t l = left; l < right; ++l) {
        arr[l] = temp[l];
    }
}
