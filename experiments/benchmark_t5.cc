#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <tuple>
#include <vector>
#include <cstring>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <sys/mman.h>
#include "../kernels/matmul.h"

// Helper function to calculate time difference in ms
double time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

// Precision types
enum class Precision { FP32, FP16, INT8, INT4, INT32 };

// Function to write dummy data to a file for memory-mapping later
void write_random_data_to_file(const char *filename, size_t data_size, Precision precision = Precision::FP32) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        std::vector<uint8_t> random_data(data_size);

        std::random_device rd;  // Random number generator
        std::mt19937 gen(rd());
        
        // Different distributions for different precisions
        if (precision == Precision::FP32) {
            std::uniform_real_distribution<float> dist(-1.0, 1.0);
            float *data_ptr = reinterpret_cast<float *>(random_data.data());
            for (size_t i = 0; i < data_size / sizeof(float); i++) {
                data_ptr[i] = dist(gen);
            }
        } else if (precision == Precision::FP16) {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            float16_t *data_ptr = reinterpret_cast<float16_t *>(random_data.data());
            for (size_t i = 0; i < data_size / sizeof(float16_t); i++) {
                data_ptr[i] = dist(gen);
            }
        } else if (precision == Precision::INT8) {
            std::uniform_int_distribution<int8_t> dist(-128, 127);
            int8_t *data_ptr = reinterpret_cast<int8_t *>(random_data.data());
            for (size_t i = 0; i < data_size / sizeof(int8_t); i++) {
                data_ptr[i] = dist(gen);
            }
        } else if (precision == Precision::INT4) {
            std::uniform_int_distribution<uint8_t> dist(0, 15);
            for (size_t i = 0; i < data_size; i++) {
                random_data[i] = (dist(gen) & 0xF) | ((dist(gen) & 0xF) << 4);  // Pack two 4-bit values into one byte
            }
        }

        file.write(reinterpret_cast<const char *>(random_data.data()), random_data.size());
        file.close();
    } else {
        std::cerr << "Error: could not open file for writing random data.\n";
    }
}

// Function to map data from file into memory without copying
void* map_file_to_memory(const char *filename, size_t data_size) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: could not open file for mapping.\n";
        return nullptr;
    }

    void *mapped_data = mmap(NULL, data_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (mapped_data == MAP_FAILED) {
        std::cerr << "Error: mmap failed.\n";
        return nullptr;
    }

    return mapped_data;
}

// Function to initialize matrix metadata and point data_ptr to the relevant memory-mapped location
void initialize_mapped_matrix(struct matrix *mat, int rows, int cols, void *mapped_data, size_t offset, Precision precision) {
    mat->row = rows;
    mat->column = cols;

    // Calculate the initial pointer with offset
    uintptr_t initial_ptr = reinterpret_cast<uintptr_t>(mapped_data) + offset;

    // Adjust offset to ensure 32-byte alignment
    size_t alignment = 32;
    size_t misalignment = initial_ptr % alignment;
    if (misalignment != 0) {
        offset += alignment - misalignment;
    }

    void *data_ptr = (uint8_t*)mapped_data + offset;

    switch (precision) {
        case Precision::FP32:
            mat->data_ptr = reinterpret_cast<float *>(data_ptr);
            break;
        case Precision::FP16:
            mat->half_data_ptr = reinterpret_cast<float16_t *>(data_ptr);
            break;
        case Precision::INT8:
            mat->int8_data_ptr = reinterpret_cast<int8_t *>(data_ptr);
            break;
        case Precision::INT4:
            mat->int4_data_ptr = reinterpret_cast<uint8_t *>(data_ptr);
            break;
    }

    // Use madvise to tell the OS to prefetch data into memory
    size_t data_size = rows * cols * sizeof(float); // Adjust based on precision
    switch (precision) {
        case Precision::FP32:
            data_size = rows * cols * sizeof(float);
            break;
        case Precision::FP16:
            data_size = rows * cols * sizeof(float16_t);
            break;
        case Precision::INT8:
            data_size = rows * cols * sizeof(int8_t);
            break;
        case Precision::INT4:
            data_size = (rows * cols + 1) / 2; // INT4 packs two values per byte
            break;
    }

    madvise(data_ptr, data_size, MADV_WILLNEED);
}

void initialize_allocated_matrix(struct matrix *mat, int rows, int cols, Precision precision) {
    mat->row = rows;
    mat->column = cols;

    size_t total_elements = rows * cols;

    // Allocate aligned memory based on the precision
    switch (precision) {
        case Precision::FP32:
            mat->data_ptr = static_cast<float *>(aligned_alloc(32, total_elements * sizeof(float)));  // 32-byte aligned for AVX
            break;
        case Precision::FP16:
            mat->half_data_ptr = static_cast<float16_t *>(aligned_alloc(32, total_elements * sizeof(float16_t)));  // Aligned for FP16
            break;
        case Precision::INT32:
            mat->int32_data_ptr = static_cast<int32_t *>(aligned_alloc(32, total_elements * sizeof(int32_t)));  // 32-byte aligned for AVX
            break;
        case Precision::INT8:
            mat->int8_data_ptr = static_cast<int8_t *>(aligned_alloc(32, total_elements * sizeof(int8_t)));  // Aligned for INT8
            break;
        case Precision::INT4:
            mat->int4_data_ptr = static_cast<uint8_t *>(aligned_alloc(32, total_elements / 2));  // Aligned for INT4
            break;
    }

    // initialize to zero
    switch (precision) {
        case Precision::FP32:
            memset(mat->data_ptr, 0, total_elements * sizeof(float));
            break;
        case Precision::FP16:
            memset(mat->half_data_ptr, 0, total_elements * sizeof(float16_t));
            break;
        case Precision::INT32:
            memset(mat->int32_data_ptr, 0, total_elements * sizeof(int32_t));
            break;
        case Precision::INT8:
            memset(mat->int8_data_ptr, 0, total_elements * sizeof(int8_t));
            break;
        case Precision::INT4:
            memset(mat->int4_data_ptr, 0, total_elements / 2);
            break;
    }
}

// Function to unmap the memory after benchmarking
void unmap_file_from_memory(void *mapped_data, size_t data_size) {
    if (munmap(mapped_data, data_size) != 0) {
        std::cerr << "Error: failed to unmap memory.\n";
    }
}

int main() {
    srand(42);

    matmul::MatmulOperator matmul_op;

    // Precision setting: change this to FP16, INT8, or INT4 to benchmark different precisions
    Precision precision = Precision::INT8;

    // Define matrix sizes for benchmarking
    std::vector<std::tuple<int, int, int, int>> benchmarks = {
        {80, 4096, 4096, 96},   // First matmul
        {80, 4096, 10240, 48},  // Second matmul
        {80, 10240, 4096, 24}   // Third matmul
    };

    // Calculate the total size needed for the dummy data file (based on the largest matrix size and precision)
    size_t total_data_size = 3000000000; // Adjust the size as needed
    const char *dummy_data_file = "dummy_data.bin";
    
    // Create the dummy data file
    write_random_data_to_file(dummy_data_file, total_data_size);

    // Map the file to memory
    void *mapped_data = map_file_to_memory(dummy_data_file, total_data_size);
    if (mapped_data == nullptr) {
        return -1;
    }

    // madvise(mapped_data, total_data_size, MADV_WILLNEED);

    struct timeval tot_start, tot_end;
    gettimeofday(&tot_start, NULL);

    size_t offset = 0;  // Offset to where we place each matrix in the memory-mapped file

    for (auto &bench : benchmarks) {
        int m = std::get<0>(bench);
        int n = std::get<1>(bench);
        int k = std::get<2>(bench);
        int num_calls = std::get<3>(bench);

        double total_time = 0.0;
        // Initialize matrices and map to relevant portions of the file
        for (int i = 0; i < num_calls; ++i) {
            if (offset >= total_data_size - k * (m+n) * 2 * sizeof(float)) {
                std::cerr << "Warning: not enough space in the dummy data file.\n";
                offset = 0;
            }
            struct matrix A, B, C;
            initialize_mapped_matrix(&A, m, k, mapped_data, offset, precision);
            offset += m * k * sizeof(float);
            initialize_mapped_matrix(&B, n, k, mapped_data, offset, precision);
            offset += n * k * sizeof(float);
            initialize_allocated_matrix(&C, m, n, Precision::INT32);

            // std::cout<<"Matrix A: "<<A.row<<"x"<<A.column<<std::endl;

            // Set up matmul parameters
            struct matmul_params params;
            params.A = A;
            params.B = B;
            params.C = C;
            params.alpha = 1.0;
            // params.beta = 0.0;

            int _, num_thread = params.opt_params.num_thread;

            // Time the matrix multiplication operation
            struct timeval start, end;
            gettimeofday(&start, NULL);
            switch (precision) {
                case Precision::FP32:
                    matmul_op.mat_mul_accelerator_transposed_fastover_column(&params);
                    break;
                
                case Precision::FP16:
                    std::cerr<<"Error: FP16 not supported in this benchmark.\n";
                    break;

                case Precision::INT8:
                    // matmul_op.mat_mul_accelerator_int8_fast_2x2_32unroll(&params);
                    matmul_op.mat_mul_mkl_int8(&params);
                    // matmul_op.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(&params);
                    break;

                case Precision::INT4:
                    matmul_op.mat_mul_accelerator_int4_fast(&params);
                    break;

                default:
                    std::cerr << "Error: Unsupported precision type.\n";
                    break;
            }

            // // test the result of multiplication
            // int32_t acc = 0;
            // for (int k = 0; k < params.A.column; k++) {
            //     acc += params.alpha * params.A.int8_data_ptr[0 * params.A.column + k] * (uint8_t)params.B.int8_data_ptr[0 * params.B.column + k];
            // }
            // int32_t expected = params.C.int32_data_ptr[0 * params.C.column + 0];
            // std::cerr<< "Result: "<<acc<<", Expected: "<<expected<<std::endl;
            // if (acc != expected) {
            //     std::cerr << "Error: multiplication result does not match.\n";
            // }

            gettimeofday(&end, NULL);
            double elapsed_time = time_diff(&start, &end);
            total_time += elapsed_time;

        }
        std::cout << "Benchmark (" << m << "x" << k << "x" << n << ") - Time elapsed (total for "
                << num_calls << " calls): " << total_time << " ms" << std::endl;
        double gflops = 2.0 * m * n * k * num_calls / (total_time * 1e6);
        std::cout << "GFLOPs: " << gflops << std::endl;
    }

    gettimeofday(&tot_end, NULL);
    double total_elapsed_time = time_diff(&tot_start, &tot_end);
    std::cout << "Total time elapsed: " << total_elapsed_time << " ms" << std::endl;

    unmap_file_from_memory(mapped_data, total_data_size);

    std::remove(dummy_data_file);

    return 0;
}