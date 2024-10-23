#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <tuple>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include "../kernels/matmul.h"

// Helper function to calculate time difference in ms
double time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

// Precision types
enum class Precision { FP32, FP16, INT8, INT4 };

// Function to write dummy data to a file for memory-mapping later
void write_dummy_data_to_file(const char *filename, size_t data_size) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        std::vector<uint8_t> dummy_data(data_size, 0); // Create dummy data
        file.write(reinterpret_cast<const char *>(dummy_data.data()), dummy_data.size());
        file.close();
    } else {
        std::cerr << "Error: could not open file for writing dummy data.\n";
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

    // Point the matrix's data pointer to the relevant portion of the memory-mapped file
    switch (precision) {
        case Precision::FP32:
            mat->data_ptr = reinterpret_cast<float *>((uint8_t*)mapped_data + offset);
            break;
        case Precision::FP16:
            mat->half_data_ptr = reinterpret_cast<float16_t *>((uint8_t*)mapped_data + offset);
            break;
        case Precision::INT8:
            mat->int8_data_ptr = reinterpret_cast<int8_t *>((uint8_t*)mapped_data + offset);
            break;
        case Precision::INT4:
            mat->int4_data_ptr = reinterpret_cast<uint8_t *>((uint8_t*)mapped_data + offset);
            break;
    }
}

void initialize_allocated_matrix(struct matrix *mat, int rows, int cols, Precision precision) {
    mat->row = rows;
    mat->column = cols;

    // Dynamically allocate memory for the matrix based on the precision
    switch (precision) {
        case Precision::FP32:
            mat->data_ptr = new float[rows * cols];  // Allocate space for FP32 matrix
            break;
        case Precision::FP16:
            mat->half_data_ptr = new float16_t[rows * cols];  // Allocate space for FP16 matrix
            break;
        case Precision::INT8:
            mat->int8_data_ptr = new int8_t[rows * cols];  // Allocate space for INT8 matrix
            break;
        case Precision::INT4:
            mat->int4_data_ptr = new uint8_t[rows * cols / 2];  // Allocate space for INT4 matrix
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
    size_t total_data_size = 1000000000; // Adjust the size as needed
    const char *dummy_data_file = "dummy_data.bin";
    
    // Create the dummy data file
    write_dummy_data_to_file(dummy_data_file, total_data_size);

    // Map the file to memory
    void *mapped_data = map_file_to_memory(dummy_data_file, total_data_size);
    if (mapped_data == nullptr) {
        return -1;
    }

    struct timeval tot_start, tot_end;
    gettimeofday(&tot_start, NULL);

    size_t offset = 0;  // Offset to where we place each matrix in the memory-mapped file

    for (auto &bench : benchmarks) {
        int m = std::get<0>(bench);
        int n = std::get<1>(bench);
        int k = std::get<2>(bench);
        int num_calls = std::get<3>(bench);

        // Initialize matrices and map to relevant portions of the file
        struct matrix A, B, C;
        initialize_mapped_matrix(&A, m, k, mapped_data, offset, precision);
        offset += m * k * sizeof(float);
        initialize_mapped_matrix(&B, n, k, mapped_data, offset, precision);
        offset += n * k * sizeof(float);
        initialize_allocated_matrix(&C, m, n, precision);

        std::cout<<"Matrix A: "<<A.row<<"x"<<A.column<<std::endl;

        if (offset >= total_data_size - k * (m+n) * sizeof(float)) {
            std::cerr << "Warning: not enough space in the dummy data file.\n";
            offset = 0;
        }

        // Set up matmul parameters
        struct matmul_params params;
        params.A = A;
        params.B = B;
        params.C = C;

        // Time the matrix multiplication operation
        struct timeval start, end;
        double total_time = 0.0;
        for (int i = 0; i < num_calls; ++i) {
            gettimeofday(&start, NULL);

            matmul_op.mat_mul_accelerator_transposed_fastover_column(&params);

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