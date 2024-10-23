#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <cmath>
#include "../kernels/matmul.h"

void initialize_matrix(struct matrix* mat, int rows, int cols) {
    mat->row = rows;
    mat->column = cols;
    mat->data_ptr = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        mat->data_ptr[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }
}

void initialize_matrix_(struct matrix* mat, int rows, int cols) {
}


void cleanup_matrix(struct matrix* mat) {
    delete[] mat->data_ptr;
}

double time_diff(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0; // in ms
}

int main() {
    srand(42);

    std::cout<<"Hello World"<<std::endl;

    matmul::MatmulOperator matmul_op;

    // Define matrix sizes for benchmarking
    std::vector<std::tuple<int, int, int>> benchmarks = {
        {40, 40, 40},
        {400, 400, 400},
        {4096, 4096, 4096},
        {1, 32000, 4096},
        {80, 4096, 4096}, // T5 matmul shape 1, pytorch 7.25 ms
        {80, 4096, 10240}, // T5 matmul shape 2, pytorch 13.6 ms
    };

    for (auto& bench : benchmarks) {
        int m = std::get<0>(bench);
        int n = std::get<1>(bench);
        int k = std::get<2>(bench);

        // Initialize matrices
        struct matrix A, B, C;
        initialize_matrix(&A, m, k);
        initialize_matrix(&B, n, k); // transposed B
        initialize_matrix(&C, m, n);

        struct matmul_params params;
        params.A = A;
        params.B = B;
        params.C = C;

        // Run and time the matrix multiplication
        struct timeval start, end;
        gettimeofday(&start, NULL);

        matmul_op.mat_mul_accelerator_transposed_fastover_column(&params);

        gettimeofday(&end, NULL);
        double elapsed_time = time_diff(&start, &end);

        // Output the results
        std::cout << "Benchmark (" << m << "x" << k << "x" << n << ") - Time elapsed: " << elapsed_time << " ms" << std::endl;

        // Calculate GOPs
        double gflops = 2.0 * m * n * k / (elapsed_time * 1e6);
        std::cout << "GFLOPs: " << gflops << std::endl;

        // Cleanup
        cleanup_matrix(&A);
        cleanup_matrix(&B);
        cleanup_matrix(&C);
    }

    return 0;
}