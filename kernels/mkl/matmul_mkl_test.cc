#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "../matmul.h"
using namespace std;

// Function to fill matrices with random int8 values
void fill_matrix_int8(int8_t *data, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        // data[i] = static_cast<int8_t>(rand() % 128); // Random values from -128 to 127
        // data[i] = static_cast<int8_t>(rand() % 256 - 128); // Random values from -128 to 127

        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::normal_distribution<> d(0, 6); // mean 0, standard deviation 64

        for (int i = 0; i < rows * cols; ++i) {
            // int value = std::round(d(gen));
            int value = rand() % 10 - 5;
            data[i] = static_cast<int8_t>(std::max(-128, std::min(127, value)));
        }
    }
}

// Function to compare two int32 matrices
bool compare_matrices(const int8_t *mat1, const int8_t *mat2, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        if (mat1[i] != mat2[i]) {
            std::cout << "Mismatch at index " << i << ": " << mat1[i] << " != " << mat2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void test_mat_mul_int8() {
    // Matrix dimensions
    int M = 64; // Rows in A and C
    int N = 128; // Columns in B and C
    int K = 32; // Columns in A and rows in B

    // Allocate matrices
    int8_t *A_data = new int8_t[M * K];
    int8_t *B_data = new int8_t[N * K];
    int8_t *C_naive_int8 = new int8_t[M * N];
    int8_t *C_mkl_int8 = new int8_t[M * N];

    // Fill matrices with random data
    srand(static_cast<unsigned>(time(0)));
    std::cout<<"filling matrix A"<<std::endl;
    fill_matrix_int8(A_data, M, K);
    std::cout<<"filling matrix B"<<std::endl;
    fill_matrix_int8(B_data, N, K);

    // // make A the identity matrix
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         if (i == j) {
    //             A_data[i * K + j] = -1;
    //         } else {
    //             A_data[i * K + j] = 0;
    //         }
    //     }
    // }

    // Print matrices A and B
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << static_cast<int>(A_data[i * K + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << static_cast<int>(B_data[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }

    // Prepare matrices
    // matrix A = {M, K, nullptr, nullptr, nullptr, nullptr, A_data, nullptr, nullptr, {}};
    matrix A = {
        .row = M,
        .column = K,
        .int8_data_ptr = A_data,
        .qparams = {
            .scale = 0.1f,
            .zero_point = 0,
            .q_min = -128,
            .q_max = 127
        }
    };

    // Initialize matrix B
    matrix B = {
        .row = K,
        .column = N,
        .int8_data_ptr = B_data,
        .qparams = {
            .scale = 0.1f,      // Set appropriate scale
            .zero_point = 0,    // Set appropriate zero point
            .q_min = -128,      // For int8_t
            .q_max = 127
        }
    };

    // Initialize output matrix C1 (for naive implementation)
    matrix C1 = {
        .row = M,
        .column = N,
        .int8_data_ptr = C_naive_int8,  // We'll store the final result as int8_t
        .qparams = {
            .scale = 1.0f,
            .zero_point = 0,
            .q_min = -128,
            .q_max = 127
        }
    };

    // Initialize output matrix C2 (for MKL implementation)
    matrix C2 = {
        .row = M,
        .column = N,
        .int8_data_ptr = C_mkl_int8,  // We'll store the final result as int8_t
        .qparams = {
            .scale = 1.0f,
            .zero_point = 0,
            .q_min = -128,
            .q_max = 127
        }
    };

    cout<<"Ok"<<endl;
    matmul_params params = {A, B, C1, {}, {}, 1.0f, 0.0f};
    params.alpha = 0.1f;

    matmul::MatmulOperator op;

    // Perform naive multiplication
    op.naive_mat_mul_int8(&params);

    cout<<"Ok"<<endl;

    // Copy result to C_mkl for comparison
    // std::copy(C_naive_int8, C_naive_int8 + M * N, C_mkl_int8);
    cout<<"Ok"<<endl;

    // Prepare parameters for MKL multiplication
    params.C = C2;

    // Perform MKL multiplication
    op.mat_mul_mkl_int8(&params);
    cout<<"Ok"<<endl;

    // Print the results
    std::cout << "Matrix C (naive):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << (int)C_naive_int8[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix C (MKL):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << (int)C_mkl_int8[i * N + j] << " ";
        }
        std::cout << std::endl;
    }


    // Validate results
    if (compare_matrices(C_naive_int8, C_mkl_int8, M, N)) {
        std::cout << "Test passed: MKL output matches naive implementation." << std::endl;
    } else {
        std::cout << "Test failed: MKL output does not match naive implementation." << std::endl;
    }

    // Clean up
    delete[] A_data;
    delete[] B_data;
    delete[] C_naive_int8;
    delete[] C_mkl_int8;
}

int main() {
    test_mat_mul_int8();
    return 0;
}
