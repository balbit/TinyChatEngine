#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "../kernels/matmul.h" // Replace with the correct path
#include "../kernels/pthread_pool.h" // Replace with the correct path

// Function prototypes for both matmul implementations
extern void mat_mul_mkl_int8(const matmul_params *params);
extern void int8_ref_matmul_nobias(const matmul_params *params);

void generate_random_matrix(matrix &mat, int rows, int cols, int8_t q_min = -128, int8_t q_max = 127) {
    mat.row = rows;
    mat.column = cols;
    mat.int8_data_ptr = new int8_t[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        mat.int8_data_ptr[i] = static_cast<int8_t>(q_min + rand() % (q_max - q_min + 1));
    }
}

bool matrices_match(const matrix &C1, const matrix &C2) {
    assert(C1.row == C2.row && C1.column == C2.column);
    for (int i = 0; i < C1.row * C1.column; i++) {
        if (C1.int8_data_ptr[i] != C2.int8_data_ptr[i]) {
            std::cout << "Mismatch at index " << i << ": "
                      << "C1 = " << static_cast<int>(C1.int8_data_ptr[i]) << ", "
                      << "C2 = " << static_cast<int>(C2.int8_data_ptr[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    srand(0); // Seed for reproducibility
    matmul::MatmulOperator matmul_op;

    // Set up dimensions and quantization parameters
    int rows_A = 4, cols_A = 4, rows_B = 4, cols_B = 4;
    assert(cols_A == rows_B); // Ensure dimensions match for multiplication

    // Initialize matrices A, B, C
    matrix A, B, C1, C2;
    generate_random_matrix(A, rows_A, cols_A);
    generate_random_matrix(B, rows_B, cols_B);
    
    // Initialize output matrices C1 and C2
    C1.row = C2.row = rows_A;
    C1.column = C2.column = cols_B;
    C1.int8_data_ptr = new int8_t[rows_A * cols_B];
    C2.int8_data_ptr = new int8_t[rows_A * cols_B];

    // Set up matmul_params structure
    matmul_params params;
    params.A = A;
    params.B = B;
    params.C = C1;  // Run with the first output matrix
    params.alpha = 1.0f;
    params.beta = 0.0f;
    C1.qparams.q_min = C2.qparams.q_min = -128;
    C1.qparams.q_max = C2.qparams.q_max = 127;

    // Run reference implementation
    params.C = C1;
    matmul_op.naive_mat_mul_int8(&params);

    // Run MKL-optimized implementation
    params.C = C2;
    matmul_op.mat_mul_mkl_int8(&params);

    // Compare results
    if (matrices_match(C1, C2)) {
        std::cout << "Test passed: Outputs match!" << std::endl;
    } else {
        std::cout << "Test failed: Outputs do not match." << std::endl;
    }

    // Clean up dynamically allocated memory
    delete[] A.int8_data_ptr;
    delete[] B.int8_data_ptr;
    delete[] C1.int8_data_ptr;
    delete[] C2.int8_data_ptr;

    return 0;
}
