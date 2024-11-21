#include <mkl.h>
#include <mkl_cblas.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../matmul.h"

namespace matmul{
void MatmulOperator::naive_mat_mul_int8(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    // CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            int acc = 0;
            for (k = 0; k < A->column; k++)
                acc += ((int32_t)data_A[i * A->column + k] - A_zp) * data_B[k * B->column + j];

            acc = (int32_t)((float)acc * effective_scale);
            acc -= C_zp;
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
}

void int8_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * n + j] = (int8_t)acc;
        }
    }
}

void int8_ref_matmul_nobias(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
    }
}

void int8_ref_matmul_nobias_batch(const struct matmul_params *params) {
    // std::cout<<"Running int8_ref_matmul_nobias_batch"<<std::endl;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[i * B->row * B->column + j * B->row + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
    }
    // std::cout<<"Finished int8_ref_matmul_nobias_batch"<<std::endl;
}

void int8_ref_matmul_bfp32_ofp32(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha + (float)(params->bias.data_ptr[j]);
        }
    }
}

void int8_ref_matmul_nobias_ofp32(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha;
        }
    }
}

void int8_ref_matmul_nobias_ofp32_batch(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[i * B->row * B->column + j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha;
        }
    }
}

void fp32_ref_matmul_bias(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *bias = params->bias.data_ptr;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc + bias[j];
            data_C[i * n + j] = acc;
        }
    }
}


void MatmulOperator::mat_mul_mkl_int8(const matmul_params *params) {
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    int M = A.row;
    int K = A.column;
    int N = B.column;

    const int8_t *data_A = A.int8_data_ptr;
    int32_t *data_C = new int32_t[M * N];  // Temporary buffer for accumulation

    const int8_t *data_B = B.int8_data_ptr;

    // Shift A instead of B by adding 128:
    // We shift A because the mkl interface expects A to be unsigned instead of B if row-major
    uint8_t *data_A_shifted = new uint8_t[M * K];
    for (int i = 0; i < M * K; ++i) {
        int16_t temp = static_cast<int16_t>(A.int8_data_ptr[i] + 128) ;
        data_A_shifted[i] = static_cast<uint8_t>(temp);
    }

    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;


    const MKL_INT8 ao = -(A.qparams.zero_point+128);
    assert(B.qparams.zero_point == 0);
    const MKL_INT8 bo = -(B.qparams.zero_point);  // Adjusted zero point for B
    MKL_INT32 co = C.qparams.zero_point;

    float effective_scale = A.qparams.scale * B.qparams.scale / C.qparams.scale;
    std::cout<<"Effective scale: "<<effective_scale<<std::endl;

    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_gemm_s8u8s32(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
        M, N, K,
        alpha,
        data_A_shifted, lda, ao,
        data_B, ldb, bo,
        beta,
        data_C, ldc, &co
    );

    // Post-process the result
    int8_t *data_C_int8 = C.int8_data_ptr;
    for (int i = 0; i < M * N; ++i) {
        int32_t acc = data_C[i];
        acc = static_cast<int32_t>(std::round(acc * effective_scale));
        if (params->bias.int8_data_ptr) {
            acc += static_cast<int32_t>(params->bias.int8_data_ptr[i % N]) * params->beta;
        }

        acc = std::max(static_cast<int32_t>(C.qparams.q_min), acc);
        acc = std::min(static_cast<int32_t>(C.qparams.q_max), acc);

        data_C_int8[i] = static_cast<int8_t>(acc);
    }

    // Clean up
    delete[] data_A_shifted;
    delete[] data_C;
}

void mat_mul_mkl_nobias_ofp32(const matmul_params *params) {
    // std::cout<<"Running mat_mul_mkl_nobias_ofp32"<<std::endl;
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    int M = A.row;
    int K = A.column;
    int N = B.column;

    const int8_t *data_A = A.int8_data_ptr;
    int32_t *data_C = new int32_t[M * N];  // Temporary buffer for accumulation

    const int8_t *data_B = B.int8_data_ptr;
    // Shift B by adding 128
    uint8_t *data_A_shifted = new uint8_t[K * M];
    for (int i = 0; i < K * M; ++i) {
        int16_t temp = static_cast<int16_t>(A.int8_data_ptr[i] + 128) ;
        data_A_shifted[i] = static_cast<uint8_t>(temp);
    }

    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;

    float alpha = 1.0f;  // Apply alpha in post-processing
    float beta = 0.0f;   // Apply beta in post-processing

    const MKL_INT8 ao = -(A.qparams.zero_point+128);
    const MKL_INT8 bo = -(B.qparams.zero_point);  // Adjusted zero point for B
    MKL_INT32 co = 0;

    // std::cout<<"Beginning cblas gemm"<<std::endl;

    // Perform the matrix multiplication
    cblas_gemm_s8u8s32(
        CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
        M, N, K,
        alpha,
        data_A_shifted, lda, ao,
        data_B, ldb, bo,
        beta,
        data_C, ldc, &co
    );

    // std::cout<<"Finished cblas gemm"<<std::endl;

    // Post-process the result
    float *data_ptr_C = C.data_ptr;
    for (int i = 0; i < M * N; ++i) {
        int32_t acc = data_C[i];

        // Apply alpha and beta
        acc = float(acc * params->alpha);
        data_ptr_C[i] = acc;
    }

    delete[] data_A_shifted;
    delete[] data_C;
}

void mat_mul_mkl_nobias_bfp32_ofp32(const matmul_params *params) {
    // std::cout<<"Running mat_mul_mkl_nobias_ofp32"<<std::endl;
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    int M = A.row;
    int K = A.column;
    int N = B.column;

    const int8_t *data_A = A.int8_data_ptr;
    int32_t *data_C = new int32_t[M * N];  // Temporary buffer for accumulation

    const int8_t *data_B = B.int8_data_ptr;
    // Shift B by adding 128
    uint8_t *data_A_shifted = new uint8_t[K * M];
    for (int i = 0; i < K * M; ++i) {
        int16_t temp = static_cast<int16_t>(A.int8_data_ptr[i] + 128) ;
        data_A_shifted[i] = static_cast<uint8_t>(temp);
    }

    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;

    float alpha = 1.0f;  // Apply alpha in post-processing
    float beta = 0.0f;   // Apply beta in post-processing

    const MKL_INT8 ao = -(A.qparams.zero_point+128);
    const MKL_INT8 bo = -(B.qparams.zero_point);  // Adjusted zero point for B
    MKL_INT32 co = 0;

    // std::cout<<"Beginning cblas gemm"<<std::endl;

    // Perform the matrix multiplication
    cblas_gemm_s8u8s32(
        CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
        M, N, K,
        alpha,
        data_A_shifted, lda, ao,
        data_B, ldb, bo,
        beta,
        data_C, ldc, &co
    );

    // Post-process the result
    float *data_ptr_C = C.data_ptr;
    for (int i = 0; i < M * N; ++i) {
        int32_t acc = data_C[i];

        // Apply alpha and beta
        acc = float(acc * params->alpha);
        data_ptr_C[i] = acc + params->bias.data_ptr[i % N]; // sample doesn't multiply beta
    }

    delete[] data_A_shifted;
    delete[] data_C;
}

void fp32_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc;
            data_C[i * n + j] = acc;
        }
    }
}

void MatmulOperator::mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_32unroll_over_column" << std::endl;
    mat_mul_mkl_int8(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_2x2_32unroll" << std::endl;
    mat_mul_mkl_int8(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch" << std::endl;
    int8_ref_matmul_nobias_ofp32_batch(params);
    // mat_mul_mkl_int8(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32" << std::endl;
    mat_mul_mkl_nobias_ofp32(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch" << std::endl;
    int8_ref_matmul_nobias_batch(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_int8_fast_2x2_32unroll_nobias" << std::endl;
    mat_mul_mkl_int8(params);
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_transposed_fastover_column" << std::endl;
    fp32_ref_matmul(params);
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column_bias(const matmul_params *params) {
    // std::cout << "Running mat_mul_accelerator_transposed_fastover_column" << std::endl;
    fp32_ref_matmul_bias(params);
}

void MatmulOperator::mat_mul_accelerator_int4_fast(const matmul_params *params) {
    // MKL doesn't support int4; you might need to provide an alternative
    std::cerr << "Error: mat_mul_accelerator_int4_fast is not supported with MKL." << std::endl;
    exit(1);
}

void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const matmul_params *params) {
    // Similar to above
    std::cerr << "Error: mat_mul_accelerator_int4_fast_no_offset is not supported with MKL." << std::endl;
    exit(1);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const matmul_params *params) {
    // std::cout<<"Running mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column"<<std::endl;
    // mat_mul_mkl_nobias_bfp32_ofp32(params);
    int8_ref_matmul_bfp32_ofp32(params);
    // std::cout<<"done!"<<std::endl;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const matmul_params *params) {
    // std::cout<<"Running mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32"<<std::endl;
    // mat_mul_mkl_nobias_bfp32_ofp32(params);
    // std::cout<<"done!"<<std::endl;
    int8_ref_matmul_bfp32_ofp32(params);
}

} // namespace matmul

