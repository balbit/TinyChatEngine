
#include <mkl.h>
void MatmulOperator::naive_mat_mul_int8(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    CHECK_MATRICES(A, B, C);

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




void MatmulOperator::mat_mul_mkl_int8(const matmul_params *params) {
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    // Ensure matrix dimensions are compatible for multiplication
    assert(A.column == B.column);
    assert(A.row == C.row);
    assert(B.row == C.column);

    int M = A.row;     // Rows in A
    int N = B.row;     // Rows in B (columns in C)
    int K = A.column;  // Columns in A and B

    // Prepare data pointers
    const int8_t *data_A = A.int8_data_ptr;
    const uint8_t *data_B = reinterpret_cast<const uint8_t *>(B.int8_data_ptr); // MKL expects uint8_t for B
    int32_t *data_C = C.int32_data_ptr;

    // Leading dimensions
    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;

    // Alpha and beta scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Zero points for quantization (if needed)
    const MKL_INT8 ao = 0; // Zero point for A
    const MKL_INT8 bo = 0; // Zero point for B
    MKL_INT32 co = 0;      // Zero point offset for C

    // Perform the matrix multiplication
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
                       M, N, K,
                       alpha,
                       data_A, lda, ao,
                       data_B, ldb, bo,
                       beta,
                       data_C, ldc, &co);
}