#include <stdio.h>

#define FT_SGEMM_MEDIUM_96_M 48
#define FT_SGEMM_MEDIUM_96_N 48
#define FT_SGEMM_MEDIUM_96_K 8
#define FT_SGEMM_MEDIUM_96_WARP_SIZE 32
#define FT_SGEMM_MEDIUM_96_ROWS_PER_WARP 16
#define FT_SGEMM_MEDIUM_96_THREAD_ROWS 4
#define FT_SGEMM_MEDIUM_96_THREAD_COLS 6

__global__ __launch_bounds__(96) void ft_sgemm_medium_96(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){
    __shared__ float tileA[FT_SGEMM_MEDIUM_96_M * FT_SGEMM_MEDIUM_96_K];
    __shared__ float tileB[FT_SGEMM_MEDIUM_96_N * FT_SGEMM_MEDIUM_96_K];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warp_id = tx / FT_SGEMM_MEDIUM_96_WARP_SIZE;
    const int lane_id = tx & (FT_SGEMM_MEDIUM_96_WARP_SIZE - 1);

    const int block_row = bx * FT_SGEMM_MEDIUM_96_M;
    const int block_col = by * FT_SGEMM_MEDIUM_96_N;

    const int warp_row = warp_id * FT_SGEMM_MEDIUM_96_ROWS_PER_WARP;
    const int thread_row = (lane_id / 8) * FT_SGEMM_MEDIUM_96_THREAD_ROWS;
    const int thread_col = (lane_id % 8) * FT_SGEMM_MEDIUM_96_THREAD_COLS;

    float acc[FT_SGEMM_MEDIUM_96_THREAD_ROWS * FT_SGEMM_MEDIUM_96_THREAD_COLS];
    #pragma unroll
    for (int i = 0; i < FT_SGEMM_MEDIUM_96_THREAD_ROWS * FT_SGEMM_MEDIUM_96_THREAD_COLS; ++i) {
        acc[i] = 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += FT_SGEMM_MEDIUM_96_K) {
        const int linear = tx * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = linear + i;
            const int row = idx % FT_SGEMM_MEDIUM_96_M;
            const int kk = idx / FT_SGEMM_MEDIUM_96_M;

            float a_val = 0.0f;
            float b_val = 0.0f;

            if ((block_row + row) < M && (k0 + kk) < K) {
                a_val = A[(k0 + kk) * M + (block_row + row)];
            }
            if ((block_col + row) < N && (k0 + kk) < K) {
                b_val = B[(k0 + kk) * N + (block_col + row)];
            }

            tileA[kk * FT_SGEMM_MEDIUM_96_M + row] = a_val;
            tileB[kk * FT_SGEMM_MEDIUM_96_N + row] = b_val;
        }

        __syncthreads();

        if ((block_row + warp_row + thread_row) < M && (block_col + thread_col) < N) {
            #pragma unroll
            for (int kk = 0; kk < FT_SGEMM_MEDIUM_96_K; ++kk) {
                float a_frag[FT_SGEMM_MEDIUM_96_THREAD_ROWS];
                float b_frag[FT_SGEMM_MEDIUM_96_THREAD_COLS];

                #pragma unroll
                for (int i = 0; i < FT_SGEMM_MEDIUM_96_THREAD_ROWS; ++i) {
                    const int row = warp_row + thread_row + i;
                    a_frag[i] = ((block_row + row) < M) ? tileA[kk * FT_SGEMM_MEDIUM_96_M + row] : 0.0f;
                }

                #pragma unroll
                for (int j = 0; j < FT_SGEMM_MEDIUM_96_THREAD_COLS; ++j) {
                    const int col = thread_col + j;
                    b_frag[j] = ((block_col + col) < N) ? tileB[kk * FT_SGEMM_MEDIUM_96_N + col] : 0.0f;
                }

                #pragma unroll
                for (int i = 0; i < FT_SGEMM_MEDIUM_96_THREAD_ROWS; ++i) {
                    #pragma unroll
                    for (int j = 0; j < FT_SGEMM_MEDIUM_96_THREAD_COLS; ++j) {
                        acc[i * FT_SGEMM_MEDIUM_96_THREAD_COLS + j] += a_frag[i] * b_frag[j];
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FT_SGEMM_MEDIUM_96_THREAD_ROWS; ++i) {
        const int row = block_row + warp_row + thread_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < FT_SGEMM_MEDIUM_96_THREAD_COLS; ++j) {
                const int col = block_col + thread_col + j;
                if (col < N) {
                    const int c_idx = col * M + row;
                    C[c_idx] = alpha * acc[i * FT_SGEMM_MEDIUM_96_THREAD_COLS + j] + beta * C[c_idx];
                }
            }
        }
    }
}
