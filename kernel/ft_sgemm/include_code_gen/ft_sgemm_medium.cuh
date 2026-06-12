 
#include <stdio.h>  
#define tab(t, a, b)t.x += a.x * b;t.y += a.y * b;  t.z += a.z * b;t.w += a.w * b;  

#define tcab(t, c, alpha, beta) \
    c.x = alpha * t.x + beta * c.x; \
    c.y = alpha * t.y + beta * c.y; \
    c.z = alpha * t.z + beta * c.z; \
    c.w = alpha * t.w + beta * c.w;
    
// #define DISABLE_ERROR_CORRECTION

__global__  __launch_bounds__(64) void ft_sgemm_medium(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta){ // , float *check_A_col, float *check_B_row, int *debug_int){
    // ms = 128, ns = 32, ks = 8
    // mw = 64, nw = 16
    // mr = 8, nr = 4
    // blockId, warpId, and threadIdx

    int ms = 32, ns = 32, ks = 8, mw = 16, nw = 32, mr = 4, nr = 4;
    
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 64 * 8, col major
    // tile B size = ns x ks = 64 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 2560 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    
    __shared__ float sAB[1024]; 

    // J - DEBUGGING START
    __shared__ float A_col[8]; // [ks=8]
    __shared__ float B_row[8]; // [ks=8]
    // J - DEBUGGIN END
    
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks; // J - 2 * 32 * 8 = 512, buffer_B starts from sAB[512]
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (128 * 8) / 128 = 8 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x); // J - 32 * 8 / 64 = 4 floats per thread -> I call a set of 4 floats a "slice" of a column
    // number of threads to load a column of tile A: 128 floats / 8 floats = 16 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread); // J - 32 / 4 = 8 threads per column of tile A -> 8 slices of a column
    // parameter for error injection
    int tx_injec = 17;
    float err_bound1 =9500.0;
    float error_inject = 10000.0;
    
    
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 8, (tx % 16 threads) * 8 + 7],
    //                              col  = (tx / 16 threads) of tile A
    // J - Point to starting element of A matrix (in global memory) for tile A for each thread
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M; // J - (tx % 8) * 4 + (int(tx / 8)) * M

    // tile B inner offset.
    // each thread load (32 * 8) / 128 = 2 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x); // J - 32 * 8 / 64 = 4 floats per thread -> I call a set of 4 floats a "slice" of a column
    // number of threads to load a column of tile B: 32 floats / 2 floats = 16 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread); // J - 32 / 4 = 8 threads per column of tile B -> 8 slices of a column
    // thread tx load 8 floats with rows = [(tx % 16 threads) * 2, (tx % 16 threads) * 2 + 1],
    //                              col  = (tx / 16 threads) of tile A
    // J - Point to starting element of B matrix (in global memory) for tile B for each thread
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N; // J - (tx % 8) * 4 + (int(tx / 8)) * N

    // prefetch the vector from A and B in global memory 
    // 
    // float{4 if ms * ks / num_thread >= 4 else 2} prefetch_vector_tile_A[{ms * ks / (4 * num_thread)}];
    // float{4 if ns * ks / num_thread >= 4 else 2} prefetch_vector_tile_B[{ns * ks / (4 * num_thread)}]
    float4 prefetch_vector_tile_A[1];
    float4 prefetch_vector_tile_B[1];
    prefetch_vector_tile_A[0] = *((float4*)A + 0); // J - Copy 4 floats from global memory to register for tile A
    prefetch_vector_tile_B[0] = *((float4*)B + 0); // J - Copy 4 floats from global memory to register for tile B
    
    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1); // J - (0 / 8) & 1 = 0
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks; // J - float* buffer_A = (float*)(sAB) + 0 + 0 = sAB
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks; // J - float* buffer_B = (float*)(sAB) + 512 + 0 = sAB + 512

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + 1 * tx + 0) = prefetch_vector_tile_A[0]; // J - *((float4*)sAB + tx) = (4 floats of tile A for this tx)
    *(((float4*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0]; // J - *((float4*)sAB + 512 + tx) = (4 floats of tile B for this tx)

    __syncthreads();
    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms / mw); // J - 32 / 16 = 2 warps along A vector
    int num_warp_B = int(ns / nw); // J - 32 / 32 = 1 warp along B vector
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32); // J - 0 for threads [0-31], 1 for threads [32-63]
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp / num_warp_A; // J - 0 for threads [0-63] because num_warp_A=2
    int idA_warp = int(id_warp % num_warp_A); // J - 0 for threads [0-31], 1 for threads [32-63] because num_warp_A=2
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw
    // offset vec B = 2D warp idB * nw
    int offset_vec_A_warp = idA_warp * mw; // J - 0 for threads [0-31], 16 for threads [32-63]
    int offset_vec_B_warp = idB_warp * nw; // J - 0 for threads [0-63] because idB_warp=0


    //2D thread idB = tx % (nw / nr)
    //          idA = tx / (nw / nr)
    int idB_thread = ((tx & 31) / ((int)(mw / mr))); // J - 0 for threads [0-7, 32-39], 1 for threads [8-15, 40-47], 2 for threads [16-23, 48-55], 3 for threads [24-31, 56-63] because (mw/nr)=4
    int idA_thread = int((tx & 31) % (mw / mr)); // J - 0 for threads [0,4,8,12,16,36,40,44], 1 for threads [1,5,9,13,17,37,41,45], 2 for threads [2,6,10,14,18,38,42,46], 3 for threads [3,7,11,15,19,39,43,47] because (mw/nr)=4

    // offset for the threads
    // offset vec A = 2D thread idA * mr
    // offset vec B = 2D thread idA * nr
    int offset_vec_A_thread = idA_thread * mr; // J - 0 for threads [0,4,8,12,16,36,40,44], 4 for threads [1,5,9,13,17,37,41,45], 8 for threads [2,6,10,14,18,38,42,46], 12 for threads [3,7,11,15,19,39,43,47] because idA_thread * mr (mr=4)
    int offset_vec_B_thread = idB_thread * nr; // J - 0 for threads [0-7, 32-39], 4 for threads [8-15, 40-47], 8 for threads [16-23, 48-55], 12 for threads [24-31, 56-63] because idB_thread * nr (nr=4)

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr and nr
    // prefetch with the double buffer
    float4 vec_A[2];
    float4 vec_B[2];
    float4 tmp_row[1];
    float4 tmp_col[1];
    float res[16];
    float C_c[4];
    float C_r[4];
    
    memset(res, 0, sizeof(res));
    // initial outer product column
    int kk = -1;
    
    // offset of register store for prefetching
    int offset_prefetch_register_kk = ((kk + 1) & 1);
    
    // offset of register to use 
    int offset_register_kk = 0;
    
    // offset of vec A and vec B w.r.t kk:
    int offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
    int offset_load_vec_B_kk = ((kk + 1) % ks) * ns;

    // load the vectors from buffer to registers
    vec_A[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
    vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
    
    // ABFT
    float4 block_level_A_c[1];
    float4 block_level_B_r[1];
    float A_c = 0., B_r = 0.;
    
    // Calculate "column checksum" from "4 floats prefetched from global memory" of "tile A" for the current "tx". Do same for "row checksum" for "tile B".
    // For a tile of A (also B), this takes 8 threads to calculate for one column (according to load_tile_B_num_threads_one_col and load_tile_B_num_floats_one_thread)
    // J - Here A_c is the checksum for one slice of a column
    A_c += prefetch_vector_tile_A[0].x; A_c += prefetch_vector_tile_A[0].y; A_c += prefetch_vector_tile_A[0].z; A_c += prefetch_vector_tile_A[0].w; 
    B_r += prefetch_vector_tile_B[0].x; B_r += prefetch_vector_tile_B[0].y; B_r += prefetch_vector_tile_B[0].z; B_r += prefetch_vector_tile_B[0].w; 
    
    // J - Accumulate A_c for whole column of tile A, for this get A_c from the other 8 threads working on the same column of tile A (i.e. 8 threads [0,1,3,4,8,32,64,128] for column 0, 8 threads [2**8...2**15] for column 1, 8 threads [2**16...2**23] for column 2, until all 32 threads from warp, then restart for next warp)
    // J - Here A_c is the checksum for a whole column of tile A
    A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32); // Share value of A_c from src thread to des thread
    A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
    A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);

    // J - Here B_r is the checksum for a whole row of tile B (actually a column due them not transposing B and still using column major indexing)
    B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
    B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);

    // Column checksum accumulation - START
    // const int col_tile = (int)(tx / load_tile_A_num_threads_one_col); // Max value: 64 / 8 = 8
    // const int col_lane = tx % load_tile_A_num_threads_one_col;

    // if (col_lane == 0)
    // {
    //     A_col[col_tile] = A_c; // Accumulate column checksum over tile A
    // }
    // __syncthreads();

    // // Once a block is finished, accumulate all column checksums of the block to the column checksums of the global matrix
    // if (by == 0 && tx < ks)
    // {
    //     atomicAdd(&check_A_col[tx], A_col[tx]);
    // }
    // __syncthreads();
    // Column checksum accumulation - END
    // Row checksum accumulation - START
    // const int row_tile = (int)(tx / load_tile_B_num_threads_one_col); // Max value: 64 / 8 = 8
    // const int row_lane = tx % load_tile_B_num_threads_one_col;

    // if (row_lane == 0)
    // {
    //     B_row[row_tile] = B_r; // Accumulate row checksum over tile B
    // }
    // __syncthreads();

    // if (bx == 0 && tx < ks)
    // {
    //     atomicAdd(&check_B_row[tx], B_row[tx]);
    // }
    // __syncthreads();
    // Row checksum accumulation - END

        // saxpy
    // J - Multiply each element from column of tile A with the "row" checksum of tile B
    // J - This provides the block level column and row checksum for C Matrix
    block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r;
    block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r; 
    block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r; 
    block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r; 
    
    block_level_B_r[0].x = prefetch_vector_tile_B[0].x * A_c; 
    block_level_B_r[0].y = prefetch_vector_tile_B[0].y * A_c; 
    block_level_B_r[0].z = prefetch_vector_tile_B[0].z * A_c; 
    block_level_B_r[0].w = prefetch_vector_tile_B[0].w * A_c; 
    
    // store into buffer

    // offset to store the saxpy result
    int offset_store_checksum = (((k / ks) + 1) & 1); // J - ((0 / 8) + 1) & 1 = 1
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks; // J - (float*) checksum_buffer_A = (float*)sAB + 0 + 1 * 32 * 8 = sAB + 256
    float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks; // J - (float*) checksum_buffer_B = (float*)sAB + 512 + 1 * 32 * 8 = sAB + 768
    
    *(((float4*)checksum_buffer_A) + tx * 1 + 0) = block_level_A_c[0]; // J - *((float4*)checksum_buffer_A + tx) = block_level_A_c[0] (4 floats of block_level_A_c[0] for this tx)
    
    *(((float4*)checksum_buffer_B) + tx * 1 + 0) = block_level_B_r[0]; // J - *((float4*)checksum_buffer_B + tx) = block_level_B_r[0] (4 floats of block_level_B_r[0] for this tx)
    
    __syncthreads(); 
    // offset C checksum each thread
    int offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks); // J - (tx < 32) ? (256) : (768), 256 if for block_level_A_c[0], 768 is for block_level_B_r[0]
    int ws = (tx < (1 * blockDim.x / 2)) ? ms: ns; // J - ws=32
    int ws_ = (tx < (1 * blockDim.x / 2)) ? ns: ms; // J - ws_=32
    int ws_1 = 1;
    int ws_2[1];
    offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
    float checksum[1];
    float checksum_[1];
    ws_2[0] = 0;
    checksum[0] = 0.;
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0 + ws_2[0])); // J - For (tx < 32) use block_level_A_c[0]
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6 + ws_2[0]));
    checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7 + ws_2[0]));
    
    __syncthreads(); 
    // K loop
    for(k = 0; k < K; k += ks){
        // tile A abd tile B global offsets move forward ks columns
        A += ks * M; 
        B += ks * N; 
        // prefetch the vector from A and B in global memory 
        if(k + ks < K){
        prefetch_vector_tile_A[0] = *((float4*)A + 0);  
        prefetch_vector_tile_B[0] = *((float4*)B + 0);  
        
        }
        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1); // J - 1 for kk=1,3,5,7,...; 0 for kk=0,2,4,6,...
            offset_prefetch_register_kk = ((kk + 1) & 1); // J - 1 for kk=0,2,4,6,...; 0 for kk=1,3,5,7,...
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms; // J - ((kk+1) % 8) * 32
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns; // J - ((kk+1) % 8) * 32
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
            vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
            
            // res[0-15] = a sub-tile of C (4x4) calculated by one thread (4x4 tile per thread -> 8x8=64 threads together calculate one tile of C 32x32)
            // res[0-3] = 1st row of tile C * 4 columns of tile C
            res[0 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].x;
            res[1 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].y;
            res[2 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].z;
            res[3 ] += vec_A[offset_register_kk * 1 + 0].x * vec_B[offset_register_kk * 1 + 0].w;
            
            // res[4-7] = 2nd row of tile C * 4 columns of tile C
            res[4 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].x;
            res[5 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].y;
            res[6 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].z;
            res[7 ] += vec_A[offset_register_kk * 1 + 0].y * vec_B[offset_register_kk * 1 + 0].w;
            
            // res[8-11] = 3rd row of tile C * 4 columns of tile C
            res[8 ] += vec_A[offset_register_kk * 1 + 0].z * vec_B[offset_register_kk * 1 + 0].x;
            res[9 ] += vec_A[offset_register_kk * 1 + 0].z * vec_B[offset_register_kk * 1 + 0].y;
            res[10] += vec_A[offset_register_kk * 1 + 0].z * vec_B[offset_register_kk * 1 + 0].z;
            res[11] += vec_A[offset_register_kk * 1 + 0].z * vec_B[offset_register_kk * 1 + 0].w;
            
            // res[12-15] = 4th row of tile C * 4 columns of tile C
            res[12] += vec_A[offset_register_kk * 1 + 0].w * vec_B[offset_register_kk * 1 + 0].x;
            res[13] += vec_A[offset_register_kk * 1 + 0].w * vec_B[offset_register_kk * 1 + 0].y;
            res[14] += vec_A[offset_register_kk * 1 + 0].w * vec_B[offset_register_kk * 1 + 0].z;
            res[15] += vec_A[offset_register_kk * 1 + 0].w * vec_B[offset_register_kk * 1 + 0].w;
            

        }
        if(((k+8) %(int(K / 20))) == 0){
            // J -  Turn off error injection
            // if(tx == (int)((k+8) / (int(K / 20)))){
            // res[0] += error_inject;
            // }
            // C row checksum (4x4 tile C)
            C_r[0 ] = res[0 ]; C_r[0 ] += res[1 ]; C_r[0 ] += res[2 ]; C_r[0 ] += res[3 ]; 
            C_r[1 ] = res[4 ]; C_r[1 ] += res[5 ]; C_r[1 ] += res[6 ]; C_r[1 ] += res[7 ]; 
            C_r[2 ] = res[8 ]; C_r[2 ] += res[9 ]; C_r[2 ] += res[10]; C_r[2 ] += res[11]; 
            C_r[3 ] = res[12]; C_r[3 ] += res[13]; C_r[3 ] += res[14]; C_r[3 ] += res[15]; 
            
            // C column checksum (4x4 tile C)
            C_c[0 ] = res[0 ]; C_c[0 ] += res[4 ]; C_c[0 ] += res[8 ]; C_c[0 ] += res[12]; 
            C_c[1 ] = res[1 ]; C_c[1 ] += res[5 ]; C_c[1 ] += res[9 ]; C_c[1 ] += res[13]; 
            C_c[2 ] = res[2 ]; C_c[2 ] += res[6 ]; C_c[2 ] += res[10]; C_c[2 ] += res[14]; 
            C_c[3 ] = res[3 ]; C_c[3 ] += res[7 ]; C_c[3 ] += res[11]; C_c[3 ] += res[15]; 
            
        __syncthreads();
        float* s = ((float*)(sAB) + ((idB_warp) * 8) + (idA_warp * 128) + idB_thread + (idA_thread * 32) + 0);
        float* s_ = ((float*)(sAB) + 256 + ((idA_warp) * 4) + (idB_warp * 256) + idA_thread + (idB_thread * 32) + 0);
        *(s_ + (0 * 8)) = C_c[0];
        *(s_ + (1 * 8)) = C_c[1];
        *(s_ + (2 * 8)) = C_c[2];
        *(s_ + (3 * 8)) = C_c[3];
        
        *(s + (0 * 8)) = C_r[0];
        *(s + (1 * 8)) = C_r[1];
        *(s + (2 * 8)) = C_r[2];
        *(s + (3 * 8)) = C_r[3];
        __syncthreads();
        checksum_[0] =  checksum[0];
        float4 r_;
        if (tx < int(1 * blockDim.x / 2)){
            r_ = *((float4*)((float*)sAB + ((tx& 31) * 1 + 0 ) * 8 + 0));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            r_ = *((float4*)((float*)sAB + ((tx& 31) * 1 + 0 ) * 8 + 4));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            
        }
        else{
            r_ = *((float4*)((float*)sAB + 256 + ((tx & 31) * 1 + 0) * 8 + 0));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            r_ = *((float4*)((float*)sAB + 256 + ((tx & 31) * 1 + 0) * 8 + 4));
            checksum_[0] -= r_.x;
            checksum_[0] -= r_.y;
            checksum_[0] -= r_.z;
            checksum_[0] -= r_.w;
            
        }
        __syncthreads();
        *((float*)sAB + (1 - int(tx / int(blockDim.x / 2))) * ns + (tx % (int(ws / 1))) * 1 + 0) = checksum_[0];
        __syncthreads();
        tmp_col[0] = (*((float4*)((float*)sAB + (idB_warp * 32 + idB_thread * 4) + 0)));
        tmp_row[0] = (*((float4*)((float*)sAB + ns + (idA_warp * 16 + idA_thread * 4) + 0)));
        // Turn of error correction for radiation test
        #ifndef DISABLE_ERROR_CORRECTION
        res[0] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[1] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[2] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 2)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[3] += int( (fabsf(*((float*)tmp_row + 0)) > err_bound1) && (fabsf(*((float*)tmp_col + 3)) > err_bound1)) * (*((float*)tmp_row + 0));
        res[4] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 1));
        res[5] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 1));
        res[6] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 2)) > err_bound1)) * (*((float*)tmp_row + 1));
        res[7] += int( (fabsf(*((float*)tmp_row + 1)) > err_bound1) && (fabsf(*((float*)tmp_col + 3)) > err_bound1)) * (*((float*)tmp_row + 1));
        res[8] += int( (fabsf(*((float*)tmp_row + 2)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 2));
        res[9] += int( (fabsf(*((float*)tmp_row + 2)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 2));
        res[10] += int( (fabsf(*((float*)tmp_row + 2)) > err_bound1) && (fabsf(*((float*)tmp_col + 2)) > err_bound1)) * (*((float*)tmp_row + 2));
        res[11] += int( (fabsf(*((float*)tmp_row + 2)) > err_bound1) && (fabsf(*((float*)tmp_col + 3)) > err_bound1)) * (*((float*)tmp_row + 2));
        res[12] += int( (fabsf(*((float*)tmp_row + 3)) > err_bound1) && (fabsf(*((float*)tmp_col + 0)) > err_bound1)) * (*((float*)tmp_row + 3));
        res[13] += int( (fabsf(*((float*)tmp_row + 3)) > err_bound1) && (fabsf(*((float*)tmp_col + 1)) > err_bound1)) * (*((float*)tmp_row + 3));
        res[14] += int( (fabsf(*((float*)tmp_row + 3)) > err_bound1) && (fabsf(*((float*)tmp_col + 2)) > err_bound1)) * (*((float*)tmp_row + 3));
        res[15] += int( (fabsf(*((float*)tmp_row + 3)) > err_bound1) && (fabsf(*((float*)tmp_col + 3)) > err_bound1)) * (*((float*)tmp_row + 3));
        #endif
        __syncthreads();
        }
            
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + 1 * tx + 0) = prefetch_vector_tile_A[0];
        *(((float4*)buffer_B) + 1 * tx + 0) = prefetch_vector_tile_B[0];
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk) + 0);
        vec_B[offset_prefetch_register_kk * 1 + 0] = *((float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk) + 0);
        
        // ABFT
        A_c = 0., B_r = 0.;
        
        A_c += prefetch_vector_tile_A[0].x; A_c += prefetch_vector_tile_A[0].y; A_c += prefetch_vector_tile_A[0].z; A_c += prefetch_vector_tile_A[0].w; 
        B_r += prefetch_vector_tile_B[0].x; B_r += prefetch_vector_tile_B[0].y; B_r += prefetch_vector_tile_B[0].z; B_r += prefetch_vector_tile_B[0].w; 
        
        A_c += __shfl_xor_sync(0xffffffff, A_c, 1, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 2, 32);
        A_c += __shfl_xor_sync(0xffffffff, A_c, 4, 32);
        
        B_r += __shfl_xor_sync(0xffffffff, B_r, 1, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 2, 32);
        B_r += __shfl_xor_sync(0xffffffff, B_r, 4, 32);

        // // Column checksum accumulation - START
        // const int col_tile = (int)(tx / load_tile_A_num_threads_one_col); // Max value: 64 / 8 = 8
        // const int col_lane = tx % load_tile_A_num_threads_one_col;

        // if (col_lane == 0)
        // {
        //     A_col[col_tile] = A_c; // Accumulate column checksum over tile A
        // }
        // __syncthreads();

        // // Once a block is finished, accumulate all column checksums of the block to the column checksums of the global matrix
        // if (by == 0 && tx < ks && k + ks < K)
        // {
        //     atomicAdd(&check_A_col[(k + ks) + tx], A_col[tx]); // (k + ks) needed due to initial checksum calculation before the K loop
        // }
        // if (bx == 0 && by == 0 && tx == 0){
        //     debug_int[0] = (k + ks);
        // }
        // __syncthreads();
        // // Column checksum accumulation - END
        // // Row checksum accumulation - START
        // const int row_tile = (int)(tx / load_tile_B_num_threads_one_col); // Max value: 64 / 8 = 8
        // const int row_lane = tx % load_tile_B_num_threads_one_col;

        // if (row_lane == 0)
        // {
        //     B_row[row_tile] = B_r; // Accumulate row checksum over tile B
        // }
        // __syncthreads();

        // if (bx == 0 && tx < ks && k + ks < K)
        // {
        //     atomicAdd(&check_B_row[(k + ks) + tx], B_row[tx]);
        // }
        // if (bx == 0 && by == 0 && tx == 0){
        //     debug_int[1] = (k + ks);
        // }
        // __syncthreads();
        // // Row checksum accumulation - END
        
        // saxpy
        block_level_A_c[0].x = prefetch_vector_tile_A[0].x * B_r; 
        block_level_A_c[0].y = prefetch_vector_tile_A[0].y * B_r; 
        block_level_A_c[0].z = prefetch_vector_tile_A[0].z * B_r; 
        block_level_A_c[0].w = prefetch_vector_tile_A[0].w * B_r; 
        
        block_level_B_r[0].x = prefetch_vector_tile_B[0].x * A_c; 
        block_level_B_r[0].y = prefetch_vector_tile_B[0].y * A_c; 
        block_level_B_r[0].z = prefetch_vector_tile_B[0].z * A_c; 
        block_level_B_r[0].w = prefetch_vector_tile_B[0].w * A_c; 
        
        // store into buffer

        // offset to store the saxpy result
        offset_store_checksum = (((k / ks)) & 1);
        
        // get the pointer to prefetched buffer A and prefetched buffer B
        float* checksum_buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_checksum * ms * ks;
        float* checksum_buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_checksum * ns * ks;
        
        *(((float4*)checksum_buffer_A) + tx * 1 + 0) = block_level_A_c[0];
        
        *(((float4*)checksum_buffer_B) + tx * 1 + 0) = block_level_B_r[0];
        
        __syncthreads(); 
        // offset C checksum each thread
        offset_A_B = (tx < (1 * blockDim.x / 2)) ? (buffer_A_offset + offset_store_checksum * ms * ks): (buffer_B_offset + offset_store_checksum * ns * ks);
        offset_A_B +=  (tx & (int)(ws / ws_1 - 1)) * ws_1;
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 0 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 1 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 2 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 3 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 4 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 5 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 6 + ws_2[0]));
        checksum[0] +=  *(((float*)(sAB) + offset_A_B + ws * 7 + ws_2[0]));
        
    __syncthreads(); 
    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    
    float4 C_res[4];
    
    C_res[0 ] = *((float4 *)(C+ M * 0) + 0 );
    C_res[1 ] = *((float4 *)(C+ M * 1) + 0 );
    C_res[2 ] = *((float4 *)(C+ M * 2) + 0 );
    C_res[3 ] = *((float4 *)(C+ M * 3) + 0 );
    
    C_res[0].x = alpha * res[0  ] + beta * C_res[0].x;
    C_res[0].y = alpha * res[4  ] + beta * C_res[0].y;
    C_res[0].z = alpha * res[8  ] + beta * C_res[0].z;
    C_res[0].w = alpha * res[12 ] + beta * C_res[0].w;
    
    C_res[1].x = alpha * res[1  ] + beta * C_res[1].x;
    C_res[1].y = alpha * res[5  ] + beta * C_res[1].y;
    C_res[1].z = alpha * res[9  ] + beta * C_res[1].z;
    C_res[1].w = alpha * res[13 ] + beta * C_res[1].w;
    
    C_res[2].x = alpha * res[2  ] + beta * C_res[2].x;
    C_res[2].y = alpha * res[6  ] + beta * C_res[2].y;
    C_res[2].z = alpha * res[10 ] + beta * C_res[2].z;
    C_res[2].w = alpha * res[14 ] + beta * C_res[2].w;
    
    C_res[3].x = alpha * res[3  ] + beta * C_res[3].x;
    C_res[3].y = alpha * res[7  ] + beta * C_res[3].y;
    C_res[3].z = alpha * res[11 ] + beta * C_res[3].z;
    C_res[3].w = alpha * res[15 ] + beta * C_res[3].w;
    
    *((float4 *)(C+ M * 0) + 0 ) = C_res[0 ];
    *((float4 *)(C+ M * 1) + 0 ) = C_res[1 ];
    *((float4 *)(C+ M * 2) + 0 ) = C_res[2 ];
    *((float4 *)(C+ M * 3) + 0 ) = C_res[3 ];
    
}
