/*
 * Copyright Paul Kurucz 2007 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#include <stdio.h>
#include <cuda.h>
#include "ccoreio.h"

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* Uncomment the define below to display the    */
/* processing timestamps.  (Linux Only)         */
//#define DISPLAY_TIMESTAMPS

#ifdef DISPLAY_TIMESTAMPS
#include <sys/time.h>
#endif

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

const char *r72cuda1_cu(void) {
        return "@(#)$Id: r72cuda1.cu,v 1.1 2008/11/22 07:58:51 jlawson Exp $";
}

#define P 0xB7E15163
#define Q 0x9E3779B9

#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_cuda_1( RC5_72UnitWork *, u32 *, void * );
#endif

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------              Local Variables               --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

#if 0
typedef struct
{
        struct {u32 hi,lo;} plain;  /* plaintext (already mixed with iv!) */
        struct {u32 hi,lo;} cypher; /* cyphertext */
        struct {u32 hi,mid,lo;} L0; /* key, changes with every unit *
PIPELINE_COUNT. */
        struct {u32 count; u32 hi,mid,lo;} check; /* counter-measure check */
} DNETC_PACKED RC5_72UnitWork;
#endif

/* Type decaration for the L0 field of the      */
/* RC5_72UnitWork structure.                    */
typedef struct {
        u32 hi;
        u32 mid;
        u32 lo;
} L0_t;

/* The number of GPU threads per thread block   */
/* to execute.  The default value of 64 makes   */
/* optimum usage of __shared__ multiprocessor   */
/* memory.  The maximum value is 512.           */
const u32 num_threads = 96;


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------     Local Helper Function Prototypes       --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

static __host__ __device__ u32 swap_u32(u32 num);

static __host__ __device__ u8 add_u32(u32 num1, u32 num2, u32 * result);

static __host__ __device__ void increment_L0(u32 * hi, u32 * mid, u32* lo, u32 amount);

static __global__ void cuda_core(const u32 plain_hi, const u32 plain_lo,
                                 const u32 cypher_hi, const u32 cypher_lo,
                                 const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                                 const u32 process_amount, u8 *results, u8 * match_found);

#ifdef DISPLAY_TIMESTAMPS
static __inline int64_t linux_read_counter(void);
#endif

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------              Core Entry Point              --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

s32 CDECL rc5_72_unit_func_cuda_1(RC5_72UnitWork *rc5_72unitwork, u32
*iterations, void * /*memblk*/)
{
        u32 i;
        u32 grid_dim;
        s32 retval = RESULT_NOTHING;

        /* Local and GPU variable pairs */
        u8 match_found;
        u8 * cuda_match_found = NULL;
        u8 * results = NULL;
        u8 * cuda_results = NULL;

#ifdef DISPLAY_TIMESTAMPS
        int64_t current_ts;
        int64_t prev_ts;
#endif

//      fprintf(stderr, "\r\nRC5 cuda: thread=%i, iterations=%i\r\n",rc5_72unitwork->threadnum, *iterations);

        cudaSetDevice(rc5_72unitwork->threadnum);

        /* Determine the grid dimensionality based on the */
        /* number of iterations.                          */
        grid_dim = (*iterations / num_threads) + 1;
        if(grid_dim > 65535) {
                grid_dim = 65535;
        }

        /* --------------------------------------------- */

        /* Allocate the cuda_match_found variable */
        if( cudaMalloc((void **)&cuda_match_found, sizeof(u8)) != CUDA_SUCCESS ) {
                retval = -1;
                fprintf(stderr, "RC5 cuda: ERROR: cudaMalloc: cuda_match_found\r\n");
                goto error_exit;
        }

        /* Allocate the results arrays */
        results = (u8 *)malloc(grid_dim * num_threads);
        if( results == NULL ) {
                retval = -1;
                fprintf(stderr, "RC5 cuda: ERROR: malloc\r\n");
                goto error_exit;
        }

        if( cudaMalloc((void **)&cuda_results, grid_dim * num_threads) != CUDA_SUCCESS ) {
                retval = -1;
                fprintf(stderr, "RC5 cuda: ERROR: cudaMalloc: cuda_results\r\n");
                goto error_exit;
        }

        /* --------------------------------------------- */

#ifdef DISPLAY_TIMESTAMPS
        prev_ts = linux_read_counter();
#endif

        for(i = 0; i < *iterations; i += (grid_dim * num_threads)) {
                dim3 block_dimension(num_threads);
                dim3 grid_dimension(grid_dim);
                u32 process_amount = *iterations - i;
                u32 j;
                u32 match_count = 0;

                /* Determine the amount of keys that we */
                /* need to process on this pass through */
                /* the for() loop.                      */
                if(process_amount > (grid_dim * num_threads)) {
                        process_amount = (grid_dim * num_threads);
                }

                /* Clear the match_found variable */
                if( cudaMemset(cuda_match_found, 0, sizeof(u8)) != CUDA_SUCCESS ) {
                        retval = -1;
                        fprintf(stderr, "RC5 cuda: ERROR: cudaMemset: cuda_match_found\r\n");
                        goto error_exit;
                }

                /* Clear the results array */
                if( cudaMemset(cuda_results, 0, grid_dim * num_threads) != CUDA_SUCCESS ) {
                        retval = -1;
                        fprintf(stderr, "RC5 cuda: ERROR: cudaMemset: cuda_results\r\n");
                        goto error_exit;
                }

#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_1=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif

                /* Execute the CUDA core */
                cuda_core<<<grid_dimension, block_dimension>>>(
			rc5_72unitwork->plain.hi, rc5_72unitwork->plain.lo,
			rc5_72unitwork->cypher.hi, rc5_72unitwork->cypher.lo,
			rc5_72unitwork->L0.hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.lo,
			process_amount, cuda_results, cuda_match_found);
                {
                        cudaError_t last_error = cudaGetLastError();
                        if(last_error != CUDA_SUCCESS) {
                                retval = -1;
                                fprintf(stderr, "RC5 cuda: CUDA CORE ERROR: %s\r\n", cudaGetErrorString(last_error));
                                goto error_exit;
                        }
                }

#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_2=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif

                /* Copy the match_found variable to the host */
                if( cudaMemcpy((void *)&match_found, (void*)cuda_match_found, sizeof(u8), cudaMemcpyDeviceToHost) != CUDA_SUCCESS ) {
                        retval = -1;
                        fprintf(stderr, "RC5 cuda: ERROR: cudaMemcpy: cuda_match_found\r\n");
                        goto error_exit;
                }

#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_3=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif

                /* Optimization: Only scan the results[] if  */
                /* the match_found flag (indicating an exact */
                /* or partial match) is set.                 */
                if(match_found) {

                        /* Copy the results[] array to the host */
                        if( cudaMemcpy((void *)results, (void*)cuda_results, process_amount, cudaMemcpyDeviceToHost) != CUDA_SUCCESS ) {
                                retval = -1;
                                fprintf(stderr, "RC5 cuda: ERROR: cudaMemcpy: cuda_results\r\n");
                                goto error_exit;
                        }

                        /* Check the results array for any matches. */
                        for(j = 0; j < process_amount; j++) {

                                /* Check if we have found a partial match */
                                if(results[j] > 0) {
                                        rc5_72unitwork->check.count++;
                                        match_count++;

                                        /* Copy over the current key */
                                        rc5_72unitwork->check.hi = rc5_72unitwork->L0.hi;
                                        rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
                                        rc5_72unitwork->check.lo = rc5_72unitwork->L0.lo;

                                        /* Offset the key index to match out current position */

					increment_L0(&rc5_72unitwork->check.hi, &rc5_72unitwork->check.mid, &rc5_72unitwork->check.lo, j);

                                        /* Check if we have found an exact match */
                                        if(results[j] > 1) {
                                                /* Correct the L0 offste value */

						increment_L0(&rc5_72unitwork->L0.hi, &rc5_72unitwork->L0.mid, &rc5_72unitwork->L0.lo, j);

                                                /* Pass back the iterations count to the callee */
                                                *iterations = i + j;

                                                /* Update the return value and jump to the exit point */
                                                retval = RESULT_FOUND;
                                                goto sucess_exit;
                                        }
                                }

                        } /* for(... j < process_amount ...) */

                } /* if(match_found) */

                /* Advance L0 by the amount that we processed */
                /* this pass through the for() loop.          */
                increment_L0(&rc5_72unitwork->L0.hi, &rc5_72unitwork->L0.mid, &rc5_72unitwork->L0.lo, process_amount);

#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_4=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif
        }

sucess_exit:
error_exit:
#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_5=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif

        if(cuda_match_found) {
                cudaFree(cuda_match_found);
        }

        if(cuda_results) {
                cudaFree(cuda_results);
        }

        if(results) {
                free(results);
        }

        if(retval == -1) {
                cudaError_t last_error = cudaGetLastError();
                fprintf(stderr, "RC5 cuda: error_exit\r\n");
                fprintf(stderr, "RC5 cuda: ERROR: %s\r\n", cudaGetErrorString(last_error));
                fflush(stderr);
        }

#ifdef DISPLAY_TIMESTAMPS
                current_ts = linux_read_counter();
                fprintf(stderr, "RC5 cuda: elapsed_time_6=%lli\r\n", current_ts - prev_ts);
                prev_ts = current_ts;
#endif

//      fprintf(stderr, "RC5 cuda: thread=%i EXIT\r\n",rc5_72unitwork->threadnum);

        return retval;
}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------           Local Helper Functions           --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* u32 byte swap */
static __host__ __device__ u32 swap_u32(u32 num)
{
        u32 retval = (num & 0xFF000000) >> 24;
        retval |= (num & 0x00FF0000) >> 8;
        retval |= (num & 0x0000FF00) << 8;
        retval |= (num & 0x000000FF) << 24;

        return retval;
}

/* Adds two u32s, returning the carry out bit.  */
static __host__ __device__ u8 add_u32(u32 num1, u32 num2, u32 * result)
{
        u8 carry = 0;
        u32 temp = num1;

        temp += num2;

        /* Check for an overflow */
        if(temp < num1) {
                carry = 1;
        }

        /* Pass back the result */
        *result = temp;

        return carry;
}

/* Increments the hi, mid and lo parts of the   */
/* L0 by the specified amount.                  */
static __host__ __device__ void increment_L0(u32 * hi, u32 * mid, u32
* lo, u32 amount)
{
        u32 temp;
        u32 result;
        u8 carry;

        /* Low uint32 */
        temp = *hi & 0xFF;
        temp |= swap_u32(*mid) << 8;
        carry = add_u32(temp, amount, &result);
        *hi = result & 0xFF;
        *mid &= 0x000000FF;
        *mid |= swap_u32(result >> 8);

        /* Mid uint32 */
        if(carry) {
                temp = *mid & 0xFF;
                temp |= swap_u32(*lo) << 8;
                carry = add_u32(temp, 1, &result);
                *mid &= 0xFFFFFF00;
                *mid |= result & 0xFF;
                *lo &= 0x000000FF;
                *lo |= swap_u32(result >> 8);
        }

        if(carry) {
                temp = *lo & 0xFF;
                carry = add_u32(temp, 1, &result);
                *lo &= 0xFFFFFF00;
                *lo |= result & 0xFF;
        }
}

/* Linux Only: Return the current uSec count */
#ifdef DISPLAY_TIMESTAMPS
static __inline int64_t linux_read_counter(void)
{
        struct timeval tv;
        int64_t retval = 0;

        gettimeofday(&tv, NULL);

        retval = (((int64_t)tv.tv_sec) * 1000000) + tv.tv_usec;

        return retval;
}
#endif /* ifdef DISPLAY_TIMESTAMPS */


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------                  GPU Core                  --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

__global__ void cuda_core(const u32 plain_hi, const u32 plain_lo,
                          const u32 cypher_hi, const u32 cypher_lo,
                          const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                          const u32 process_amount, u8 * results, u8 *match_found)
{
        /* Grid of blocks dimension */
//      int gd = gridDim.x;

        /* Block index */
        int bx = blockIdx.x;

        /* Block of threads dimension */
        int bd = blockDim.x;

        /* Thread index */
        int tx = threadIdx.x;

        /* RC5 local state variables */
        u32 A, B;
        u32 S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12,
          S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24, S25;
        u32 L[3];

        /* Drop out early if we don't have any data to process */
        if( ((bx * bd) + tx) > process_amount) {
                /* Warning... Make sure you DON'T use  */
                /* __syncthreads() anywhere after this */
                /* point in the core!!!                */
                return;
        }

        /* Initialize the S[] with constants */
#define KEY_INIT(_i) S##_i = (P + _i*Q)
        KEY_INIT(0);
        KEY_INIT(1);
        KEY_INIT(2);
        KEY_INIT(3);
        KEY_INIT(4);
        KEY_INIT(5);
        KEY_INIT(6);
        KEY_INIT(7);
        KEY_INIT(8);
        KEY_INIT(9);
        KEY_INIT(10);
        KEY_INIT(11);
        KEY_INIT(12);
        KEY_INIT(13);
        KEY_INIT(14);
        KEY_INIT(15);
        KEY_INIT(16);
        KEY_INIT(17);
        KEY_INIT(18);
        KEY_INIT(19);
        KEY_INIT(20);
        KEY_INIT(21);
        KEY_INIT(22);
        KEY_INIT(23);
        KEY_INIT(24);
        KEY_INIT(25);

        /* Initialize L0[] based on our block    */
        /* and thread index.                     */
        L[2] = L0_hi;
        L[1] = L0_mid;
        L[0] = L0_lo;
        increment_L0(&L[2], &L[1], &L[0], (bx * bd) + tx);

        /* ------------------------------------- */
        /* ------------------------------------- */
        /* ------------------------------------- */

#define ROTL_BLOCK_k(i, j, k) ROTL_BLOCK_k##k (i, j)

#define ROTL_BLOCK_k0(i, j) \
        S##i = ROTL3(S##i+(S##j+L[2])); \
        L[0] = ROTL(L[0]+(S##i+L[2]),(S##i+L[2])); \

#define ROTL_BLOCK_k1(i, j) \
        S##i = ROTL3(S##i+(S##j+L[0])); \
        L[1] = ROTL(L[1]+(S##i+L[0]),(S##i+L[0])); \

#define ROTL_BLOCK_k2(i, j) \
        S##i = ROTL3(S##i+(S##j+L[1])); \
        L[2] = ROTL(L[2]+(S##i+L[1]),(S##i+L[1])); \

#define ROTL_BLOCK_i0_j1 \
        S0 = ROTL3(S0+(S25+L[0])); \
        L[1] = ROTL(L[1]+(S0+L[0]),(S0+L[0])); \

#define ROTL_BLOCK_i0_j2 \
        S0 = ROTL3(S0+(S25+L[1])); \
        L[2] = ROTL(L[2]+(S0+L[1]),(S0+L[1])); \

        /* ---------- */

        S0 = ROTL3(S0);
        L[0] = ROTL(L[0]+S0,S0);

        /* ---------- */

        ROTL_BLOCK_k(1, 0, 1);
        ROTL_BLOCK_k(2, 1, 2);
        ROTL_BLOCK_k(3, 2, 0);
        ROTL_BLOCK_k(4, 3, 1);
        ROTL_BLOCK_k(5, 4, 2);
        ROTL_BLOCK_k(6, 5, 0);
        ROTL_BLOCK_k(7, 6, 1);
        ROTL_BLOCK_k(8, 7, 2);
        ROTL_BLOCK_k(9, 8, 0);
        ROTL_BLOCK_k(10, 9, 1);
        ROTL_BLOCK_k(11, 10, 2);
        ROTL_BLOCK_k(12, 11, 0);
        ROTL_BLOCK_k(13, 12, 1);
        ROTL_BLOCK_k(14, 13, 2);
        ROTL_BLOCK_k(15, 14, 0);
        ROTL_BLOCK_k(16, 15, 1);
        ROTL_BLOCK_k(17, 16, 2);
        ROTL_BLOCK_k(18, 17, 0);
        ROTL_BLOCK_k(19, 18, 1);
        ROTL_BLOCK_k(20, 19, 2);
        ROTL_BLOCK_k(21, 20, 0);
        ROTL_BLOCK_k(22, 21, 1);
        ROTL_BLOCK_k(23, 22, 2);
        ROTL_BLOCK_k(24, 23, 0);
        ROTL_BLOCK_k(25, 24, 1);

        /* ---------- */

        ROTL_BLOCK_i0_j2;

        /* ---------- */

        ROTL_BLOCK_k(1, 0, 0);
        ROTL_BLOCK_k(2, 1, 1);
        ROTL_BLOCK_k(3, 2, 2);
        ROTL_BLOCK_k(4, 3, 0);
        ROTL_BLOCK_k(5, 4, 1);
        ROTL_BLOCK_k(6, 5, 2);
        ROTL_BLOCK_k(7, 6, 0);
        ROTL_BLOCK_k(8, 7, 1);
        ROTL_BLOCK_k(9, 8, 2);
        ROTL_BLOCK_k(10, 9, 0);
        ROTL_BLOCK_k(11, 10, 1);
        ROTL_BLOCK_k(12, 11, 2);
        ROTL_BLOCK_k(13, 12, 0);
        ROTL_BLOCK_k(14, 13, 1);
        ROTL_BLOCK_k(15, 14, 2);
        ROTL_BLOCK_k(16, 15, 0);
        ROTL_BLOCK_k(17, 16, 1);
        ROTL_BLOCK_k(18, 17, 2);
        ROTL_BLOCK_k(19, 18, 0);
        ROTL_BLOCK_k(20, 19, 1);
        ROTL_BLOCK_k(21, 20, 2);
        ROTL_BLOCK_k(22, 21, 0);
        ROTL_BLOCK_k(23, 22, 1);
        ROTL_BLOCK_k(24, 23, 2);
        ROTL_BLOCK_k(25, 24, 0);

        /* ---------- */

        ROTL_BLOCK_i0_j1;

        /* ---------- */

        ROTL_BLOCK_k(1, 0, 2);
        ROTL_BLOCK_k(2, 1, 0);
        ROTL_BLOCK_k(3, 2, 1);
        ROTL_BLOCK_k(4, 3, 2);
        ROTL_BLOCK_k(5, 4, 0);
        ROTL_BLOCK_k(6, 5, 1);
        ROTL_BLOCK_k(7, 6, 2);
        ROTL_BLOCK_k(8, 7, 0);
        ROTL_BLOCK_k(9, 8, 1);
        ROTL_BLOCK_k(10, 9, 2);
        ROTL_BLOCK_k(11, 10, 0);
        ROTL_BLOCK_k(12, 11, 1);
        ROTL_BLOCK_k(13, 12, 2);
        ROTL_BLOCK_k(14, 13, 0);
        ROTL_BLOCK_k(15, 14, 1);
        ROTL_BLOCK_k(16, 15, 2);
        ROTL_BLOCK_k(17, 16, 0);
        ROTL_BLOCK_k(18, 17, 1);
        ROTL_BLOCK_k(19, 18, 2);
        ROTL_BLOCK_k(20, 19, 0);
        ROTL_BLOCK_k(21, 20, 1);
        ROTL_BLOCK_k(22, 21, 2);
        ROTL_BLOCK_k(23, 22, 0);
        ROTL_BLOCK_k(24, 23, 1);
        ROTL_BLOCK_k(25, 24, 2);

        /* ---------- */

        A = plain_lo + S0;
        B = plain_hi + S1;

        /* ---------- */

#define FINAL_BLOCK_k(i, j) \
        A = ROTL(A^B,B)+S##i; \
        B = ROTL(B^A,A)+S##j;

        FINAL_BLOCK_k(2, 3);
        FINAL_BLOCK_k(4, 5);
        FINAL_BLOCK_k(6, 7);
        FINAL_BLOCK_k(8, 9);
        FINAL_BLOCK_k(10, 11);
        FINAL_BLOCK_k(12, 13);
        FINAL_BLOCK_k(14, 15);
        FINAL_BLOCK_k(16, 17);
        FINAL_BLOCK_k(18, 19);
        FINAL_BLOCK_k(20, 21);
        FINAL_BLOCK_k(22, 23);
        FINAL_BLOCK_k(24, 25);

        /* ------------------------------------- */
        /* ------------------------------------- */
        /* ------------------------------------- */

        /* Check the results for a match.        */
        if (A == cypher_lo) {

                /* Set the match_found flag */
                *match_found = 1;

                /* Record the "check_*" match   */
                /* in the results array.        */
                results[(bx * bd) + tx] = 1;

                if (B == cypher_hi) {
                        /* Record the RESULT_FOUND match  */
                        /* in the results array.          */
                        results[(bx * bd) + tx] = 2;
                }
        }
}
