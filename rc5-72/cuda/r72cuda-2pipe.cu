/* -*-C-*-
 *
 * Copyright Paul Kurucz 2007 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * With modifications by Greg Childers
*/

#include <stdio.h>
#include <cuda.h>
#include "r72cuda.h"
#include "r72cuda-helper.cu"

#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_cuda_2_64( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_2_128( RC5_72UnitWork *, u32 *, void * );
#endif

static __global__ void cuda_2pipe(const u32 plain_hi, const u32 plain_lo,
                                  const u32 cypher_hi, const u32 cypher_lo,
                                  const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                                  const u32 process_amount, u8 * results, u8 * match_found);

static s32 CDECL rc5_72_run_cuda_2(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, int device, u32 num_threads, int waitmode);


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------              Core Entry Point              --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

s32 CDECL rc5_72_unit_func_cuda_2_64(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  /* The number of GPU threads per thread block   */
  /* to execute.  The default value of 64 makes   */
  /* optimum usage of __shared__ multiprocessor   */
  /* memory.  The maximum value is 512.           */
  const u32 num_threads = 64;

  return rc5_72_run_cuda_2(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, num_threads, default_wait_mode);
}

s32 CDECL rc5_72_unit_func_cuda_2_128(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_2(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 128, default_wait_mode);
}

static s32 CDECL rc5_72_run_cuda_2(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, int device, u32 num_threads, int waitmode)
{
  const u32 pipeline_count = 2;
  const u32 max_grid_dim = 65535;
  const u32 optimal_process_amount = num_threads * max_grid_dim * pipeline_count; // optimal GPU utilization during a single GPU core invocation

  int currentdevice;
  u32 i;
  u32 grid_dim;
  s32 retval = RESULT_NOTHING;

  /* Local and GPU variable pairs */
  u8 match_found;
  u8 * cuda_match_found = NULL;
  u8 * results = NULL;
  u8 * cuda_results = NULL;

  cudaStream_t core;
  cudaEvent_t stop;
  cudaError_t last_error;

#ifdef DISPLAY_TIMESTAMPS
  int64_t current_ts;
  int64_t prev_ts;
#endif

  struct timeval tv_core_start;
  struct timeval tv_core_elapsed;

  //fprintf(stderr, "\r\nRC5 cuda: iterations=%i\r\n", *iterations);

  if( cudaGetDevice(&currentdevice) != (cudaError_t) CUDA_SUCCESS ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: cudaGetDevice\r\n");
    goto error_exit;
  }

  if (currentdevice != device) {
    if( cudaSetDevice(device) != (cudaError_t) CUDA_SUCCESS ) {
      retval = -1;
      fprintf(stderr, "RC5 cuda: ERROR: cudaSetDevice\r\n");
      goto error_exit;
    }
  }

  if( cudaStreamCreate(&core) != (cudaError_t) CUDA_SUCCESS ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: cudaStreamCreate\r\n");
    goto error_exit;
  }

  if( cudaEventCreate(&stop) != (cudaError_t) CUDA_SUCCESS ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: cudaEventCreate\r\n");
    goto error_exit;
  }

  /* Determine the grid dimensionality based on the */
  /* number of iterations.                          */
  grid_dim = min_u32((*iterations / pipeline_count + num_threads - 1) / num_threads, max_grid_dim);

  /* --------------------------------------------- */

  /* Allocate the cuda_match_found variable */
  if( cudaMalloc((void **)&cuda_match_found, sizeof(u8)) != (cudaError_t) CUDA_SUCCESS ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: cudaMalloc: cuda_match_found\r\n");
    goto error_exit;
  }

  /* Allocate the results arrays */
  results = (u8 *)malloc(grid_dim * num_threads * 2 * sizeof(u8));
  if( results == NULL ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: malloc\r\n");
    goto error_exit;
  }

  if( cudaMalloc((void **)&cuda_results, grid_dim * num_threads * 2 * sizeof(u8)) != (cudaError_t) CUDA_SUCCESS ) {
    retval = -1;
    fprintf(stderr, "RC5 cuda: ERROR: cudaMalloc: cuda_results\r\n");
    goto error_exit;
  }

  /* --------------------------------------------- */

#ifdef DISPLAY_TIMESTAMPS
  prev_ts = linux_read_counter();
#endif

  for (i = 0; i < *iterations; i += optimal_process_amount) {
    dim3 block_dimension(num_threads);
    dim3 grid_dimension(grid_dim);
    u32 process_amount = min_u32(*iterations - i, optimal_process_amount);
    u32 j;
    u32 match_count = 0;

    /* Clear the match_found variable */
    if( cudaMemset(cuda_match_found, 0, sizeof(u8)) != (cudaError_t) CUDA_SUCCESS ) {
      retval = -1;
      fprintf(stderr, "RC5 cuda: ERROR: cudaMemset: cuda_match_found\r\n");
      goto error_exit;
    }

    /* Clear the results array */
    if( cudaMemset(cuda_results, 0, grid_dim * num_threads * 2 * sizeof(u8)) != (cudaError_t) CUDA_SUCCESS ) {
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

    if (waitmode == 1)
      CliTimer(&tv_core_start);

    cuda_2pipe<<<grid_dimension, block_dimension, 0, core>>>(
      rc5_72unitwork->plain.hi, rc5_72unitwork->plain.lo,
      rc5_72unitwork->cypher.hi, rc5_72unitwork->cypher.lo,
      rc5_72unitwork->L0.hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.lo,
      (process_amount + pipeline_count - 1) / pipeline_count,
      cuda_results, cuda_match_found);
    /* (process_amount+1)/2 just to handle the (impossible) case that process_amount is odd */
    last_error = cudaGetLastError();
    if(last_error != (cudaError_t) CUDA_SUCCESS) {
      retval = -1;
      fprintf(stderr, "RC5 cuda: CUDA CORE ERROR: %s\r\n", cudaGetErrorString(last_error));
      goto error_exit;
    }

    if (cudaEventRecord(stop, core) != CUDA_SUCCESS) {
      retval = -1;
      fprintf(stderr, "RC5 cuda: ERROR: cudaEventRecord\r\n");
      goto error_exit;
    }

#ifdef DISPLAY_TIMESTAMPS
    current_ts = linux_read_counter();
    fprintf(stderr, "RC5 cuda: elapsed_time_2=%lli\r\n", current_ts - prev_ts);
    prev_ts = current_ts;
#endif

    if (waitmode == 0) {
      int slept;
      for (slept = 0; cudaEventQuery(stop) == cudaErrorNotReady; ++slept)
        NonPolledUSleep(min_sleep_interval);
      //fprintf(stderr, "\rRC5 cuda: slept=%d process_amount=%d\n", slept, process_amount);
    } else if (waitmode == 1) {
      static long best_time = -1;
      long expected = 0, elapsed = 0;
      int slept = 0;
      if (best_time <= 0) {
        for (slept = 0; cudaEventQuery(stop) == cudaErrorNotReady; ++slept)
          NonPolledUSleep(min_sleep_interval);
      } else {
        expected = best_time * process_amount / optimal_process_amount;
        if (expected / 2 >= min_sleep_interval) {
          //fprintf(stderr, "\rRC5 cuda: sleep_now=%d\n", expected / 2);
          NonPolledUSleep(expected / 2);
          ++slept;
        }
        while (cudaEventQuery(stop) == cudaErrorNotReady) {
          CliTimerDiff(&tv_core_elapsed, &tv_core_start, NULL);
          elapsed = tv_core_elapsed.tv_sec * 1000000 + tv_core_elapsed.tv_usec;
          long sleep_now = (expected - elapsed) / 2;
          //fprintf(stderr, "\rRC5 cuda: sleep_now=%d\n", sleep_now);
          if (sleep_now < min_sleep_interval) {
            if (cudaEventSynchronize(stop) != CUDA_SUCCESS) {
              retval = -1;
              fprintf(stderr, "RC5 cuda: ERROR: cudaEventSynchronize\r\n");
              goto error_exit;
            }
            break;
          } else {
            NonPolledUSleep(sleep_now);
            ++slept;
          }
        }
      }
      CliTimerDiff(&tv_core_elapsed, &tv_core_start, NULL);
      elapsed = tv_core_elapsed.tv_sec * 1000000 + tv_core_elapsed.tv_usec;
      if (process_amount == optimal_process_amount) {
        // update best speed
        if (best_time <= 0 || best_time > elapsed)
          best_time = elapsed;
      }
      //fprintf(stderr, "\rRC5 cuda: slept=%d best_time=%ld elapsed=%ld process_amount=%d\n", slept, best_time, elapsed, process_amount);
    } else {
      if (cudaEventSynchronize(stop) != CUDA_SUCCESS) {
        retval = -1;
        fprintf(stderr, "RC5 cuda: ERROR: cudaEventSynchronize\r\n");
        goto error_exit;
      }
    }

    /* Copy the match_found variable to the host */
    if( cudaMemcpy((void *)&match_found, (void *)cuda_match_found, sizeof(u8), cudaMemcpyDeviceToHost) != (cudaError_t) CUDA_SUCCESS ) {
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
      if( cudaMemcpy((void *)results, (void *)cuda_results, process_amount * sizeof(u8), cudaMemcpyDeviceToHost) != (cudaError_t) CUDA_SUCCESS ) {
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

      }       /* for(... j < process_amount ...) */

    }     /* if(match_found) */

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

  if(core) {
    cudaStreamDestroy(core);
  }

  if(stop) {
    cudaEventDestroy(stop);
  }

  if(results) {
    free(results);
  }

  if(retval == -1) {
    last_error = cudaGetLastError();
    fprintf(stderr, "RC5 cuda: error_exit\r\n");
    fprintf(stderr, "RC5 cuda: ERROR: %s\r\n", cudaGetErrorString(last_error));
    fflush(stderr);
  }

#ifdef DISPLAY_TIMESTAMPS
  current_ts = linux_read_counter();
  fprintf(stderr, "RC5 cuda: elapsed_time_6=%lli\r\n", current_ts - prev_ts);
  prev_ts = current_ts;
#endif

  return retval;
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------                  GPU Core                  --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

__global__ void cuda_2pipe(const u32 plain_hi, const u32 plain_lo,
                           const u32 cypher_hi, const u32 cypher_lo,
                           const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                           const u32 process_amount, u8 * results, u8 * match_found)
{
  /* Grid of blocks dimension */
//	int gd = gridDim.x;

  /* Block index */
  int bx = blockIdx.x;

  /* Block of threads dimension */
  int bd = blockDim.x;

  /* Thread index */
  int tx = threadIdx.x;

  /* RC5 local state variables */
  u32 A1, A2, B1, B2;
  u32 S1[26], S2[26];
  u32 L1[3], L2[3];

  /* Drop out early if we don't have any data to process */
  if( ((bx * bd) + tx) > process_amount) {
    /* This processes two extra keys at the end of a non-full block */
    /* But this takes no extra time so might as well */
    /* Warning... Make sure you DON'T use  */
    /* __syncthreads() anywhere after this */
    /* point in the core!!!                */
    return;
  }

  /* Initialize the S[] with constants */
#define KEY_INIT(i) S1[i] = S2[i] = P + i*Q;
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
  L1[2] = L2[2] = L0_hi;
  L1[1] = L2[1] = L0_mid;
  L1[0] = L2[0] = L0_lo;
  increment_L0(&L1[2], &L1[1], &L1[0], 2*((bx * bd) + tx));
  increment_L0(&L2[2], &L2[1], &L2[0], 2*((bx * bd) + tx) + 0x01);

  /* ------------------------------------- */
  /* ------------------------------------- */
  /* ------------------------------------- */

#define ROTL_BLOCK(i,j) ROTL_BLOCK_j ## j (i)

#define ROTL_BLOCK_i0_j1 \
  S1[0] = ROTL3(S1[0]+(S1[25]+L1[0])); \
  S2[0] = ROTL3(S2[0]+(S2[25]+L2[0])); \
  L1[1] = ROTL(L1[1]+(S1[0]+L1[0]),(S1[0]+L1[0])); \
  L2[1] = ROTL(L2[1]+(S2[0]+L2[0]),(S2[0]+L2[0])); \

#define ROTL_BLOCK_i0_j2 \
  S1[0] = ROTL3(S1[0]+(S1[25]+L1[1])); \
  S2[0] = ROTL3(S2[0]+(S2[25]+L2[1])); \
  L1[2] = ROTL(L1[2]+(S1[0]+L1[1]),(S1[0]+L1[1])); \
  L2[2] = ROTL(L2[2]+(S2[0]+L2[1]),(S2[0]+L2[1])); \

#define ROTL_BLOCK_j0(i) \
  S1[i] = ROTL3(S1[i]+(S1[i-1]+L1[2])); \
  S2[i] = ROTL3(S2[i]+(S2[i-1]+L2[2])); \
  L1[0] = ROTL(L1[0]+(S1[i]+L1[2]),(S1[i]+L1[2])); \
  L2[0] = ROTL(L2[0]+(S2[i]+L2[2]),(S2[i]+L2[2])); \

#define ROTL_BLOCK_j1(i) \
  S1[i] = ROTL3(S1[i]+(S1[i-1]+L1[0])); \
  S2[i] = ROTL3(S2[i]+(S2[i-1]+L2[0])); \
  L1[1] = ROTL(L1[1]+(S1[i]+L1[0]),(S1[i]+L1[0])); \
  L2[1] = ROTL(L2[1]+(S2[i]+L2[0]),(S2[i]+L2[0])); \

#define ROTL_BLOCK_j2(i) \
  S1[i] = ROTL3(S1[i]+(S1[i-1]+L1[1])); \
  S2[i] = ROTL3(S2[i]+(S2[i-1]+L2[1])); \
  L1[2] = ROTL(L1[2]+(S1[i]+L1[1]),(S1[i]+L1[1])); \
  L2[2] = ROTL(L2[2]+(S2[i]+L2[1]),(S2[i]+L2[1])); \

  /* ---------- */

  S1[0] = ROTL3(S1[0]);
  S2[0] = ROTL3(S2[0]);
  L1[0] = ROTL(L1[0]+S1[0],S1[0]);
  L2[0] = ROTL(L2[0]+S2[0],S2[0]);

  /* ---------- */

  ROTL_BLOCK(1,1);
  ROTL_BLOCK(2,2);
  ROTL_BLOCK(3,0);
  ROTL_BLOCK(4,1);
  ROTL_BLOCK(5,2);
  ROTL_BLOCK(6,0);
  ROTL_BLOCK(7,1);
  ROTL_BLOCK(8,2);
  ROTL_BLOCK(9,0);
  ROTL_BLOCK(10,1);
  ROTL_BLOCK(11,2);
  ROTL_BLOCK(12,0);
  ROTL_BLOCK(13,1);
  ROTL_BLOCK(14,2);
  ROTL_BLOCK(15,0);
  ROTL_BLOCK(16,1);
  ROTL_BLOCK(17,2);
  ROTL_BLOCK(18,0);
  ROTL_BLOCK(19,1);
  ROTL_BLOCK(20,2);
  ROTL_BLOCK(21,0);
  ROTL_BLOCK(22,1);
  ROTL_BLOCK(23,2);
  ROTL_BLOCK(24,0);
  ROTL_BLOCK(25,1);

  /* ---------- */

  ROTL_BLOCK_i0_j2;

  /* ---------- */

  ROTL_BLOCK(1,0);
  ROTL_BLOCK(2,1);
  ROTL_BLOCK(3,2);
  ROTL_BLOCK(4,0);
  ROTL_BLOCK(5,1);
  ROTL_BLOCK(6,2);
  ROTL_BLOCK(7,0);
  ROTL_BLOCK(8,1);
  ROTL_BLOCK(9,2);
  ROTL_BLOCK(10,0);
  ROTL_BLOCK(11,1);
  ROTL_BLOCK(12,2);
  ROTL_BLOCK(13,0);
  ROTL_BLOCK(14,1);
  ROTL_BLOCK(15,2);
  ROTL_BLOCK(16,0);
  ROTL_BLOCK(17,1);
  ROTL_BLOCK(18,2);
  ROTL_BLOCK(19,0);
  ROTL_BLOCK(20,1);
  ROTL_BLOCK(21,2);
  ROTL_BLOCK(22,0);
  ROTL_BLOCK(23,1);
  ROTL_BLOCK(24,2);
  ROTL_BLOCK(25,0);

  /* ---------- */

  ROTL_BLOCK_i0_j1;

  /* ---------- */

  ROTL_BLOCK(1,2);
  ROTL_BLOCK(2,0);
  ROTL_BLOCK(3,1);
  ROTL_BLOCK(4,2);
  ROTL_BLOCK(5,0);
  ROTL_BLOCK(6,1);
  ROTL_BLOCK(7,2);
  ROTL_BLOCK(8,0);
  ROTL_BLOCK(9,1);
  ROTL_BLOCK(10,2);
  ROTL_BLOCK(11,0);
  ROTL_BLOCK(12,1);
  ROTL_BLOCK(13,2);
  ROTL_BLOCK(14,0);
  ROTL_BLOCK(15,1);
  ROTL_BLOCK(16,2);
  ROTL_BLOCK(17,0);
  ROTL_BLOCK(18,1);
  ROTL_BLOCK(19,2);
  ROTL_BLOCK(20,0);
  ROTL_BLOCK(21,1);
  ROTL_BLOCK(22,2);
  ROTL_BLOCK(23,0);
  ROTL_BLOCK(24,1);
  ROTL_BLOCK(25,2);

  /* ---------- */

  A1 = plain_lo + S1[0];
  A2 = plain_lo + S2[0];
  B1 = plain_hi + S1[1];
  B2 = plain_hi + S2[1];

  /* ---------- */

#define FINAL_BLOCK(i) \
  A1 = ROTL(A1^B1,B1)+S1[2*i]; \
  A2 = ROTL(A2^B2,B2)+S2[2*i]; \
  B1 = ROTL(B1^A1,A1)+S1[2*i+1]; \
  B2 = ROTL(B2^A2,A2)+S2[2*i+1];

  FINAL_BLOCK(1);
  FINAL_BLOCK(2);
  FINAL_BLOCK(3);
  FINAL_BLOCK(4);
  FINAL_BLOCK(5);
  FINAL_BLOCK(6);
  FINAL_BLOCK(7);
  FINAL_BLOCK(8);
  FINAL_BLOCK(9);
  FINAL_BLOCK(10);
  FINAL_BLOCK(11);
  FINAL_BLOCK(12);

  /* ------------------------------------- */
  /* ------------------------------------- */
  /* ------------------------------------- */

  /* Check the results for a match.        */
  if (A1 == cypher_lo) {

    /* Set the match_found flag */
    *match_found = 1;

    /* Record the "check_*" match   */
    /* in the results array.        */
    results[2*((bx * bd) + tx)] = 1;

    if (B1 == cypher_hi) {
      /* Record the RESULT_FOUND match  */
      /* in the results array.          */
      results[2*((bx * bd) + tx)] = 2;
    }
  }

  if (A2 == cypher_lo) {

    /* Set the match_found flag */
    *match_found = 1;

    /* Record the "check_*" match   */
    /* in the results array.        */
    results[2*((bx * bd) + tx)+1] = 1;

    if (B2 == cypher_hi) {
      /* Record the RESULT_FOUND match  */
      /* in the results array.          */
      results[2*((bx * bd) + tx)+1] = 2;
    }
  }
}

// vim: syntax=cpp
