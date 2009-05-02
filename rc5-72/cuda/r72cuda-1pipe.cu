/* -*-C-*-
 *
 * Copyright Paul Kurucz 2007 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * With modifications by Greg Childers, Robin Harmsen and Andreas Beckmann
 *
 * $Id: r72cuda-1pipe.cu,v 1.28 2009/05/02 12:15:04 andreasb Exp $
*/

#include <stdio.h>
#include <cuda.h>
#include "r72cuda.h"
#include "r72cuda-helper.cu"

#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_128( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_256( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_bw( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_s0( RC5_72UnitWork *, u32 *, void * );
extern "C" s32 CDECL rc5_72_unit_func_cuda_1_64_s1( RC5_72UnitWork *, u32 *, void * );
#endif

static __global__ void cuda_1pipe(const u32 plain_hi, const u32 plain_lo,
                                  const u32 cypher_hi, const u32 cypher_lo,
                                  const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                                  const u32 process_amount, u8 * results, u8 * match_found);

static s32 CDECL rc5_72_run_cuda_1(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, int device, u32 num_threads, int waitmode);


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------              Core Entry Point              --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

s32 CDECL rc5_72_unit_func_cuda_1_64(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 64, default_wait_mode);
}

s32 CDECL rc5_72_unit_func_cuda_1_128(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 128, default_wait_mode);
}

s32 CDECL rc5_72_unit_func_cuda_1_256(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 256, default_wait_mode);
}

s32 CDECL rc5_72_unit_func_cuda_1_64_bw(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 64, -1);
}

s32 CDECL rc5_72_unit_func_cuda_1_64_s0(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 64, 0);
}

s32 CDECL rc5_72_unit_func_cuda_1_64_s1(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  return rc5_72_run_cuda_1(rc5_72unitwork, iterations, rc5_72unitwork->threadnum, 64, 1);
}

static s32 CDECL rc5_72_run_cuda_1(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, int device, u32 num_threads, int waitmode)
{
  const u32 pipeline_count = 1;
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

  cudaStream_t core = -1;
  cudaEvent_t start = -1, stop = -1;
  cudaError_t cuda_error, last_error;

#ifdef DISPLAY_TIMESTAMPS
  int64_t current_ts;
  int64_t prev_ts;
#endif

  struct timeval tv_core_start;
  struct timeval tv_core_elapsed;

  const char *failed = NULL;

  //fprintf(stderr, "\r\nRC5 cuda: iterations=%i (%i*%i+%i)\r\n", *iterations, *iterations / optimal_process_amount, optimal_process_amount, *iterations % optimal_process_amount);

  if ((cuda_error = cudaGetDevice(&currentdevice)) != cudaSuccess) {
    failed = "cudaGetDevice";
    goto error_exit;
  }

  if (currentdevice != device) {
    if ((cuda_error = cudaSetDevice(device)) != cudaSuccess) {
      failed = "cudaSetDevice";
      goto error_exit;
    }
  }

  if ((cuda_error = cudaStreamCreate(&core)) != cudaSuccess || core == -1) {
    failed = "cudaStreamCreate";
    goto error_exit;
  }

  if ((cuda_error = cudaEventCreate(&start)) != cudaSuccess || start == -1) {
    failed = "cudaEventCreate";
    goto error_exit;
  }

  if ((cuda_error = cudaEventCreate(&stop)) != cudaSuccess || stop == -1) {
    failed = "cudaEventCreate";
    goto error_exit;
  }

  /* Determine the grid dimensionality based on the */
  /* number of iterations.                          */
  grid_dim = min_u32((*iterations / pipeline_count + num_threads - 1) / num_threads, max_grid_dim);

  /* --------------------------------------------- */

  /* Allocate the cuda_match_found variable */
  if ((cuda_error = cudaMalloc((void **)&cuda_match_found, sizeof(u8))) != cudaSuccess) {
    failed = "cudaMalloc: cuda_match_found";
    goto error_exit;
  }

  /* Allocate the results arrays */
  results = (u8 *)malloc(grid_dim * num_threads * sizeof(u8));
  if( results == NULL ) {
    failed = "malloc: results";
    goto error_exit;
  }

  if ((cuda_error = cudaMalloc((void **)&cuda_results, grid_dim * num_threads * sizeof(u8))) != cudaSuccess) {
    failed = "cudaMalloc: cuda_results";
    goto error_exit;
  }

  /* --------------------------------------------- */

#ifdef DISPLAY_TIMESTAMPS
  prev_ts = read_counter();
#endif

  for (i = 0; i < *iterations; i += optimal_process_amount) {
    dim3 block_dimension(num_threads);
    dim3 grid_dimension(grid_dim);
    u32 process_amount = min_u32(*iterations - i, optimal_process_amount);
    u32 j, k;
    u32 match_count = 0;

    /* Clear the match_found variable */
    if ((cuda_error = cudaMemset(cuda_match_found, 0, sizeof(u8))) != cudaSuccess) {
      failed = "cudaMemset: cuda_match_found";
      goto error_exit;
    }

    /* Clear the results array */
    if ((cuda_error = cudaMemset(cuda_results, 0, grid_dim * num_threads * sizeof(u8))) != cudaSuccess) {
      failed = "cudaMemset: cuda_results";
      goto error_exit;
    }

#ifdef DISPLAY_TIMESTAMPS
    current_ts = read_counter();
    fprintf(stderr, "RC5 cuda: elapsed_time_1=%lli\r\n", current_ts - prev_ts);
    prev_ts = current_ts;
#endif

    /* Execute the CUDA core */

    if (waitmode == 1) {
      CliTimer(&tv_core_start);
    } else if (waitmode == 2) {
      if ((cuda_error = cudaEventRecord(start, core)) != cudaSuccess) {
        failed = "cudaEventRecord";
        goto error_exit;
      }
    }

    cuda_1pipe<<<grid_dimension, block_dimension, 0, core>>>(
      rc5_72unitwork->plain.hi, rc5_72unitwork->plain.lo,
      rc5_72unitwork->cypher.hi, rc5_72unitwork->cypher.lo,
      rc5_72unitwork->L0.hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.lo,
      (process_amount + pipeline_count - 1) / pipeline_count,
      cuda_results, cuda_match_found);
    last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
      retval = -1;
      Log("RC5 CUDA CORE ERROR [%d]: %s\n", device, cudaGetErrorString(last_error));
      goto error_exit;
    }

    if ((cuda_error = cudaEventRecord(stop, core)) != cudaSuccess) {
      failed = "cudaEventRecord";
      goto error_exit;
    }

#ifdef DISPLAY_TIMESTAMPS
    current_ts = read_counter();
    fprintf(stderr, "RC5 cuda: elapsed_time_2=%lli\r\n", current_ts - prev_ts);
    prev_ts = current_ts;
#endif

    if (waitmode == 0) {
      int slept;
      for (slept = 0; (cuda_error = cudaEventQuery(stop)) == cudaErrorNotReady && slept <= max_sleep_iterations; ++slept)
        NonPolledUSleep(min_sleep_interval);
      //fprintf(stderr, "\rRC5 cuda: slept=%d process_amount=%d\n", slept, process_amount);
    } else if (waitmode == 1) {
      long best_time = rc5_72unitwork->best_time;
      long expected = 0, elapsed = 0;
      int slept = 0;
      if (best_time <= 0) {
        for (slept = 0; (cuda_error = cudaEventQuery(stop)) == cudaErrorNotReady && slept <= max_sleep_iterations; ++slept)
          NonPolledUSleep(min_sleep_interval);
        //fprintf(stderr, "\rRC5 cuda: slept=%d process_amount=%d\n", slept, process_amount);
      } else {
        expected = best_time * process_amount / optimal_process_amount;
        if (expected / 2 >= min_sleep_interval) {
          //fprintf(stderr, "\rRC5 cuda: sleep_now=%d\n", expected / 2);
          NonPolledUSleep(expected / 2);
          ++slept;
        }
        while ((cuda_error = cudaEventQuery(stop)) == cudaErrorNotReady && slept <= max_sleep_iterations) {
          CliTimerDiff(&tv_core_elapsed, &tv_core_start, NULL);
          elapsed = tv_core_elapsed.tv_sec * 1000000 + tv_core_elapsed.tv_usec;
          long sleep_now = (expected - elapsed) / 2;
          //fprintf(stderr, "\rRC5 cuda: sleep_now=%d\n", sleep_now);
          if (sleep_now < min_sleep_interval) {
            if ((cuda_error = cudaEventSynchronize(stop)) != cudaSuccess) {
              failed = "cudaEventSynchronize";
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
          rc5_72unitwork->best_time = best_time = elapsed;
      }
      //fprintf(stderr, "\rRC5 cuda: slept=%d best_time=%ld elapsed=%ld process_amount=%d\n", slept, best_time, elapsed, process_amount);
    } else if (waitmode == 2) {
      long best_time = rc5_72unitwork->best_time;
      long expected = 0, elapsed = 0;
      int slept = 0;
      if (best_time <= 0) {
        for (slept = 0; (cuda_error = cudaEventQuery(stop)) == cudaErrorNotReady && slept <= max_sleep_iterations; ++slept)
          NonPolledUSleep(min_sleep_interval);
        //fprintf(stderr, "\rRC5 cuda: slept=%d process_amount=%d\n", slept, process_amount);
      } else {
        expected = (long) ((double)best_time * ((double)process_amount / (double)optimal_process_amount));
        //fprintf(stderr, "\rRC5 cuda: sleep_now=%d best_time=%d\n", expected, best_time);
        if (expected >= min_sleep_interval) {
          // Sleep for entire expected time (since it is an accurate time from the GPU clock) rounded
          // down to the nearest multiple of min_sleep_interval
          NonPolledUSleep((expected/min_sleep_interval) * min_sleep_interval);
          ++slept;
        }
      }
      if ((cuda_error = cudaEventSynchronize(stop)) != cudaSuccess) {
        failed = "cudaEventSynchronize";
        goto error_exit;
      }
      if (process_amount == optimal_process_amount) {
        // update best speed
        float gpu_time;
        if ((cuda_error = cudaEventElapsedTime(&gpu_time, start, stop)) != cudaSuccess) {
          failed = "cudaEventElapsedTime";
          goto error_exit;
        }
        elapsed = (long)(1000.0 * gpu_time);
        if (best_time <= 0 || best_time > elapsed ) {
          rc5_72unitwork->best_time = best_time = elapsed;
        }
      }
      //fprintf(stderr, "\rRC5 cuda: slept=%d best_time=%ld elapsed=%ld process_amount=%d\n", slept, best_time, elapsed, process_amount);
    } else {
      if ((cuda_error = cudaEventSynchronize(stop)) != cudaSuccess) {
        failed = "cudaEventSynchronize";
        goto error_exit;
      }
    }

    /* Copy the match_found variable to the host */
    if ((cuda_error = cudaMemcpy((void *)&match_found, (void *)cuda_match_found, sizeof(u8), cudaMemcpyDeviceToHost)) != cudaSuccess) {
      failed = "cudaMemcpy";
      goto error_exit;
    }

#ifdef DISPLAY_TIMESTAMPS
    current_ts = read_counter();
    fprintf(stderr, "RC5 cuda: elapsed_time_3=%lli\r\n", current_ts - prev_ts);
    prev_ts = current_ts;
#endif

    /* Optimization: Only scan the results[] if  */
    /* the match_found flag (indicating an exact */
    /* or partial match) is set.                 */
    if(match_found) {

      /* Copy the results[] array to the host */
      if ((cuda_error = cudaMemcpy((void *)results, (void *)cuda_results, (process_amount + pipeline_count - 1) / pipeline_count * sizeof(u8), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        failed = "cudaMemcpy: cuda_results";
        goto error_exit;
      }

      /* Check the results array for any matches. */
      for (j = 0; j < (process_amount + pipeline_count - 1) / pipeline_count; j++) {
        for (k = 0; k < pipeline_count; k++) {
          /* Check if we have found a partial match */
          if(((results[j] >> 2*k) & 3) > 0) {
            rc5_72unitwork->check.count++;
            match_count++;

            /* Copy over the current key */
            rc5_72unitwork->check.hi = rc5_72unitwork->L0.hi;
            rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
            rc5_72unitwork->check.lo = rc5_72unitwork->L0.lo;

            /* Offset the key index to match out current position */
            increment_L0(&rc5_72unitwork->check.hi, &rc5_72unitwork->check.mid, &rc5_72unitwork->check.lo, pipeline_count * j + k);

            /* Check if we have found an exact match */
            if(((results[j] >> 2*k) & 3) > 1) {
              /* Correct the L0 offset value */
              increment_L0(&rc5_72unitwork->L0.hi, &rc5_72unitwork->L0.mid, &rc5_72unitwork->L0.lo, pipeline_count * j + k);

              /* Pass back the iterations count to the callee */
              *iterations = i + pipeline_count * j + k;

              /* Update the return value and jump to the exit point */
              retval = RESULT_FOUND;
              goto sucess_exit;
            }
          }
        }
      }       /* for(... j < process_amount ...) */

    }     /* if(match_found) */

    /* Advance L0 by the amount that we processed */
    /* this pass through the for() loop.          */
    increment_L0(&rc5_72unitwork->L0.hi, &rc5_72unitwork->L0.mid, &rc5_72unitwork->L0.lo, process_amount);

#ifdef DISPLAY_TIMESTAMPS
    current_ts = read_counter();
    fprintf(stderr, "RC5 cuda: elapsed_time_4=%lli\r\n", current_ts - prev_ts);
    prev_ts = current_ts;
#endif
  }

sucess_exit:
error_exit:
  if (failed) {
    retval = -1;
    last_error = cudaGetLastError();
    Log("RC5 CUDA ERROR [%d]: #%d '%s' (%s)\n", device, cuda_error, cudaGetErrorString(cuda_error), failed);
    if (cuda_error != last_error)
      Log("RC5 CUDA ERROR [%d]: #%d '%s' (%s)\n", device, last_error, cudaGetErrorString(last_error), failed);
  }

#ifdef DISPLAY_TIMESTAMPS
  current_ts = read_counter();
  fprintf(stderr, "RC5 cuda: elapsed_time_5=%lli\r\n", current_ts - prev_ts);
  prev_ts = current_ts;
#endif

  if(cuda_match_found) {
    cudaFree(cuda_match_found);
  }

  if(cuda_results) {
    cudaFree(cuda_results);
  }

  if (stop != -1) {
    cudaEventDestroy(stop);
  }

  if (start != -1) {
    cudaEventDestroy(start);
  }

  if (core != -1) {
    cudaStreamDestroy(core);
  }

  if(results) {
    free(results);
  }

#ifdef DISPLAY_TIMESTAMPS
  current_ts = read_counter();
  fprintf(stderr, "RC5 cuda: elapsed_time_6=%lli\r\n", current_ts - prev_ts);
  prev_ts = current_ts;
#endif

  /* tell the client about the optimal timeslice increment for this core
     (with current parameters) */
  rc5_72unitwork->optimal_timeslice_increment = optimal_process_amount;

  return retval;
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------                  GPU Core                  --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

__global__ void cuda_1pipe(const u32 plain_hi, const u32 plain_lo,
                           const u32 cypher_hi, const u32 cypher_lo,
                           const u32 L0_hi, const u32 L0_mid, const u32 L0_lo,
                           const u32 process_amount, u8 * results, u8 * match_found)
{
  const int pipeline_count = 1;
  int k;

  /* Grid of blocks dimension */
  //int gd = gridDim.x;

  /* Block index */
  int bx = blockIdx.x;

  /* Block of threads dimension */
  int bd = blockDim.x;

  /* Thread index */
  int tx = threadIdx.x;

  /* RC5 local state variables */
  u32 A, B;
  u32 S[26];
  u32 L[3];

  /* Drop out early if we don't have any data to process */
  if( ((bx * bd) + tx) > process_amount) {
    /* This processes one extra key at the end of a non-full block */
    /* But this takes no extra time so might as well */
    /* Warning... Make sure you DON'T use  */
    /* __syncthreads() anywhere after this */
    /* point in the core!!!                */
    return;
  }

  for (k = 0; k < pipeline_count; ++k) {
    /* Initialize the S[] with constants */
#define KEY_INIT(i) S[i] = P + i*Q;
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
    increment_L0(&L[2], &L[1], &L[0], pipeline_count * ((bx * bd) + tx) + k);

    /* ------------------------------------- */
    /* ------------------------------------- */
    /* ------------------------------------- */

#define ROTL_BLOCK(i,j) ROTL_BLOCK_j ## j (i)

#define ROTL_BLOCK_i0_j1 \
  S[0] = ROTL3(S[0]+(S[25]+L[0])); \
  L[1] = ROTL(L[1]+(S[0]+L[0]),(S[0]+L[0]));

#define ROTL_BLOCK_i0_j2 \
  S[0] = ROTL3(S[0]+(S[25]+L[1])); \
  L[2] = ROTL(L[2]+(S[0]+L[1]),(S[0]+L[1]));

#define ROTL_BLOCK_j0(i) \
  S[i] = ROTL3(S[i]+(S[i-1]+L[2])); \
  L[0] = ROTL(L[0]+(S[i]+L[2]),(S[i]+L[2]));

#define ROTL_BLOCK_j1(i) \
  S[i] = ROTL3(S[i]+(S[i-1]+L[0])); \
  L[1] = ROTL(L[1]+(S[i]+L[0]),(S[i]+L[0]));

#define ROTL_BLOCK_j2(i) \
  S[i] = ROTL3(S[i]+(S[i-1]+L[1])); \
  L[2] = ROTL(L[2]+(S[i]+L[1]),(S[i]+L[1]));

    /* ---------- */

    S[0] = ROTL3(S[0]);
    L[0] = ROTL(L[0]+S[0],S[0]);

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

    A = plain_lo + S[0];
    B = plain_hi + S[1];

    /* ---------- */

#define FINAL_BLOCK(i) \
  A = ROTL(A^B,B)+S[2*i]; \
  B = ROTL(B^A,A)+S[2*i+1];

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
    if (A == cypher_lo) {

      /* Set the match_found flag */
      *match_found = 1;

      /* Record the "check_*" match   */
      /* in the results array.        */
      results[(bx * bd) + tx] += (1 << (2 * k));

      if (B == cypher_hi) {
        /* Record the RESULT_FOUND match  */
        /* in the results array.          */
        results[(bx * bd) + tx] += (1 << (2 * k)); /* add another 1 */
      }
    }
  }
}

// vim: syntax=cpp:et:sw=2:ts=2
