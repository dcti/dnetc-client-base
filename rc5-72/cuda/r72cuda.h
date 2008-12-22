/* -*-C-*-
 *
 * Copyright Paul Kurucz 2007 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * With modifications by Greg Childers
 *
 * $Id: r72cuda.h,v 1.13 2008/12/22 11:46:33 andreasb Exp $
*/

#include "ccoreio.h"
#include "clitime.h"
#include "sleepdef.h"

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
const int min_sleep_interval = 15000;   // microseconds
#else
const int min_sleep_interval = 100;     // microseconds
#endif

// choices are -1 = busy wait, 0 = simple sleep loop, 1 = adaptive sleep loop, 2 = gpu timed sleep
const int default_wait_mode = 2;

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* Uncomment the define below to display the    */
/* processing timestamps.                       */
//#define DISPLAY_TIMESTAMPS

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

#define P 0xB7E15163
#define Q 0x9E3779B9

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* ---------------     Local Helper Function Prototypes       --------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

static __host__ __device__ u32 swap_u32(u32 num);
static __host__ __device__ u8 add_u32(u32 num1, u32 num2, u32 * result);
static __host__ __device__ void increment_L0(u32 * hi, u32 * mid, u32 * lo, u32 amount);

#ifdef DISPLAY_TIMESTAMPS
static __inline int64_t read_counter(void);
#endif

static __inline u32 min_u32(u32 a, u32 b)
{
  return (a < b) ? a : b;
}

/* Type decaration for the L0 field of the      */
/* RC5_72UnitWork structure.                    */
typedef struct {
  u32 hi;
  u32 mid;
  u32 lo;
} L0_t;

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))

#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)

