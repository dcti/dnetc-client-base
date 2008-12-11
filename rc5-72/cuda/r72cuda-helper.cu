/* -*-C-*-
 *
 * Copyright Paul Kurucz 2007 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * With modifications by Greg Childers
*/

/* nvcc requires all __device__ functions to be in the same source file as
 * the kernel, so this source file can't be compiled separately but has to
 * be included from the cores.
*/

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
static __host__ __device__ void increment_L0(u32 * hi, u32 * mid, u32 * lo, u32 amount)
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

#ifdef DISPLAY_TIMESTAMPS
/* Return the current uSec count */
static __inline int64_t linux_read_counter(void)
{
  struct timeval tv;
  int64_t retval = 0;

  CliTimer(&tv);

  retval = (((int64_t)tv.tv_sec) * 1000000) + tv.tv_usec;

  return retval;
}
#endif /* ifdef DISPLAY_TIMESTAMPS */

// vim: syntax=cpp
