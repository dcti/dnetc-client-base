// Implementation of 64 bit unsigned integers using
// 32 bit unsigned integers
//
// $Log: u64class.cpp,v $
// Revision 1.1  1998/10/05 11:04:49  fordbr
// A reasonably complete implementation of a 64 bit unsigned integer(u64) using
// 32 bit unsigned integers(u32).
//
// All arithmetic (+ - * / %), bitwise logical (& | ^ ~ << >>) and their assignment
// equivalents (+= -= *= /= %= &= |= ^= ~= <<= >>=) are overloaded as well as the
// comparison operators (< <= > >= == !=).
//
// Explicit constructors from one u32, two u32 (a high and a low part) and another u64
// are provided.
//
// Mainly meant to be portable rather than fast.
//

#include "u64class.h"

u64 operator + (const u64 &a, const u64 &b)
{
   u32 carry;

   carry = (a.lo & 0xffff) + (b.lo & 0xffff);
   carry = (a.lo >> 16) + (b.lo >> 16) + (carry >> 16);
   return u64(a.hi + b.hi + (carry >> 16), a.lo + b.lo);
}

u64 operator - (const u64 &a, const u64 &b)
{
   u32 borrow;

   borrow = (a.lo & 0xffff) - (b.lo & 0xffff);
   borrow = (a.lo >> 16) - (b.lo >> 16) - ((borrow >> 16) & 1);
   return u64(a.hi - b.hi - ((borrow >> 16) & 1), a.lo - b.lo);
}

u64 operator * (const u64 &a, const u64 &b)
{
   u32 result_hi, result_lo, carry, temp;

   result_lo = a.lo * b.lo;
   temp = (a.lo & 0xffff) * (b.lo & 0xffff);
   temp = (a.lo >> 16) * (b.lo & 0xffff) + (temp >> 16);
   carry = temp >> 16;
   temp = (b.lo >> 16) * (a.lo & 0xffff) + (temp & 0xffff);
   carry += (a.lo >> 16) * (b.lo >> 16) + (temp >> 16);
   result_hi = a.hi * b.lo + a.lo * b.hi + carry;

   return u64(result_hi, result_lo);
}

u64 division (const u64 &a, const u64 &b, int remainder)
{
   if ((a.hi == 0 && a.lo == 0) ||
       (a.hi < b.hi) ||
       (a.hi == b.hi && a.lo < b.lo)) {
      if (remainder) {
         return a;
      } else {
         return u64(0);
      }
   } else if (b.hi == 0 && b.lo == 0) {
      return u64(0xffffffff, 0xffffffff);
   } else {
      u32 dividend[7], divisor[7], quotient[4];
      u32 dividend_max, divisor_max, trial_q, test_value, test_dividend, temp;
      u32 shift, i, j;

      quotient[0] = 0;
      quotient[1] = 0;
      quotient[2] = 0;
      quotient[3] = 0;

      dividend[0] = 0;
      dividend[1] = 0;
      dividend[2] = a.lo & 0xffff;
      dividend[3] = a.lo >> 16;
      dividend[4] = a.hi & 0xffff;
      dividend[5] = a.hi >> 16;
      dividend[6] = 0;

      for (dividend_max = 5; dividend[dividend_max] == 0; dividend_max--)
         ;
      dividend_max++;

      divisor[0] = 0;
      divisor[1] = 0;
      divisor[2] = b.lo & 0xffff;
      divisor[3] = b.lo >> 16;
      divisor[4] = b.hi & 0xffff;
      divisor[5] = b.hi >> 16;
      divisor[6] = 0;

      for (divisor_max = 5; divisor[divisor_max] == 0; divisor_max--)
         ;

      for (shift = 0; divisor[divisor_max] < 0x8000; shift++) {
         for (i = dividend_max; i > 2 ; i--) {
            dividend[i] = ((dividend[i] << 1) & 0xffff) + ((dividend[i-1] & 0x8000) >> 15);
         }
         dividend[2] = (dividend[2] << 1) & 0xfffe;
         for (i = divisor_max; i > 2 ; i--) {
            divisor[i] = ((divisor[i] << 1) & 0xffff) + ((divisor[i-1] & 0x8000) >> 15);
         }
         divisor[2] = (divisor[2] << 1) & 0xfffe;
      }

      while (dividend_max > divisor_max) {
         test_dividend = (dividend[dividend_max] << 16) + dividend[dividend_max-1];
         if (dividend[dividend_max] == divisor[divisor_max]) {
            trial_q = 0xffff;
         } else {
            trial_q =  test_dividend / divisor[divisor_max];
         }
         test_value = divisor[divisor_max-1] * trial_q;
         test_dividend = ((test_dividend - trial_q * divisor[divisor_max]) << 16) + dividend[dividend_max-2];
         if (test_value > test_dividend) {
            trial_q--;
            test_value -= divisor[divisor_max-1];
            temp = (test_dividend >> 16) + divisor[divisor_max];
            test_dividend = (temp << 16) + (test_dividend & 0xffff);
            if ((temp & 0xffff0000) == 0 && test_value > test_dividend) {
               trial_q--;
            }
         }
         temp = divisor[2] * trial_q;
         for (i = 3, j = dividend_max - divisor_max + 1; i <= divisor_max; i++, j++) {
            dividend[j] -= temp & 0xffff;
            temp = divisor[i] * trial_q + (temp >> 16) + ((dividend[j] >> 16) & 1);
            dividend[j] &= 0xffff;
         }
         dividend[j] -= temp & 0xffff;
         temp = (temp >> 16) + ((dividend[j] >> 16) & 1);
         dividend[j] &= 0xffff;
         if (dividend[dividend_max] < temp) {
            trial_q--;
            temp = divisor[2];
            for (i = 3, j = dividend_max - divisor_max + 1; i <= divisor_max; i++, j++) {
               dividend[j] += temp;
               temp = divisor[i] + (dividend[j] >> 16);
               dividend[j] &= 0xffff;
            }
         }
         dividend_max--;
         quotient[dividend_max - divisor_max] = trial_q;
      }
      if (remainder) {
         // calculate remainder here
         for (;shift > 0; shift--) {
            for (i = 2; i < dividend_max; i++) {
               dividend[i] = ((dividend[i] >> 1) & 0xffff) + ((dividend[i+1] & 0x0001) << 15);
            }
            dividend[dividend_max] = (dividend[dividend_max] >> 1);
         }
         for (i = dividend_max + 1; i < 6; i++) {
            dividend[i] = 0;
         }
         return u64((dividend[5] << 16) + dividend[4], (dividend[3] << 16) + dividend[2]);
      } else {
         return u64((quotient[3] << 16) + quotient[2], (quotient[1] << 16) + quotient[0]);
      }
   }
}

u64 operator / (const u64 &a, const u64 &b)
{
   return division(a, b, 0);
}

u64 operator % (const u64 &a, const u64 &b)
{
   return division(a, b, 1);
}

u64 operator >> (const u64 &a, const u32 b)
{
   if (b > 63) {
      return u64(0);
   } else if (b > 31) {
      return u64(a.hi >> (b-32));
   } else if (b == 0) {
      return a;
   }
   return u64(a.hi >> b, (a.hi << (32-b)) | (a.lo >> b));
}

u64 operator << (const u64 &a, const u32 b)
{
   if (b > 63) {
      return u64(0);
   } else if (b > 31) {
      return u64(a.lo << (b-32), 0);
   } else if (b == 0) {
      return a;
   }
   return u64((a.hi << b) | (a.lo >> (32-b)), a.lo << b);
}
