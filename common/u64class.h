// Implementation of 64 bit unsigned integers using
// 32 bit unsigned integers
//
// $Log: u64class.h,v $
// Revision 1.1  1998/10/05 11:04:52  fordbr
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

#ifndef _U64CLASS_H
#define _U64CLASS_H

class u64
{
   public:
      u32 hi;
      u32 lo;

      u64 ();
      u64 (const u32);
      u64 (const u32, const u32);
      u64 (const u64 &v);
      ~u64 ();

      u64 & operator -= (const u64 &v);
      u64 & operator += (const u64 &v);
      u64 & operator *= (const u64 &v);
      u64 & operator /= (const u64 &v);
      u64 & operator %= (const u64 &v);

      u64 & operator &= (const u64 &v);
      u64 & operator |= (const u64 &v);
      u64 & operator ^= (const u64 &v);
      u64 & operator <<= (const u32 v);
      u64 & operator >>= (const u32 v);

      u64 operator ++ ();     // prefix increment
      u64 operator -- ();     // prefix decrement
      u64 operator ++ (int);  // postfix increment
      u64 operator -- (int);  // postfix decrement

      // At present these don't have to be friend functions
      // as hi and lo are public.  But in the future who knows.
      friend int operator >  (const u64 &a, const u64 &b);
      friend int operator >= (const u64 &a, const u64 &b);
      friend int operator <  (const u64 &a, const u64 &b);
      friend int operator <= (const u64 &a, const u64 &b);
      friend int operator == (const u64 &a, const u64 &b);
      friend int operator != (const u64 &a, const u64 &b);

      friend u64 operator &  (const u64 &a, const u64 &b);
      friend u64 operator |  (const u64 &a, const u64 &b);
      friend u64 operator ^  (const u64 &a, const u64 &b);
      friend u64 operator >> (const u64 &a, const u32 b);
      friend u64 operator << (const u64 &a, const u32 b);

      friend u64 operator + (const u64 &a, const u64 &b);
      friend u64 operator - (const u64 &a, const u64 &b);
      friend u64 operator * (const u64 &a, const u64 &b);
      friend u64 operator / (const u64 &a, const u64 &b);
      friend u64 operator % (const u64 &a, const u64 &b);

      // Helper function for / and %
      // remainder == 0 returns quotient
      // remainder != 0 returns remainder
      friend u64 division (const u64 &a, const u64 &b, const int remainder = 0);
};

inline
u64::u64 ()
{
   hi = lo = 0;
}

inline
u64::u64 (const u32 v)
{
   hi = 0;
   lo = v;
}

inline
u64::u64 (const u32 high, const u32 low)
{
   hi = high;
   lo = low;
}

inline
u64::u64 (const u64 &v)
{
   hi = v.hi;
   lo = v.lo;
}

inline
u64::~u64 ()
{
}

inline u64 &
u64::operator -= (const u64 &v)
{
   *this = *this - v;
   return *this;
}

inline u64 &
u64::operator += (const u64 &v)
{
   *this = *this + v;
   return *this;
}

inline u64 &
u64::operator *= (const u64 &v)
{
   *this = *this * v;
   return *this;
}

inline u64 &
u64::operator /= (const u64 &v)
{
   *this = *this / v;
   return *this;
}

inline u64 &
u64::operator %= (const u64 &v)
{
   *this = *this % v;
   return *this;
}

inline u64 &
u64::operator &= (const u64 &v)
{
   hi &= v.hi;
   lo &= v.lo;
   return *this;
}

inline u64 &
u64::operator |= (const u64 &v)
{
   hi |= v.hi;
   lo |= v.lo;
   return *this;
}

inline u64 &
u64::operator ^= (const u64 &v)
{
   hi ^= v.hi;
   lo ^= v.lo;
   return *this;
}

inline u64 &
u64::operator <<= (const u32 v)
{
   if (v > 63) {
      hi = lo = 0;
   } else if (v > 31) {
      hi = lo << (v-32);
      lo = 0;
   } else if (v != 0) {
      hi = (hi << v) | (lo >> (32-v));
      lo <<= v;
   }
   return *this;
}

inline u64 &
u64::operator >>= (const u32 v)
{
   if (v > 63) {
      hi = lo = 0;
   } else if (v > 31) {
      lo = hi >> (v-32);
      hi = 0;
   } else if (v != 0) {
      lo = (lo >> v) | (hi << (32-v));
      hi >>= v;
   }
   return *this;
}

inline u64
u64::operator ++ ()
{
   if (++lo == 0) {
      ++hi;
   }
   return *this;
}

inline u64
u64::operator -- ()
{
   if (lo-- == 0) {
      --hi;
   }
   return *this;
}

inline u64
u64::operator ++ (int dummy)
{
   u64 oldValue = *this;
   if (++lo == 0) {
      ++hi;
   }
   return oldValue;
}

inline u64
u64::operator -- (int dummy)
{
   u64 oldValue = *this;
   if (lo-- == 0) {
      --hi;
   }
   return oldValue;
}

inline u64
operator & (const u64 &a, const u64 &b)
{
   return u64(a.hi & b.hi, a.lo & b.lo);
}

inline u64
operator | (const u64 &a, const u64 &b)
{
   return u64(a.hi | b.hi, a.lo | b.lo);
}

inline u64
operator ^ (const u64 &a, const u64 &b)
{
   return u64(a.hi ^ b.hi, a.lo ^ b.lo);
}

inline int
operator == (const u64 &a, const u64 &b)
{
   return a.lo == b.lo && a.hi == b.hi;
}

inline int
operator != (const u64 &a, const u64 &b)
{
   return a.lo != b.lo || a.hi != b.hi;
}

inline int
operator > (const u64 &a, const u64 &b)
{
   return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

inline int
operator >= (const u64 &a, const u64 &b)
{
   return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

inline int
operator < (const u64 &a, const u64 &b)
{
   return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

inline int
operator <= (const u64 &a, const u64 &b)
{
   return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}

#endif
