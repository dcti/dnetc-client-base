/*
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogr.cpp,v 1.1.2.7 2000/09/17 10:33:54 cyp Exp $
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HAVE_STATIC_CHOOSEDAT /* choosedat table is static, pre-generated */
/* #define CRC_CHOOSEDAT_ANYWAY */ /* you'll need to link crc32 if this is defd */

/* --- various optimization option overrides ----------------------------- */

/* baseline/reference == ogr.cpp without optimization == old ogr.cpp */
#if defined(NO_OGR_OPTIMIZATION) || defined(GIMME_BASELINE_OGR_CPP)
  #define OGROPT_BITOFLIST_DIRECT_BIT 0           /* 0/1 - default is 1 ('yes')) */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0   /* 0/1 - default is hw dependant */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS  0       /* 0-2 - default is 1 */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 0-2 - default is 2 */
#else
  #if (defined(ASM_X86) || defined(__386__)) && defined(OGR_NOFFZ)
    /* the bsr instruction is very slow on some cpus */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
    /* #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_noffz */
  #endif
  #if defined(ASM_68K) 
    #define OGROPT_BITOFLIST_DIRECT_BIT 0          /* we want 'no' */
  #endif
  #if defined(ASM_PPC) || defined(__PPC__)
    /* I'm not sure whether this is compiler or arch specific, */ 
    /* it doesn't seem to make any difference for x86/gcc 2.7x  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0    /* no optimization */
  #endif
#endif  

/* -- various optimization option defaults ------------------------------- */

/* optimization for machines where mem access is faster than a shift+sub+and.
   Particularly effective with small data cache.
   If not set to 1, BITOFLIST will use a pre-computed memtable lookup, 
   otherwise it will compute the value at runtime (0x80000000>>((x-1)&0x1f))
*/
#ifndef OGROPT_BITOFLIST_DIRECT_BIT
#define OGROPT_BITOFLIST_DIRECT_BIT 1 /* the default is "yes" */
#endif


/* optimization for available hardware insn(s) for 'find first zero bit',
   counting from highest bit, ie 0xEFFFFFFF returns 1, and 0xFFFFFFFE => 32 
   This is the second (or first) most effective speed optimization.
*/
#if defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) && \
           (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 0)
  #undef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
#else
  #undef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
  #if (defined(__PPC__) || defined(ASM_PPC)) || \
      (defined(__WATCOMC__) && defined(__386__)) || \
      (defined(__GNUC__) && (defined(ASM_SPARC) || defined(ASM_ALPHA) \
                           || defined(ASM_X86) \
                           || (defined(ASM_68K) && (defined(mc68020) \
                           || defined(mc68030) || defined(mc68040) \
                           || defined(mc68060)))))
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 1
    /* #define FIRSTBLANK_ASM_TEST */ /* define this to test */
  #endif
#endif


/* optimize COPY_LIST_SET_BIT macro for when jumps are expensive. 
   0=no reduction (6 'if'), 1=one 'if'+manual copy; 2=one 'if' plus memcpy; 
   This is the most (or second most) effective speed optimization.
   If your compiler has an intrinsic memcpy() AND optimizes that for size
   and alignment (ie, doesn't just inline memcpy) AND/OR the target arch
   is register-rich, then 2 is faster than 1.
*/
#ifndef OGROPT_COPY_LIST_SET_BIT_JUMPS
#define OGROPT_COPY_LIST_SET_BIT_JUMPS 1        /* 0 (no opt) or 1 or 2 */
#endif


/* reduction of found_one maps by using single bits intead of whole chars
   0=no reduction (upto 1024 octets); 1=1024 bits in 128 chars; 2=120 chars
   opt 1 or 2 adds two shifts, two bitwise 'and's and one bitwise 'or'.
   NOTE: that found_one() is not a speed critical function, and *should*
   have no effect on -benchmark at all since it is never called for -bench,
   Some compilers/archs show a negative impact on -benchmark because 
   they optimize register usage in ogr_cycle while tracking those used in 
   found_one and/or are size sensitive, and the increased size of found_one() 
   skews -benchmark. [the latter can be compensated for by telling the 
   compiler to align functions, the former by making found_one() non static]
*/
#ifndef OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE
#define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 2 /* 0 (no opt) or 1 or 2 */
#endif

/* ----------------------------------------------------------------------- */


#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
#include "crc32.h" /* only need to crc choose_dat if its not static */
#endif
#include "ogr.h"

#define CHOOSEBITS 12
#define MAXBITS    12
#define ttmMAXBITS (32-MAXBITS)

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef HAVE_STATIC_CHOOSEDAT  /* choosedat table is static, pre-generated */
extern const unsigned char ogr_choose_dat[]; /* this is in choosedat.h|c */
#define choose(x,y) (ogr_choose_dat[CHOOSEBITS*(x)+(y+3)]) /*+3 skips header */
#else
static const unsigned char *choosedat;/* set in init_load_choose() */
#define choose(x,y) (choosedat[CHOOSEBITS*(x)+(y)])
#endif

static const int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};
#if !defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) || defined(FIRSTBLANK_ASM_TEST)
static char ogr_first_blank[65537]; /* first blank in 16 bit COMP bitmap, range: 1..16 */
#endif
#if (OGROPT_BITOFLIST_DIRECT_BIT == 0)
static U ogr_bit_of_LIST[200]; /* which bit of LIST to update */
#endif

static int init_load_choose(void);
static int found_one(struct State *oState);
static int ogr_init(void);
static int ogr_create(void *input, int inputlen, void *state, int statelen);
static int ogr_cycle(void *state, int *pnodes);
static int ogr_getresult(void *state, void *result, int resultlen);
static int ogr_destroy(void *state);
#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
static int ogr_count(void *state);
static int ogr_save(void *state, void *buffer, int buflen);
static int ogr_load(void *buffer, int buflen, void **state);
#endif
static int ogr_cleanup(void);

#ifndef OGR_GET_DISPATCH_TABLE_FXN
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table
#endif  
extern CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void);

#if defined(__cplusplus)
}
#endif

/* ------------------------------------------------------------------ */

#define COMP_LEFT_LIST_RIGHT(lev,s)                             \
  {                                                             \
    register int ss = 32 - s;                                   \
    lev->comp[0] = (lev->comp[0] << s) | (lev->comp[1] >> ss);  \
    lev->comp[1] = (lev->comp[1] << s) | (lev->comp[2] >> ss);  \
    lev->comp[2] = (lev->comp[2] << s) | (lev->comp[3] >> ss);  \
    lev->comp[3] = (lev->comp[3] << s) | (lev->comp[4] >> ss);  \
    lev->comp[4] <<= s;                                         \
    lev->list[4] = (lev->list[4] >> s) | (lev->list[3] << ss);  \
    lev->list[3] = (lev->list[3] >> s) | (lev->list[2] << ss);  \
    lev->list[2] = (lev->list[2] >> s) | (lev->list[1] << ss);  \
    lev->list[1] = (lev->list[1] >> s) | (lev->list[0] << ss);  \
    lev->list[0] >>= s;                                         \
  }

#define COMP_LEFT_LIST_RIGHT_32(lev)              \
  lev->comp[0] = lev->comp[1];                    \
  lev->comp[1] = lev->comp[2];                    \
  lev->comp[2] = lev->comp[3];                    \
  lev->comp[3] = lev->comp[4];                    \
  lev->comp[4] = 0;                               \
  lev->list[4] = lev->list[3];                    \
  lev->list[3] = lev->list[2];                    \
  lev->list[2] = lev->list[1];                    \
  lev->list[1] = lev->list[0];                    \
  lev->list[0] = 0;

#if (OGROPT_BITOFLIST_DIRECT_BIT == 0)
  #define BITOFLIST(x) ogr_bit_of_LIST[x]
#else
  #define BITOFLIST(x) 0x80000000>>((x-1)&0x1f) /*0x80000000 >> ((x-1) % 32)*/
#endif


#if (OGROPT_COPY_LIST_SET_BIT_JUMPS == 1)
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    lev2->list[0] = lev->list[0];                 \
    lev2->list[1] = lev->list[1];                 \
    lev2->list[2] = lev->list[2];                 \
    lev2->list[3] = lev->list[3];                 \
    lev2->list[4] = lev->list[4];                 \
    if (d <= (32*5))                              \
      lev2->list[(d-1)>>5] |= BITOFLIST( d );     \
  }    
#elif (OGROPT_COPY_LIST_SET_BIT_JUMPS == 2)
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    memcpy( &(lev2->list[0]), &(lev->list[0]), sizeof(lev2->list[0])*5 ); \
    if (d <= (32*5))                              \
      lev2->list[(d-1)>>5] |= BITOFLIST( d );     \
  }    
#else
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    register int bit = BITOFLIST( d );            \
    if (d <= 32) {                                \
       lev2->list[0] = lev->list[0] | bit;        \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 64) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1] | bit;        \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 96) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2] | bit;        \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 128) {                        \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3] | bit;        \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 160) {                        \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4] | bit;        \
    } else {                                      \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    }                                             \
  }
#endif  

#define COPY_DIST_COMP(lev2,lev)                  \
  lev2->dist[0] = lev->dist[0] | lev2->list[0];   \
  lev2->dist[1] = lev->dist[1] | lev2->list[1];   \
  lev2->dist[2] = lev->dist[2] | lev2->list[2];   \
  lev2->dist[3] = lev->dist[3] | lev2->list[3];   \
  lev2->dist[4] = lev->dist[4] | lev2->list[4];   \
  lev2->comp[0] = lev->comp[0] | lev2->dist[0];   \
  lev2->comp[1] = lev->comp[1] | lev2->dist[1];   \
  lev2->comp[2] = lev->comp[2] | lev2->dist[2];   \
  lev2->comp[3] = lev->comp[3] | lev2->dist[3];   \
  lev2->comp[4] = lev->comp[4] | lev2->dist[4];

#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
static const unsigned chooseCRC32[24] = {
  0x00000000,   /* 0 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 5 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 10 */
  0x00000000,
  0x01138a7d,
  0x00000000,
  0x00000000,
  0x00000000,   /* 15 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 20 */
  0x00000000,
  0x00000000,
  0x00000000
};
#endif

static int init_load_choose(void)
{
#ifndef HAVE_STATIC_CHOOSEDAT
  #error choose_dat needs to be created/loaded here
#endif  
  if (MAXBITS != ogr_choose_dat[2]) {
    return CORE_E_FORMAT;
  }
#ifndef HAVE_STATIC_CHOOSEDAT
  /* skip over the choose.dat header */
  choosedat = &ogr_choose_dat[3];
#endif  

#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
  /* CRC32 check */
  {
    int i, j;
    unsigned crc32 = 0xffffffff;
    crc32 = CRC32(crc32, ogr_choose_dat[0]);
    crc32 = CRC32(crc32, ogr_choose_dat[1]);
    crc32 = CRC32(crc32, ogr_choose_dat[2]); /* This varies a lot */
    for (j = 0; j < (1 << MAXBITS); j++) {
      for (i = 0; i < CHOOSEBITS; ++i) crc32 = CRC32(crc32, choose(j, i));
    }
    crc32 = ~crc32;
    if (chooseCRC32[MAXBITS] != crc32) {
      /* printf("Your choose.dat (CRC=%08x) is corrupted! Oh well, continuing anyway.\n", crc32); */
      return CORE_E_FORMAT;
    }
  }

#endif
  return CORE_S_OK;
}

/*-----------------------------------------*/
/*  found_one() - print out golomb rulers  */
/*-----------------------------------------*/

static int found_one(struct State *oState)
{
  /* confirm ruler is golomb */
  {
    register int i, j;
    #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2)
    char diffs[((1024-64)+7)/8];
    #elif (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1)
    char diffs[((1024)+7)/8];
    #else
    char diffs[1024];
    #endif
    register int max = oState->max;
    register int maxdepth = oState->maxdepth;
    #if 1 /* (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1) || \
             (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2) */
    memset( diffs, 0, sizeof(diffs) );
    #else
    for (i = max>>1; i>=1; i--) diffs[i] = 0;
    #endif
    for (i = 1; i < maxdepth; i++) {
      register int marks_i = oState->marks[i];
      for (j = 0; j < i; j++) {
        register int diff = marks_i - oState->marks[j];
        if (diff+diff <= max) {        /* Principle 1 */
          if (diff <= 64) break;      /* 2 bitmaps always tracked */
	  #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2) || \
              (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1)
	  {
	    register int mask;
            #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2)
            diff -= 64;
            #endif
            mask = 1<<(diff&7);
            diff >>= 3;   
	    if ((diffs[diff] & mask)!=0) return 0;
	    diffs[diff] |= (char)mask;
	  }
	  #else    
          if (diffs[diff]) return 0;
          diffs[diff] = 1;
	  #endif
        }
      }
    }
  }
  return 1;
}

#if !defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) /* 0 <= x < 0xfffffffe */
  #define LOOKUP_FIRSTBLANK(x) ((x < 0xffff0000) ? \
      (ogr_first_blank[x>>16]) : (16 + ogr_first_blank[x - 0xffff0000]))
#elif defined(__PPC__) || defined(ASM_PPC) /* CouNT Leading Zeros Word */
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
    { i = ~i; __asm__ ("cntlzw %0,%1" : "=r" (i) : "r" (i)); return i+1; }
  #else /* macos */
    #error "Please check this (define FIRSTBLANK_ASM_TEST to test)"
    #define LOOKUP_FIRSTBLANK(x) (__cntlzw(~((unsigned int)(x)))+1)
  #endif    
#elif defined(ASM_ALPHA) && defined(__GNUC__)
  #error "Please check this (define FIRSTBLANK_ASM_TEST to test)"
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
  { i = ~i; __asm__ ("cntlzw %0,%0" : "=r"(i) : "0" (i)); return i+1; }
#elif defined(ASM_SPARC) && defined(__GNUC__)    
  #error "Please check this (define FIRSTBLANK_ASM_TEST to test)"
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
  { register int count; __asm__ ("scan %1,0,%0" : "r=" (count)
    : "r" ((unsigned int)(~i)) );  return count+1; }
#elif defined(ASM_X86) && defined(__GNUC__) || \
      defined(__386__) && defined(__WATCOMC__)
  #if defined(__GNUC__)      
    #define asm_lookup_first_0(result,input) \
              __asm__("notl %1\n\t"     \
		      "movl $33,%0\n\t" \
		      "bsrl %1,%1\n\t"  \
		      "jz   0f\n\t"     \
		      "subl %1,%0\n\t"  \
		      "decl %0\n\t"     \
		      "0:"              \
		      :"=r"(result), "=r"(input) : "1"(input) : "cc" );
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i) 
    { register unsigned int s; asm_lookup_first_0(s,i); return s; }
  #else /* WATCOMC */
    int LOOKUP_FIRSTBLANK(unsigned int);
    #pragma aux LOOKUP_FIRSTBLANK =  \
                      "not  eax"     \
		      "mov  edx,21h" \
                      "bsr  eax,eax" \
                      "jz   f0"      \
                      "sub  edx,eax" \
                      "dec  edx"     \
                      "f0:"          \
		      value [edx] parm [eax] modify exact [eax edx] nomemory;
  #endif
#elif defined(ASM_68K) && defined(__GNUC__) /* Bit field find first one set (020+) */
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
  { i = ~i; __asm__ ("bfffo %1,0,0,%0" : "=d" (i) : "d" (i)); return ++i; }  
#else
  #error OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM is defined, and no code to match
#endif
	 
/*
     0- (0x0000+0x8000-1) = 1    0x0000-0x7fff = 1 (0x7fff) 1000 0000 0000 0000
0x8000- (0x8000+0x4000-1) = 2    0x8000-0xBfff = 2 (0x3fff) 1100 0000 0000 0000
0xC000- (0xC000+0x2000-1) = 3    0xC000-0xDfff = 3 (0x1fff) 1110 0000 0000 0000
0xE000- (0xE000+0x1000-1) = 4    0xE000-0xEfff = 4 (0x0fff) 1111 0000 0000 0000
0xF000- (0xF000+0x0800-1) = 5    0xF000-0xF7ff = 5 (0x07ff) 1111 1000 0000 0000
0xF800- (0xF800+0x0400-1) = 6    0xF800-0xFBff = 6 (0x03ff) 1111 1100 0000 0000
*/

static int ogr_init(void)
{
  int r = init_load_choose();
  if (r != CORE_S_OK) {
    return r;
  }

  #if (OGROPT_BITOFLIST_DIRECT_BIT == 0)
  {
    int n;
    ogr_bit_of_LIST[0] = 0;
    for( n=1; n < 200; n++) {
       ogr_bit_of_LIST[n] = 0x80000000 >> ((n-1) % 32);
    }
  }    
  #endif

  #if !defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) || defined(FIRSTBLANK_ASM_TEST)
  {
    /* first zero bit in 16 bits */
    int i, j, k = 0, m = 0x8000;
    for (i = 1; i <= 16; i++) {
      for (j = k; j < k+m; j++) ogr_first_blank[j] = (char)i;
      k += m;
      m >>= 1;
    }
    ogr_first_blank[0xffff] = 17;     /* just in case we use it */
  }    
  #endif

  #if defined(FIRSTBLANK_ASM_TEST)
  {
    static int done_test = -1;
    if ((++done_test) == 0)
    {
      unsigned int q, err_count = 0;
      printf("begin firstblank test\n"
             "(this may take a looooong time and requires a -KILL to stop)\n");   
      for (q = 0; q <= 0xfffffffe; q++)
      {
        int s1 = ((q < 0xffff0000) ? \
          (ogr_first_blank[q>>16]) : (16 + ogr_first_blank[q - 0xffff0000]));
        int s2 = LOOKUP_FIRSTBLANK(q);
        if (s1 != s2)
        {
          printf("\nfirstblank error %d != %d (q=%u/0x%08x)\n", s1, s2, q, q);
          err_count++;
        }  
        else if (q == 0xfffffffe || (q & 0xfffff) == 0xfffff)      
        {
          printf("\rfirstblank done 0x%08x-0x%08x ", q & 0xfff00000, q);
          fflush(stdout);
        }
      }
      printf("\nend firstblank test (%u errors)\n", err_count);    
    }
    done_test = 0;
  }
  #endif

  return CORE_S_OK;
}

#ifdef OGR_DEBUG
static void dump(int depth, struct Level *lev, int limit)
{
  printf("--- depth %d\n", depth);
  printf("list=%08x%08x%08x%08x%08x\n", lev->list[0], lev->list[1], lev->list[2], lev->list[3], lev->list[4]);
  printf("dist=%08x%08x%08x%08x%08x\n", lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3], lev->dist[4]);
  printf("comp=%08x%08x%08x%08x%08x\n", lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3], lev->comp[4]);
  printf("cnt1=%d cnt2=%d limit=%d\n", lev->cnt1, lev->cnt2, limit);
  //sleep(1);
}
#endif

static int ogr_create(void *input, int inputlen, void *state, int statelen)
{
  struct State *oState;
  struct WorkStub *workstub = (struct WorkStub *)input;

  if (input == NULL || inputlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }

  if (((unsigned int)statelen) < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  oState = (struct State *)state;
  if (oState == NULL) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct State));

  oState->maxdepth = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;

  if (((unsigned int)oState->maxdepth) > (sizeof(OGR)/sizeof(OGR[0]))) {
    return CORE_E_FORMAT;
  }

  oState->max = OGR[oState->maxdepth-1];

  /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */
  oState->half_depth2 = oState->half_depth = ((oState->maxdepth+1) >> 1) - 1;
  if (!(oState->maxdepth % 2)) oState->half_depth2++;  /* if even, use 2 marks */

  /* Simulate GVANT's "KTEST=1" */
  oState->half_depth--;
  oState->half_depth2++;
  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
  ------------------*/
                               oState->half_length = (oState->max-4) >> 1;
  if ( !(oState->maxdepth%2) ) oState->half_length = (oState->max-7) >> 1;

  oState->depth = 1;

  {
    int i, n;
    struct Level *lev, *lev2;

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }
    lev = &oState->Levels[1];
    for (i = 0; i < n; i++) {
      int limit;
      if (oState->depth <= oState->half_depth2) {
        if (oState->depth <= oState->half_depth) {
          limit = oState->max - OGR[oState->maxdepthm1 - oState->depth];
          limit = limit < oState->half_length ? limit : oState->half_length;
        } else {
          limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - oState->depth);
          limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
        }
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - oState->depth);
      }
      lev->limit = limit;
      register int s = workstub->stub.diffs[i];
      //dump(oState->depth, lev, 0);
      oState->marks[i+1] = oState->marks[i] + s;
      lev->cnt2 += s;
      register int t = s;
      while (t >= 32) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        t -= 32;
      }
      if (t > 0) {
        COMP_LEFT_LIST_RIGHT(lev, t);
      }
      lev2 = lev + 1;
      COPY_LIST_SET_BIT(lev2, lev, s);
      COPY_DIST_COMP(lev2, lev);
      lev2->cnt1 = lev->cnt2;
      lev2->cnt2 = lev->cnt2;
      lev++;
      oState->depth++;
    }
    oState->depth--; // externally visible depth is one less than internal
  }

  oState->startdepth = workstub->stub.length;

/*
  printf("sizeof      = %d\n", sizeof(struct State));
  printf("max         = %d\n", oState->max);
  printf("maxdepth    = %d\n", oState->maxdepth);
  printf("maxdepthm1  = %d\n", oState->maxdepthm1);
  printf("half_length = %d\n", oState->half_length);
  printf("half_depth  = %d\n", oState->half_depth);
  printf("half_depth2 = %d\n", oState->half_depth2);
  {
    int i;
    printf("marks       = ");
    for (i = 1; i < oState->depth; i++) {
      printf("%d ", oState->marks[i]-oState->marks[i-1]);
    }
    printf("\n");
  }
*/

  return CORE_S_OK;
}

#ifdef OGR_DEBUG
static void dump_ruler(struct State *oState, int depth)
{
  int i;
  printf("max %d ruler ", oState->max);
  for (i = 1; i < depth; i++) {
    printf("%d ", oState->marks[i] - oState->marks[i-1]);
  }
  printf("\n");
}
#endif


static int ogr_cycle(void *state, int *pnodes)
{
  struct State *oState = (struct State *)state;
  int depth = oState->depth+1;      /* the depth of recursion */
  struct Level *lev = &oState->Levels[depth];
  struct Level *lev2;
  int nodes = 0;
  int nodeslimit = *pnodes;
  int retval = CORE_S_CONTINUE;
  int limit;
  int s;
  U comp0;

#ifdef OGR_DEBUG
  oState->LOGGING = 1;
#endif
  for (;;) {

    oState->marks[depth-1] = lev->cnt2;
#ifdef OGR_DEBUG
    if (oState->LOGGING) dump_ruler(oState, depth);
#endif
    if (depth <= oState->half_depth2) {
      if (depth <= oState->half_depth) {
        //dump_ruler(oState, depth);
        if (nodes >= nodeslimit) {
          break;
        }
        limit = oState->max - OGR[oState->maxdepthm1 - depth];
        limit = limit < oState->half_length ? limit : oState->half_length;
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - depth);
        limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
      }
    } else {
      limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - depth);
    }

#ifdef OGR_DEBUG
    if (oState->LOGGING) dump(depth, lev, limit);
#endif

    nodes++;

    /* Find the next available mark location for this level */
stay:
    comp0 = lev->comp[0];
#ifdef OGR_DEBUG
    if (oState->LOGGING) printf("comp0=%08x\n", comp0);
#endif

    if (comp0 < 0xfffffffe) {
      #if defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) /* 0 <= x < 0xfffffffe */
      s = LOOKUP_FIRSTBLANK( comp0 );
      #else
      if (comp0 < 0xffff0000) 
        s = ogr_first_blank[comp0 >> 16];
      else {    
        /* s = 16 + ogr_first_blank[comp0 & 0x0000ffff]; slow code */
        s = 16 + ogr_first_blank[comp0 - 0xffff0000];
      }        
      #endif
#ifdef OGR_DEBUG
  if (oState->LOGGING) printf("depth=%d s=%d len=%d limit=%d\n", depth, s+(lev->cnt2-lev->cnt1), lev->cnt2+s, limit);
#endif
      if ((lev->cnt2 += s) > limit) goto up; /* no spaces left */
      COMP_LEFT_LIST_RIGHT(lev, s);
    } else {
      /* s>32 */
      if ((lev->cnt2 += 32) > limit) goto up; /* no spaces left */
      COMP_LEFT_LIST_RIGHT_32(lev);
      if (comp0 == 0xffffffff) goto stay;
    }


    /* New ruler? */
    if (depth == oState->maxdepthm1) {
      oState->marks[oState->maxdepthm1] = lev->cnt2;       /* not placed yet into list arrays! */
      if (found_one(oState)) {
        retval = CORE_S_SUCCESS;
        break;
      }
      goto stay;
    }

    /* Go Deeper */
    lev2 = lev + 1;
    COPY_LIST_SET_BIT(lev2, lev, lev->cnt2-lev->cnt1);
    COPY_DIST_COMP(lev2, lev);
    lev2->cnt1 = lev->cnt2;
    lev2->cnt2 = lev->cnt2;
    lev->limit = limit;
    lev++;
    depth++;
    continue;

up:
    lev--;
    depth--;
    if (depth <= oState->startdepth) {
      retval = CORE_S_OK;
      break;
    }
    limit = lev->limit;

    goto stay; /* repeat this level till done */
  }

  #if 0 /* oState->Nodes is unused (count is returned through *pnodes) */
  // oState->Nodes += nodes;
  {
    U new_hi = oState->Nodes.hi;
    U new_lo = oState->Nodes.lo;
    new_lo += nodes;
    if (new_lo < oState->Nodes.lo)
    {
      if ((++new_hi) < oState->Nodes.hi)
        new_hi = new_lo = ((U)ULONG_MAX);
    } 
    oState->Nodes.hi = new_hi;
    oState->Nodes.lo = new_lo;
  }    
  #endif
  oState->depth = depth-1;

  *pnodes = nodes;

  return retval;
}

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }
  workstub->stub.marks = (u16)oState->maxdepth;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    workstub->stub.diffs[i] = (u16)(oState->marks[i+1] - oState->marks[i]);
  }
  workstub->worklength = oState->depth;
  if (workstub->worklength > STUB_MAX) {
    workstub->worklength = STUB_MAX;
  }
  return CORE_S_OK;
}

static int ogr_destroy(void *state)
{
  #if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  if (state) free(state); 
  #else
  state = state;
  #endif
  return CORE_S_OK;
}

#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
static int ogr_count(void *state)
{
  return sizeof(struct State);
}

static int ogr_save(void *state, void *buffer, int buflen)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_MEMORY;
  }
  memcpy(buffer, state, sizeof(struct State));
  return CORE_S_OK;
}

static int ogr_load(void *buffer, int buflen, void **state)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  *state = malloc(sizeof(struct State));
  if (*state == NULL) {
    return CORE_E_MEMORY;
  }
  memcpy(*state, buffer, sizeof(struct State));
  return CORE_S_OK;
}
#endif

static int ogr_cleanup(void)
{
  return CORE_S_OK;
}

CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void)
{
  static CoreDispatchTable dispatch_table;
  dispatch_table.init      = ogr_init;
  dispatch_table.create    = ogr_create;
  dispatch_table.cycle     = ogr_cycle;
  dispatch_table.getresult = ogr_getresult;
  dispatch_table.destroy   = ogr_destroy;
#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  dispatch_table.count     = ogr_count;
  dispatch_table.save      = ogr_save;
  dispatch_table.load      = ogr_load;
#endif
  dispatch_table.cleanup   = ogr_cleanup;
  return &dispatch_table;
}

