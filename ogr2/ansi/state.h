/* Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ogrstate.h should be included from ogr.cpp only.
 * perhaps it will become integrated into ogr.cpp or ogr_*.cor
*/
#ifndef __STATE_H__
#define __STATE_H__ "@(#)$Id: state.h,v 1.1.2.2 2001/07/08 18:25:32 andreasb Exp $"

#include "ogr.h"

//FIXME
// Default, cores should redefine it if needed
#define OGR_MEM_ALIGNMENT 3

#include <limits.h>
#if (UINT_MAX < 0xfffffffful)
  #error "ogr needs an int thats not less than 32bits"
#elif (UINT_MAX == 0xffffffff)
  #define OGR_INT_SIZE 4
#elif (UINT_MAX == 0xffffffffffffffff)
  #define OGR_INT_SIZE 8
#else
  #error "What's up Doc?"
#endif

#ifdef __VEC__
  #error  #define OGR_VEC_SIZE ???
#else
  #define OGR_VEC_SIZE 0
#endif


// Internal stuff that's not part of the interface but we need for
// declaring the problem work area size.

// I have to reserve memory for all possible OGR cruncher setups because
// memory reservation happens inside problem.h/.c and I cannot know what
// cruncher is going to get used :(

#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40

typedef u32 U;

//#define OGR_PROFILE
// OGR_WINDOW is used to test register windowing in the core
//#define OGR_WINDOW 10


struct Level {
  /* If AltiVec is possible we must reserve memory, just in case */
  #ifdef __VEC__   // unused if OGROPT_ALTERNATE_CYCLE == 0 || == 1
  vector unsigned int listV0, listV1, compV0, compV1;
  #endif
  U list[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 2
  U dist[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 1 || 2
  U comp[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 2
  int cnt1;        // unused if OGROPT_ALTERNATE_CYCLE == 1 || == 2     TODO: this is a duplicate of previous_level->cnt2;
  int cnt2;        // always needed
  int limit;       // always needed
};

#define OGR_LEVEL_SIZE ((128*4)+((4*BITMAPS)*3)+(OGR_INT_SIZE*3))

#if 0
struct State {
  #if 0 /* unused - see notes for ogr_cycle() above */
  struct { U hi,lo; } Nodes;      /* counts "tree branches" */
  //double Nodes;                 /* counts "tree branches" */  
  #endif
  int max;                        /* maximum length of ruler */
  int maxdepth;                   /* maximum number of marks in ruler */
  int maxdepthm1;                 /* maxdepth-1 */
  int half_length;                /* half of max */
  int half_depth;                 /* half of maxdepth */
  int half_depth2;                /* half of maxdepth, adjusted for 2nd mark */
  int marks[MAXDEPTH+1];          /* current length */
  int startdepth;
  int depth;
  int limit; // unused
  #ifdef OGR_DEBUG
    int LOGGING;
  #endif
  #ifdef  OGR_WINDOW /* used by OGRtestbench */
    int wind;                     /* depth window base */
    int turn;                     /* window turn counter */
  #endif
  #ifdef OGR_PROFILE /* used by OGRtestbench */
    struct {
      long hd;                    /* Half depth */
      long hd2;                   /* Half depth 2 */
      long ghd;                   /* Greater than Half depth */
      long lt16;                  /* shift <16 */
      long lt32;                  /* shift < 32 */
      long ge32;                  /* shift >= 32 */
      long fo;                    /* found one? */
      long push;                  /* Go deeper */
   } prof;
  #endif
  /* If AltiVec is possible we must reserve memory, just in case */
  #ifdef __VEC__     /* only used by OGROPT_ALTERNATE_CYCLE == 2 */
    vector unsigned int distV0, distV1;
  #endif
  U dist[BITMAPS];   /* only used by OGROPT_ALTERNATE_CYCLE == 1 */
  struct Level Levels[MAXDEPTH];
#else
struct State {
  /* all variables will be initialized by ogr_create() */
  /* State may not contain pointers pointing into State itself! */
  
  /* Part 1: variables that won't get changed after ogr_create() */
  int core;                       /* our core id */
  int stub_error;                 /* don't process stub if not zero */
  int max;                        /* maximum length of ruler */
  int maxmarks;                   /* was: maxdepth */ /* maximum number of marks in ruler */
  int maxdepth;                   /* was: maxdepthm1 */ /* maximum number of first differences in ruler = marks - 1 */
  int half_length;                /* maximum length of left segment */
  int half_depth;                 /* depth of left/right segment */
  int half_depth2;                /* depth of left+middle segment */
  int startdepth;                 /* depth of the stub */
  int stopdepth;                  /* ogr_cycle() stops if this level is reached; either startdepth or startdepth-1 */
  int depththdiff;                /* save stub2->diffs[depth-1] of a type 2 stub, 0 if type 1 stub */
  int cycle;                      /* count recycles (from 1), 0 if manually generated stub => defines pseudo fifo order */
  
  /* Part 2: variables that will be changed by ogr_cycle() and read by 
             ogr_getresult(). Do not read these values while ogr_cycle() is running!
             The state represented by parts 1&2 and returned by ogr_getresult
             is safe to be saved to disk. */
  int depth;                      /* depth of last placed mark */
//int markpos[MAXDEPTH];          /* duplicates Levels[].cnt2 */ /* was: marks */ /* current positions of the marks */
  struct { u32 hi, lo; } nodes;   /* our internal nodecounter */
  
  /* Part 3: Variables that may be used ONLY by ogr_cycle() */
  #ifdef OGR_DEBUG
    int LOGGING;
  #endif
  #ifdef  OGR_WINDOW /* used by OGRtestbench */
    int wind;                     /* depth window base */
    int turn;                     /* window turn counter */
  #endif
  #ifdef OGR_PROFILE /* used by OGRtestbench */
    struct {
      long hd;                    /* Half depth */
      long hd2;                   /* Half depth 2 */
      long ghd;                   /* Greater than Half depth */
      long lt16;                  /* shift <16 */
      long lt32;                  /* shift < 32 */
      long ge32;                  /* shift >= 32 */
      long fo;                    /* found one? */
      long push;                  /* Go deeper */
   } prof;
  #endif
  /* If AltiVec is possible we must reserve memory, just in case */
  #ifdef __VEC__     /* only used by OGROPT_ALTERNATE_CYCLE == 2 */
    vector unsigned int distV0, distV1;
  #endif
  U dist[BITMAPS];   /* only used by OGROPT_ALTERNATE_CYCLE == 1 */
  struct Level Levels[MAXDEPTH];
};
#endif

#if 0
#define OGR_PROBLEM_SIZE (/*16+*/(6*OGR_INT_SIZE)+(OGR_INT_SIZE*(MAXDEPTH+1))+ \
                         (4*OGR_INT_SIZE)+(128*2)+(OGR_INT_SIZE*BITMAPS)+ \
                         (OGR_LEVEL_SIZE*MAXDEPTH)+64)
                         /* sizeof(struct State) */
#else
#ifdef OGR_DEBUG
  #define OGR_STATE_SIZE_DEBUG (OGR_INT_SIZE)
#else
  #define OGR_STATE_SIZE_DEBUG 0
#endif
#ifdef OGR_WINDOW
  #define OGR_STATE_SIZE_WINDOW (2*(OGR_INT_SIZE))
#else
  #define OGR_STATE_SIZE_WINDOW 0
#endif
#ifdef OGR_PROFILE
  #define OGR_STATE_SIZE_PROFILE (8*(OGR_LONG_SIZE))
#else
  #define OGR_STATE_SIZE_PROFILE 0
#endif
#define XOGR_PROBLEM_SIZE ((8*OGR_INT_SIZE)+ \
                         (OGR_INT_SIZE)+((MAXDEPTH)*OGR_INT_SIZE)+(2*4)+ \
                         (OGR_STATE_SIZE_DEBUG)+(OGR_STATE_SIZE_WINDOW)+ \
                         (OGR_STATE_SIZE_PROFILE)+(2*OGR_VEC_SIZE)+ \
                         (BITMAPS*4)+ \
                         (MAXDEPTH*OGR_LEVEL_SIZE)+64)
                         /* sizeof(struct State) */
#endif


/* some constraints */
#if (STUB_MAX > MAXDEPTH)
#error STUB_MAX > MAXDEPTH
#endif

#endif /* !__STATE_H__ */
