/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __OGR_H__
#define __OGR_H__ "@(#)$Id: ogr.h,v 1.1.2.18.2.3 2001/04/01 20:19:09 andreasb Exp $"

// define this to use the new struct Stub
//#define OGR_NEW_STUB_FORMAT

// stubmap generation
//#define OGR_CALLBACK


#ifndef u16
#include "cputypes.h"
#endif

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

/* ===================================================================== */

/*
 * Constants for return values from all the below functions.
 * Those starting with CORE_S are success codes, and those starting
 * with CORE_E or STUB_E are error codes.
 */
#define CORE_S_OK       0
#define CORE_S_CONTINUE 1
#define CORE_S_SUCCESS  2
#define CORE_E_MEMORY    (-1)
//#define CORE_E_IO      (-2)
#define CORE_E_FORMAT    (-3)
//#define CORE_E_STOPPED (-4)

/* different STUB_E_ my be ORed together */
#define STUB_OK          0
/* Stub is not Golomb, i.e. measures a distance twice */
#define STUB_E_GOLOMB    1
/* Stub exceeds a limit, so it's too long to contain an optimal ruler */
#define STUB_E_LIMIT     2
/* current ogr_choose_dat2 has not enough entries for this rulers marks */
#define STUB_E_MARKS     4

/* ===================================================================== */

/*
 * You need to define this here in ogr.h, because files other than ogr.cpp 
 * (e.g. selftest) depend on this setting.
 */

/* either undef'd or 3 is implemented in common/selftest.cpp */
//#undef OGR_ALTERNATE_TESTCASES
#define OGR_ALTERNATE_TESTCASES 3

/* ===================================================================== */

#ifndef MIPSpro
#pragma pack(1)
#endif

/*
 * Dispatch table structure. A pointer to one of these should be returned
 * from an exported function in the module.
 *
 * The result callback is weird, it implies that only one client in the
 * entire process will use this core (but that client can have multiple
 * threads). Oh well.
 */
typedef struct {
  /*
   * Initialize the core, called once for all threads.
   */
  int (*init)(void);

  /*
   * Returns the core name.
   */
  const char* (*name)(void);

  /*
   * Returns the cores RCS Id list
   */
  const char* (*core_id)(void);
  
  /*
   * Create a new work unit, called once for each thread.
   * The format of input is defined by the core.
   */
  int (*create)(void *input, int inputlen, void *state, int statelen);

  /*
   * Continue working, return CORE_S_OK if no more work to do, or
   * CORE_S_CONTINUE if things need to keep going.
   * On input, nodes should contain the number of algorithm iterations
   * to do. On output, nodes will contain the actual number of iterations
   * done.
   *
   * If with_time_constraints is 0, the OGR cruncher uses the nodeslimit 
   * merely as a hint when to leave the cruncher rather than adhering to it 
   * exactly. There is a slight speed increase (negligible in a preemptive 
   * non-realtime environment) when applying the hint scheme, but cannot 
   * be used in a cruncher that is bound by time constraints (eg 
   * non-preemptive or real-time environments) because it can happen 
   * that a nodeslimit request of 1000 will end up with 32000 nodes done.
  */
  int (*cycle)(void *state, int *steps, int with_time_constraints);

  /*
   * If cycle returns CORE_S_SUCCESS, call getresult to get the successful
   * result. If called at other times, returns the last stable state of the 
   * search.
   */
  int (*getresult)(void *state, void *result, int resultlen);

  /*
   * Clean up state structure.
   */
  int (*destroy)(void *state);

#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  /*
   * Return the number of bytes needed to serialize this state.
   */
  int (*count)(void *state);

  /*
   * Serialize the state into a flat data structure suitable for
   * persistent storage.
   * buflen must be at least as large as the number of bytes returned
   * by count().
   * Does not destroy the state structure.
   */
  int (*save)(void *state, void *buffer, int buflen);

  /*
   * Load the state from persistent storage buffer.
   */
  int (*load)(void *buffer, int buflen, void **state);
#endif

  /*
   * Clean up anything allocated in init().
   */
  int (*cleanup)(void);

} CoreDispatchTable;

/* ===================================================================== */

// define this to enable LOGGING code
//#undef OGR_DEBUG
//#define OGR_PROFILE
// OGR_WINDOW is used to test register windowing in the core
//#define OGR_WINDOW 10

#ifndef OGR_NEW_STUB_FORMAT
// specifies the number of ruler diffs can be represented.
// Warning: increasing this will cause all structures based
// on workunit_t in packets.h to change, possibly breaking
// network and buffer structure operations.
#define STUB_MAX 10

struct Stub { /* size is 24 */
  u16 marks;           /* N-mark ruler to which this stub applies */
  u16 length;          /* number of valid elements in the diffs[] array */
  u16 diffs[STUB_MAX]; /* first <length> differences in ruler */
};

struct WorkStub { /* size is 28 */
  Stub stub;           /* stub we're working on */
  u32 worklength;      /* depth of current state */
};

#else /* OGR_NEW_STUB_FORMAT */
#define STUB_MAX 32

/* we are safe to use bytes for the diffs up to OGR-29 at least, because
 * an upper limit for the length of a first difference is
 * max_first_diff_length = OGR_length[depth] - sum(i = 1 .. depth - 1, i)
 * and max_first_diff_length < 255 for depth up to 29
 * But check in ogr_getresult() whether a diff exceeds this, anyway. 
 * Some limits might be not strong enough.
 */

/* The client may read these values, but NOT MODIFY them !!! (except ntoh/hton()) */
/* maximum size is 48 */
struct NewStub { /* size is 1+1+1+32+8 = 43 [+1 = 44] [+1+1 = 46] */
  u8 marks;            /* N-mark ruler to which this stub applies */
  u8 depth;            /* startdepth, can calculate stopdepth from depth,depththdiff */
  u8 workdepth;        /* depth of current state */
  u8 diffs[STUB_MAX];  /* first <depth> differences in ruler */
  struct { u32 hi, lo; } nodes;
  /* anything else needed ? */
  u8 depththdiff;      /* if 0: stopdepth = startdepth
                          else: stopdepth = startdepth - 1 ==> finish the remaining stubs on this branch 
                                ==> depththdiff = diffs[depth-1] */
  /* marks, depth, diffs[0 .. depth-2], depththdiff are CONST. Do not modify them in the core!
     workdepth, diffs[depth-1 .. STUB_MAX-1], nodes may be modified by the core. */
  /* we need an ogr_resetstub() function. */
  
  u8 core;             /* save the core number */
  u8 cycle;            /* count recycles for fifo-buffers */
};

#endif /* OGR_NEW_STUB_FORMAT */

// Internal stuff that's not part of the interface but we need for
// declaring the problem work area size.

// I have to reserve memory for all possible OGR cruncher setups because
// memory reservation happens inside problem.h/.c and I cannot know what
// cruncher is going to get used :(

#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40

typedef u32 U;

#ifdef OGR_CORE_INTERNAL_STRUCTURES
struct Level {
  /* If AltiVec is possible we must reserve memory, just in case */
  #ifdef __VEC__   // unused if OGROPT_ALTERNATE_CYCLE == 0 || == 1
  vector unsigned int listV0, listV1, compV0, compV1;
  #endif
  U list[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 2
  U dist[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 1 || 2
  U comp[BITMAPS]; // unused if OGROPT_ALTERNATE_CYCLE == 2
  int cnt1;        // unused if OGROPT_ALTERNATE_CYCLE == 1 || == 2
  int cnt2;        // always needed
  int limit;       // always needed
};
#endif /* OGR_CORE_INTERNAL_STRUCTURES */

#define OGR_LEVEL_SIZE ((128*4)+((4*BITMAPS)*3)+(OGR_INT_SIZE*3))

#ifdef OGR_CORE_INTERNAL_STRUCTURES
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
  int stub_error;                 /* don't process stub if not zero */
  int max;                        /* maximum length of ruler */
  int maxmarks;                   /* was: maxdepth */ /* maximum number of marks in ruler */
  int maxdepth;                   /* was: maxdepthm1 */ /* maximum number of first differences in ruler = marks - 1 */
  int half_length;                /* maximum length of left segment */
  int half_depth;                 /* depth of left/right segment */
  int half_depth2;                /* depth of left+middle segment */
  int startdepth;                 /* depth of the stub */
  int stopdepth;                  /* */
  
  /* Part 2: variables that will be changed by ogr_cycle() and read by 
             ogr_getresult(). Do not read these values while ogr_cycle() is running!
             The state represented by parts 1&2 and returned by ogr_getresult
             is safe to be saved to disk. */
  int depth;                      /* depth of last placed mark */
  int markpos[MAXDEPTH];          /* was: marks */ /* current positions of the marks */
  u32 nodeshi, nodeslo;           /* our internal nodecounter */
  
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
#endif /* OGR_CORE_INTERNAL_STRUCTURES */

#ifndef MIPSpro
#pragma pack()
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
#define OGR_PROBLEM_SIZE ((8*OGR_INT_SIZE)+ \
                         (OGR_INT_SIZE)+((MAXDEPTH)*OGR_INT_SIZE)+(2*4)+ \
                         (OGR_STATE_SIZE_DEBUG)+(OGR_STATE_SIZE_WINDOW)+ \
                         (OGR_STATE_SIZE_PROFILE)+(2*OGR_VEC_SIZE)+ \
                         (BITMAPS*4)+ \
                         (MAXDEPTH*OGR_LEVEL_SIZE)+64)
                         /* sizeof(struct State) */
#endif

#if defined(__cplusplus)
extern "C" {
#endif

  extern const int choose_version;
  extern const int choose_dist_bits;
  extern const int choose_max_depth;
  extern const int choose_alignment;

  extern const unsigned char ogr_choose_dat2[];

#if defined(__cplusplus)
}
#endif

unsigned long ogr_nodecount(const struct Stub *);
const char *ogr_stubstr_r(const struct Stub *stub, 
                          char *buffer, unsigned int bufflen,
                          int worklength);
const char *ogr_stubstr(const struct Stub *stub);


/* some constraints */
#if (STUB_MAX > MAXDEPTH)
#error STUB_MAX > MAXDEPTH
#endif

#endif /* __OGR_H__ */
