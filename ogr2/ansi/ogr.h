/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogr.h,v 1.1.2.9 2001/01/14 02:37:09 andreasb Exp $
*/
#ifndef __OGR_H__
#define __OGR_H__ 

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

/* ===================================================================== */

/*
 * Constants for return values from all the below functions.
 * Those starting with CORE_S are success codes, and those starting
 * with CORE_E are error codes.
 */
#define CORE_S_OK       0
#define CORE_S_CONTINUE 1
#define CORE_S_SUCCESS  2
#define CORE_E_MEMORY   (-1)
#define CORE_E_IO       (-2)
#define CORE_E_FORMAT   (-3)
#define CORE_E_STOPPED  (-4)
/* Stub is not Golomb or exceeds a limit, so it can't  */
#define CORE_E_STUB     (-5)
/* current ogr_choose_dat2 has not enough entries for this ruler length */
#define CORE_E_CHOOSE   (-6)

/* ===================================================================== */

/*
 * use the new ogr_choose_dat2
 * - correct precalculated values for choose(bitmap, 13)
 * - may be extended up to CHOOSE_MARKS = 16
 * - alignment on 16 byte borders
 * - header moved to variables
 * - choose(x,y) needs less instructions !
 *
 * You need to define this here in ogr.h, because files other than ogr.cpp 
 * (e.g. selftest) depend on this setting.
 */
//#define OGROPT_NEW_CHOOSEDAT

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
   * result. If called at other times, returns the current state of the 
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
#undef OGR_DEBUG
//#define OGR_PROFILE
// OGR_WINDOW is used to test register windowing in the core
//#define OGR_WINDOW 10

// specifies the number of ruler diffs can be represented.
// Warning: increasing this will cause all structures based
// on workunit_t in packets.h to change, possibly breaking
// network and buffer structure operations.
#define STUB_MAX 10

struct Stub { /* size is 24 */
  u16 marks;           /* N-mark ruler to which this stub applies */
  u16 length;          /* number of valid elements in the stub[] array */
  u16 diffs[STUB_MAX]; /* first <length> differences in ruler */
};

struct WorkStub { /* size is 28 */
  Stub stub;           /* stub we're working on */
  u32 worklength;      /* depth of current state */
};

// Internal stuff that's not part of the interface but we need for
// declaring the problem work area size.

// I have to reserve memory for all possible OGR cruncher setups because
// memory reservation happens inside problem.h/.c and I cannot know what
// cruncher is going to get used :(

#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40

typedef u32 U;

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

#define OGR_LEVEL_SIZE ((128*4)+((4*BITMAPS)*3)+(OGR_INT_SIZE*3))

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
//  int limit; // unused
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

#ifndef MIPSpro
#pragma pack()
#endif

#define OGR_PROBLEM_SIZE (/*16+*/(6*OGR_INT_SIZE)+(OGR_INT_SIZE*(MAXDEPTH+1))+ \
                         (4*OGR_INT_SIZE)+(128*2)+(OGR_INT_SIZE*BITMAPS)+ \
                         (OGR_LEVEL_SIZE*MAXDEPTH)+64)
                         /* sizeof(struct State) */

#ifdef OGROPT_NEW_CHOOSEDAT

#if defined(__cplusplus)
extern "C" {
#endif

  extern const int choose_version;
  extern const int choose_distbits;
  extern const int choose_max_marks;
  extern const int choose_align_marks;

  extern const unsigned char ogr_choose_dat2[];

#if defined(__cplusplus)
}
#endif

#endif /* OGROPT_NEW_CHOOSEDAT */

unsigned long ogr_nodecount(const struct Stub *);
const char *ogr_stubstr_r(const struct Stub *stub, 
                          char *buffer, unsigned int bufflen,
                          int worklength);
const char *ogr_stubstr(const struct Stub *stub);

#endif /* __OGR_H__ */
