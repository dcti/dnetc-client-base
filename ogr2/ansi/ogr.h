/* Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __OGR_H__
#define __OGR_H__ "@(#)$Id: ogr.h,v 1.1.2.18.2.6 2001/07/08 18:25:32 andreasb Exp $"

//#define OGR_FORCE_OLD_STUB

#ifndef OGR_FORCE_OLD_STUB
// define this to use the new struct Stub
#define OGR_NEW_STUB_FORMAT
#endif

// stubmap generation
//#define OGR_CALLBACK


#ifndef u16
#include "cputypes.h"
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
//#define CORE_E_MEMORY    (-1)
//#define CORE_E_IO      (-2)
#define CORE_E_FORMAT    (-3)
//#define CORE_E_STOPPED (-4)

// new error codes, old ones should be replaced someday
// client allocated core memory block is too small for struct State:
#define CORE_E_LOWMEM    (-5)
#define CORE_E_NOMEM     (-6)
#define CORE_E_8BIT      (-7)
#define CORE_E_CORENO    (-8)

/*
CORE_E_MEMORY
CORE_E_ALIGNMENT
CORE_E_
*/

/* different STUB_E_ may be ORed together, do not use more than 8 bit ! */
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

// define this to enable LOGGING code
//#undef OGR_DEBUG

/* ===================================================================== */

#ifndef MIPSpro
#pragma pack(4) 
// FIXME: pack CoreDispatchTable on longwork boundaries - 4 or 8 ?
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
   * Returns sizeof(State) and required alignment.
   */
  int (*get_size)(int* alignment);
  
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

// These structures get saved to disk/sent through network
#ifndef MIPSpro
#pragma pack(1) 
#endif

#ifndef OGR_NEW_STUB_FORMAT
#define OGR_OLD_STUB
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
struct Stub2 { /* size is 8+1+1+1+32+1+1+1 = 46 [+2 = 48] */
  struct { u32 hi, lo; } nodes;
  u8 marks;            /* N-mark ruler to which this stub applies */
  u8 depth;            /* startdepth, can calculate stopdepth from depth,depththdiff */
  u8 workdepth;        /* depth of current state */
  u8 diffs[STUB_MAX];  /* first <depth> differences in ruler */
  u8 depththdiff;      /* if 0: stopdepth = startdepth
                          else: stopdepth = startdepth - 1 ==> finish the remaining stubs on this branch 
                                ==> depththdiff = diffs[depth-1] */
  /* marks, depth, diffs[0 .. depth-2], depththdiff are CONST. Do not modify them in the core!
     workdepth, diffs[depth .. STUB_MAX-1], nodes may be modified by the core. 
     diffs[depth-1] may be modified by the core if depththdiff != 0. */
  
  u8 core;             /* save the core number */
  u8 cycle;            /* count recycles for fifo-buffers, start with 1, so 0 = manual generated, process first */
  u8 dummy[2];
};

/* maximum size is 36 */
#define NET_STUB_MAX (36-8-1-1-1-(0)-1-1-1)
struct NetStub2 { /* size is 36 */
  struct { u32 hi, lo; } nodes;
  u8 marks;            /* N-mark ruler to which this stub applies */
  u8 depth;            /* startdepth, can calculate stopdepth from depth,depththdiff */
  u8 workdepth;        /* depth of current state */
  u8 diffs[NET_STUB_MAX];  /* first <depth> differences in ruler */
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

#ifndef MIPSpro
#pragma pack()
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

#ifdef OGR_OLD_STUB
//unsigned long ogr_nodecount(const struct Stub *);
const char *ogr_stubstr_r(const struct Stub *stub, 
                          char *buffer, unsigned int bufflen,
                          int worklength);
const char *ogr_stubstr(const struct Stub *stub);
#else
int ogr_init_stub2_from_testcasedata(struct Stub2 *stub, u32 marks, u32 ndiffs, const u32 *diffs);

/* hton(struct Stub2) / ntoh(struct Stub2) */
int ogr_switch_byte_order(struct Stub2 *stub);

int ogr_reset_stub(struct Stub2 *stub);

/* format = 0 without workpos / format = 1 with workpos */
const char *ogr_stubstr_r(const struct Stub2 *stub, 
                          char *buffer, unsigned int bufflen,
                          int format);
const char *ogr_stubstr(const struct Stub2 *stub);

int ogr_netstub2_to_stub2(const NetStub2 *netstub, Stub2 *stub, int switch_byte_order);
int ogr_stub2_to_netstub2(const Stub2 *stub, NetStub2 *netstub, int switch_byte_order);

int ogr_benchmark_stub(struct Stub2* stub);
#endif

// increase this value if the client reports "not enough core memory for OGR"
#define OGR_PROBLEM_SIZE 2960

#endif /* __OGR_H__ */
