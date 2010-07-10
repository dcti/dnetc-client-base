/*
 * Copyright distributed.net 1998-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *stress_r72_cpp(void) {
return "@(#)$Id: stress.cpp,v 1.9 2010/07/10 17:34:24 stream Exp $"; }

#include "cputypes.h"
#include "client.h"
#include "baseincs.h"  // standard #includes
#include "problem.h"   // Problem class
#include "triggers.h"  // CheckExitRequestTrigger()
#include "logstuff.h"  // LogScreen()
#include "rsadata.h"
#include "selftest.h"
#include "ansi/rotate.h"
#include "ccoreio.h"

/*
** PURPOSE :
**    The stress-test is designed to perform an extensive review of RC5-72
**    cores effectiveness. It is an incremental test-suite in that it first
**    tests the most direct code path. If successful, the tests target the
**    key incrementation block and the main processing loop. Then, the tests
**    focus on every possible paths possibly taken while incrementing the
**    keys. Finally, a long run test is performed on a part of a known
**    keyblock.
**
**    This arrangement allows the test suite to build up on preceding tests
**    in order to hint at locations where a bug, if any, is likely to reside.
**
**    The basic (simplified) block diagram of an RC5 core looks as follow :
**
**    +-----------------------------------+
**    |             CORE INIT             |
**    |                                   |
**    | -> RC5_72UnitWork *rc5_72unitwork |
**    | -> u32 *iterations                |
**    +-----------------------------------+
**                      |<----------------------------------+
**                      v                                   |
**    +-----------------------------------+                 |
**    |             PROCESSING            |                 |
**    |                                   |                 |
**    | - Upto 3 nested loops             |                 |
**    | - 1, 2, 3, 4, 8, 12 or 24         |                 |
**    |   processing pipelines            |                 |
**    +-----------------------------------+                 |
**                      |                                   |
**                      v                                   |
**    +-----------------------------------+                 |
**    |          SUCCESS DETECTION        |                 |
**    |                                   |                 |
**    | - Records partial successes       |                 |
**    | - Check for a full match (exit    |                 |
**    |   early)                          |                 |
**    +-----------------------------------+                 |
**                      |                                   |
**                      v                                   |
**    +-----------------------------------+                 |
**    |       KEY(s) INCREMENTATION       |                 |
**    +-----------------------------------+                 |
**                      |-----------------------------------+
**                      v
**    +-----------------------------------+
**    |             CORE EXIT             |
**    | <- RC5_72UnitWork *rc5_72unitwork |
**    | <- u32 *iterations                |
**    +-----------------------------------+
**
** Test #1 : Test all pipelines (straight path from CORE_INIT through to
**           CORE_EXIT - No key iteration)
** Test #2 : Test for every possible key (partial match) taken sequentially
**           within a small subset of a keyblock (about 1000 keys). Only L0hi
**           and the high-order byte of L0mid are affected.
** Test #3 : Test for every possible key (success) taken sequentially in a
**           very small subset so that a carry should be propagated from L0hi,
**           through L0mid, and up to L0lo. Ran several times to cover all
**           possible combinations.
** Test #4 : Long run test from CA:DB0EF306:30000000 through to
**           CA:DB0EF306:61000000. The 5 partial matches to be discovered are
**           #1 : CA:DB0EF306:30F0C71F
**           #2 : CA:DB0EF306:357C03CC
**           #3 : CA:DB0EF306:3EF068E2
**           #4 : CA:DB0EF306:3EF57E4E
**           #5 : CA:DB0EF306:609CDAE6
**           The test is arranged so that partial matches #3 and #4 should be
**           found in a single run (timeslice).
**
** WARNING :
**  It is possible for a core to trash one of the key to be used in the next
**  iteration without being catched by any of these tests, providing the core
**  state is restored at the end of the next iteration. The reason for this is
**  that it's not possible to find two consecutive (or close to each other)
**  partial matches (or a partial match and a full match), so we have no way
**  to assert the core effectiveness right after a partial match occurs.
**  As a result, the code that deals with partial matches detection MUST BE
**  CHECKED CAREFULY to prevent the occurence of such invisible bugs.
*/

#define KEYBASE_HI  0x000000CA
#define KEYBASE_MID 0xDB0EF306
#define KEYBASE_LO  0x30000000

#define MINIMUM_ITERATIONS 48     /* Borrowed from problem.cpp */


struct RC5_Key {
  u32 hi;
  u32 mid;
  u32 lo;
};


static void  __SwitchRC572Format(u32 *hi, u32 *mid, u32 *lo)
{
    register u32 tempkeylo = *lo;
    register u32 tempkeymid = *mid;
    register u32 tempkeyhi = *hi;

    *lo  = ((tempkeyhi)        & 0x000000FFL) |
           ((tempkeymid >> 16) & 0x0000FF00L) |
           ((tempkeymid)       & 0x00FF0000L) |
           ((tempkeymid << 16) & 0xFF000000L);
    *mid = ((tempkeymid)       & 0x000000FFL) |
           ((tempkeylo >> 16)  & 0x0000FF00L) |
           ((tempkeylo)        & 0x00FF0000L) |
           ((tempkeylo << 16)  & 0xFF000000L);
    *hi  = ((tempkeylo)        & 0x000000FFL);
}


static void __IncrementKey(u32 *keyhi, u32 *keymid, u32 *keylo, u32 iters)
{
  *keylo = *keylo + iters;
  if (*keylo < iters)
  {
    *keymid = *keymid + 1;
    if (*keymid == 0) *keyhi = *keyhi + 1;
  }
}


/*****************************************************************************
** Encrypt the plain text (plain.lo and plain.hi) to build known full match.
** The matching key is contestwork->bigcrypto.key + offset, and it is
** returned in 'matchkey' for further reference.
*/

#define P 0xB7E15163
#define Q 0x9E3779B9

static void __cypher_text(ContestWork *contestwork, RC5_Key *matchkey, u32 offset)
{
  u32 i, j, k;
  u32 A, B;
  u32 S[26];
  u32 L[3];
  u32 key_hi, key_mid, key_lo;
  u32 plain_hi, plain_lo;

  key_hi  = contestwork->bigcrypto.key.hi;
  key_mid = contestwork->bigcrypto.key.mid;
  key_lo  = contestwork->bigcrypto.key.lo;
  __IncrementKey(&key_hi, &key_mid, &key_lo, offset);
  matchkey->hi  = key_hi;
  matchkey->mid = key_mid;
  matchkey->lo  = key_lo;

  __SwitchRC572Format(&key_hi, &key_mid, &key_lo);

  L[2] = key_hi;
  L[1] = key_mid;
  L[0] = key_lo;

  plain_hi = contestwork->bigcrypto.plain.hi ^ contestwork->bigcrypto.iv.hi;
  plain_lo = contestwork->bigcrypto.plain.lo ^ contestwork->bigcrypto.iv.lo;

  for (S[0] = P, i = 1; i < 26; i++)
    S[i] = S[i-1] + Q;

  for (A = B = i = j = k = 0;
       k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
  {
    A = S[i] = ROTL3(S[i]+(A+B));
    B = L[j] = ROTL(L[j]+(A+B),(A+B));
  }
  A = plain_lo + S[0];
  B = plain_hi + S[1];
  for (i = 1; i <= 12; i++)
  {
    A = ROTL(A^B,B)+S[2*i];
    B = ROTL(B^A,A)+S[2*i+1];
  }

  contestwork->bigcrypto.cypher.lo = A;
  contestwork->bigcrypto.cypher.hi = B;
}


/*
** Basic initialization of our crypto stuff. Some members are later
** overwritten for specific purposes.
*/
static void __init_contest(ContestWork *contestwork, u32 iters)
{
  memset(contestwork, 0, sizeof(ContestWork));
  contestwork->bigcrypto.key.hi        = KEYBASE_HI;
  contestwork->bigcrypto.key.mid       = KEYBASE_MID;
  contestwork->bigcrypto.key.lo        = KEYBASE_LO;
  contestwork->bigcrypto.iv.lo         = RC572_IVLO;
  contestwork->bigcrypto.iv.hi         = RC572_IVHI;
  contestwork->bigcrypto.plain.lo      = RC572_PLAINLO;
  contestwork->bigcrypto.plain.hi      = RC572_PLAINHI;
  contestwork->bigcrypto.cypher.lo     = RC572_CYPHERLO;
  contestwork->bigcrypto.cypher.hi     = RC572_CYPHERHI;
  contestwork->bigcrypto.keysdone.lo   = 0;
  contestwork->bigcrypto.keysdone.hi   = 0;
  contestwork->bigcrypto.iterations.lo = iters;
  contestwork->bigcrypto.iterations.hi = 0;
  contestwork->bigcrypto.randomsubspace = 0xffff; /* invalid, tests don't propagate random subspaces */
}


static long __check_result(int test, ContestWork *contestwork, int pipenum,
      u32 expected_count, u32 expected_iters, RC5_Key *match)
{
  long success = 1L;
  if (contestwork->bigcrypto.check.count != expected_count) {
    success = -1L;
    Log("\rRC5-72: Stress-test %d: Pipe #%d fails to set 'check.count'\n", test, pipenum);
    Log("Got 0x%08X, expected 0x%08X\n", contestwork->bigcrypto.check.count, expected_count);
  }
  if (contestwork->bigcrypto.check.hi != match->hi
        || contestwork->bigcrypto.check.mid != match->mid
        || contestwork->bigcrypto.check.lo != match->lo) {
    success = -1L;
    Log("\rRC5-72: Stress-test %d: Pipe #%d fails to set 'check.hi/mid/lo'\n", test, pipenum);
    Log("check:  %02X:%08X:%08X, expected %02X:%08X:%08X\n",
        contestwork->bigcrypto.check.hi, contestwork->bigcrypto.check.mid, contestwork->bigcrypto.check.lo,
        match->hi, match->mid, match->lo);
  }
  if (contestwork->bigcrypto.keysdone.hi != 0 || contestwork->bigcrypto.keysdone.lo != expected_iters) {
    success = -1L;
    Log("\rRC5-72: Stress-test %d: Pipe #%d - Iterations count not updated\n", test, pipenum);
    Log("Got 0x%08X, expected 0x%08X\n", contestwork->bigcrypto.keysdone.lo, expected_iters);
  }
  return success;
}


/*****************************************************************************
** Test #1 : Test each pipeline sequentially and only once (straight path from
**           CORE_INIT through to CORE_EXIT - No key iteration)
**
** Failures in this test point to :
** - Prolog. Some datas are not initialized.
** - Miscalculations. The algorithm is not implemented properly.
** - Full match detection. The core doesn't notice there's a full match or
**      fail to copy relevant datas to 'check.' members.
** - Epilog. Bogus cleanup.
*/
static long __test_1(void)
{
  ContestWork contestwork;
  Problem *thisprob;
  u32 pipes = MINIMUM_ITERATIONS;   /* Max number of pipelines */
  u32 tslice = MINIMUM_ITERATIONS;
  u32 iters = 0;
  long success = 1L;
  RC5_Key matchkey;

  if (CheckExitRequestTrigger())
    return -1L;

  do {
    int resultcode;

    __init_contest(&contestwork, MINIMUM_ITERATIONS);
    thisprob = ProblemAlloc();
    if (thisprob) {
      __cypher_text(&contestwork, &matchkey, iters);
      if (ProblemLoadState(thisprob, &contestwork, RC5_72, tslice, 0, 0, 0, 0, NULL) == 0) {
        pipes = thisprob->pub_data.pipeline_count;

        ProblemRun(thisprob);
        if (CheckExitRequestTrigger())
          success = 0L;

        resultcode = ProblemRetrieveState(thisprob, &contestwork, NULL, 1, 0);

        /* Check the number of pipelines here, once and for all. */
        if (pipes != 1 && pipes != 2 && pipes != 3 && pipes != 4 &&
            pipes != 8 && pipes != 12 && pipes != 16 && pipes != 24) {
          Log("\rRC5-72 : INTERNAL ERROR - Number of pipes = %d\n", pipes);
          success = -1L;
        }

        if (success != 0L) {
          if (resultcode != RESULT_FOUND) {
            success = -1L;
            Log("\rRC5-72: Stress-test 1: Pipe #%d missed a full match\n", iters+1);
          }
          success |= __check_result(1, &contestwork, iters+1, 1, iters, &matchkey);
        }
      }
    }
    ProblemFree(thisprob);
  } while (++iters < pipes && success != 0L);

  if (success < 0) {
    Log("RC5-72: Stress-test 1 FAILED\n");
    Log("Possible errors locations :\n");
    Log("- Prolog/Epilog\n");
    Log("- Miscalculations in the main body\n");
    Log("- Full match detection\n");
  }
  else if (success == 0) {
    success = -1L;
    Log("RC5-72: *** break ***\n");
  }
  else
    Log("RC5-72: Stress-test 1 passed\n");

  return success;
}


/*****************************************************************************
** Test #2 : Test for every possible key (partial match) taken sequentially
**           within a small subset of a keyblock (about 1000 keys). Only L0hi
**           and the high-order byte of L0mid are affected.
**
** Failures in this test point to :
** - Partial match detection.
** - Key incrementation.
** - Main loop re-initialization after a key incrementation occurs.
*/
static long __test_2(void)
{
  ContestWork contestwork;
  Problem *thisprob;
  u32 maxkeys = 0x420;      /* Must be an even multiple of MINIMUM_ITERATIONS */
                            /* and must be > 768 to test 3-pipe cores.        */
  u32 pipes = 1;
  u32 tslice = maxkeys;
  u32 iters = 0;
  long success = 1L;
  RC5_Key matchkey;

  if (CheckExitRequestTrigger())
    return -1L;

  do {
    int resultcode;

    __init_contest(&contestwork, maxkeys);
    thisprob = ProblemAlloc();
    if (thisprob) {
      __cypher_text(&contestwork, &matchkey, iters);
      /* kludge : Convert a full match into a partial match */
      contestwork.bigcrypto.cypher.hi = ~contestwork.bigcrypto.cypher.hi;

      if (ProblemLoadState(thisprob, &contestwork, RC5_72, tslice, 0, 0, 0, 0, NULL) == 0) {
        pipes = thisprob->pub_data.pipeline_count;

        do {
          if (CheckExitRequestTrigger()) {
            success = 0L;
            break;
          }
        } while(ProblemRun(thisprob) == RESULT_WORKING);

        resultcode = ProblemRetrieveState(thisprob, &contestwork, NULL, 1, 0);

        if (success != 0L) {
          int cpipe = iters % pipes + 1;

          if (resultcode == RESULT_FOUND) {
            success = -1L;      /* A partial match was expected */
            Log("\rRC5-72: Stress-test 2: Pipe #%d found a full match\n", cpipe);
          }
          success |= __check_result(2, &contestwork, cpipe, 1, maxkeys, &matchkey);
        }
      }
    }
    ProblemFree(thisprob);
  } while (++iters < maxkeys && success > 0L);

  if (success < 0) {
    Log("RC5-72: Stress-test 2 FAILED\n");
    Log("Possible errors :\n");
    Log("- Partial match detection fails\n");
    Log("- Miscalculations in key iteration block\n");
    Log("- Main loop re-initialization\n");
  }
  else if (success == 0) {
    success = -1L;
    Log("RC5-72: *** break ***\n");
  }
  else
    Log("RC5-72: Stress-test 2 passed\n");

  return success;
}


/*****************************************************************************
** Test #3 : Test for every possible key (success) taken sequentially in a
**           very small subset so that a carry should be propagated from L0hi,
**           through L0mid, and eventually up to L0lo. Ran several times to
**           cover all possible combinations.
**
** Failures in this test point to :
** - Full match detection
** - Key incrementation.
** - Main loop re-initialization after a key incrementation occurs.
*/

static RC5_Key key_masks[] = {
  {0x00, 0x00000000, 0x00000000},
  {0x00, 0x00000000, 0x0000FF00},
  {0x00, 0x00000000, 0x00FFFF00},
  {0x00, 0x00000000, 0xFFFFFF00},
  {0x00, 0x000000FF, 0xFFFFFF00},
  {0x00, 0x0000FFFF, 0xFFFFFF00},
  {0x00, 0x00FFFFFF, 0xFFFFFF00},
  {0x00, 0xFFFFFFFF, 0xFFFFFF00}
};

static long __test_3(void)
{
  ContestWork contestwork;
  Problem *thisprob;
  u32 maxkeys = MINIMUM_ITERATIONS * 4;
  u32 pipes = 1;
  u32 tslice = maxkeys;
  long success = 1L;
  RC5_Key matchkey;
  RC5_Key basekey;
  int i;

  if (CheckExitRequestTrigger())
    return -1L;

  for (i = 0; i < 8 && success > 0L; i++) {
    u32 iters = 0;

    basekey.hi  = KEYBASE_HI  | key_masks[i].hi;
    basekey.mid = KEYBASE_MID | key_masks[i].mid;
    basekey.lo  = KEYBASE_LO  | key_masks[i].lo;

    /* Arrange the base key so that L0hi will overflow. As a result, we'll
    ** check all possible keys right before and right after the overflow
    ** occurs. This allow us to test for key incrementation bugs, even if
    ** the incrementation is anticipated (pipelined cores).
    */
    basekey.lo = (basekey.lo & 0xFFFFFF00) + (0x100 - MINIMUM_ITERATIONS);
    basekey.lo -= basekey.lo % MINIMUM_ITERATIONS;

    do {
      int resultcode;

      __init_contest(&contestwork, maxkeys);
      contestwork.bigcrypto.key.hi  = basekey.hi;
      contestwork.bigcrypto.key.mid = basekey.mid;
      contestwork.bigcrypto.key.lo  = basekey.lo;

      thisprob = ProblemAlloc();
      if (thisprob) {
        __cypher_text(&contestwork, &matchkey, iters);
        if (ProblemLoadState(thisprob, &contestwork, RC5_72, tslice, 0, 0, 0, 0, NULL) == 0) {
          pipes = thisprob->pub_data.pipeline_count;

          do {
            if (CheckExitRequestTrigger()) {
              success = 0L;
              break;
            }
          } while(ProblemRun(thisprob) == RESULT_WORKING);

          resultcode = ProblemRetrieveState(thisprob, &contestwork, NULL, 1, 0);

          if (success != 0) {
            int cpipe = iters % pipes + 1;

            if (resultcode != RESULT_FOUND) {
              success = -1L;
              Log("\rRC5-72: Stress-test 3: Pipe #%d missed a full match\n", cpipe);
            }
            success |= __check_result(3, &contestwork, cpipe, 1, iters, &matchkey);
          }
        }
      }
      ProblemFree(thisprob);
    } while (++iters < maxkeys && success > 0L);
  }

  if (success < 0) {
    Log("RC5-72: Stress-test 3 FAILED\n");
    Log("Possible errors :\n");
    Log("- Full match detection fails\n");
    Log("- Miscalculations in key iteration block\n");
    Log("- Main loop re-initialization\n");
  }
  else if (success == 0) {
    success = -1L;
    Log("RC5-72: *** break ***\n");
  }
  else
    Log("RC5-72: Stress-test 3 passed\n");

  return success;
}


/*****************************************************************************
** Test #4 : Long run test from CA:DB0EF306:30000000 through to
**           CA:DB0EF306:61000000. The 5 partial matches to be discovered are
**           #1 : CA:DB0EF306:30F0C71F
**           #2 : CA:DB0EF306:357C03CC
**           #3 : CA:DB0EF306:3EF068E2
**           #4 : CA:DB0EF306:3EF57E4E
**           #5 : CA:DB0EF306:609CDAE6
**           The test is arranged so that partial matches #3 and #4 should be
**           found in a single run (timeslice).
*/
static long __test_4(void)
{
  ContestWork contestwork;
  Problem *thisprob;
  u32 maxkeys = 0x33006600;   /* must be > 0x309CDAE6 */
  u32 tslice = 0x00080010;    /* large enough to find partial matches #3 and
                                 #4 in the same run */
  u32 pipes = 1;
  u32 iters = 0;
  long success = 1L;
  RC5_Key cmc_key;
  u32 cmc_count;
  int resultcode;

  if (CheckExitRequestTrigger())
    return -1L;

  __init_contest(&contestwork, maxkeys);
  cmc_key.hi = cmc_key.mid = cmc_key.lo = cmc_count = 0;
  thisprob = ProblemAlloc();

  /* Unlike the three other tests, we reuse the ContestWork datas to emulate
  ** the real behaviour of the client. Any core that fails to collect all
  ** partial matches (one or more in each run) will fail this test.
  */

  if (thisprob && ProblemLoadState(thisprob, &contestwork, RC5_72, tslice, 0, 0, 0, 0, NULL) == 0) {
    u32 sec = 0;
    pipes = thisprob->pub_data.pipeline_count;
    do {
      u32 basekey = KEYBASE_LO + iters;

      ProblemInfo info;
      u32 itersDonehi, itersDonelo, remoteconn=0;

      do {
        if (CheckExitRequestTrigger()) {
          success = 0L;
          break;
        }
        if (ProblemGetInfo(thisprob, &info, P_INFO_DCOUNT) == -1) {
          success = 0L;
          break;
        }
        itersDonehi=info.dcounthi; itersDonelo=info.dcountlo;
        ProblemRun(thisprob);
        if (ProblemGetInfo(thisprob, &info, P_INFO_DCOUNT) == -1) {
          success = 0L;
          break;
        }
        if ((itersDonehi-info.dcounthi)||(itersDonelo-info.dcountlo)) {
          break;
        }
        remoteconn = 1;
      } while(1);

      if (remoteconn) {
        Log("\rRC5-72: Restarting stress-test 4\n");
        iters = 0;
        cmc_key.hi = cmc_key.mid = cmc_key.lo = cmc_count = 0;
        ProblemFree(thisprob);
        thisprob = ProblemAlloc();
        ProblemLoadState(thisprob, &contestwork, RC5_72, tslice, 0, 0, 0, 0, NULL);
        continue;
      }

      if (thisprob->pub_data.runtime_sec >= sec) {
        unsigned long perc = (iters / 120 * 100) / (maxkeys / 120);
        LogScreen("\rRC5-72: Stress-test 4 - %lu%%", perc);
        sec = thisprob->pub_data.runtime_sec + 1;
      }

      resultcode = ProblemRetrieveState(thisprob, &contestwork, NULL, 0, 0);

      iters += tslice;
      basekey += tslice;
      if (basekey > 0x609CDAE6) {
        cmc_key.hi  = KEYBASE_HI;
        cmc_key.mid = KEYBASE_MID;
        cmc_key.lo  = 0x609CDAE6;
        cmc_count = 5;
      }
      else if (basekey > 0x3EF57E4E) {
        cmc_key.hi  = KEYBASE_HI;       /* The test is designed so that      */
        cmc_key.mid = KEYBASE_MID;      /* partial match #3 is overwritten   */
        cmc_key.lo  = 0x3EF57E4E;       /* by partial match #4 since both of */
        cmc_count = 4;                  /* them are found in the same run    */
      }
      else if (basekey > 0x357C03CC) {
        cmc_key.hi  = KEYBASE_HI;
        cmc_key.mid = KEYBASE_MID;
        cmc_key.lo  = 0x357C03CC;
        cmc_count = 2;
      }
      else if (basekey > 0x30F0C71F) {
        cmc_key.hi  = KEYBASE_HI;
        cmc_key.mid = KEYBASE_MID;
        cmc_key.lo  = 0x30F0C71F;
        cmc_count = 1;
      }

      if (success != 0L) {
        int cpipe = cmc_key.lo % pipes + 1; /* The pipe that should find the match */

        if (resultcode == RESULT_FOUND) {
          success = -1L;    /* partial match expected */
          Log("\rRC5-72: Stress-test 4: Found a non-existing full match\n");
        }
        success |= __check_result(4, &contestwork, cpipe, cmc_count, iters, &cmc_key);
      }
    } while (iters < maxkeys && success > 0L);
    ProblemFree(thisprob);
  } // if (thisprob)

  if (success < 0) {
    Log("RC5-72: Stress-test 4 FAILED\n");
    Log("Possible errors :\n");
    Log("- Multiple partial match detection fails\n");
    /* Other errors should have been found by test 1 to 3 */
  }
  else if (success == 0) {
    success = -1L;
    Log("RC5-72: *** break ***\n");
  }
  else
    Log("\rRC5-72: Stress-test 4 passed\n");

  return success;
}


long StressRC5_72(void)
{
  long result = 1L;

  if (!IsProblemLoadPermitted(-1, RC5_72))
    return 0;

  if ((result = __test_1()) <= 0L)
    return result;

  if ((result = __test_2()) <= 0L)
    return result;

  if ((result = __test_3()) <= 0L)
    return result;

  return __test_4();
}
