#include "stubsplit.h"

static int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};

int stub_split(struct stub *stub, stub_callback callback, void *userdata)
{
  struct stub newstub = *stub;
  int used[1000];
  int i, j, k, t, tmax, limit;

  for (i = 0; i < 1000; i++) {
    used[0] = 0;
  }

  /*
   * First mark all the differences that occur in the current stub.
   * If we come across the same difference twice (for two different
   * combinations of adjacent differences), the stub is not a Golomb
   * stub so we return 0.
   */

  tmax = 0;
  for (i = 0; i < stub->length; i++) {
    t = stub->stub[i];
    j = i + 1;
    while (1) {
      if (used[t]) {
        return 0;
      }
      used[t] = 1;
      if (t > tmax) {
        tmax = t;
      }
      if (j >= stub->length) {
        break;
      }
      t += stub->stub[j];
      j++;
    }
  }

  /*
   * Next, mark all the differences for which a new mark would use
   * an existing difference.
   */

  i = tmax;
  while (i > 0) {
    if (used[i] == 1) {
      j = stub->length - 1;
      t = i;
      while (j >= 0) {
        t -= stub->stub[j];
        if (t <= 0) {
          break;
        }
        if (used[t] == 0) {
          used[t] = 2;
        }
        j--;
      }
    }
    i--;
  }

  /*
   * Finally, construct the list of new stubs from the unmarked differences.
   */

  t = 0;
  for (i = 0; i < stub->length; i++) {
    t += stub->stub[i];
  }
  newstub.length++;
  limit = OGR[stub->marks] - OGR[stub->marks-1 - stub->length];
  j = 0;
  for (i = 1; t+i <= limit; i++) {
    if (!used[i]) {
      j++;
      newstub.stub[newstub.length-1] = i;
      if (callback(userdata, &newstub) == 0) {
        break;
      }
    }
  }
  return j;
}
