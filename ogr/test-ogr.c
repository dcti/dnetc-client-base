#include <stdio.h>

#ifdef WIN32

#include <windows.h>

#else // WIN32

#if (macintosh)
#include <sys_time.h>
#else
#include <sys/time.h>
#endif
#include <unistd.h>

typedef unsigned long DWORD;

DWORD GetTickCount()
{
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return tv.tv_sec*1000 + tv.tv_usec/1000;
}

#endif // WIN32

#include "client2.h"
#include "stub.h"

extern CoreDispatchTable *ogr_get_dispatch_table();

#define NTESTS 1

int BestKnownRulers[][40] = {
  //{21,333,2,22,32,21,5,1,12,34,15,35,7,9,60,10,20,8,3,14,19,4},
  {22,356,1,8,5,29,27,36,16,2,4,31,20,25,19,30,10,7,21,39,11,12,3},
  {23,372,3,4,10,44,5,25,8,15,45,12,28,1,26,9,11,31,39,13,19,2,16,6},
  {24,425,9,24,4,1,59,25,7,11,2,10,39,14,3,44,26,8,40,6,21,15,16,19,22},
  {25,480,12,17,10,33,19,55,11,3,1,5,25,16,7,44,32,26,38,18,22,2,35,28,8,13},
  {26,492,5,12,11,8,16,10,44,30,13,25,4,56,7,2,46,3,15,22,39,14,6,21,50,32,1},
  //{26,492,1,32,50,21,6,14,39,22,15,3,46,2,7,56,4,25,13,30,44,10,16,8,11,12,5},
  {27,553,3,12,26,25,29,2,9,36,10,68,1,4,17,53,35,8,16,28,6,14,13,71,18,19,23,7},
  {28,585,3,12,26,25,29,2,9,36,10,68,1,4,17,53,35,8,16,28,6,14,13,71,18,19,23,7,32},
  {29,623,3,25,5,8,54,61,2,35,14,26,19,15,71,1,16,39,9,18,52,6,23,21,47,10,12,20,4,7},
  /*{30 680 9 26 22 71 21 2 29 25 13 63 11 44 28 6 8 16 75 4 1 40 19 47 15 3 33 10 7 20 12
  {31 747 9 3 42 25 64 2 16 31 13 21 85 38 14 19 4 53 20 7 8 28 58 1 10 30 68 6 26 24 5 17
  {32 784 7 8 11 2 29 55 6 18 40 1 4 30 3 44 51 9 23 48 14 53 20 36 16 54 46 43 25 49 17 10 12
  {33 859 12 80 4 55 3 20 1 51 60 14 35 8 11 99 6 40 2 29 10 26 30 7 27 88 5 13 15 17 44 9 16 22
  {34 938 4 15 16 42 22 26 23 14 74 13 33 8 76 21 17 27 28 11 36 34 45 94 6 18 43 25 78 20 9 1 2 50 7
  {35 987 2 22 12 74 9 8 33 93 30 27 4 21 14 45 1 70 19 49 29 43 26 6 73 15 13 64 23 44 55 7 3 37 11 5
  {36 1005 5 11 37 3 7 55 44 23 64 13 15 73 6 26 43 29 49 19 70 1 45 14 21 4 27 30 93 33 8 9 74 12 22 2 18
  {37 1099 8 1 4 57 27 21 19 85 15 23 7 34 52 29 17 134 31 6 10 12 32 11 25 58 18 2 33 39 3 66 51 56 26 24 49 14
  {38 1146 5 51 40 112 4 15 16 42 22 26 23 14 74 13 33 8 76 21 17 27 28 11 36 34 45 94 6 18 43 25 78 20 9 1 2 50 7
  {39 1252 7 9 88 20 25 24 10 28 67 14 19 23 99 17 40 63 77 29 6 2 46 18 12 68 47 3 41 11 82 71 39 21 1 4 27 43 15 36
  {40 1282 9 6 64 18 34 16 59 39 57 28 4 74 13 58 49 37 8 17 10 46 44 95 47 22 11 3 40 23 38 92 20 1 30 103 2 5 19 29 12*/
};

int ogr_result(void *state, void *result, int resultlen)
{
  int i, j;
  int *marks = (int *)result;

  printf("\n");
  for(i = 0; i < resultlen; i++) printf("%4d", marks[i]);
  printf("\n  ");
  for(i = 1; i < resultlen; i++) printf("%4d", marks[i]-marks[i-1]);
  printf("\n");

  /* exit(1); /* don't miss it */
}

int main(int argc, char *argv[])
{
  CoreDispatchTable *disp;
  int r, i, j;
  int done;
  struct {
    void *state;
    int nodeslimit;
    DWORD runtotal;
    double totalnodes;
  } Tests[NTESTS];

  disp = ogr_get_dispatch_table();
  disp->result = ogr_result;
  r = disp->init();
  if (r != CORE_S_OK) {
    printf("init failed: %d\n", r);
    return 1;
  }
  if (argc == 1) {
    for (i = 0; i < NTESTS; i++) {
      struct Stub stub;
      stub.marks = BestKnownRulers[i][0];
      stub.length = 5; //BestKnownRulers[i][0] / 5 + 1;
      for (j = 0; j < stub.length; j++) {
        stub.stub[j] = BestKnownRulers[i][2+j];
      }
      r = disp->create(&stub, sizeof(stub), &Tests[i].state);
      if (r != CORE_S_OK) {
        printf("create failed: %d\n", r);
        return 1;
      }
      Tests[i].totalnodes = 0;
      Tests[i].nodeslimit = 256;
    }
  } else {
    if (argc > 2) {
      struct Stub stub;
      stub.marks = atoi(argv[1]);
      stub.length = argc - 2;
      for (i = 0; i < stub.length; i++) {
        stub.stub[i] = atoi(argv[2+i]);
      }
      r = disp->create(&stub, sizeof(stub), &Tests[0].state);
      if (r != CORE_S_OK) {
        printf("create failed: %d\n", r);
        return 1;
      }
      Tests[0].totalnodes = 0;
      Tests[0].nodeslimit = 256;
    } else {
      printf("Usage: %s marks stub1 stub2 ...\n", argv[0]);
      return 1;
    }
  }
  done = 0;
  while (done < NTESTS) {
    for (i = 0; i < NTESTS; i++) {
      int nodes;
      DWORD start, end;

      if (Tests[i].state == NULL) continue;
      nodes = Tests[i].nodeslimit;
      start = GetTickCount();
      r = disp->cycle(Tests[i].state, &nodes);
      end = GetTickCount();
      if (r >= 0) {
        Tests[i].totalnodes += nodes;
      }
      if (end - start < 1000) {
        Tests[i].nodeslimit <<= 1;
      }
      printf("%d nodes=%d, nodes/sec=%d\n", BestKnownRulers[i][0], nodes, (nodes*1000)/(end-start+1));
      if (r != CORE_S_CONTINUE) {
        if (r != CORE_S_OK) {
          printf("cycle failed: %d\n", r);
          return 1;
        }
        printf("%d done, totalnodes=%g\n", BestKnownRulers[i][0], Tests[i].totalnodes);
        r = disp->destroy(Tests[i].state);
        if (r != CORE_S_OK) {
          printf("destroy failed: %d\n", r);
          return 1;
        }
        Tests[i].state = NULL;
        done++;
      }
    }
  }
  r = disp->cleanup();
  if (r != CORE_S_OK) {
    printf("cleanup failed: %d\n", r);
    return 1;
  }
  return 0;
}
