/*
 *        It is not yet ready for 64 bit operations, specifically
 *        in the choose algorithms (though we have fixes already
 *        set in another unlinked version) and in the bmBit[] array,
 *        and possibly other areas.
 *
 *        It is not yet ready to handle 3-stubs, etc.
 *        It is currently being tested on 5-stubs such as 3-13-7-17-1
 *        and we have not put the finishes needed for optimal 3-stub
 *        stuff.  (Again, a simple fix, but not yet implemented here)
 *
 *        To run a test just enter a
 *        send file with a 5-stub like the following:
 *               22 355  3 13 7 17 1    -3 -13 -7 -17 -1
 *
 *        Don't forget to use "-dsend 2 -dsave 1" to make the 5-stub work correctly
 *
 *   Best regards ... Mark
 *
 *   Note from Nate (sampo): try out the #define VISUALC on different compilers.
 *   on some it is faster, some slower.  Above test runs in 114 seconds on a G3/266
 */

/*-----INCLUDES----*/
#ifdef (macintosh)
#define stricmp strcmp
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
/* #include <assert.h> */
#include <signal.h>

#ifdef OS2
#define INCL_DOSPROCESS
#include <os2.h>
#endif

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef TIMINGTEST
#include <sys/timeb.h>
struct timeb start2,finish2;
#endif

/*-----TYPEDEFs-----*/
typedef unsigned long   U;

/*-----DEFINES-----*/
#define compress(B) (F3[(B)>>16] | F2[(B)&0xFFFF])
#define choose(x,y) (choose2[x][y])

#define MAX_CHOOSEBITS 16                    /* maximum bits in choose bitmap */
#define MAX_CHOOSEDIM (1<<MAX_CHOOSEBITS)    /* 1st dimension of choose */

#ifndef BITMAPS_64
#define BITMAP1  32
#define BITMAP2  64
#define BITMAP3  96
#define BITMAP4 128
#define BITMAP5 160
#define BITMAP6 192
#else
#define BITMAP1  64
#define BITMAP2 128
#define BITMAP3 192
#endif

#define OGR22             /* This sets the code to run fastest for OGR-22 */
#define MAXDEPTH  32      /* Leave this as is for OGR-22 */
//#define VISUALC

#ifdef WIN32
#define Win32_fastcall __fastcall
#define Win32_cdecl    __cdecl
#else
#define Win32_fastcall
#define Win32_cdecl
#endif

#define OR |              /* Use as test: "+" better give same answer! */
#define DD 48             /* used to cut # of bitmaps early */
/* ------------------------------------------------------------------
 *  DD 48 appeared to be optimal after testing different OGR-22 stubs
 * ------------------------------------------------------------------
 *  theory: DD quickens the pace of reducing # of bitmaps in use.
 *          This can cause extra work at end of ruler (DD units from
 *          right edge) but is often cancelled anyway by other
 *          marks (e.g., both COMP and DIST can cancel a possible
 *          spot, and DIST is still accurate for small diffs!)
 *
 *          For consistency, do not change this from 48 for OGR-22
 * ------------------------------------------------------------------
 */

/* The limits for OGR 22 */
#ifdef OGR22
#define HALFDEPTH  9
#define HALFDEPTH2 12
#define MAXDEPTHM1 21
#define MAXLENGTH  355
#define HALFLENGTH ((MAXLENGTH-8) >> 1)  /* see below */
#else
#define HALFDEPTH  half_depth
#define HALFDEPTH2 half_depth2
#define MAXDEPTHM1 maxdepthm1
#define MAXLENGTH  maxlength
#define HALFLENGTH half_length
#endif


/*-----GLOBALS-----*/

unsigned char choose[MAX_CHOOSEDIM][9]; /* ruler lengths */
unsigned char choose2[MAX_CHOOSEDIM][5]; /* ruler lengths */
U F3[1<<16];         /* The "compressed" bitmap for the 16 left bits */
U F2[1<<16];         /* The "compressed" bitmap for the 16 right bits */
U bmBit[256];        /* bits set mod 32 style */

U LD1[32];

unsigned char Sum[65536][17]; /* #open & Weighted sum of open bits */
unsigned char first[65536];   /* first blank in 16 bit bitmap, range: 1..16 */
int AVG[MAXDEPTH];            /* average length for n dists */
int OGR[]={0,1,3,6,11,17,25,34,44,55,72,85,106,127,151,177,199,216,246,283,333,356,372};

int gnodes = 0l;                /* integer node counter */
int cnodes = 0l;				/* choose nodes */

int ChooseDepth1 = 7;           /* when to start cacl'g chooose */
int ChooseDepth2 = 7;           /* when to stop  calc'g chooose */
int FillChooseTrigger = 25;     /* calc if cnt2 < AVG[] * Trigger */
int ChooseBits;                 /* bits in choose bitmap */
int FilledAtDepth = 99;         /* depth that choose was calc'd */
int ChooseRefill;
int dChooseBits[] = {16, 16, 16, 16, 16, 16, 15, 11, 6, 1, 1, 1};
int ChooseLimit = 165;          /* don't bother calc'g above this length */
int max;             /* maximum length of ruler */
U F1;                /* The required "extra" shift for the high 16 bits */
U bit32[256];        /* bits set only if index = 33 - 64 */
U bmStub[8];
U bitmask;           /* 32-bit mask using the input stub */

int ttmChooseBits;              /* 32-ChooseBits, precomputed in golombInit() */
int maxlength;                  /* maximum length of ruler */
int maxdepth;                   /* maximum number of marks in ruler */
int maxdepthm1;                 /* maxdepth-1, precomputed in golombInit() */

int half_length;                /* half of maxlength */
int half_length2;               /* maxlength-half_length-1 == 2nd halfmark */
int half_depth;                 /* half of maxdepth */
int half_depth2;                /* half of maxdepth, adjusted for 2nd mark */
int count[MAXDEPTH];            /* current length */

time_t  start0, start;          /* used to time search */
time_t  middle, finish;         /* used to time search */
volatile int DUMPTIME=60;       /* minimum time between saves, in seconds */
volatile int DUMPDEPTH;         /* save.txt updated at this level */
char *save_file_name = "save.txt";
char *send_file_name = "send.txt";
int FORCEDUMP;                  /* send.txt updated at this level */
int restart=999,n_start,n_end,etime,re_depth;
int start_stub[MAXDEPTH],end_stub[MAXDEPTH],re_stub[MAXDEPTH];
int dsave=1,dsend=2;            /* alter the save/send update depths */
int IDLE_state = 1;             /* turns off the 'idle' priority */
int OGR_TEST=0;                 /* Allows quick testing of small rulers */
int pri_class = 1;              /* # = Priority class (1=Idle, 2=Regular) */
int pri_delta = 1;              /* # = delta within class can be 0 thru 31 */
volatile int KILL_signal = 0;   /* Used to save output ASAP, then kill */
double GNodes = 0.0;			/* GARSP nodes */
double CNodes = 0.0;			/* CHOOSE nodes */
#ifdef WIN32
FILETIME ft1,ft2,ft3,ft4;
FILETIME ft5,ft6,ft7,ft8;
#endif



/*-----SPECIAL "WORKHORSE" DEFINES-----*/

/* Shift the comp arrays to the left */
#define ShiftComp1z              comp0 = (comp0 << s) OR (comp1 >> ss);
#define ShiftComp2z  ShiftComp1z comp1 = (comp1 << s) OR (comp2 >> ss);
#define ShiftComp3z  ShiftComp2z comp2 = (comp2 << s) OR (comp3 >> ss);
#define ShiftComp4z  ShiftComp3z comp3 = (comp3 << s) OR (comp4 >> ss);
#define ShiftComp5z  ShiftComp4z comp4 = (comp4 << s) OR (comp5 >> ss);
#define ShiftComp6z  ShiftComp5z comp5 = (comp5 << s) OR (comp6 >> ss);

#define ShiftComp1   ShiftComp1z comp1 <<= s;
#define ShiftComp2   ShiftComp2z comp2 <<= s;
#define ShiftComp3   ShiftComp3z comp3 <<= s;
#define ShiftComp4   ShiftComp4z comp4 <<= s;
#define ShiftComp5   ShiftComp5z comp5 <<= s;
#define ShiftComp6   ShiftComp6z comp6 <<= s;

/* Shift the list arrays to the right */
#define ShiftList0   list0 >>= s;
#define ShiftList1   list1 = (list1 >> s) OR (list0 << ss); ShiftList0
#define ShiftList2   list2 = (list2 >> s) OR (list1 << ss); ShiftList1
#define ShiftList3   list3 = (list3 >> s) OR (list2 << ss); ShiftList2
#define ShiftList4   list4 = (list4 >> s) OR (list3 << ss); ShiftList3
#define ShiftList5   list5 = (list5 >> s) OR (list4 << ss); ShiftList4
#define ShiftList6   list6 = (list6 >> s) OR (list5 << ss); ShiftList5

/* Shift the comp arrays to the left by 32 */
#define ShiftC32_1z              comp0 = comp1;
#define ShiftC32_2z  ShiftC32_1z comp1 = comp2;
#define ShiftC32_3z  ShiftC32_2z comp2 = comp3;
#define ShiftC32_4z  ShiftC32_3z comp3 = comp4;
#define ShiftC32_5z  ShiftC32_4z comp4 = comp5;
#define ShiftC32_6z  ShiftC32_5z comp5 = comp6;

#define ShiftC32_1   ShiftC32_1z comp1 = 0;
#define ShiftC32_2   ShiftC32_2z comp2 = 0;
#define ShiftC32_3   ShiftC32_3z comp3 = 0;
#define ShiftC32_4   ShiftC32_4z comp4 = 0;
#define ShiftC32_5   ShiftC32_5z comp5 = 0;
#define ShiftC32_6   ShiftC32_6z comp6 = 0;

/* Shift the list arrays to the right by 32 */
#define ShiftL32_0   list0 = 0;
#define ShiftL32_1   list1 = list0; ShiftL32_0
#define ShiftL32_2   list2 = list1; ShiftL32_1
#define ShiftL32_3   list3 = list2; ShiftL32_2
#define ShiftL32_4   list4 = list3; ShiftL32_3
#define ShiftL32_5   list5 = list4; ShiftL32_4
#define ShiftL32_6   list6 = list5; ShiftL32_5


/* Define the paramters for calling the next level */
#define Parms1   depth+1, cnt2, cnt2,                                 \
                 list0a, dist0 OR list0a, comp0 | (dist0 OR list0a)
#define Parms2   depth+1, cnt2, cnt2,                                 \
                 list0a, dist0 OR list0a, comp0 | (dist0 OR list0a),  \
                 list1a, dist1 OR list1a, comp1 | (dist1 OR list1a)
#define Parms3   depth+1, cnt2, cnt2,                                 \
                 list0a, dist0 OR list0a, comp0 | (dist0 OR list0a),  \
                 list1a, dist1 OR list1a, comp1 | (dist1 OR list1a),  \
                 list2a, dist2 OR list2a, comp2 | (dist2 OR list2a)
#define Parms4   depth+1, cnt2, cnt2,                                 \
                 list0a, dist0 OR list0a, comp0 | (dist0 OR list0a),  \
                 list1a, dist1 OR list1a, comp1 | (dist1 OR list1a),  \
                 list2a, dist2 OR list2a, comp2 | (dist2 OR list2a),  \
                 list3a, dist3 OR list3a, comp3 | (dist3 OR list3a)
#define Parms5   depth+1, cnt2, cnt2,                                 \
                 list0a, dist0 OR list0a, comp0 | (dist0 OR list0a),  \
                 list1a, dist1 OR list1a, comp1 | (dist1 OR list1a),  \
                 list2a, dist2 OR list2a, comp2 | (dist2 OR list2a),  \
                 list3a, dist3 OR list3a, comp3 | (dist3 OR list3a),  \
                 list4a, dist4 OR list4a, comp4 | (dist4 OR list4a)

/* Define the paramters being sent to the next level */
#define Parms1a  int depth, int cnt1, int cnt2,    \
                 U list0, U dist0, U comp0
#define Parms2a  int depth, int cnt1, int cnt2,    \
                 U list0, U dist0, U comp0,        \
                 U list1, U dist1, U comp1
#define Parms3a  int depth, int cnt1, int cnt2,    \
                 U list0, U dist0, U comp0,        \
                 U list1, U dist1, U comp1,        \
                 U list2, U dist2, U comp2
#define Parms4a  int depth, int cnt1, int cnt2,    \
                 U list0, U dist0, U comp0,        \
                 U list1, U dist1, U comp1,        \
                 U list2, U dist2, U comp2,        \
                 U list3, U dist3, U comp3
#define Parms5a  int depth, int cnt1, int cnt2,    \
                 U list0, U dist0, U comp0,        \
                 U list1, U dist1, U comp1,        \
                 U list2, U dist2, U comp2,        \
                 U list3, U dist3, U comp3,        \
                 U list4, U dist4, U comp4

/* Define routines that find next avail spot and shift bitmaps */
#ifndef BITMAPS_64
#define NEXTSPOT( SC32, SL32, SC, SL )                         \
   /* Find the next available mark location for this level */   \
   {                                                             \
      int s; /* count bits of next free distance */               \
stay:                                                             \
      if( comp0 < 0xffff0000 ) {                                  \
         s = first[comp0>>16];                                    \
      } else {                                                    \
         if( comp0 < 0xfffffffe ) {                               \
            /* s = 16 + first[comp0 & 0x0000ffff]; slow code */   \
            s = 16 + first[comp0 - 0xffff0000];                   \
         } else {                                                 \
            /* s>=32 */                                           \
            U comptemp=comp0;                                     \
            if( (cnt2 += BITMAP1) > limit ) return;               \
            { SC32; SL32; }                                       \
            if(comptemp==0xffffffff) goto stay; /* no free bit */ \
            goto skip_out; /* s==32 */                            \
         }                                                        \
      }                                                           \
      if( (cnt2 += s) > limit ) return; /* no spaces left */      \
                                                                  \
      { int ss=BITMAP1-s; SC; SL; } /* shift arrays */            \
   }                                                              \
skip_out:
#else
#define NEXTSPOT( SC32, SL32, SC, SL )                         \
   /* Find the next available mark location for this level */   \
   {                                                             \
      int s; /* count bits of next free distance */               \
stay:                                                             \
      if( comp0 < 0xffff000000000000 ) {                          \
         s =      first[comp0>>48];                               \
      } else if( comp0 < 0xffffffff00000000 ) {                   \
         s = 16 + first[comp0>>32 & 0x000000000000ffff];          \
      } else if( comp0 < 0xffffffffffff0000 ) {                   \
         s = 32 + first[comp0>>16 & 0x000000000000ffff];          \
      } else if( comp0 < 0xfffffffffffffffe ) {                   \
         s = 48 + first[comp0     & 0x000000000000ffff];          \
      } else { /* s>=64 */                                        \
         if( (cnt2 += 64) > limit ) return; /* no spaces left */  \
         {                                                        \
            U comptemp=comp0;                                     \
            { SC32; SL32; }                                       \
            if(comptemp==0xffffffffffffffff) goto stay;           \
            goto skip_out; /* s==64 */                            \
         }                                                        \
      }                                                           \
      if( (cnt2 += s) > limit ) return; /* no spaces left */      \
                                                                  \
      { int ss=BITMAP1-s; SC; SL; } /* shift arrays */            \
   }                                                              \
skip_out:
#endif


/* define routines that simplify copying list arrays into next level */
#define L0  list0a = list0;
#define L1  list1a = list1;
#define L2  list2a = list2;
#define L3  list3a = list3;
#define L4  list4a = list4;
#define L5  list5a = list5;
#define X0  list0a = list0 | bmBit[d];
#define X1  list1a = list1 | bmBit[d];
#define X2  list2a = list2 | bmBit[d];
#define X3  list3a = list3 | bmBit[d];
#define X4  list4a = list4 | bmBit[d];
#define X5  list5a = list5 | bmBit[d];

/* function prototypes */
void found_one(int cnt2);
void Time_to_save(int depth);
void Win32_fastcall Recursion0(Parms2a);
int Win32_fastcall Front(int depth, U dist0, U dist1, int cnt2);
U decompress(U bitmask, U bitmap);
int init_compress(void);
void fill_choose (int);

/*-----------------------------------------*/
/*  The golomb workhorse engine!           */
/*-----------------------------------------*/



void Win32_fastcall Recursion2( Parms2a )
{
   int limit;
   LD1[depth]++;
   count[depth-1]=cnt2;
   if (depth > HALFDEPTH2) limit = MAXLENGTH-choose(compress(dist0),MAXDEPTHM1-4-depth);
   else limit = Front(depth, dist0, dist1, cnt2);
   gnodes++;
   while(1) {
      NEXTSPOT( ShiftC32_1, ShiftL32_1, ShiftComp1, ShiftList1 )
      {
         U list0a,list1a;
         int d = cnt2-cnt1;
              if( d <= BITMAP1 ) { X0 L1 }
         else if( d <= BITMAP2 ) { L0 X1 }
         else                    { L0 L1 }

         if (depth < MAXDEPTHM1-4) Recursion2( Parms2 );
         else Recursion0( Parms2 );
      }
#ifndef OGR22
      if ( depth <= DUMPDEPTH ) Time_to_save(depth);
#endif
   }
}

void Win32_fastcall Recursion3( Parms3a )
{
   int limit;
   LD1[depth]++;
   count[depth-1]=cnt2;
   if (depth > HALFDEPTH2) limit = MAXLENGTH-choose(compress(dist0),MAXDEPTHM1-4-depth);
   else limit = Front(depth, dist0, dist1, cnt2);
   gnodes++;

   while(1) {
      NEXTSPOT( ShiftC32_2, ShiftL32_2, ShiftComp2, ShiftList2 )
      {
         U list0a,list1a,list2a;
         int d = cnt2-cnt1;

         if (depth < MAXDEPTHM1-4) {
            if (MAXLENGTH-cnt2>BITMAP2+DD) {
#ifdef VISUALC
                     if( d <= BITMAP1 ) { X0 L1 L2 }
                else if( d <= BITMAP2 ) { L0 X1 L2 }
                else if( d <= BITMAP3 ) { L0 L1 X2 }
                else                    { L0 L1 L2 }
#else
                                        { X0 L1 L2 }
#endif
                Recursion3( Parms3 );
            } else {
                     if( d <= BITMAP1 ) { X0 L1 }
                else if( d <= BITMAP2 ) { L0 X1 }
                else                    { L0 L1 }
                Recursion2( Parms2 );
            }
         }
         else {
                  if( d <= BITMAP1 ) { X0 L1 }
             else if( d <= BITMAP2 ) { L0 X1 }
             else                    { L0 L1 }
             Recursion0( Parms2 );
         }
      }
#ifndef OGR22
      if ( depth <= DUMPDEPTH ) Time_to_save(depth);
#endif
   }
}


/* division by 3 for integer x <= 49151 */
#define DIV3(x) ((x)*0x0000AAAB >> 17)

int Win32_fastcall Front(int depth, U dist0, U dist1, int cnt2)
{
    int limit;
    if (depth > HALFDEPTH) {
        /*      M I D D L E   R U L E R   T E S T S
         * Define the tests that are done in the middle of the ruler
         */
        limit = MAXLENGTH-count[HALFDEPTH]-1;
        if (depth < HALFDEPTH2) limit -= Sum[dist0>>16][HALFDEPTH2-depth];
        if (limit > MAXLENGTH-OGR[MAXDEPTHM1-depth]) limit = MAXLENGTH-OGR[MAXDEPTHM1-depth];
    } else {
        /*      F I R S T   H A L F   T E S T S
         * Restart code placed here to keep speed up
         * depth = 1 for first mark after the left edge
         * restart = 999 means this is a new run, =1+ means otherwise
         * We evaluate the second+ time we enter this routine
         */
        if( depth-1 == restart ) {
            if (count[restart]-count[restart-1] < re_stub[restart]) { gnodes--; return 0; }
            if (count[restart]-count[restart-1] > re_stub[restart]) {
                gnodes-=restart;  /* correct the node count for restarts */
                restart=999;  /* started or restarted */
            } else {
                if (restart == re_depth) { gnodes--; return 0; }
                restart++;
            }
        }
        {
            int d = ((HALFDEPTH2*2+HALFDEPTH)-depth*3) >> 1;
            limit = (MAXLENGTH-1-Sum[dist0 >> 16][d]) >> 1;
            if (limit > MAXLENGTH-OGR[MAXDEPTHM1-depth])
                limit = MAXLENGTH-OGR[MAXDEPTHM1-depth];
        }
        while (depth <= ChooseDepth2) {
            if (depth > FilledAtDepth) break;  /* Choose already filled! */
            if (depth < ChooseDepth1) { FilledAtDepth = 99; ChooseRefill=1; break; }
            if (depth < ChooseDepth2) {
                if (cnt2 - count[depth-2] < FillChooseTrigger) { ChooseRefill=1; break; }
                if ((depth < FilledAtDepth) || ChooseRefill) {
                    ChooseBits = dChooseBits[depth-1];
                    ttmChooseBits = BITMAP1 - ChooseBits;
                    fill_choose(depth-1);
                    FilledAtDepth = depth;
                    ChooseRefill = 0;
                }
            } else {
                if (cnt2 - count[depth-2] < FillChooseTrigger) {
                    ChooseBits = dChooseBits[depth];
                    ttmChooseBits = BITMAP1 - ChooseBits;
                    fill_choose(depth);
                    FilledAtDepth = 99;
                    ChooseRefill = 1;
                } else if (ChooseRefill) {
                    ChooseBits=dChooseBits[depth-1];
                    ttmChooseBits = BITMAP1 - ChooseBits;
                    fill_choose(depth-1);
                    FilledAtDepth = depth;
                    ChooseRefill = 0;
                }
            }
            break;
        }
        if (gnodes >= 0x40000000) {
			GNodes += (double) gnodes;
			gnodes = 0;
        }
    }

    if (cnt2 <= AVG[depth])
    {
         int l;
         unsigned char *pSum1 = &Sum[dist0 >> 16][0];
         int adist1 = pSum1[0];
         unsigned char *pSum2 = &Sum[dist0 & 0xffff][0];
         int adist2 = pSum2[0];

         /* simple minsum pattern */
         int ndist = MAXDEPTHM1-depth;
         if (ndist <= adist1) l = pSum1[ndist];
         else {
            ndist -= adist1;
            l = pSum1[adist1]+ndist*16 + pSum2[ndist];
         }
         if (limit > MAXLENGTH-l) limit = MAXLENGTH-l;

         /* double minsum pattern */
         ndist = 2*(MAXDEPTHM1-depth)-1;
         if (ndist <= adist1) l = pSum1[2] + pSum1[ndist];
         else { /* Note: ndist always >= 17 for OGR-22 */
            ndist -= adist1;
            l = pSum1[2] + pSum1[adist1]+ndist*16;
            if (ndist <= adist2) l += pSum2[ndist];
            else {
                ndist -= adist2;
                l += pSum2[adist2]+ndist*16 + Sum[dist1 >> 16][ndist];
            }
         }
         l = MAXLENGTH-DIV3(l+2);
         if (limit > l) limit = l;
    }
    return limit;
}


#ifndef BITMAPS_64
/* The following 3 routines are not needed for 64 bit version */


void Win32_fastcall Recursion4(Parms4a)
{
   int limit;
   LD1[depth]++;
   count[depth-1]=cnt2;
   if (depth > HALFDEPTH2) limit = MAXLENGTH-choose(compress(dist0),MAXDEPTHM1-4-depth);
   else limit = Front(depth, dist0, dist1, cnt2);
   gnodes++;

   while(1) {
      NEXTSPOT( ShiftC32_3, ShiftL32_3, ShiftComp3, ShiftList3 )
      {
         U list0a,list1a,list2a,list3a;
         int d = cnt2-cnt1;

#ifdef VISUALC
              if( d <= BITMAP1 ) { X0 L1 L2 L3 }
         else if( d <= BITMAP2 ) { L0 X1 L2 L3 }
         else if( d <= BITMAP3 ) { L0 L1 X2 L3 }
         else if( d <= BITMAP4 ) { L0 L1 L2 X3 }
         else                    { L0 L1 L2 L3 }
#else
              if( d <= BITMAP1 ) { X0 L1 L2 L3 }
         else if( d <= BITMAP2 ) { L0 X1 L2    }
         else                    { L0 L1       }
#endif
         if (depth < MAXDEPTHM1-4) {
            if      (cnt2 < MAXLENGTH-BITMAP3-DD) Recursion4(Parms4);
            else if (cnt2 < MAXLENGTH-BITMAP2-DD) Recursion3(Parms3);
            else                                  Recursion2(Parms2);
         } else Recursion0(Parms2);
      }
#ifndef OGR22
      if ( depth <= DUMPDEPTH ) Time_to_save(depth);
#endif
   }
}

void Win32_fastcall Recursion5(Parms5a)
{
   int limit;
   LD1[depth]++;
   count[depth-1]=cnt2;
   if (depth > HALFDEPTH2) limit = MAXLENGTH-choose(compress(dist0),MAXDEPTHM1-4-depth);
   else limit = Front(depth, dist0, dist1, cnt2);
   gnodes++;

   while(1) {
      NEXTSPOT( ShiftC32_4, ShiftL32_4, ShiftComp4, ShiftList4 )
      {
         U list0a,list1a,list2a,list3a,list4a;
         int d = cnt2-cnt1;
              if( d <= BITMAP1 ) { X0 L1 L2 L3 L4 }
         else if( d <= BITMAP2 ) { L0 X1 L2 L3 L4 }
         else if( d <= BITMAP3 ) { L0 L1 X2 L3 L4 }
         else if( d <= BITMAP4 ) { L0 L1 L2 X3 L4 }
         else if( d <= BITMAP5 ) { L0 L1 L2 L3 X4 }
         else                    { L0 L1 L2 L3 L4 }

         if (depth < MAXDEPTHM1-4) {
            if (cnt2 < MAXLENGTH-BITMAP4-DD)      Recursion5(Parms5);
            else if (cnt2 < MAXLENGTH-BITMAP3-DD) Recursion4(Parms4);
            else if (cnt2 < MAXLENGTH-BITMAP2-DD) Recursion3(Parms3);
            else                                  Recursion2(Parms2);
         }
         else Recursion0(Parms2);
      }
      if ( depth <= DUMPDEPTH ) Time_to_save(depth);
   }
}

#endif   /* for #ifndef BITMAPS_64 for the past 2 routines */


void Win32_fastcall Recursion0( Parms2a )
{
   int limit;             /* limit for this mark */
   LD1[depth]++;
   count[depth-1]=cnt2;

   if (MAXDEPTHM1-depth > 0) limit = MAXLENGTH-Sum[dist0>>16][MAXDEPTHM1-depth];
   else limit = MAXLENGTH;

   gnodes++;
   while(1) {
      NEXTSPOT( ShiftC32_1, ShiftL32_1, ShiftComp1, ShiftList1 )
      if (depth < MAXDEPTHM1) {
         U list0a,list1a;
         int d = cnt2-cnt1;
              if( d <= BITMAP1 ) { X0 L1 }
         else if( d <= BITMAP2 ) { L0 X1 }
         else                    { L0 L1 }

         Recursion0( Parms2 );
      } else found_one(cnt2);
#ifndef OGR22
      if ( depth <= DUMPDEPTH ) Time_to_save(depth);
#endif
   }
}



U decompress(U bitmask, U bitmap)
{
   long unsigned i,j;
   unsigned long map=0l;
   for( i=j=(1<<31); j>0; j>>=1 ){
      if( ! (bitmask & j) ) {      /* if not in the original bitmask... */
         if( bitmap & i ) map |= j;
         i>>=1;
      }
   }
   if( map & bitmask ) {printf("decompress error");exit(1);}
   return (map | bitmask);
}


int init_compress(void)
{
   U leftmask,rightmask;  /* 16-bit masks using the input stub */
   U i,j1,j2,k;
   U bit[33];
   int s;

// Set and count_+16 the taken bits
   F1 = k = 0;
   for( i=1; i <= 32; i++ ) { bit[i]=1&(bitmask>>(32-i)); if(bit[i])k++; }
   for( i=1; i <= 16; i++ ) if(bit[i]) F1++;
   leftmask = bitmask >> 16;
   rightmask = bitmask & 0xffff;


   printf("Building choose using %d (+%d preset<32 =%d) bits, ChooseLimit=%d ...",
           ChooseBits,(int)k,ChooseBits+(int)k,ChooseLimit);

   if(ChooseBits+k>32) {printf("\n\nYou must lower ChooseBits!");exit(1);}


// Build the "compression" indexing arrays
   for( i=0; i < (1<<16); i++ ) {             /* i = 16-bit map */
   if( (i & leftmask) == leftmask ) {
     k = 0;                               /* k = new 16-bit map */
     j2 = 1;                              /* j2 = bit in k */
     for( j1=1; j1<=16; j1++ ) {          /* j1 = bit in i */
        if( !bit[  j1 ] ) {               /* skip the LEFT presets */
           if( i & 1<<(16-j1) ) k = k | (1<<(16-j2));
           j2++;
        }
     }
     F3[i] = k;
   }
   if( (i & rightmask) == rightmask ) {
     k = 0;
     j2 = 1;
     for( j1=1; j1<=16; j1++ ) {
        if( !bit[j1+16] ) {               /* skip the RIGHT presets */
           if( i & 1<<(16-j1) ) k = k | (1<<(16-j2));
           j2++;
        }
     }
     F2[i] = k;
   }
//   #define compress(B) ( (((U) F3[(B)>>16]) << (16-ttmChooseBits)) | (F2[(B)&0xFFFF] << (F1-ttmChooseBits)) )
     s = 16-ttmChooseBits;
     if( s > 0 ) F3[i] <<= s; else F3[i] >>= (-s);
     s = F1-ttmChooseBits;
     if( s > 0 ) F2[i] <<= s; else F2[i] >>= (-s);
   }
   return 0;
}



/*-----------------------------------------*/
/*  Get command line arguments             */
/*-----------------------------------------*/
void parse_arguments (int argc, char **argv)
{
  int i = 1;
  while (i < argc) {
    char *arg = argv[i];
    /* printf("arg = %s",argv[i]); */
    if (stricmp(arg,"-save") == 0) {
      save_file_name = argv[i+1];
      i += 2;
    } else if (stricmp (arg, "-send") == 0) {
      send_file_name = argv[i+1];
      i += 2;
    } else if (stricmp (arg, "-class") == 0) {
      pri_class = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-delta") == 0) {
      pri_delta = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-dsave") == 0) {
      dsave = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-dsend") == 0) {  /* do not advertise this */
      dsend = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-ogr") == 0) {    /* do not advertise this */
      maxdepth = atoi(argv[i+1]);
      maxlength = OGR[maxdepth-1];
      OGR_TEST=1;
      i += 2;
    } else if (stricmp (arg, "-dumptime") == 0) {
      DUMPTIME = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-cl") == 0) {
      ChooseLimit = atoi(argv[i+1]);
      if ((ChooseLimit > 255) || (ChooseLimit <= 0)) {
          fprintf(stderr,"Warning: Choose Limit restored to default 165.\n");
          ChooseLimit = 165;
      }
      i += 2;
    } else if (stricmp (arg, "-ct") == 0) {
      FillChooseTrigger = atoi(argv[i+1]);
      i += 2;
    } else if (stricmp (arg, "-cd1") == 0) {
      ChooseDepth1 = atoi(argv[i+1]);
      if ((ChooseDepth1 > HALFDEPTH) || (ChooseDepth1 <= 0)) {
          fprintf(stderr,"Warning: Choose depth1 restored to default 7!\n");
          ChooseDepth1 = 7;
      }
      i += 2;
    } else if (stricmp (arg, "-cd2") == 0) {
      ChooseDepth2 = atoi(argv[i+1]);
      if ((ChooseDepth2 > HALFDEPTH) || (ChooseDepth2 <= 0)) {
          fprintf(stderr,"Warning: Choose depth2 restored to default 7!\n");
          ChooseDepth2 = 7;
      }
      i += 2;
    } else if (stricmp (arg, "-cb") == 0) {
      int depth = atoi(argv[i+1]);
      if ((depth > 0) && (depth <= HALFDEPTH)) {
          dChooseBits[depth] = atoi(argv[i+2]);
          if (dChooseBits[depth] <= 0) {
              fprintf(stderr,"Warning: # bits in choose for depth %d adjusted to 1!\n",depth);
              dChooseBits[depth] = 1;
          }
          if (dChooseBits[depth] >= MAX_CHOOSEBITS) {
              fprintf(stderr,"Warning: # bits in choose for depth %d adjusted to %d!\n",depth,MAX_CHOOSEBITS);
              dChooseBits[depth] = MAX_CHOOSEBITS;
          }
      } else fprintf(stderr,"Warning: -cbits %s %s ignored!\n",argv[i+1],argv[i+2]);
      i += 3;
    } else {
      if ((stricmp (arg, "-?") != 0) && (stricmp (arg, "-h") != 0) && (stricmp (arg, "-help") != 0))
          fprintf(stderr,"Illegal command line option:  %s\n", arg);
      fprintf(stderr," options:  -save $        $  =   save.txt filename\n");
      fprintf(stderr,"           -send $        $  =   send.txt filename\n");
      fprintf(stderr,"           -dumptime #    #  = minimum seconds between saves\n");
      fprintf(stderr,"           -dsave #       #  = extra save depth (use 1 for really hard stubs)\n");
      fprintf(stderr,"           -class #       #  = Priority class (1=Idle, 2=Regular)\n");
      fprintf(stderr,"           -delta #       #  = delta within class ( can be 0 upto 31)\n");
      fprintf(stderr,"           -cl #          #  = choose's length limit (default=165)\n");
      fprintf(stderr,"           -ct #          #  = choose's trigger (default=25)\n");
      fprintf(stderr,"           -cd1 #         #  = choose depth1 (default=7)\n");
      fprintf(stderr,"           -cd2 #         #  = choose depth2 (default=7)\n");
      fprintf(stderr,"           -cb #1 #2      #2 = bits in choose for depth #1 \n");
      exit(1);
    }
  }
}


/*-----------------------------------------*/
/*  found_one() - print out golomb rulers  */
/*-----------------------------------------*/
void found_one(int cnt2)
{
   FILE *fout;
   int i,j;

   count[maxdepth-1] = cnt2;       /* not placed yet into list arrays! */

   /* confirm ruler is golomb */
   {
      int diff;
      char diffs[256];
      for( i=65; i <= MAXLENGTH/2; i++ ) diffs[i]=0;
      for( i=1; i < maxdepth; i++ ) {
         for( j=0; j<i; j++ ) {
            diff = count[i]-count[j];
            if( diff+diff <= MAXLENGTH ) {        /* Principle 1 */
               if( diff <= 64 ) break;      /* 2 bitmaps always tracked */
               if( diffs[diff] ) return;
               diffs[diff] = 1;
            }
         }
      }
   }

   fout = fopen("newruler.txt","a");
   fprintf(fout,"found: length = %d\n", count[maxdepth-1]);
   for( i=0; i<maxdepth; i++ ) fprintf(fout,"%4d",count[i]);
   fprintf(fout,"\n  ");
   for( i=1; i<maxdepth; i++ ) fprintf(fout,"%4d",count[i]-count[i-1]);
   fprintf(fout,"\n");
   fclose(fout);

   fout = fopen(send_file_name,"a");
   fprintf(fout,"0 found: length = %d\n", count[maxdepth-1]);
   for( i=0; i<maxdepth; i++ ) fprintf(fout,"%4d",count[i]);
   fprintf(fout,"\n  ");
   for( i=1; i<maxdepth; i++ ) fprintf(fout,"%4d",count[i]-count[i-1]);
   fprintf(fout,"\n");
   fclose(fout);

   printf("\n");
   for( i=0; i<maxdepth; i++ ) printf("%4d",count[i]);
   printf("\n  ");
   for( i=1; i<maxdepth; i++ ) printf("%4d",count[i]-count[i-1]);
   printf("\n");
}

/*-----------------------------------------*/
/*  get_input() get the input to use       */
/*-----------------------------------------*/
void get_input(void)
{
   FILE *fout;
   int i,j;

   restart = 1;

   /* File Read */
   fout = fopen(save_file_name,"r");
   if( !fout ) {printf("Cannot open %s",save_file_name);exit(1);}

   fscanf(fout,"%d %d",&maxdepth,&maxlength);
   /* check maxdepth */
   if ((maxdepth <= 1) || (maxdepth > 22)) {
       printf("Number of marks must be between 2 and 22.\n");
       exit(1);
   }
   if( maxlength <= 0 ) maxlength = OGR[maxdepth-1];

   /* The remainder of the input file is optional */

   re_stub[1] = start_stub[1] = 1;
   /* Starting Stub */
   for(i=1; ;i++) {
      j = fscanf(fout,"%d",&start_stub[i]);
      if( j == EOF || j == 0 ) {
         n_start = re_depth = i-1;
         fclose(fout);
         fout = fopen(save_file_name,"a");
         if( n_start == 0 ) fprintf(fout," 1");
         fprintf(fout," -999\n\n");
         fclose(fout);
         re_stub[re_depth]--;
         return;
      }
      if( (re_stub[i]=start_stub[i]) < 0 ) {
         re_depth = n_start = i-1;
         re_stub[re_depth]--;  /* needed because we will start AFTER this */
         break;
      }
   }

   /* Ending Stub */
   end_stub[1] = -start_stub[i];
   for(i=2; ;i++) {
      j = fscanf(fout,"%d",&end_stub[i]);
      if( j == EOF || j == 0 ) {
         n_end=i-1;
         fclose(fout);
         fout = fopen(save_file_name,"a");
         if( n_end == 0 ) fprintf(fout," -999");
         fprintf(fout,"\n\n");
         fclose(fout);
         return;
      }
      if( end_stub[i] > 0 ) {n_end=i-1;break;}
      end_stub[i] = -end_stub[i];
   }

   /* Restart Stub -- overwrite the restart stub if work already completed */
   re_depth = end_stub[i];
   while(1) {
      if(re_depth==999){printf("\nThis run is already completed!\n");exit(0);}
      for( i=1; i<=re_depth; i++ ) {
         if( fscanf(fout,"%d",&re_stub[i]) != 1) {
            printf("\n There is an error in the input file! \n"); exit(2);
         }
      }
      if( fscanf(fout,"%d %lf %lf",&etime,&GNodes,&CNodes) != 3 ) {
         printf("\n There is an error in the input file! \n"); exit(2);
      }
      if( fscanf(fout,"%d",&i ) != 1 ) break;
      re_depth = i;
   }
   fclose(fout);

   /* Screen Dump if we are restarting */
   printf("\n\n-----------------------------------------");
   printf("\n          GARSP is resuming after...\n");
   printf(" Searched through");
   for( i=1; i<=re_depth; i++ ) printf(" %d",re_stub[i]);
   printf(" ...in %d seconds (%.0f nodes)\n",etime,GNodes);
   printf("-----------------------------------------\n\n");

}



/*-----------------------------------------*/
/*  store_temp() store state for future use*/
/*-----------------------------------------*/
void store_temp(int depth)
{
   FILE *fout;
   int i;

   /* Screen Dump */
   printf("Searched thru");
   for( i=1; i<=depth; i++ ) printf(" %d",count[i]-count[i-1]);
   GNodes += (double)gnodes; gnodes = 0;
   printf(" ... %d secs, %.0f nodes\n",etime+(int)difftime(finish,start),GNodes);
   fflush(stdout);
   middle = finish;

   /* File Storage */
   fout = fopen(save_file_name,"a");

   fprintf(fout,"%d   ",depth);
   for( i=1; i<=depth; i++ ) fprintf(fout," %d",count[i]-count[i-1]);
   fprintf(fout,"   %d %.0f\n",etime+(int)difftime(finish,start),GNodes);
   fclose(fout);

   if( KILL_signal ) {
       exit(1);  /* exit with error, so batch files work OK */
   }
}


/*-----------------------------------------*/
/*  store() - store state for future use   */
/*-----------------------------------------*/
void store(void)
{
    FILE *fout;
   int i;

   /* Screen Dump */
   printf("Searched thru");
   for( i=1; i<=FORCEDUMP; i++ ) printf(" %d",count[i]-count[i-1]);
   GNodes += (double)gnodes; gnodes = 0;
   printf(" ... %d secs, %.0f nodes",etime+(int)difftime(finish,start),GNodes);
   printf("  *** saved in %s\n",send_file_name);
   fflush(stdout);

   middle = finish;

   /* Permanent File Storage */
   fout = fopen(send_file_name,"a");
   fprintf(fout,"%d %d",maxdepth,MAXLENGTH);
   for( i=1; i<=FORCEDUMP; i++ ) fprintf(fout," %d",count[i]-count[i-1]);
   fprintf(fout,"  %d %.0f %.0f\n",etime+(int)difftime(finish,start),GNodes,CNodes);
   fclose(fout);

   /* Reset Temp File Storage */
   fout = fopen(save_file_name,"w");
   fprintf(fout,"%d %d  ",maxdepth,MAXLENGTH);
   for( i=1;i<=n_start;i++) fprintf(fout,"%d ",start_stub[i]);
   if( ! n_start ) fprintf(fout," 1");
   for( i=1;i<=n_end;i++) fprintf(fout," %d",-end_stub[i]);
   if( ! n_end ) fprintf(fout," -999");

   fprintf(fout,"\n\n%d   ",FORCEDUMP);
   for( i=1; i<=FORCEDUMP; i++ ) fprintf(fout," %d",count[i]-count[i-1]);
   fprintf(fout,"   0 0 0\n");   /* time and node counts reset to zero */
   fclose(fout);

   { GNodes = CNodes = 0.0; etime=0; start=finish; }

}



/*-----------------------------------------*/
/*  Is it time to make saves? ...          */
/*-----------------------------------------*/
void Time_to_save(int depth)
{
   if( depth  == FORCEDUMP ) {
      if( restart == 999 ) store();
   } else {
      time(&finish);
      if( difftime(finish,middle) >= DUMPTIME )
         store_temp(depth);
   }
   if( depth <= n_end ) {
      int i;
      for( i=1; i <= n_end; i++) {
         if(count[i]-count[i-1] > end_stub[i]) break;
         if(count[i]-count[i-1] < end_stub[i]) return;
      }
      printf("\n The assigned Stub Range is finished! ");
      time (&finish);
      printf("\n Session lasted %.0f seconds\n\n", difftime(finish,start0));
#ifdef WIN32
    GetProcessTimes(GetCurrentProcess(),&ft5,&ft6,&ft7,&ft8);
    printf("Process Time: %.1lf seconds\n",
        ((double)ft8.dwLowDateTime+(double)ft8.dwHighDateTime*4294967296.0-
        (double)ft4.dwLowDateTime+(double)ft4.dwHighDateTime*4294967296.0)*100e-9);
#endif
      for(i=7;i<22;i++) printf("%3d %10lu\n",i,LD1[i]);
      {
         FILE *fout;
         fout = fopen(save_file_name,"a");
         fprintf(fout,"999 This run is finished\n");
         fclose(fout);
         fout = fopen(send_file_name,"a");
         fprintf(fout,"999 This run is finished\n");
         fclose(fout);
      }
      exit(0);
   }
}

void MinSumInit()
{
    /*
     * Sum[dist][0]    holds the number of zero bits in dist
     * Sum[dist][1-16] hold sums of the first 1-16 zero bits
     *
     * AVG[] length for n dists used to avoid unnecessary
     * minimum sum limit calculations.
     */
    unsigned int i,j,dist;
    for (dist = 0; dist < 65536; dist++) {
        j = Sum[dist][0] = 0;
        for (i=1; i<=16; i++) {
            if ((dist & 1<<(16-i) ) == 0) {
                j++;
                Sum[dist][j] = Sum[dist][j-1] + i;
            }
        }
        Sum[dist][0] = (unsigned char) j;
        for( j++; j <= 16; j++ ) Sum[dist][j] = Sum[dist][j-1] + 16;
    }
    Sum[65535][16]=255;   /* corrects for 16*16 = 256 == 0 as unsigned char */
    for (i = 0; i < maxdepth; i++)
        AVG[i] = (i*MAXLENGTH*2+(maxdepth-1))/(maxdepth*2-2);
}


/*--------------------------------------------*/
/*      The Garsp Init Routine                */
/*--------------------------------------------*/
void golombInit()
{
   U comp0;        /* comparison bitmap */
   U list0;        /* ruler bitmap */
   U dist0;        /* distance bitmap */
   int depth;      /* the depth of recursion */

   depth    = 1;
   list0  = 0;
   dist0  = 0;
   comp0  = 0;
   count[1] = count[0] = 0;

   /* precomputed here, used in Recursion() */
   maxdepthm1=maxdepth-1;

#ifdef BITMAPS_64
   Recursion3(
      depth,0,0,
      list0,dist0,comp0,
      list0,dist0,comp0,
      list0,dist0,comp0);
#else
   /* starting with Recursion5 (instead of Recursion6) appeared */
   /* to be up to 7% faster (same principle as DD parameter)    */
   Recursion5(
      depth,0,0,
      list0,dist0,comp0,
      list0,dist0,comp0,
      list0,dist0,comp0,
      list0,dist0,comp0,
      list0,dist0,comp0);
#endif
}



/*---------------------------------------*/
/*      The Signal Catch Routine         */
/*---------------------------------------*/
void Win32_cdecl MyHandler( int sig_number )
{
   DUMPDEPTH = half_depth-1;
   DUMPTIME  = 0;
   KILL_signal = 1;
}



/*---------------------------------------*/
/*      The Main Routine                 */
/*---------------------------------------*/
int Win32_cdecl main(argc,argv)
   int argc;
   char *argv[];
{
   int i,j,k,m;                     /* counters */
   FILE *fout;

   if( 8*(int)sizeof(U) != BITMAP1 ) {
      printf("GARSP only works with %d bit ints\n\n",BITMAP1);
      printf("sizeof(char)=%d sizeof(short)=%d sizeof(int)=%d sizeof(long)=%d\n\n",
         (int)sizeof(char),(int)sizeof(short),(int)sizeof(int),(int)sizeof(long));
      exit(1);
   }


   /*--- init data ---*/
   n_end = n_start = etime = 0;

   parse_arguments (argc, argv);


   if (!OGR_TEST) get_input();


  /*
   *   Bump priority into idle class -- let others get CPU first
   */
   if (pri_class < 1 || pri_class > 2) {
     fprintf(stderr,"class must be either 1 or 2\n");
     fprintf(stderr,"A value of 1 (Idle) is used\n");
     pri_class = 1;
   }
   if (pri_delta < 0 || pri_delta > 31) {
     fprintf(stderr,"delta must be a value between 0 & 31\n");
     fprintf(stderr,"A value of 0 (lowest) is used\n");
     pri_delta = 0;
   }
#ifdef WIN32
   SetPriorityClass(GetCurrentProcess(), (pri_class == 1)  ? IDLE_PRIORITY_CLASS : NORMAL_PRIORITY_CLASS);
   SetThreadPriority(GetCurrentThread(), pri_delta /*THREAD_PRIORITY_IDLE*/);
#endif
#ifdef OS2
   DosSetPriority(PRTYS_THREAD, pri_class /*PRTYC_IDLETIME*/,pri_delta/*PRTYD_MINIMUM*/, 0);
#endif


   printf("\nProgram GARSP, Version 6.00 using %d bits/map\n", ChooseBits);

#ifdef OGR22
   if(maxdepth != 22)
   {
      fprintf(stderr,"This compilation is specifically optimized for OGR-22\n You may not use it for length %d!\n",maxdepth);
      exit(10);
   }
#endif


   FORCEDUMP = (maxdepth >> 2)-2 +dsend; /* SEND.TXT file contents norm=-2 */
   DUMPDEPTH = (maxdepth >> 2)   +dsave; /* SAVE.TXT file contents norm=+0 */

   if( FORCEDUMP < 0 ) FORCEDUMP = 0;
   printf("\n saving at depth = %2d (%s)\n",FORCEDUMP,send_file_name);
   printf("  and at depths <= %2d (%s)",DUMPDEPTH,save_file_name);
   printf(" if elapsed time >= %d seconds\n\n",DUMPTIME);
   fflush(stdout);


   /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */

   /* Simulate GVANT's "KTEST=1" - if maxdepth even, use 2 marks */
   half_depth = ((maxdepth-1) >> 1)-1;
   half_depth2 = (maxdepth & 1) ? half_depth+2 : half_depth+3;

   /*------------------
   Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
   We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
   But:    half_length2 + half_length <= maxlength-1    (our midpoint reduction)
   So:     half_length + 3 (6 if maxdepth even) + half_length <= maxlength-1

   But:    hl | 1 2 | hl+1 and  hl | 2 1 | hl+1 ,
           or hl | 2 3 1 | hl+1 and hl | 1 3 2 | hl+1 all have duplicate diffs
   So:     half_length = (maxlength-5)/2 (or (maxlength-8)/2 if maxdepth even)
   Note:   The first valid cases are hl | 3 1 | hl+1 or hl | 2 1 4 | hl+1
   ------------------*/
   half_length = (maxdepth & 1) ? (maxlength-5) >> 1 : (maxlength-8) >> 1;

   printf(" Midmarks=%d-%d, Halflength=%d\n",HALFDEPTH,HALFDEPTH2,HALFLENGTH);//exit(1);

   /* Init tables for minimum sums */
   MinSumInit();

   /* Permanent File Storage */
   fout = fopen(send_file_name,"a");
   fprintf(fout,"-99 Version 6.00 "); /* DO NOT change this line! */
   fprintf(fout,"%2d-bit ",BITMAP1);  /* DO NOT change this line! */
   fprintf(fout,"(V523)");    /* you may enter a 4 char code here */
   fprintf(fout," continuing after 0");
   for( i=1; i<=re_depth; i++ ) fprintf(fout,"-%d",re_stub[i]);
   fprintf(fout,"  (OGR-%d<=%d; %d bits\n",maxdepth,MAXLENGTH,ChooseBits);
   fclose(fout);

   /* first zero bit in 16 bits */
   k = 0; m = 0x8000;
   for( i=1; i <= 16; i++) {
       for (j = k; j < k+m; j++) first[j] = i;
       k += m;
       m >>= 1;
   }
   first[0xffff] = 17;     /* just in case we use it */

   /* Prepare bit arrays for choose calcs */
   for( i=1; i < 256; i++ ) {
       bit32[i] = 0;
       bmBit[i] = 0x80000000 >> ((i-1) & 31);
   }
   for( i=1; i <  33; i++ ) bit32[i+32] = 1 << (32-i);

   //n_end   = n_end < FORCEDUMP ? n_end   : FORCEDUMP;
   /* this is for testing */
   if (FORCEDUMP < n_end) {
       FORCEDUMP = n_end;
       DUMPDEPTH = n_end+1;
   }
   printf(" OGR(%d) limited to %d length",maxdepth,MAXLENGTH);
   if( n_start ) {
      printf(" is being searched from");
      for(i=1;i<=n_start;i++) printf(" %d",start_stub[i]);
   }
   if( n_end ) {
      printf(" through");
      for(i=1;i<=n_end;i++) printf(" %d",end_stub[i]);
   }
   printf("\n\n");
   fflush(stdout);


   /* Prepare ctrl-break functions prior to starting loops */
   /* Note: UNIX & DJGPP (GCC's DOS port) have no SIGBREAK defined */
#ifndef SIGBREAK
#define SIGBREAK SIGINT
#endif
    signal( SIGBREAK, MyHandler );       /* Ctrl-Break */
    signal( SIGINT  , MyHandler );       /* Ctrl-C     */

    middle = time(&start);
    do start0 = time(&start); while (start0 == middle);
    middle = start0;
#ifdef WIN32
    GetProcessTimes(GetCurrentProcess(),&ft1,&ft2,&ft3,&ft4);
#endif

   //middle = start0 = time(&start);
#ifdef TIMINGTEST
   ftime(&start2);
   golombInit();
   ftime(&finish2);
   printf ("\n Session lasted %.2f seconds\n",((double)(finish2.time-start2.time))+
           (double)((signed)finish2.millitm-(signed)start2.millitm)/(double)1000.0);
#else
   golombInit();
   time (&finish);
   printf ("\n Session lasted %.0f seconds\n", difftime(finish,start0));
#ifdef WIN32
    GetProcessTimes(GetCurrentProcess(),&ft5,&ft6,&ft7,&ft8);
    printf("Process Time: %.1lf seconds\n",
        ((double)ft8.dwLowDateTime+(double)ft8.dwHighDateTime*4294967296.0-
        (double)ft4.dwLowDateTime+(double)ft4.dwHighDateTime*4294967296.0)*100e-9);
#endif
#endif
   return 0;
}



         /*
          * XXXxxxXXX  C H O O S E  !! !! !!
          */

#define MAX_C_DEPTH  9          /* we fill choose array to MAX_C_DEPTH-1 */

int Min_Sum(U dist0, int ndist)
{
    unsigned int l,adist;
    unsigned char *pSum;

    pSum = &Sum[dist0 >> 16][0];
    adist = pSum[0];
    if (ndist <= adist) l = pSum[ndist];
    else {
        l = pSum[adist];
        ndist -= adist;
        pSum = &Sum[dist0 & 0xffff][0];
        adist = pSum[0];
        if (ndist <= adist) l += pSum[ndist]+ndist*16;
        else {
            l += pSum[adist]+adist*16;
            ndist -= adist;
            pSum = &Sum[bmStub[1] >> 16][0];
            adist = pSum[0];
            if (ndist <= adist) l += pSum[ndist]+ndist*32;
            else {
                l += pSum[adist]+adist*32;
                ndist -= adist;
                l += Sum[bmStub[1] & 0xffff][ndist]+ndist*48;
            }
        }
    }
    return l;
}


/*---------------------------------*/
/*    CHOOSE Workhorse routine     */
/*---------------------------------*/

void Win32_fastcall golomb(
   unsigned long bitmap,          /* the uncompressed bitmap index */
   int maxdepth_m1,               /* maxdepth - 1 */
   int half_depth ,               /* half of maxdepth */
   int half_depth2,               /* half of maxdepth...2nd mark */
   int minz                       /* a known shortest limit */
){
   U comp0[MAX_C_DEPTH],comp1[MAX_C_DEPTH];       /* comparison bitmap */
   U list0[MAX_C_DEPTH],list1[MAX_C_DEPTH];       /* ruler bitmap */
   U dist0[MAX_C_DEPTH],dist1[MAX_C_DEPTH];       /* distance bitmap */
   int limit[MAX_C_DEPTH];                     /* limit for this mark */
   int count[MAX_C_DEPTH];                     /* current length */
   int depth=1;                             /* current number of marks */
   int i,s;                                 /* counter */
   int half_length = (max-1) >> 1;          /* 1st center mark limit */
   int half_length2;                        /* 2nd center mark limit */


   comp0[1] = dist0[1]  = bmStub[0] | bitmap;
   comp1[1] = dist1[1]  = bmStub[1];
   list0[1] = list1[1] = 0;
   count[1] = count[0] = 0;

   limit[1] = max-choose[compress(bitmap)][maxdepth_m1-1];
   if (limit[1] > half_length) limit[1] = half_length;
   limit[maxdepth_m1] = max;

start:
{
   U t=comp0[depth];
   if (t < 0xffff0000) s = first[t>>16];
   else if (t != 0xffffffff) s = first[t-0xffff0000] + 16;
   else {
      t = comp1[depth];
      if (t < 0xffff0000) s = first[t>>16] + 32;
      else s = first[t-0xffff0000] + 48;
   }
   if ((count[depth] + s) > limit[depth] ) goto up_level;
   count[depth] = count[depth] + s;
   if (s <= 31 ) {
       comp0[depth] = (comp0[depth] << s) | (comp1[depth] >> (32-s));
       list1[depth] = (list1[depth] >> s) | (list0[depth] << (32-s));
       comp1[depth] <<= s;
       list0[depth] >>= s;
   } else if (s <= 63) {
      comp0[depth] = comp1[depth] << (s-32);
      list1[depth] = list0[depth] >> (s-32);
      comp1[depth] = 0;
      list0[depth] = 0;
   } else {
      comp0[depth] = comp1[depth] = 0;
      list0[depth] = list1[depth] = 0;
   }
}

   cnodes++;
   if (depth < maxdepth_m1) {
       int d = count[depth]-count[depth-1];
       dist0[depth+1] = dist0[depth] | list0[depth] | bit32[d+32];
       depth++;
       if (depth < maxdepth_m1) {
          unsigned char *ptr = &choose[compress(dist0[depth])][maxdepth_m1-depth];
          if (count[depth-1] > max - ptr[1])  goto up_level;
          limit[depth] = max - ptr[0];
          if (depth <= half_depth2) {
              if (depth <= half_depth) {
                 limit[depth] = (limit[depth] > half_length) ? half_length : limit[depth];
              } else {
                 half_length2 = max - count[half_depth] - 1;
                 limit[depth] = (limit[depth] > half_length2) ? half_length2 : limit[depth];
              }
          }
       }
       count[depth] = count[depth-1];
       list1[depth] = list1[depth-1] | bit32[d];  /* ==0 if [x] > [32] */
       list0[depth] = list0[depth-1] | bit32[d+32];
       dist1[depth] = dist1[depth-1] | list1[depth];
       comp0[depth] = comp0[depth-1] | dist0[depth];
       comp1[depth] = comp1[depth-1] | dist1[depth];
       goto start;
   } else {
      if (count[depth] > 128) {
          U bmDiffs[2];
          bmDiffs[0] = bmStub[2]; bmDiffs[1] = bmStub[3];
          for (i = 1; i <= maxdepth_m1; i++) {
            for (s = i-1 ; s >= 0; s--) {
              int diff = count[i]-count[s];
              if (diff > 64) {       /* only need check 64<diff<=max/2 */
                if (diff+diff > count[depth]) break;
                if (bmDiffs[(diff-65) >> 5] & bmBit[diff]) goto start;
                bmDiffs[(diff-65) >> 5] |= bmBit[diff];
              }
            }
          }
      }
      max=count[depth]-1;
      if( max < minz ) return;
      /* reset the limits for when we go back down! */
      limit[maxdepth_m1] = max;
      for( i=1; i < maxdepth_m1; i++ ) {
         limit[i]= max-choose[compress(dist0[i])][maxdepth_m1-i];
      }
      half_length = (max-1) >> 1;
      limit[half_depth] = (limit[half_depth] > half_length) ? half_length : limit[half_depth];
      if (half_depth2 != half_depth) {
          half_length2 = max - count[half_depth] - 1;
          limit[half_depth2] = (limit[half_depth2] > half_length2) ? half_length2 : limit[half_depth2];
      }
      goto up_level;
   }

up_level:
   if (--depth == 0) return;
   goto start;
}

void Win32_fastcall bitwise(
   int nummax,
   int numbits,
   U   bitmap,
   int bit,
   int *params,
   int *newbit
){
   int n,minz;
   if (numbits) {
      for( n = bit; n <= ChooseBits-numbits; n++ ) {
         newbit[numbits]=n;
         bitwise(nummax,numbits-1,bitmap+(1<<n),n+1,params,newbit);
      }
   } else {
      U bitmap_u = decompress(bitmask,(bitmap << ttmChooseBits));
      int m1 = params[0];  /* m1 = maxdepth-1 */
      max = choose[bitmap][m1]-1;  /* Why search for SAME length? */
      minz = Min_Sum(bitmap_u,m1);
      if( minz > max ) {
		  if ((max+1 != minz) && (max+1 < ChooseLimit)) {
              printf("\nMinZ Error: %08lX %08lX %i: %i %i.\n",bitmap,bitmap_u,m1,max+1,minz);
			  //exit(1);
		  }
		  max=minz-1;
	  } else golomb(bitmap_u,m1,params[1],params[2],minz);
      choose[bitmap][m1] = (++max < 255) ? max : 255;
      if (m1 >= 4) choose2[bitmap][m1-4] = max;
      for( bit=1; bit <= nummax; bit++ ) {
         if (choose[bitmap-(1<<newbit[bit])][m1] > max ) {
             choose[bitmap-(1<<newbit[bit])][m1] = max;
         }
      }
   }
}


void fill_choose (int stubdepth)
{
   int i,j,d,numbits;         /* counters */
   time_t  start0,start1, finish; /* used to time search */
   unsigned long bitmap;
   int maxdepth;                   /* maximum number of marks in ruler */
   int params[3];                  /* stores info for workhorse subroutine */
   int newbit[MAX_CHOOSEBITS+1];

   /* get stub */
   //printf("\n Generating choose for current stub: 0");
   bmStub[3] = bmStub[2] = bmStub[1] = bmStub[0] = 0;
   for (i = 1; i < stubdepth; i++) {
           //printf("-%d",count[i]);
           d = 0;
           for (j = i; j < stubdepth; j++) {
              d += count[j]-count[j-1];
              if (d > 128) break;
              if (d < 1) { printf("Error in Stub.\n"); exit(3); }
              if (bmStub[(d-1) >> 5] & bmBit[d]) { printf("Stub isn't golomb.\n"); exit(2); }
              bmStub[(d-1) >> 5] |= bmBit[d];
           }
       }
   //printf("\nbmStub: %08lX %08lX %08lX %08lX",bmStub[0],bmStub[1],bmStub[2],bmStub[3]);

   bitmask = bmStub[0];
   if( init_compress() ) {printf("Error in init_compress");exit(1);}

   /* Preset the choose array */
   //printf("Setting Max to %d\n",ChooseLimit);
   for (bitmap = 0; bitmap < (1 << ChooseBits); bitmap++) {
       choose[bitmap][0] = 0;
       choose[bitmap][1] = Min_Sum(decompress(bitmask,(bitmap << ttmChooseBits)),1);
       for (j = 2; j < MAX_C_DEPTH; j++) choose[bitmap][j] = ChooseLimit;
   }

   cnodes = 0;
   time (&start0);
   for (maxdepth=3; maxdepth <= MAX_C_DEPTH; maxdepth++) {
      time (&start1);
      //printf("%2d: ",maxdepth);

      /* precalculate 3 numbers used often in workhorse engine */
      params[0] = (maxdepth-1);              /* maxdepth_m1 */
      params[1] = (maxdepth-1) >> 1;         /* half_depth  */
      params[2] = (maxdepth-1) - params[1];  /* half_depth2 */

      for (numbits = ChooseBits; numbits >= 0; numbits--) {
         //printf("%d",numbits); fflush(stdout);
         bitwise(numbits,numbits,0l,0,params,newbit);
         //printf(" "); fflush(stdout);
      }
      time (&finish);
   }
   CNodes += (double)cnodes;
   printf (" %d sec, %.0f nodes\n",finish-start0,CNodes);
}