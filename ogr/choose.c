/*--------------------------------------------------------------------*/
/*                                                                    */
/*                    http://members.aol.com/golomb20                 */
/*                                                                    */
/*                 This program is a pre-program for use              */
/*                 with GARSP, a routine to find Optimal              */
/*                 Golomb Rulers.  Please see the OGR                 */
/*                 home page for more details.                        */
/*                                                                    */
/*                                                                    */
/*            Precompiled outputs are available for download          */
/*                                                                    */
/*--------------------------------------------------------------------*/
/*  Choose.c ...creates CHOOSE.DAT file of MAXBITS bits/map for GARSP */
/*                                                                    */
/*  Usage:                                                            */
/*      If you already have a good "choose.dat" keep it there!        */
/*      Change the MAXBITS to what you need, and compile this source  */
/*      Run choose.c (It builds from the old choose.dat as needed!)   */
/*      If you stop the run at any point, just restart it to finish   */
/*                                                                    */
/*  When done:                                                        */
/*         * delete the "choose.ck1" file                             */
/*         * move your old choose.dat file to a safe place            */
/*         * rename choose.new to choose.dat                          */
/*--------------------------------------------------------------------*/
/*                                                                    */
/*  choose[0][N] = OGR length of N segments, with no bits preset      */
/*                                                                    */
/*  Version 5, Alexey Guzeev (aga@permonline.ru) & M.Garry  4/98      */
/*       Added ability to shrink or grow existing choose file.        */
/*       Added ability to generate choose.dat in several runs         */
/*       Added idle priority code for OS/2 and Win32                  */
/*       Added AMD K5 specific optimization                           */
/*                                                                    */
/*  Version 4  modified by Geoffrey Faivre-Malloy 3/5/98              */
/*       Added parens around 4 "<<" and ">>" operations so this       */
/*       compiles correctly on Borland c++ v5.0  "x<<y" -> "(x<<y)"   */
/*                                                                    */
/*  Version 3  modified by M.Garry 2/4/98                             */
/*       Replaces Version 2 ...                                       */
/*       Now includes a version identifier, and also                  */
/*       Outputs the array in inverted format                         */
/*           (6% speed up in GARSP and also simplifies GARSP code     */
/*                                                                    */
/*  Version 2  modified by M.Garry                                    */
/*       Fixes error for when s>=32 which allowed an undefined        */
/*       (non-ansi) x<<32 operation.  This error  was already         */
/*       fixed in GARSP 5.06, but not in CHOOSE.                      */
/*                                                                    */
/*--------------------------------------------------------------------*/


/*-----INCLUDES----*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef OS2
#define INCL_DOSPROCESS
#include <os2.h>
#endif

#ifdef WIN32
#include <windows.h>
#endif


#ifdef AMDK5
# ifndef __GNUC__
#  error K5 optimization designed to be compiled using GCC
# endif
# ifndef __i386__
#  error K5 optimization is not accessible at non-x86 machines
# endif
#endif


/*-----TYPEDEFs----*/
typedef unsigned long U;


/*-----DEFINES-----*/
/* Fill choose array up to choose[10][ffff] */
#define MAXDEPTH 12             /* we fill choose array to MAXDEPTH-1 */
#define MAXBITS 12              /* we fill choose array to MAXBITS */
#define MAXFILE (1<<MAXBITS)    /* needed to dimension choose */

#define MIN_CHECKPOINT_INTERVAL 3600 /* minimum seconds between checkpoints */
                                     /* set to 0 if no checkpoints required */
/*                                   warning: not ANSI compliant
#define DOUBLE_CHECKPOINT            
*/


#ifdef DOUBLE_CHECKPOINT
# include <io.h>
#endif /* DOUBLE_CHECKPOINT */


/*-----GLOBALS-----*/
int max;                        /* maximum length of ruler */
int depth;                      /* current number of marks in ruler */
int maxdepth;                   /* maximum number of marks in ruler */
unsigned char choose[MAXDEPTH][MAXFILE]; /* array of ruler lengths */
int half_length;                /* half of max */
int half_length2;               /* half of max using 2nd center mark */
int half_depth;                 /* half of maxdepth */
int half_depth2;                /* half of maxdepth, adjusted for 2nd mark */
int count[MAXDEPTH+1];          /* current length */
#ifndef AMDK5
int first[0x10000];             /* first open bit in a 16 bit bitmap */
#endif

int newbit[MAXBITS+1];
int new;
int counter;

int diff,diffs[1000],bit[1000];
char BUILDUP = 0;                /* used to make big choose from small one */


/*---------------------------------*/
/*    GOLOMB Workhorse routine     */
/*---------------------------------*/

#ifdef WIN32
   __fastcall golomb(U bitmap)
#else
   void golomb(U bitmap)
#endif
{
   U c[MAXDEPTH];          /* comparison bitmap */
   U list[MAXDEPTH];       /* ruler bitmap */
   U dist[MAXDEPTH];       /* distance bitmap */
   U t;                    /* temp bitmap */
   int limit[MAXDEPTH];    /* limit for this mark */
   int i,s;

   depth = 1;                                /* 1st mark @ left edge
                                                begin placing 2nd mark */
   c[1]     = dist[1]  = bitmap;
   list[1]  = count[1] = 0;
   limit[1] = max-choose[maxdepth-2][0];

start:
   t = ~c[depth];

#ifdef AMDK5
  __asm__ ("
             movl $33, %%ebx
             bsrl %%eax, %%eax
             jz   0f
             subl %%eax, %%ebx
             decl %%ebx
           0:
           "
           : "=b"(s) : "a"(t) : "%eax", "%ebx", "cc");
#else /* AMDK5 */
   if( t > 0x0000ffff ) {
      s = first[t>>16];
   } else {
         if( t > 0 ) {
            s = 16 + first[t];
         } else {
            s = 33;
         }
   }
#endif /* AMDK5 */

   if( (count[depth] += s) > limit[depth] ) goto up_level;

   if (depth == maxdepth-1) {
      for( i=1; i<maxdepth; i++ ) {
         for( s=i-1; s >= 0; s-- ) {
            diff = count[i]-count[s];
            if( diff > 32 ) {       /* only need check 32<diff<=max/2 */
               if( diff+diff > count[depth] ) break;
               if( diffs[diff] ) goto skip;
               diffs[diff] = 1;
            }
         }
      }
      new=count[depth];
      max=new-1;
      /* reset the limits for when we go back down! */
      for( i=1; i<maxdepth; i++ ) {
         limit[i]= max-choose[maxdepth-i-1][(dist[i])>>(32-MAXBITS)];
      }
      if( maxdepth > 5 ) {
         half_length = ((max + 1) >> 1) - 1;
         limit[half_depth] = (limit[half_depth] > half_length) ? half_length : limit[half_depth];
         if(half_depth2-half_depth) {
            half_length2 = max - count[half_depth] - 1;
            limit[depth] = (limit[depth] > half_length2) ? half_length2 : limit[depth];
         }
      }
skip:
      for( i=33; i <= count[depth]>>1; i++ ) diffs[i]=0;
      goto up_level;
   }

   if( s > 31 ) {
      c[depth] = 0;
      list[depth] = 0;
   } else {
      c[depth] <<= s;
      list[depth] >>= s;
   }

   depth++;
   count[depth] = count[depth-1];
   list[depth] = list[depth-1] | bit[count[depth-1]-count[depth-2]];
   dist[depth] = dist[depth-1] | list[depth];
   c[depth]    = c[depth-1]    | dist[depth];

   limit[depth] = max-choose[maxdepth - depth - 1][(dist[depth])>>(32-MAXBITS)];

   if (depth <= half_depth2) {
      if (depth <= half_depth) {
         limit[depth] = (limit[depth] > half_length) ? half_length : limit[depth];
      } else {
         half_length2 = max - count[half_depth] - 1;
         limit[depth] = (limit[depth] > half_length2) ? half_length2 : limit[depth];
      }
   }
   goto start;

up_level:
   if (--depth == 0) return;
   goto start;
}





void bitwise(
   int nummax,
   int numbits,
   U   bitmap,
   int bit
){
   int m1,n;
   if( numbits ) {
      for( n = bit; n <= MAXBITS-numbits; n++ ) {
         newbit[numbits]=n;
         bitwise(nummax,numbits-1,bitmap+(1<<n),1+n);
      }
   } else {
      m1 = maxdepth-1;
      max = choose[m1][bitmap]-1;      /* Why search to find SAME length? */
      half_length = ((max + 1) >> 1) - 1;
         new = 0;
         if( BUILDUP==0 || bitmap%(1<<BUILDUP) ) {
            golomb(bitmap<<(32-MAXBITS));
         }
         if( new ) choose[m1][bitmap] = new;
      for( bit=1; bit <= nummax; bit++ ) {
         if( choose[m1][bitmap] < choose[m1][bitmap-(1<<newbit[bit])] ) {
            choose[m1][bitmap-(1<<newbit[bit])] = choose[m1][bitmap];
         }

      }
   }
}


int main (void)
{
   int i,j,numbits;             /* counters */
   time_t  start, finish;       /* used to time search */
   FILE *length_file;           /* file to write ruler lengths to */
   FILE *checkpoint_file;	/* checkpoint file */
   time_t last_checkpoint_save;
   char restored=0;
   int load_depth=0;
   int load_bits=0;
   char maxd,maxb;

  /* bump priority into idle class -- let others get CPU first */
#ifdef WIN32
   SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
   SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
#endif 
#ifdef OS2
   DosSetPriority(PRTYS_THREAD, PRTYC_IDLETIME, PRTYD_MINIMUM, 0);
#endif

   for (i=0; i<MAXFILE; i++) {  /* Preset the choose array */
      choose[0][i] = 0;
      for (j=1; j<MAXDEPTH; j++) choose[j][i] = 255;  /* max for char */
   }
#ifndef AMDK5
   for (i=0; i<=0xffff; i++) {  /* loc of first 1 in a 16 bit bitmap */
      for( j=15; j >= 0; j--) if( i & (1<<j) ) {first[i] = 16-j; break;}
   }
#endif /* AMDK5 */

   for( i=1; i < 1000; i++ ) bit[i] = diffs[i] = 0;
   for( i=1; i <  33; i++ ) bit[i] = 1 << (32-i);

   if (MIN_CHECKPOINT_INTERVAL) { /* read checkpoint */
     unsigned char version;
     checkpoint_file=fopen("choose.ck1", "rb");
     if (!checkpoint_file) {
#ifdef DOUBLE_CHECKPOINT
       if (access("choose.ck2", 0)==0) {
         printf("Error: Cannot open choose.ck1 to read\n");
         exit(1);
       } else
#endif /* DOUBLE_CHECKPOINT */
       {
         printf("\n Creating %d-bit CHOOSE.NEW from scratch.\n", MAXBITS);
       }
     } else {
       if (1!=fread(&version , sizeof(version), 1, checkpoint_file)) {
         printf("Error: choose.ck1 size is 0\n");
         exit(1);
       }
       if (version!=0) {
         printf("Error: Wrong choose.ck1 version\n");
         exit(1);
       }
       if (1!=fread(&maxd, sizeof(maxd), 1, checkpoint_file)) exit(1);
       if (maxd != MAXDEPTH) {
         printf("Error: Wrong version (MAXDEPTH) in choose.ck1!\n");
         exit(1);
       }

       if (1!=fread(&maxb, sizeof(maxb), 1, checkpoint_file)) exit(1);
       if (maxb != MAXBITS) {
         printf("  Error: Wrong version (MAXBITS) in choose.ck1!\n");
         printf("(Did you forget to delete an old choose.ck1 file?)\n");
         exit(1);
       }

       if (1!=fread(&load_depth, sizeof(load_depth), 1, checkpoint_file)) exit(1);
       if (1!=fread(&load_bits , sizeof(load_bits ), 1, checkpoint_file)) exit(1);
       if (1!=fread(choose     , sizeof(choose    ), 1, checkpoint_file)) exit(1);
       if (1!=fread(&BUILDUP   , sizeof(BUILDUP   ), 1, checkpoint_file)) exit(1);
       fclose(checkpoint_file);
       printf("Creating CHOOSE.NEW starting from choose.ck1 at %d-%d\n", load_depth, load_bits);
       restored=1;
     }
   }

   if (!restored)
   {
     unsigned char version;
     checkpoint_file=fopen("choose.dat", "rb");
     if (checkpoint_file) {
       printf(" \n Using CHOOSE.DAT to build from. \n");

       if (1!=fread(&version , sizeof(version ), 1, checkpoint_file)) {
         printf("Error: choose.dat size is 0\n");
         exit(1);
       }
       if ((version!=1) && (version!=101)) {
         printf("Error: choose.dat has unknown version\n");
         exit(1);
       }

       if (1!=fread(&maxd, sizeof(maxd), 1, checkpoint_file)) exit(1);
       if (maxd != MAXDEPTH) {
         printf("Error: Wrong version (MAXDEPTH) in choose.dat!\n");
         exit(1);
       }

       if (1!=fread(&maxb, sizeof(maxb), 1, checkpoint_file)) exit(1);
       if (maxb >= MAXBITS) {
         if ((length_file = fopen("choose.new","wb")) == NULL) {
           printf("\n\nError: Cannot open output file\n\n"); exit(1);
         }
         printf("Reading %d bits/map from the choose.dat (%d bits/map)\n\n", MAXBITS, maxb);
         {
           unsigned char data[22];       /* data being copied */
           unsigned char maxbb=MAXBITS;
           fwrite(&version, sizeof(char), 1, length_file);
           fwrite(&maxd   , sizeof(char), 1, length_file);
           fwrite(&maxbb  , sizeof(char), 1, length_file);
           for (j=0; j<1<<MAXBITS; ++j) {
             fread ( data, sizeof(char), maxd, checkpoint_file);
             fwrite( data, sizeof(char), maxd, length_file);
             for (i=1; i<1<<(maxb-MAXBITS); ++i) {
               fread(data, sizeof(char), maxd, checkpoint_file);
             }
           }
         }
         fclose(checkpoint_file);
         fclose(length_file);
         printf("\n  --- Choose.New is ready to use after being renamed to Choose.Dat --- \n\n");
         exit(0);
       }

       BUILDUP = MAXBITS-maxb;
       printf("Reading %d bits/map from choose.dat\n\n",maxb);

       {
          char dummy[MAXDEPTH];
          for( j=0; j<MAXFILE; j+=(1<<BUILDUP) ) {
             if (MAXDEPTH!=fread(dummy,sizeof(char),MAXDEPTH,checkpoint_file)) {
                printf("Error reading choose.dat file!\n");
                exit(1);
             }
             for( i=0; i<MAXDEPTH; i++ ) choose[i][j]=dummy[i];
          }
       }

       fclose(checkpoint_file);
     }
   }

   time(&last_checkpoint_save);

   for (maxdepth=2; maxdepth<=MAXDEPTH; maxdepth++) {
      counter=0;
      time (&start);
      printf("%2d marks bits  ",maxdepth);
      half_depth2 = half_depth  = ((maxdepth+1)>>1)-1;
      if ( !(maxdepth%2) ) half_depth2++;
      if (maxdepth < 5) half_depth = half_depth2 = 0;
      for (numbits = MAXBITS; numbits >= 0; numbits--) {
         printf("%d",numbits); fflush(stdout);
         if ( (maxdepth<load_depth) ||
              ((maxdepth==load_depth) && (numbits>=load_bits)) ) {
           printf(" "); fflush(stdout);
           continue;
         }

         bitwise(numbits,numbits,0l,0);

         /* write checkpoint */
         if ( (MIN_CHECKPOINT_INTERVAL==0) ||
              (time(NULL)-last_checkpoint_save<MIN_CHECKPOINT_INTERVAL) ) {
           printf(" "); fflush(stdout);
         } else {
           unsigned char version=0;
           printf("."); fflush(stdout);

#ifdef DOUBLE_CHECKPOINT
           if (access("choose.ck2", 0)==0) {
             if (remove("choose.ck2")) {
               printf("\nremove unsuccessful. Trying to continue anyway\n");
             }
           }
           if (rename("choose.ck1", "choose.ck2")) {
             if (restored) {
               printf("\nrename unsuccessful. Trying to continue anyway\n");
             }
           }
#endif /* DOUBLE_CHECKPOINT */

           checkpoint_file=fopen("choose.ck1", "wb");
           if (!checkpoint_file) {
             printf("Error: Cannot open choose.ck1 to write to it\n");
             exit(1);
           }
           if (1!=fwrite(&version , sizeof(version ), 1, checkpoint_file)) exit(1);
           {
             maxd = MAXDEPTH;
             maxb = MAXBITS;
             if (1!=fwrite(&maxd, sizeof(maxd), 1, checkpoint_file)) exit(1);
             if (1!=fwrite(&maxb, sizeof(maxb), 1, checkpoint_file)) exit(1);
           }  
           if (1!=fwrite(&maxdepth, sizeof(maxdepth), 1, checkpoint_file)) exit(1);
           if (1!=fwrite(&numbits , sizeof(numbits ), 1, checkpoint_file)) exit(1);
           if (1!=fwrite(choose   , sizeof(choose  ), 1, checkpoint_file)) exit(1);
           if (1!=fwrite(&BUILDUP , sizeof(BUILDUP ), 1, checkpoint_file)) exit(1);
           fclose(checkpoint_file);
           restored=1;
           time(&last_checkpoint_save);
         }
      }
      time (&finish);
      printf (" %.0f sec", difftime(finish,start));
      if(counter) printf(" %d better",counter);
      printf("\n");
   }
   {
      unsigned char version = 1;        /* identifies specific files */
      char max_golomb = MAXDEPTH;       /* char for output file */
      char max_bits = MAXBITS;          /* char for output file */
      if ((length_file = fopen("choose.new","wb")) == NULL) {
         printf("Error: Cannot open choose.new to write to it\n");
         exit(1);
      }
      for( i=0; i<MAXFILE; i++) choose[0][i]=0;
      fwrite(&version,sizeof(char),1,length_file);
      fwrite(&max_golomb,sizeof(char),1,length_file);
      fwrite(&max_bits,sizeof(char),1,length_file);

      /*
       * write inverted format ... makes GARSP 6% faster & other benefits
       *
       * fwrite(&(choose[1][0]),sizeof(char),(max_golomb-1)*MAXFILE,length_file);
       *   (above line replaced with ...)
       */

      for( i=0; i<MAXFILE; i++ ) {
         for( j=0; j<max_golomb; j++ ) {
            fwrite(&(choose[j][i]),sizeof(char),1,length_file);
         }
      }

      fclose(length_file);
   }
   printf("\n  --- Choose.New is ready to use after being renamed to Choose.Dat --- \n\n");
   return 0;
}

