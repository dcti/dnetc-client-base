/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrng_layered.cpp,v 1.2 2008/12/30 23:05:52 kakace Exp $
 *
 * This file demonstrates the implementation of a layered, two stages, OGR-NG
 * core. If you wish to test it, proceed as follows :
 * - Create a core by the usual method (see ogr_readme.txt for details)
 * - Define OGROPT_ALTERNATE_CYCLE to 1
 * - Include "ogrng_layered.cpp" after the inclusion of the code base
 *   (ogrng_codebase.cpp)
 * Depending on your CPU, this core may or may not be faster.
 */


static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose)
{
   struct OgrLevel *lev = &oState->Levels[oState->depth];
   int depth       = oState->depth;
   int maxlen_m1   = oState->max - 1;
   int nodes       = *pnodes;
   int limit       = lev->limit;
   int mark        = lev->mark;
   
   SETUP_TOP_STATE(lev);
   if (depth > oState->half_depth2)
      goto stage2;

   do {
      for (;;) {
         if (comp0 < (SCALAR)~1) {
            int s = LOOKUP_FIRSTBLANK(comp0);
            
            if ((mark += s) > limit)
               break;
            COMP_LEFT_LIST_RIGHT(lev, s);
         }
         else {
            if ((mark += SCALAR_BITS) > limit)
               break;
            if (comp0 == (SCALAR)~0) {
               COMP_LEFT_LIST_RIGHT_WORD(lev);
               continue;
            }
            COMP_LEFT_LIST_RIGHT_WORD(lev);
         }
         
         lev->mark = mark;
         PUSH_LEVEL_UPDATE_STATE(lev);
         ++lev; ++depth;
         limit = choose(dist0, depth);
         if (depth > oState->half_depth2) {
            lev->limit = limit;
            if (--nodes <= 0) {
               lev->mark = mark;
               goto exit;
            }
stage2:     //---------- Stage #2 begins here ----------
            do {
               for (;;) {
                  if (comp0 < (SCALAR)~1) {
                     int s = LOOKUP_FIRSTBLANK(comp0);
                     
                     if ((mark += s) > limit)
                        break;
                     COMP_LEFT_LIST_RIGHT(lev, s);
                  }
                  else {
                     if ((mark += SCALAR_BITS) > limit)
                        break;
                     if (comp0 == (SCALAR)~0) {
                        COMP_LEFT_LIST_RIGHT_WORD(lev);
                        continue;
                     }
                     COMP_LEFT_LIST_RIGHT_WORD(lev);
                  }
                  
                  lev->mark = mark;
                  if (depth == oState->maxdepthm1)
                     goto exit;
                  
                  PUSH_LEVEL_UPDATE_STATE(lev);
                  ++lev; ++depth;
                  lev->limit = limit = choose(dist0, depth);
                  if (--nodes <= 0) {
                     lev->mark = mark;
                     goto exit;
                  }
               }  // for (;;)
               --lev; --depth;
               POP_LEVEL(lev);
               limit = lev->limit;
               mark  = lev->mark;
            } while (depth >= oState->half_depth2);
            //---------- Stage #2 ends here ----------
         }
         else {
            if (depth > oState->half_depth) {
               int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;
               
               if (depth < oState->half_depth2) {
                  #if (SCALAR_BITS <= 32)
                  temp -= LOOKUP_FIRSTBLANK(dist0);
                  #else
                  temp -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));
                  #endif
               }

               if (limit > temp) {
                  limit = temp;
               }
            }
            lev->limit = limit;
            if (--nodes <= 0) {
               lev->mark = mark;
               goto exit;
            }
         }
      }  // for (;;)
      --lev;
      --depth;
      POP_LEVEL(lev);
      limit = lev->limit;
      mark  = lev->mark;
   } while (depth > oState->stopdepth);
   
exit:
   SAVE_FINAL_STATE(lev);
   *pnodes -= nodes;
   return depth;
}
