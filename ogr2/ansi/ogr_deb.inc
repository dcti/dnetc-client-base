/*
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file (ogr_deb.inc) contains all the OGR debug stuff
 *
 * $Id: ogr_deb.inc,v 1.1.2.3.2.1 2001/04/01 22:02:15 andreasb Exp $
 */

#ifdef OGR_DEBUG
//  #define OGR_DEBUG_CYCLE(__x) __x
#endif

#ifndef OGR_DEBUG_CYCLE
  #define OGR_DEBUG_CYCLE(__x)
#endif


#ifdef OGR_DEBUG
static void dump(int depth, struct Level *lev, int limit)
{
  printf("--- depth %d\n", depth);
  printf("list=%08x%08x%08x%08x%08x\n", lev->list[0], lev->list[1], lev->list[2], lev->list[3], lev->list[4]);
  printf("dist=%08x%08x%08x%08x%08x\n", lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3], lev->dist[4]);
  printf("comp=%08x%08x%08x%08x%08x\n", lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3], lev->comp[4]);
  printf("cnt1=%d cnt2=%d limit=%d\n", lev->cnt1, lev->cnt2, limit);
  //sleep(1);
}

static void dump_ruler(struct State *oState, int depth)
{
  int i;
  printf("max %d ruler ", oState->max);
  for (i = 1; i < depth; i++) {
    printf("%d ", oState->Levels[i].cnt2 - oState->Levels[i-1].cnt2);
  }
  printf("\n");
}
#endif
