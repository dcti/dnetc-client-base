/* Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/ 
const char *probman_cpp(void) {
return "@(#)$Id: probman.cpp,v 1.9.2.5 2001/02/23 03:38:07 sampo Exp $"; }

#include "baseincs.h"  // malloc()/NULL/memset()
#include "problem.h"   // Problem class
#include "probman.h"   // thats us. Keep prototypes in sync.

// -----------------------------------------------------------------------

static struct
{
  Problem **probtable;
  unsigned int tablesize;
  unsigned int probcount;
} probmanstatics = {((Problem **)0),0,0};

// -----------------------------------------------------------------------

Problem *GetProblemPointerFromIndex(unsigned int probindex)
{
  if (probmanstatics.probcount && probindex < probmanstatics.probcount )
    return probmanstatics.probtable[probindex];
  return ((Problem *)0);
}  

// -----------------------------------------------------------------------

//Note: Problems managed from Benchmark etc are not managed by ProblemManager.

int GetProblemIndexFromPointer( Problem *prob )
{
  unsigned int probindex;
  if (probmanstatics.probcount)
  {
    for (probindex = 0; probindex < probmanstatics.probcount; probindex++ )
    {
      if (probmanstatics.probtable[probindex] == prob)
        return (int)probindex;
    }
  }
  return -1;
}

// -----------------------------------------------------------------------

// InitializeProblemManager() allocates maxnumproblems, and returns that
// value, or returns error value of -1 if it could not allocate the number
// of problems requested.

// XXX
// if this is in fact the wrong behaviour, for example, the client should
// return success if it only allocated *some* of the crunchers, then we can
// change the code to reflect this, but this seems more bulletproof. feel
// free to prove me wrong.  - sampo.

int InitializeProblemManager(unsigned int maxnumproblems)
{
  unsigned int i, probcount;
  
  if (maxnumproblems == 0 || probmanstatics.probtable != ((Problem **)0))
    return -1;
  if (((int)(maxnumproblems)) < 0)
    maxnumproblems = (16*1024);     // XXX
                                    // a comment to explain this magic
                                    // number would be nice.  Why not 128
                                    // as in GetMaxCrunchersPermitted() ?
                                    // a #define for magic numbers would be
                                    // good.

  probmanstatics.probtable=(Problem **)
                             malloc(maxnumproblems * sizeof(Problem *));
  if (probmanstatics.probtable == ((Problem **)0))
    return -1;

  probmanstatics.tablesize = maxnumproblems;
  memset((void *)probmanstatics.probtable,0,
                             (maxnumproblems * sizeof(Problem *)));
    
  probcount = 0;
  for (i=0;i<maxnumproblems;i++)
  {
    probmanstatics.probtable[i]=ProblemAlloc();
    if (probmanstatics.probtable[i]==((Problem *)0))
      break;
    probcount++;
  }
  if (probcount != maxnumproblems)
  {
    unsigned int j;
    
    for(j=0; j<probcount; j++)
        ProblemFree(probmanstatics.probtable[j]);
        
    free((void *)probmanstatics.probtable);
    probmanstatics.probtable = ((Problem **)0);
    probmanstatics.probcount = 0;
    probmanstatics.tablesize = 0;
    return -1;
  }
  probmanstatics.probcount = probcount;
  return (int)(probmanstatics.probcount);
}    

// -----------------------------------------------------------------------

int DeinitializeProblemManager(void)
{
  Problem **probtable = probmanstatics.probtable;

  if (probtable != ((Problem **)0))
  {
    for (;probmanstatics.probcount>0;probmanstatics.probcount--)
    {
      if (probtable[probmanstatics.probcount-1])
        ProblemFree(probtable[probmanstatics.probcount-1]);
      probtable[probmanstatics.probcount-1] = ((Problem *)0);
    }
    free((void *)probtable);
  }

  probmanstatics.probcount = 0;
  probmanstatics.tablesize = 0;
  probmanstatics.probtable = ((Problem **)0);
  return 0;
}

// -----------------------------------------------------------------------
