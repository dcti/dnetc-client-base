/* Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/ 
const char *probman_cpp(void) {
return "@(#)$Id: probman.cpp,v 1.9.2.4 2000/11/12 17:16:43 cyp Exp $"; }

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

int InitializeProblemManager(unsigned int maxnumproblems)
{
  unsigned int i, probcount;
  
  if (maxnumproblems == 0 || probmanstatics.probtable!= ((Problem **)0))
    return -1;
  if (((int)(maxnumproblems)) < 0)
    maxnumproblems = (16*1024);

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
  if (probcount == 0)
  {
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
