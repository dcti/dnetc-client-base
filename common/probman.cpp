// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probman.cpp,v $
// Revision 1.2  1998/11/06 02:32:27  cyp
// Ok, no more restrictions (at least from the client's perspective) on the
// number of processors that the client can run on.
//
// Revision 1.1  1998/09/28 02:36:33  cyp
// Created. Just stubs for now.
//
// 
#if (!defined(lint) && defined(__showids__))
const char *probman_cpp(void) {
return "@(#)$Id: probman.cpp,v 1.2 1998/11/06 02:32:27 cyp Exp $"; }
#endif

#include "baseincs.h"  // malloc()/NULL/memset()
#include "problem.h"   // Problem class
#include "probman.h"   // thats us. Keep prototypes in sync.

// -----------------------------------------------------------------------

static struct
{
  Problem **probtable;
  unsigned int tablesize;
  unsigned int probcount;
} probmanstatics = {NULL,0,0};

// -----------------------------------------------------------------------

Problem *GetProblemPointerFromIndex(unsigned int probindex)
{
  if (probmanstatics.probcount && probindex < probmanstatics.probcount )
    return probmanstatics.probtable[probindex];
  return NULL;
}  

// -----------------------------------------------------------------------

int InitializeProblemManager(unsigned int maxnumproblems)
{
  unsigned int i, probcount;
  
  if (maxnumproblems == 0 || probmanstatics.probtable!= NULL)
    return -1;
  if (((int)(maxnumproblems)) < 0)
    maxnumproblems = (16*1024);
    
  probmanstatics.probtable=(Problem **)
                             malloc(maxnumproblems * sizeof(Problem *));
  if (probmanstatics.probtable == NULL)
    return -1;

  probmanstatics.tablesize = maxnumproblems;
  memset((void *)probmanstatics.probtable,0,
                             (maxnumproblems * sizeof(Problem *)));
    
  probcount = 0;
  for (i=0;i<maxnumproblems;i++)
    {
    probmanstatics.probtable[i]=new Problem;
    if (probmanstatics.probtable[i]==NULL)
      break;
    probcount++;
    }
  if (probcount == 0)
    {
    free((void *)probmanstatics.probtable);
    probmanstatics.probtable = NULL;
    probmanstatics.probcount = 0;
    probmanstatics.tablesize = 0;
    return -1;
    }
  probmanstatics.probcount = probcount+1;
  return (int)(probmanstatics.probcount);
}    

// -----------------------------------------------------------------------

int DeinitializeProblemManager(void)
{
  unsigned int i, tablesize = probmanstatics.tablesize;
  Problem **probtable = probmanstatics.probtable;
  probmanstatics.tablesize = 0;
  probmanstatics.probtable = NULL;
  probmanstatics.probcount = 0;

  if (probtable!= NULL)
    {
    for (i=0;i<tablesize;i++)
      {
      if (probtable[i])
        delete probtable[i];
      }
    free((void *)probtable);
    }
  return 0;
}

// -----------------------------------------------------------------------
