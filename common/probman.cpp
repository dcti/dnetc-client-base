// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probman.cpp,v $
// Revision 1.7  1998/12/14 11:47:36  cyp
// Thread index (probman index) is assigned through the problem constructor.
//
// Revision 1.6  1998/11/14 13:57:42  cyp
// ProbMan saves its index to the Problem object. (needed for chrisb's copro
// board code)
//
// Revision 1.5  1998/11/13 21:08:49  cyp
// Changed DeinitializeProblemManager() for chrisb's x86 option board support
// so that the index is still valid when the problem is destroyed. This
// change does not affect DeinitializeProblemManager()s functionality.
//
// Revision 1.4  1998/11/12 18:50:57  cyp
// Created GetProblemIndexFromPointer(). Note that the function returns -1 if
// the pointer is not to an object managed by ProblemManager (ie was created
// from Benchmark etc).
//
// Revision 1.3  1998/11/06 03:55:01  cyp
// Fixed InitializeProblemManager(): was returning 1 problem more than it was
// being asked for.
//
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
return "@(#)$Id: probman.cpp,v 1.7 1998/12/14 11:47:36 cyp Exp $"; }
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
    probmanstatics.probtable[i]=new Problem(i);
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
  probmanstatics.probcount = probcount;
  return (int)(probmanstatics.probcount);
}    

// -----------------------------------------------------------------------

int DeinitializeProblemManager(void)
{
  Problem **probtable = probmanstatics.probtable;

  if (probtable!= NULL)
    {
    for (;probmanstatics.probcount>0;probmanstatics.probcount--)
      {
      if (probtable[probmanstatics.probcount-1])
        delete probtable[probmanstatics.probcount-1];
      probtable[probmanstatics.probcount-1] = NULL;
      }
    free((void *)probtable);
    }

  probmanstatics.probcount = 0;
  probmanstatics.tablesize = 0;
  probmanstatics.probtable = NULL;
  return 0;
}

// -----------------------------------------------------------------------
