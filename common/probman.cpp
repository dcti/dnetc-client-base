/* Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/ 
const char *probman_cpp(void) {
return "@(#)$Id: probman.cpp,v 1.15 2000/06/02 06:24:58 jlawson Exp $"; }

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
    probmanstatics.probtable[i]=new Problem();
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

#if (CLIENT_OS == OS_FREEBSD)
#include <sys/mman.h>
int TBF_MakeProblemsVMInheritable(void)
{
  Problem **probtable = probmanstatics.probtable;
  unsigned int probcount  = probmanstatics.probcount;

  if (probtable != NULL && probcount != 0)
  {
    unsigned int i;  
    int failed = 0, mflag = 0; /*VM_INHERIT_SHARE*/ /*MAP_SHARED|MAP_INHERIT*/;

    for (i=0;(!failed && i<probcount);i++)
    {
      if (probtable[i]==NULL)
        break;
      failed = (minherit((void *)probtable[i],sizeof(Problem),mflag)!=0);
      //if (failed)
      //  fprintf(stderr,"probman_inherit:1:%u %s\n",i,strerror(errno));
    }
    if (!failed)
    {
      failed = (minherit((void *)probtable, 
          probmanstatics.tablesize * sizeof(Problem *), mflag)!=0);
      //if (failed)
      //  perror("probman_inherit:2");
    }
    if (!failed)
    {
      failed=(minherit((void*)&probmanstatics,sizeof(probmanstatics),mflag)!=0);
      //if (failed)
      //  perror("probman_inherit:3");
    }
    if (!failed)
      return 0;
  }   
  return -1;
}
#endif

// -----------------------------------------------------------------------
