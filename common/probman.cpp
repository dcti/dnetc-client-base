// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probman.cpp,v $
// Revision 1.1  1998/09/28 02:36:33  cyp
// Created. Just stubs for now.
//
// 

#if (!defined(lint) && defined(__showids__))
const char *probman_cpp(void) {
return "@(#)$Id: probman.cpp,v 1.1 1998/09/28 02:36:33 cyp Exp $"; }
#endif

#include "cputypes.h"  // MAX!CPUS
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "probman.h"   // thats us. Keep prototypes in sync.

#ifdef MAXCPUS         // we don't use this anymore
#undef MAXCPUS         
#endif

#define MAXPROBLEMS 32+2 // still in the skeleton phase
                         // (added +2 to show that we can go higher)

// -----------------------------------------------------------------------

static Problem problem[ MAXPROBLEMS ];
static int initialized;

// -----------------------------------------------------------------------

Problem *GetProblemPointerFromIndex(unsigned int probindex)
{
  //if (initialized)
    {
    if (probindex < (sizeof(problem)/sizeof(problem[0])))
      return &(problem[probindex]);
    }
  return NULL;
}  

// -----------------------------------------------------------------------

int InitializeProblemManager(void)
{
  initialized = 1;
  return 0;
}

// -----------------------------------------------------------------------

int DeinitializeProblemManager(void)
{
  initialized = 0;
  return 0;
}

// -----------------------------------------------------------------------
