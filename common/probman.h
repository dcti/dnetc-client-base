/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __PROBMAN_H__
#define __PROBMAN_H__ "@(#)$Id: probman.h,v 1.4.2.1 1999/04/13 19:45:28 jlawson Exp $"

//Return a specific Problem object or NULL if that problem doesn't exist
Problem *GetProblemPointerFromIndex(unsigned int probindex);

//Return the index (>=0) of a specific Problem object or -1 if not found.
//Note: Problems created from Benchmark etc are not managed by ProblemManager
int GetProblemIndexFromPointer( Problem *prob );

//Initialize the problem manager
int InitializeProblemManager(unsigned int maxnumproblems);

//Deinitialize the problem manager
int DeinitializeProblemManager(void);

#endif /* __PROBMAN_H__ */
