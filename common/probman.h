/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __PROBMAN_H__
#define __PROBMAN_H__ "@(#)$Id: probman.h,v 1.11 2008/12/30 20:58:42 andreasb Exp $"

//Return a specific Problem object or NULL if that problem doesn't exist
Problem *GetProblemPointerFromIndex(unsigned int probindex);

//Return the index (>=0) of a specific Problem object or -1 if not found.
//Note: Problems created from Benchmark etc are not managed by ProblemManager
int GetProblemIndexFromPointer( Problem *prob );

//Initialize the problem manager
int InitializeProblemManager(unsigned int maxnumproblems);

//Deinitialize the problem manager
int DeinitializeProblemManager(void);

// returns the number of problems managed by ProblemManager
unsigned int GetManagedProblemCount(void);

#endif /* __PROBMAN_H__ */
