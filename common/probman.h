// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probman.h,v $
// Revision 1.4  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.3  1998/11/12 18:50:59  cyp
// Created GetProblemIndexFromPointer(). Note that the function returns -1 if
// the pointer is not to an object managed by ProblemManager (ie was created
// from Benchmark etc).
//
// Revision 1.2  1998/11/06 02:32:26  cyp
// Ok, no more restrictions (at least from the client's perspective) on the
// number of processors that the client can run on.
//
// Revision 1.1  1998/09/28 02:36:39  cyp
// Created. Just stubs for now.
//
// 

#ifndef __PROBMAN_H__
#define __PROBMAN_H__

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
