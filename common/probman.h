// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probman.h,v $
// Revision 1.1  1998/09/28 02:36:39  cyp
// Created. Just stubs for now.
//
// 

#ifndef __PROBMAN_H__
#define __PROBMAN_H__

//Return a specific Problem object or NULL if that problem doesn't exist
Problem *GetProblemPointerFromIndex(unsigned int probindex);

//Initialize the problem manager
int InitializeProblemManager(void);

//Deinitialize the problem manager
int DeinitializeProblemManager(void);

#endif /* __PROBMAN_H__ */
