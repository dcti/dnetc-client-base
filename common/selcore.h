// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: selcore.h,v $
// Revision 1.2  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/08/21 23:24:50  cyruspatel
// Created
//
//
// 

#ifndef __SELCORE_H__
#define __SELCORE_H__

//returns name for core number (0...) or "" if no such core
const char *GetCoreNameFromCoreType( unsigned int coretype ); 

#endif //__SELCORE_H__
