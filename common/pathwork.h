// Hey, Emacs, this is a -*-C++-*- file !

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: pathwork.h,v $
// Revision 1.2  1998/07/06 01:47:05  cyruspatel
// Added signature so that emacs recognizes this to be a c++ file.
//
// Revision 1.1  1998/07/05 13:09:09  cyruspatel
// Created - see pathwork.cpp for documentation 
//

#ifndef __PATHWORK_H__
#define __PATHWORK_H__

// -------------------------------------------------------------------
// Get the working directory previously initialized with 
// InitWorkingDirectoryFromSamplePaths(). returns NULL if buffer is too
// small otherwise returns buffer. The directory returned can be 
// strcat()ed with a filename (ie it always ends with '/' or whatever)

const char *GetWorkingDirectory( char *buffer, unsigned int maxlen );

// -------------------------------------------------------------------
// the working directory is the app's directory, unless the ini filename
// contains a dirspec, in which case the path to the ini file is used.
// Exception to the rule: win32/win16 which always use the app's directory.

int InitWorkingDirectoryFromSamplePaths( const char *inipath, const char *apppath );

// -------------------------------------------------------------------
// prepends the working directory to 'filename' if the filename is
// not already pathified. Returns a pointer to the pathified filename
// or to a zero length string on error (which will cause fopen() to fail)

const char *GetFullPathForFilename( const char *filename );

// -------------------------------------------------------------------

#endif //__PATHWORK_H__
