// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
//
//
// $Log: pathwork.h,v $
// Revision 1.1  1998/07/05 13:09:09  cyruspatel
// Created new pathwork.cpp which contains functions for determining/setting
// the "work directory" and pathifying a filename that has no dirspec.
// GetFullPathForFilename() is ideally suited for use in (or just prior to) a
// call to fopen(). This obviates the neccessity to pre-parse filenames or
// maintain separate filename buffers. In addition, each platform has its own
// code section to avoid cross-platform assumptions. More doc in pathwork.cpp
// #define DONT_USE_PATHWORK if you don't want to use these functions.
//
//

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
