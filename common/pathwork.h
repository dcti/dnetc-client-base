/* Hey, Emacs, this is a -*-C++-*- file !
 *
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PATHWORK_H__
#define __PATHWORK_H__ "@(#)$Id: pathwork.h,v 1.6 1999/04/04 16:28:04 cyp Exp $"

/* -------------------------------------------------------------------
 * Get the working directory previously initialized with 
 * InitWorkingDirectoryFromSamplePaths(). returns NULL if buffer is too
 * small otherwise returns buffer. The directory returned can be 
 * strcat()ed with a filename (ie it always ends with '/' or whatever)
 * ------------------------------------------------------------------ */
const char *GetWorkingDirectory( char *buffer, unsigned int maxlen );


/* -------------------------------------------------------------------
 * the working directory is the app's directory, unless the ini filename
 * contains a dirspec, in which case the path to the ini file is used.
 * Exception to the rule: win32/win16 which always use the app's directory.
 *------------------------------------------------------------------- */
int InitWorkingDirectoryFromSamplePaths( const char *inipath, const char *apppath );


/* -------------------------------------------------------------------
 * prepends the working directory to 'filename' if the filename is
 * not already pathified. Returns a pointer to the pathified filename
 * or to a zero length string on error (which will cause fopen() to fail)
 *------------------------------------------------------------------- */
const char *GetFullPathForFilename( const char *filename );


/* -------------------------------------------------------------------
 * prepends the directory 'dir' to 'fname' if the filename is
 * not already pathified. Returns a pointer to the pathified filename
 * or to a zero length string on error (which will cause fopen() to fail)
 * Equivalent to GetFullPathForFilename() if 'dir' is NULL.
 *------------------------------------------------------------------- */
const char *GetFullPathForFilenameAndDir( const char *fname, const char *dir );


/* -------------------------------------------------------------------
 * get the offset of the filename component in 'fullpath' 
 *------------------------------------------------------------------- */
unsigned int GetFilenameBaseOffset( const char *fullpath );


/* ------------------------------------------------------------------
 * Suffix separator (called an 'extension' in the windos world)
 * ------------------------------------------------------------------ */

#if (CLIENT_OS == OS_RISCOS)
  #define EXTN_SEP   "/"
#else
  #define EXTN_SEP   "."
#endif

#endif /* __PATHWORK_H__ */

