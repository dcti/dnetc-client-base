/* Hey, Emacs, this a -*-C++-*- file !
**
** memfile.c created 1998/08/16 Cyrus Patel <cyp@fb14.uni-mainz.de>
**
** This code is POSIXly correct. Please keep it that way.
**
** This is a posix stream interface to memory, standard FILE functions are
** emulated. Multiple 'files' can be open simultaneously. The size of a 'file' 
** is limited only by available memory. The following functions are available: 
** mfopen(), mfread(), mfwrite(), mfeof(), mftell(), mfseek(), mrewind(), 
** mflush[all](), mfclose[all](), mfileno(), mfilelength(), mftruncate(), 
** mdup(). The 'stream' object used/created by these functions is a MEMFILE *. 
**
*/
#ifndef __MEMFILE_H__
#define __MEMFILE_H__ "@(#)$Id: memfile.h,v 1.2 1999/04/05 17:56:52 cyp Exp $"

#include <stdio.h>   /* required for the size_t typedef */

struct memfilebuff
{
  struct memfilebuff *next;
  unsigned long blocksize;
  unsigned long usedsize;
  char *buffer;
};

typedef struct       /* mem stream structure  */
{
  int fd;
  int mode;
  int ateof;
  char fname[128];
  void *next;
  unsigned int opencount;
  unsigned long offset;
  unsigned long length;
  size_t grow_increment;
  struct memfilebuff *mbuff;
} MEMFILE;


extern size_t  mfread( void *buf, size_t ecount, size_t esize, MEMFILE *mp );
extern size_t  mfwrite( void *buf, size_t ecount, size_t esize, MEMFILE *mp );
extern int     mftruncate( int mfd, unsigned long newlen );
extern long    mfilelength( int mfd );
extern int     mfileno( MEMFILE *mp );
extern int     mfeof( MEMFILE *mp );
extern long    mftell( MEMFILE *mp );
extern int     mfseek( MEMFILE *mp, long offset, int whence );
extern void    mrewind( MEMFILE *mp );
extern int     mfflushall(void);
extern int     mfflush( MEMFILE *mp );
extern int     mfcloseall(void);
extern int     mfclose( MEMFILE *mp );
extern int     mdup( int mfd );
extern MEMFILE *mfopen( const char *name, const char *cmode );
extern size_t  _mfsetgrowincrement( MEMFILE *mp, size_t growby );

#endif /* __MEMFILE_H__ */

