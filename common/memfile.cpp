/*
** memfile.c created 1998/08/16 Cyrus Patel <cyp@fb14.uni-mainz.de>
**
** This code is POSIXly correct. Please keep it that way.
*/
/*
  This is a posix stream interface to memory, standard FILE functions are
  emulated. Multiple 'files' can be open simultaneously. The size of a 'file' 
  is limited only by available memory. The following functions are available: 
  mfopen(), mfread(), mfwrite(), mfeof(), mftell(), mfseek(), mrewind(), 
  mflush[all](), mfclose[all](), mfileno(), mfilelength(), mftruncate(), 
  mdup(). The stream object used/created by these functions is a MEMFILE *. 
*/
/*
** $Log: memfile.cpp,v $
** Revision 1.2  1998/10/03 05:34:28  sampo
** comment out <malloc.h> for MacOS builds.
**
** Revision 1.1  1998/08/20 19:21:50  cyruspatel
** Created
**
**
**
*/

#include <stdio.h>
#include <string.h>
#if (CLIENT_OS != OS_MACOS)
#include <malloc.h>
#endif
#include "memfile.h" /* thats us */

/* ====================================================================== */

#if !defined(lint)
const char *memfile_c(void) {
return "@(#)$Id: memfile.cpp,v 1.2 1998/10/03 05:34:28 sampo Exp $"; }
#endif

/* #define TEST */

/* ====================================================================== */

#ifndef __MEMFILE_H__

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

#endif

/* ====================================================================== */

static MEMFILE *__memstreambase = NULL;

/* ---------------------------------------------------------------------- */

static MEMFILE *__find_mp( MEMFILE *mp, MEMFILE **prev )
{
  MEMFILE *last, *here;
  
  if (mp) 
    {
    last = NULL;
    here = __memstreambase;
    while (here && mp!=here)
      {
      last = here;
      here = (MEMFILE *)here->next;
      }
    if (here == mp)
      {
      if (prev) 
        *prev = last;
      return (mp);
      }
    }
  return (NULL);
}  

/* ---------------------------------------------------------------------- */
  
int mfclose( MEMFILE *mp )
{
  MEMFILE *prev;
  struct memfilebuff *mbuff;

  if ( __find_mp( mp, &prev ) == NULL )
    return -1;

  if (mp->opencount>1)
    mp->opencount--;
  else
    {
    if (prev) 
      prev->next = mp->next;
    else 
      __memstreambase = (MEMFILE *)mp->next;
    while( mp->mbuff )
      {
      mbuff = mp->mbuff->next;
      free( (void *)mp->mbuff );
      mp->mbuff = mbuff;
      }
    free( (void *)mp );
    }
  return 0;
}  

/* ---------------------------------------------------------------------- */

int mfcloseall(void)
{
  while ( __memstreambase )
    mfclose( __memstreambase );
  return 0;
}    

/* ---------------------------------------------------------------------- */

int mfflushall(void)
{ return 0; }  

int mfflush( MEMFILE *mp )
{
  if ( !mp ) /* flushall */
    return 0;
  if ( __find_mp( mp, NULL ) == NULL )
    return -1;
  return 0;
}

/* ---------------------------------------------------------------------- */

long mftell( MEMFILE *mp )
{
  if ( __find_mp( mp, NULL ) == NULL )
    return (long)(-1L);
  return ((long)(mp->offset));
}  

/* ---------------------------------------------------------------------- */

int mfeof( MEMFILE *mp )
{
  if ( __find_mp( mp, NULL ) == NULL )
    return -1;
  return (mp->ateof != 0);
}  

/* ---------------------------------------------------------------------- */

int mfileno( MEMFILE *mp )
{
  if ( __find_mp( mp, NULL ) == NULL )
    return -1;
  return mp->fd;
}

/* ---------------------------------------------------------------------- */

#define MFO_WRONLY 0x0001
#define MFO_RDONLY 0x0002
#define MFO_RDWR   (MFO_WRONLY|MFO_RDONLY)
#define MFO_APPEND 0x0010
#define MFO_CREAT  0x0100  
#define MFO_TRUNC  0x0200
#define MFO_TEXT   0x1000 /* posixly correct. ignored, everything is binary */
#define MFO_BINARY 0x2000 /* posixly correct. ignored, everything is binary */

/* ---------------------------------------------------------------------- */

static int __cmode2umode( const char *cmode, int *umode )
{
  int oflags, aflags;

  if ( cmode[0]=='a')
    {
    oflags = MFO_WRONLY;
    aflags = (MFO_APPEND | MFO_CREAT);
    }
  else if (cmode[0]=='w')
    {
    oflags = ((cmode[1]=='+' || cmode[2]=='+')?(MFO_RDWR):(MFO_WRONLY));
    aflags = (MFO_CREAT | MFO_TRUNC);
    }
  else if (cmode[0]=='r')
    {
    oflags = ((cmode[1]=='+' || cmode[2]=='+')?(MFO_RDWR):(MFO_RDONLY));
    aflags = 0;
    }
  else
    return -1;
  
  aflags |= (( cmode[1]=='b' || cmode[2]=='b' )?(MFO_BINARY):(MFO_TEXT));
  *umode = (aflags | oflags);
  return 0;
}    

/* ---------------------------------------------------------------------- */

MEMFILE *mfopen( const char *filename, const char *cmode )
{
  int fd, mode;
  struct memfilebuff *mbuff;
  MEMFILE *mp, *prev, *tmp;

  if (!filename || !cmode)
    return NULL;

  if ( __cmode2umode( cmode, &mode ) != 0)
    return NULL;

  tmp = prev = NULL;    
  mp = __memstreambase;
  while (mp && strcmp(mp->fname, filename)!=0)
    {
    if (mp->mode == 0) /* unused */
      tmp = mp;
    prev = mp;
    mp = (MEMFILE *)mp->next;
    }

  if (!mp)
    {
    if (( mode & (MFO_CREAT | MFO_TRUNC) )==0)
      return NULL;
    mp = tmp;
    if ( mp )
      fd = mp->fd;
    else 
      {
      if ( ( mp = (MEMFILE *)malloc( sizeof( MEMFILE ) ) ) == NULL )
        return NULL;
      if (!prev)
        {
        fd = (int)0x4001;
        __memstreambase = mp;
        }
      else
        {
        prev->next = (void *)mp;
        fd = 1 + prev->fd;
        }
      }
    memset( mp, 0, sizeof( MEMFILE ));
    mp->mode = mode;
    strncpy( mp->fname, filename, sizeof(mp->fname)-1 );
    mp->fd = fd;
    }

  mp->opencount++;
  mp->offset = 0;
  
  if ((mp->mode & MFO_TRUNC) != 0)
    {
    while( mp->mbuff )
      {
      mbuff = mp->mbuff->next;
      free( (void *)mp->mbuff );
      mp->mbuff = mbuff;
      }
    }

  return mp;
}  

/* ---------------------------------------------------------------------- */

int mdup( int mfd )
{ 
  MEMFILE *mp;
  
  mp = __memstreambase;
  while ( mp && mp->fd != mfd )
    mp = (MEMFILE *)mp->next;
  
  if ( mp )
    return -1;

  mp->opencount++;
  return ( mp->fd );
}  

/* ---------------------------------------------------------------------- */

int mfseek( MEMFILE *mp, long offset, int whence )
{
  if ( __find_mp( mp, NULL ) == NULL )
    return -1;

  mp->ateof = 0;
  if ((mp->mode & MFO_APPEND)!=0) 
    return -1;

  if (whence == SEEK_CUR)
    offset += (long)(mp->offset);
  else if (whence == SEEK_END)
    offset += (long)(mp->length);
  
  if ((offset < 0) || (((unsigned long)(offset)) > mp->length))
    return -1;
  
  mp->offset = (unsigned long)(offset);
  return 0;
}        

void mrewind( MEMFILE *mp ) { mfseek( mp, 0, SEEK_SET ); }

/* ---------------------------------------------------------------------- */

int mftruncate( int mfd, unsigned long newlen )
{
  MEMFILE *mp;
  struct memfilebuff *mbuff, *prev, *next;
  unsigned long origin, offset, blocksize;
  
  mp = __memstreambase;
  while ( mp && mp->fd != mfd )
    mp = (MEMFILE *)mp->next;
  
  if ( !mp )
    return -1;

  if (( newlen > mp->length ) || ( mp->mode & MFO_WRONLY ) == 0)
    return -1;

  mp->ateof = 0;

  if ( newlen == mp->length ) 
    return 0;

  mbuff = mp->mbuff;
  prev = NULL;
  origin = 0;
  offset = newlen;
  mp->length = 0;
  
  while ( mbuff )
    {  
    next = mbuff->next;
    blocksize = mbuff->blocksize;
    if ((offset >=origin) && (offset <(origin+( mbuff->usedsize))))
      {
      mbuff->usedsize = ((offset)-origin);
      offset = origin + mbuff->blocksize;
      if ( mbuff->usedsize != 0 )
        mp->length = origin + mbuff->usedsize;
      else
        {
        if ( !prev )
          mp->mbuff = mbuff->next;
        else
          prev->next = mbuff->next;
        free( mbuff );
        mbuff = prev;
        }
      }
    origin += blocksize;
    prev = mbuff;
    mbuff = next;
    }

  if (mp->offset > mp->length)
    mp->offset = mp->length;
  
  return 0;
}  

/* ---------------------------------------------------------------------- */

long int mfilelength( int mfd )
{ 
  MEMFILE *mp;
  
  mp = __memstreambase;
  while ( mp && mp->fd != mfd )
    mp = (MEMFILE *)mp->next;
  
  if ( !mp )
    return (long int)(-1L);

  return (long int)mp->length;
}  

/* ---------------------------------------------------------------------- */

#define MAXIMUM_GROW_INCREMENT (8192-(sizeof(struct memfilebuff)+16)) 
#define DEFAULT_GROW_INCREMENT (1024-(sizeof(struct memfilebuff)+16)) 
#define MINIMUM_GROW_INCREMENT (   2*(sizeof(struct memfilebuff)+16)) 
                      /* 16 bytes to compensate for malloc overhead */

size_t _mfsetgrowincrement( MEMFILE *mp, size_t growincrement )
{
  size_t old_grow;
  if ( __find_mp( mp, NULL ) == NULL )
    return 0;
  old_grow = mp->grow_increment;
  if ( old_grow == 0 )
    old_grow = DEFAULT_GROW_INCREMENT;
  if ( growincrement != 0 )
    {
    if ( growincrement < MINIMUM_GROW_INCREMENT )
      growincrement = MINIMUM_GROW_INCREMENT;
    if ( growincrement > MAXIMUM_GROW_INCREMENT )
      growincrement = MAXIMUM_GROW_INCREMENT;
    mp->grow_increment = growincrement;
    }
  return old_grow;
}  

/* ---------------------------------------------------------------------- */

size_t mfwrite( void *buffer, size_t elemcount, size_t elemsize, MEMFILE *mp)
{
  char *charbuff;
  size_t grow_by;
  struct memfilebuff *mbuff, *prev;
  unsigned long dosize = (unsigned long)(elemcount)*elemsize;
  unsigned long pos, total, origin, remain;

  if ( __find_mp( mp, NULL ) == NULL )
    return 0;
  mp->ateof = (mp->offset >= mp->length);
  if (!buffer || dosize == 0 || ( mp->mode & MFO_WRONLY ) == 0)
    return 0;

  mbuff = mp->mbuff;
  prev  = NULL;
  origin = total = 0;

  while ( dosize > 0 )
    {  
    if ( mbuff == NULL )
      {
      if (mp->grow_increment == 0)
        mp->grow_increment = DEFAULT_GROW_INCREMENT;
      grow_by = mp->grow_increment;
      do{
        mbuff = (struct memfilebuff *)
                   ( malloc( grow_by + sizeof(struct memfilebuff) ));
        if ( !mbuff )
          {
          if ( grow_by < (MINIMUM_GROW_INCREMENT * 2))
            break;
          grow_by -= MINIMUM_GROW_INCREMENT;
          }
        } while ( !mbuff );
      if (!mbuff)
        break;
      mbuff->next = NULL;
      mbuff->usedsize = 0;
      mbuff->blocksize = grow_by;
      mbuff->buffer = ((char *)((void *)(mbuff)))+sizeof(struct memfilebuff);
      if ( prev )
        prev->next = mbuff;
      else
        mp->mbuff = mbuff;
      }
    if ((mp->offset>=origin) && (mp->offset<(origin+( mbuff->blocksize))))
      {
      pos = (mp->offset)-origin;
      remain = (mbuff->blocksize)-pos;
      if (dosize < remain) 
        remain = dosize;
      memcpy( (void *)((mbuff->buffer)+pos), buffer, remain );
      total += remain;
      mp->offset += remain;
      mbuff->usedsize = pos+remain;
      charbuff = ((char *)(buffer))+remain;
      buffer = ((void *)(charbuff));
      dosize -= remain;
      }
    prev = mbuff;
    origin += mbuff->blocksize;
    mbuff = mbuff->next;
    }

  mbuff = mp->mbuff;
  mp->length = 0;
  while ( mbuff )
    {
    mp->length += mbuff->usedsize;
    mbuff = mbuff->next;
    }

  mp->ateof = (mp->offset >= mp->length);
  return ( total / elemsize );
}    

/* ---------------------------------------------------------------------- */

size_t mfread( void *buffer, size_t elemcount, size_t elemsize, MEMFILE *mp )
{
  char *charbuf;
  struct memfilebuff *mbuff;
  unsigned long dosize = (unsigned long)(elemcount)*elemsize;
  unsigned long pos, total, remain, origin;

  if ( __find_mp( mp, NULL ) == NULL )
    return 0;
  mp->ateof = (mp->offset >= mp->length);
  if ( !buffer || dosize == 0 || mp->mbuff == NULL || ( mp->ateof ) )
    return 0;
  if (( mp->mode & MFO_RDONLY ) == 0)
    return 0;

  mbuff = mp->mbuff;
  total = origin = 0;

  while ( dosize > 0 )
    {  
    if (mbuff == NULL)
      break;
    if ((mp->offset>=origin) && (mp->offset<(origin+( mbuff->usedsize))))
      {
      pos = (mp->offset)-origin;
      remain = (mbuff->usedsize)-pos;
      if (dosize < remain) 
        remain = dosize;
      memcpy( buffer, (void *)((mbuff->buffer)+pos), remain );
      total +=  remain;
      mp->offset += remain;
      if (mp->offset >= mp->length)
        break;
      dosize -= remain;
      charbuf = ((char *)(buffer))+remain;
      buffer = (void *)(charbuf);
      }
    origin += mbuff->blocksize;
    mbuff = mbuff->next;
    }

  mp->ateof = (mp->offset >= mp->length);
  return (size_t)( total / elemsize );
}

/* ---------------------------------------------------------------------- */

#ifdef TEST

unsigned int count_open_streams(void)
{
  unsigned int count = 0;
  MEMFILE *mp = __memstreambase;
  while ( mp )
    {
    count++;
    mp = (MEMFILE *)mp->next;
    }
  return count;
}  
  

int main(void)
{
  MEMFILE *file, *file2, *file3;
  char test[] = "This is a very long string that just goes on and on, but it can surely tell us a few things about how this code is working.";
  char test2[] = "This is the second string, and should get cat'd right up behind the first string.";
  char buffer[512];              

  printf("\n--------- open should fail (file doesn't exist)\n");
  file = mfopen( "dummy", "r+b" );
  printf("fopen(\"dummy\",\"r+b\") -> 0x%x\n", file );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- open should succeed (truncated on open)\n");
  file = mfopen( "dummy", "w+b" );
  printf("fopen(\"dummy\",\"w+b\") -> 0x%x\n", file );
  printf("#of open memstreams: %d\n", count_open_streams() );
  printf("growth increment was %d\n", _mfsetgrowincrement( file, 64 ) );
  printf("growth increment is now %d\n", _mfsetgrowincrement( file, 0 ) );

  printf("\n--------- should be zero offset, and eof == false\n");
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- should write %d bytes\n", strlen(test));
  printf( "fwrite( [], %d, %d, file )  -> %d\n", strlen(test), sizeof(char),
                        mfwrite( test, strlen(test), sizeof(char), file ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- should read the string back in\n");
  printf( "rewind(file) \n" ); mrewind(file);
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );
  memset( buffer, 0, sizeof( buffer ));
  printf( "fread( [], %d, %d, file )  -> %d\n", sizeof(buffer), sizeof(char), 
                       mfread( buffer, sizeof(buffer), sizeof(char), file ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- should cat the second string to the first\n");
  printf( "fwrite( [], %d, %d, file )  -> %d\n", sizeof(char), strlen(test2), 
                        mfwrite( test2, strlen(test2), sizeof(char), file ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- should read zero (fp is at eof)\n");
  memset( buffer, 0, sizeof( buffer ));
  printf( "fread( [], %d, %d, file )  -> %d\n", sizeof(buffer), sizeof(char), 
                       mfread( buffer, sizeof(buffer), sizeof(char), file ) );
  printf( "ftell(file) -> %ld  feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- seek test\n");
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf( "seek(file,64,0)-> %d\n", mfseek( file, 64, SEEK_SET ) );
  printf( "ftell(file) -> %ld  feof(file)-> %d\n", mftell( file ), mfeof( file ) );
  
  printf("\n--------- read after seek test, should read filelen-64 bytes\n");
  memset( buffer, 0, sizeof( buffer ));
  printf( "fread( [], %d, %d, file )  -> %d\n", sizeof(buffer), sizeof(char), 
                       mfread( buffer, sizeof(buffer), sizeof(char), file ) );
  printf( "-> \"%s\"\n", buffer );
  
  printf("\n--------- change size test\n");
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf( "ftell(file) -> %ld  feof(file)-> %d\n", mftell( file ), mfeof( file ) );
  printf( "ftruncate(fileno(file),10)   ->  %d\n", mftruncate( mfileno( file ), 10 ) );
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf( "ftell(file) -> %ld  feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- negative seek test\n");
  printf( "fseek(file,-5,2)->%d\n", mfseek( file, -5, SEEK_END ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );

  printf("\n--------- read after seek test, should read 5 bytes\n");
  memset( buffer, 0, sizeof( buffer ));
  printf( "fread( [], %d, %d, file )  -> %d\n", sizeof(buffer), sizeof(char), 
                       mfread( buffer, sizeof(buffer), sizeof(char), file ) );
  printf( "-> \"%s\"\n", buffer );

  printf("\n--------- open and write a second file\n");
  file2 = mfopen( "dummy2", "w+b" );
  printf("fopen(\"dummy2\",\"w+b\") -> 0x%x\n", file2 );
  printf( "fwrite( [], %d, %d, file2 )  -> %d\n", strlen("TEST"), sizeof(char),
                        mfwrite( "TEST", strlen("TEST"), sizeof(char), file2 ) );
  printf( "filelength( fileno( file2 ) )-> %ld\n", mfilelength( mfileno( file2 ) ) );
  printf( "ftell(file2) -> %ld feof(file2)-> %d\n", mftell( file2 ), mfeof( file2 ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- open and write a third file\n");
  file3 = mfopen( "dummy3", "w+b" );
  printf("fopen(\"dummy3\",\"w+b\") -> 0x%x\n", file3 );
  printf( "fwrite( [], %d, %d, file3 )  -> %d\n", strlen("BLAHBLAH"), sizeof(char),
            mfwrite( "BLAHBLAH", strlen("BLAHBLAH"), sizeof(char), file3 ) );
  printf( "filelength( fileno( file3 ) )-> %ld\n", mfilelength( mfileno( file3 ) ) );
  printf( "ftell(file3) -> %ld feof(file3)-> %d\n", mftell( file3 ), mfeof( file3 ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- close second\n");
  printf( "fclose(file2)    -> %d\n", mfclose( file2 ) );
  printf( "filelength( fileno( file2 ) )-> %ld\n", mfilelength( mfileno( file2 ) ) );
  printf( "ftell(file2) -> %ld feof(file2)-> %d\n", mftell( file2 ), mfeof( file2 ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- check first\n");
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );
  printf("#of open memstreams: %d\n", count_open_streams() );
  
  printf("\n--------- truncate to zero test\n");
  printf( "ftruncate(fileno(file),0)   ->  %d\n", mftruncate( mfileno( file ), 0 ) );
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- close first\n");
  printf( "fclose(file)    -> %d\n", mfclose( file ) );
  printf( "filelength( fileno( file ) )-> %ld\n", mfilelength( mfileno( file ) ) );
  printf( "ftell(file) -> %ld feof(file)-> %d\n", mftell( file ), mfeof( file ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  printf("\n--------- close third\n");
  printf( "fclose(file3)    -> %d\n", mfclose( file3 ) );
  printf( "filelength( fileno( file3 ) )-> %ld\n", mfilelength( mfileno( file3 ) ) );
  printf( "ftell(file3) -> %ld feof(file3)-> %d\n", mftell( file3 ), mfeof( file3 ) );
  printf("#of open memstreams: %d\n", count_open_streams() );

  return 0;
}
#endif
