/* 
 * This module consists of functions required by plib[mt]3s.lib when 
 * building C++ apps which (theoretically) requires linking Watcom's 
 * static CLIB. 
 * 
 * The latter is overloaded with redundancy, inefficiency, outdated apis,
 * in no way makes an application clib independant and essentially just
 * adds bulk.
 *
 * Written December 1997 Cyrus Patel <cyp@fb14.uni-mainz.de>
*/   
#if defined(__showids__)
const char *nwwatemu_c(void) {
return "@(#)$Id: nwwatemu.c,v 1.1.2.1 2001/01/21 15:10:31 cyp Exp $"; }
#endif

#ifdef __cplusplus
  #error Error: This needs to be a 'C' not 'CPP' file.
#endif   

/* ************************* cvtbuf ******************************* */

void *__CVTBuffer( void )      /* used for float/double conversions */
{                              /* Watcom CLIB mallocs 1 buf per "thread" */
  static unsigned char cvtbuf[64];
  return (void *)(&cvtbuf[0]);
}  

/* ************************* fltused ******************************* */

unsigned int _fltused_=0;  /* are floating point routines linked? */
unsigned int __ppfltused_=0;
/* ************************* old8087 ******************************* */

void clearx87(void);
#pragma aux clearx87 = "fldz" "fldz" "fldz" "fldz";
void __old_8087(void) { clearx87(); }  /* Clear the x87 stack. */

/* ****************** misc error handling routines **************** */

extern void ExitThread(int actioncode, int terminationcode);
extern void ConsolePrintf(char *s, ... );

int __EnterWVIDEO(int wdstate) /* returns zero if entered debugger */
         { wdstate=0; return wdstate; }
void __exit_with_msg( char *msg, int exitcode )
         { ConsolePrintf( msg ); ExitThread( 0 /* EXIT_THREAD*/, exitcode ); }
void __fatal_runtime_error( int wdstate, char *msg, int exitcode )
         { if (__EnterWVIDEO(wdstate)) __exit_with_msg( msg, exitcode ); }

/* ****************** watcom's prelude substitutions ************ */

void __WATCOM_Prelude() { return; }

#ifdef UNUSED

extern free( void *__memptr );
extern void *malloc( unsigned int __size );

extern memcpy( void *__dstptr, void *__srcptr, unsigned int __count );
extern memset( void *__memptr, int __ch, unsigned int __count );

//void memset( void *__memptr, int __ch, unsigned int __count )
//{ unsigned int i; char *mem=__memptr; for (i=0;i<__count;i++) *mem++=__ch; }

void __NullRtn() { return; }
void __NullAccessRtn() { return; }
int  __NullInitAccessRtn(int x) { return x; }

void *_AccessIOB     = (void *)(__NullRtn);
void *_ReleaseIOB    = (void *)(__NullRtn);
void *_AccessFileH   = (void *)(__NullAccessRtn);
void *_ReleaseFileH  = (void *)(__NullAccessRtn);
void *_FiniAccessH   = (void *)(__NullAccessRtn);
void *_AccessTDList  = (void *)(__NullRtn);
void *_ReleaseTDList = (void *)(__NullRtn);

void *_NW_free( void *__memptr ) { free(__memptr); return (void *)(0); }

void *_NW_malloc( unsigned int __size ) { return malloc( __size ); }
void *_NW_calloc( unsigned int __items, unsigned int __itemsz ) 
{ 
  unsigned int totalsz;
  void *amem; 
  if ( ( amem = malloc( totalsz = (__items * __itemsz ) ) ) != (void *)(0) )
    memset( amem, 0, totalsz );
  return amem;
}

void *_NW_realloc(void *__oldptr,unsigned int __newsize,unsigned int __oldsize)
{
  void *newptr = (void *)0;

  if (!__oldptr)
    newptr = malloc( __newsize ); 
  else if (!__newsize)
    free( __oldptr ); 
  else if (__newsize <= __oldsize)
    newptr = __oldptr;
  else if ((newptr = malloc( __newsize )) != (void *)(0))
  {
    if (__newsize < __oldsize )
      __oldsize = __newsize;
    memcpy ( newptr, __oldptr, __oldsize );
    free( __oldptr );
  }
  return newptr;
}  

#endif
