//#define TRACE_STAT
/*
 * ANSI/POSIX functions not available, poorly implemented, or incompatible
 *                      between NetWare versions
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Includes:
 *   ftime(2)  [ANSI/SYS V] - doesn't exist on NetWare 
 *   daylight  [ANSI/SYS V] - compatibility for NetWare 4 and above
 *   timezone  [ANSI/SYS V] - totally broken on NetWare 4 and above
 *   usleep(2) [BSD 4.3]    - doesn't exist on NetWare 
 *   sleep(3)  [POSIX 1]    - doesn't exist on NetWare 
 *   stat(2)   [POSIX 1]    - portability for stat_XXX crap 
 *   fstat(2)  [POSIX 1]    - portability for fstat_XXX crap 
 *   chdir(3)  [POSIX 1]    - workaround to work with or without trailing slash
 *   access(2) [POSIX 1]    - w/a broken NCPGetEntryAttributes on 5.0-pre SP3 
 *   fopen(3)  [POSIX 1]    - workaround for exclusive open 
 *
 * $Id: nwlcomp.c,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
*/

/* ===================================================================== */

#define TRACE_OUT(x) ConsolePrintf x; ThreadSwitchLowPriority()
#ifdef TRACE_STAT
  #undef TRACE_STAT
  #define TRACE_STAT(x) TRACE_OUT(x)
#else
  #define TRACE_STAT(x) /* nothing */
#endif

/* ===================================================================== */

#include <time.h>    /* time_t and time(time_t *) */

#pragma pack(1)                         
struct timeb {                          /* from <sys/timeb.h> */
        time_t  time;                   /* seconds since the Epoch */
        unsigned short millitm;         /* + milliseconds since the Epoch */
        short   timezone;               /* minutes west of CUT */
        short   dstflag;                /* DST == non-zero */
};
#pragma pack()

#ifdef __cplusplus
extern "C" {
#endif
int ftime(struct timeb *tb);
extern unsigned long GetCurrentTicks(void); /* DON'T INCLUDE NWTIME.H */

extern int timezone; /* declared in nwtime.h as macros to functions */
extern int daylight; /* but they're only on 4.x and bad decls anyway */
                     /* eg timezone is declared as (unsigned) time_t! */

#if 0 /* Here is an extract of the screw ups: */
  #ifndef _TIME_T
  # define _TIME_T 
  typedef unsigned long time_t; /* BAD BAD BAD BAD BAD */
  #endif
  extern time_t timezone;       /* SEE WHY! SEE WHY! SEE WHY! :) */
  extern time_t *__get_timezone( void ); /* HERE TOO! HERE TOO! */
#endif

int     *__get_daylight( void ); /* We support these here just in ... */
time_t  *__get_timezone( void ); /* case <nwtime.h> got included somewhere */

#ifdef __cplusplus
}
#endif

int     *__get_daylight( void ) { return &daylight;  }
time_t  *__get_timezone( void ) { return ((time_t *)&timezone);  }

int ftime(struct timeb *tb)
{
  if (tb)
  {
    unsigned long cas[3];
    unsigned long usec,secs;

    /* GetClockStatus() is stubbed/emulated in nwlemu.c
       Contrary to documentation, the real GetClockStatus() 
       is *NOT* MP-clean (will force a migration to CPU0)
    */
    GetClockStatus(cas); /* possibly emulated */
  
    secs = (time_t)cas[0]; /* full secs */
    usec = cas[1]; /* frac secs */
  
    _asm mov eax, usec
    _asm xor edx, edx
    _asm mov ecx, 1000000
    _asm mul ecx
    _asm xor ecx, ecx
    _asm dec ecx
    _asm div ecx
    _asm mov usec, eax

    tb->time = (time_t)secs;                   /* seconds since the Epoch */
    tb->millitm = (unsigned short)(usec/1000); /* millisecs since the Epoch */
    tb->timezone = (short)(-(((int)timezone)/60)); /* minutes west of CUT */
    tb->dstflag = (daylight != 0);
  }   
  return 0;
}  

/* ===================================================================== */

#ifdef __cplusplus   /* avoid pulling in *any* non-standard headers */
extern "C" {
#endif
extern unsigned long GetCurrentTicks(void);
extern void ThreadSwitchLowPriority(void);
extern void delay(unsigned int);
#define DELAY delay
#define SCHED_YIELD ThreadSwitchLowPriority

unsigned int sleep(unsigned int);
void usleep(unsigned long);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* wrapper for utilization suppression support */
extern "C" void nwCliMillisecSleep(unsigned long millisecs);
#undef DELAY
#define DELAY nwCliMillisecSleep
#ifdef __cplusplus
}
#endif

static unsigned int _xsleep(register unsigned int secs, 
                            register unsigned long usecs )
{
  unsigned int remainder = 0;
  SCHED_YIELD();
  if ((secs*1000) < secs) /* wrap */
  {
    //errno = EINVAL;
    remainder = secs;
  }
  else if (secs == 0 && usecs < 1000)
  {  
    /* already done above */
  }
  else if (secs == 0 && usecs <= 55000)
  {
    DELAY(0);
  }
  else      
  {  
    unsigned int millisecs = ((secs*1000)+((usecs+500)/1000));
    unsigned long sleptfor, t1 = GetCurrentTicks();
    DELAY( millisecs );
    sleptfor = GetCurrentTicks();
    if (sleptfor < t1)
    {
      sleptfor += ~t1;
      t1 = 0;
    }
    if (( sleptfor = (((sleptfor - t1)*182)/10) ) < millisecs )
      remainder = (millisecs - sleptfor);
  }
  return remainder;
}  

void usleep(unsigned long usec)       { _xsleep(0,usec);return; }
unsigned int sleep(unsigned int secs) { return _xsleep(secs,0); }

/* ===================================================================== */

#ifdef __cplusplus
extern "C" {
#endif
#include <sys/stat.h>   /* struct stat, stat(), fstat() */
#include <process.h>    /* GetThreadGroupID(), GetNLMHandle() */
#include <stdlib.h>     /* malloc */
#include <string.h>     /* memset, memcpy */
#include <unistd.h>     /* chdir */
#include <errno.h>      /* errno */
#include <nwfile.h>     /* FE*(), pulls in nwnamspc.h */
/* #include <nwnamspc.h> CONFLICT WITH NWFILE!! *//* SetTargetNamespace */
#ifdef __cplusplus
}
#endif

#if 0
int chdir(const char *newdir)
{
  /* On NetWare 4.x 'newdir' is not allowed to have a trailing slash.
  ** On NetWare 3.x 'newdir' must have a trailing slash
  */
  int thrgrid = GetThreadGroupID(); /* ensure this before we do anything */
  int rc = -1, new_errno = EBADHNDL; /* (22) what clib returns if no context */
  if (thrgrid != 0 && thrgrid != -1) /* do we have CLIB context? */
  {
    if (!newdir)
      new_errno = EINVAL;
    else if (!*newdir)
      new_errno = EINVAL;
    else
    {
      int nlmHandle = GetNLMHandle();
      const char *symname = "\x05""chdir";
      int (*__chdir)(const char *) = 
            (int (*)(const char *))ImportPublicSymbol(nlmHandle,symname);

      if (!__chdir)
        new_errno = ENOSYS;
      else
      {
        if (((*__chdir)(newdir)) == 0)
        {
ConsolePrintf("chdir('%s') [1] worked\n", newdir);
          rc = 0;
        }
        else
        {
          unsigned int len = strlen(newdir);
          char *newdir_alt = (char *)malloc(len+5);
          if (!newdir_alt)
            new_errno = ENOMEM;
          else
          {
            strcpy(newdir_alt,newdir);
            if (newdir[len-1] == '\\' || newdir[len-1] == '/')
              newdir_alt[len-1] = '\0';
            else
            {
              newdir_alt[len+0] = '/'; 
              newdir_alt[len+1] = '\0';
            }
            new_errno = 0;
            if (((*__chdir)(newdir_alt)) == 0)
            {
ConsolePrintf("chdir('%s') [2] worked\n", newdir_alt);
              rc = 0;
            }
            free((char *)newdir_alt);
          }
        }
        UnimportPublicSymbol(nlmHandle, symname);
      } /* got symbol */
    } /* newdir && *newdir */
  }
  if (rc != 0 && new_errno)
    errno = new_errno;
  return rc;
}
#endif

/* ------------------------------------------------------------------- */

#pragma pack(1)
struct my_stat500
{
   dev_t          st_dev;        /* volume number                         */
   ino_t          st_ino;        /* directory entry number of the st_name */
   unsigned short st_mode;       /* emulated file mode                    */
   unsigned short st_pad1;       /* reserved for alignment                */
   unsigned long  st_nlink;      /* count of hard links (always 1)        */
   unsigned long  st_uid;        /* object id of owner                    */
   unsigned long  st_gid;        /* group-id (always 0)                   */
   dev_t          st_rdev;       /* device type (always 0)                */
   off_t          st_size;       /* total file size--files only           */
   time_t         st_atime;      /* last access date--files only          */
   time_t         st_mtime;      /* last modify date and time             */
   time_t         st_ctime;      /* POSIX: last status change time...     */
                                 /* ...NetWare: creation date/time        */
   time_t         st_btime;      /* last archived date and time           */
   unsigned long  st_attr;       /* file attributes                       */
   unsigned long  st_archivedID; /* user/object ID of last archive        */
   unsigned long  st_updatedID;  /* user/object ID of last update         */
   unsigned short st_inheritedRightsMask;  /* inherited rights mask       */
   unsigned short st_pad2;       /* reserved for alignment                */
   unsigned int   st_originatingNameSpace; /* namespace of creation       */
   size_t         st_blksize;    /* block size for allocation--files only */
   size_t         st_blocks;     /* count of blocks allocated to file     */
   unsigned int   st_flags;      /* user-defined flags                    */
   unsigned long  st_spare[4];   /* for future use                        */
   unsigned char  st_name[255+1];/* TARGET_NAMESPACE name                 */
};

struct my_stat411                   /* v4.11 */
{
   dev_t          st_dev;
   ino_t          st_ino;
   unsigned short st_mode;
   short          st_nlink;
   unsigned long  st_uid;
   short          st_gid;
   dev_t          st_rdev;
   off_t          st_size;
   time_t         st_atime;
   time_t         st_mtime;
   time_t         st_ctime;
   time_t         st_btime;
   unsigned long  st_attr;
   unsigned long  st_archivedID;
   unsigned long  st_updatedID;
   unsigned short st_inheritedRightsMask;
   unsigned char  st_originatingNameSpace;
   /*----------------- new fields starting in v4.11 ------------------------- */
   unsigned char  st_name[255+1];
   size_t         st_blksize;
   size_t         st_blocks;
   unsigned int   st_flags;
   unsigned long  st_spare[4];
};

struct my_stat410              /* v3.11, v3.12, v4.0, v4.01, v4.02 and v4.10 */
{
   dev_t          st_dev;
   ino_t          st_ino;
   unsigned short st_mode;
   short          st_nlink;
   unsigned long  st_uid;
   short          st_gid;
   dev_t          st_rdev;
   off_t          st_size;
   time_t         st_atime;
   time_t         st_mtime;
   time_t         st_ctime;
   time_t         st_btime;
   unsigned long  st_attr;
   unsigned long  st_archivedID;
   unsigned long  st_updatedID;
   unsigned short st_inheritedRightsMask;
   unsigned char  st_originatingNameSpace;
   unsigned char  st_name[13];
};
#pragma pack()


static int xx_stat(int xxtype, int handle, const char *filename, 
                   struct stat *statblk, size_t size_of_statstruct )
{
  int retcode = -1;
  int new_errno = EINVAL;
  
  if (xxtype == 'fd' && handle == -1)
  {
    new_errno = EBADF;
  }
  else if (statblk && ((xxtype == 'fd')?(handle != -1):(filename!=0)))
  {
    int thrgrid = GetThreadGroupID(); /* ensure this before we do anything */
    new_errno = EBADHNDL; /* (22) what clib returns if no context */

    if (thrgrid != 0 && thrgrid != -1) /* do we have CLIB context? */
    {
      int nlmHandle = GetNLMHandle();
      long nwversion = ((GetFileServerMajorVersionNumber() * 100)
                       +GetFileServerMinorVersionNumber());
      //int argsize = sizeof(struct stat);
      int (*_stat_vector)(const char *, void *);
      int (*_fstat_vector)(int, void *);
      unsigned int len; 
      const char *symname;
      new_errno = ENOSYS;

      TRACE_STAT(("size_of_statstruct=%d\r\n"
                   "sizeof(struct my_stat410)=%d\r\n"
                   "sizeof(struct my_stat411)=%d\r\n"
                   "sizeof(struct my_stat500)=%d\r\n"
                   "nwver=%d\r\n",
                   size_of_statstruct, 
                   sizeof(struct my_stat410), 
                   sizeof(struct my_stat411), 
                   sizeof(struct my_stat500), 
                   nwversion ));

      if (size_of_statstruct == sizeof(struct my_stat410) || /* is pre-411 */
         nwversion < 411)
      {
        symname = ((xxtype == 'fd')?("\x05""fstat"):("\x04""stat"));

        _stat_vector = (int (*)(const char *, void *))
             ImportPublicSymbol(nlmHandle, symname);
        _fstat_vector = (int (*)(int , void *))_stat_vector;
        
        TRACE_STAT(("1. fxn=%s => %08x\r\n", symname+1, _stat_vector ));
        if (_stat_vector)
        {
          struct my_stat410 s410;
          if (xxtype == 'fd')
            retcode = (*_fstat_vector)(handle, (void *)&s410);
          else
            retcode = (*_stat_vector)(filename, (void *)&s410);

          TRACE_STAT(("1. retcode=%d\r\n", retcode));            
          UnImportPublicSymbol(nlmHandle, symname );
          if (retcode != 0)
            new_errno = errno;
          else if (size_of_statstruct == sizeof(struct my_stat410))
            memcpy((void *)statblk, (void *)&s410, size_of_statstruct);
          else /* statblk is 411 or 500 format */
          {
            unsigned char oldnamespace;
            /* clear everything to zap 'pad' space */
            memset((void *)statblk, 0, size_of_statstruct);
            statblk->st_dev         = s410.st_dev;
            statblk->st_ino         = s410.st_ino;
            statblk->st_mode        = s410.st_mode;
            statblk->st_nlink       = s410.st_nlink;
            statblk->st_uid         = s410.st_uid; 
            statblk->st_gid         = s410.st_gid; 
            statblk->st_rdev        = s410.st_rdev;
            statblk->st_size        = s410.st_size;
            statblk->st_atime       = s410.st_atime;
            statblk->st_mtime       = s410.st_mtime;
            statblk->st_ctime       = s410.st_ctime;
            statblk->st_btime       = s410.st_btime;
            statblk->st_attr        = s410.st_attr;
            statblk->st_archivedID  = s410.st_archivedID;
            statblk->st_updatedID   = s410.st_updatedID;
            statblk->st_inheritedRightsMask = s410.st_inheritedRightsMask;
            statblk->st_originatingNameSpace = s410.st_originatingNameSpace;
            statblk->st_blksize     = 512;
            statblk->st_blocks      = (s410.st_size+511)/512;
            /* 255+1 from 13 */
            memcpy(statblk->st_name, s410.st_name, sizeof(s410.st_name));
            oldnamespace = SetTargetNameSpace(0);
            if (oldnamespace != 0 && oldnamespace != 255)
            {
              int got_nsname = 0;
              unsigned char oldcurrns;
              unsigned long new_ino;

              SetTargetNameSpace(oldnamespace);
              oldcurrns = SetCurrentNameSpace(oldnamespace);
              if (FEConvertDirectoryNumber( 0, s410.st_dev,
                  s410.st_ino, ((int)oldnamespace)&0xff, &new_ino ) == 0 )
              {
                unsigned char *pathbuf = (unsigned char *)malloc(8192);
                if (pathbuf)
                {
                  unsigned long pathcount;
                  if (FEMapVolumeAndDirectoryToPath( s410.st_dev, new_ino,
                           pathbuf, &pathcount ) == 0)
                  {
                    unsigned char *p = pathbuf;
                    for (len = 0; len < (pathcount-1); len++)
                    {
                      p+=*p+1;
                    }
                    len = *p++;
                    if (len < (sizeof(statblk->st_name)-1))
                    {
                      memcpy( statblk->st_name, p, len );
                      statblk->st_name[len] = '\0';
                      statblk->st_ino = new_ino;
                      got_nsname = 1;
                    }
                  } 
                  free((void *)pathbuf);
                }
              }                                                 
              SetCurrentNameSpace(oldcurrns);

              if (!got_nsname && filename) /* xxtype != fd */
              {              
                unsigned int pos; 
                pos = len = strlen(filename);
                while (pos > 0)
                {
                  pos--;
                  if (filename[pos]!='\\' && filename[pos]!='/' && 
                      filename[pos]!=':')
                  {
                    pos++;
                    if ((len-pos) < (sizeof(statblk->st_name)-1))
                    {
                      memcpy(statblk->st_name, filename+pos, (len-pos)+1);
                    }    
                    break;  
                  }
                }
              }
            } /* if (oldnamespace != 0) */
          } /* if (rescode == 0) */
        } /* if (_stat_vector) */
      } /* struct of nwversion is pre-411 */
      else if (size_of_statstruct == sizeof(struct my_stat500) &&
              nwversion >= 500)
      {
        symname = ((xxtype == 'fd')?("\x09""fstat_500"):("\x08""stat_500"));

        _stat_vector = (int (*)(const char *, void *))
             ImportPublicSymbol(nlmHandle, symname);
        _fstat_vector = (int (*)(int , void *))_stat_vector;

        //new_errno = ENOSYS;
        if (_stat_vector)
        {
          if (xxtype == 'fd')
            retcode = (*_fstat_vector)(handle, (void *)statblk );
          else
            retcode = (*_stat_vector)(filename, (void *)statblk );
          if (retcode != 0)
            new_errno = errno;
          UnImportPublicSymbol(nlmHandle, symname );
        }
      }
      else /* stat struct is 411/500 format, and we're on 411 or later */
      {
        symname = ((xxtype == 'fd')?("\x09""fstat_411"):("\x08""stat_411"));

        _stat_vector = (int (*)(const char *, void *))
             ImportPublicSymbol(nlmHandle, symname);
        _fstat_vector = (int (*)(int , void *))_stat_vector;

        TRACE_STAT(("2. fxn=%s => %08x\r\n", symname+1, _stat_vector ));             
        if (_stat_vector)
        {
          struct my_stat411 *s411 = (struct my_stat411 *)statblk;
          int didmalloc = 0;

          new_errno = ENOMEM;
          if (size_of_statstruct != sizeof(struct my_stat411)) 
          {                                          /* must be stat500 */
            s411 = (struct my_stat411 *)malloc(sizeof(struct my_stat411));
            didmalloc = 1;
          }

          if (s411)
          {
            if (xxtype == 'fd')
              retcode = (*_fstat_vector)(handle, (void *)s411 );
            else
              retcode = (*_stat_vector)(filename, (void *)s411 );

            TRACE_STAT(("2. retcode=%d\r\n", retcode));
            if (retcode != 0)
              new_errno = errno;
            else if (didmalloc)
            {
              /* clear everything to zap 'pad' space */
              memset((void *)statblk,0,sizeof(struct stat));
              statblk->st_dev         = s411->st_dev;
              statblk->st_ino         = s411->st_ino;
              statblk->st_mode        = s411->st_mode;
              /* 500 nlink is ulong, pre-500 nlink is short */
              statblk->st_nlink       = s411->st_nlink;
              statblk->st_uid         = s411->st_uid;
              /* 500 gid is ulong, pre-500 gid is short */ 
              statblk->st_gid         = s411->st_gid; 
              statblk->st_rdev        = s411->st_rdev;
              statblk->st_size        = s411->st_size;
              statblk->st_atime       = s411->st_atime;
              statblk->st_mtime       = s411->st_mtime;
              statblk->st_ctime       = s411->st_ctime;
              statblk->st_btime       = s411->st_btime;
              statblk->st_attr        = s411->st_attr;
              statblk->st_archivedID  = s411->st_archivedID;
              statblk->st_updatedID   = s411->st_updatedID;
              statblk->st_inheritedRightsMask = s411->st_inheritedRightsMask;
              statblk->st_originatingNameSpace = s411->st_originatingNameSpace;
              statblk->st_blksize     = s411->st_blksize;
              statblk->st_blocks      = s411->st_blocks;
              statblk->st_flags       = s411->st_flags;
              /* four unsigned longs to four unsigned longs */
              statblk->st_spare[0]    = s411->st_spare[0];
              statblk->st_spare[1]    = s411->st_spare[1];
              statblk->st_spare[2]    = s411->st_spare[2];
              statblk->st_spare[3]    = s411->st_spare[3];
              /* unsigned char[255+1] to unsigned char[255+1] */
              memcpy(statblk->st_name, s411->st_name, sizeof(s411->st_name));
            } /* if (retcode == 0) */

            if (didmalloc)
            {
              free((void *)s411);
            }
          }  /* if (s411) */
          UnImportPublicSymbol(nlmHandle, symname );
        } /* if (_stat_vector) */
      } /* stat struct is 411/500 format, and we're on 411 or later */
    } /* have CLIB context */
  } /* if (filename && statblk) */

  if (retcode != 0 && new_errno == ENOENT && xxtype != 'fd')
  {
    handle = sopen(filename, (0x0000|0x0200), 0x40, 0);
    if (handle == -1 && errno != ENOENT)
      handle = sopen(filename, (0x0000|0x0200), 0, 0);      
    if (handle != -1)
    {
      retcode = xx_stat('fd', handle, 0, statblk, size_of_statstruct);
      new_errno = errno;
      close(handle);
    }
  }

  if (retcode == 0)
    return 0;

  errno = new_errno;
  return -1;
}  
int stat(const char *filename, struct stat *statblk)
{
  TRACE_STAT(("called plain stat()\n"));
  return xx_stat(0, -1, filename, statblk, sizeof(struct stat));
}  
int fstat(int handle, struct stat *statblk)
{
  TRACE_STAT(("called plain fstat()\n"));
  return xx_stat('fd', handle, 0, statblk, sizeof(struct stat));
}  
int stat_411(const char *filename, struct stat *statblk)
{
  TRACE_STAT(("called stat_411()\n"));
  return xx_stat(0, -1, filename, statblk, sizeof(struct my_stat411));
}
int fstat_411(int handle, struct stat *statblk)
{
  TRACE_STAT(("called fstat_411()\n"));
  return xx_stat('fd', handle, 0, statblk, sizeof(struct my_stat411));
}
int fstat_500(int handle, struct stat *statblk)
{
  TRACE_STAT(("called fstat_500()\n"));
  return xx_stat('fd', handle, 0, statblk, sizeof(struct my_stat500));
}
int stat_500(const char *filename, struct stat *statblk)
{
  TRACE_STAT(("called stat_500()\n"));
  return xx_stat(0, -1, filename, statblk, sizeof(struct my_stat500));
}

/* ==================================================================== */

#ifdef __cplusplus
extern "C" {
#endif
#include <unistd.h>   /* access() */
#include <sys/stat.h> /* stat(), struct stat */
#include <errno.h>    /* errno */
#ifdef __cplusplus
}
#endif

#include <nwdos.h>

/* either access or stat is broken on NetWare 5.0-pre-SP3 */
int access( const char *filename, int mode )
{
  int new_errno = EINVAL;
  if (filename)
  {
    struct stat statblk;
    if (stat(filename, &statblk) != 0)
    {  
      TRACE_STAT(("\raccess('%s',%x) failed 1: %d/%s\r\n", filename, mode, errno, strerror(errno)));
      if (errno == ENOENT)
      {
        if (strlen(filename) > 2)
        {        
          if (filename[1] == ':' && 
             ((filename[0] >= 'a' && filename[0] <= 'z') ||
              (filename[0] >= 'A' && filename[0] <= 'Z')) )
          {
            struct find_t dta;
            if (DOSFindFirstFile( (char *)filename, 
                  (_A_RDONLY|_A_HIDDEN|_A_SYSTEM), &dta) == 0)
            {
              if ((mode & W_OK)==0 || (dta.attrib & _A_RDONLY)==0)
                return 0;
              new_errno = EACCES;
            }
          }
        }
      }
    }  
    else if ((mode & X_OK)!=0 && (statblk.st_mode & S_IXUSR)==0)
      new_errno = EACCES;
    else if ((mode & R_OK)!=0 && (statblk.st_mode & S_IRUSR)==0)
      new_errno = EACCES;
    else if ((mode & W_OK)!=0 && (statblk.st_mode & S_IWUSR)==0)
      new_errno = EACCES;
    else
    {
      TRACE_STAT(("\raccess('%s',%x) success\r\n", filename, mode));
      return 0;
    }  
    TRACE_STAT(("\raccess('%s',%x) failed 2: (mode=%x) %d/%s\r\n", filename, mode, statblk.st_mode, new_errno, strerror(new_errno)));
  }
  errno = new_errno;
  return -1;
}

/* ==================================================================== */

#if 0
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h> /* fopen */
#include <errno.h> /* errno, E* constants */
#include <string.h> /* strerror() */
#include <unistd.h> /* close()|unlink() */
#include <fcntl.h> /* O_* contants */
#include <nwfile.h>
#include <dirent.h>
#ifdef __cplusplus
}
#endif

static int __set_sharepurge_bits(const char *filename, long addflags)
{                                        /* addflags = _A_SHARE|_A_IMMPURG */
  int exists = 0;
  DIR *dir = opendir( filename );
  if (dir)
  {
    struct dirent *rdir = readdir( dir );
    exists = 1;
    if (rdir && addflags && (rdir->d_attr & addflags)!=addflags)
    {
      long mdatetime = (rdir->d_date | rdir->d_time << 16);
      long cdatetime = (rdir->d_cdatetime >> 16 | rdir->d_cdatetime << 16);
      long adatetime = (rdir->d_adatetime >> 16 | rdir->d_adatetime << 16);
      long bdatetime = (rdir->d_bdatetime >> 16 | rdir->d_bdatetime << 16);

      SetFileInfo( (char *)(filename), 0x06 /* search attrib = rwhs*/,
                   (rdir->d_attr|addflags),
                   (char *)(&cdatetime), (char *)(&adatetime),
                   (char *)(&mdatetime), (char *)(&bdatetime),
                   rdir->d_uid  );
    }
    closedir( dir );
  }
  return exists;
}


FILE *fopen( const char *filename, const char *modestr )
{
  int new_errno = EINVAL;
  if (filename && modestr)
  {
    int obinmode = -1, rwflags = 0, ctaflags = 0;

    if ( *modestr == 'r')     
    {
      rwflags = O_RDONLY;
      new_errno = 0;
    }  
    else if (*modestr == 'w') 
    {
      rwflags = O_WRONLY;
      ctaflags = O_CREAT|O_TRUNC;
      new_errno = 0;
    }
    else if (*modestr == 'a')
    {
      rwflags = O_WRONLY;
      ctaflags = O_CREAT|O_APPEND;
      new_errno = 0;
    }

    if (new_errno == 0)
    {
      if (modestr[1] == '+')
      {
        rwflags = O_RDWR;
        if (modestr[2] == 'b')
          obinmode = 1;       
        else if (modestr[2] == 't')
          obinmode = 0;
        else if (modestr[2])
          new_errno = EINVAL;
      }
      else if (modestr[1] == 'b' 
            || modestr[1] == 't')
      {
        obinmode = 1;
        if (modestr[1] == 't');
          obinmode = 0;
        if (modestr[2] == '+')
          rwflags = O_RDWR;
        else if (modestr[2])
          new_errno = EINVAL;
      }
      else if (modestr[1])
      {
        new_errno = EINVAL;
      }
    }
      
    if (new_errno == 0)
    {
      int handle, existed;
      FILE *file;

      if (obinmode == 0)
        obinmode = O_TEXT;
      else if (obinmode > 0)
        obinmode = O_BINARY;
      else 
        obinmode = 0;

      existed = __set_sharepurge_bits( filename, _A_SHARE );

      handle = sopen( filename, rwflags|ctaflags|obinmode, SH_DENYRW, 
                                              0 /* S_IWRITE|S_IREAD */ );
      if (handle == -1)
      {
//ConsolePrintf("fopen('%s','%s') failed 1:%s\r\n", filename, modestr, strerror(errno) );
        return (FILE *)0; /* errno is set */
      }

      file = fdopen( handle, modestr );
      if (file)
      {
//ConsolePrintf("fopen('%s','%s') success\r\n", filename, modestr );
        return file;
      }

      new_errno = errno;
      close(handle);
      if (!existed)
      {
        __set_sharepurge_bits( filename, _A_IMMPURG );
        remove( filename );
      }  
    }
//ConsolePrintf("fopen('%s','%s') failed 2:%s\r\n", filename, modestr, strerror(new_errno) );
  }
  errno = new_errno;
  return (FILE *)0;
}
#endif

/* ==================================================================== */
