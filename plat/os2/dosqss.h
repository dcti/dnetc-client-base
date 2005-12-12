/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/

#ifndef DOSQSS_H
#define DOSQSS_H "@(#)$Id: dosqss.h,v 1.1.2.5 2005/12/12 17:55:54 stream Exp $"


#ifdef __cplusplus
extern "C" {
#endif
APIRET APIENTRY DosQuerySysState(ULONG func,ULONG arg1,ULONG arg2,
                                 ULONG _res_,PVOID buf,ULONG bufsz);

#if defined(__EMX__)
  USHORT _THUNK_FUNCTION (Dos16QProcStatus) (PVOID buffer, USHORT buffer_size);

  USHORT DosQProcStatus(PVOID buffer, USHORT buffer_size)
    {
      return (USHORT)
        ( _THUNK_PROLOG (sizeof(PVOID)+sizeof(USHORT));
          _THUNK_FLAT (buffer);
          _THUNK_SHORT (buffer_size);
          _THUNK_CALL (Dos16QProcStatus));
    }
#else
  APIRET16 APIENTRY16 DosQProcStatus(PVOID buffer, USHORT buffer_size);
#endif

typedef struct {
    ULONG   threadcnt;
    ULONG   proccnt;
    ULONG   modulecnt;
} QGLOBAL, *PQGLOBAL;

typedef struct {
    ULONG   rectype;        /* 256 for thread */
    USHORT  threadid;
    USHORT  slotid;
    ULONG   sleepid;
    ULONG   priority;
    ULONG   systime;
    ULONG   usertime;
    UCHAR   state;
    UCHAR   _reserved1_;    /* padding to ULONG */
    USHORT  _reserved2_;    /* padding to ULONG */
} QTHREAD, *PQTHREAD;

typedef struct {
    USHORT  sfn;
    USHORT  refcnt;
    ULONG   flags;
    ULONG   accmode;
    ULONG   filesize;
    USHORT  volhnd;
    USHORT  attrib;
    USHORT  _reserved_;
} QFDS, *PQFDS;

typedef struct qfile {
    ULONG           rectype;        /* 8 for file */
    struct qfile    *next;
    ULONG           opencnt;
    PQFDS           filedata;
    char            name[1];
} QFILE, *PQFILE;

typedef struct {
    ULONG   rectype;        /* 1 for process */
    PQTHREAD threads;
    USHORT  pid;
    USHORT  ppid;
    ULONG   type;
    ULONG   state;
    ULONG   sessid;
    USHORT  hndmod;
    USHORT  threadcnt;
    ULONG   privsem32cnt;
    ULONG   _reserved2_;
    USHORT  sem16cnt;
    USHORT  dllcnt;
    USHORT  shrmemcnt;
    USHORT  fdscnt;
    PUSHORT sem16s;
    PUSHORT dlls;
    PUSHORT shrmems;
    PUSHORT fds;
} QPROCESS, *PQPROCESS;

typedef struct sema {
    struct sema *next;
    USHORT  refcnt;
    UCHAR   sysflags;
    UCHAR   sysproccnt;
    ULONG   _reserved1_;
    USHORT  index;
    CHAR    name[1];
} QSEMA, *PQSEMA;

typedef struct {
    ULONG   rectype;        /**/
    ULONG   _reserved1_;
    USHORT  _reserved2_;
    USHORT  syssemidx;
    ULONG   index;
    QSEMA   sema;
} QSEMSTRUC, *PQSEMSTRUC;

typedef struct {
    USHORT  pid;
    USHORT  opencnt;
} QSEMOWNER32, *PQSEMOWNER32;

typedef struct {
    PQSEMOWNER32    own;
    PCHAR           name;
    PVOID           semrecs; /* array of associated sema's */
    USHORT          flags;
    USHORT          semreccnt;
    USHORT          waitcnt;
    USHORT          _reserved_;     /* padding to ULONG */
} QSEMSMUX32, *PQSEMSMUX32;

typedef struct {
    PQSEMOWNER32    own;
    PCHAR           name;
    PQSEMSMUX32     mux;
    USHORT          flags;
    USHORT          postcnt;
} QSEMEV32, *PQSEMEV32;

typedef struct {
    PQSEMOWNER32    own;
    PCHAR           name;
    PQSEMSMUX32     mux;
    USHORT          flags;
    USHORT          refcnt;
    USHORT          thrdnum;
    USHORT          _reserved_;     /* padding to ULONG */
} QSEMMUX32, *PQSEMMUX32;

typedef struct semstr32 {
    struct semstr32 *next;
    QSEMEV32 evsem;
    QSEMMUX32  muxsem;
    QSEMSMUX32 smuxsem;
} QSEM32STRUC, *PQSEM32STRUC;

typedef struct shrmem {
    struct shrmem *next;
    USHORT  hndshr;
    USHORT  selshr;
    USHORT  refcnt;
    CHAR    name[1];
} QSHRMEM, *PQSHRMEM;

typedef struct module {
    struct module *next;
    USHORT  hndmod;
    USHORT  type;
    ULONG   refcnt;
    ULONG   segcnt;
    PVOID   _reserved_;
    PCHAR   name;
    USHORT  modref[1];
} QMODULE, *PQMODULE;

typedef struct {
    PQGLOBAL        gbldata;
    PQPROCESS       procdata;
    PQSEMSTRUC      semadata;
    PQSEM32STRUC    sem32data;      /* not always present */
    PQSHRMEM        shrmemdata;
    PQMODULE        moddata;
    PVOID           _reserved2_;
    PQFILE          filedata;       /* only present in FP19 or later or W4 */
} QTOPLEVEL, *PQTOPLEVEL;

#define DOSQSS_PROCESS 0x01

#define DOSQSS_MODULE  0x04

// -----------------------------------------------------------------------

/*** Signal support (16-bit) ***/

/* Signal Numbers for DosSetSigHandler  */

#define SIG_CTRLC                  1       /* Control C                  */
#define SIG_BROKENPIPE             2       /* Broken Pipe                */
#define SIG_KILLPROCESS            3       /* Program Termination        */
#define SIG_CTRLBREAK              4       /* Control Break              */
#define SIG_PFLG_A                 5       /* Process Flag A             */
#define SIG_PFLG_B                 6       /* Process Flag B             */
#define SIG_PFLG_C                 7       /* Process Flag C             */
#define SIG_CSIGNALS               8       /* number of signals plus one */

/* Flag Numbers for DosFlagProcess */

#define PFLG_A                     0       /* Process Flag A             */
#define PFLG_B                     1       /* Process Flag B             */
#define PFLG_C                     2       /* Process Flag C             */

/* Signal actions */

#define SIGA_KILL                  0
#define SIGA_IGNORE                1
#define SIGA_ACCEPT                2
#define SIGA_ERROR                 3
#define SIGA_ACKNOWLEDGE           4

/* DosFlagProcess codes */

#define FLGP_SUBTREE               0
#define FLGP_PID                   1

#define DosSetSigHandler        Dos16SetSigHandler
#define DosFlagProcess          Dos16FlagProcess

#if defined(__EMX__)
   /* Nothing. This stuff must be not used under EMX (already emulated by EMX) */
#else
typedef void (pascal __far16 *PFNSIGHANDLER)(USHORT, USHORT);

APIRET16 APIENTRY16 DosSetSigHandler(PFNSIGHANDLER pfnSigHandler,
                                     PFNSIGHANDLER FAR * ppfnPrev, PUSHORT pfAction,
                                     USHORT fAction, USHORT usSigNum);

APIRET16 APIENTRY16 DosFlagProcess(PID pid, USHORT fScope, USHORT usFlagNum,
                                   USHORT usFlagArg);
#endif /* __EMX__ */

#ifdef __cplusplus
}
#endif

#endif
