/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CLIENT_H__
#define __CLIENT_H__ "@(#)$Id: client.h,v 1.142 2000/01/04 01:31:34 michmarc Exp $"


enum {
  RC5, // http://www.rsa.com/rsalabs/97challenge/
  DES, // http://www.rsa.com/rsalabs/des3/index.html
  OGR, // http://members.aol.com/golomb20/
  CSC  // http://www.cie-signaux.fr/security/index.htm
};
#define CONTEST_COUNT       4  /* RC5,DES,OGR,CSC */

#include "problem.h"           /* ContestWork structure */
#include "lurk.h"              /* client structure copy of lurk_conf */

#ifndef MIPSpro
#pragma pack(1)               /* no padding allowed */
#endif /* ! MIPSpro */

typedef struct
{
  ContestWork work;/* {key,iv,plain,cypher,keysdone,iter} or {stub,pad} */
  u32  resultcode; /* core state: RESULT_WORKING:0|NOTHING:1|FOUND:2 */
  char id[59];     /* d.net id of worker that last used this */
  u8   contest;    /* 0=rc5,1=des,etc. If this is changed, make this u32 */
  u8   cpu;        /* 97.11.25 If this is ever changed, make this u32 */
  u8   os;         /* 97.11.25 If this is ever changed, make this u32 */
  u8   buildhi;    /* 97.11.25 If this is ever changed, make this u32 */
  u8   buildlo;    /* 97.11.25 If this is ever changed, make this u32 */
} WorkRecord;

#ifndef MIPSpro
# pragma pack()
#endif /* ! MIPSpro */

#define __T(x) #x
#define _T(x) __T(x)

#define MAXBLOCKSPERBUFFER             2000  /* Now in work units */
#define BUFTHRESHOLD_DEFAULT             25  /* Now in work units */
#define PREFERREDBLOCKSIZE_DEFAULT       30
#define PREFERREDBLOCKSIZE_MIN           28
#define PREFERREDBLOCKSIZE_MAX           33
#define BUFFER_DEFAULT_IN_BASENAME  "buff-in"
#define BUFFER_DEFAULT_OUT_BASENAME "buff-out"
#define MINCLIENTOPTSTRLEN   64 /* no asciiz var is smaller than this */

struct membuffstruct 
{ 
  unsigned long count; 
  WorkRecord *buff[MAXBLOCKSPERBUFFER];
};

typedef struct
{
public:
  /* non-user-configurable */
  int  nonewblocks;
  int  randomchanged;
  int  randomprefix;
  int  rc564closed;
  int  stopiniio;
  u32  scheduledupdatetime;
  char inifilename[MINCLIENTOPTSTRLEN*2];
  struct { struct membuffstruct in, out; } membufftable[CONTEST_COUNT];

  /* -- general -- */
  char id[MINCLIENTOPTSTRLEN];
  int  quietmode;
  int  blockcount;
  int  minutes;
  int  percentprintingoff;
  int  noexitfilecheck;
  char pausefile[MINCLIENTOPTSTRLEN*2];
  char loadorder_map[CONTEST_COUNT];

  /* -- buffers -- */
  int  nodiskbuffers;
  char in_buffer_basename[MINCLIENTOPTSTRLEN*2];
  char out_buffer_basename[MINCLIENTOPTSTRLEN*2];
  char checkpoint_file[MINCLIENTOPTSTRLEN*2];
  int  offlinemode;
    int  nettimeout;
    int  nofallback;
    int  autofindkeyserver;
    char keyproxy[MINCLIENTOPTSTRLEN];
    int  keyport;
    char httpproxy[MINCLIENTOPTSTRLEN];
    int  httpport;
    int  uuehttpmode;
    char httpid[MINCLIENTOPTSTRLEN*2];
  int  noupdatefromfile;
    char remote_update_dir[MINCLIENTOPTSTRLEN*2];
  #ifdef LURK 
  struct dialup_conf lurk_conf;
  #endif
  int  connectoften;

  // In general, don't use inthreshold anymore;
  // Use ClientGetInThreshold(client, contest)
  int inthreshold[CONTEST_COUNT]; 
  int outthreshold[CONTEST_COUNT];
  int timethreshold[CONTEST_COUNT];  /* in hours */

  int preferred_blocksize[CONTEST_COUNT];

  /* -- perf -- */
  int  numcpu;
  int  priority;
  int  coretypes[CONTEST_COUNT];

  /* -- log -- */
  char logname[MINCLIENTOPTSTRLEN*2];
  char logfiletype[MINCLIENTOPTSTRLEN]; /* "none", "no limit", "rotate", "restart", "fifo" */
  char logfilelimit[MINCLIENTOPTSTRLEN]; /* "nnn K|M|days" etc */
  int  messagelen;
  char smtpsrvr[MINCLIENTOPTSTRLEN];
  int  smtpport;
  char smtpfrom[MINCLIENTOPTSTRLEN];
  char smtpdest[MINCLIENTOPTSTRLEN];

} Client;

void ResetClientData(Client *client); /* reset everything */
int ClientGetInThreshold(Client *client, int contestid, bool force = false);
int ClientGetOutThreshold(Client *client, int contestid, bool force = false);
int ClientRun(Client *client);  /* run the loop, do the work */

#endif /* __CLIENT_H__ */
