/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CLIENT_H__
#define __CLIENT_H__ "@(#)$Id: client.h,v 1.128 1999/04/04 16:28:04 cyp Exp $"

#define MAXBLOCKSPERBUFFER  500
#define CONTEST_COUNT       3  /* RC5,DES,OGR */

#include "problem.h"          /* ContestWork structure */
#pragma pack(1)               /* no padding allowed */

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

#pragma pack()

struct membuffstruct 
{ 
  unsigned long count; 
  WorkRecord *buff[MAXBLOCKSPERBUFFER];
};

class Client
{
public:
  /* non-user-configurable */
  int  nonewblocks;
  int  randomprefix;
  int  randomchanged;
  int  rc564closed;
  int  stopiniio;
  u32  scheduledupdatetime;
  char inifilename[128];

  /* -- block/buffer -- */
  char id[64];
  char checkpoint_file[128];
  int  nodiskbuffers;
  s32  connectoften;
  s32  preferred_blocksize;
  struct { struct membuffstruct in, out; } membufftable[CONTEST_COUNT];
  char in_buffer_basename[128];
  char out_buffer_basename[128];
  char remote_update_dir[128];
  char loadorder_map[CONTEST_COUNT];
  volatile s32 inthreshold[CONTEST_COUNT];
  volatile s32 outthreshold[CONTEST_COUNT];

  /* -- net -- */
  s32  offlinemode;
  s32  nettimeout;
  s32  nofallback;
  int  autofindkeyserver;
  char keyproxy[64];
  s32  keyport;
  char httpproxy[64];
  s32  httpport;
  s32  uuehttpmode;
  char httpid[128];

  /* -- log -- */
  char logname[128];
  s32  messagelen;
  char smtpsrvr[128];
  s32  smtpport;
  char smtpfrom[128];
  char smtpdest[128];

  /* -- perf -- */
  s32  numcpu;
  s32  cputype;
  s32  priority;

  /* -- misc -- */
  int  quietmode;
  s32  blockcount;
  s32  minutes;
  s32  percentprintingoff;
  s32  noexitfilecheck;
  char pausefile[128];


  /* --------------------------------------------------------------- */

  long PutBufferRecord(const WorkRecord * data);
  long GetBufferRecord( WorkRecord * data, unsigned int contest, int use_out_file);
  long GetBufferCount( unsigned int contest, int use_out_file, unsigned long *normcountP );

  Client();
  ~Client() {};

  int Main( int argc, const char *argv[] );
    // encapsulated main().  client.Main() may restart itself

  int ParseCommandline( int runlevel, int argc, const char *argv[], 
                        int *retcodeP, int logging_is_initialized );
    // runlevel=0 = parse cmdline, >0==exec modes && print messages for init'd cmdline options
    // returns !0 if app should be terminated

  int CheckpointAction(int action, unsigned int load_problem_count );
    // CHECKPOINT_OPEN (copy from checkpoint to in-buffer), *_REFRESH, *_CLOSE
    // returns !0 if checkpointing is disabled

  int  Configure( void );
    // runs the interactive configuration setup

  int Run( void );
    // run the loop, do the work

  int BufferUpdate( int updatereq_flags, int interactive );
    // pass flags ORd with BUFFERUPDATE_FETCH/*_FLUSH. 
    // if interactive, prints "Input buffer full. No fetch required" etc.
    // returns updated flags or < 0 if offlinemode!=0 or NetOpen() failed.

  int SelectCore(int quietly);
    // always returns zero.
    // to configure for cpu. called before Run() from main(), or for 
    // "modes" (Benchmark()/Test()) from ParseCommandLine().

  unsigned int LoadSaveProblems(unsigned int load_problem_count, int retmode);
    // returns number of actually loaded problems 

};

#endif /* __CLIENT_H__ */
