/*
 * Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Log: client.h,v $
 * Revision 1.127  1999/04/01 03:38:29  cyp
 * Combined half-a-dozen [in|out]_buffer_file vars in in|out_buffer_basename.
 * Created remote_update_dir var - serves as an alternate buffer directory.
 *
 * Revision 1.126  1999/03/20 07:45:01  cyp
 * Modified for/to use new WorkRecord structure.
 *
 * Revision 1.125  1999/03/19 15:45:03  gregh
 * Pad ogr union members to be the same size as crypto members.
 *
 * Revision 1.124  1999/03/18 07:38:02  gregh
 * Add #ifdef GREGH blocks so we can safely leave CONTEST_COUNT at 2 for
 * current builds (until OGR is ready).
 *
 * Revision 1.123  1999/03/18 03:55:09  cyp
 * cleaned up client class variables.
 *
 * Revision 1.122  1999/03/09 07:15:45  gregh
 * Various OGR changes.
 *
 * Revision 1.121  1999/03/05 04:59:46  gregh
 * struct Packet now has a union to hold crypto and OGR data.
 *
 * Revision 1.120  1999/03/01 09:18:31  gregh
 * Fix CONTEST_COUNT so the config doesn't crash, yet the client won't try
 * to process OGR right now.
 *
 * Revision 1.119  1999/03/01 07:54:52  gregh
 * Change WorkRecord to a union that contains crypto (RC5/DES) data and OGR
 * data overlaid on the same bytes.
 *
 * Revision 1.118  1999/02/21 21:44:58  cyp
 * tossed all redundant byte order changing. all host<->net order conversion
 * is done at [get|put][net|disk] points and nowhere else.
 *
 * Revision 1.117  1999/02/20 02:56:59  gregh
 * Added OGR
 *
 * Revision 1.116  1999/02/09 10:55:55  silby
 * Updated OPs from proxy tree.
 *
 * Revision 1.115  1999/01/31 20:19:08  cyp
 * Discarded all 'bool' type wierdness. See cputypes.h for explanation.
 *
 * Revision 1.114  1999/01/29 19:03:14  jlawson
 * fixed formatting.  changed some int vars to bool.
 *
 * Revision 1.113  1999/01/17 13:50:11  cyp
 * buffer thresholds must be volatile.
 *
 * Revision 1.112  1999/01/04 02:49:09  cyp
 * Enforced single checkpoint file for all contests.
 *
 * Revision 1.110  1998/12/25 02:32:11  silby
 * ini writing functions are now not part of client object.
 * This allows the win32 (and other) guis to have
 * configure modules that act on a dummy client object.
 * (Client::Configure should be seperated as well.)
 * Also fixed bug with spaces being taken out of pathnames.
 *
 * Revision 1.109  1998/12/23 00:30:24  silby
 * Stepped back to 1.106.
 *
 * Revision 1.106  1998/12/21 01:38:34  silby
 * scheduledtime should've been a signed integer, fixed.
 *
 * Revision 1.105  1998/12/21 00:56:56  silby
 * Connectrequested is no longer used, removed the variable.
 *
 * Revision 1.104  1998/12/20 23:57:02  silby
 * Added variable scheduledupdatetime; used to coordinate massive
 * update at start of new contests.
 *
 * Revision 1.103  1998/12/20 23:00:35  silby
 * Descontestclosed value is now stored and retrieved from the ini file,
 * additional updated of the .ini file's contest info when fetches and
 * flushes are performed are now done.  Code to throw away old des blocks
 * has not yet been implemented.
 *
 * Revision 1.102  1998/12/12 02:21:40  daa
 * update to add iterationshi to rc5_packet_t and BIG ops to op enum
 *
 * Revision 1.101  1998/12/01 23:31:43  cyp
 * Removed ::totalBlocksDone. count was inaccurate if client was restarted.
 * ::Main() (as opposed to realmain()) now controls restart.
 *
 * Revision 1.100  1998/11/28 19:44:34  cyp
 * InitializeLogging() and DeinitializeLogging() are no longer Client class
 * methods.
 *
 * Revision 1.99  1998/11/26 22:15:54  cyp
 * ::WriteFullConfig() is now ::WriteConfig(1) [default is 0]; threw out
 * useless ::CheckForcedKeyport() and ::CheckForcedKeyproxy()
 *
 * Revision 1.98  1998/11/26 07:27:34  cyp
 * Updated to reflect deleted/new buffwork and buffupd methods.
 *
 * Revision 1.97  1998/11/19 20:48:51  cyp
 * Rewrote -until/-h handling. Did away with useless client.hours (time-to-die
 * is handled by client.minutes anyway). -until/-h/hours all accept "hh:mm"
 * format now (although they do continue to support the asinine "hh.mm").
 *
 * Revision 1.96  1998/11/09 20:05:14  cyp
 * Did away with client.cktime altogether. Time-to-Checkpoint is calculated
 * dynamically based on problem completion state and is now the greater of 1
 * minute and time_to_complete_1_percent (an average change of 1% that is).
 *
 * Revision 1.95  1998/11/08 18:52:07  cyp
 * DisplayHelp() is no longer a client method.
 *
 * Revision 1.94  1998/11/07 14:15:15  cyp
 * InternalCountBuffer() (optionally) returns the number of 2^28 blocks in a
 * buffer.
 *
 * Revision 1.93  1998/11/04 21:28:12  cyp
 * Removed redundant ::hidden option. ::quiet was always equal to ::hidden.
 *
 * Revision 1.92  1998/11/02 04:40:10  cyp
 * Removed redundant ::numcputemp. ::numcpu does it all.
 *
 * Revision 1.91  1998/10/26 03:02:50  cyp
 * Version tag party.
 *
 * Revision 1.90  1998/10/21 12:50:09  cyp
 * Promoted u8 contestids in Fetch/Flush/Update to unsigned ints.
 *
 * Revision 1.89  1998/10/19 12:29:51  cyp
 * completed implementation of 'priority'.
 *
 * Revision 1.88  1998/10/11 00:49:46  cyp
 * Removed Benchmark(), SelfTest() [both are now standalone] and
 * CkpointToBufferInput() which has been superceded by UndoCheckpoint()
 *
 * Revision 1.87  1998/10/09 12:25:23  cyp
 * ValidateProcessorCount() is no longer a client method [is now standalone].
 *
 * Revision 1.86  1998/10/09 00:42:47  blast
 * Benchmark was looking at contest 2=DES, other=RC5 and cmdline.cpp
 * was setting 0=RC5, 1=DES, made it run two rc5 benchmarks. FIXED
 * Changed Calling convention for Benchmark() from u8 to unsigned int.
 *
 * Revision 1.85  1998/10/08 20:57:14  cyp
 * Removed Client::UnlockBuffer() [is now standalone]. Changed all buffer
 * function prototypes that took 'u8' as contest id to use 'unsigned int'
 * instead.
 *
 * Revision 1.84  1998/10/08 10:13:33  cyp
 * Removed GetProcessorType() from the client class. No need for it to be a
 * method.
 *
 * Revision 1.83  1998/10/06 21:17:08  cyp
 * Changed prototype of InitializeLogging() to take an argument.
 *
 * Revision 1.82  1998/10/05 05:13:09  cyp
 * Fixed missing #ifdef MMX_RC5 for client.usemmx
 *
 * Revision 1.81  1998/10/03 03:52:24  cyp
 * Removed ::Install() and ::Uninstall() [os specific functions needed to be
 * called directly from ParseCommandLine()], changed ::runhidden to an 'int',
 * removed win16 specific SurrenderCPU() [uses win api function instead]
 *
 * Revision 1.80  1998/09/28 02:36:13  cyp
 * new method LoadSaveProblems() [probfill.cpp]; removed MAXCPUS [obsolete];
 * removed SetNiceness() [SetPriority() setprio.cpp]; removed DisplayBanner()
 * [static function with restart-tracking in client.cpp]; removed RunStartup()
 * [static function in client.cpp]; changed return value of ::Run() from
 * s32 to int since this is the client's exit code.
 *
 * Revision 1.79  1998/09/19 08:50:20  silby
 * Added in beta test client timeouts.  Enabled/controlled from version.h by 
 *
 * Revision 1.78  1998/08/28 22:31:09  cyp
 * Added prototype for Client::Main().
 *
 * Revision 1.77  1998/08/10 22:02:25  cyruspatel
 * Removed xxxTriggered and pausefilefound statics (now f() in triggers.cpp)
 *
 * Revision 1.76  1998/08/05 18:28:47  cyruspatel
 * Converted more printf()s to LogScreen()s
 *
 * Revision 1.75  1998/08/02 16:17:47  cyruspatel
 * Completed support for logging.
 *
 * Revision 1.74  1998/08/02 06:51:40  silby
 * Slight logging changes to bring win32gui in sync with rest of tree.
 *
 * Revision 1.73  1998/08/02 03:16:42  silby
 * Log,LogScreen, and LogScreenf are now in logging.cpp
 *
 * Revision 1.72  1998/07/30 05:09:08  silby
 * Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being included
 * , now they are not. Also, added the logic for dialwhenneeded, which is a 
 * new lurk feature.
 *
 * Revision 1.71  1998/07/29 05:14:49  silby
 * Changes to win32 so that LurkInitiateConnection now works 
 *
 * Revision 1.70  1998/07/26 12:46:04  cyruspatel
 * new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
 * xx.v27.distributed.net then that will be interpreted by Network::Resolve()
 * to mean 'find a keyserver that covers the timezone I am in'. Network
 * constructor extended to take this as an argument.
 *
 * Revision 1.69  1998/07/25 06:31:44  silby
 * Added lurk functions to initiate a connection and hangup a connection.  win32 hangup is functional.
 *
 * Revision 1.68  1998/07/25 05:29:57  silby
 * Changed all lurk options to use a LURK define (automatically set in client.h) so that lurk integration of mac/amiga clients needs only touch client.h and two functions in client.cpp
 *
 * Revision 1.67  1998/07/15 06:58:12  silby
 * Changes to Flush, Fetch, and Update so that when the win32 gui sets connectoften to initiate one of the above more verbose feedback will be given.  Also, when force=1, a connect will be made regardless of offlinemode and lurk.
 *
 * Revision 1.66  1998/07/11 00:37:29  silby
 * Documented the connectrequested variable better.
 *
 * Revision 1.65  1998/07/08 09:22:48  remi
 * Added support for the MMX bitslicer.
 * Wrapped $Log comments to some reasonable value.
 *
 * Revision 1.64  1998/07/08 05:19:26  jlawson
 * updates to get Borland C++ to compile under Win32.
 *
 * Revision 1.63  1998/07/07 21:55:18  cyruspatel
 * client.h has been split into client.h and baseincs.h 
 *
 * Revision 1.62  1998/07/06 03:15:31  jlawson
 * prototype for InternalGetLocalFilename no longer defined by default.
 *
 * Revision 1.61  1998/07/05 15:54:02  cyruspatel
 * Implemented EraseCheckpointFile() and TruncateBufferFile() in buffwork.cpp;
 * substituted unlink() with EraseCheckpointFile() in client.cpp; modified
 * client.h to #include buffwork.h; moved InternalGetLocalFilename() to
 * cliconfig.cpp; cleaned up some.
 *
 * Revision 1.60  1998/07/05 13:44:10  cyruspatel
 * Fixed an inadvertent wrap of one of the long single-line revision headers.
 *
 * Revision 1.59  1998/07/05 13:09:04  cyruspatel
 * Created new pathwork.cpp. Threw away icky client methods.
 *
 * Revision 1.58  1998/07/05 12:42:37  cyruspatel
 * Created cpucheck.h to support makefiles that rely on autodependancy info
 * to detect file changes.
 *
 * Revision 1.57  1998/07/04 21:05:34  silby
 * Changes to lurk code; win32 and os/2 code now uses the same variables,
 * and has been integrated into StartLurk and LurkStatus functions so they
 * now act the same.  Additionally, problems with lurkonly clients trying to
 * connect when contestdone was wrong should be fixed.
 *
 * Revision 1.56  1998/07/02 13:09:28  kbracey
 * A couple of RISC OS fixes - printf format specifiers made long.
 * Changed a "blocks" to "block%s", n==1?"":"s".
 *
 * Revision 1.55  1998/07/01 03:29:02  silby
 * Added prototype for CheckForcedKeyproxy (used in cliconfig)
 *
 * Revision 1.54  1998/06/29 08:43:59  jlawson
 * More OS_WIN32S/OS_WIN16 differences and long constants added.
 *
 * Revision 1.53  1998/06/29 07:51:54  ziggyb
 * OS/2 lurk and DOD header additions. (platforms\os2cli\dod.h)
 *
 * Revision 1.52  1998/06/29 06:57:43  jlawson
 * added new platform OS_WIN32S to make code handling easier.
 *
 * Revision 1.51  1998/06/28 23:40:23  silby
 * Changes to path handling code so that path validation+adding to filenames
 * will be more reliable (especially on win32).
 *
 * Revision 1.50  1998/06/25 04:43:32  silby
 * Changes to Internalgetfilename for win32 (+ other platforms in the future)
 * to make path handling better (now it won't miss / and : on win32)
 *
 * Revision 1.49  1998/06/25 03:02:32  blast
 * Moved the version #defines from client.h to version.h and added a version
 * string called CLIENT_VERSIONSTRING...
 *
 * Revision 1.48  1998/06/25 02:30:38  jlawson
 * put back public Client::connectrequested for use by win32gui
 *
 * Revision 1.47  1998/06/24 16:24:05  cyruspatel
 * Added prototype for GetTimesliceBaseline() (in cpucheck.cpp)
 *
 * Revision 1.46  1998/06/24 15:54:18  jlawson
 * added pragma pack(1) around structures.
 *
 * Revision 1.45  1998/06/22 19:42:01  daa
 * bump to 2.7025.410 this is likely to become 2.7100.411
 *
 * Revision 1.44  1998/06/22 01:04:56  cyruspatel
 * DOS changes. Fixes various compile-time errors: removed extraneous ')' in
 * sleepdef.h, resolved h.tonl()/n.tohl() conflict with same def in client.h
 * (is now inline asm), added NO!NETWORK wrapper around Network::Resolve()
 *
 * Revision 1.43  1998/06/21 17:10:26  cyruspatel
 * Fixed some NetWare smp problems. Merged duplicate numcpu validation code
 * in ::ReadConfig()/::ValidateConfig() into ::ValidateProcessorCount() and
 * spun that off, together with what used to be ::x86id() or ::ArmId(), into
 * cpucheck.cpp. Adjusted and cleaned up client.h accordingly.
 *
 * Revision 1.42  1998/06/15 09:17:57  jlawson
 * removed include of pthread.h since threadcd.h knows when to include it
 *
 * Revision 1.41  1998/06/14 11:22:37  ziggyb
 * Fixed the OS/2 headers and added an os2defs.h and adjusted for the
 * separate sleep defines header.
 *
 * Revision 1.40  1998/06/14 08:12:36  friedbait
 * 'Log' keywords added to maintain automatic change history
 *
 *
*/

#ifndef __CLIENT_H__
#define __CLIENT_H__

#define MAXBLOCKSPERBUFFER  500
#define CONTEST_COUNT       3 /* RC5,DES,OGR */

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

/* -------------------------------------------------------------------------- */

#endif // __CLIENT_H__

