// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This file contains the basic types used in a lot of places: Client class;
// Operation, contest_id_t enums; Packet, FileHeader and FileEntry structs; 
// none of them depend on anything other than cputypes.h, and network.h
// ----------------------------------------------------------------------
//
// $Log: client.h,v $
// Revision 1.119  1999/03/01 07:54:52  gregh
// Change FileEntry to a union that contains crypto (RC5/DES) data and OGR
// data overlaid on the same bytes.
//
// Revision 1.118  1999/02/21 21:44:58  cyp
// tossed all redundant byte order changing. all host<->net order conversion
// as well as scram/descram/checksumming is done at [get|put][net|disk] points
// and nowhere else.
//
// Revision 1.117  1999/02/20 02:56:59  gregh
// Added IDCONTEST_OGR
//
// Revision 1.116  1999/02/09 10:55:55  silby
// Updated OPs from proxy tree.
//
// Revision 1.115  1999/01/31 20:19:08  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.114  1999/01/29 19:03:14  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.113  1999/01/17 13:50:11  cyp
// buffer thresholds must be volatile.
//
// Revision 1.112  1999/01/04 02:49:09  cyp
// Enforced single checkpoint file for all contests.
//
// Revision 1.110  1998/12/25 02:32:11  silby
// ini writing functions are now not part of client object.
// This allows the win32 (and other) guis to have
// configure modules that act on a dummy client object.
// (Client::Configure should be seperated as well.)
// Also fixed bug with spaces being taken out of pathnames.
//
// Revision 1.109  1998/12/23 00:30:24  silby
// Stepped back to 1.106.
//
// Revision 1.106  1998/12/21 01:38:34  silby
// scheduledtime should've been a signed integer, fixed.
//
// Revision 1.105  1998/12/21 00:56:56  silby
// Connectrequested is no longer used, removed the variable.
//
// Revision 1.104  1998/12/20 23:57:02  silby
// Added variable scheduledupdatetime; used to coordinate massive
// update at start of new contests.
//
// Revision 1.103  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.102  1998/12/12 02:21:40  daa
// update to add iterationshi to rc5_packet_t and BIG ops to op enum
//
// Revision 1.101  1998/12/01 23:31:43  cyp
// Removed ::totalBlocksDone. count was inaccurate if client was restarted.
// ::Main() (as opposed to realmain()) now controls restart.
//
// Revision 1.100  1998/11/28 19:44:34  cyp
// InitializeLogging() and DeinitializeLogging() are no longer Client class
// methods.
//
// Revision 1.99  1998/11/26 22:15:54  cyp
// ::WriteFullConfig() is now ::WriteConfig(1) [default is 0]; threw out
// useless ::CheckForcedKeyport() and ::CheckForcedKeyproxy()
//
// Revision 1.98  1998/11/26 07:27:34  cyp
// Updated to reflect deleted/new buffwork and buffupd methods.
//
// Revision 1.97  1998/11/19 20:48:51  cyp
// Rewrote -until/-h handling. Did away with useless client.hours (time-to-die
// is handled by client.minutes anyway). -until/-h/hours all accept "hh:mm"
// format now (although they do continue to support the asinine "hh.mm").
//
// Revision 1.96  1998/11/09 20:05:14  cyp
// Did away with client.cktime altogether. Time-to-Checkpoint is calculated
// dynamically based on problem completion state and is now the greater of 1
// minute and time_to_complete_1_percent (an average change of 1% that is).
//
// Revision 1.95  1998/11/08 18:52:07  cyp
// DisplayHelp() is no longer a client method.
//
// Revision 1.94  1998/11/07 14:15:15  cyp
// InternalCountBuffer() (optionally) returns the number of 2^28 blocks in a
// buffer.
//
// Revision 1.93  1998/11/04 21:28:12  cyp
// Removed redundant ::hidden option. ::quiet was always equal to ::hidden.
//
// Revision 1.92  1998/11/02 04:40:10  cyp
// Removed redundant ::numcputemp. ::numcpu does it all.
//
// Revision 1.91  1998/10/26 03:02:50  cyp
// Version tag party.
//
// Revision 1.90  1998/10/21 12:50:09  cyp
// Promoted u8 contestids in Fetch/Flush/Update to unsigned ints.
//
// Revision 1.89  1998/10/19 12:29:51  cyp
// completed implementation of 'priority'.
//
// Revision 1.88  1998/10/11 00:49:46  cyp
// Removed Benchmark(), SelfTest() [both are now standalone] and
// CkpointToBufferInput() which has been superceded by UndoCheckpoint()
//
// Revision 1.87  1998/10/09 12:25:23  cyp
// ValidateProcessorCount() is no longer a client method [is now standalone].
//
// Revision 1.86  1998/10/09 00:42:47  blast
// Benchmark was looking at contest 2=DES, other=RC5 and cmdline.cpp
// was setting 0=RC5, 1=DES, made it run two rc5 benchmarks. FIXED
// Changed Calling convention for Benchmark() from u8 to unsigned int.
//
// Revision 1.85  1998/10/08 20:57:14  cyp
// Removed Client::UnlockBuffer() [is now standalone]. Changed all buffer
// function prototypes that took 'u8' as contest id to use 'unsigned int'
// instead.
//
// Revision 1.84  1998/10/08 10:13:33  cyp
// Removed GetProcessorType() from the client class. No need for it to be a
// method.
//
// Revision 1.83  1998/10/06 21:17:08  cyp
// Changed prototype of InitializeLogging() to take an argument.
//
// Revision 1.82  1998/10/05 05:13:09  cyp
// Fixed missing #ifdef MMX_RC5 for client.usemmx
//
// Revision 1.81  1998/10/03 03:52:24  cyp
// Removed ::Install() and ::Uninstall() [os specific functions needed to be
// called directly from ParseCommandLine()], changed ::runhidden to an 'int',
// removed win16 specific SurrenderCPU() [uses win api function instead]
//
// Revision 1.80  1998/09/28 02:36:13  cyp
// new method LoadSaveProblems() [probfill.cpp]; removed MAXCPUS [obsolete];
// removed SetNiceness() [SetPriority() setprio.cpp]; removed DisplayBanner()
// [static function with restart-tracking in client.cpp]; removed RunStartup()
// [static function in client.cpp]; changed return value of ::Run() from
// s32 to int since this is the client's exit code.
//
// Revision 1.79  1998/09/19 08:50:20  silby
// Added in beta test client timeouts.  Enabled/controlled from version.h by 
//
// Revision 1.78  1998/08/28 22:31:09  cyp
// Added prototype for Client::Main().
//
// Revision 1.77  1998/08/10 22:02:25  cyruspatel
// Removed xxxTriggered and pausefilefound statics (now f() in triggers.cpp)
//
// Revision 1.76  1998/08/05 18:28:47  cyruspatel
// Converted more printf()s to LogScreen()s
//
// Revision 1.75  1998/08/02 16:17:47  cyruspatel
// Completed support for logging.
//
// Revision 1.74  1998/08/02 06:51:40  silby
// Slight logging changes to bring win32gui in sync with rest of tree.
//
// Revision 1.73  1998/08/02 03:16:42  silby
// Log,LogScreen, and LogScreenf are now in logging.cpp
//
// Revision 1.72  1998/07/30 05:09:08  silby
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being included
// , now they are not. Also, added the logic for dialwhenneeded, which is a 
// new lurk feature.
//
// Revision 1.71  1998/07/29 05:14:49  silby
// Changes to win32 so that LurkInitiateConnection now works 
//
// Revision 1.70  1998/07/26 12:46:04  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.69  1998/07/25 06:31:44  silby
// Added lurk functions to initiate a connection and hangup a connection.  win32 hangup is functional.
//
// Revision 1.68  1998/07/25 05:29:57  silby
// Changed all lurk options to use a LURK define (automatically set in client.h) so that lurk integration of mac/amiga clients needs only touch client.h and two functions in client.cpp
//
// Revision 1.67  1998/07/15 06:58:12  silby
// Changes to Flush, Fetch, and Update so that when the win32 gui sets connectoften to initiate one of the above more verbose feedback will be given.  Also, when force=1, a connect will be made regardless of offlinemode and lurk.
//
// Revision 1.66  1998/07/11 00:37:29  silby
// Documented the connectrequested variable better.
//
// Revision 1.65  1998/07/08 09:22:48  remi
// Added support for the MMX bitslicer.
// Wrapped $Log comments to some reasonable value.
//
// Revision 1.64  1998/07/08 05:19:26  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.63  1998/07/07 21:55:18  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.62  1998/07/06 03:15:31  jlawson
// prototype for InternalGetLocalFilename no longer defined by default.
//
// Revision 1.61  1998/07/05 15:54:02  cyruspatel
// Implemented EraseCheckpointFile() and TruncateBufferFile() in buffwork.cpp;
// substituted unlink() with EraseCheckpointFile() in client.cpp; modified
// client.h to #include buffwork.h; moved InternalGetLocalFilename() to
// cliconfig.cpp; cleaned up some.
//
// Revision 1.60  1998/07/05 13:44:10  cyruspatel
// Fixed an inadvertent wrap of one of the long single-line revision headers.
//
// Revision 1.59  1998/07/05 13:09:04  cyruspatel
// Created new pathwork.cpp which contains functions for determining/setting
// the "work directory" and pathifying a filename that has no dirspec.
// GetFullPathForFilename() is ideally suited for use in (or just prior to) a
// call to fopen(). This obviates the neccessity to pre-parse filenames or
// maintain separate filename buffers. In addition, each platform has its own
// code section to avoid cross-platform assumptions. More doc in pathwork.cpp
// #define DONT_USE_PATHWORK if you don't want to use these functions.
//
// Revision 1.58  1998/07/05 12:42:37  cyruspatel
// Created cpucheck.h to support makefiles that rely on autodependancy info
// to detect file changes.
//
// Revision 1.57  1998/07/04 21:05:34  silby
// Changes to lurk code; win32 and os/2 code now uses the same variables,
// and has been integrated into StartLurk and LurkStatus functions so they
// now act the same.  Additionally, problems with lurkonly clients trying to
// connect when contestdone was wrong should be fixed.
//
// Revision 1.56  1998/07/02 13:09:28  kbracey
// A couple of RISC OS fixes - printf format specifiers made long.
// Changed a "blocks" to "block%s", n==1?"":"s".
//
// Revision 1.55  1998/07/01 03:29:02  silby
// Added prototype for CheckForcedKeyproxy (used in cliconfig)
//
// Revision 1.54  1998/06/29 08:43:59  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.53  1998/06/29 07:51:54  ziggyb
// OS/2 lurk and DOD header additions. (platforms\os2cli\dod.h)
//
// Revision 1.52  1998/06/29 06:57:43  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.51  1998/06/28 23:40:23  silby
// Changes to path handling code so that path validation+adding to filenames
// will be more reliable (especially on win32).
//
// Revision 1.50  1998/06/25 04:43:32  silby
// Changes to Internalgetfilename for win32 (+ other platforms in the future)
// to make path handling better (now it won't miss / and : on win32)
//
// Revision 1.49  1998/06/25 03:02:32  blast
// Moved the version #defines from client.h to version.h and added a version
// string called CLIENT_VERSIONSTRING...
//
// Revision 1.48  1998/06/25 02:30:38  jlawson
// put back public Client::connectrequested for use by win32gui
//
// Revision 1.47  1998/06/24 16:24:05  cyruspatel
// Added prototype for GetTimesliceBaseline() (in cpucheck.cpp)
//
// Revision 1.46  1998/06/24 15:54:18  jlawson
// added pragma pack(1) around structures.
//
// Revision 1.45  1998/06/22 19:42:01  daa
// bump to 2.7025.410 this is likely to become 2.7100.411
//
// Revision 1.44  1998/06/22 01:04:56  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h, resolved h.tonl()/n.tohl() conflict with same def in client.h
// (is now inline asm), added NO!NETWORK wrapper around Network::Resolve()
//
// Revision 1.43  1998/06/21 17:10:26  cyruspatel
// Fixed some NetWare smp problems. Merged duplicate numcpu validation code
// in ::ReadConfig()/::ValidateConfig() into ::ValidateProcessorCount() and
// spun that off, together with what used to be ::x86id() or ::ArmId(), into
// cpucheck.cpp. Adjusted and cleaned up client.h accordingly.
//
// Revision 1.42  1998/06/15 09:17:57  jlawson
// removed include of pthread.h since threadcd.h knows when to include it
//
// Revision 1.41  1998/06/14 11:22:37  ziggyb
// Fixed the OS/2 headers and added an os2defs.h and adjusted for the
// separate sleep defines header.
//
// Revision 1.40  1998/06/14 08:12:36  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


#ifndef __CLIBASICS_H__
#define __CLIBASICS_H__

#include "cputypes.h"
#include "stub.h"

// --------------------------------------------------------------------------

#define PACKET_VERSION      0x03
#define MAXBLOCKSPERBUFFER  500
#define CONTEST_COUNT       2 /* 3 contests, only 2 right now (no OGR) */

// --------------------------------------------------------------------------

typedef enum
{
  OP_REQUEST,                     //     obsolete (use OP_REQUEST_MULTI)
  OP_DATA,                        //     transition (use OP_BIGDATA)
  OP_SUCCESS,                     //     obsolete (use OP_SUCCESS_MULTI)
  OP_DONE,                        //     obsolete (use OP_DONE_MULTI)
  OP_FAIL,                        //     never used
  OP_MAX,                         //     never used
  OP_PERMISSION = 42,             // ==> current: full<>master only
  OP_DONE_NOCLOSE,                //     obsolete (use OP_DONE_MULTI)
  OP_DONE_NOCLOSE_ACK,            //     transition (use OP_BIGDONE_ACK)
  OP_PROXYCHANGE,                 // ==> current: full<>master only
  OP_PERREQUEST = 50,             //     obsolete (use OP_PERREQUEST_MULTI)
  OP_PERDATA,                     //     transition (use OP_BIGDATA)
  OP_SCRAMREQUEST = 69,           // ==> current: client->full only
  OP_SCRAM = 96,                  // ==> current: client<-full only
  OP_SUCCESS_ACK,                 // ==> current
  OP_COMPRESSED_FOLLOWS = 100,    //     never used
  OP_REQUEST_MULTI,               //     obsolete (use OP_BIGREQUEST_MULTI)
  OP_SUCCESS_MULTI,               // ==> current
  OP_DONE_MULTI,                  //     transition (use OP_BIGDONE_MULTI)
  OP_PERREQUEST_MULTI,            //     transition (use OP_BIGPERREQUEST_MULTI)
  OP_BIGREQUEST_MULTI,            // ==> current
  OP_BIGDATA,                     // ==> current
  OP_BIGDONE_MULTI,               // ==> current
  OP_BIGDONE_ACK,                 // ==> current
  OP_BIGPERREQUEST_MULTI          // ==> current
} Operation;

// --------------------------------------------------------------------------

typedef enum
{
  IDCONTEST_ANY = 0,
  IDCONTEST_DES = 1,       /* obsolete */
  IDCONTEST_RC556,         /* obsolete */
  IDCONTEST_RC564,
  IDCONTEST_DESII,
  IDCONTEST_OGR
} contest_id_t;

// --------------------------------------------------------------------------

#pragma pack(1)               // no padding allowed

typedef struct
{
  u32 lock;
  u32 count;
} FileHeader;

// note this begins exactly as a ContestWork struct,
// so that we can just pass it to LoadWork.

typedef struct
{
  union {
    struct {
      u64  key;               // starting key
      u64  iv;                // initialization vector
      u64  plain;             // plaintext we're searching for
      u64  cypher;            // cyphertext
      u64  keysdone;          // iterations done (also current position in block)
      u64  iterations;        // iterations to do
    } crypto;
   struct {
     Stub stub;
   } ogr;
  } data;
  u32  op;                // (out)OP_SUCCESS, (out)OP_DONE, or (in)OP_DATA
  char id[59];            // email address of original worker...
  u8   contest;           //
  u8   cpu;               // added 97.11.25
  u8   os;                // added 97.11.25
  u8   buildhi;           // added 97.11.25
  u8   buildlo;           // added 97.11.25
  u32  checksum;          // checksum for file curruption
  u32  scramble;          // scramble key for this entry (NOT! the same as OP_KEY)
} FileEntry;

// remember that for now, the u64's come from the server in the "wrong" order
// lo|hi, not hi|lo like they should
// - everything in network byte order.

typedef struct Packet
{
  u32  op;            // operation code                     }--|    }--|
  u64  key;           // the Key starting point                |       |
  u64  iv;            // the IV                                |       |
  u64  plain;         // the Plaintext                         |       |
  u64  cypher;        // the Cyphertext                        |       |
  u32  iterations;    // number of iterations (the low 32bits) |       |
  char id[64];        // identifier (email address)            |       |
  u32  ip;            // IP Address (proxy filled)             |       |
  u32  iterationshi;  // number of iterations (the high 32bits)|       |
  char other[20];     // extra space                           |       |
  u32  rc564contestdone; //  set to h.tonl(0xBEEFF00D) by the proxies if the rc564 contest is finished
  u32  descontestdone;   //   set to h.tonl(0xBEEFF00D) by the proxies if current des contest is over
  u32  contestid;     // contest identifier                    |       |
  u32  build;         // build number                          |       |
  u32  os;            // OS - see defines                      |       |
  u32  cpu;           // CPU - see defines                     |       |
  u32  version;       // currently set to = 0x00000002      }--|       |
  u32  checksum;      // pre scrambled checksum            ----|    }--|
  u32  scramble;      // the key we're using to scrambling this -------|
} Packet;

#pragma pack()

class Network;            // prototype for referenced classes
class IniSection;

struct membuffstruct 
{ 
  unsigned long count; 
  FileEntry *buff[MAXBLOCKSPERBUFFER];
};


class Client
{
public:
  int  quietmode;
  char inifilename[128];
  char id[64];
  volatile s32  inthreshold[CONTEST_COUNT];
  volatile s32  outthreshold[CONTEST_COUNT];
  s32  blockcount;
  s32  minutes;
  s32  timeslice;
  s32  priority;
  char keyproxy[64];
  s32  keyport;
  char httpproxy[64];
  s32  httpport;
  s32  uuehttpmode;
  char httpid[128];
  s32  cputype;
  s32  offlinemode;
  int  stopiniio;

  s32  messagelen;
  char smtpsrvr[128];
  s32  smtpport;
  char smtpfrom[128];
  char smtpdest[128];

  char logname[128];
  char exit_flag_file[128];
  char checkpoint_file[128];
  char pausefile[128];

  s32 numcpu;
  s32 percentprintingoff;
  s32 connectoften;

  int nodiskbuffers;
  struct { struct membuffstruct in, out; } membufftable[CONTEST_COUNT];
  char in_buffer_file[CONTEST_COUNT][128];
  char out_buffer_file[CONTEST_COUNT][128];
  long PutBufferRecord(const FileEntry *data);
  long GetBufferRecord( FileEntry *data, unsigned int contest, int use_out_file);
  long GetBufferCount( unsigned int contest, int use_out_file, unsigned long *normcountP );
  
  s32 nofallback;
  u32 randomprefix;
  int randomchanged;
  s32 consecutivesolutions[CONTEST_COUNT];
  int autofindkeyserver;
  s32 nonewblocks;
  s32 nettimeout;
  s32 noexitfilecheck;
  s32 preferred_contest_id;  // 0 for RC564, 1 for DESII 
  s32 preferred_blocksize;
  s32 contestdone[CONTEST_COUNT];
  u32 descontestclosed;
  s32 scheduledupdatetime;

#if defined(MMX_BITSLICER) || defined(MMX_RC5)
  int usemmx;
#endif

#if defined(NEEDVIRTUALMETHODS)
  // methods that can be overriden to provide additional functionality
  // virtuals required for OS2 & win GUI clients...
  virtual void InternalReadConfig( IniSection &ini ) {};
  virtual void InternalValidateConfig( void) {};
  virtual void InternalWriteConfig( IniSection &ini ) {};
#endif

  Client();
  ~Client() {};


  int Main( int argc, const char *argv[] );
    // encapsulated main().  client.Main() may restart itself

  int ParseCommandline( int runlevel, int argc, const char *argv[], 
                        int *retcodeP, int logging_is_initialized );
                        
  // runlevel == 0: ReadConfig() (-quiet, -ini, -guistart etc done here too)
  //          >= 1: post-readconfig (override ini options)
  //          == 2: run "modes"

  int CheckpointAction(int action, unsigned int load_problem_count );
    // CHECKPOINT_OPEN (copy from checkpoint to in-buffer), *_REFRESH, *_CLOSE
    // returns !0 if checkpointing is disabled

#if defined(NEEDVIRTUALMETHODS)
  virtual int Configure( void );
    // runs the interactive configuration setup
#else
  int  Configure( void );
    // runs the interactive configuration setup
#endif

  int Run( void );
    // run the loop, do the work
    // returns:
    //    -2 = exit by error (all contests closed)
    //    -1 = exit by error (critical)
    //     0 = exit for unknown reason
    //     1 = exit by user request
    //     2 = exit by exit file check
    //     3 = exit by time limit expiration
    //     4 = exit by block count expiration

  int BufferUpdate( int updatereq_flags, int interactive );
    // pass flags ORd with BUFFERUPDATE_FETCH/*_FLUSH. 
    // if interactive, prints "Input buffer full. No fetch required" etc.
    // returns updated flags or < 0 if offlinemode!=0 or NetOpen() failed.

  int SelectCore(int quietly);
    // always returns zero.
    // to configure for cpu. called before Run() from main(), or for 
    // "modes" (Benchmark()/Test()) from ParseCommandLine().

  unsigned int LoadSaveProblems(unsigned int load_problem_count, int retmode);
    // returns actually loaded problems 
    
};

// --------------------------------------------------------------------------

#endif // __CLIBASICS_H__

