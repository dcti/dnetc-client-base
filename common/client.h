// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.


// For WinNT Service:
//#define WINNTSERVICE "bovrc5nt"
// For Win32 hidden console:
//#define WIN32HIDDEN
// For GUIs and such
//#define NOMAIN

#define NEW_STATS_AND_LOGMSG_STUFF    //if you want a 'better looking' screen
#define NEW_LOGSCREEN_PERCENT_SINGLE  //if the percbar is to stay < 80 chars
//#define PERCBAR_ON_ONE_LINE         //percents for all threads on 1 line


#ifndef CLIENT_H
#define CLIENT_H

// --------------------------------------------------------------------------

#define CLIENT_CONTEST      70
#define CLIENT_BUILD        23
#define CLIENT_BUILD_FRAC   408

// all revision comments moved to the changeLog.txt

#include "cputypes.h"
#include "problem.h"
#include "iniread.h"
#include "network.h"
#include "scram.h"
#include "mail.h"
#include "convdes.h"
#include "sleepdef.h"

#define OPTION_SECTION "parameters"

#define PACKET_VERSION      0x03

#define MAXCPUS             16
#define BUFFER_RETRY        10
#define FETCH_RETRY         10
#define NETTIMEOUT          60

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  #define NEEDVIRTUALMETHODS
#endif

// --------------------------------------------------------------------------

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>

#if (CLIENT_OS == OS_RISCOS)
#include <sys/fcntl.h>
#include <unistd.h>
#else
#include <fcntl.h>
#endif

#if (CLIENT_OS == OS_IRIX)
  #include <limits.h>
  #include <sys/types.h>
  #include <sys/prctl.h>
  #include <sys/schedctl.h>
#elif (CLIENT_OS == OS_OS2)
  // Note: Look in network.h for os2.h defines
  #include <dos.h>
  #include <sys/timeb.h>
  #include <conio.h>
  #include <share.h>
  #include <direct.h>
  #ifndef QSV_NUMPROCESSORS       /* This is only defined in the SMP toolkit */
    #define QSV_NUMPROCESSORS     26
  #endif
#elif (CLIENT_OS == OS_RISCOS)
  extern "C"
  {
    #include <stdarg.h>
    #include <machine/endian.h>
    #include <swis.h>
    extern unsigned int ARMident(), IOMDident();
    extern void riscos_clear_screen();
    extern bool riscos_check_taskwindow();
    extern int riscos_find_local_directory(const char *argv0);
    extern char *riscos_localise_filename(const char *filename);
    extern int getch();

    #define fileno(f) ((f)->__file)
    #define isatty(f) ((f) == 0)
  }
  extern bool riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <types.h>
  #define unlink remove
#elif (CLIENT_OS == OS_SCO)
  #include <sys/time.h>
#elif (CLIENT_OS == OS_WIN16)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <dos.h>
  #include <share.h>
  #include <dir.h>
#elif (CLIENT_OS == OS_WIN32)
  #include <sys/timeb.h>
  #include <process.h>
  #include <ras.h>
  #include <conio.h>
  #include <share.h>
#ifdef __TURBOC__
  #include <dir.h>
#endif
#elif (CLIENT_OS == OS_DOS)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <share.h>
  #if defined(DJGPP)
    #include <dir.h>
    #define ntohl(x) ((((x)<<24) & 0xFF000000) | (((x)<<8) & 0x00FF0000) | (((x)>>8) & 0x0000FF00) | (((x)>>24) & 0x000000FF))
    #define htonl(x) ((((x)<<24) & 0xFF000000) | (((x)<<8) & 0x00FF0000) | (((x)>>8) & 0x0000FF00) | (((x)>>24) & 0x000000FF))
  #else
    #include <sys/stat.h> //S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
  #endif
#elif (CLIENT_OS == OS_BEOS)
// nothing  #include <share.h>
#elif (CLIENT_OS == OS_NETWARE)
  #include <sys/time.h>
  #include <process.h>
  #include <conio.h>     //ConsolePrintf()
  //#include <nwcntask.h>  //ThreadSwitch()
  //#include <nwconn.h>    //for login to remote server
  //#include <nwbindry.h>  //for login to remote server OT_USER
  #include <direct.h>    //for chdir(), getcwd()
  #include <nwfile.h>    //ScanErasedFiles() PurgeErasedFile()
  #include <share.h>

  //used in client.cpp
  extern int  CliIsNetworkAvailable(int dummy);
  extern int  CliSetThreadName( int threadID, int crunchernumber );
  extern int  CliClearThreadContextSpecifier( int threadID );
  extern int  CliGetProcessorCount( void );
  extern int  CliMigrateThreadToSMP( void );
  extern int  CliRunProblemAsCallback( Problem *problem, int timeslice, int cpu_i, int niceness );
  extern int  CliWaitForThreadExit( int threadID );
  extern int  CliInitClient( int argc, char *argv[], void *client );
  extern int  CliValidateSinglePath( char *dest, unsigned int destsize,
                  char *defaultval, int defaultnoneifempty, char *source );
  extern int  CliSetScreenDestructionMode(int newmode);
  extern int  CliKickWatchdog(void);
  extern int  CliExitClient(void); //also used by signal handler/cliconfig.cpp

  //used in cliconfig.cpp
  extern int  CliValidateProcessorCount( int numcpu );
  extern int  CliIsClientRunning(void);
  extern void CliThreadSwitchLowPriority(void);
  extern void CliThreadSwitchWithDelay(void);
  extern int  CliGetSystemConsoleScreen(void);
  extern unsigned int CliGetCurrentTicks(void); //also used in clitime.cpp
  extern int  CliActivateConsoleScreen(void);
  extern void CliConsolePrintf(char *fmt,...);
  extern void CliForceClientShutdown(void);

  //used in clitime.cpp
  extern unsigned int CliConvertTicksToSeconds( unsigned int ticks, unsigned int *secs, unsigned int *hsecs );
  extern unsigned int CliConvertSecondsToTicks( unsigned int secs, unsigned int hsecs, unsigned int *ticks );

  //symbol redefinitions
  int CliGetHostName(char *, unsigned int);  //used in mail.cpp
  long my_inet_addr( char *hostname ); //(in hbyname.cpp) used in network.cpp
  extern int sizonly_stat( const char *fn, struct stat *statblk ); //buffwork.cpp
  extern int purged_unlink( const char *filename ); //buffwork.cpp

  #ifdef gethostname // emulated function in hbyname.cpp
  #undef gethostname
  #endif
  #ifdef inet_addr // emulated function in hbyname.cpp
  #undef inet_addr
  #endif
  #ifdef stat // stat and fstat have problems on nw 3.x
  #undef stat
  #endif
  #ifdef unlink  //purging unlink
  #undef unlink
  #endif

  #define inet_addr(p)           my_inet_addr(p)
  #define gethostname(b,s)       CliGetHostName(b,s)
  #define stat( _fn, _statblk )  sizonly_stat( _fn, _statblk )
  #define unlink( _fn )          purged_unlink( _fn )

#elif (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_SOLARIS)
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int); // Keep g++ happy.
#endif


#if defined(MULTITHREAD) && (CLIENT_OS != OS_WIN32) && (CLIENT_OS != OS_OS2) && (CLIENT_OS != OS_NETWARE) && (CLIENT_OS != OS_BEOS)
#include <pthread.h>
#endif



#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

// --------------------------------------------------------------------------

#ifdef max
#undef max
#endif
#define max(a,b)            (((a) > (b)) ? (a) : (b))

#ifdef min
#undef min
#endif
#define min(a,b)            (((a) < (b)) ? (a) : (b))

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_NETWARE)
#define PATH_SEP   "\\"
#define PATH_SEP_C '\\'
#define EXTN_SEP   "."
#define EXTN_SEP_C '.'
#elif (CLIENT_OS == OS_MACOS)
#define PATH_SEP   ":"
#define PATH_SEP_C ':'
#define EXTN_SEP   "."
#define EXTN_SEP_C '.'
#elif (CLIENT_OS == OS_RISCOS)
#define PATH_SEP   "."
#define PATH_SEP_C '.'
#define EXTN_SEP   "/"
#define EXTN_SEP_C '/'
#else
#define PATH_SEP   "/"
#define PATH_SEP_C '/'
#define EXTN_SEP   "."
#define EXTN_SEP_C '.'
#endif

// --------------------------------------------------------------------------

typedef enum
{
  OP_REQUEST,
  OP_DATA,
  OP_SUCCESS,
  OP_DONE,
  OP_FAIL,
  OP_MAX,
  OP_PERMISSION = 42,
  OP_DONE_NOCLOSE,
  OP_DONE_NOCLOSE_ACK,
  OP_PROXYCHANGE,
  OP_PERREQUEST = 50,
  OP_PERDATA,
  OP_SCRAMREQUEST = 69,
  OP_SCRAM = 96,
  OP_SUCCESS_ACK,
  OP_COMPRESSED_FOLLOWS = 100,
  OP_REQUEST_MULTI,
  OP_SUCCESS_MULTI,
  OP_DONE_MULTI,
  OP_PERREQUEST_MULTI
} Operation;

// --------------------------------------------------------------------------

typedef enum
{
  IDCONTEST_ANY = 0,
  IDCONTEST_DES = 1,       /* obsolete */
  IDCONTEST_RC556,         /* obsolete */
  IDCONTEST_RC564,
  IDCONTEST_DESII
} contest_id_t;

// --------------------------------------------------------------------------

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
  char other[24];     // extra space                           |       |
  u32  rc564contestdone; //  set to htonl(0xBEEFF00D) by the proxies if the rc564 contest is finished
  u32  descontestdone;   //   set to htonl(0xBEEFF00D) by the proxies if current des contest is over
  u32  contestid;     // contest identifier                    |       |
  u32  build;         // build number                          |       |
  u32  os;            // OS - see defines                      |       |
  u32  cpu;           // CPU - see defines                     |       |
  u32  version;       // currently set to = 0x00000002      }--|       |
  u32  checksum;      // pre scrambled checksum            ----|    }--|
  u32  scramble;      // the key we're using to scrambling this -------|
} Packet;

// --------------------------------------------------------------------------

typedef struct
{
  u32 lock;
  u32 count;
} FileHeader;

// --------------------------------------------------------------------------

// note this begins exactly as a ContestWork struct,
// so that we can just pass it to LoadWork.

typedef struct
{
  u64  key;               // starting key
  u64  iv;                // initialization vector
  u64  plain;             // plaintext we're searching for
  u64  cypher;            // cyphertext
  u64  keysdone;          // iterations done (also current position in block)
  u64  iterations;        // iterations to do
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

// --------------------------------------------------------------------------

class Client
{
public:
  char exename[64];
  char exepath[128];
  char inifilename[128];
  char id[64];
  s32  inthreshold[2];
  s32  outthreshold[2];
  s32  blockcount;
  char hours[64];
  s32  minutes;
  s32  timeslice;
  s32  niceness;
  char logname[128];// Logfile name as used by the client
  char ini_logname[128];// Logfile name as is in the .ini
  char keyproxy[64];
  s32  keyport;
  char httpproxy[64];
  s32  httpport;
  s32  uuehttpmode;
  char httpid[128];
  s32  cputype;
  s32  messagelen;
  char smtpsrvr[128];
  s32  smtpport;
  char smtpfrom[128];
  char smtpdest[128];
  s32  offlinemode;
  char in_buffer_file[2][128];
  char ini_in_buffer_file[2][128];
  char out_buffer_file[2][128];
  char ini_out_buffer_file[2][128];
  char exit_flag_file[128];
  char ini_exit_flag_file[128];
  MailMessage mailmessage;
  s32  numcpu, numcputemp;
  char checkpoint_file[2][128];
  char ini_checkpoint_file[2][128];
  s32 checkpoint_min;
  s32 percentprintingoff;
  s32 connectoften;
  s32 nodiskbuffers;
  s32 membuffcount[2][2];
  FileEntry *membuff[2][500][2];
  s32 nofallback;
  s32 randomprefix;
  s32 randomchanged;
  s32 consecutivesolutions[2];
  s32 quietmode;
  s32 nonewblocks;
  s32 nettimeout;
  s32 noexitfilecheck;
  s32 exitfilechecktime;
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
  s32 lurk;
#if (!defined(WINNTSERVICE))
  s32 win95hidden;
#endif
#endif
#if (CLIENT_OS == OS_OS2)
  s32 os2hidden;
  s32 connectstatus;          // 0 is not connected, 1 is connected
#endif
  char pausefile[128];
  char ini_pausefile[128];
  s32 preferred_contest_id;  // 0 for RC564, 1 for DESII (unlike config)
  s32 preferred_blocksize;
  s32 contestdone[2];

protected:
  char proxymessage[64];
  char logstr[1024];

  ContestWork contestwork;
  RC5Result rc5result;

  u32 totalBlocksDone[2];
  u32 old_totalBlocksDone[2];
  u32 timeStarted;

#if defined(MULTITHREAD)
  u32 connectrequested;
#endif

#if (CLIENT_OS == OS_WIN16)
  virtual void SurrenderCPU( void ) {};
    // pause for other tasks
#endif

#ifndef NEW_STATS_AND_LOGMSG_STUFF //Substituted by CliTimeString(NULL,1)
  const char * Time( void );
    // Returns: string with GMT time in "mm/dd/yy hh:mm:ss"
#endif

  void RandomWork( FileEntry * data );
    // returns a random block.

  // these put/get data blocks to the file buffers
  // and trigger Fetch/Flush when needed
  s32  PutBufferInput( const FileEntry * data );  // hidden
    // dump an entry into in buffer
    // Returns: -1 on error, otherwise number of entries now in file.

  s32  GetBufferInput( FileEntry * data, u8 contest );        // "user"
    // get work from in buffer
    // Returns: -1 on error, otherwise number of blocks now in file.

  s32  CountBufferInput(u8 contest);
    // returns number of blocks currently in input buffer

  s32  PutBufferOutput( const FileEntry * data );  // "user"
    // dump an entry into output buffer
    // Returns: -1 on error, otherwise number of entries now in file.

  s32  GetBufferOutput( FileEntry * data , u8 contest);       // hidden
    // get work from output buffer
    // Returns: -1 on error, otherwise number of blocks now in file.

  s32  CountBufferOutput(u8 contest);
    // returns number of blocks currently in output buffer

  s32 InternalPutBuffer( const char *filename, const FileEntry * data );
  s32 InternalGetBuffer( const char *filename, FileEntry * data, u32 *optype , u8 contest);
  s32 InternalCountBuffer( const char *filename , u8 contest);
  static const char *InternalGetLocalFilename( const char *filename );
  s32 EnsureBufferConsistency( const char *filename );

#if defined(NEEDVIRTUALMETHODS)
  // methods that can be overriden to provide additional functionality
  // virtuals required for OS2 & win GUI clients...
  virtual void InternalReadConfig( IniSection &ini ) {};
  virtual void InternalValidateConfig( void) {};
  virtual void InternalWriteConfig( IniSection &ini ) {};
#endif

  bool CheckForcedKeyport(void);

  void SetNiceness(void);
    // set the client niceness

public:
  Client();
  ~Client();

  void PrintBanner(const char * clname);
    // prints out a version banner to screen

  void ParseCommandlineOptions(int Argc, char *Argv[], s32 &inimissing);
    // parses commandline options, setting parsed items to NULL

  s32 CkpointToBufferInput(u8 contest);
    // Copies info in checkpint file back to input buffer

  void DoCheckpoint( int load_problem_count );
    // Make the checkpoint file represent current blocks being worked on


  void DisplayHelp( const char * unrecognized_option );
    // Displays the interactive command line help screen.

#if defined(NEEDVIRTUALMETHODS)
  virtual s32  Configure( void );
    // runs the interactive configuration setup
#else
  s32  Configure( void );
    // runs the interactive configuration setup
#endif


  s32  ConfigureGeneral( s32 currentmenu );
    // part of the interactive setup

  static s32 yesno(char *str);
    // Checks whether user typed yes or no, used in interactive setup
    // Returns 1=yes, 0=no, -1=unknown

  static s32 findmenuoption( s32 menu, s32 option);
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.

  void setupoptions( void );
    // Sets all the pointers/etc for optionstruct options

  static void killwhitespace( char *string );
    // Removes all spaces from a string

  static int isstringblank( char *string );
    // returns 1 if a string is blank (or null), 0 if it is not

  static void clearscreen( void );
    // Clears the screen. (Platform specific ifdefs go inside of it.)

  s32  ReadConfig( void );
    // returns -1 if no ini exits, 0 otherwise

  void ValidateConfig( void );
    // verifies configuration and forces valid values

  s32  WriteConfig( void );
    // returns -1 on error, 0 otherwise

  s32  WriteContestandPrefixConfig( void );
    // returns -1 on error, 0 otherwise
    // only writes contestdone and randomprefix .ini entries

  u32  Benchmark( u8 contest, u32 numk );
    // returns keys/second

  s32  SelfTest( u8 contest );
    // run all the test keys
    // Returns: number of tests passed, or negative number of test that failed

  s32 Run( void );
    // run the loop, do the work
    // returns:
    //    -2 = exit by error (all contests closed)
    //    -1 = exit by error (critical)
    //     0 = exit for unknown reason
    //     1 = exit by user request
    //     2 = exit by exit file check
    //     3 = exit by time limit expiration
    //     4 = exit by block count expiration

  s32  Fetch( u8 contest, Network *netin = NULL );
    // fills up all of the input buffers
    // this is for sneakernet support amung other things
    // Returns: number of buffers received, negative if some error occured

  s32  ForceFetch( u8 contest, Network *netin = NULL );
    // Like fetch, but keeps trying until done or until buffer size doesn't get bigger
    // Basically, ignores premature disconnection.

  s32  Flush( u8 contest, Network *netin = NULL );
    // flushes out result buffers, useful when a SUCCESS happens
    // Also remove buffer file stub if done
    // this is for sneakernet support
    // Returns: number of buffers sent, negative if some error occured

  s32  ForceFlush( u8 contest, Network *netin = NULL );
    // Like flush, but keeps trying until done or until buffer size doesn't get smaller
    // Basically, ignores '31' and '32' type errors -- bad buffer file entry problems.

  s32 Update( u8 contest, s32 fetcherr, s32 flusherr);
    // flushes out buffer, and fills input buffers
    // returns: number of buffers sent, negative if some error occurred
    // Return value is the return from fetch*fetcherr +
    //                      value from flush*flusherr

  void Log( const char *format, ... );
    // logs message to screen and file (append mode)
    // if logname isn't set, then only to screen

  void LogScreenf( const char *format, ... );
    // logs message to screen only

#if defined(NEEDVIRTUALMETHODS)
  virtual void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#else
  void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#endif


#if defined(NEEDVIRTUALMETHODS)
  virtual void LogScreenPercentSingle(u32 percent, u32 lastpercent, bool restarted);
  virtual void LogScreenPercentMulti(u32 cpu, u32 percent, u32 lastpercent, bool restarted);
    // progress percentage printing to screen only.
#else
  void LogScreenPercentSingle(u32 percent, u32 lastpercent, bool restarted);
  void LogScreenPercentMulti(u32 cpu, u32 percent, u32 lastpercent, bool restarted);
    // progress percentage printing to screen only.
#endif


  s32 Install();
    // installs the clients into autolaunch configuration
    // returns: non-zero on failure

  s32 Uninstall(void);
    // removes the client from autolaunch configuration
    // returns: non-zero on failure

  s32 RunStartup(void);
    // to be called before calling Run() for the first time
    // returns: non-zero on failure

  s32 SelectCore(void);
    // to be called before Run(), Benchmark(), or Test() to configure for cpu
    // returns: non-zero on failure

  s32 UnlockBuffer( const char *filename );
    // unlock buffer 'filename'

#if (CLIENT_CPU == CPU_X86)
  int x86id();
    // Identify CPU type
#elif (CLIENT_CPU == CPU_ARM)
  int ARMid();
    // Identify CPU type
#endif

  s32 SetContestDoneState( Packet * packet);
    // Set the contest state appropriately based on packet information
    // Returns 1 if a change to contest state was detected

};

// --------------------------------------------------------------------------

#ifdef NEW_STATS_AND_LOGMSG_STUFF
  #include "clitime.h"
  #include "clirate.h"
  #include "clisrate.h"
  #define Time() (CliGetTimeString(NULL,1))
#else
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
    #ifdef __WATCOMC__
      // disable "Warning! W481: col(1) class/enum has the same name as the function/variable 'timezone'"
      #pragma warning 481 9 ;
    #endif
    struct timezone
    {
      int  tz_minuteswest;    /* of Greenwich */
      int  tz_dsttime;        /* type of dst correction to apply */
    };
  #endif
  #if (((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU == CPU_68K)) ||   \
     (CLIENT_OS == OS_MACOS) ||                              \
     (CLIENT_OS == OS_SCO) ||                                \
     (CLIENT_OS == OS_OS2) ||                                \
     (CLIENT_OS == OS_WIN32) ||                              \
     (CLIENT_OS == OS_AMIGAOS) ||                            \
     (CLIENT_OS == OS_NETWARE) ||                            \
     (CLIENT_OS == OS_WIN16) ||                              \
     (CLIENT_OS == OS_DOS) ||                                \
     ((CLIENT_OS == OS_VMS) && !defined(MULTINET)))
  extern "C" gettimeofday(struct timeval *tv, struct timezone *);
  #endif
#endif //#ifdef NEW_STATS_AND_LOGMSG_STUFF

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)
  #ifdef __WATCOMC__
    #define x86ident _x86ident
//    #define checklocks _checklocks
  #endif
  #if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
    extern "C" u32 x86ident( void ) asm ("x86ident");
  #else
    extern "C" u32 x86ident( void );
  #endif
//  extern "C" u16 checklocks( void );
#endif


extern Problem problem[2*MAXCPUS];

extern volatile u32 SignalTriggered, UserBreakTriggered;
extern volatile s32 pausefilefound;
extern void CliSetupSignals( void );

#if (CLIENT_OS == OS_RISCOS)
extern s32 guiriscos, guirestart;
#endif

#if (CLIENT_OS == OS_WIN32)
typedef DWORD (CALLBACK *rasenumconnectionsT)(LPRASCONN, LPDWORD, LPDWORD);
typedef DWORD (CALLBACK *rasgetconnectstatusT)(HRASCONN, LPRASCONNSTATUS);
extern rasenumconnectionsT rasenumconnections;
extern rasgetconnectstatusT rasgetconnectstatus;
#endif

// --------------------------------------------------------------------------


#endif

