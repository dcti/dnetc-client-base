// For WinNT Service:
//#define WINNTSERVICE "bovrc5nt"
// For Win32 hidden console:
//#define WIN32HIDDEN
// For GUIs and such
//#define NOMAIN

#define NEW_STATS_AND_LOGMSG_STUFF   //if you want a 'better looking' screen
#define NEW_LOGSCREEN_PERCENT_SINGLE //if the percbar is to stay < 80 chars
//#define PERCBAR_ON_ONE_LINE //prints percs for all (max 16) threads on one line
                     //platform must support "\r" to return to start of line

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#ifndef CLIENT_H
#define CLIENT_H

// --------------------------------------------------------------------------

#define CLIENT_CONTEST      70
#define CLIENT_BUILD        21
#define CLIENT_BUILD_FRAC   405
// v2.7021.405 98.05.21 OS/2 changes from Oscar Chang
// v2.7021.405 98.05.21 New parisc rc5 core from Katsuomi Hamajima
// v2.7021.405 98.05.21 Added Banner message crediting Matthew Kwan
// v2.7021.405 98.05.21 OS390 changes from Friedemann Baitinger
// v2.7021.405 98.05.21 New cpu/os codes added OS_OS390=39, CPU_S390=14
// v2.7021.405 98.05.21 ARM/riscos changes from Chris Berry
// v2.7021.405 98.05.21 Rebuilt bdeslow.aout.o/bbdeslow.aout.o and updated des-x86.cpp
// v2.7021.404 98.05.19 Increased temp space allocated for headers in http mode
// v2.7021.404 98.05.17 "-n" setting now forces a minimum of 1 block
// v2.7021.404 98.05.04 ForcedKeyPort changes for euro23.v27.distributed.net/jp.v27.distributed.net/port 3064
// v2.7021.404 98.05.04 New directory structure, configure script
// v2.7020.403 98.04.28 Fixed starting of NT service from command line
// v2.7020.403 98.04.28 Fixed DOS buffer problems
// v2.7020.403 98.04.28 Fixed bug in -offlinemode, when ini file doesn't know that DES is over.
// v2.7020.403 98.04.23 Improved x86 chip identification
// v2.7020.403 98.04.23 New sboxes-kwan3.cpp and sboxes-kwan4.cpp
// v2.7020.403 98.04.21 Fixed final buffer flush when using "nodisk" mode.
// v2.7020.403 98.04.21 stricmp references changed to strcmpi in cliconfig.cpp
// v2.7020.403 98.04.21 Change to ansi rc5 core rc5ansi2-rg.cpp (4 macros).
// v2.7020.403 98.04.21 Very minor change to p5 core (not even a cycle, really).
// v2.7020.403 98.04.21 Added "-forcefetch"/"-forceflush".  These deal with corrupted buffer entries much better.
// v2.7020.403 98.04.21 RISCOS changes
// v2.7020.403 98.04.21 Updated strongARM core (in source -- it was used in last compile already)
// v2.7019.402 98.04.14 Fixed CPU detection for K6/6x86
// v2.7019.402 98.04.14 Win16 changes
// v2.7019.402 98.04.13 Renamed problem.H to problem.h
// v2.7019.402 98.04.13 Fixed Socks4/Socks5 support
// v2.7019.402 98.04.13 Arm/StrongArm fixes
// v2.7019.402 98.04.13 Added aflags to open call in buffwork/sh_fopen
// v2.7019.402 98.04.03 Added SOCKS5 support (with username/password and no auth)
// v2.7019.402 98.04.02 Fixed SOCKS4 support which wasn't working.
// v2.7019.402 98.04.02 Use only correct port with well-known keyserver RRs.
// v2.7018.401 98.04.02 Client::Run returns 'fuller' return codes
// v2.7018.401 98.04.02 Benchmark / test now do both contests
// v2.7018.401 98.04.02 Renamed rc5.* to problem.*
// v2.7018.401 98.04.02 Removed OS2_PM #defines
// v2.7018.401 98.04.02 Fixed dates in mail messages
// v2.7018.401 98.04.02 SUNOS changes
// v2.7018.401 98.04.02 adjusted the #defines in des-x86.cpp to link successfully under linux
// v2.7018.401 98.04.02 Fixed sopen() problem in buffwork.cpp which affected many OSes
// v2.7018.401 98.04.02 New HP-PA RC5 core from Katsuomi Hamajima <hamajima@ydc.co.jp>
// v2.7018.401 98.04.02 Lurk mode 'sticks' now in win32gui client.
// v2.7017.400 98.03.15 New K6 core
// v2.7017.400 98.03.15 Updated Alpha RC5 core (by Pedro Miguel Teixeira)
// v2.7016.399 98.03.13 Fixed buffer flushing problem
// v2.7015.398 98.03.08 Memory buffers are now allocated only as they're needed.
// v2.7015.398 98.03.07 Added 'contestdone' flags to ini to smooth contest transitions
// v2.7015.398 98.03.06 Fixed problem with "-2" sized buffers
// v2.7015.398 98.03.06 Changed default exitfilecheck time to 30 seconds
// v2.7015.398 98.03.06 Sun/68K changes
// v2.7015.398 98.03.06 Fixed shared buffer open problem
// v2.7015.398 98.03.06 Fixed bug where mail might be sent, even in offline mode.
// v2.7015.398 98.03.06 Fixed treatment of "empty" checkpoint filenames.
// v2.7015.398 98.03.06 Fixed key speed calculation for 2^31 size DES blocks
// v2.7015.398 98.03.06 Fixed block size reported for 2^31 DES blocks
// v2.7014.397b 98.02.27 New x86 CPP cores which fix the core dump problem
// v2.7014.397b 98.02.23 Netware changes
// v2.7014.397 98.02.17 Minor code cleanup
// v2.7014.397 98.02.12 AIX Changes, OSF changes
// v2.7013.396c 98.02.09 Older slice routines moved to oldslice.zip
// v2.7013.396c 98.02.09 HTTP modes no longer force port keyport=2064.
// v2.7013.396b 98.02.08 Made "-quiet" even quieter
// v2.7013.396b 98.02.08 Fixed fault when missing final parameter of 2-param option
// v2.7013.396b 98.02.08 Digital Unix patches
// v2.7013.396 98.02.08 AIX / SPARC changes
// v2.7013.396 98.02.08 BDESLOW.S/BDESLW1.S cores from Sven Mikkelsen (AT&T syntax)
// v2.7013.396 98.02.08 Fixed mail bug related to sending empty mail messages
// v2.7012.395 98.02.05 Yet another new bitslice driver from Andrew Meggs
// v2.7011.394b 98.02.01 "Connecting to..." message in network.cpp
// v2.7011.394 98.02.01 Watcom link order changed
// v2.7011.394 98.02.01 OS2 GUI changes for DES tests
// v2.7011.394 98.02.01 Macro collission problem fixed
// v2.7011.394 98.02.01 Netware changes
// v2.7010.393 98.01.30 New bitslice driver and s-boxes from Andrew Meggs
// v2.7010.393 98.01.30 Got RC5 assembly running on AIX/PowerPC client.
// v2.7010.393 98.01.30 Fixed mailing to multiple destinations
// v2.7010.392 98.01.30 NT Service version default startup option changed to auto-start
// v2.7010.392 98.01.30 Warning message about "-hide" when running win32 client on NT
// v2.7010.392 98.01.28 BEOS, HP, Netware changes
// v2.7009.391 98.01.25 Added duplicate x86 core to allow 2 DES threads.
// v2.7008.390 98.01.25 New non-x86 DES core routines
// v2.7008.390 98.01.25 Netware changes
// v2.7008.390 98.01.24 -runbuffers/-runoffline ignored when doing fetch/flush/update
// v2.7007.389 98.01.21 "sent to server"/"received from server" messages include DES/RC5 type
// v2.7007.389 98.01.20 CPU identification for non-unix X86 from Cyrus Patel
// v2.7006.388 98.01.20 Newer x86 DES core -- 8% faster on PPros
// v2.7006.388 98.01.20 Fixed contestdone[] treatment in client.cpp
// v2.7005.387 98.01.19 Added 2nd method for proxies to notify clients about contest status.
// v2.7004.387 98.01.18 DES Bitslice cores from Remi
// v2.7004.386 98.01.17 Client will properly identify a 3*2^28 block now.
// v2.7004.386 98.01.17 Fixed negative block sizes
// v2.7004.386 98.01.17 Added ansi core des routine
// v2.7004.386 98.01.17 632 byte memory leak on some failed network Open()s
// v2.7003.385 98.01.15 Don't process a partial block started on another cpu/os/build
// v2.7002.384 98.01.13 RISCOS changes
// v2.7002.383 98.01.13 Banner problem
// v2.7002.382 98.01.13 Client shows size of block being processed
// 382 98.01.13 Win32 clients no longer set  processor affinity when only 1
//              cpu is configured.
// 382 98.01.13 Fixed RC5 blocks being sent to server with wrong contestid
//              when they were downloaded by older non-dual clients.
// 381 98.01.12 New RC5 cores (cpp wasn't updated before)
// 380 98.01.12 Fixed up code to detect end of DES contest (again.  Ugh.)
// 379 98.01.12 Fixed speed reporting on individual blocks.
// 378 98.01.12 Fixed up code to detect end of DES contest
// 377 98.01.12 Svend Olaf Mikkelsen credit
// 376 98.01.12 Problem with block counting fixed
// 375 98.01.12 Cosmetic changes to key speed reporting
// 369 98.01.11 des key incrementation stuff from Remi, new des-x86.cpp
// 368 98.01.10 X86 des core from Remi Guyomarch
// 357 98.01.06 RISCOS changes
// 357 98.01.06 QNX Changes
// 365 98.01.06 Initial Changes to incorporate DESII contest
// 356 98.01.06 SOCKS4 support added
// 356 98.01.06 HTTP network fix
// 355 98.01.06 BEOS changes
// 355 98.01.06 K6 core removed (486 core used in this case as it's actually faster)
// 350 98.01.05 New p5/k5/486/6x86 cores (p5 by Bruce Ford b.ford@qut.edu.au)
// 346 98.01.04 minor change in network.cpp
// 345 98.01.03 Limited checking of exitrc5.now file to once every 3 seconds
// 340 98.01.03 Checkpoint files re-written immediately after blocks finished
// 340 98.01.03 more natural http/uue mode selection for Network class
// 340 98.01.03 keyserver port can now be explictly specified when using http
// 340 98.01.03 PutBufferOutput() no longer returns error if Update() fails.
//              Only actual buffer updating problems result in an error.
// 340 98.01.01 VMS changes
// 340 97.12.31 Sparc changes (mt)
// 340 97.12.31 Random prefix changes cause ini file to be rewritten sooner
// 340 97.12.31 Firemodes 2/3 now default to rc5proxy23.distributed.net
// 340 97.12.31 Fixed exit problem that occasionally caused blocks to be lost
// 340 97.12.31 OS2 GUI changes
// 335 97.12.25 Fixed issue that some clients don't read their checkpoint files
// 335 97.12.25 PPC Core changes
// 330 97.12.19 StrongArm/RiscOS changes
// 325 97.12.16 PPC/Linux changes for core selection
// 320 97.12.16 Fixed date field in mail messages
// 315 97.12.16 Buffers were all being labelled as version 6401.  fixed.  ugh.
// 310 97.12.15 Fixed problem with "-ini" command line option
// 305 97.12.15 New Random number generator added (with much longer periodicity)
// 305 97.12.15 Fixed problem with sharing 6401 buffers (lost email/cpu/ver information)
// 305 97.12.15 Additions to source tree of Service / GUI / Screensaver changes
// 300 97.12.10 PPC changes
// 295 97.12.10 NetBSD changes
// 290 97.12.08 Fixed problems caused by buffers of size 0
// 285 97.12.08 Increased network pause for Solaris clients
// 285 97.12.08 New network.h/network.cpp updates from Jeff Lawson (no changes
//              that affect clients, but server code updated)
// 280 97.12.08 Fixed serious bug affecting some blocks re-retrieved from buff-in.rc5
//              Version number updated to 6403
// 275 97.12.04 Updated for win16 GUI changes
//     97.12.04 Added 5 second pause in MT clients when quitting to allow
//              child threads to quit.
//     97.12.04 InternalReadConfig/WriteConfig/ValidateConfig embedded in #if
// 270 97.12.01 Random blocks now use (prior prefix) + 1 to avoid
//              generating blocks that have been checked
//     97.12.01 "The proxy says" messages not printed when network errors occur
//              to prevent logfile overflow
//     97.12.01 Build "fraction" added to logfile/mail logs
//     97.12.02 Made InternalReadConfig/InternalWriteConfig/InternalValidateConfig virtual
//              for Win32/OS2
//     97.12.02 Small message on x86 clients when autodetecting CPU indicating that
//              it's only a guess
//     97.12.02 Client will now 'pause' when a 'pause' file is detected
//              (configure with "-pausefile filename" on command line, or pausefile=fn in ini)
// 265 97.12.01 Added Y2K support to print routines
// 260 97.11.29 Fixed support for >=2 crack threads (broken 11.12)
// 255          Move to 6402 -- *.rc5 files hold cpu/os information

#include "cputypes.h"
#include "problem.h"
#include "iniread.h"
#include "network.h"
#include "scram.h"
#include "mail.h"
#include "convdes.h"

#define OPTION_SECTION "parameters"

#define PACKET_VERSION      0x03

#define MAXCPUS             16
#define BUFFER_RETRY        10
#define FETCH_RETRY         10
#define NETTIMEOUT          60

// --------------------------------------------------------------------------

#if ((CLIENT_OS == OS_AMIGA) || (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <stdarg.h>
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
    #include <kernel.h>
    extern unsigned int find_core(void);
    extern void riscos_check_taskwindow(int *);
    extern int riscos_find_local_directory(const char *argv0);
    extern char *riscos_localise_filename(const char *filename);

    extern int my_stat(const char *filename, struct stat *buf);
    extern FILE *my_fopen(const char *filename, const char *mode);
    extern int my_unlink(const char *filename);

    #define fopen(f,m) my_fopen(f,m)
    #define unlink(f) my_unlink(f)
    #define stat(f,s) my_stat(f,s)
    }
  extern int riscos_in_taskwindow;
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
#elif (CLIENT_OS == OS_DOS)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <share.h>
  #if defined(DJGPP)
    #include <dir.h>
  #else //if (!defined(DJGPP))
    #include <dos.h> //sleep, usleep
    #include <sys/stat.h> //S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
  #endif
#elif (CLIENT_OS == OS_WIN32)
  #include <sys/timeb.h>
  #include <process.h>
  #include <ras.h>
  #include <conio.h>
  #include <share.h>
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

#elif (CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K)
  extern "C" int nice(int); // Keep g++ happy.
#endif

#if defined(MULTITHREAD) && (CLIENT_OS != OS_WIN32) && (CLIENT_OS != OS_OS2) && (CLIENT_OS != OS_NETWARE) && (CLIENT_OS != OS_BEOS)
#include <pthread.h>
#endif



#if ((CLIENT_OS == OS_AMIGA) || (CLIENT_OS == OS_RISCOS))
}
#endif

// --------------------------------------------------------------------------

#ifndef NEW_STATS_AND_LOGMSG_STUFF
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
#endif

#if ((CLIENT_OS == OS_DOS) && defined(DOS4G))
#define ntohl(x) ((((x)<<24) & 0xFF000000) | (((x)<<8) & 0x00FF0000) | (((x)>>8) & 0x0000FF00) | (((x)>>24) & 0x000000FF))
#define htonl(x) ((((x)<<24) & 0xFF000000) | (((x)<<8) & 0x00FF0000) | (((x)>>8) & 0x0000FF00) | (((x)>>24) & 0x000000FF))
#endif

#define max(a,b)            (((a) > (b)) ? (a) : (b))
#if (CLIENT_OS == OS_OS2)
#undef min
#endif
#define min(a,b)            (((a) < (b)) ? (a) : (b))

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
//  char id[64];            // email address of original worker...
  char id[59];            // email address of original worker...
  u8   contest;
  u8   cpu;        // added 97.11.25
  u8   os;         // added 97.11.25
  u8   buildhi;    // added 97.11.25
  u8   buildlo;    // added 97.11.25
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
  char logname[128];
  s32  firemode;
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
  char out_buffer_file[2][128];
  char exit_flag_file[128];
  MailMessage mailmessage;
  s32  numcpu, numcputemp;
  char checkpoint_file[2][128];
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
#if (CLIENT_OS == OS_WIN32)
  s32 lurk;
#if (!defined(WINNTSERVICE))
  s32 win95hidden;
#endif
#endif
#if (CLIENT_OS == OS_OS2)
  s32 os2hidden;
#endif
  char pausefile[128];
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
  char *InternalGetLocalFilename( char *filename );
  s32 EnsureBufferConsistency( const char *filename );

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
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

  void PrintBanner(char * clname);
    // prints out a version banner to screen

  void ParseCommandlineOptions(int argc, char *argv[], s32 &inimissing);
    // parses commandline options, setting parsed items to NULL

  s32 CkpointToBufferInput(u8 contest);
    // Copies info in checkpint file back to input buffer

  void DoCheckpoint(void);
    // Make the checkpoint file represent current blocks being worked on

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  // virtuals required for OS2 & win GUI clients...
  virtual s32  Configure( void );
#else
  s32  Configure( void );
#endif
    // runs the interactive configuration setup

  s32  ConfigureGeneral( int currentmenu );
    // part of the interactive setup

  s32 yesno(char *str);
    // Checks whether user typed yes or no, used in interactive setup
    // Returns 1=yes, 0=no, -1=unknown

  void clearscreen( void );
    // Clears the screen. (Platform specific ifdefs go inside of it.)

  s32  ReadConfig( void );
    // returns -1 if no ini exits, 0 otherwise

  void ValidateConfig( void );
    // verifies configuration and forces valid values

  s32  WriteConfig( void );
    // returns -1 on error, 0 otherwise

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

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  // virtuals required for OS2 & win GUI clients...
  virtual void LogScreen ( const char *text );
#else
  void LogScreen ( const char *text );
#endif
    // logs preformated message to screen only.  can be overriden.

#ifdef SMART_COMPLETION_CHECKING
  int GetProblemState(void);
  #if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  virtual void LogScreenPercent( u32 percent, u32 lastpercent, u32 restarted );
  #else
  virtual void LogScreenPercent( u32 percent, u32 lastpercent, u32 restarted );
  #endif
#else
  #if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  // virtual required for OS2 and win GUI clients...
  virtual void LogScreenPercentSingle(u32 percent, u32 lastpercent, bool restarted);
  virtual void LogScreenPercentMulti(u32 cpu, u32 percent, u32 lastpercent, bool restarted);
  #else
  void LogScreenPercentSingle(u32 percent, u32 lastpercent, bool restarted);
  void LogScreenPercentMulti(u32 cpu, u32 percent, u32 lastpercent, bool restarted);
  #endif
    // progress percentage printing to screen only.
#endif // SMART_COMPLETION_CHECKING

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
#endif

  s32 Client::SetContestDoneState( Packet * packet);
    // Set the contest state appropriately based on packet information
    // Returns 1 if a change to contest state was detected

};

// --------------------------------------------------------------------------
#if (CLIENT_CPU == CPU_X86)
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_OS2)
  #ifdef __WATCOMC__
    #define x86ident _x86ident
//    #define checklocks _checklocks
  #endif
  extern "C" u32 x86ident( void );
//  extern "C" u16 checklocks( void );
#endif
#endif

#ifndef NEW_STATS_AND_LOGMSG_STUFF
#if ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU == CPU_68K) ||   \
     (CLIENT_OS == OS_MACOS) ||                              \
     (CLIENT_OS == OS_SCO) ||                                \
     (CLIENT_OS == OS_OS2) ||                                \
     (CLIENT_OS == OS_WIN32) ||                              \
     (CLIENT_OS == OS_AMIGA) ||                              \
     (CLIENT_OS == OS_NETWARE) ||                            \
     (CLIENT_OS == OS_WIN16) ||                              \
     (CLIENT_OS == OS_DOS) ||                                \
     ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
  extern "C" gettimeofday(struct timeval *tv, struct timezone *);
#endif
#endif //#ifdef NEW_STATS_AND_LOGMSG_STUFF

extern Problem problem[2*MAXCPUS];
#if defined(MULTITHREAD)
extern volatile s32 ThreadIsDone[2*MAXCPUS];
#endif

extern volatile u32 SignalTriggered, UserBreakTriggered;
extern volatile s32 pausefilefound;
extern void CliSetupSignals( void );


#if (CLIENT_OS == OS_WIN32)
typedef DWORD (CALLBACK *rasenumconnectionsT)(LPRASCONN, LPDWORD, LPDWORD);
typedef DWORD (CALLBACK *rasgetconnectstatusT)(HRASCONN, LPRASCONNSTATUS);
extern rasenumconnectionsT rasenumconnections;
extern rasgetconnectstatusT rasgetconnectstatus;
#endif

// --------------------------------------------------------------------------

#ifdef NEW_STATS_AND_LOGMSG_STUFF
  #include "clitime.h" //CliTimer, CliGetTimeString
  #include "clirate.h"  //CliGetKeyrateForProblem
  #include "clisrate.h" //CliGetSummaryStringForContest, CliGetMessageForFileentryLoaded, CliGetMessageForProblemCompleted
  #define Time() (CliGetTimeString(NULL,1))
  /*
  int gettimeofday(struct timeval *tv, struct timezone *tz)
    { tz=tz; CliTimer( tv ); return 0; }
  const char * Client::Time( void )  
    { return CliGetTimeString(NULL,1); }
  */
#endif

#endif
