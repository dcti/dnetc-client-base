// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: client.cpp,v $
// Revision 1.118  1998/08/07 10:59:11  cberry
// Changed handling of -benchmarkXXX so it performs the benchmark rather than giving the menu.
//
// Revision 1.117  1998/08/05 18:28:40  cyruspatel
// Converted more printf()s to LogScreen()s, changed some Log()/LogScreen()s
// to LogRaw()/LogScreenRaw()s, ensured that DeinitializeLogging() is called,
// and InitializeLogging() is called only once (*before* the banner is shown)
//
// Revision 1.116  1998/08/02 16:17:37  cyruspatel
// Completed support for logging.
//
// Revision 1.115  1998/08/02 05:36:19  silby
// Lurk functionality is now fully encapsulated inside the Lurk Class, much less code floating inside client.cpp now.
//
// Revision 1.114  1998/08/02 03:16:31  silby
// Major reorganization:  Log,LogScreen, and LogScreenf are now in logging.cpp, and are global functions - client.h #includes logging.h, which is all you need to use those functions.  Lurk handling has been added into the Lurk class, which resides in lurk.cpp, and is auto-included by client.h if lurk is defined as well. baseincs.h has had lurk-specific win32 includes moved to lurk.cpp, cliconfig.cpp has been modified to reflect the changes to log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead of printf. client.cpp has had variable names changed as well, etc.
//
// Revision 1.113  1998/07/30 05:08:59  silby
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being included, now they are not. Also, added the logic for dialwhenneeded, which is a new lurk feature.
//
// Revision 1.112  1998/07/30 02:18:18  blast
// AmigaOS update
//
// Revision 1.111  1998/07/29 05:14:40  silby
// Changes to win32 so that LurkInitiateConnection now works - required the addition of a new .ini key connectionname=.  Username and password are automatically retrieved based on the connectionname.
//
// Revision 1.110  1998/07/26 12:45:52  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.109  1998/07/25 06:31:39  silby
// Added lurk functions to initiate a connection and hangup a connection.  win32 hangup is functional.
//
// Revision 1.108  1998/07/25 05:29:49  silby
// Changed all lurk options to use a LURK define (automatically set in client.h) so that lurk integration of mac/amiga clients needs only touch client.h and two functions in client.cpp
//
// Revision 1.107  1998/07/20 00:32:19  silby
// Changes to facilitate 95 CLI/NT service integration
//
// Revision 1.106  1998/07/19 14:42:12  cyruspatel
// NetWare SMP adjustments
//
// Revision 1.105  1998/07/16 19:19:36  remi
// Added -cpuinfo option (you forget this one cyp! :-)
//
// Revision 1.104  1998/07/16 16:58:58  silby
// x86 clients in MMX mode will now permit des on > 2 processors.  Bryddes is still set at two, however.
//
// Revision 1.103  1998/07/16 08:25:07  cyruspatel
// Added more NONETWORK wrappers around calls to Update/Fetch/Flush. Balanced
// the '{' and '}' in Fetch and Flush. Also, Flush/Fetch will now end with
// 100% unless there was a real send/retrieve fault.
//
// Revision 1.101  1998/07/15 06:58:03  silby
// Changes to Flush, Fetch, and Update so that when the win32 gui sets connectoften to initiate one of the above more verbose feedback will be given.  Also, when force=1, a connect will be made regardless of offlinemode and lurk.
//
// Revision 1.100  1998/07/15 06:10:54  silby
// Fixed an improper #ifdef
//
// Revision 1.99  1998/07/14 04:37:33  cramer
// -runbuffers now works (as far as I've been able to tell)  It will process
// blocks from the buffer files until all blocks have been exhausted and then
// exit.  [look in common/client.cpp for the three "cramer magic" lines]
//
//    Ricky Beam <cramer@foobar.interpath.net>
//
// Revision 1.98  1998/07/13 22:52:29  cramer
// Fixed Alde's divide by zero error in printing the % for block flush/fetch.
// (Leave it to NFS to create a race condition!)
//
// Revision 1.97  1998/07/13 13:18:45  kbracey
// Put back in .ini filename selection code. CYP! STOP MESSING ABOUT!
//
// Revision 1.96  1998/07/13 03:29:49  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// ambiguities. ("declaration/type or an expression")
//
// Revision 1.95  1998/07/12 12:51:36  cyruspatel
// Added static functions for use by ::Run() et al: IsFilenameValid() which
// checks for zero length and "none" filenames, and DoesFileExist() which
// expands the filename to the full path (if not DONT_USE_PATHWORK) before
// doing an access(filename, 0) check.
//
// Revision 1.94  1998/07/12 09:05:59  silby
// Fixed path handling code *again* so that the .ini name will follow the executible's name. This is how it's been done for hundreds of releases, now is a bad time to change.
//
// Revision 1.93  1998/07/11 07:26:26  silby
// Fixed my fix for the exitrc5.now path problem.  Now the path is added on at each read instead of once at the beginning of the code (where it could be wrong depending on the os.)
//
// Revision 1.92  1998/07/11 04:31:43  silby
// Made exitrc5.now files GetFullPathed so they act properly.
//
// Revision 1.91  1998/07/11 01:53:14  silby
// Change in logging statements - all have full timestamps now so they look correct in the win32gui.
//
// Revision 1.90  1998/07/11 01:36:28  silby
// switched order of lurk and connectrequested code so that it would work properly with the win32 gui, also fixed a bug with win32gui forced fetches.
//
// Revision 1.89  1998/07/10 04:34:55  silby
// Changes to connectrequested variable to allow better GUI interface with the main thread for requested flushes,fetches, and updates.
//
// Revision 1.88  1998/07/09 09:43:25  remi
// Give an error message when the user ask for '-ident' and there is no support
// for it in the client.
//
// Revision 1.87  1998/07/09 04:37:56  jlawson
// cleared integer cast warning.
//
// Revision 1.86  1998/07/08 23:31:27  remi
// Cleared a GCC warning.
// Tweaked $Id: client.cpp,v 1.118 1998/08/07 10:59:11 cberry Exp $.
//
// Revision 1.85  1998/07/08 09:28:10  jlawson
// eliminate integer size warnings on win16
//
// Revision 1.84  1998/07/08 09:22:52  remi
// Added support for the MMX bitslicer.
// Wrapped $Log comments to some reasonable value.
//
// Revision 1.83  1998/07/08 05:19:23  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.82  1998/07/07 21:55:14  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.81  1998/07/07 02:52:57  silby
// Another slight change to logging where date will now be shown for proper
// alignment in the win32gui.
//
// Revision 1.80  1998/07/07 02:12:14  silby
// Removed an over-zealous #if DONT_USE_PATHWORK - was preventing .inis from
// following the executible's name.
//
// Revision 1.79  1998/07/07 01:14:09  silby
// Fixed a bug where the default rc5 out buffer was buff-out.des. 413 needs
// to be recalled now.
//
// Revision 1.78  1998/07/05 21:49:15  silby
// Modified logging so that manual wrapping is not done on win32gui, as it
// looks terrible in a non-fixed spaced font.
//
// Revision 1.77  1998/07/05 15:53:58  cyruspatel
// Implemented EraseCheckpointFile() and TruncateBufferFile() in buffwork.cpp;
// substituted unlink() with EraseCheckpointFile() in client.cpp; modified
// client.h to #include buffwork.h; moved InternalGetLocalFilename() to
// cliconfig.cpp; cleaned up some.
//
// Revision 1.76  1998/07/05 13:44:07  cyruspatel
// Fixed an inadvertent wrap of one of the long single-line revision headers.
//
// Revision 1.75  1998/07/05 13:08:58  cyruspatel
// Created new pathwork.cpp which contains functions for determining/setting
// the "work directory" and pathifying a filename that has no dirspec.
// GetFullPathForFilename() is ideally suited for use in (or just prior to) a
// call to fopen(). This obviates the neccessity to pre-parse filenames or
// maintain separate filename buffers. In addition, each platform has its own
// code section to avoid cross-platform assumptions. More doc in pathwork.cpp
// #define DONT_USE_PATHWORK if you don't want to use these functions.
//
// Revision 1.74  1998/07/04 22:05:53  silby
// Fixed problem found by peterd in which message mailing would have
// overridden offlinemode
//
// Revision 1.73  1998/07/04 21:05:30  silby
// Changes to lurk code; win32 and os/2 code now uses the same variables, and
// has been integrated into StartLurk and LurkStatus functions so they now act
// the same.  Additionally, problems with lurkonly clients trying to connect
// when contestdone was wrong should be fixed.
//
// Revision 1.72  1998/07/04 10:30:10  jlawson
// fixed printing of info of invalid block when -runbuffers consumes
// all blocks from buffin
//
// Revision 1.71  1998/07/04 01:57:33  cyruspatel
// Fixed the pause file fix :) - on mt systems inner cpu_i loop was being run
// even if pausefilefound.
//
// Revision 1.70  1998/07/03 23:13:55  remi
// Fix the pause file bug in non-multithreaded clients.
//
// Revision 1.69  1998/07/02 13:09:25  kbracey
// A couple of RISC OS fixes - printf format specifiers made long.
// Changed a "blocks" to "block%s", n==1?"":"s".
//
// Revision 1.68  1998/07/01 10:52:04  ziggyb
// Fixed the problem of a mail message being attemped when run in offline mode
// after the main thread is quit.
//
// Revision 1.67  1998/07/01 03:12:42  blast
// AmigaOS changes...
//
// Revision 1.66  1998/07/01 00:06:27  cyruspatel
// Added full buffering to ::LogScreenPercentSingle() so that the write is
// done in one go, regardless of whether stdout is buffered or not.
//
// Revision 1.65  1998/06/30 05:57:33  jlawson
// benchmark will exit before status message is printed if SignalTriggered
// is already set on function entry.
//
// Revision 1.64  1998/06/30 03:11:46  fordbr
// Fixed blockcount bug with 'x' to exit
//
// Revision 1.63  1998/06/29 08:43:53  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.62  1998/06/29 07:57:30  ziggyb
// Redid the lurk detection for OS/2. Also gave the text output functions
// a priority boost so the display isn't so sluggish.
//
// For the general client, I added an if(!offlinemode) on all the mailmessages,
// since it seemed that even in offline mode, or lurk mode, a smtp attempt was
// being made.
//
// Revision 1.61  1998/06/29 06:57:38  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.60  1998/06/29 04:22:17  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.59  1998/06/25 02:30:33  jlawson
// put back public Client::connectrequested for use by win32gui
//
// Revision 1.58  1998/06/25 01:34:31  blast
//
// AmigaOS changes, also changed makefile so that there now is only ONE
// AmigaOS makefile for both cpu's.
//
// Revision 1.57  1998/06/24 21:53:40  cyruspatel
// Created CliGetMessageForProblemCompletedNoSave() in clisrate.cpp. It
// is similar to its non-nosave pendant but doesn't affect cumulative
// statistics.  Modified Client::Benchmark() to use the function.
//
// Revision 1.56  1998/06/23 20:19:52  cyruspatel
// Adjusted NetWare specific stuff in ::Benchmark to obtain the timeslice
// from the new GetTimesliceBaseline()
//
// Revision 1.55  1998/06/22 03:25:30  silby
// changed so that pausefile is only checked every exitfilechecktime now.
//
// Revision 1.54  1998/06/21 16:08:02  cyruspatel
// NetWare change: first thread no longer migrates to smp.
//
// Revision 1.53  1998/06/17 10:51:23  kbracey
// Fixed printfs for machines with 64-bit longs.
//
// Revision 1.52  1998/06/17 07:53:02  cberry
// replaced the % in [%s] Both RC5 and DES are marked as finished.  Quitting.
//
// Revision 1.51  1998/06/15 12:03:49  kbracey
// Lots of consts.
//
// Revision 1.50  1998/06/15 08:28:02  jlawson
// fixed signed/unsigned comparison
//
// Revision 1.49  1998/06/15 06:18:32  dicamillo
// Updates for BeOS
//
// Revision 1.48  1998/06/14 12:24:15  ziggyb
// Fixed the OS/2 lurk mode so that it updates less freqently. It only
// does a -update after a completed block. I've also OS2 defined -fetch/-flush
// so that if a connection is lost while in lurk mode, it will stop trying to
// connect and return with a value of -3.
//
// Win32 clients may want to look at that to reduce the client trying to -update
// after a connection is lost in lurk mode.
//
// Revision 1.47  1998/06/14 08:26:38  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.46  1998/06/14 08:12:33  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

#if (!defined(lint) && defined(__showids__))
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.118 1998/08/07 10:59:11 cberry Exp $"; }
#endif

// --------------------------------------------------------------------------

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "version.h"
#include "problem.h"
#include "network.h"
#include "mail.h"
#include "scram.h"
#include "convdes.h"
#include "sleepdef.h"  //sleep(), usleep()
#include "threadcd.h"
#include "buffwork.h"
#include "clitime.h"
#include "clirate.h"
#include "clisrate.h"
#include "clicdata.h"
#include "pathwork.h"
#include "cpucheck.h"  //GetTimesliceBaseline()
#include "cliident.h"  // CliIdentifyModules()
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()

#if ( ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32)) && defined(MULTITHREAD) )
#include "lurk.h"      //lurk stuff
#endif

#define Time() (CliGetTimeString(NULL,1))

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_AMIGAOS)
#if (CLIENT_CPU == CPU_68K)
long __near __stack  = 65536L;  // AmigaOS has no automatic stack extension
      // seems standard stack isn't enough
#endif // (CLIENT_CPU == CPU_68K)
#endif // (CLIENT_OS == OS_AMIGAOS)

#if (CLIENT_OS == OS_RISCOS)
s32 guiriscos, guirestart;
#endif

// --------------------------------------------------------------------------

Problem problem[2*MAXCPUS];
volatile u32 SignalTriggered, UserBreakTriggered;
volatile s32 pausefilefound = 0;

// --------------------------------------------------------------------------

#define TEST_CASE_COUNT 32

// note this is in .lo, .hi, .lo... order...

// RC5-32/12/8 test cases -- generated by gentests64.cpp:
static const u32 rc5_test_cases[TEST_CASE_COUNT][8] = {
  {0x9CC718F9L,0x82E51B9FL,0xF839A5D9L,0xC41F78C1L,0x20656854L,0x6E6B6E75L,0xB74BE041L,0x496DEF29L},
  {0xAD53E9EEL,0x503307FFL,0x801617EFL,0xD692CED8L,0x9E8077E3L,0x0DFB3219L,0x5C3DDEFEL,0x8352F7E8L},
  {0x0DBF2929L,0xE03E6FC8L,0xE7DBE5BFL,0x7B751357L,0xFC2D9B0EL,0xC5EAAA09L,0x376AEAD6L,0x4A7C9366L},
  {0xA17777A1L,0x02C1E29BL,0xF1AA7427L,0xE68AD1B4L,0x1E17F769L,0x39A94DB1L,0x78CFF380L,0x53803F15L},
  {0xD2725B6FL,0xDE74A850L,0x412D03CEL,0x0B4E5828L,0x97EEF36CL,0xD64AC9C5L,0x63B95CC3L,0x317A694BL},
  {0xC96A1F6AL,0x5DD2CB81L,0x3ED3951FL,0xA0FEBBACL,0xC8E05F32L,0xCF4F69ACL,0xFD354429L,0x3B484423L},
  {0x70D8CB2EL,0x28121486L,0x0BC8EB42L,0x1A98CAF8L,0xD6B4200EL,0x5A52D62DL,0x0C108DFAL,0x8586C175L},
  {0x6FF52A11L,0xA82E0B78L,0x8FFA8420L,0xADD71687L,0xA4AAA33AL,0xBB16D7CBL,0x15D7D840L,0xE49390DBL},
  {0x2FBAC32EL,0x07E0FA31L,0x6D15A362L,0x503AF090L,0xC2C33AF0L,0xBB46A730L,0x5CD685C4L,0xEE8923ADL},
  {0xD9E0DF5EL,0x2C9FEA4AL,0x0C864872L,0xB6FD690EL,0x7C11D122L,0xD634A7ADL,0xE61CB70FL,0xF648AB05L},
  {0x56DF8939L,0xC2A5A41AL,0x8F7B3577L,0x541C52B8L,0xDF866B91L,0x8865735EL,0x79744D69L,0x136B18BBL},
  {0x4FF28818L,0x31EBB0BDL,0xDCE0E95CL,0x61563C09L,0x0B72DC92L,0x41E67F05L,0x9A6CE9DCL,0x19501B69L},
  {0x2E106515L,0xA32A5709L,0x9962A7C9L,0xD0277838L,0x980DE39EL,0xA92C58E6L,0x8C51EC30L,0x9E132667L},
  {0x1BF46B08L,0xFFDBA499L,0x296F7027L,0x57CC173FL,0xA3E5327DL,0x35DDB044L,0x573076F0L,0xF5926ACEL},
  {0xFE15A18BL,0xEF365EC6L,0xB233039FL,0x6B43EBD7L,0x7D833BA5L,0xBE32181AL,0xBED66A62L,0x3569D778L},
  {0x83AED0F6L,0xDD360FA8L,0x1A9BE31BL,0x40478379L,0x66DA385EL,0x4FC79A9CL,0x46CF6792L,0x32F61EFDL},
  {0x10B68262L,0xF193FF18L,0x04555142L,0xCC56315EL,0xECB8D9D9L,0xF6D6F5E0L,0x356ACF47L,0x8256B2B6L},
  {0xCFE700A9L,0x14C538B0L,0xD6CD4C7FL,0xC4AE077FL,0xF235A559L,0xA86AA7A7L,0x8FB2C30BL,0x7865C1BDL},
  {0xAABA5363L,0xEF0783C8L,0xB63197FAL,0x9D4BD494L,0x5522F33AL,0xDA7E541AL,0x1A752327L,0x2BC13FEBL},
  {0x496843EAL,0xEC506879L,0x876DB29CL,0x8BE92A18L,0x8DC2F7B0L,0xF6795A2BL,0x5A4092A3L,0x6FC7DBD8L},
  {0x15EA5B56L,0x325A329DL,0xEF2EDE0FL,0x84075B43L,0x26866105L,0xC0F0FB0FL,0x95606F49L,0xD99306DDL},
  {0x37F9E181L,0xAC9FE7CDL,0x53E11DBAL,0x3DE1760DL,0x9B38424DL,0x446F1874L,0xCFE2DCA1L,0xBF02F214L},
  {0x980EE103L,0x02565361L,0xD8B42FC8L,0x2B744E31L,0xC86B006AL,0x95271537L,0xCE92B9F6L,0x34B38F55L},
  {0xE1622235L,0x9D79FD72L,0x64949621L,0x827E7326L,0xB62DCFFBL,0x5550394BL,0x16FFA94FL,0x0F018F39L},
  {0x7CED2D31L,0xA6C12FDAL,0x9A2C916EL,0x387A3526L,0xB72354F0L,0x3FC4695EL,0xED740B75L,0xE509631AL},
  {0x916A4DCFL,0x06A7F131L,0xE0EB2318L,0x02A6A72AL,0xEC04D2B2L,0x874224B3L,0x57FF02F2L,0x09A93B11L},
  {0x095089A9L,0x67640DD0L,0x5BFE0C48L,0x540099ECL,0x7B698791L,0x9D546A5DL,0x1A6D6D0FL,0x927E09F6L},
  {0x8DDAAA17L,0x31F00CD1L,0xF051CEE8L,0x65449CE3L,0x431B87FEL,0x57515625L,0xBA4BEDD5L,0x54E57D62L},
  {0x86FF3B32L,0x8D06370DL,0x4491A8A0L,0x28EE0149L,0x88B5A51EL,0x622E4B51L,0x7DE6E50CL,0xE4FA09AEL},
  {0x1E7983D4L,0x641E961BL,0xBC2B9ED9L,0x523DD916L,0x58EBA9CDL,0xDAC66FDDL,0x674A743EL,0x979ADDF5L},
  {0x3CC18C96L,0x5F70F357L,0x7D4D6EBCL,0x5A2DF505L,0x6B8101BBL,0x7166229BL,0x3D467CB4L,0x8363EB0DL},
  {0x8B101ED0L,0xE8F7D6D7L,0x6DE39B32L,0x737BE68EL,0xF0D36A77L,0xA47FB3F6L,0x85659E76L,0x7BB2E391L}
};

// DES test cases -- key/iv/plain/cypher -- generated by gentestsdes.c
// DES test cases are now true DES keys, and can be checked with any DES package
static const u32 des_test_cases[TEST_CASE_COUNT][8] = {
  {0x54159B85L,0x316B02CEL,0xB401FC0EL,0x737B34C9L,0x96CB8518L,0x4A6649F9L,0x34A74BC8L,0x47EE6EF2L},
  {0x1AD9EA4AL,0x15527592L,0x805A8EF3L,0x21F68C4CL,0xC4B9980EL,0xD5C3BD8AL,0x5AFD5D57L,0x335981ADL},
  {0x5B199D6EL,0x40C40E6DL,0xF5756816L,0x36088043L,0x4EF1DF24L,0x6BF462ECL,0xC657515FL,0xABB6EBB0L},
  {0xAD8C0DECL,0x68385DF1L,0x19FB0D4FL,0x288D2CD6L,0x03FA0F6FL,0x038E92F8L,0x2FA04E4CL,0xBFAB830AL},
  {0xC475C22CL,0xDFE3B67AL,0x5614A37EL,0xD70F8E2DL,0xCA620ACEL,0xA1CF54BBL,0xB5BF73A1L,0xB2BB55BDL},
  {0x2FABC40DL,0xE03B8CE6L,0xF825C0CFL,0x47BDC4A9L,0x639F0904L,0x354EFC8BL,0xC745E11CL,0x698BF15FL},
  {0x80940E61L,0xDCBC7F73L,0xA30685EAL,0x67CDA3FEL,0x6E538AA3L,0xC34993BBL,0xF6DBDCE9L,0x6FCE1832L},
  {0x4A701329L,0x450D5D0BL,0x93D406FAL,0x96C9CD56L,0xAF7D2E73L,0xA1A9F844L,0x9428CB49L,0x1F93460FL},
  {0x2A73B06EL,0x8C855D6BL,0x3FC6F9D5L,0x3F07BC65L,0x9A311C3BL,0x8FC62B22L,0x0E71ECD9L,0x003B4F0BL},
  {0x255DFBB0L,0xB5290115L,0xE4663D24L,0x702B8D86L,0xC082814FL,0x6DFA89ACL,0xB76E630DL,0xF54F4D24L},
  {0xBA1A3B6EL,0x9158E3C4L,0x4C3E8CBCL,0xA19D4133L,0x7F8072ECL,0x6A19424EL,0xE09F06DAL,0x6508CD88L},
  {0xFB32138AL,0xF4F73175L,0x87C55A28L,0xC5FAA7A2L,0xDAE86B44L,0x629B5CAEL,0xAEC607BCL,0x9DD8816DL},
  {0x5B0BDA4FL,0x025B2502L,0x1F6A43E5L,0x0006E17EL,0xB0E11412L,0x64EB87EBL,0x2AED8682L,0x6A8BC154L},
  {0xB05837B3L,0xFBE083AEL,0x3248AE33L,0xD92DDA07L,0xFAF4525FL,0x4E90B16DL,0x6725DBD5L,0x746A4432L},
  {0x76BC4570L,0xBFB5941FL,0x8F2A8068L,0xCE638F26L,0xA21EBDF0L,0x585A6F8AL,0x65A3766EL,0x95B6663AL},
  {0xC7610E85L,0x5DDCBC51L,0xB0747E7FL,0x8A52D246L,0x3825CE90L,0xD70EA566L,0x50BC63A5L,0xDF9DD8FAL},
  {0xB9B02615L,0x017C3745L,0x21BAECACL,0x4771B2AAL,0x32703B09L,0x0CBEF2BCL,0x69907E24L,0x0B3928A6L},
  {0x0D7C8F0DL,0xFDC2DF6EL,0x3BBCE282L,0x7C62A9D8L,0x4E18FA5AL,0x2D942C4EL,0x5BF53685L,0x23E40E20L},
  {0xBAA426B6L,0xAED92F13L,0xC0DAC03CL,0x3382923AL,0x25F6F952L,0x3C691477L,0x49B7862AL,0x6520E509L},
  {0x7C37682AL,0x164A43B3L,0x9D31C0D1L,0x884B1EE5L,0x2DCBB169L,0xB4530B74L,0x3C93D6C3L,0x9A9CE765L},
  {0x79B55B8FL,0x6B8AC2B5L,0xE9545371L,0x004E203EL,0xA3170E57L,0x9F71563DL,0xF5DE356FL,0xBD0191DFL},
  {0xC8F80132L,0xD532972FL,0xBC2145BCL,0x42E174FEL,0xBA4DCA59L,0x6F65FA58L,0xB276ADD5L,0xA0A9F7B1L},
  {0x6E497043L,0x7C402CC2L,0x0039BB42L,0xBD8438A2L,0x508592BFL,0x1A2F40D6L,0x0F1EB5BCL,0x6B0C42E7L},
  {0xB3C4FD31L,0xD619314AL,0x39B2DBF7L,0x0295F93AL,0x4D547967L,0x36149936L,0x44B02FEEL,0xEECC0B2DL},
  {0x7FA12954L,0x08737CA8L,0x8ECDCE90L,0x5DACCF36L,0x7AA693B0L,0x62C8CA9CL,0x948CB25EL,0xF4781028L},
  {0x01BFDC08L,0x7558CD0EL,0x7D6D82DAL,0x19ACD958L,0x1EDF3781L,0x195110A7L,0x021EB315L,0xE2EA34C9L},
  {0x5161A2C4L,0x4F043B43L,0x17D76130L,0xDCB7695CL,0xA70ADBC0L,0x843A8801L,0xAEE16715L,0xE1AF0F07L},
  {0x943DF4E3L,0xB6D6CEF2L,0xC763AAA3L,0xA0179248L,0xEB61626FL,0x1B130032L,0x5630226FL,0x1C9DBFB2L},
  {0xE997049EL,0x37D5E085L,0x07C372A8L,0x3669C801L,0x689B4583L,0xDA05F0A2L,0xFA70DACDL,0x3F031F6CL},
  {0x4C2F1083L,0x5D8A6B32L,0xC38544FAL,0x017883F5L,0xD06D9EAAL,0xEE0DFBF6L,0xB1A728B7L,0x12C311C4L},
  {0x5225BCB0L,0xE51C98B6L,0x2B7ABF2DL,0xD714717EL,0xC867B0B7L,0xF24322B6L,0x0A6BF211L,0xB0B7C1CAL},
  {0xCE6823E9L,0x16A8A476L,0xCDC4DBA4L,0xD93B6603L,0xC6E231B9L,0xD84C2204L,0xDB623F7CL,0x3477E4B2L},
};

// --------------------------------------------------------------------------

Client::Client()
{
  id[0] = 0;
  inthreshold[0] = 10;
  outthreshold[0] = 10;
  inthreshold[1] = 10;
  outthreshold[1] = 10;
  blockcount = 0;
  minutes = 0;
  strcpy(hours,"0.0");
  keyproxy[0] = 0;
  keyport = 2064;
  httpproxy[0] = 0;
  httpport = 80;
  uuehttpmode = 1;
  strcpy(httpid,"");
  totalBlocksDone[0] = totalBlocksDone[1] = 0;
  timeStarted = 0;
  cputype=-1;
  offlinemode = 0;
  autofindkeyserver = 1;  //implies 'only if keyproxy==dnetkeyserver'

#ifdef DONT_USE_PATHWORK
  strcpy(ini_logname, "none");
  strcpy(ini_in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(ini_out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(ini_in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(ini_out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(ini_exit_flag_file, "exitrc5" EXTN_SEP "now");
  strcpy(ini_checkpoint_file[0],"none");
  strcpy(ini_checkpoint_file[1],"none");
  strcpy(ini_pausefile,"none");

  strcpy(logname, "none");
  strcpy(inifilename, InternalGetLocalFilename("rc5des" EXTN_SEP "ini"));
  strcpy(in_buffer_file[0], InternalGetLocalFilename("buff-in" EXTN_SEP "rc5"));
  strcpy(out_buffer_file[0], InternalGetLocalFilename("buff-out" EXTN_SEP "rc5"));
  strcpy(in_buffer_file[1], InternalGetLocalFilename("buff-in" EXTN_SEP "des"));
  strcpy(out_buffer_file[0], InternalGetLocalFilename("buff-out" EXTN_SEP "des"));
  strcpy(exit_flag_file, InternalGetLocalFilename("exitrc5" EXTN_SEP "now"));
  strcpy(checkpoint_file[1],"none");
  strcpy(pausefile,"none");
#else
  strcpy(logname, "none");
  strcpy(inifilename, "rc5des" EXTN_SEP "ini");
  strcpy(in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(exit_flag_file, "exitrc5" EXTN_SEP "now");
  strcpy(checkpoint_file[1],"none");
  strcpy(pausefile,"none");
#endif
  messagelen = 0;
  smtpport = 25;
  strcpy(smtpsrvr,"your.smtp.server");
  strcpy(smtpfrom,"RC5Notify");
  strcpy(smtpdest,"you@your.site");
  numcpu = -1;
  numcputemp=1;
  strcpy(checkpoint_file[0],"none");
  checkpoint_min=5;
  percentprintingoff=0;
  connectoften=0;
  nodiskbuffers=0;
  membuffcount[0][0]=0;
  membuffcount[1][0]=0;
  membuffcount[0][1]=0;
  membuffcount[1][1]=0;
  for (int i1=0;i1<2;i1++) {
    for (int i2=0;i2<500;i2++) {
      for (int i3=0;i3<2;i3++) {
        membuff[i1][i2][i3]=NULL;
      }
    }
  }
  nofallback=0;
  randomprefix=100;
  preferred_contest_id = 1;
  preferred_blocksize=30;
  randomchanged=0;
  consecutivesolutions[0]=0;
  consecutivesolutions[1]=0;
  quietmode=0;
  nonewblocks=0;
  nettimeout=60;
  noexitfilecheck=0;
  exitfilechecktime=30;
#if defined(LURK)
  dialup.lurkmode=0;
  dialup.dialwhenneeded=0;
#endif
#if (CLIENT_OS==OS_WIN32) 
  win95hidden=0;
#endif
#if (CLIENT_OS == OS_OS2)
   os2hidden=0;
#endif
  contestdone[0]=contestdone[1]=0;
  srand( (unsigned) time( NULL ) );
  InitRandom();
#ifdef MMX_BITSLICER
  usemmx = 1;
#endif
}


// --------------------------------------------------------------------------

Client::~Client()
{
  cputype=-1; //dummy to suppress compiler 'Warning:'
}

// --------------------------------------------------------------------------

void Client::RandomWork( FileEntry * data )
{
  u32 randompref2;

  randompref2 = ( ( (u32) randomprefix) + 1 ) & 0xFF;

  data->key.lo = htonl( Random( NULL, 0 ) & 0xF0000000L );
  data->key.hi = htonl( (Random( NULL, 0 ) & 0x00FFFFFFL) + ( randompref2 << 24) ); // 64 bits significant

  data->iv.lo = htonl( 0xD5D5CE79L );
  data->iv.hi = htonl( 0xFCEA7550L );
  data->cypher.lo = htonl( 0x550155BFL );
  data->cypher.hi = htonl( 0x4BF226DCL );
  data->plain.lo = htonl( 0x20656854L );
  data->plain.hi = htonl( 0x6E6B6E75L );
  data->keysdone.lo = htonl( 0 );
  data->keysdone.hi = htonl( 0 );
  data->iterations.lo = htonl( 0x10000000L );
  data->iterations.hi = htonl( 0 );
  data->id[0] = 0;
//82E51B9F:9CC718F9 -- sample problem from RSA pseudo-contest...
//  data->key.lo = htonl(0x9CC718F9L & 0xFF000000L );
//  data->key.hi = htonl(0x82E51B9FL & 0xFFFFFFFFL );
//  data->iv.lo = htonl( 0xF839A5D9L );
//  data->iv.hi = htonl( 0xC41F78C1L );
//  data->cypher.lo = htonl( 0xB74BE041L );
//  data->cypher.hi = htonl( 0x496DEF29L );
//  data->plain.lo = htonl( 0x20656854L );
//  data->plain.hi = htonl( 0x6E6B6E75L );
//  data->iterations.lo = htonl( 0x01000000L );
//END SAMPLE PROBLEM
  data->op = htonl( OP_DATA );
  data->os = 0;
  data->cpu = 0;
  data->buildhi = 0;
  data->buildlo = 0;

  data->contest = 0; // Random blocks are always RC5, not DES.

  data->checksum =
    htonl( Checksum( (u32 *) data, ( sizeof(FileEntry) / 4 ) - 2 ) );
  data->scramble = htonl( Random( NULL, 0 ) );
  Scramble( ntohl(data->scramble), (u32 *) data, ( sizeof(FileEntry) / 4 ) - 1 );

}

// --------------------------------------------------------------------------

s32 Client::ForceFetch( u8 contest, Network *netin )
{
  s32 temp1, temp2;
  s32 ret = -1;

  temp1 = CountBufferInput(contest);
  temp2 = temp1-1;
  while ((!UserBreakTriggered) && (temp2 < temp1) && (temp1 < inthreshold[contest]))
  {
    // If we're actually making the buffer file bigger, then keep trying to fetch...
    ret = Fetch(contest, netin);
    temp2 = temp1;
    temp1 = CountBufferInput(contest);

    LogScreen("[%s] %d block%s remain%s in %s\n",Time(),temp1,temp1==1?"":"s",temp1==1?"s":"",
#ifdef DONT_USE_PATHWORK
    ini_in_buffer_file[contest]);
#else
    in_buffer_file[contest]);
#endif
  }
  return ret;
}

// --------------------------------------------------------------------------

#if defined(NONETWORK)
s32 Client::Fetch( u8 /*contest*/, Network * /*netin*/, s32 /*quietness*/, s32 /*force*/ )
{ return -2; }

#else

s32 Client::Fetch( u8 contest, Network *netin, s32 quietness, s32 force )
{
  Network *net;
  Packet packet;
  u32 scram;
  u32 scram2;
  s32 err;
  FileEntry data;
  s32 count = 0;
  s32 more;
  s32 retry;
  u32 thisrandomprefix;
  u8 tmpcontest;

if (force == 0) // check to see if fetch should be done
  {
    if (offlinemode
      #if (CLIENT_OS == OS_NETWARE)
          || !nwCliIsNetworkAvailable(0)
      #endif
    ) return( -1 );
  };

  if (contestdone[contest]) return( -1 );

  // first, find out if we are already full
  if (CountBufferInput(contest) >= inthreshold[contest]) 
    {
    if (force == 1) LogScreen("%s in buffer is already full, no connect needed.\n",(contest == 1 ? "DES":"RC5"));
    return 0;
    };

  #if defined(LURK)
  // Check if we need to dial, and if update started this connect or not
  if (!netin) // check if update started the connect, or if we must
    if (dialup.DialIfNeeded(force) < 0) return -1; // connection failed
  #endif


  // ready the net
  if (!netin)
  {
    net = new Network( (const char *) (keyproxy[0] ? keyproxy : NULL ),
        (const char *) (nofallback ? (keyproxy[0] ? keyproxy : NULL) : DEFAULT_RRDNS),
        (s16) keyport, autofindkeyserver );
    net->quietmode = quietmode;

    switch (uuehttpmode)
    {
    case 1:  // uue
      net->SetModeUUE(true);
      break;

    case 2:  // http
      net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
      break;

    case 3:  // uue + http
      net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
      net->SetModeUUE(true);
      break;

    case 4:  // SOCKS4
      net->SetModeSOCKS4(httpproxy, (s16) httpport, httpid);
      break;

    case 5:  // SOCKS5
      net->SetModeSOCKS5(httpproxy, (s16) httpport, httpid);
      break;
    }

  } else {
    net = netin;
  }

  retry = 0;
  while ( net->Open() && ( retry++ < FETCH_RETRY ) )
  {
    if ( retry == FETCH_RETRY )
    {
      if (!netin) delete net;
      return( -1 );
    }
    LogScreen( "[%s] Network::Open Error %d - sleeping for 3 seconds\n", Time(), retry );
    sleep( 3 );

#if (CLIENT_OS == OS_AMIGAOS)
    if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C )
      SignalTriggered = UserBreakTriggered = 1;
#endif

#if defined(LURK)
    if (dialup.CheckForStatusChange() == -1)
      {
      LogScreen("TCPIP Connection Lost - Aborting fetch\n");
      return -3;                 // so stop trying and return
      };
#endif

    if ( SignalTriggered )
      return -2;
    // at this point we need more data, so try until we get a connection
  }

  // request a valid key for communications
  memset( &packet, 0, sizeof(Packet) );
  packet.op = htonl( OP_SCRAMREQUEST );
  packet.version = htonl( PACKET_VERSION );
    packet.os = htonl( CLIENT_OS );
    packet.cpu = htonl( CLIENT_CPU );
    packet.build = htonl( CLIENT_CONTEST*100 + CLIENT_BUILD );

    if (contest == 0) {
      packet.contestid = htonl( IDCONTEST_RC564 );
    } else {
      packet.contestid = htonl( IDCONTEST_DESII );
    }

  packet.checksum =
    htonl( Checksum( (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) );
  scram = Random( NULL, 0 );
  Scramble( scram, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
  packet.scramble = htonl( scram );

  if ( net->Put( sizeof( Packet ), (char *) &packet ) )
  {
    LogScreen( "[%s] Network::Error Unable to send data 1\n", Time() );
    if (!netin) delete net;
    return( -1 );
  }

  if ( ( err = net->Get( sizeof( Packet ), (char *) &packet, nettimeout ) ) == sizeof(Packet) )
  {
    Descramble( scram, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
    if ( ( ntohl( packet.scramble ) != scram ) || // key is valid
         ( !CheckChecksum( ntohl( packet.checksum ),
           (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) ) || // checksum valid
         ( ntohl( packet.version ) != PACKET_VERSION ) || // version valid
         ( ntohl( packet.op ) != OP_SCRAM ) )  // expected packet
    {
      LogScreen( "[%s] Network::Error Bad Data 2\n", Time() );
      if (!netin) delete net;
      return( -1 );
    }
  }
  else
  {
    LogScreen( "[%s] Network::Error Read failed 3/%d\n", Time(), (int) err );
    if (!netin) delete net;
    return( -1 );
  }

  // this is the issued scramble key...
  // finish communications with it...
  scram2 = ntohl( packet.key.lo );

  strncpy( proxymessage, packet.id, 63 );
#if (CLIENT_OS == OS_OS390)
  __atoe(proxymessage);
#endif

  SetContestDoneState(&packet);
  if (contestdone[contest])
  {
    if (!netin) delete net;
    return( -1 );
  }

  do
  {
    memset( &packet, 0, sizeof(Packet) );
    packet.op = htonl( OP_REQUEST_MULTI );
#if (CLIENT_CPU == CPU_X86)
    // x86 cores require <=31
    packet.iterations = htonl( 1 << min(31, (int) preferred_blocksize) );
#else
    //packet.iterations = 0x10000000L;     // Request 1 block
    packet.iterations = htonl( 1 << preferred_blocksize );
#endif
    packet.version = htonl( PACKET_VERSION );
    packet.os = htonl( CLIENT_OS );
    packet.cpu = htonl( CLIENT_CPU );
    packet.build = htonl( CLIENT_CONTEST*100 + CLIENT_BUILD );

    if (contest == 0) {
      packet.contestid = htonl( IDCONTEST_RC564 );
    } else {
      packet.contestid = htonl( IDCONTEST_DESII );
    }

    packet.checksum =
      htonl( Checksum( (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) );
    Scramble( scram2, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
    packet.scramble = htonl( scram2 );

    if ( net->Put( sizeof( Packet ), (char *) &packet ) )
    {
      LogScreen( "\n[%s] Network::Error Unable to send data 4\n", Time() );
      if (!netin) delete net;
      return( count ? count : -1 );
    }
    if ( ( err = net->Get( sizeof( Packet ), (char *) &packet, nettimeout ) ) == sizeof(Packet) )
    {
      Descramble( ntohl( packet.scramble ),
                  (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
      if ( ( ntohl( packet.scramble ) != scram2 ) || // key is valid
           ( !CheckChecksum( ntohl( packet.checksum ),
             (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) ) || // checksum valid
           ( ntohl( packet.version ) != PACKET_VERSION ) || // version valid
           ( ntohl( packet.op ) != OP_DATA ) )  // expected packet
      {

        LogScreen( "\n[%s] Network::Error Bad Data 5\n", Time() );
        if (!netin) delete net;
        return( count ? count : -1 );
      }
    }
    else
    {
      LogScreen( "\n[%s] Network::Error Read failed 6/%d\n", Time(), (int) err );
      if (!netin) delete net;
      return( count ? count : -1 );
    }

    // incoming is in network byte order, and so are the data strctures,
    // so we can just copy...
    data.key.hi = packet.key.hi;
    data.key.lo = packet.key.lo;

    if (ntohl(packet.contestid) == IDCONTEST_RC564) {
      data.contest = 0;
    } else {
      if (ntohl(packet.contestid) == IDCONTEST_DESII) {
        data.contest = 1;
      } else {
        if (ntohl(packet.contestid) == 0) {
          // Connecting to a non-dual server will return contestid=0. This means RC564.
          data.contest = 0;
        } else {
          Log("\n[%s] Received block for unknown contest #%ld\n",Time(),(long)ntohl(packet.contestid));
          if (!netin) delete net;
          return( count ? count : -1 );
        }
      }
    }

    tmpcontest = data.contest;

    if (data.contest == 0) {
      // Remember the prefix on the rc564 blocks.
      thisrandomprefix = (ntohl(packet.key.hi) & 0xFF000000L) >> 24;
      if (thisrandomprefix != ((u32) randomprefix) ) { // Has the high byte changed?  If so, then remember it.
        if (ntohl(packet.iterations) > 0x08000000L) { // only if this wasn't a test block...
          randomprefix=thisrandomprefix;                 // and use it to generate random blocks.
          randomchanged=1;
        }
      }
    }
    data.iv.hi = packet.iv.hi;
    data.iv.lo = packet.iv.lo;
    data.plain.hi = packet.plain.hi;
    data.plain.lo = packet.plain.lo;
    data.cypher.hi = packet.cypher.hi;
    data.cypher.lo = packet.cypher.lo;
    data.keysdone.hi = 0;
    data.keysdone.lo = 0;
    if (packet.iterations == 0)
    {
      packet.iterations = (u32) 1<<31;
      LogScreen("\n[%s] Received 2^32 block.  Truncating to 2^31\n", Time());
    }
    data.iterations.hi = (packet.iterations == 0 ? 1 : 0 );
    data.iterations.lo = packet.iterations;
    data.op = packet.op;
    strcpy( data.id, "rc5@distributed.net" );
#if (CLIENT_OS == OS_OS390)
    __etoa(data.id);
#endif
    data.cpu=0;
    data.os=0;
    data.buildlo=0;
    data.buildhi=0;

//Log( "\niv:        %08lX:%08lX \n"
//     "plain:     %08lX:%08lX \n"
//     "cypher:    %08lX:%08lX \n"
//     "key:       %08lX:%08lX \n",
//     ntohl( data.iv.hi ) , ntohl( data.iv.lo ),
//      ntohl( data.plain.hi ) , ntohl( data.plain.lo ),
//      ntohl( data.cypher.hi ) , ntohl( data.cypher.lo ),
//      ntohl( data.key.hi ) , ntohl( data.key.lo ));


    data.checksum =
      htonl( Checksum( (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 2 ) );
    scram = Random( NULL, 0 ); // new random key
    data.scramble = htonl( scram );
    Scramble( scram, (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 1 );
    // now have a safe file entry.

    // stuff it in the queue immediately.
    if ( ( more = PutBufferInput( &data ) ) == -1 )
    {
      Log( "\n[%s] Fatal Buffering Error\n", Time() );
      if (!netin) delete net;
      return( -1 );
    }
    count++;
    if (proxymessage[0] != 0) {
      if (quietness <= 1) // if > 1, don't show proxy message
        {
        #if (CLIENT_OS == OS_WIN32)
          //uses proportional font, so reformatting looks wierd
          Log( "[%s] The proxy says: \"%.64s\"\n", Time(), proxymessage );
        #else
          Log( CliReformatMessage( "The proxy says: ", proxymessage ) );
        #endif
        }
      proxymessage[0]=0;
    }
#if !defined(NOMAIN)
    if (!isatty(fileno(stdout))) 
      Log( "<" );  // use simple output for redirected stdout
    else
#endif
      {
      u32 percent2div = (inthreshold[contest]-more)+count;
      if (((s32)(percent2div)) <= 0) percent2div = 0; 
      u32 percent2 = ((percent2div)?((count*10000)/percent2div):(10000));
       LogScreen( "\r[%s] Retrieved block %u of %u (%u.%02u%% transferred) ",
          CliGetTimeString(NULL,1), count, percent2div?percent2div:count,
          percent2/100, percent2%100 );
      }    
    if (tmpcontest != contest)
    {
      // We didn't get the block we were requesting.
      Log("\n[%s] Received block for wrong contest.\n", Time());
      if (!netin) delete net;
      return( -1 );
    }
#if (CLIENT_OS == OS_AMIGAOS)
    if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C )
      SignalTriggered = UserBreakTriggered = 1;
#endif
  } while ( more < inthreshold[contest]  && (!SignalTriggered) );

  // close this connection
  if (!netin) delete net;

  #if defined(LURK)
  // Check if we need to hangup (unless update started this connect)
  if (!netin)
    dialup.HangupIfNeeded();
  #endif

  LogScreen( "\n" );
  Log( "[%s] Retrieved %d %s block%s from server              \n", Time(), (int) count, (contest == 1 ? "DES":"RC5"), count == 1 ? "" : "s" );
  return ( count );
}

#endif //NONETWORK

// ---------------------------------------------------------------------------

s32 Client::ForceFlush( u8 contest , Network *netin )
{
  u32 temp1, temp2;
  s32 ret = -1;

  temp1 = CountBufferOutput(contest);
  temp2 = 1 + temp1;
  while ((!UserBreakTriggered) && (temp1 < temp2) ) {
    // If we're actually making the buffer file smaller, then keep trying to flush...
    ret = Flush(contest, netin);
    temp2 = temp1;
    temp1 = CountBufferOutput(contest);
    Log("[%s] %d block%s remain%s in %s\n",Time(), temp1, temp1==1?"":"s", temp1==1?"s":"",
#ifdef DONT_USE_PATHWORK
    ini_out_buffer_file[contest]);
#else
    out_buffer_file[contest]);
#endif
  }
  return ret;
}

// ---------------------------------------------------------------------------

#if defined(NONETWORK)
s32 Client::Flush( u8 /*contest*/, Network * /*netin*/, s32 /*quietness*/, s32 /*force*/ )
{ return -2; }

#else

s32 Client::Flush( u8 contest , Network *netin, s32 quietness, s32 force )
{
  // send data, look for ack...
  // we must differentiate between OP_DONE and OP_SUCCESS so we can look for
  // OP_DONE_NOCLOSE_ACK or OP_SUCCESS_ACK

  Network *net;
  Packet packet;
  u32 scram;
  u32 scram2;
  s32 err;
  u32 oper;
  FileEntry data;
  s32 count;
  s32 more;
  u32 i, retry;

if (force == 0) // Check if flush should be done
  {
    if (offlinemode
    #if (CLIENT_OS == OS_NETWARE)
          || !nwCliIsNetworkAvailable(0)
    #endif
     ) return( -1 );
  };

  if (contestdone[contest]) return( -1 );

  // first, find out if we are empty already
  if (CountBufferOutput(contest) < 1)
    {
    if (force == 1) LogScreen("%s out buffer is already empty, no connect needed.\n",(contest == 1 ? "DES":"RC5"));
    return 0;
    };

  #if defined(LURK)
  // Check if we need to dial, and if update started this connect or not
  if (!netin) // check if update started the connect, or if we must
    if (dialup.DialIfNeeded(force) < 0) return -1; // connection failed
  #endif

  // ready the net
  if (!netin)
  {
    net = new Network( (const char *) (keyproxy[0] ? keyproxy : NULL ),
        (const char *) (nofallback ? (keyproxy[0] ? keyproxy : NULL) : DEFAULT_RRDNS),
        (s16) keyport, autofindkeyserver );
    net->quietmode = quietmode;

    switch (uuehttpmode)
    {
    case 1:  // uue
      net->SetModeUUE(true);
      break;

    case 2:  // http
        net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
      break;

    case 3:  // uue + http
      net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
        net->SetModeUUE(true);
      break;

    case 4:  // SOCKS4
      net->SetModeSOCKS4(httpproxy, (s16) httpport, httpid);
      break;
    case 5:  // SOCKS5
      net->SetModeSOCKS5(httpproxy, (s16) httpport, httpid);
      break;
    }

  } else {
    net = netin;
  }

  retry = 0;
  while ( net->Open() )
  {
    if ( retry++ == FETCH_RETRY ) {
      if (!netin) delete net;
      return( -1 );
    }

    LogScreen( "\n[%s] Network::Open Error - Sleeping for 3 seconds\n", Time() );
    sleep( 3 );
#if (CLIENT_OS == OS_AMIGAOS)
    if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C)
      SignalTriggered = UserBreakTriggered = 1;
    #endif

#if defined(LURK)
    if (dialup.CheckForStatusChange() == -1)
       {
       LogScreen("TCPIP Connection Lost - Aborting flush\n");
       return -3;                 // so stop trying and return
       };
#endif

    if ( 1==SignalTriggered ) return -1;
    // Note that this ignores signaltriggered==2 (used when -nodisk is specified)

    // at this point we need more data, so try until we get a connection
  }

  // request a valid key for communications
  memset( &packet, 0, sizeof(Packet) );
  packet.op = htonl( OP_SCRAMREQUEST );
  packet.version = htonl( PACKET_VERSION );

  if (contest == 0) {
    packet.contestid = htonl( IDCONTEST_RC564 );
  } else {
    packet.contestid = htonl( IDCONTEST_DESII );
  }

  packet.checksum =
    htonl( Checksum( (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) );
  scram = Random( NULL, 0 );
  Scramble( scram, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
  packet.scramble = htonl( scram );

  if ( net->Put( sizeof( Packet ), (char *) &packet ) )
  {
    LogScreen( "[%s] Network::Error Unable to send 21\n", Time() );
    if (!netin) delete net;
    return( -1 );
  }

  if ( ( err = net->Get( sizeof( Packet ), (char *) &packet, nettimeout ) ) == sizeof(Packet) )
  {
    Descramble( scram, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
    if ( ( ntohl( packet.scramble ) != scram ) || // key is valid
         ( !CheckChecksum( ntohl( packet.checksum ),
           (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) ) || // checksum valid
         ( ntohl( packet.version ) != PACKET_VERSION ) || // version valid
         ( ntohl( packet.op ) != OP_SCRAM ) )  // expected packet
    {
      LogScreen( "[%s] Network::Error Bad Data 22\n", Time() );
      if (!netin) delete net;
      return( -1 );
    }
  }
  else
  {
    LogScreen( "[%s] Network::Error Read failed 23/%d\n", Time(), (int) err );
    if (!netin) delete net;
    return( -1 );
  }

  SetContestDoneState(&packet);
  if (contestdone[contest]) {
    if (!netin) delete net;
    return( -1 );
  }

  // this is the issued scramble key...
  // finish communications with it...
  scram2 = ntohl( packet.key.lo );

  // get the text message and display to the client
  strncpy( proxymessage, packet.id, 63 );
#if (CLIENT_OS == OS_OS390)
  __atoe(proxymessage);
#endif

  count = 0;
  do
  {
    if ( ( more = GetBufferOutput( &data , contest) ) == -1 )
    {
      // no more to do
      if (proxymessage[0] != 0)
        {
        if (quietness <= 1) // if > 1, don't show proxy message
          {
          #if (CLIENT_OS == OS_WIN32)
            //uses proportional font, so reformatting looks wierd
            Log( "[%s] The proxy says: \"%.64s\"\n", Time(), proxymessage );
          #else
            Log( CliReformatMessage( "The proxy says: ", proxymessage ) );
          #endif
          }
        proxymessage[0]=0;
      }
      //Log( "\n[%s] Sent %d %s block(s) to server\n", Time(), (int) count, (contest == 1 ? "DES":"RC5") );
      //if (!netin) delete net;
      //return( count );
      break;
    }

    // descramble - GetBufferOutbut already told us this is valid data...
    Descramble( ntohl( data.scramble ),
                (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 1 );

    // compose real packet...
    // data is in network byte order, and so are the packet strctures,
    // so we can just copy...

    memset( &packet, 0, sizeof(Packet) );

    // need only op, id, iterations, key...

    //if (strcmp(data.id,"rc5@distributed.net") == 0)
      // This was an earlier version than 6402
      // (Earlier versions always set id to rc5@distributed.net in
      // the buffer, and submitted blocks using the configured id)
      // ......The above didn't work properly, as some clients didn't set this field
      // to rc5@distributed.net as expected.  Changed 12.15.1997 to the following...
//printf("%s\n",data.id);
//printf("%d %d %d %d \n",htonl( (u32) data.os ) , htonl((u32) data.cpu),htonl((u32) data.buildhi),htonl((u32) data.buildlo));

    if ((64 != (u32) data.buildhi ) && (70 != (u32) data.buildhi) && (71 != (u32) data.buildhi))
    {
      // Well, if 64 (or 70 or 71) isn't saved in data.buildhi, then this record was written with an
      // older version of the client.
      strncpy( packet.id, id, sizeof(packet.id) - 1 );
#if (CLIENT_OS == OS_OS390)
      __etoa(packet.id);
#endif
      packet.os = htonl( CLIENT_OS );
      packet.cpu = htonl( CLIENT_CPU );
      packet.build = htonl( 6401 );
    }
    else
    {
      // This block was written to the file by a client ver > 6401
      strncpy( packet.id, data.id, sizeof(data.id) - 1 );
      packet.id[sizeof(data.id)-1]=0; // in case data.id was >58 bytes
#if (CLIENT_OS == OS_OS390)
      __etoa(packet.id);
#endif
      packet.os = htonl( (u32) data.os );
      packet.cpu = htonl( (u32) data.cpu );
      packet.build = htonl( ( (u32) data.buildhi) * 100 + ( (u32) data.buildlo ) );
    }
    for ( i = strlen(packet.id) ; i < sizeof(packet.id) ; i++ )
      packet.id[(int) i] = 0;

    oper = ntohl( data.op );
    // oper in host order

    // no need to change byte order.
    packet.key.hi = data.key.hi;
    packet.key.lo = data.key.lo;
    packet.iv.hi = data.iv.hi;
    packet.iv.lo = data.iv.lo;
    packet.plain.hi = data.plain.hi;
    packet.plain.lo = data.plain.lo;
    packet.cypher.hi = data.cypher.hi;
    packet.cypher.lo = data.cypher.lo;

    if ( data.contest == 0 ) {
        packet.contestid = htonl( IDCONTEST_RC564 );
    } else {
        packet.contestid = htonl( IDCONTEST_DESII );
    }

    if ( (oper == OP_DONE) || (oper == OP_DONE_MULTI) )
    {
      // block we did
      packet.op = htonl( OP_DONE_MULTI );
      packet.key.hi = data.key.hi;
      packet.key.lo = data.key.lo;
      packet.iterations = data.iterations.lo;
    }
    else if ( (oper == OP_SUCCESS) || (oper == OP_SUCCESS_MULTI) )
    {
      // exact key that is solution
      packet.op = htonl( OP_SUCCESS_MULTI );

      // network order problem
      packet.key.hi = data.key.hi;
      packet.key.lo = data.key.lo;
      packet.iterations = 0;
    }
    else
    {
      // not a valid packet??? - discard
      Log( "\n[%s] Error - Bad Data 24\n", Time() );
      if (!netin) delete net;
      return( count ? count : -1 );
    }

//Log( "\nFlushing key %08lX:%08lX   contestid= %d  cpu/os/build: %d/%d/%d\n",
//      ntohl( packet.key.hi ) , ntohl( packet.key.lo ),packet.contestid,
//     ntohl(packet.cpu),ntohl(packet.os),ntohl(packet.build));

    packet.version = htonl( PACKET_VERSION );
    packet.checksum =
      htonl( Checksum( (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) );
    Scramble( scram2, (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
    packet.scramble = htonl( scram2 );
    if ( net->Put( sizeof( Packet ), (char *) &packet ) )
    {
      LogScreen( "\n[%s] Network::Error Unable to send 25\n", Time() );

      // return packet to buffer...
      Scramble( ntohl( data.scramble ),
                (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 1 );
      PutBufferOutput( &data );
      if (!netin) delete net;
      return( count ? count : -1 );
    }

    // ok, it's sent.
    if ( ( err = net->Get( sizeof( Packet ), (char *) &packet, nettimeout ) ) == sizeof(Packet) )
    {
      Descramble( ntohl( packet.scramble ),
                  (u32 *) &packet, ( sizeof(Packet) / 4 ) - 1 );
      if ( ( ntohl( packet.scramble ) != (u32) scram2 ) || // key is valid
           ( !CheckChecksum( ntohl( packet.checksum ),
             (u32 *) &packet, ( sizeof(Packet) / 4 ) - 2 ) ) || // checksum valid
           ( ntohl( packet.version ) != PACKET_VERSION ) || // version valid
           ( ( ntohl( packet.op ) != // expected packet
               (u32) ( ( (oper == OP_DONE) || (oper == OP_DONE_MULTI) ) ?
                 OP_DONE_NOCLOSE_ACK : OP_SUCCESS_ACK ) ) )
            )
      {
        LogScreen( "Network::Error Bad Data 26\n", Time() );
        // return packet to buffer...
        Scramble( ntohl( data.scramble ),
                  (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 1 );
        PutBufferOutput( &data );
        if (!netin) delete net;
        return( count ? count : -1 );
      }
      count++;
      if (proxymessage[0] != 0)
      {
        if (quietness <= 1) // if > 1, don't show proxy message
          {
          #if (CLIENT_OS == OS_WIN32)
            //uses proportional font, so reformatting looks wierd
            Log( "[%s] The proxy says: \"%.64s\"\n", Time(), proxymessage );
          #else
            Log( CliReformatMessage( "The proxy says: ", proxymessage ) );
          #endif
          }
        proxymessage[0]=0;
      }
#if !defined(NOMAIN)
    if (!isatty(fileno(stdout))) 
      Log( "<" );  // use simple output for redirected stdout
    else
#endif
      {
      u32 percent2div = (more+count);  //hmm, can more be negative?
      if (((s32)(percent2div)) <= 0) percent2div = 0;
      u32 percent2 = ((percent2div)?((count*10000)/percent2div):(10000));
       LogScreen( "\r[%s] Sent block %u of %u (%u.%02u%% transferred) ",
          CliGetTimeString(NULL,1), count, percent2div?percent2div:count,
          percent2/100, percent2%100 );
      }
    }
    else //Get()...
    {
      LogScreen( "\n[%s] Network::Error Read Failed 27/%d\n", Time(), (int) err );
      // return packet to buffer...
      Scramble( ntohl( data.scramble ),
                (u32 *) &data, ( sizeof(FileEntry) / 4 ) - 1 );
      PutBufferOutput( &data );
      if (!netin) delete net;
      return( count ? count : -1 );
    }

#if (CLIENT_OS == OS_AMIGAOS)
    if (SetSignal(0L,0L) & SIGBREAKF_CTRL_C)
      SignalTriggered = UserBreakTriggered = 1;
#endif
  }
  while (more != -1 && !(SignalTriggered == 1));

  // close this connection
  if (!netin) delete net;

  #if defined(LURK)
  // Check if we need to hangup (unless update started this connect)
  if (!netin)
    dialup.HangupIfNeeded();
  #endif

  LogScreen( "\n" );
  Log( "[%s] Sent %d %s block%s to server                \n", Time(), (int) count, (contest == 1 ? "DES":"RC5"), count == 1 ? "" : "s" );
  return( count );
}

#endif //NONETWORK

// ---------------------------------------------------------------------------

#ifdef NONETWORK
s32 Client::Update (u8 /*contest*/, s32 /*fetcherr*/, s32 /*flusherr*/, s32 /*force*/ )
{ return -1; }

#else

s32 Client::Update (u8 contest, s32 fetcherr, s32 flusherr, s32 force )
{
  s32 retcode,retcode1,retcode2;
  Network *net;

if (force == 0) // We need to check if we're allowed to connect
  // If force==1, we don't care about the offline mode
  {
    if (offlinemode
      #if (CLIENT_OS == OS_NETWARE)
        || !nwCliIsNetworkAvailable(0)
      #endif
    )
    return( -1 );
  }

    LogFlush(0); //checktosend(x)

  // Ready the net

  #if defined(LURK)
  // Check if we need to dial
  if (dialup.DialIfNeeded(force) < 0) return -1; // connection failed
  #endif

  net = new Network( (const char *) (keyproxy[0] ? keyproxy : NULL ),
      (const char *) (nofallback ? (keyproxy[0] ? keyproxy : NULL) : DEFAULT_RRDNS),
      (s16) keyport, autofindkeyserver );
  net->quietmode = quietmode;

  switch (uuehttpmode)
  {
  case 1:  // uue
    net->SetModeUUE(true);
    break;

  case 2:  // http
    net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
    break;

  case 3:  // uue + http
    net->SetModeHTTP(httpproxy, (s16) httpport, httpid);
    net->SetModeUUE(true);
    break;

  case 4:  // SOCKS4
    net->SetModeSOCKS4(httpproxy, (s16) httpport, httpid);
    break;

  case 5:  // SOCKS5
    net->SetModeSOCKS5(httpproxy, (s16) httpport, httpid);
    break;
  }

  // do the fetching and flushing
  if ((!fetcherr) && flusherr)
  {
    // If we only need the flush to succeed, then do it first
    // to prevent timeout problems.
    retcode2 = Flush(contest, net,0,force);
    if (retcode2 >= 1) // some blocks were transferred
      retcode1 = Fetch(contest, net, 2,force);// No proxy message display a second time
    else retcode1 = Fetch(contest, net,0,force);

  } else {                        // This is either a flush request, or an update request.
    retcode1 = Fetch(contest, net,0,force);
    if (retcode1 >= 1) // some blocks were transferred
      retcode2 = Flush(contest, net, 2,force);// No proxy message display a second time
    else retcode2 = Flush(contest, net, 0,force);
  }
  if (contest == 0) {
    if (randomchanged) {
      if (!WriteContestandPrefixConfig()) randomchanged=0;
    }
  }

  // delete the net
  delete net;

  #if defined(LURK)
  // Check if we need to hangup
  dialup.HangupIfNeeded();
  #endif

  if (fetcherr && flusherr) {
     // This is an explicit call to update() by the user.
     // Report if either type of error (flush/fetch) occurred
     retcode = min( retcode1*fetcherr , retcode2*flusherr );
  } else {
     // We're only worried about one of the flush/fetch.
     // Report its error and throw the other one away...
     retcode = (retcode1 * fetcherr) + (retcode2 * flusherr);
  }

  if ((retcode1>0) || (retcode1==-1) || (retcode2>0) || (retcode2==-1))
  {
    // did any net traffic occur?
    LogFlush(0); //checktosend(x)
  }
  return( retcode );
}

#endif //NONETWORK

// ---------------------------------------------------------------------------

u32 Client::Benchmark( u8 contest, u32 numk )
{
  ContestWork contestwork;

  u32 numkeys = 10000000L;
  u32 tslice;

  if (SelectCore() || SignalTriggered) return 0;
  if (numk != 0) numkeys = max(numk,1000000L);

  if (contest == 1)
    LogScreen( "Benchmarking RC5 with %d tests:\n", (int) numkeys );
  else if (contest == 2)
    LogScreen( "Benchmarking DES with %d tests:\n", (int) numkeys * 2 );
  else return 0;

  contestwork.key.lo = htonl( 0 );
  contestwork.key.hi = htonl( 0 );
  contestwork.iv.lo = htonl( 0 );
  contestwork.iv.hi = htonl( 0 );
  contestwork.plain.lo = htonl( 0 );
  contestwork.plain.hi = htonl( 0 );
  contestwork.cypher.lo = htonl( 0 );
  contestwork.cypher.hi = htonl( 0 );
  contestwork.keysdone.lo = htonl( 0 );
  contestwork.keysdone.hi = htonl( 0 );
  contestwork.iterations.lo = htonl( numkeys );
  contestwork.iterations.hi = htonl( 0 );

  (problem[0]).LoadState( &contestwork , (u32) (contest - 1) );

  (problem[0]).percent = 0;
  tslice = ( 100000L / PIPELINE_COUNT );

  #if (CLIENT_OS == OS_NETWARE)
    tslice = GetTimesliceBaseline(); //in cpucheck.cpp
  #endif

  while ( (problem[0]).Run( tslice , 0 ) == 0 )
    {
    if (!percentprintingoff)
      LogScreenPercent( 1 ); //logstuff.cpp - number of loaded problems

    #if (CLIENT_OS == OS_NETWARE)   //yield
      nwCliThreadSwitchLowPriority();
    #endif

    #if (CLIENT_OS == OS_AMIGAOS)
      if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C)
        SignalTriggered = UserBreakTriggered = 1;
    #endif
    if ( SignalTriggered )
      return 0;
    }
  LogScreenPercent( 1 ); //finish the percent bar

  #if 0 //could use this, but it shows the block number ("00000000:00000000")
    LogScreen("\n[%s] %s\n", CliGetTimeString( NULL, 1 ),
                CliGetMessageForProblemCompletedNoSave( &(problem[0]) ) );
    return (u32)0;  //unused, so we don't care
  #else
    struct timeval tv;
    char ratestr[32];
    double rate = CliGetKeyrateForProblemNoSave( &(problem[0]) );
    tv.tv_sec = (problem[0]).timehi;  //read the time the problem:run started
    tv.tv_usec = (problem[0]).timelo;
    CliTimerDiff( &tv, &tv, NULL );    //get the elapsed time
    LogScreen("\nCompleted in %s [%skeys/sec]\n", CliGetTimeString( &tv, 2 ),
                             CliGetKeyrateAsString( ratestr, rate ) );
    return (u32)(rate);
  #endif
}

// ---------------------------------------------------------------------------

s32 Client::SelfTest( u8 contest )
{
  s32 run;
  s32 successes = 0;
  const u32 (*test_cases)[TEST_CASE_COUNT][8];
  ContestWork contestwork;
  RC5Result rc5result;
  u64 expectedsolution;

  if (SelectCore()) return 0;
  if (contest == 1)
    {
    test_cases = (const u32 (*)[TEST_CASE_COUNT][8])&rc5_test_cases[0][0];
    LogScreen("Beginning RC5 Self-test.\n");
    }
  else if (contest == 2)
    {
    test_cases = (const u32 (*)[TEST_CASE_COUNT][8])&des_test_cases[0][0];
    LogScreen("Beginning DES Self-test.\n");
    }
  else return 0;

  for ( s32 i = 0 ; i < TEST_CASE_COUNT ; i++ )
  {
    // load test case
    if (contest == 1) {
      // RC5-64
      expectedsolution.lo = (*test_cases)[(int) i][0];
      expectedsolution.hi = (*test_cases)[(int) i][1];
    } else {
      // DES
      expectedsolution.lo = (*test_cases)[(int) i][0];
      expectedsolution.hi = (*test_cases)[(int) i][1];

      convert_key_from_des_to_inc ( (u32 *) &expectedsolution.hi, (u32 *) &expectedsolution.lo);

      // to test also success on complementary keys
      if (expectedsolution.hi & 0x00800000L)
      {
        expectedsolution.hi ^= 0x00FFFFFFL;
        expectedsolution.lo = ~expectedsolution.lo;
      }
    }
    contestwork.key.lo = htonl( expectedsolution.lo & 0xFFFF0000L);
    contestwork.key.hi = htonl( expectedsolution.hi );
    contestwork.iv.lo = htonl( (*test_cases)[(int) i][2] );
    contestwork.iv.hi = htonl( (*test_cases)[(int) i][3] );
    contestwork.plain.lo = htonl( (*test_cases)[(int) i][4] );
    contestwork.plain.hi = htonl( (*test_cases)[(int) i][5] );
    contestwork.cypher.lo = htonl( (*test_cases)[(int) i][6] );
    contestwork.cypher.hi = htonl( (*test_cases)[(int) i][7] );
    contestwork.keysdone.lo = htonl( 0 );
    contestwork.keysdone.hi = htonl( 0 );
    contestwork.iterations.lo = htonl( 0x00010000L );  // only need to get to xDEEO
    contestwork.iterations.hi = htonl( 0 );

    (problem[0]).LoadState( &contestwork , (u32) (contest-1) );
    while ( ( run = (problem[0]).Run( 0x4000 , 0 ) ) == 0 )
    {
      #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      SurrenderCPU();
      #elif (CLIENT_OS == OS_NETWARE)
      nwCliThreadSwitchLowPriority();
      #endif

      (problem[0]).GetResult( &rc5result );
    }

#if (CLIENT_OS == OS_AMIGAOS)
    if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C)
      SignalTriggered = UserBreakTriggered = 1;
#endif
    if ( SignalTriggered ) return 0;

    // switch on value of run
    if ( run == 1 )
    {
      s32 tmpcontest=(problem[0]).GetResult( &rc5result );

      successes++;
      if ( rc5result.result == RESULT_FOUND )
      {
        if ( (ntohl( rc5result.key.hi ) + ntohl( rc5result.keysdone.hi ) ) != expectedsolution.hi
          || (ntohl( rc5result.key.lo ) + ntohl( rc5result.keysdone.lo ) ) != expectedsolution.lo )
        {
          // failure occurred (wrong key)
          LogScreen( "Test %d FAILED: %08X:%08X - %08X:%08X\n", successes,
              (ntohl( rc5result.key.hi ) + ntohl( rc5result.keysdone.hi ) ),
              (ntohl( rc5result.key.lo ) + ntohl( rc5result.keysdone.lo ) ),
              expectedsolution.hi, expectedsolution.lo );
          successes *= -1;
          return( successes );
        } else {
          // match found
          if (tmpcontest == 1)
          {
            // DES...
            u32 hi = (*test_cases)[(int) i][1];
            u32 lo = (*test_cases)[(int) i][0];

            u32 hi2 = ntohl( rc5result.key.hi ) + (u32) ntohl( rc5result.keysdone.hi );
            u32 lo2 = ntohl( rc5result.key.lo ) + (u32) ntohl( rc5result.keysdone.lo );
            convert_key_from_inc_to_des (&hi2, &lo2);

            LogScreen( "Test %d Passed: %08X:%08X - %08X:%08X\n", successes,
              (u32) hi2, (u32) lo2, (u32) hi, (u32) lo);

          } else {
            // RC5...
            LogScreen( "Test %d Passed: %08X:%08X\n", successes,
              (u32) ntohl( rc5result.key.hi ) + (u32) ntohl( rc5result.keysdone.hi ),
              (u32) ntohl( rc5result.key.lo ) + (u32) ntohl( rc5result.keysdone.lo ));
          }
        }
      }
      else
      {
        // failure occurred (no solution)
        LogScreen( "Test %d FAILED: %08X:%08X - %08X:%08X\n", successes,
              0, 0, expectedsolution.hi, expectedsolution.lo );
        successes *= -1;
        return( successes );
      }
    }
  }
  LogScreen( "\n%d/%d %s Tests Passed\n", 
      (int) successes, (int) TEST_CASE_COUNT, ((contest == 1)?"RC5":"DES") );
  return( successes );
}

// ---------------------------------------------------------------------------

static int IsFilenameValid( const char *filename )
{ return ( *filename != 0 && strcmp( filename, "none" ) != 0 ); }

static int DoesFileExist( const char *filename )
{
  if ( !IsFilenameValid( filename ) )
    return 0;
#ifdef DONT_USE_PATHWORK
  return ( access( filename, 0 ) == 0 );
#else
  return ( access( GetFullPathForFilename( filename ), 0 ) == 0 );
#endif
}

// ---------------------------------------------------------------------------

#if defined(MULTITHREAD)
void Go_mt( void * parm )
{
// Serve both problem[cpunum] and problem[cpunum+numcputemp] until interrupted.
// 2 are used to avoid stalls when network traffic becomes required.
// The main thread of execution will remove finished blocks &
// insert new ones.
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
  char * FAR *argv = (char * FAR *) parm;
  #elif (CLIENT_OS == OS_NETWARE)
  char * *argv = (char * *) parm;
  #else
  char * *argv = (char * *) parm;
  sigset_t signals_to_block;
  #endif
  s32 tempi2;
  s32 numcputemp;
  s32 timeslice;
  u32 run;
  s32 niceness;

  #if (CLIENT_OS == OS_WIN32)
  DWORD LAffinity, LProcessAffinity, LSystemAffinity;
  OSVERSIONINFO osver;
  #endif

  tempi2 = atol(argv[0]);
  numcputemp = atol(argv[1]);
  timeslice = atol(argv[2]);
  niceness = atol(argv[3]);
//LogScreen("tempi2: %d\n",tempi2);
//LogScreen("numcpu: %d\n",numcputemp);
//LogScreen("timeslice: %d\n",timeslice);
//LogScreen("niceness: %d\n",niceness);


#if (CLIENT_OS == OS_WIN32)
  if (niceness == 0)
    SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_IDLE);

  osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&osver);
  if ((VER_PLATFORM_WIN32_NT == osver.dwPlatformId) && (numcputemp > 1))
  {
    if (GetProcessAffinityMask(GetCurrentProcess(), &LProcessAffinity, &LSystemAffinity))
    {
      LAffinity = 1L << tempi2;
      if (LProcessAffinity & LAffinity)
        SetThreadAffinityMask(GetCurrentThread(), LAffinity);
    }
  }
#elif (CLIENT_OS == OS_NETWARE)
  {
  nwCliInitializeThread( tempi2+1 ); //in netware.cpp
  }
#elif (CLIENT_OS == OS_OS2)
#elif (CLIENT_OS == OS_BEOS)
#else
  sigemptyset(&signals_to_block);
  sigaddset(&signals_to_block, SIGINT);
  sigaddset(&signals_to_block, SIGTERM);
  sigaddset(&signals_to_block, SIGKILL);
  sigaddset(&signals_to_block, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &signals_to_block, NULL);
#endif

  while (!SignalTriggered)
  {
    for (s32 tempi = tempi2; tempi < 2*numcputemp ; tempi += numcputemp)
    {
      run = 0;
      while (!SignalTriggered && (run == 0))
      {
#if (CLIENT_OS == OS_AMIGAOS)
        if (SetSignal(0L,0L) & SIGBREAKF_CTRL_C)
          SignalTriggered = UserBreakTriggered = 1;
#endif
        // This will return without doing anything if uninitialized...
        if (!pausefilefound) {
#if (CLIENT_OS == OS_NETWARE)
              //sets up and uses a polling procedure that runs as
              //an OS callback when the system enters an idle loop.
          run = nwCliRunProblemAsCallback( &(problem[tempi]),
                         timeslice / PIPELINE_COUNT, tempi2, niceness );
#else
          run = (problem[tempi]).Run( timeslice / PIPELINE_COUNT , tempi2 );
#endif
        } else {
          run = 0;
          sleep( 1 ); // don't race in this loop
        }
      }
    }
    sleep( 1 ); // Avoid looping in case no buffers become available, or pausefile found
  }
//   LogScreen("Exiting child thread %d\n",tempi2+1);
#if (CLIENT_OS == OS_BEOS)
  exit(0);
#endif
}
#endif

// ---------------------------------------------------------------------------

// returns:
//    -2 = exit by error (all contests closed)
//    -1 = exit by error (critical)
//     0 = exit for unknown reason
//     1 = exit by user request
//     2 = exit by exit file check
//     3 = exit by time limit expiration
//     4 = exit by block count expiration
s32 Client::Run( void )
{
  FileEntry fileentry;
  RC5Result rc5result;

#if defined(MULTITHREAD)
  char buffer[MAXCPUS][4][40];
  #if (CLIENT_OS == OS_BEOS)
    static char * thstart[MAXCPUS][4];
  #else
    char * thstart[MAXCPUS][4];
  #endif
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_NETWARE)
    unsigned long threadid[MAXCPUS];
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadid[MAXCPUS];
    char thread_name[32];
    char thread_error;
    long be_priority;
    static status_t be_exit_value;
  #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
    pthread_attr_t thread_sched[MAXCPUS];
    pthread_t threadid[MAXCPUS];
  #else
    pthread_t threadid[MAXCPUS];
  #endif
#endif

  s32 count = 0, nextcheckpointtime = 0;
  s32 TimeToQuit = 0, getbuff_errs = 0;

#if defined(MULTITHREAD)
  #if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
    connectrequested = 0;         // uses public class member
  #else
    u32 connectrequested = 0;
  #endif
  u32 connectloops = 0;
#endif

  s32 cpu_i;
  s32 exitchecktime;
  s32 tmpcontest;
  s32 exitcode = 0;


  // --------------------------------------
  // Recover blocks from checkpoint files
  // --------------------------------------

  if ( DoesFileExist( checkpoint_file[0] ) )
    {
    s32 recovered = CkpointToBufferInput(0); // Recover any checkpointed information in case we abnormally quit.
    if (recovered != 0) Log("Recovered %d block%s from RC5 checkpoint file\n",recovered,recovered==1?"":"s");
    }
  if ( DoesFileExist( checkpoint_file[1] ) )
    {
    s32 recovered = CkpointToBufferInput(1); // Recover any checkpointed information in case we abnormally quit.
    if (recovered != 0) Log("Recovered %d block%s from DES checkpoint file\n",recovered,recovered==1?"":"s");
    }

  // --------------------------------------
  // Select an appropriate core, niceness and timeslice setting
  // --------------------------------------

  if (SelectCore())
    return -1;

  #if (CLIENT_CPU == CPU_POWERPC) //this should be in SelectCore()
  switch (whichcrunch)
    {
    case 0:
      Log("Using the 601 core.\n\n");
      break;
    case 1:
      Log("Using the 603/604/750 core.\n\n");
      break;
  }
#endif

  SetNiceness();

  // --------------------------------------
  // Initialize the timers
  // --------------------------------------

  timeStarted = time( NULL );
  exitchecktime = timeStarted + 5;

  // --------------------------------------
  // Clear the signal triggers
  // --------------------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
    SetSignal(0L, SIGBREAKF_CTRL_C);
  #endif
  SignalTriggered = UserBreakTriggered = pausefilefound = 0;

  // --------------------------------------
  // Determine the number of problems to work with. Number is used everywhere.
  // --------------------------------------

  int load_problem_count = 1;
  #ifdef MULTITHREAD
    load_problem_count = 2*numcputemp;
    #if (CLIENT_OS == OS_NETWARE)
    {
      if (numcputemp == 1) //NetWare client prefers non-MT if only one 
        load_problem_count = 1; //thread/processor is to used
    }
    #endif
  #endif

  // --------------------------------------
  // Set up initial state of each problem[]...
  // uses 2 active buffers per CPU to avoid stalls
  // --------------------------------------

  for (cpu_i = 0; cpu_i < load_problem_count; cpu_i++ )
  {
#if (CLIENT_CPU == CPU_X86)
    if ((cpu_i != 0) && (cpu_i != numcputemp) 
#if defined(MMX_BITSLICER)
       && (des_unit_func!=des_unit_func_mmx) // we're not using the mmx cores
#endif
       && (cpu_i != 1) && (cpu_i != 1 + numcputemp))
      {
      // Not the 1st or 2nd cracking thread...
      // Must do RC5.  DES x86 cores aren't multithread safe.  There are 2 separate cores.
      // Note that if rc5 contest is over, this will return -2...
      count = GetBufferInput( &fileentry , 0);
      if (contestdone[0])
        count=-2; // means that this thread won't actually start.
      }
    else
#endif
    {
      if (getbuff_errs == 0)
      {
        if (!contestdone[ (int) preferred_contest_id ])
        {
          // Neither contest is done...
          count = GetBufferInput( &fileentry , (u8) preferred_contest_id);
          if (contestdone[ (int) preferred_contest_id ]) // This contest just finished.
          {
            goto PreferredIsDone1;
          }
          else
          {
            if (count == -3)
            {
              // No DES blocks available while in offline mode.  Do rc5...
              count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
            }
          }
        }
        else
        {
          // Preferred contest is done...
PreferredIsDone1:
          count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
          if (contestdone[ ! preferred_contest_id ])
          {
            // This contest just finished.
            count = -2; // Both contests finished!
          }
        }
      }
    }

    if (count == -1)
    {
      getbuff_errs++;
    }
    else if ((!nonewblocks) && (count != -2))
    {
      // LoadWork expects things descrambled.
      Descramble( ntohl( fileentry.scramble ),
                     (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
      // If a block was finished with an 'odd' number of keys done, then make it redo the last
      // key -- this will prevent a 2-pipelined core from looping forever.
      if ((ntohl(fileentry.iterations.lo) & 0x00000001L) == 1)
      {
        fileentry.iterations.lo = htonl((ntohl(fileentry.iterations.lo) & 0xFFFFFFFEL) + 1);
        fileentry.key.lo = htonl(ntohl(fileentry.key.lo) & 0xFEFFFFFFL);
      }
      if (fileentry.contest != 1)
        fileentry.contest=0;

      // If this is a partial DES block, and completed by a different cpu/os/build, then
      // reset the keysdone to 0...
      if (fileentry.contest == 1)
      {
        if ( (ntohl(fileentry.keysdone.lo)!=0) || (ntohl(fileentry.keysdone.hi)!=0) )
        {
          if ((fileentry.cpu != CLIENT_CPU) || (fileentry.os != CLIENT_OS) ||
              (fileentry.buildhi != CLIENT_CONTEST) || (fileentry.buildlo != CLIENT_BUILD))
          {
            fileentry.keysdone.lo = fileentry.keysdone.hi = htonl(0);
            LogScreen("Read partial DES block from another cpu/os/build.\n");
            LogScreen("Marking entire block as unchecked.\n");
          }
        }
      }

      {
        if (cpu_i==0 && load_problem_count>1)
          Log( "[%s] %s\n", CliGetTimeString(NULL,1),
                                  "Loading two blocks per thread...");

        Log( "[%s] %s\n", CliGetTimeString(NULL,1),
                       CliGetMessageForFileentryLoaded( &fileentry ) );

        //only display the "remaining blocks in file" once
        static char have_loaded_buffers[2]={0,0};
        have_loaded_buffers[fileentry.contest]=1;

        if (cpu_i == (load_problem_count-1)) //last loop?
        {
          if (load_problem_count == 2)
            Log("[%s] 1 Child thread has been started.\n", Time());
          else if (load_problem_count > 2)
            Log("[%s] %d Child threads ('A'%s'%c') have been started.\n",
              Time(), load_problem_count>>1,
              ((load_problem_count>4)?("-"):(" and ")),
              'A'+((load_problem_count>>1)-1));

          for (s32 tmpc = 0; tmpc < 2; tmpc++) //once for each contest
          {
            if (have_loaded_buffers[(int) tmpc]) //load any of this type?
            {
              int in = (int) CountBufferInput((u8) tmpc);
              int out = (int) CountBufferOutput((u8) tmpc);
              Log( "[%s] %d %s block%s remain%s in file %s\n", CliGetTimeString(NULL,1),
                in,
                CliGetContestNameFromID((int) tmpc),
                in == 1 ? "" : "s",
                in == 1 ? "s" : "",
                (nodiskbuffers ? "(memory-in)" :
#ifdef DONT_USE_PATHWORK
                ini_in_buffer_file[(int) tmpc]));
#else
                in_buffer_file[(int) tmpc]));
#endif
              Log( "[%s] %d %s block%s %s in file %s\n", CliGetTimeString(NULL,1),
                out,
                CliGetContestNameFromID((int) tmpc),
                out == 1 ? "" : "s",
                out == 1 ? "is" : "are",
                (nodiskbuffers ? "(memory-out)" :
#ifdef DONT_USE_PATHWORK
                ini_out_buffer_file[(int) tmpc]));
#else
                out_buffer_file[(int) tmpc]));
#endif
            }
          }
        }
      }

      (problem[(int) cpu_i]).LoadState( (ContestWork *) &fileentry , (u32) (fileentry.contest) );

      //----------------------------
      //spin off a thread for this problem
      //----------------------------

#if defined(MULTITHREAD)
      {
        //Only launch a thread if we have really loaded 2*threadcount buffers
        if ((load_problem_count > 1) && (cpu_i < numcputemp))
        {
          // Start the thread for this cpu
          sprintf(buffer[cpu_i][0],"%d",(int)cpu_i);
          sprintf(buffer[cpu_i][1],"%d",(int)numcputemp);
          sprintf(buffer[cpu_i][2],"%d",(int)timeslice);
          sprintf(buffer[cpu_i][3],"%d",(int)niceness);
          thstart[cpu_i][0] = &buffer[cpu_i][0][0];
          thstart[cpu_i][1] = &buffer[cpu_i][1][0];
          thstart[cpu_i][2] = &buffer[cpu_i][2][0];
          thstart[cpu_i][3] = &buffer[cpu_i][3][0];
#if (CLIENT_OS == OS_WIN32)
          threadid[cpu_i] = _beginthread( Go_mt, 8192, thstart[cpu_i]);
          //if ( threadid[cpu_i] == 0)
          //  threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_OS2)
          threadid[cpu_i] = _beginthread( Go_mt, NULL, 8192, thstart[cpu_i]);
          if ( threadid[cpu_i] == -1)
            threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_NETWARE)
          threadid[cpu_i] = BeginThread( Go_mt, NULL, 8192, thstart[cpu_i]);
          if ( threadid[cpu_i] == -1)
            threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_BEOS)
          switch(niceness)
          {
            case 0: be_priority = B_LOW_PRIORITY; break;
            case 1: be_priority = (B_LOW_PRIORITY + B_NORMAL_PRIORITY) / 2; break;
            case 2: be_priority = B_NORMAL_PRIORITY; break;
            default: be_priority = B_LOW_PRIORITY; break;
          }
          sprintf(thread_name, "RC5DES crunch#%d", cpu_i + 1);
          threadid[cpu_i] = spawn_thread((long (*)(void *)) Go_mt, thread_name,
                be_priority, (void *)thstart[cpu_i]);
          thread_error = (threadid[cpu_i] < B_NO_ERROR);
          if (!thread_error)
            thread_error =  (resume_thread(threadid[cpu_i]) != B_NO_ERROR);
          if (thread_error)
            threadid[cpu_i] = NULL; //0
#elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
          pthread_attr_init(&thread_sched[cpu_i]);
          pthread_attr_setscope(&thread_sched[cpu_i],PTHREAD_SCOPE_SYSTEM);
          pthread_attr_setinheritsched(&thread_sched[cpu_i],PTHREAD_INHERIT_SCHED);
          if (pthread_create( &threadid[cpu_i], &thread_sched[cpu_i], (void *(*)(void*)) Go_mt, thstart[cpu_i]) )
            threadid[cpu_i] = (pthread_t) NULL; //0
#else
          if (pthread_create( &threadid[cpu_i], NULL, (void *(*)(void*)) Go_mt, thstart[cpu_i]) )
            threadid[cpu_i] = (pthread_t) NULL; //0
#endif

          if ( !threadid[cpu_i] )
          {
            Log("[%s] Could not start child thread '%c'.\n",Time(),cpu_i+'A');
            return(-1);  //All those loaded blocks are gonna get lost
          }
        }
      }
#endif
    } //if ((!nonewblocks) && (count != -2))
  } //for (cpu_i = 0; cpu_i < load_problem_count; cpu_i ++)


  //------------------------------------
  // display the percent bar so the user sees some action
  //------------------------------------

  if (!percentprintingoff)
    LogScreenPercent( load_problem_count ); //logstuff.cpp

  //============================= MAIN LOOP =====================
  //now begin looping until we have a reason to quit
  //------------------------------------

  // -- cramer - until we have a better way of telling how many blocks
  //             are loaded and if we can get more, this is gonna be a
  //             a little complicated.  getbuff_errs and nonewblocks
  //             control the exit process.  getbuff_errs indicates the
  //             number of attempts to load new blocks that failed.
  //             nonewblocks indcates that we aren't get anymore blocks.
  //             Together, they can signal when the buffers have been
  //             truely exhausted.  The magic below is there to let
  //             the client finish processing those blocks before exiting.

  // Start of MAIN LOOP
  while (TimeToQuit == 0)
  {
    //------------------------------------
    //Do keyboard stuff for clients that allow user interaction during the run
    //------------------------------------

    #if ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)) && !defined(NEEDVIRTUALMETHODS)
    {
      while ( kbhit() )
      {
        int hitchar = getch();
          if (hitchar == 0) //extended keystroke
            getch();
          else
          {
            if (hitchar == 3 || hitchar == 'X' || hitchar == 'x' || hitchar == '!')
            {
              // exit after current blocks
              if (blockcount > 0)
              {
                blockcount = min(blockcount, (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp));
              } else {
                blockcount = (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp);
              }
              Log("Exiting after current block\n");
              exitcode = 1;
            }
        #if (defined(MULTITHREAD) && !defined(NONETWORK))
            if (hitchar == 'u' || hitchar == 'U')
            {
              Log("Keyblock Update forced\n");
              connectrequested = 1;
            }
        #endif
          }
        }
    }
    #endif

    //------------------------------------
    //special update request (by keyboard or by lurking) handling
    //------------------------------------

    #if (defined(MULTITHREAD) && !defined(NONETWORK))
    {
      if ((connectoften && ((connectloops++)==19)) || (connectrequested > 0) )
      {
        // Connect every 20*3=60 seconds
        // Non-MT 60 + (time for a client.run())
        connectloops=0;
        if (connectrequested == 1) // forced update by a user
          {
          Update(0 ,1,1,1);  // RC5 We care about the errors, force update.
          Update(1 ,1,1,1);  // DES We care about the errors, force update.
          LogScreen("Keyblock Update completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 2) // automatic update
          {
          Update(0 ,0,0);  // RC5 We don't care about any of the errors.
          Update(1 ,0,0);  // DES 
          connectrequested=0;
          }
        else if (connectrequested == 3) // forced flush
          {
          Flush(0,NULL,1,1); // Show errors, force flush
          Flush(1,NULL,1,1);
          LogScreen("Flush request completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 4) // forced fetch
          {
          Fetch(0,NULL,1,1); // Show errors, force fetch
          Fetch(1,NULL,1,1);
          LogScreen("Fetch request completed.\n");
          connectrequested=0;
          };


      }
    }
    #endif

    //------------------------------------
    // Lurking
    //------------------------------------

#if defined(LURK)
if(dialup.lurkmode) // check to make sure lurk mode is enabled
  connectrequested=dialup.CheckIfConnectRequested();
#endif

    //------------------------------------
    //sleep, run or pause...
    //------------------------------------

    if (load_problem_count > 1) //ie MUTLITHREAD
      {
      // prevent the main thread from racing & bogging everything down.
      sleep(3);
      }
    else if (pausefilefound) //threads have their own sleep section
      {
      #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        SurrenderCPU();
      #elif (CLIENT_OS != OS_DOS)
        sleep(1);
      #endif
      }
    else //only one problem and we are not paused
      {
      //Actually run a problem
      #if (CLIENT_OS == OS_NETWARE)
        {
        //sets up and uses a polling procedure that runs as
        //an OS callback when the system enters an idle loop.
        nwCliRunProblemAsCallback( &(problem[0]),
                                timeslice/PIPELINE_COUNT, 0 , niceness );
        }
      #else
        {
        (problem[0]).Run( timeslice / PIPELINE_COUNT , 0 );
        #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        SurrenderCPU();
        #endif
        }
      #endif //if non-mt, netware or not
      }
   

    //------------------------------------
    //update the status bar
    //------------------------------------

    if (!percentprintingoff)
      LogScreenPercent( load_problem_count ); //logstuff.cpp


    //------------------------------------
    //now check all problems for change, do checkpointing, reloading etc
    //------------------------------------

    for (cpu_i = 0; ((!pausefilefound) && (!SignalTriggered) && (cpu_i < load_problem_count)); cpu_i++)
    {

      // -------------
      // check for finished blocks that need reloading
      // -------------

      // Did any threads finish a block???
      if ((problem[(int) cpu_i]).finished == 1)
      {
        (problem[(int) cpu_i]).GetResult( &rc5result );

        //-----------------
        //only do something if RESULT_FOUND or RESULT_NOTHING
        //Q: when can it be finished AND result_working?
        //-----------------

        if ((rc5result.result == RESULT_FOUND) || (rc5result.result == RESULT_NOTHING))
        {
          //---------------------
          //print the keyrate and update the totals for this contest
          //---------------------

          {
            Log( "\n[%s] %s", CliGetTimeString(NULL,1), /* == Time() */
                   CliGetMessageForProblemCompleted( &(problem[(int) cpu_i]) ) );
          }

          //----------------------------------------
          // Figure out which contest block was from, and increment totals
          //----------------------------------------

          tmpcontest = (u8) (problem[(int) cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );

          totalBlocksDone[(int) tmpcontest]++;

          //----------------------------------------
          // Print contest totals
          //----------------------------------------

          // Detect/report any changes to the total completed blocks...


          {
          //display summaries only of contests w/ more than one block done
          int i = 1;
          for (s32 tmpc = 0; tmpc < 2; tmpc++)
            {
              if (totalBlocksDone[(int) tmpc] > 0)
                {
                  Log( "%c%s%c Summary: %s\n",
                       ((i == 1) ? ('[') : (' ')), CliGetTimeString(NULL, i),
                       ((i == 1) ? (']') : (' ')), CliGetSummaryStringForContest((int) tmpc) );
                  if ((--i) < 0) i = 0;
                }
            }
          }

          //---------------------
          //put the completed problem away
          //---------------------

          tmpcontest = fileentry.contest = (u8) (problem[(int) cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );

          // make it into a reply
          if (rc5result.result == RESULT_FOUND)
          {
            consecutivesolutions[fileentry.contest]++;
            if (keyport == 3064)
                LogScreen("Success\n");
            fileentry.op = htonl( OP_SUCCESS_MULTI );
            fileentry.key.lo = htonl( ntohl( fileentry.key.lo ) +
                                ntohl( fileentry.keysdone.lo ) );
          }
          else
          {
            if (keyport == 3064)
              LogScreen("Success was not detected!\n");
            fileentry.op = htonl( OP_DONE_MULTI );
          }

          fileentry.os = CLIENT_OS;
          fileentry.cpu = CLIENT_CPU;
          fileentry.buildhi = CLIENT_CONTEST;
          fileentry.buildlo = CLIENT_BUILD;
          strncpy( fileentry.id, id , sizeof(fileentry.id)-1); // set id for this block
          fileentry.id[sizeof(fileentry.id)-1]=0;  // in case id>58 bytes, truncate.

          fileentry.checksum =
              htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          Scramble( ntohl( fileentry.scramble ),
                     (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if ( PutBufferOutput( &fileentry ) == -1 )
            {
            Log( "PutBuffer Error\n" );

            // Block didn't get put into a buffer, subtract it from the count.
            totalBlocksDone[(int)tmpcontest]--;
            };

          //---------------------
          //delete the checkpoint file, info is outdated
          //---------------------

          // Checkpoint info just became outdated...

          if ( DoesFileExist( checkpoint_file[0] ) )
            EraseCheckpointFile(checkpoint_file[0]); //buffwork.cpp
          if ( DoesFileExist( checkpoint_file[1] ) )
            EraseCheckpointFile(checkpoint_file[1]); //buffwork.cpp

          //---------------------
          // See if the request to quit after the completed block
          //---------------------
          if(exitcode == 1) TimeToQuit=1; // Time to quit

          //---------------------
          //now load another block for this contest
          //---------------------

          // Get another block...

#if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
          if ((cpu_i%numcputemp)>=2)
          {
              // Not the 1st or 2nd cracking thread...
              // Must do RC5.  DES x86 cores aren't multithread safe.
              // Note that if rc5 contest is over, this will return -2...
              count = GetBufferInput( &fileentry , 0);
            if (contestdone[0])
              count = -2;
          }
          else
#endif
          {
            if (getbuff_errs == 0)
            {
              if (!contestdone[ (int) preferred_contest_id ])
              {
                // Neither contest is done...
                count = GetBufferInput( &fileentry , (u8) preferred_contest_id);
                if (contestdone[ (int) preferred_contest_id ]) // This contest just finished.
                {
                  goto PreferredIsDone2;
                }
                else
                {
                  if (count == -3)
                  {
                    // No DES blocks available while in offline mode.  Do rc5...
                    count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
                  }
                }
              }
              else
              {
                // Preferred contest is done...
      PreferredIsDone2:
                count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
                if (contestdone[ ! preferred_contest_id ])
                {
                  // This contest just finished.
                  count = -2; // Both contests finished!
                }
              }
            }
            else if (nonewblocks) getbuff_errs++; // cramer magic #1 (mt)
          }

          if (count < 0)
          {
            getbuff_errs++; // cramer magic #2 (non-mt)
            if (!nonewblocks)
            {
              TimeToQuit=1; // Force blocks to be saved
              exitcode = -2;
              continue;  //break out of the next cpu_i loop
            }
          }

          //---------------------
          // correct any potential problems in the freshly loaded fileentry
          //---------------------

          if (!nonewblocks)
          {
            // LoadWork expects things descrambled.
            Descramble( ntohl( fileentry.scramble ),
                       (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
            // If a block was finished with an 'odd' number of keys done, then make it redo the last
            // key -- this will prevent a 2-pipelined core from looping forever.
            if ((ntohl(fileentry.iterations.lo) & 0x00000001L) == 1)
            {
              fileentry.iterations.lo = htonl((ntohl(fileentry.iterations.lo) & 0xFFFFFFFEL) + 1);
              fileentry.key.lo = htonl(ntohl(fileentry.key.lo) & 0xFEFFFFFFL);
            }
            if (fileentry.contest != 1)
              fileentry.contest=0;

            // If this is a partial DES block, and completed by a different
            // cpu/os/build, then reset the keysdone to 0...
            if (fileentry.contest == 1)
            {
              if ( (ntohl(fileentry.keysdone.lo)!=0) || (ntohl(fileentry.keysdone.hi)!=0) )
              {
                if ((fileentry.cpu != CLIENT_CPU) || (fileentry.os != CLIENT_OS) ||
                    (fileentry.buildhi != CLIENT_CONTEST) || (fileentry.buildlo != CLIENT_BUILD))
                {
                  fileentry.keysdone.lo = fileentry.keysdone.hi = htonl(0);
                  LogScreen("Read partial DES block from another cpu/os/build.\n");
                  LogScreen("Marking entire block as unchecked.\n");
                }
              }
            }
          }

          //---------------------
          // display the status of the file buffers
          //---------------------

          if (!nonewblocks)
          {
            int outcount = (int) CountBufferOutput((u8) fileentry.contest);
            Log( "[%s] %s\n", CliGetTimeString(NULL,1), /* == Time() */
                              CliGetMessageForFileentryLoaded( &fileentry ) );
            Log( "[%s] %d %s block%s remain%s in file %s\n"
                 "[%s] %d %s block%s %s in file %s\n",
                 CliGetTimeString(NULL,1), count, CliGetContestNameFromID(fileentry.contest),
                 count == 1 ? "" : "s", count == 1 ? "s" : "",
                 (nodiskbuffers ? "(memory-in)" :
#ifdef DONT_USE_PATHWORK
                 ini_in_buffer_file[(int)fileentry.contest]),
#else
                 in_buffer_file[(int)fileentry.contest]),
#endif
                 CliGetTimeString(NULL,1), outcount, CliGetContestNameFromID(fileentry.contest),
                 outcount == 1 ? "" : "s", outcount == 1 ? "is" : "are",
                 (nodiskbuffers ? "(memory-out)" :
#ifdef DONT_USE_PATHWORK
                 ini_out_buffer_file[(int)fileentry.contest]) );
#else
                 out_buffer_file[(int)fileentry.contest]) );
#endif
          }

          //---------------------
          // now load the problem with the fileentry
          //---------------------
          if (!nonewblocks)
            (problem[(int)cpu_i]).LoadState( (ContestWork *) &fileentry , (u32) (fileentry.contest) );

        } // end (if 'found' or 'nothing')

        DoCheckpoint( load_problem_count );
      } // end(if finished)
    } // endfor(cpu_i)

    //----------------------------------------
    // Check for time limit...
    //----------------------------------------

    if ( ( minutes > 0 ) &&
           (s32) ( time( NULL ) > (s32) ( timeStarted + ( 60 * minutes ) ) ) )
    {
      Log( "\n[%s] Shutdown - %u.%02u hours expired\n", Time(), minutes/60, (minutes%60) );
      TimeToQuit = 1;
      exitcode = 3;
    }

    //----------------------------------------
    // Check for user break
    //----------------------------------------

#if (CLIENT_OS == OS_AMIGAOS)
    if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C )
      SignalTriggered = UserBreakTriggered = 1;
#endif
    if ( SignalTriggered || UserBreakTriggered )
    {
      Log( "\n[%s] Shutdown message received - Block being saved.\n", Time() );
      TimeToQuit = 1;
      if (UserBreakTriggered) exitcode = 1;
    }

    //----------------------------------------
    // Check for 32 consecutive solutions
    //----------------------------------------

    for (int tmpc = 0; tmpc < 2; tmpc++)
    {
      const char *contname = CliGetContestNameFromID( tmpc ); //clicdata.cpp
      if ((consecutivesolutions[tmpc] >= 32) && !contestdone[tmpc])
      {
        Log( "\n[%s] Too many consecutive %s solutions detected.\n", Time(), contname );
        Log( "[%s] Either the contest is over, or this client is pointed at a test port.\n", Time() );
        Log( "[%s] Marking %s contest as over\n", Time(), contname );
        Log( "[%s] Further %s blocks will not be processed.\n", Time(), contname );
        contestdone[tmpc] = 1;
        WriteContestandPrefixConfig( );
      }
    }
    if (contestdone[0] && contestdone[1])
    {
      TimeToQuit = 1;
      Log( "\n[%s] Both RC5 and DES are marked as finished.  Quitting.\n", Time() );
      exitcode = -2;
    }

    //----------------------------------------
    // Has -runbuffers exhausted all buffers?
    //----------------------------------------

    // cramer magic (voodoo)
#ifdef MULTITHREAD
    if (nonewblocks > 0 && (getbuff_errs >= 2*numcputemp))
#else
    if (nonewblocks > 0 && getbuff_errs)
#endif
    {
      TimeToQuit = 1;
      exitcode = 4;
    }

    //----------------------------------------
    // Reached the -b limit?
    //----------------------------------------

    // Done enough blocks?
    if ( ( blockcount > 0 ) && ( totalBlocksDone[0]+totalBlocksDone[1] >= (u32) blockcount ) )
    {
      Log( "[%s] Shutdown - %d blocks completed\n", Time(), (u32) totalBlocksDone[0]+totalBlocksDone[1] );
      TimeToQuit = 1;
      exitcode = 4;
    }

    if ( time( NULL ) > exitchecktime )
    {
    exitchecktime = time( NULL ) + exitfilechecktime; // At most once every 30 seconds by default
    if (!noexitfilecheck) // Check if file "exitrc5.now" exists...
      {
        //----------------------------------------
        // Found an "exitrc5.now" flag file?
        //----------------------------------------
        if( DoesFileExist( exit_flag_file ) )
        {
          Log( "[%s] Found \"exitrc5" EXTN_SEP "now\" file.  Exiting.\n", Time() );
          TimeToQuit = 1;
          exitcode = 2;
        }
      }
    }
    //------------------------------------
    // Does a pausefile exist?
    //------------------------------------

    pausefilefound = DoesFileExist( pausefile );

    //----------------------------------------
    // Are we quitting?
    //----------------------------------------

    if ( TimeToQuit )
    {
      // ----------------
      // Shutting down: shut down threads
      // ----------------

      SignalTriggered = 1; // will make other threads exit

#if defined(MULTITHREAD)

      LogScreen("Quitting...\n");

      // Wait for all threads to end...
      for (cpu_i = 0; cpu_i < numcputemp; cpu_i++)
      {
#if (CLIENT_OS == OS_OS2)
        DosWaitThread(&threadid[cpu_i], DCWW_WAIT);
#elif (CLIENT_OS == OS_WIN32)
        WaitForSingleObject((HANDLE)threadid[cpu_i], INFINITE);
#elif (CLIENT_OS == OS_BEOS)
        wait_for_thread(threadid[cpu_i], &be_exit_value);
#elif (CLIENT_OS == OS_NETWARE)
        nwCliWaitForThreadExit( threadid[cpu_i] ); //in netware.cpp
#else
        pthread_join(threadid[cpu_i], NULL);
#endif
      }
#endif

      // ----------------
      // Shutting down: save problem buffers
      // ----------------

      for (cpu_i = (load_problem_count - 1); cpu_i >= 0; cpu_i-- )
      {
        if ((problem[(int)cpu_i]).IsInitialized())
        {
          fileentry.contest = (u8) (problem[(int)cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );
          fileentry.op = htonl( OP_DATA );

          fileentry.os = CLIENT_OS;
          fileentry.cpu = CLIENT_CPU;
          fileentry.buildhi = CLIENT_CONTEST;
          fileentry.buildlo = CLIENT_BUILD;

          fileentry.checksum =
                 htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          u32 temphi = ntohl( fileentry.key.hi );
          u32 templo = ntohl( fileentry.key.lo );
          u32 percent2 = (u32) ( (double) 10000.0 *
                           ( (double) ntohl(fileentry.keysdone.lo) /
                              (double) ntohl(fileentry.iterations.lo) ) );
          Scramble( ntohl( fileentry.scramble ),
                       (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if ( PutBufferInput( &fileentry ) == -1 )
          {
            Log( "Buffer Error\n" );
          }
          else
          {
            Log( "[%s] Saved block %08lX:%08lX (%d.%02d percent complete)\n",
                Time(), (unsigned long) temphi, (unsigned long) templo,
                percent2/100, percent2%100 );
          }
        }
      } //endfor(cpu_i)

      // ----------------
      // Shutting down: delete checkpoint files
      // ----------------

      if ( DoesFileExist( checkpoint_file[0] ) )
        EraseCheckpointFile(checkpoint_file[0]);
      if ( DoesFileExist( checkpoint_file[1] ) )
        EraseCheckpointFile(checkpoint_file[1]);

      // ----------------
      // Shutting down: do a net flush if we don't have diskbuffers
      // ----------------

      // no disk buffers -- we had better flush everything.
      if (nodiskbuffers)
      {
        SignalTriggered = 2; // ignored by flush.
        ForceFlush((u8) preferred_contest_id ) ;
        ForceFlush((u8) ! preferred_contest_id );
        SignalTriggered = 1; // not really required.
      }

    } // TimeToQuit

    //----------------------------------------
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit)
    {
      // Time to checkpoint?
      if ((IsFilenameValid( checkpoint_file[0] ) ||
           IsFilenameValid( checkpoint_file[1] ))
           && (!nodiskbuffers) && (!pausefilefound))
        {
       if ( (!TimeToQuit ) && ( ( (s32) time( NULL ) ) > ( (s32) nextcheckpointtime ) ) )

        {
          nextcheckpointtime = time(NULL) + checkpoint_min * 60;
          //Checkpoints may be slightly late (a few seconds). However,
          //this eliminates checkpoint catchup due to pausefiles/clock
          //changes/other nasty things that change the clock
          DoCheckpoint(load_problem_count);
        }
      } // Checkpointing
    }

  }  // End of MAIN LOOP

  //======================END OF MAIN LOOP =====================

  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif
  return exitcode;
}

// ---------------------------------------------------------------------------

void Client::DoCheckpoint( int load_problem_count )
{
  FileEntry fileentry;

  for (int j = 0; j < 2; j++)
  {
    if ( IsFilenameValid(checkpoint_file[j] ) )
    {
      EraseCheckpointFile(checkpoint_file[j]); // Remove prior checkpoint information (if any).

      for (int cpu_i = 0 ; cpu_i < (int) load_problem_count ; cpu_i++)
      {
        fileentry.contest = (u8) (problem[cpu_i]).RetrieveState( (ContestWork *) &fileentry , 0 );
        if (fileentry.contest == j)
        {
          fileentry.op = htonl( OP_DATA );

          fileentry.os = CLIENT_OS;
          fileentry.cpu = CLIENT_CPU;
          fileentry.buildhi = CLIENT_CONTEST;
          fileentry.buildlo = CLIENT_BUILD;

          fileentry.checksum =
              htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          Scramble( ntohl( fileentry.scramble ),
                      (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if (InternalPutBuffer( this->checkpoint_file[j], &fileentry ) == -1)
            Log( "Checkpoint Buffer Error\n" );
        }
      } //endfor(cpu_i)
    }
  }
}

// ---------------------------------------------------------------------------

s32 Client::SetContestDoneState( Packet * packet)
{
  u32 detect;

  // Set the contestdone state, if possible...
  // Move contestdone[] from 0->1, or 1->0.
  detect = 0;
  if (packet->descontestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[1]==0) {detect = 2; contestdone[1] = 1;}
  } else {
    if (contestdone[1]==1) {detect = 2; contestdone[1] = 0;}
  }
  if (detect == 2) {
    Log( "Received notification: %s contest %s.\n",
         (detect == 2 ? "DES" : "RC5"),
         (contestdone[(int)detect-1]?"is not currently active":"has started") );
  }

  if (packet->rc564contestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[0] == 0) {detect = 1; contestdone[0] = 1;}
  } else {
    if (contestdone[0] == 1) {detect = 1; contestdone[0] = 0;}
  }
  if (detect == 1) {
    Log( "Received notification: %s CONTEST %s\n",
        (detect == 2 ? "DES" : "RC5"),
        (contestdone[(int)detect-1]?"IS OVER":"HAS STARTED") );
  }

  if (detect != 0) {
    WriteContestandPrefixConfig();
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------

#if !defined(NOMAIN)
int main( int argc, char *argv[] )
{
  s32 inimissing;
  int i;

#if (CLIENT_OS == OS_RISCOS)
  riscos_in_taskwindow = riscos_check_taskwindow();
  if (riscos_find_local_directory(argv[0])) 
    return -1;
#endif

  // This is the main client object.  'Static' since allocating such
  // large objects off of the stack may cause problems for some and
  // also the NT Service keeps a pointer to it.
  static Client client;

#if (CLIENT_OS == OS_NETWARE) // create stdout/screen, set cwd etc.
  // and save pointer to client so functions in netware.cpp can get at the
  // filenames and niceness level
  if ( nwCliInitClient( argc, argv, &client )  ) 
    return -1;
#endif

  // set up break handlers
  CliSetupSignals();

  // determine the filename of the ini file
  if ( getenv( "RC5INI" ) != NULL )
    {
    strncpy( client.inifilename, getenv( "RC5INI" ), 127 );
    }
  else
    {
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || \
    (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN32) || \
    (CLIENT_OS == OS_OS2)
    char fndrive[_MAX_DRIVE], fndir[_MAX_DIR], fname[_MAX_FNAME], fext[_MAX_FNAME];
    _splitpath(argv[0], fndrive, fndir, fname, fext);
    _makepath(client.inifilename, fndrive, fndir, fname, EXTN_SEP "ini");
    strcpy(client.exepath, fndrive);   // have the drive
    strcat(client.exepath, fndir);     // append dir for fully qualified path
    strcpy(client.exename, fname);     // exe filename
    strcat(client.exename, fext);      // tack on extention
    #elif (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_DOS)
    //not really needed for netware (appname in argv[0] won't be anything 
    //except what I tell it to be at link time.)
    client.inifilename[0] = 0;
    if (argv[0] != NULL && ((strlen(argv[0])+5) < sizeof(client.inifilename)))
      {
      strcpy( client.inifilename, argv[0] );
      char *slash = strrchr( client.inifilename, '/' );
      char *slash2 = strrchr( client.inifilename, '\\');
      if (slash2 > slash ) slash = slash2;
      slash2 = strrchr( client.inifilename, ':' );
      if (slash2 > slash ) slash = slash2;
      if ( slash == NULL ) slash = client.inifilename;
      if ( ( slash2 = strrchr( slash, '.' ) ) != NULL ) // ie > slash
        strcpy( slash2, ".ini" );
      else if ( strlen( slash ) > 0 )
        strcat( slash, ".ini" );
      }
    if ( client.inifilename[0] == 0 )
      strcpy( client.inifilename, "rc5des.ini" );
    #elif (CLIENT_OS == OS_VMS)
    strcpy( client.inifilename, "rc5des" EXTN_SEP "ini" );
    #else
    strcpy( client.inifilename, argv[0] );
    strcat( client.inifilename, EXTN_SEP "ini" );
    #endif
    }

  // See if there's a command line parameter to override the INI filename...
  for (i = 1; i < argc; i++)
    {
    if ( strcmp(argv[i], "-ini" ) == 0)
      {
      strcpy( client.inifilename, argv[i+1] );
      argv[i][0] = argv[i+1][0] = 0;
      i++; // Don't try and parse the next argument
      }
    }

#ifndef DONT_USE_PATHWORK
  InitWorkingDirectoryFromSamplePaths( client.inifilename, argv[0] );
#endif

  // read in the ini file
  inimissing = client.ReadConfig();

  // See if there's a command line parameter for the quiet setting...
  for (i = 1; i < argc; i++)
    {
    if ( strcmp(argv[i], "-quiet" ) == 0)
      {
      client.quietmode=1;
      argv[i][0] = 0;
      }
    }

#if (CLIENT_OS == OS_RISCOS)
  for (i = 1; i < argc; i++)
    {
    // See if we are a subtask of the GUI
    if ( strcmp(argv[i], "-guiriscos" ) == 0) 
      {                       
      guiriscos=1;
      argv[i][0] = 0;
      }
    // See if are restarting (hence less banners wanted)
    if ( strcmp(argv[i], "-guirestart" ) == 0) 
      {                  
      guirestart=1;
      argv[i][0] = 0;
      }
    }
#endif

  //'magic number' so ::Run gets executed only when unchanged
  #define OK_TO_RUN (-123)
  int retcode = OK_TO_RUN;

  //let quietmode take effect
  client.InitializeLogging(); 

  // print a banner
  client.PrintBanner(argv[0]);

  // start parsing the command line
  client.ParseCommandlineOptions(argc, argv, &inimissing);

  if (inimissing)
    {
    // prompt the user to do the configuration if there wasn't an ini file
    if (client.Configure() ==1 ) 
      client.WriteConfig();
    retcode = 0;
    }

  #if defined(NONETWORK)
     client.offlinemode=1;
  #endif

  // parse command line "modes" - returns 0 if it didn't do anything
  if (retcode == OK_TO_RUN )
    {
    if ( client.RunCommandlineModes( argc, argv, &retcode ) == 0 ) 
      retcode = OK_TO_RUN;
    }

  if (retcode == OK_TO_RUN)
    {
    if (client.RunStartup())
      retcode = 0;
    }

  if (retcode == OK_TO_RUN)
    {
    // otherwise we are running

    if (!client.offlinemode)  NetworkInitialize();

    #if (CLIENT_OS == OS_RISCOS)
    if (!guirestart)
    #endif
    LogRaw("\nRC5DES Client v2.%d.%d started.\nUsing %s as distributed.net ID.\n\n",
             CLIENT_CONTEST*100+CLIENT_BUILD,CLIENT_BUILD_FRAC,client.id);

    client.Run();

    if (!client.offlinemode)   NetworkDeinitialize();
    if (client.randomchanged)  client.WriteContestandPrefixConfig();

    retcode = ( UserBreakTriggered ? -1 : 0 );
    } //if (retcode == OK_TO_RUN)

  client.DeinitializeLogging(); //flush and close logs/mail

  #if (CLIENT_OS == OS_NETWARE)
  nwCliExitClient(); // destroys AES process, screen, polling procedure
  #endif
  #if (CLIENT_OS == OS_AMIGAOS)
  if (retcode) retcode = 5; // 5 = Warning
  #endif // (CLIENT_OS == OS_AMIGAOS)

  return retcode;
}
#endif

// ---------------------------------------------------------------------------

//parse command line "modes" - returns 0 if did nothing
int Client::RunCommandlineModes( int argc, char *argv[], int *retcodeP )
{
  int i, retcode = -12345;  // 'magic number' 

  for (i = 1; ((retcode == -12345) && (i < argc)); i++)
    {
    if ( argv[i][0] == 0 ) 
      continue;
    else if (( strcmp( argv[i], "-fetch" ) == 0 ) || 
             ( strcmp( argv[i], "-forcefetch" ) == 0 ) || 
             ( strcmp( argv[i], "-flush"      ) == 0 ) || 
             ( strcmp( argv[i], "-forceflush" ) == 0 ) || 
             ( strcmp( argv[i], "-update"     ) == 0 ))
      {
      int dofetch = 0, doflush = 0, doforce = 0;
      if ( strcmp( argv[i], "-fetch" ) == 0 )           dofetch=1;
      else if ( strcmp( argv[i], "-flush" ) == 0 )      doflush=1;
      else if ( strcmp( argv[i], "-forcefetch" ) == 0 ) dofetch=doforce=1;
      else if ( strcmp( argv[i], "-forceflush" ) == 0 ) doflush=doforce=1;
      else /* ( strcmp( argv[i], "-update" ) == 0) */   dofetch=doflush=1;
    
      offlinemode=0;
      NetworkInitialize();

      if ( dofetch )
        {
        int retcode2;
        if ( doforce )
          {
          retcode2 = ForceFetch(0); // RC5 Fetch
          retcode = ForceFetch(1); // DES Fetch
          }
        else
          {
          retcode2 = Fetch(0); // RC5 Fetch
          retcode = Fetch(1); // DES Fetch
          }
        if (contestdone[0]) 
          retcode2 = 0;
        if (randomchanged) 
          WriteContestandPrefixConfig();
        if (contestdone[1]) 
          retcode=1;
        if (retcode < 0 || retcode2 < 0)
          {
          if (retcode2 < retcode) 
            retcode = retcode2;
          dofetch = retcode;
          }
        }
      if ( doflush )
        {
        int retcode2;
        if (doforce)
          {
          retcode2 = ForceFlush(0); // RC5 Flush
          retcode = ForceFlush(1); // DES Flush
          }
        else
          {
          retcode2 = Flush(0); // RC5 Flush
          retcode = Flush(1); // DES Flush
          }
        if (contestdone[0]) 
          retcode2 = 0;
        if (contestdone[1]) 
          retcode = 0;
        if (retcode < 0 || retcode2 < 0)
          {
          if (retcode2 < retcode) 
            retcode = retcode2;
          doflush = retcode;
          }
        }

      LogFlush(1); //checktosend(1)
      NetworkDeinitialize();

      retcode = 0;
      if (dofetch >= 0 && doflush >= 0) //no errors
        LogScreen( "%s completed.\n", argv[i]+1 );
      else
        {
        #if defined(NONETWORK)
        retcode = -1; //always error
        #elif (CLIENT_OS == OS_NETWARE)
        if (!nwCliIsNetworkAvailable(0)) retcode =-1; //detect TCP services 
        #endif
        
        if (retcode) //was handled above
          {
          LogScreen( "TCP/IP services are not available. Without TCP/IP the "
          "client cannot\nsupport the -flush, -forceflush, -fetch, "
          "-forcefetch or -update options.\n");
          }
        else
          {
          LogScreen( "\nAn error occured trying to %s. "
                     "Please try again later\n", argv[i]+1 );
          retcode = dofetch;
          if (doflush < dofetch)
            retcode = doflush;
          }
        }
      }  
    else if ( strcmp(argv[i], "-ident" ) == 0)
      {
      CliIdentifyModules();
      retcode = 0;
      }
    else if ( strcmp( argv[i], "-test" ) == 0 )
      {
      if ( SelfTest(1) > 0 && SelfTest(2) > 0 ) //both OK
        retcode = 0;
      else if ( UserBreakTriggered )
        retcode = -1;
      else  //one of them failed
        retcode = 1; 
      }
    else if (strcmp( argv[i], "-benchmark" ) == 0 )
      {
      int dobench = '1'; 
      if ( isatty(fileno(stdout)) )
        {
        LogScreen( "Block type combinations to benchmark:\n" 
                   "\n1. Both a short RC5 block and a short DES block."
                   "\n2. Only a short RC5 block."
                   "\n3. Only a short DES block."
                   "\n4. Both a long RC5 block and a long DES block."
                   "\n5. Only a long RC5 block."
                   "\n6. Only a long DES block."
                   "\n\nSelect block type(s) to benchmark or "
                   "press any other key to quit: " );
        do{
          fflush(stdin);
          #if (CLIENT_OS == OS_DOS || CLIENT_OS == OS_NETWARE || \
               CLIENT_OS == OS_WIN32 || CLIENT_OS == OS_WIN16 || \
               CLIENT_OS == OS_WIN16S || CLIENT_OS == OS_OS2)
          dobench = getch();
          if (!dobench) getch(); //purge the extended keystroke
          #else
          //could do a select() here in lieu of kbhit()
          dobench = getchar();
          #endif
          if (isprint(dobench) || dobench == 0x1B) //ESC
            break;
          usleep(100000); //in case getchar()/getch() is non-blocking
          #if (CLIENT_OS == OS_AMIGAOS)
          if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C )
            SignalTriggered = UserBreakTriggered = 1;
          #endif
          } while (!SignalTriggered);
        LogScreenRaw("%c\n", ((isprint(dobench))?(dobench):('\n')) );
        } 
      if ( !SignalTriggered && ( dobench == '1' || dobench == '2' ))
        Benchmark(1, 1000000L);
      if ( !SignalTriggered && ( dobench == '1' || dobench == '3' ))
        Benchmark(2, 1000000L);
      if ( !SignalTriggered && ( dobench == '4' || dobench == '5' ))
        Benchmark(1, 0 );
      if ( !SignalTriggered && ( dobench == '4' || dobench == '6' ))
        Benchmark(2, 0 );
      retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if( strcmp( argv[i], "-benchmark2" ) == 0 )
      {
	  Benchmark(1, 1000000L);
	  if (!SignalTriggered)
	  {
	      Benchmark(2, 1000000L);
	  }
	  retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if( strcmp( argv[i], "-benchmark2rc5" ) == 0 )
      {
	  Benchmark(1, 1000000L);
	  retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if( strcmp( argv[i], "-benchmark2des" ) == 0 )
      {
	  Benchmark(2, 1000000L);
	  retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if( strcmp( argv[i], "-benchmarkrc5" ) == 0 )
      {
	  Benchmark(1, 0);
	  retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if( strcmp( argv[i], "-benchmarkdes" ) == 0 )
      {
	  Benchmark(2, 0);
	  retcode = ( UserBreakTriggered ? -1 : 0 ); //and break out of loop
      }
    else if ( strcmp( argv[i], "-cpuinfo" ) == 0 )
      {
      const char *scpuid, *smaxcpus, *sfoundcpus;  //cpucheck.cpp
      GetProcessorInformationStrings( &scpuid, &smaxcpus, &sfoundcpus );
      LogScreen("Automatic processor detection tag:\n\t%s\n"
      "Number of processors detected by this client:\n\t%s\n"
      "Number of processors supported by each instance of this client:\n\t%s\n",
      scpuid, sfoundcpus, smaxcpus );
      retcode = 0; //and break out of loop
      }
    else if ( strcmp( argv[i], "-config" ) == 0 )
      {
      if (Configure()==1) 
        WriteConfig();
      //only write config if 1 is returned
      retcode = 0; //and break out of loop
      }
    else if ( strcmp( argv[i], "-install" ) == 0 )
      {
      #if ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2))
      Install();
      retcode = 0; //and break out of loop
      #endif
      }
    else if ( strcmp( argv[i], "-uninstall" ) == 0 )
      {
      #if ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2))
      Uninstall();
      retcode = 0; //and break out of loop
      #endif
      }
    else if ( strcmp( argv[i], "-forceunlock" ) == 0 )
      {
      retcode = UnlockBuffer(argv[i+1]);
      }
    else
      {
      DisplayHelp(argv[i]);
      retcode = 0;
      }
    }
  if ( retcode == -12345 ) //no change?
    return 0;
  *retcodeP = retcode;
  return 1;
}  

// --------------------------------------------------------------------------

