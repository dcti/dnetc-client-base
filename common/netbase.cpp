/*
 * TCP/IP networking API abstraction layer, with automatic stack and
 * dialup-device initialization and shutdown.
 * Written October 2000 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This module is entirely self-contained - No net API specific
 * structures, functions or defines need to be known outside this
 * module.
 *
 * An overview of the public functions with short descriptions for
 * each is in the header file.
 *
 * Implementation notes:
 *
 * - error codes and error recovery:
 *
 *   All (public-) functions herein that return 'int' return zero on success,
 *   or an code on failure (which can then be described using net_strerror()).
 *
 *   The functions are designed such that retrying is uneccesary, that is,
 *   a non-zero return value can always be considered a permanent/fatal/
 *   unrecoverable error.
 *
 * - blocking vs non-blocking I/O:
 *
 *   From the caller's perspective, all 'socket' i/o functions herein
 *   are non-blocking and the caller has complete control over I/O timeout.
 *   A timeout argument value of zero reduces net_xxx() calls to a poll.
 *   Positive timeout values are interpreted as the maximum number of
 *   milliseconds to wait for the function to complete.
 *   A timeout argument value that is negative generally implies an
 *   indefinite wait, but may be converted internally to some 'sensible'
 *   value if appropriate.
 *
 *   The reason why the functions here have to appear to be non-blocking
 *   is because a) signals may need to be checked and not all net api
 *   implementations return EINTR, b) higher level functions that process
 *   http or uue streams don't always know exactly how much data to read().
 *
 *   Since blocking endpoints are always more efficient, and since
 *   non-blocking I/O does not require that the socket itself be non-
 *   blocking, this module was designed to use blocking sockets as long as
 *   the following condition is satisfied: net_read() with BSD sockets
 *   can only appear to be non-blocking if it can detect how much data may
 *   be atomically recv()d without blocking, ie using ioctl(FIONREAD).
 *   Thus, the presence of FIONREAD (ie, it is defined) determines whether
 *   recv() operation can be executed on a blocking endpoint or not. If your
 *   plaform has FIONREAD but does not support it completely/properly, then
 *   undefine it! The net_xxx() functions will then (have to) create/use
 *   non-blocking endpoints.
 *
 * - automatic stack and dialup initialization/shutdown:
 *
 *   net_open() will initialize the stack/dialup for the first endpoint
 *   opened and net_close() will deinitialize stack/dialup when the last
 *   endpoint is closed. Consequently, an endpoint must be open before
 *   other net_xxx() functions can be called. This was a design decision,
 *   but can easily be changed if warranted.
 *
*/
const char *netbase_cpp(void) {
return "@(#)$Id: netbase.cpp,v 1.1.2.14 2001/01/13 22:40:50 andreasb Exp $"; }

#define TRACE /* expect trace to _really_ slow I/O down */
#define TRACE_STACKIDC(x) //TRACE_OUT(x) /* stack init/shutdown/check calls */
#define TRACE_ERRMGMT(x)  //TRACE_OUT(x) /* error string/number calls */
#define TRACE_POLL(x)     //TRACE_OUT(x) /* net_poll1() */
#define TRACE_CONNECT(x)  //TRACE_OUT(x) /* net_connect() */
#define TRACE_ACCEPT(x)   //TRACE_OUT(x) /* net_accept() */
#define TRACE_FIONBIO(x)  //TRACE_OUT(x) /* (non-)blocking state change*/
#define TRACE_OPEN(x)     //TRACE_OUT(x) /* net_open() */
#define TRACE_CLOSE(x)    //TRACE_OUT(x) /* net_close() */
#define TRACE_READ(x)     //TRACE_OUT(x) /* net_read() */
#define TRACE_WRITE(x)    //TRACE_OUT(x) /* net_write() */
#define TRACE_NETDB(x)    //TRACE_OUT(x) /* net_resolve() */

#include "cputypes.h"
#if ((CLIENT_OS == OS_AMIGAOS)|| (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <errno.h>
#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  #define WIN32_LEAN_AND_MEAN /* don't want winsock.h included here */
  #ifndef STRICT
    #define STRICT
  #endif
  #include <windows.h>
  #include "w32sock.h" //winsock wrappers
  #include "w32util.h" //winGetVersion()
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <socklib.h>
  #include <inetlib.h>
  #include <unixlib.h>
  #include <sys/ioctl.h>
  //#include <unistd.h>
  #include <netdb.h>
  #undef FIONREAD
  }
#elif (CLIENT_OS == OS_DOS)
  //ntohl()/htonl() defines are in...
  #include "platforms/dos/clidos.h"
#elif (CLIENT_OS == OS_VMS)
  #include <signal.h>
  #ifdef __VMS_UCX__ // define for UCX instead of Multinet on VMS
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <netdb.h>
    #include <unixio.h>
  #elif defined(MULTINET)
    #include "multinet_root:[multinet.include.sys]types.h"
    #include "multinet_root:[multinet.include.sys]ioctl.h"
    #include "multinet_root:[multinet.include.sys]param.h"
    #include "multinet_root:[multinet.include.sys]time.h"
    #include "multinet_root:[multinet.include.sys]socket.h"
    #include "multinet_root:[multinet.include]netdb.h"
    #include "multinet_root:[multinet.include.netinet]in.h"
    #include "multinet_root:[multinet.include.netinet]in_systm.h"
    #ifndef multinet_inet_addr
      extern "C" unsigned long int inet_addr(const char *cp);
    #endif
    #ifndef multinet_inet_ntoa
      extern "C" char *inet_ntoa(struct in_addr in);
    #endif
  #endif
#elif (CLIENT_OS == OS_OS2)
  #define BSD_SELECT
  #include <sys/types.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/select.h>
  #include <sys/ioctl.h>
  #if defined(__EMX__)
    // this has to stay as long as the define below is needed
    #include <io.h>
  #endif
#elif (CLIENT_OS == OS_AMIGAOS)
  extern "C" {
  #include "platforms/amiga/amiga.h"
  #include <assert.h>
  #define _KERNEL
  #include <sys/socket.h>
  #undef _KERNEL
  #include <proto/socket.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #define inet_ntoa(addr) Inet_NtoA(addr.s_addr)
  }
#elif (CLIENT_OS == OS_BEOS)
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
#else
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
  #if (CLIENT_OS == OS_LINUX) && (CLIENT_CPU == CPU_ALPHA)
    #include <asm/byteorder.h>
  #elif (CLIENT_OS == OS_QNX)
    #include <sys/select.h>
  #elif (CLIENT_OS == OS_DYNIX) && defined(NTOHL)
    #define ntohl(x)  NTOHL(x)
    #define htonl(x)  HTONL(x)
    #define ntohs(x)  NTOHS(x)
    #define htons(x)  HTONS(x)
  #elif ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K))
    #if defined(_SUNOS3_)
      #define _SOCKET_H_ALREADY_
      extern "C" int fcntl(int, int, int);
    #endif
    extern "C" {
    int socket(int, int, int);
    int setsockopt(int, int, int, char *, int);
    int connect(int, struct sockaddr *, int);
    }
  #elif (CLIENT_OS == OS_AIX)
    #include <sys/select.h>
    #include <strings.h>
  #elif (CLIENT_OS == OS_ULTRIX)
    extern "C" {
      int socket(int, int, int);
      int setsockopt(int, int, int, char *, int);
      int connect(int, struct sockaddr *, int);
    }
  #elif (CLIENT_OS == OS_NETWARE)
    extern "C" {
    #pragma pack(1)
    #include <tiuser.h> //using TLI
    #include <poll.h>
    #define HAVE_POLL_SYSCALL
    #pragma pack()
    }
  #endif
#endif

#if (defined(__GLIBC__) && (__GLIBC__ >= 2)) \
    || (CLIENT_OS == OS_MACOS) \
    || (CLIENT_OS == OS_NETBSD) \
    || ((CLIENT_OS == OS_FREEBSD) && (__FreeBSD__ >= 4))
  /* nothing - socklen_t already defined */
#elif ((CLIENT_OS == OS_BSDOS) && (_BSDI_VERSION > 199701))
  #define socklen_t size_t
#elif ((CLIENT_OS == OS_NTO2) || (CLIENT_OS == OS_QNX))
  #define socklen_t size_t
#elif (CLIENT_OS == OS_AMIGAOS)
  #define socklen_t long
#else
  #define socklen_t int
#endif

/* ======================================================================== */

#include "cputypes.h"
#include "triggers.h" // CheckExitRequestTriggerNoIO()
#include "util.h"     // trace
#include "lurk.h"     // #ifdef LURK
#include "netbase.h"  // ourselves

#if !defined(__NETBASE_H__)  /* defines in netbase.h */
#ifndef INVALID_SOCKET
  typedef int SOCKET;
  #define INVALID_SOCKET ((SOCKET)-1)
#endif

#define ps_stdneterr    -1 /* look at errno/WSAGetLastError()/sock_errno() etc */
#define ps_stdsyserr    -2 /* look at errno */
#define ps_bsdsockerr   -3 /* look at getsockopt(fd,SOL_SOCKET,SO_ERROR,...) */
#define ps_oereserved   -4 /* special to net_open() - err has been cached */
#define ps_EBADF        -5
#define ps_ENETDOWN     -6
#define ps_EINVAL       -7
#define ps_EINTR        -8
#define ps_ETIMEDOUT    -9
#define ps_EDISCO      -10
#define ps_ENOSYS      -11 /* function not implemented */
#define ps_ENODATA     -12 /* Valid name, no data record of requested type */
#define ps_ENOENT      -13 /* no entry for requested name */
#define ps_EINPROGRESS -14
#define ps_ELASTERR ps_EINPROGRESS

#endif

/* ======================================================================== */
/* STACK INITIALIZATION/CONTROL                                             */
/* ======================================================================== */

/* one shot init/deinit. Must be called to init once before any network
 * I/O (anywhere) can happen, and once to deinit before application
 * shutdown. Multiple init/deinit calls are ok as long as they are in
 * init/deinit pairs and the very last deinit has 'final_call' set.
*/
static int __global_init_deinit_check(int doWhat, int final_call)
{
  static int init_mark = -1; /* never before */
  int first_call = 0, rc = 0; /* assume ok */

  if (init_mark == -2) /* never again */
    return -1;
  if (init_mark == -1) /* never before */
  {
    if (doWhat <= 0) /* not initialize call */
      return -1;
    first_call = 1;
    init_mark = 0;    /* not initialized */
  }

  TRACE_STACKIDC((+1,"__global_init_deinit_check(doWhat=%d,first_call=%d,last_call=%d)\n", doWhat,first_call,final_call));
  if (doWhat > 0)                            //initialize
  {
    if (init_mark == 0)
    {
      #ifdef SOCKS
      if (first_call) /* only done once */
      {
        LIBPREFIX(init)("rc5-client");
      }
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (first_call) /* only done once */
      {
        /* this must be done globally for win32 for two reasons:
           a) dialup-detection depends on it
           b) Winsock2 WSACleanup() bug: (affects any application using
              winsock2, not just this one): If WSAStartup() is called
              with any version less than 2, the first SendMessage()
              or Sleep() and perhaps other user32.dll/kernel32.dll 'wait'
              call made after WSACleanup() will (silently) crash ws2_32.dll
              and take the application with it. To be safe, simply ensure
              WSACleanup() is called only on termination.
        */
        if (winGetVersion() >= 400) /* win9x and winnt only */
        {         /* win16 and win32s do it on a per-conn basis */
          WSADATA wsaData;
          if ( WSAStartup( 0x0101, &wsaData ) != 0 )
            rc = -1;
        }
      }
      #elif ((CLIENT_OS == OS_OS2) && !defined(__EMX__))
      if (first_call) /* only done once */
      {
        sock_init();
      }
      #endif
      if (rc == 0)
        init_mark = 1;
      else if (first_call)
        init_mark = -2;
    }
  }
  else if (doWhat < 0)                      /* deinitialize */
  {
    if (init_mark == 1)
    {
      #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (final_call)
      {
        if (winGetVersion() >= 400) /* win9x and winnt only */
        {         /* win16 and win32s do it on a per-conn basis */
          if (WSACleanup() !=0)
            rc = -1;
        }
      }
      #endif
      if (rc == 0)
        init_mark = 0;
      if (final_call)
        init_mark = -2; /* don't allow another init/deinit */
    }
  }
  else /* if (doWhat == 0) */ /* query state */
  {
    if (init_mark != 1)
      rc = -1;
  }
  TRACE_STACKIDC((-1,"__global_init_deinit_check() => %d\n", rc));
  if (rc != 0)
    rc = ps_ENETDOWN;
  return rc;
}

/* --------------------------------------------------------------------- */

int net_initialize(void)
{
  return __global_init_deinit_check(+1,0);
}

int net_deinitialize(int final_call)
{
  return __global_init_deinit_check(-1, final_call);
}

/* --------------------------------------------------------------------- */

static int __dialupsupport_action(int doWhat)
{
  int rc = 0;
  #if defined(LURK)
  {
    //'redial_if_needed' is used here as follows:
    //   If a connection had been previously initiated with DialIfNeeded()
    //   AND there has been no HangupIfNeeded() since then AND the connection
    //   has dropped, THEN kickoff a new DialIfNeeded().
    // Should this behaviour be integrated in lurk.cpp?
    static int redial_if_needed = 0;
    // LurkIsWatching() returns zero if 'dialup' isn't initialized.
    // Otherwise it returns a bitmask of things it is configured to do,
    // ie CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DOD
    int confbits = LurkIsWatching();
    if (confbits) /* 'dialup' initialized and have LURK[ONLY] and/or DOD */
    {
      TRACE_STACKIDC((+1,"__dialupsupport_action(%d)=>%d\n",doWhat));
      if (doWhat < 0) /* request to de-initialize? */
      {
        // HangupIfNeeded will hang up a connection if previously
        // initiated with DialIfNeeded(). Otherwise it does nothing.
        LurkHangupIfNeeded();
        redial_if_needed = 0;
      }
      // IsConnected() returns non-zero when 'dialup' is initialized and
      // a link is up. Otherwise it returns zero.
      else if (!LurkIsConnected()) /* not online/no longer online? */
      {
        rc = ps_ENETDOWN; /* conn dropped and assume not (re)startable */
        if (doWhat > 0 || redial_if_needed) /* request to initialize? */
        {
          if ((confbits & CONNECT_DOD)!=0) /* configured for dial-on-demand?*/
	  {
            // DialIfNeeded(1) returns zero if already connected OR
            // not-configured-for-dod OR dial success. Otherwise it returns -1
            // (either 'dialup' isn't initialized or dialing failed).
            // Passing '1' makes it ignore any lurkonly restriction.
      	    if (LurkDialIfNeeded(1) == 0) /* reconnect to complete */
            {                                /* whatever we were doing */
   	      rc = 0; /* (re-)dial was successful */
              redial_if_needed = 1;
            }
          }
        } /* request to initialize? */
      } /* !LurkIsConnected() */
      TRACE_STACKIDC((-1,"__dialupsupport_action()=>%d\n",rc));
    } /* if LurkIsWatching() */
  } /* if defined(LURK) */
  #endif /* LURK */
  doWhat = doWhat; /* possible unused */
  return rc;
}

/* --------------------------------------------------------------------- */

/*
  net_init_check_deinit( ... ) combines both init and deinit so statics can
  be localized. The function is called with (> 0) to init, (< 0) to deinint
  and (== 0) to return the current 'isOK' state.

  'only_test_api_avail' is non-zero, then the function will only check if the
  network api is present. (ie, it does not check connectivity)
  This option is meaningless for init and deinit requests.
*/
static int net_init_check_deinit( int doWhat, int only_test_api_avail )
{
  static int init_level = 0;
  int rc = 0; /* assume success */

  TRACE_STACKIDC((+1,"net_init_check_deinit(doWhat=%d,api_only=%d)\n", doWhat,only_test_api_avail));

  /* ----------------------- */

  if (rc == 0 && doWhat < 0)       //request to deinitialize
  {
    if (init_level == 0) /* ACK! */
    {
      printf("Beep! Beep! Unbalanced Network Init/Deinit!\n");
      rc = ps_ENETDOWN;
    }
    else if ((--init_level)==0)  //don't deinitialize more than once
    {
      __dialupsupport_action(doWhat);
      #if (CLIENT_OS == OS_AMIGAOS)
      amigaNetworkingDeinit();
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (winGetVersion() < 400) /* win16 and win32s only */
        WSACleanup();            /* winnt and win9x do it as a one-shot */
      #endif
    }
  }

  /* ----------------------- */

  if (rc == 0 && doWhat > 0)  //request to initialize
  {
    rc = ps_ENOSYS;
    if (__global_init_deinit_check(0,0) == 0) /* if global init was ok */
    {
      int plat_init_done = 0;
      rc = 0;
      if ((++init_level)==1) //don't initialize more than once
      {
        #if (!defined(_TIUSER_) && !defined(SOCK_STREAM))
          rc = ps_ENOSYS;  /* no networking capabilities */
        #elif (CLIENT_OS == OS_AMIGAOS)
          int openalllibs = 1;
          #if defined(LURK)
          openalllibs = !LurkIsWatching(); /*some libs unneeded if lurking*/
          #endif
          if (!amigaNetworkingInit(openalllibs))
            rc = ps_ENOSYS;
        #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
        if (winGetVersion() < 400) /* win16 and win32s only */
        {                          /* win9x and winnt do it as a one-shot */
          WSADATA wsaData;
          if ( WSAStartup( 0x0101, &wsaData ) != 0 )
            rc = ps_ENOSYS;
        }
        #endif
        if (rc == 0)
        {
          plat_init_done = 1;
          rc = __dialupsupport_action(+1);
        }
      }
      if (rc == 0)
        rc = net_init_check_deinit(0,0); /* check */
      if (rc != 0)
      {
        if (plat_init_done)
          net_init_check_deinit(-1,0); /*de-init (and decrement init_level)*/
        else
          init_level--;
      }
    } /* global initialization suceeded */
  }

  /* ----------------------- */

  if ( rc == 0 && doWhat == 0 )     //request to check online mode
  {
    if (init_level == 0) /* ACK! haven't been initialized yet */
      rc = ps_ENOSYS;
    else
    {
      #if (!defined(_TIUSER_) && !defined(SOCK_STREAM))
        rc = ps_ENOSYS;  //no networking capabilities
      #elif(CLIENT_OS == OS_AMIGAOS)
      if (!amigaIsNetworkingActive())  // tcpip still available, if not lurking?
        rc = ps_ENETDOWN;
      #elif (CLIENT_OS == OS_NETWARE)
      if (!FindNLMHandle("TCPIP.NLM")) /* tcpip is still loaded? */
        rc = ps_ENOSYS;
      #endif
    }
    if (rc == 0 && !only_test_api_avail)
      rc = __dialupsupport_action(doWhat);
  } /* if ( rc == 0 && doWhat == 0 ) */

  /* ----------------------- */

  TRACE_STACKIDC((-1,"net_init_check_deinit() => %d\n", rc));
  return rc;
}

#define is_netapi_callable() (net_init_check_deinit(0,1) == 0)

/* ======================================================================== */
/* MISC helpers                                                             */
/* ======================================================================== */

#if defined(TRACE)
/* forward reference */
static const char *internal_net_strerror(const char *, int , SOCKET );
#define trace_expand_ps_rc(__rc, __fd) \
          ((__rc != 0)?(internal_net_strerror(" ", __rc, __fd )):(""))
#define trace_expand_api_rc(__rc, __fd) \
          ((__rc < 0)?(internal_net_strerror(" ", ps_stdneterr, __fd )):(""))
#endif /* if defined(TRACE) */

static void __calc_timeout_metrics(int iotimeout, /* millisecs */
                                   int *maxloops, /* max number of loops */
                                   int *mssleep ) /* sleep time per loop */
{
  *maxloops = *mssleep = 0;
  if (iotimeout != 0)
  {
    if (iotimeout < 0) /* "blocking" */
      iotimeout = 90*1000;  /* == 90 second timeout */
    *maxloops = (iotimeout+249)/250; /* quarter sec loops */
    *mssleep = 250; /* 250 millisecs */
  }
  return;
}


/* ======================================================================== */
/* ERROR CODE MANAGEMENT                                                    */
/* ======================================================================== */

/* errcode retrieval may not be possible after a failed net_open(), so
   on net_open failure, the values are cached here, and net_open() returns
   a 'magic value' which causes __read_errnos() to pick up the error number
   from here rather than from the system.
*/
static struct
{
  int ps_errnum;
  int syserr;
  int neterr;
  int extra; /* _TIUSER_ needs this to TLOOK, TOUTSTATE etc  */
} ps_oereserved_cache = {0,0,0,0};

/* translate lower case ps_xxx error numbers into error numbers from
   the system (or from ps_oereserved_cache if the ps_errnum is ps_oereserved)
*/
static int ___read_errnos(SOCKET fd, int ps_errnum,
                          int *syserr, int *neterr, int *extra )
{
  *syserr = *neterr = *extra = 0;
  fd = fd; /* possibly unused */

  /* don't make this a switch() or an if ... else ... */

  if (ps_errnum == ps_oereserved) /* error codes were cached */
  {
    ps_errnum = ps_oereserved_cache.ps_errnum;
    *syserr   = ps_oereserved_cache.syserr;
    *neterr   = ps_oereserved_cache.neterr;
    *extra    = ps_oereserved_cache.extra;
    return ps_errnum;
  }

  if (ps_errnum == ps_bsdsockerr) /* only happens for BSD sox */
  {
    #if defined(_TIUSER_)
    ps_errnum = ps_stdneterr;
    /* *** fallthrough *** */
    #elif defined(SOL_SOCKET) && defined(SO_ERROR)
    if (is_netapi_callable())
    {
      int rc, so_err = 0;
      socklen_t szint = (socklen_t) sizeof(so_err);
      TRACE_ERRMGMT((+1,"getsockopt(s,SOL_SOCKET,SO_ERROR, &so_err, &%d\n",(int)szint));
      rc = getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *)&so_err, &szint);
      TRACE_ERRMGMT((-1,"getsockopt(...)=>%d%s [so_err=%d]\n", rc, trace_expand_api_rc(rc,fd), so_err ));
      if (rc == 0)
      {
        *neterr = so_err;
        return ps_stdneterr; /* we have the error number */
      }
    }
    #endif
    if (ps_errnum == ps_bsdsockerr) /* didn't change (retrieval failed) */
      return ps_errnum; /* "cannot retrieve socket error number" */
  }

  if (ps_errnum == ps_stdneterr)
  {
    if (is_netapi_callable())
    {
      #if defined(_TIUSER_)
      *neterr = t_errno;
      if (*neterr == TLOOK)
      {
        *extra = t_look(fd);
        if (*extra == T_DISCONNECT)
        {
          struct t_discon tdiscon;
          tdiscon.udata.buf = (char *)0;
          tdiscon.udata.maxlen = 0;
          tdiscon.udata.len = 0;
          if (t_rcvdis(fd, &tdiscon) == 0) /* otherwise TNODIS etc */
          {
            *syserr = tdiscon.reason;
            *neterr = *extra = 0;
            ps_errnum = ps_stdsyserr;
          }
        }
      }
      else if (*neterr == TOUTSTATE)
        *extra = t_getstate(fd);
      else if (*neterr == TSYSERR)
      {
        *syserr = errno;
        *neterr = 0;
        ps_errnum = ps_stdsyserr;
      }
      #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
      *neterr = WSAGetLastError();
      TRACE_ERRMGMT((0,"WSAGetLastError() => %d\n", *neterr ));
      #elif (CLIENT_OS == OS_AMIGAOS)
      *syserr = Errno();
      TRACE_ERRMGMT((0,"Errno() => %d\n", *syserr ));
      ps_errnum = ps_stdsyserr;
      #elif (CLIENT_OS == OS_OS2) && !defined(__EMX__)
      *syserr = sock_errno();
      TRACE_ERRMGMT((0,"sock_errno() => %d\n", *syserr ));
      ps_errnum = ps_stdsyserr;
      #else
      *syserr = errno;
      TRACE_ERRMGMT((0,"(net) errno => %d\n", *syserr ));
      ps_errnum = ps_stdsyserr;
      #endif
      return ps_errnum;
    }
    return ps_ELASTERR - 1; /* translation error */
  }

  if (ps_errnum == ps_stdneterr)
  {
    *syserr = errno;
    TRACE_ERRMGMT((0, "(sys) errno => %d\n", *syserr ));
    return ps_errnum;
  }

  return ps_errnum;
}

/* --------------------------------------------------------------------- */

static const char *internal_net_strerror(const char *ctx, int ps_errnum, SOCKET fd)
{
  static char msgbuf[256];
  char scratch[sizeof(msgbuf)];
  int syserr, neterr, extraerr, got_message = 0;
  const char *msg;
  unsigned int msgbuflen;

  TRACE_ERRMGMT((+1,"net_strerror(ctx=%p, ps_errnum=%d, fd=%d)\n", ctx, ps_errnum, fd));
  ps_errnum = ___read_errnos(fd, ps_errnum, &syserr, &neterr, &extraerr );

  msgbuflen = 0;
  if (ctx)
  {
    strncpy(msgbuf, ctx, sizeof(msgbuf) );
    msgbuf[sizeof(msgbuf)-1] = '\0';
    msgbuflen = strlen(msgbuf);
    if (msgbuflen > 0 && msgbuflen < (sizeof(msgbuf)-2))
      msgbuf[msgbuflen++] = ' ';
  }
  msgbuf[msgbuflen] = '\0';

  msg = (const char *)0;
  if (ps_errnum < ps_ELASTERR) /* translation error */
    msg = "(unable to translate error code)";
  else if (ps_errnum == ps_bsdsockerr) /* failed to get socket error */
    msg = "(unable to determine socket error number)";
  else
  {
    switch (ps_errnum)
    {
      case 0: msg = "error 0"; break;
      case ps_EBADF: msg = "EBADF: invalid socket descriptor"; break;
      case ps_ENETDOWN: msg = "ENETDOWN: network down or not available"; break;
      case ps_EINVAL: msg = "EINVAL: invalid argument"; break;
      case ps_EINTR: msg = "EINTR: interrupted system call"; break;
      case ps_ETIMEDOUT: msg = "ETIMEDOUT: operation timed out"; break;
      case ps_EDISCO: msg = "EDISCONNECT: connection lost"; break;
      case ps_ENOSYS: msg = "ENOSYS: unsupported system call"; break;
      case ps_ENODATA: msg = "ENODATA: no data of requested type"; break;
      case ps_ENOENT: msg = "ENOENT: no entry for requested name"; break;
      default: break;
    }
  }
  if (msg)
  {
    strncpy( &msgbuf[msgbuflen], msg, (sizeof(msgbuf)-msgbuflen)-1 );
    msgbuf[sizeof(msgbuf)-1] = '\0';
    got_message = 1;
  }

  /* -------------------------- */

  if (!got_message && ps_errnum == ps_stdneterr)
  {
#if defined(_TIUSER_)
    #ifndef DEBUGTHIS
    #define debugtli(_x,_y) /* nothing */
    #else
    #define debugtli(__ctx, __fd) printf("%s\n", net_strerror(__ctx, ps_stdneterr, __fd))
    #endif
    if (neterr == TLOOK)
    {
      static struct { int num;      const char *name; } look_tab[] = {
                    { T_LISTEN,     "t_listen" },
                    { T_CONNECT,    "t_connect" },
                    { T_DATA,       "t_data" },
                    { T_EXDATA,     "t_exdata" },
                    { T_DISCONNECT, "t_disconnect" },
                    { T_ORDREL,     "t_ordrel" },
                    { T_UDERR,      "t_uderr" },
                    { T_GODATA,     "t_godata" },
                    { T_GOEXDATA,   "t_goexdata" },
                    { T_EVENTS,     "t_events" } };
      unsigned int i;
      msg = (const char *)0;
      for (i = 0; i < (sizeof(look_tab)/sizeof(look_tab[0])); i++)
      {
        if (look_tab[i].num == extraerr)
        {
          msg = look_tab[i].name;
          break;
        }
      }
      sprintf(scratch, "TLOOK: asynchronous event %d%s%s%s",
              extraerr, ((msg)?(" ("):("")), ((msg)?(msg):("")),
              ((msg)?(") "):("")) );
      strncpy( &msgbuf[msgbuflen], scratch, (sizeof(msgbuf)-msgbuflen)-1 );
      msgbuf[sizeof(msgbuf)-1] = '\0';
      got_message = 1;
    }
    else if (neterr == TOUTSTATE)
    {
      static struct { int num;     const char *name; } state_tab[] = {
                    { T_UNBND,     "T_UNBND" },
                    { T_IDLE,      "T_IDLE" },
                    { T_OUTCON,    "T_OUTCON" },
                    { T_INCON,     "T_INCON" },
                    { T_DATAXFER,  "T_DATAXFER" },
                    { T_OUTREL,    "T_OUTREL" } };
      unsigned int i;
      msg = "???";
      for (i = 0; i < (sizeof(state_tab)/sizeof(state_tab[0])); i++)
      {
        if (state_tab[i].num == extraerr)
        {
          msg = state_tab[i].name;
          break;
        }
      }
      sprintf(scratch, "TOUTSTATE: primitive issued in wrong sequence (%d:%s)",
                       extraerr, msg );
      strncpy( &msgbuf[msgbuflen], scratch, (sizeof(msgbuf)-msgbuflen)-1 );
      msgbuf[sizeof(msgbuf)-1] = '\0';
      got_message = 1;
    }
    else /* (neterr!=TSYSERR) TSYSERR will have been translated to ps_stdsyserr */
    {
      int slen = sprintf(scratch, "error %d: ", neterr );
      msg = "(unrecognized cause)";
      if (neterr > 0 && neterr < t_nerr)
        msg = t_errlist[neterr];
      strncpy( &scratch[slen], msg, (sizeof(scratch)-slen)-1 );
      scratch[ sizeof(scratch)-1 ] = '\0';
      strncpy( &msgbuf[msgbuflen], scratch, (sizeof(msgbuf)-msgbuflen)-1 );
      msgbuf[sizeof(msgbuf)-1] = '\0';
      got_message = 1;
    }
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    msg = (const char *)0;
    switch (neterr)
    {
      case WSAEINTR:           msg = "EINTR: Interrupted system call"; break;
      case WSAEBADF:           msg = "EBADF: Bad file number"; break;
      case WSAEACCES:          msg = "EACCES: Permission denied"; break;
      case WSAEFAULT:          msg = "EFAULT: Bad address passed"; break;
      case WSAEINVAL:          msg = "EINVAL: Invalid parameter passed"; break;
      case WSAEMFILE:          msg = "EMFILE: Too many open files"; break;
      case WSAEWOULDBLOCK:     msg = "EWOULDBLOCK: Operation would block"; break;
      case WSAEINPROGRESS:     msg = "EINPROGRESS: Operation is now in progress"; break;
      case WSAEALREADY:        msg = "EALREADY: Operation is already in progress"; break;
      case WSAENOTSOCK:        msg = "ENOTSOCK: Socket operation on non-socket"; break;
      case WSAEDESTADDRREQ:    msg = "EDESTADDRREQ: Destination address required"; break;
      case WSAEMSGSIZE:        msg = "EMSGSIZE: Message is too long"; break;
      case WSAEPROTOTYPE:      msg = "EPROTOTYPE: The protocol is of the wrong type for the socket"; break;
      case WSAENOPROTOOPT:     msg = "ENOPROTOOPT: The requested protocol is not available"; break;
      case WSAEPROTONOSUPPORT: msg = "EPROTONOSUPPORT: The requested protocol is not supported"; break;
      case WSAESOCKTNOSUPPORT: msg = "ESOCKTNOSUPPORT: The specified socket type is not supported"; break;
      case WSAEOPNOTSUPP:      msg = "EOPNOTSUPP: The specified operation is not supported"; break;
      case WSAEPFNOSUPPORT:    msg = "EPFNOSUPPORT: The specified protocol family is not supported"; break;
      case WSAEAFNOSUPPORT:    msg = "EAFNOSUPPORT: The specified address family is not supported"; break;
      case WSAEADDRINUSE:      msg = "EADDRINUSE: The specified address is already in use"; break;
      case WSAEADDRNOTAVAIL:   msg = "EADDRNOTAVAIL: The requested address is unassignable"; break;
      case WSAENETDOWN:        msg = "ENETDOWN: The network appears to be down"; break;
      case WSAENETUNREACH:     msg = "ENETUNREACH: The network is unreachable"; break;
      case WSAENETRESET:       msg = "ENETRESET: The network dropped the connection on reset"; break;
      case WSAECONNABORTED:    msg = "ECONNABORTED: Software caused a connection abort"; break;
      case WSAECONNRESET:      msg = "ECONNRESET: Connection was reset by peer"; break;
      case WSAENOBUFS:         msg = "ENOBUFS: Out of buffer space"; break;
      case WSAEISCONN:         msg = "EISCONN: Socket is already connected"; break;
      case WSAENOTCONN:        msg = "ENOTCONN: Socket is not presently connected"; break;
      case WSAESHUTDOWN:       msg = "ESHUTDOWN: Can't send data because socket is shut down"; break;
      case WSAETOOMANYREFS:    msg = "ETOOMANYREFS: Too many references, unable to splice"; break;
      case WSAETIMEDOUT:       msg = "ETIMEDOUT: The connection timed out"; break;
      case WSAECONNREFUSED:    msg = "ECONNREFUSED: The connection was refused"; break;
      case WSAELOOP:           msg = "ELOOP: Too many symbolic link levels"; break;
      case WSAENAMETOOLONG:    msg = "ENAMETOOLONG: File name is too long"; break;
      case WSAEHOSTDOWN:       msg = "EHOSTDOWN: The host appears to be down"; break;
      case WSAEHOSTUNREACH:    msg = "EHOSTUNREACH: The host is unreachable"; break;
      case WSAENOTEMPTY:       msg = "ENOTEMPTY: The directory is not empty"; break;
      case WSAEPROCLIM:        msg = "EPROCLIM: There are too many processes"; break;
      case WSAEUSERS:          msg = "EUSERS: There are too many users"; break;
      case WSAEDQUOT:          msg = "EDQUOT: The disk quota is exceeded"; break;
      case WSAESTALE:          msg = "ESTALE: Bad NFS file handle"; break;
      case WSAEREMOTE:         msg = "EREMOTE: There are too many levels of remote in the path"; break;
      //se WSAEDISCO:          msg = "EDISCO: Disconnect"; break;
      case WSASYSNOTREADY:     msg = "WSASYSNOTREADY: Network sub-system is not ready or unusable"; break;
      case WSAVERNOTSUPPORTED: msg = "WSAVERNOTSUPPORTED: The requested version is not supported"; break;
      case WSANOTINITIALISED:  msg = "WSANOTINITIALISED: Socket system is not initialized"; break;
      default: break;
    }
    if (!msg)
    {
      sprintf(scratch, "error %d: (no description available)", neterr );
      msg = scratch;
    }
    strncpy( &msgbuf[msgbuflen], msg, (sizeof(msgbuf)-msgbuflen)-1 );
    msgbuf[sizeof(msgbuf)-1] = '\0';
    got_message = 1;
#else /* should never happen (should have been translated by __net_readerrno) */
    syserr = neterr;
    ps_errnum = ps_stdsyserr;
#endif
  } /* if (ps_errnum == ps_stdneterr) */

  /* -------------------------- */

  if (!got_message && ps_errnum == ps_stdsyserr)
  {
    int slen = sprintf( scratch, "error %d: ", syserr );
    strncpy( &scratch[slen], strerror(syserr), (sizeof(scratch)-slen)-1 );
    scratch[sizeof(scratch)-1] = '\0';
    strncpy( &msgbuf[msgbuflen], scratch, (sizeof(msgbuf)-msgbuflen)-1 );
    msgbuf[sizeof(msgbuf)-1] = '\0';
    got_message = 1;
  }

  if (!got_message)
  {
    /* "how did you get here?" */
    sprintf( scratch, "error %d: (no error description available)", ps_errnum );
    strncpy( &msgbuf[msgbuflen], scratch, (sizeof(msgbuf)-msgbuflen)-1 );
    msgbuf[sizeof(msgbuf)-1] = '\0';
  }

  TRACE_ERRMGMT((-1,"net_strerror(...)=>'%s'\n", msgbuf ));
  return msgbuf;
}

/* get a descriptive error message for an error number returned by one
 * of the net_xxx() functions that return 'int'
*/
const char *net_strerror(int ps_errnum, SOCKET fd)
{
  return internal_net_strerror(0, ps_errnum, fd );
}

/* --------------------------------------------------------------------- */

#if !defined(_TIUSER_) && defined(SOCK_STREAM) /* only needed for BSD sox */
/* only the errnos we're actually testing for internally are checked */
static int net_match_errno(register int which_ps_err)
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  if (which_ps_err == ps_EINPROGRESS)
  {
    int err = WSAGetLastError();
    TRACE_CONNECT((0,"err = %d\n", err));
    return (err == WSAEINPROGRESS || err == WSAEWOULDBLOCK);
    /* WSAEINPROGRESS has a completely different meaning from that on BSD */
  }
  if (which_ps_err == ps_EINTR) return (WSAGetLastError() == WSAEINTR);
  #else
  {
    #if (CLIENT_OS == OS_AMIGAOS)
    int err = Errno();
    #elif (CLIENT_OS == OS_OS2) && !defined(__EMX__)
    int err = sock_errno();
    #else
    int err = errno;
    #endif
    #if defined(EINTR)
    if (which_ps_err == ps_EINTR)       return (err == EINTR);
    #endif
    #if defined(EINPROGRESS)
    if (which_ps_err == ps_EINPROGRESS) return (err == EINPROGRESS);
    #endif
    if (which_ps_err == 0) /* dummy */  return (err == 0);
  }
  #endif
  return 0;
}
#endif /* #if defined(_TIUSER_) || defined(SOCK_STREAM) */

/* ======================================================================== */
/* STATE DETECTION                                                          */
/* ======================================================================== */

#define LOOK_NOT_POLL

#if !defined(LOOK_NOT_POLL) && !defined(HAVE_POLL_SYSCALL) && !defined(POLLIN)
#  define POLLIN          01      /* message available on read queue */
#  define POLLPRI         02      /* priority message available */
#  define POLLOUT         04      /* stream is writable */
#  define POLLERR         010     /* error message has arrived */
#  define POLLHUP         020     /* hangup has occurred */
// define POLLNVAL        040     /* invalid descriptor */
   /* our net_poll1 returns EBADF on invalid descriptor */
#endif

#if defined(TRACE) && (defined(_TIUSER_) || !defined(LOOK_NOT_POLL))
const char *__trace_expand_pollmask(int events) /* can't make this static :( */
{
  static char buffer[sizeof("POLLIN|POLLOUT|POLLPRI|POLLERR|POLLHUP  ")];
  if (events == 0)
    strcpy(buffer,"0");
  else
  {
    unsigned int pos, count = 0;
    static struct { int mask; const char *name;} poll_tab[] = {
                  { POLLIN,  "IN"  },
                  { POLLOUT, "OUT" },
                  { POLLPRI, "PRI" },
                  { POLLERR, "ERR" },
                  { POLLHUP, "HUP" } };
    buffer[0] = '\0';
    for (pos = 0; pos < (sizeof(poll_tab)/sizeof(poll_tab[0])); pos++)
    {
      if ((events & poll_tab[pos].mask)!=0)
      {
        if (count == 0)
          strcpy(buffer,"POLL");
        else
          strcat(buffer,"|");
        strcat( buffer, poll_tab[pos].name );
        events ^= poll_tab[pos].mask;
        count++;
      }
    }
    if (events)
    {
      if (count != 0)
        strcat(buffer,"|");
      sprintf(&buffer[strlen(buffer)],"0x%x",events);
    }
  }
  return buffer;
}
#endif /* defined(TRACE) && defined(TRACE_POLL) */

/* --------------------------------------------------------------------- */

#define ps_T_LISTEN     0x001  /* connect indication received */
#define ps_T_CONNECT    0x002  /* connect confirmation received */
#define ps_T_DISCONNECT 0x004  /* disconnect received */
#define ps_T_ORDREL     0x008  /* orderly release indication */
#define ps_T_UDERR      0x010  /* datagram error indication */
#define ps_T_DATA       0x100  /* normal data received */
#define ps_T_EXDATA     0x200  /* expedited data received */
#define ps_T_GODATA     0x400  /* flow control restrictions on normal data flow have been lifted */
#define ps_T_GOEXDATA   0x800  /* flow control restrictions on expedited data flow have been lifted */

#if !defined(_TIUSER_) && defined(SOCK_STREAM) /* BSD sox only */
static int __bsd_quick_disco_look(SOCKET fd, int fdr_is_ready)
{
  int rc = 1;

  TRACE_POLL((+1,"__bsd_quick_disco_look(fd,%d)\n", fdr_is_ready));
  if (!fdr_is_ready)
  {
    struct timeval tv;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd,&rfds);
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    TRACE_POLL((+1,"ql: select()\n"));
    rc = select( fd+1, &rfds, 0, 0, &tv );
    TRACE_POLL((-1,"ql: select(...) =>%d%s\n", rc, trace_expand_api_rc(rc,fd) ));

    if (rc < 0)
      rc = 0; /* we don't deal with select errors here */
  }
  if (rc > 0)
  {
    char ch = 0;
    #if (CLIENT_OS == OS_MACOS) /* broken gusi */
      ch = 1; /* msg_peek removes 1 byte from the queue */
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    if (winGetVersion() < 400) /* broken 16bit winsock.dll */
      ch = 1; /* msg_peek removes 1 byte from the queue */
    #endif
    if (ch == 0)
    {  
      TRACE_POLL((+1,"ql: recv(...,MSG_PEEK)\n"));
      rc = recv(fd, &ch, 1, MSG_PEEK);
      TRACE_POLL((-1,"ql: recv(...,MSG_PEEK) =>%d%s\n",rc,trace_expand_api_rc(rc,fd)));
    }
    if (rc == 0) /* 0 means graceful shutdown */
      rc = ps_T_ORDREL;
    else if (rc < 0) /* means reset */
      rc = ps_T_DISCONNECT;
    else
      rc = 0; /* plain data */
  }
  TRACE_POLL((-1,"__bsd_quick_disco_look(...)=>%s\n", ((rc==ps_T_ORDREL)?("ordrel"):((rc)?("disco"):("(nothing)"))) ));
  return rc;
}
#endif

/* net_look() looks for the events asked for (sleep until they happen or */
/* timeout) and returns _one_ of the events in revents, or rc is non-zero */
/* on error. NB BSD sox: because select() can't check for disconnect and */
/* write-ready simultaneously without also checking for read, a net_look for */
/* ps_T_GODATA requires at least two select() operations so you'd better */
/* have a small timeout when checking for ps_T_GODATA (net_write() does) */

static int net_look( SOCKET fd, int events, int *revents, int mstimeout )
{
  int rc = ps_ENETDOWN, result_events = 0;
  events = events; revents = revents; mstimeout = mstimeout;

  TRACE_POLL((+1,"net_look(s,%d,revents,%d)\n", events, mstimeout));

  if ( fd == INVALID_SOCKET )
  {
    rc = ps_EBADF;
  }
  else if (is_netapi_callable())
  {
    int retry_count = 0;
    int break_pending = CheckExitRequestTriggerNoIO();
    for (;;)
    {
      #if defined(_TIUSER_)
      int look = t_look(fd);
      rc = 0;
      if (look == 0)
      {
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = pfd.revents = 0;
        if ((events & (ps_T_LISTEN|ps_T_DATA))!=0)
          pfd.events |= POLLIN;
        if ((events & (ps_T_CONNECT|ps_T_GODATA))!=0)
          pfd.events |= POLLOUT;
        if ((events & (ps_T_GOEXDATA|ps_T_EXDATA))!=0)
          pfd.events |= POLLPRI;
        TRACE_POLL((+1,"poll( &pfd, 1, %d ) [events=%s]\n", mstimeout, __trace_expand_pollmask(pfd.events) ));
        rc = poll( &pfd, 1, mstimeout );
        if (rc < 0) t_errno = TSYSERR; /* for trace */
        TRACE_POLL((-1,"poll(...)=>%d%s [revents=%s]\n",rc,trace_expand_api_rc(rc,fd),__trace_expand_pollmask(pfd.revents)));
        if (rc == 0)
          break;
        if (rc > 0)
        {
          rc = 0;
          if ((pfd.revents & POLLNVAL)!=0)
          {
            rc = ps_EBADF;
            break;
          }
          look = t_look(fd);
          t_errno = TLOOK; /* for trace */
          TRACE_POLL((0,"t_look() =>%d (%s)\n",look,trace_expand_api_rc(-1,fd)));
          if (look == 0)
          {
            if ((pfd.revents & POLLERR)!=0)
              result_events = ps_T_ORDREL; /* mmm */
            else if ((pfd.revents & POLLHUP)!=0)
              result_events = ps_T_ORDREL;
            else if ((pfd.revents & POLLIN)!=0)
              result_events = ((events & ps_T_LISTEN)?(ps_T_LISTEN):(ps_T_DATA));
            else if ((pfd.revents & POLLOUT)!=0)
              result_events = ((events & ps_T_CONNECT)?(ps_T_CONNECT):(ps_T_GODATA));
            else if ((pfd.revents & POLLPRI)!=0)
              result_events = ps_T_EXDATA;
            break;
          }
          /* fallthrough */
        }
        else /* rc < 0 */
        {
          if (errno != EINTR)
          {
            rc = ps_stdsyserr; /* not stdneterr! */
            break;
          }
        }
      }
      if (rc == 0 && look)
      {
        switch (look) {
          case T_LISTEN:   result_events = ps_T_LISTEN; break;
          case T_CONNECT:  result_events = ps_T_CONNECT; break;
          case T_ORDREL:   result_events = ps_T_ORDREL; break;
          case T_UDERR:    result_events = ps_T_UDERR; break;
          case T_DATA:     result_events = ps_T_DATA; break;
          case T_EXDATA:   result_events = ps_T_EXDATA; break;
          case T_GODATA:   result_events = ps_T_GODATA; break;
          case T_GOEXDATA: result_events = ps_T_GOEXDATA; break;
          default:         rc = ps_stdneterr; break; /* T_DISCONNECT */
        }
        break;
      }
      #elif defined(SOCK_STREAM)
      struct timeval tv;
      struct timeval *tvP;
      fd_set wfds, rfds, xfds;
      fd_set *wfdsP, *rfdsP;

      FD_ZERO(&xfds);
      FD_SET(fd,&xfds);

      rfdsP = (fd_set *)0;
      if ((events & (ps_T_LISTEN|ps_T_DATA))!=0)
      {
        rfdsP = &rfds;
        FD_ZERO(rfdsP);
        FD_SET(fd,rfdsP);
      }
      wfdsP = (fd_set *)0;
      if ((events & (ps_T_CONNECT|ps_T_GODATA))!=0)
      {
        wfdsP = &wfds;
        FD_ZERO(wfdsP);
        FD_SET(fd,wfdsP);
      }
      /* if we're only waiting for "ready to accept data for send" */
      /* then quickly check if the read bit is set. If it is, then */
      /* that could mean two things: a) there is data to be read,  */
      /* or there is a disconnect indication (or error) pending.   */
      if (!rfdsP && (events & ps_T_GODATA)!=0)
      {
        rc = 0;
        result_events = __bsd_quick_disco_look(fd, 0);
        if (result_events != 0) /* ps_T_ORDREL or ps_T_DISCONNECT */
          break;                /* if ps_T_DISCONNECT, then errno is set */
      }
      tvP = (struct timeval *)0;
      if (mstimeout >= 0)
      {
        tv.tv_sec = mstimeout/1000;
        tv.tv_usec = (mstimeout%1000)*1000;
        tvP = &tv;
      }

      /* select() is sooooo damn ooogly! (its saving grace is one bit per fd) */
      TRACE_POLL((+1,"select(n, %p, %p, %p, %d:%d )\n", rfdsP,wfdsP,&xfds,tv.tv_sec,tv.tv_usec));
      rc = select( fd+1, rfdsP, wfdsP, &xfds, tvP );
      TRACE_POLL((-1,"select(...) =>%d%s\n", rc, trace_expand_api_rc(rc,fd) ));

      if (rc == 0) /* timed out */
      {
        break; /* result_events = 0, return 0 */
      }
      else if (rc > 0) /* something set in one of more of the fd_sets */
      {
        int isx = FD_ISSET(fd,&xfds);
        rc = 0;
        if (isx)
        {
         TRACE_POLL((0,"select says exception set\n"));
          #if 0 /* fallthrough to read to get errno primed */
          if ((events & ps_T_CONNECT)!=0)
          {
            result_events = ps_T_DISCONNECT;
            break;
          }
          #endif
          if ((events & ps_T_EXDATA)!=0)
          {
            result_events = ps_T_EXDATA;
            break;
          }
          /* fallthrough */
        }
        if (wfdsP)  /* CONNECT or GODATA */
        {
          if (FD_ISSET(fd,wfdsP))
          {
            TRACE_POLL((0,"select says write ready\n"));
            if ((events & ps_T_CONNECT)!=0)
              result_events = ps_T_CONNECT;
            else
              result_events = ps_T_GODATA;
            break;
          }
          if (!rfdsP)
          {
            rc = 0;
            TRACE_POLL((0,"CONNECT/GODATA. Doing quick read test.\n"));
            result_events = __bsd_quick_disco_look(fd, 0);
            if (result_events != 0) /* ps_T_ORDREL or ps_T_DISCONNECT */
              break;                /* if ps_T_DISCONNECT, then errno is set */
          }
        }
        if (rfdsP)
        {
          if (FD_ISSET(fd, rfdsP))
          {
            TRACE_POLL((0,"select says read ready.\n"));
            rc = 0;
            if ((events & ps_T_LISTEN)!=0)
              result_events = ps_T_LISTEN;
            else
            {
              result_events = __bsd_quick_disco_look(fd, 1);
              if (result_events == 0)
                result_events = ps_T_DATA;
            }
            break;
          }
        } /* rfdsP */
        rc = 0;
        if (isx) /* neither wfds or rfds were set, but xfds is set */
        {
          TRACE_POLL((0,"neither wfds nor rfds are are, but xfds is set. Disconnect assumed.\n"));
          result_events = ps_T_ORDREL;
          break;
        }
      } /* if (rc > 0)  */
      else if (!net_match_errno(ps_EINTR)) /* its not EINTR */
      {
        rc = ps_stdneterr;
        break;
      }
      #endif /* SOCK_STREAM */

      /* we have EINTR if we got here */
      /* we do *not* checkexitrequesttrigger unless
         eintr _and_ there was no exit pending when we started,
         otherwise we'd never be able to send mail on shutdown.
      */
      if (!break_pending && CheckExitRequestTriggerNoIO())
      {
        rc = ps_EINTR;
        break;
      }
      if ((++retry_count) == 5) /* stuck in EINTR loop */
      {
        result_events = ps_T_ORDREL;
        rc = 0;
        break;
      }
      /* otherwise retry */
    } /* for (;;) */
  } /* if (is_netapi_callable()) */

  if (rc == 0)
  {
    if (result_events == ps_T_DISCONNECT)
      rc = ps_stdneterr;
    else if (result_events == ps_T_ORDREL)
      rc = ps_EDISCO;
    else if (revents)
      *revents = result_events;
  }

  TRACE_POLL((-1,"net_look() => %d%s [revents=%d]\n", rc, trace_expand_ps_rc(rc,fd), result_events ));
  return rc;
}

/* --------------------------------------------------------------------- */

#ifndef LOOK_NOT_POLL
/* similar to the poll() syscall, but returns an error code, not just -1/0 */
/* like the poll() syscall, net_poll1() does not modify revents on error   */
/*                                                                         */
/* this implementation of poll() diverges from the standard as follows:    */
/* a) it returns EBADF on invalid fd (rather than set POLLNVAL)            */
/* b) see special remark about select()+hangup checks below                */

static int net_poll1( SOCKET fd, int events, int *revents, int mstimeout )
{
  int rc = ps_ENETDOWN, result_events = 0;
  events = events; revents = revents; mstimeout = mstimeout;

  TRACE_POLL((+1,"net_poll(fd, %s, %p, %d)\n",
                __trace_expand_pollmask(events), revents, mstimeout ));
  if ( fd == INVALID_SOCKET )
  {
    rc = ps_EBADF;
  }
  else if ((rc = net_init_check_deinit(0,0)) == 0) /* check */
  {
#if defined(_TIUSER_) || defined(HAVE_POLL_SYSCALL)
    int retry_count = 0;
    struct pollfd pfd[1];
    pfd[0].fd = fd;
    pfd[0].events = pfd[0].revents = 0;

    /* Documented compatibility issue on BSD and some others:
       in some instances these platforms deviate from the historical
       implementation with respect to return of error:
       Where the historical implementation would return an error, the
       BSD implementations copy events to revents, and leave the
       following I/O operation to error out.
       This is _not_ nice, since a recv() on a broken connection would
       then sigpipe.
       To work around this wierdness, we add POLLERR to the events mask.
    */
    pfd[0].events = POLLERR;
    if ((events & POLLIN)!=0)  pfd[0].events |= POLLIN;
    if ((events & POLLPRI)!=0) pfd[0].events |= POLLPRI;
    if ((events & POLLOUT)!=0) pfd[0].events |= POLLOUT;

    if (mstimeout < 0) /* "wait indefinitely" */
      mstimeout = -1;  /* INFINTIM */

    for (;;)
    {
      TRACE_POLL((+1,"poll(%s, 1, %d)\n", __trace_expand_pollmask(pfd[0].events),
                                       mstimeout));
      rc = poll( &(pfd[0]), 1, mstimeout );
      TRACE_POLL((-1,"poll(...)=>%d%s [revents=%s]\n", rc, trace_expand_api_rc(rc,fd),
                                       __trace_expand_pollmask(pfd[0].revents)));
      if (rc == 0) /* timed out */
      {
        break; /* result_events = 0, return 0 */
      }
      if (rc > 0) /* got result */
      {
        if ((pfd[0].revents & POLLNVAL)!=0)
        {
          rc = ps_EBADF;
          break;
        }
        if ((pfd[0].revents & (POLLERR|POLLHUP))!=0) /* for BSD wierdness */
          pfd[0].revents &= (POLLERR|POLLHUP);       /* above */
        /* easier to simply copy, but oh well */
        if ((pfd[0].revents & POLLIN)!=0)  result_events |= POLLIN;
        if ((pfd[0].revents & POLLPRI)!=0) result_events |= POLLPRI;
        if ((pfd[0].revents & POLLOUT)!=0) result_events |= POLLOUT;
        if ((pfd[0].revents & POLLERR)!=0) result_events |= POLLERR;
        if ((pfd[0].revents & POLLHUP)!=0) result_events |= POLLHUP;
        rc = 0;
        break;
      }
      if (errno != EINTR)
      {
        rc = ps_stdsyserr;
        break; /* return ps_stdsyserr */
      }
      if (mstimeout < 0 || (++retry_count) == 5) /* hung in retry */
      {
        result_events = POLLERR;
        rc = 0; /* yes, success */
        break; /* success */
      }
      /* otherwise retry */
    }
#elif defined(SOCK_STREAM) /* BSD sox */
    /* note: this implementation of poll() using select() does not
       unconditionally add the read fds for testing for disconnect
       indications. The caller must either do that separately, or
       explicitely add POLLIN to events.
       We do it that way because we want the timeout to be honored
       and don't have a portable way to measure elapsed time here.
    */
    int retry_count = 0;
    struct timeval tv;
    struct timeval *tvP;
     wfds, rfds, xfds;
    fd_set *wfdsP, *rfdsP;

    /* this has to be outside the loop in case tv is modified */
    tvP = (struct timeval *)0;
    tv.tv_sec = (time_t)-1;
    tv.tv_usec = 0;
    if (mstimeout >= 0)
    {
      tv.tv_sec = mstimeout/1000;
      tv.tv_usec = (mstimeout%1000)*1000;
      tvP = &tv;
    }
    rfdsP = (fd_set *)0;
    if ((events & POLLIN)!=0)
    {
      rfdsP = &rfds;
      FD_ZERO(rfdsP);
      FD_SET(fd,rfdsP);
    }
    wfdsP = (fd_set *)0;
    if ((events & POLLOUT)!=0)
    {
      wfdsP = &wfds;
      FD_ZERO(wfdsP);
      FD_SET(fd,wfdsP);
    }
    FD_ZERO(&xfds);
    FD_SET(fd,&xfds);

    for (;;)
    {
      /* select() is sooooo damn ooogly! (its saving grace is one bit per fd) */
      TRACE_POLL((+1,"select(n, %p, %p, %p, %d:%d )\n", rfdsP,wfdsP,&xfds,tv.tv_sec,tv.tv_usec));
      rc = select( fd+1, rfdsP, wfdsP, &xfds, tvP );
      TRACE_POLL((-1,"select(...) =>%d%s\n", rc, trace_expand_api_rc(rc,fd) ));

      if (rc == 0) /* timed out */
      {
        break; /* result_events = 0, return 0 */
      }
      else if (rc > 0) /* have fd */
      {
        int isx = FD_ISSET(fd,&xfds);
        TRACE_POLL((0,"FD_ISSET(fd,xfds)=>%d\n",isx));
        if (wfdsP)
        {
          if (FD_ISSET(fd,wfdsP))
            result_events |= POLLOUT;
          TRACE_POLL((0,"FD_ISSET(fd,wfdsP)=>%d\n",((result_events&POLLOUT)!=0)));
        }
        if (rfdsP)
        {
          if (FD_ISSET(fd,rfdsP))
          {
            char ch;
            TRACE_POLL((0,"FD_ISSET(fd,rfds)=>1\n"));
            #if defined(MSG_OOB)
            if (isx && (events & POLLPRI)!=0)
            {
              TRACE_POLL((+1,"recv(...,MSG_PEEK|MSG_OOB)\n"));
              rc = recv(fd, &ch, 1, MSG_PEEK|MSG_OOB);
              TRACE_POLL((-1,"recv(...,MSG_PEEK|MSG_OOB) =>%d\n",rc));
              if (rc > 0)
                result_events |= POLLPRI;
            }
            #endif
            if ((result_events & (POLLPRI|POLLHUP|POLLERR))==0)
            {
              TRACE_POLL((+1,"recv(...,MSG_PEEK)\n"));
              rc = recv(fd, &ch, 1, MSG_PEEK);
              TRACE_POLL((-1,"recv(...,MSG_PEEK) =>%d\n",rc));
              if (rc == 0) /* 0 means graceful shutdown */
                result_events |= POLLHUP;
              else if (rc < 0) /* means reset */
                result_events |= POLLERR;
              else if ((events & POLLIN)!=0)
                result_events |= POLLIN;
            }
          }
        }
        rc = 0; /* success */
        if (result_events == 0 && isx && (rfdsP || wfdsP))
        {
          result_events = POLLERR;
          #if 1 /* not truly spec, but we don't want to do the getsockopt */
          rc = ps_bsdsockerr; /* ... if its not going to be needed later */
          #endif
        }
        break;
      } /* have fd */
      else if (!net_match_errno(ps_EINTR)) /* its not EINTR */
      {
        rc = ps_stdneterr;
        break;
      }
      if (mstimeout < 0 || (++retry_count) == 5) /* stuck in EINTR loop */
      {
        result_events = POLLERR;
        rc = 0; /* yes, success */
        break; /* success */
      }
      /* otherwise retry */
    } /* for (;;) */
#endif /* BSD sox */
  }

  if (rc == 0 && revents)
    *revents = result_events;

  TRACE_POLL((-1,"net_poll1(...) => %d (revents=%s)\n",
                rc, __trace_expand_pollmask(result_events) ));
  return rc;
}
#endif /* LOOK_NOT_POLL */

/* ======================================================================== */
/* FD/FLOW/PROTO CONTROL PRIMITIVES (BSD socks only)                        */
/* ======================================================================== */

/* wrapper around various ioctl calling conventions. net_ioctl
   cannot/will never be called with an illegal/unknown opt.
*/
#if (!defined(_TIUSER_) && defined(SOCK_STREAM)) /* BSD sox only */
  #if (defined(FIONBIO) || defined(FIONREAD))
    #define HAVE_NET_IOCTL
  #endif
  #if (!defined(FIONBIO)) /* don't need fcntl if we already have FIONBIO */ \
    && defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK))
    #define HAVE_FCNTL_FIONBIO
  #endif
#endif

#if defined(HAVE_NET_IOCTL) /* have BSD sox + FIONBIO or FIONREAD */
static int net_ioctl( SOCKET sock, unsigned long opt, int *i_optval )
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    unsigned long optval = (unsigned long)*i_optval; //hmm!
    if (ioctlsocket(sock, opt, &optval)!=0)
      return ps_stdneterr;
    #if (UINT_MAX != ULONG_MAX)
    *i_optval = (int)((optval > UINT_MAX) ? (UINT_MAX) : (optval));
    #else
    *i_optval = (int)optval;
    #endif
    return 0;
  #elif (CLIENT_OS == OS_VMS)
    unsigned long optval = (unsigned long)*i_optval;
    #if defined(FIONBIO) && defined(__VMS_UCX__)
    // nonblocking sockets not directly supported by UCX
    // - DIGITAL's work around requires system privileges to use
    if (opt = FIONBIO) { errno = EPERM; return ps_stdneterr; }
    #endif
    /* defined(MULTINET) or opt!=FIONBIO */
    if (socket_ioctl(sock, opt, &optval)!=0)
      return ps_stdneterr;
    #if (UINT_MAX != ULONG_MAX)
    *i_optval = (int)((optval > UINT_MAX) ? (UINT_MAX) : (optval));
    #else
    *i_optval = (int)optval;
    #endif
    return 0;
  #elif (CLIENT_OS == OS_RISCOS)
    int optval = (int)*i_optval;
    if (ioctl(sock, opt, i_optval) != 0) return ps_stdneterr;
    *i_optval = optval;
    return 0;
  #elif (CLIENT_OS == OS_OS2)
    int optval = (int)*i_optval;
    if (ioctl(sock, opt, i_optval) != 0) return ps_stdneterr;
    *i_optval = optval;
    return 0;
  #elif (CLIENT_OS == OS_AMIGAOS)
    if (IoctlSocket(sock, opt, (char *)i_optval)!=0) return ps_stdneterr;
    return 0;
  #elif (CLIENT_OS == OS_LINUX) /*use ioctl to avoid 2.0+2.1 vs 2.2+ trouble*/
    unsigned int optval = (unsigned int)*i_optval;
    if (ioctl(sock, opt, &optval )!=0) return ps_stdneterr;
    *i_optval = (int)optval;
    return 0;
  #else
    return ((ioctl(sock, opt, (void *)i_optval )==0) ? 0 : ps_stdneterr);
  //#else
  // return ps_ENOSYS;
  #endif
}
#endif /* only required by BSD sox */

/* --------------------------------------------------------------------- */

#if defined(HAVE_FCNTL_FIONBIO)  /* no FIONBIO but have fcntl() */
static int ___fcntl_O_NONBLOCK(int sock, int set_blocking)
{
  int res, flag;
  #if (defined(FNDELAY))
  flag = FNDELAY;
  #else
  flag = O_NONBLOCK;
  #endif
  if (( res = fcntl(sock, F_GETFL, flag ) ) != -1)
  {
    //printf("want: %sblocking, was: %sblocking, ", ((set_blocking==0)?("non-"):("")), ((res & flag)?("non-"):("")) );
    if ((set_blocking == 0 /* off */ && (res & flag) != 0) ||
        (set_blocking != 0 /* on */  && (res & flag) == 0))
    {
      //printf("no change needed\n");
      return 0;
    }
    if (fcntl(sock, F_SETFL, (res ^ flag)) != -1)
    {
      if (( res = fcntl(sock, F_GETFL, flag ) ) != -1)
      {
        if ((set_blocking == 0 /* off */ && (res & flag) != 0) ||
            (set_blocking != 0 /* on */  && (res & flag) == 0))
        {
          //printf("changed ok\n");
          return 0;
        }
        //printf("fcntl() said 'ok' but did not change. still %sblocking\n", ((res & flag)?("non-"):("")) );
        errno = EPERM;
      }
    }
  }
  //printf("fcntl err: '%s'\n", strerror(errno));
  return ps_stdneterr;
}
#endif /* only required by BSD sox */

/* --------------------------------------------------------------------- */

#if (!defined(_TIUSER_) && defined(SOCK_STREAM)) /* BSD sox only */
static int net_set_blocking(SOCKET fd)
{
  #if defined(HAVE_FCNTL_FIONBIO)
  return ___fcntl_O_NONBLOCK(fd, 1);
  #elif defined(HAVE_NET_IOCTL)
  int rc, flagon = 0; /* blocking */
  TRACE_FIONBIO((+1,"ioctl(%d, FIONBIO, %d)\n", fd, flagon ));
  rc = net_ioctl(fd, FIONBIO, &flagon );
  TRACE_FIONBIO((-1,"ioctl(%d, FIONBIO, %d) => %d%s\n", fd, flagon, rc, trace_expand_ps_rc(rc,fd) ));
  return rc;
  #else
  #error either ioctl or fcntl support is required
  #endif
}
#endif /* BSD sox */

/* --------------------------------------------------------------------- */

#if (!defined(_TIUSER_) && defined(SOCK_STREAM)) /* BSD sox only */
static int net_set_nonblocking(SOCKET fd)
{
  #if defined(HAVE_FCNTL_FIONBIO)
  return ___fcntl_O_NONBLOCK(fd, 0);
  #elif defined(HAVE_NET_IOCTL)
  int rc, flagon = 1; /* non-blocking */
  TRACE_FIONBIO((+1,"ioctl(%d, FIONBIO, %d)\n", fd, flagon ));
  rc = net_ioctl(fd, FIONBIO, &flagon );
  TRACE_FIONBIO((-1,"ioctl(%d, FIONBIO, %d) => %d%s\n", fd, flagon, rc, trace_expand_ps_rc(rc,fd) ));
  return rc;
  #else
  #error either ioctl or fcntl support is required
  #endif
}
#endif /* BSD sox */

/* ======================================================================== */
/* ENDPOINT OPEN/CLOSE                                                      */
/* ======================================================================== */

/* create/close a tcp endpoint. May cause an api library to be
 * loaded/unloaded. open may implicitely (lib load) or exlicitely
 * (user wants dialup control) cause a dialup connection. If
 * a dialup was explicitely caused, net_close() will disconnect.
*/

static unsigned int open_endpoint_count = 0;

int net_close(SOCKET fd)
{
  int rc = ps_EBADF;
  TRACE_CLOSE((+1,"net_close(s)\n" ));
  if ( fd != INVALID_SOCKET )
  {
#if defined( _TIUSER_ )                                //TLI
    //ioctl( fd, I_SETDELAY, 0); /* turn blocking back on */
    if ( t_getstate( fd ) != T_UNBND )
    {
      t_rcvrel( fd );   /* wait for conn release by peer */
      t_sndrel( fd );   /* initiate close */
      t_unbind( fd );   /* close our own socket */
    }
    rc = t_close( fd );
    if (rc != 0)
      rc = ps_stdneterr;
#elif defined(SOCK_STREAM) /* BSD sox */
    #ifndef FIONREAD /* default socket state is non-blocking */
    net_set_blocking(fd);  /* so turn blocking back on */
    #endif
    TRACE_CLOSE((+1,"shutdown(s,2)\n"));
    rc = shutdown( fd, 2 );
    TRACE_CLOSE((-1,"shutdown(s,2) => %d%s\n", rc, trace_expand_api_rc(rc,fd) ));
    TRACE_CLOSE((+1,"close(s)\n"));
    #if (CLIENT_OS == OS_OS2) && !defined(__EMX__)
    rc = (int)soclose( fd );
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    rc = (int)closesocket( fd );
    #elif (CLIENT_OS == OS_AMIGAOS)
    rc = (int)CloseSocket( fd );
    #elif (CLIENT_OS == OS_BEOS)
    rc = (int)closesocket( fd );
    #elif (CLIENT_OS == OS_VMS) && defined(MULTINET)
    rc = (int)socket_close( fd );
    #else
    rc = (int)close( fd );
    #endif
    TRACE_CLOSE((-1,"close(s) = %d%s\n",rc, trace_expand_api_rc(rc,fd)));
    if (rc != 0)
      rc = ps_stdneterr;
#endif
    if (rc == 0)
    {
      TRACE_CLOSE((0,"next endpoint count = %d\n", open_endpoint_count-1 ));
      if ((--open_endpoint_count) == 0)
         net_init_check_deinit(-1,0);
    }
  }
  TRACE_CLOSE((-1,"net_close(s) => %d%s\n", rc, trace_expand_ps_rc(rc,fd)));
  return rc;
}

/* --------------------------------------------------------------------- */

#if !defined(_TIUSER_) && defined(SOCK_STREAM) /* BSD sox only */
static int bsd_condition_new_socket(SOCKET fd, int as_listener)
{
  int rc = 0, min_buf_size = 0;

  #if (CLIENT_OS == OS_RISCOS)
  if (rc == 0)
  {
    // allow blocking socket calls to preemptively multitask
    int fon = 1;
    //ioctl(fd, FIOSLEEPTW, &fon);
  }
  #endif
  if (rc == 0 && as_listener) /* server */
  {
    min_buf_size = 0; /* servers don't need to adjust buffer size */
    rc = net_set_nonblocking(fd); /* listeners are always non-blocking */
    if (rc == 0)
    {
      #if defined(SOL_SOCKET) && defined(SO_REUSEADDR)
      int fon = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*)&fon, sizeof(int));
      #endif
    }
  }
  else if (rc == 0) /* not a server */
  {
    min_buf_size = 2048;
    #if !defined(FIONREAD) /* needs to be non-blocking */
    rc = net_set_nonblocking(fd);
    #else
    rc = net_set_blocking(fd); /* can be non-blocking */
    #endif
  }
  if (rc == 0 && min_buf_size > 0)
  {
    #if (defined(SOL_SOCKET) && defined(SO_RCVBUF) && defined(SO_SNDBUF))
    int which;
    for (which = 0; which < 2; which++ )
    {
      int sz = 0, type = ((which == 0)?(SO_RCVBUF):(SO_SNDBUF));
      socklen_t szint = (socklen_t)sizeof(int);
      if (getsockopt(fd, SOL_SOCKET, type, (char *)&sz, &szint) < 0)
        ;
      else if (sz < min_buf_size)
      {
        sz = min_buf_size;
        setsockopt(fd, SOL_SOCKET, type, (char *)&sz, szint);
      }
    }
    #endif
  }
  return rc;
}
#endif /* BSD sox */


/* --------------------------------------------------------------------- */

int net_open(SOCKET *sockP, u32 local_addr, int local_port)
{
  int rc = 0;

  if (!sockP)
    return ps_EINVAL;

  TRACE_OPEN((+1,"net_open(%p, %s:%d)\n", sockP, net_ntoa(local_addr), local_port ));
  if (local_port<0 || local_port>0xffff || local_addr==0xffffffff)
  {
    rc = ps_EINVAL;
  }
  if ((++open_endpoint_count) == 1)
  {
    if ((rc = net_init_check_deinit(+1,0)) != 0)
    {
      open_endpoint_count--;
    }
  }
  else if ((rc = net_init_check_deinit(0,0)) != 0)
  {
    ; /* rc = ps_ENETDOWN|ps_ENOSYS; above */
  }

  if (rc == 0)
  {
    #if defined(AF_INET)
    #endif

    #if defined(_TIUSER_)
    {
      int fd = t_open("/dev/tcp", O_RDWR|O_NDELAY, ((struct t_info *)0) );
      rc = ps_stdneterr;
      if (fd != -1)
      {
        rc = 0;
        if ( local_port != 0)
        {
          struct t_bind bnd;
          struct sockaddr_in saddr;
          memset((void *) &saddr, 0, sizeof(saddr));
          saddr.sin_family = AF_INET;
          saddr.sin_port = htons(((u16)local_port));
          saddr.sin_addr.s_addr = local_addr;
          bnd.addr.maxlen = bnd.addr.len = sizeof(saddr);
          bnd.addr.buf = (char *)&saddr;
          bnd.qlen = 5; /* MAX_CIND */
          if ( t_bind( fd, &bnd, ((struct t_bind *)0) ) == -1 )
            rc = ps_stdneterr;
        }
      }
      if (rc == 0)
      {
        *sockP = fd;
        rc = 0;
      }
      else
      {
        t_close(fd);
      }
    }
    #elif defined(SOCK_STREAM)                       /* BSD sox */
    {
      int fd;
      TRACE_OPEN((+1,"socket( AF_INET, SOCK_STREAM, 0 )\n" ));
      fd = (int)socket(AF_INET, SOCK_STREAM, 0);
      TRACE_OPEN((-1,"socket( AF_INET, SOCK_STREAM, 0 ) => %d%s\n",fd, trace_expand_api_rc(((fd==-1)?(-1):(0)),fd) ));
      rc = ps_stdneterr;
      if (fd != -1)
      {
        rc = bsd_condition_new_socket( fd, (local_port != 0) );
        if (rc == 0 && local_port) /* server */
        {
          struct sockaddr_in saddr;
          memset((void *) &saddr, 0, sizeof(saddr));
          saddr.sin_family = AF_INET;
          saddr.sin_port = htons(((u16)local_port));
          saddr.sin_addr.s_addr = local_addr;
          TRACE_OPEN((+1,"bind(fd,%s:%d,...)\n",net_ntoa(local_addr),local_port));
          while (bind(fd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
          {
            if (net_match_errno(ps_EINTR)) /* caught a signal */
              continue;                    /* retry */
            rc = ps_stdneterr;
            break;
          }
          TRACE_OPEN((-1,"bind(fd,%s:%d,...) => %d%s\n",net_ntoa(local_addr),local_port,rc, trace_expand_ps_rc(rc,fd) ));
        }
        if (rc == 0 && local_port) /* server */
        {
          TRACE_OPEN((+1,"listen(fd,5)\n"));
          while (listen(fd, 5) < 0)
          {
            if (net_match_errno(ps_EINTR)) /* caught a signal */
              continue;                    /* retry */
            rc = ps_stdneterr;
            break;
          }
          TRACE_OPEN((-1,"listen(fd,5) =>%d%s\n",rc,trace_expand_ps_rc(rc,fd)));
        }
        if (rc == 0)
          *sockP = fd;
        else
          net_close(fd);
      } /* if (fd != -1) */
    } /* BSD sox */
    #endif
    if (rc != 0)
    {
      if (rc == ps_stdneterr) /* need to save before deinit */
      {
        ps_oereserved_cache.ps_errnum = ___read_errnos( INVALID_SOCKET, rc,
                                        &ps_oereserved_cache.syserr,
                                        &ps_oereserved_cache.neterr,
                                        &ps_oereserved_cache.extra );
        rc = ps_oereserved;
      }
      if ((--open_endpoint_count)==0)
        net_init_check_deinit(-1,0);
    }
  } /* net check ok */
  TRACE_OPEN((-1,"net_open(...)=%d%s\n",rc,trace_expand_ps_rc(rc,*sockP)));
  return rc;
}

/* ======================================================================== */
/* CONNECT/ACCEPT                                                           */
/* ======================================================================== */

int net_connect( SOCKET sock, u32 *that_address, int *that_port,
                              u32 *this_address, int *this_port,
                              int iotimeout /* millisecs */ )
{
  int rc = ps_ENETDOWN;
  this_address = this_address; this_port = this_port; /* shaddup compiler */

  TRACE_CONNECT((+1, "net_connect(s, %s:%d, %d)\n",
                     net_ntoa((that_address)?(*that_address):(0)),
                     ((that_port)?(*that_port):(0)), iotimeout ));

  if ( sock == INVALID_SOCKET )
  {
    rc = ps_EBADF;
  }
  else if (!that_address || !that_port)
  {
    rc = ps_EINVAL;
  }
  else if (!*that_address || *that_port <= 0 || *that_port > 0xffff)
  {
    rc = ps_EINVAL;
  }
  else if ((rc = net_init_check_deinit(0,0)) == 0)  /* check */
  {
    int break_pending = CheckExitRequestTriggerNoIO();
    int loopcount = 0, maxloops = 0, mssleep = 0;
    __calc_timeout_metrics(iotimeout, /* millisecs */
                           &maxloops, /* max number of loops */
                           &mssleep ); /* sleep time per loop */

#if defined(_TIUSER_)                                         //OSI/XTI/TLI
    {
      struct sockaddr_in saddr;
      struct t_bind bnd;
      struct t_bind *bndP = (struct t_bind *)0;

      memset( &bnd, 0, sizeof(struct t_bind) );
      memset( &saddr, 0, sizeof(struct sockaddr_in) );

      bnd.addr.maxlen = bnd.addr.len = sizeof(saddr);
      bnd.addr.buf = (char *)&saddr;
      bnd.qlen = 0; /* listen depth */

      if (this_address)
      {
        if (*this_address)
        {
          saddr.sin_family = AF_INET;
          saddr.sin_addr.s_addr = *this_address;
          bndP = &bnd;
        }
      }
      rc = ps_stdneterr;
      if (t_bind( sock, bndP, &bnd ) != -1)
      {
        u32 bound_addr = saddr.sin_addr.s_addr;
        int bound_port = 0xffff & ((int)htons(saddr.sin_port));
        struct t_call sndcall;
        int err;

        memset((void *) &saddr, 0, sizeof(saddr));
        saddr.sin_family = AF_INET;
        saddr.sin_port = (u16)htons(((u16)*that_port));
        saddr.sin_addr.s_addr = *that_address;

        memset((void *) &sndcall, 0, sizeof(sndcall));
        sndcall.addr.len = sizeof(saddr);
        sndcall.addr.maxlen = sizeof(saddr);
        sndcall.addr.buf = (char *)&saddr;

        err = 0;
        rc = t_connect( sock, &sndcall, &sndcall );
        if (rc < 0)
          err = t_errno;
        while (rc < 0 && err == TNODATA)
        {
          struct pollfd pfd;
          pfd.fd = sock;
          pfd.events = POLLOUT;
          pfd.revents = 0;
          if (poll(&pfd, 1, mssleep) < 0)
          {
            t_errno = err = TSYSERR;
            break;
          }
          if (pfd.revents != 0) /* POLLOUT/POLLERR/POLLHUP */
          {
            rc = t_rcvconnect(sock, &sndcall );
            if (rc < 0)
              err = t_errno;
          }
          else if (!break_pending && CheckExitRequestTriggerNoIO())
          {
            t_errno = err = TSYSERR;
            errno = EINTR;
          }
          else if ((++loopcount) > maxloops)
          {
            t_errno = err = TSYSERR;
            errno = ETIMEDOUT;
          }
        } /* while (rc < 0 && err == TNODATA) */
        if (rc < 0)
        {
          rc = ps_stdneterr;
          if (err == TLOOK)
          {
            int look = t_look(sock);
            if (look == T_CONNECT)
            {
              if (t_rcvconnect(sock, &sndcall )==0)
                rc = 0;
              else
              {
                err = t_errno;
                if (err == TLOOK);
                  look = t_look(sock);
              }
            }
            if (look == T_DISCONNECT)
            {
              struct t_discon tdiscon;
              tdiscon.udata.buf = (char *)0;
              tdiscon.udata.maxlen = 0;
              tdiscon.udata.len = 0;
              if (t_rcvdis(sock, &tdiscon) < 0)
                tdiscon.reason = ECONNREFUSED;
              t_errno = TSYSERR;
              errno = tdiscon.reason;
            }
          } /* err == TLOOK */
        } /* if (rc < 0) */
        if (rc == 0)
        {
          if (this_address)
            *this_address = bound_addr;
          if (this_port)
            *this_port = bound_port;
          if (that_address)
            *that_address = saddr.sin_addr.s_addr;
          if (that_port)
            *that_port = 0xffff & ((int)htons(saddr.sin_port));
        }
      }
    } /* _TIUSER_ */
    #elif defined(SOCK_STREAM) //BSD sox
    {
      struct sockaddr_in saddr;

      rc = 0;
      if (this_address)
      {
        if (*this_address)
        {
          memset((void *) &saddr, 0, sizeof(saddr));
          saddr.sin_family = AF_INET;
          saddr.sin_addr.s_addr = *this_address;
          TRACE_CONNECT((+1,"bind(fd,%s:%d,...)\n",net_ntoa(*this_address),0));
          while (bind(sock, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
          {
            if (net_match_errno(ps_EINTR)) /* caught a signal */
              continue;                    /* retry */
            rc = ps_stdneterr;
            break;
          }
          TRACE_CONNECT((-1,"bind(...) => %d%s\n",rc, trace_expand_ps_rc(rc,sock) ));
        }
      }

      if (rc == 0)
      {
        int in_progress = 0, is_async = 1;
        memset((void *) &saddr, 0, sizeof(saddr));
        saddr.sin_family = AF_INET;
        saddr.sin_port = htons(((u16)*that_port));
        saddr.sin_addr.s_addr = *that_address;

        #if defined(FIONREAD)  /* socket default state is blocking */
        is_async = 0;
        if (iotimeout >= 0) /* we have a non-blocking timeout value */
        {
          if (net_set_nonblocking(sock) == 0) /* so convert to non-blocking */
            is_async = 1;
        }
        #endif

        TRACE_CONNECT((+1,"connect(s, %s:%d)\n", net_ntoa(*that_address), *that_port));
        rc = connect(sock, (struct sockaddr *)&saddr, sizeof(saddr));
        TRACE_CONNECT((-1,"connect(s) => %d%s\n", rc, trace_expand_api_rc(rc,sock) ));

        TRACE_CONNECT((0,"phase 1: rc=%d is_async=%d in_progress=%d\n",rc,is_async,in_progress));
        if (rc != 0) /* connect error or not complete */
        {
          rc = ps_stdneterr;
          if (is_async && net_match_errno(ps_EINPROGRESS))
          {
            in_progress = 1;
            rc = 0;
          }
        }
        TRACE_CONNECT((0,"phase 2: rc=%d is_async=%d in_progress=%d\n",rc,is_async,in_progress));
        if (is_async) /* connect was executed asynchronously */
        {
          #if defined(FIONREAD) /* socket default state is blocking */
          int rc2 = net_set_blocking(sock); /* so return to blocking state */
          if (rc == 0) rc = rc2;
          #endif
        }
        TRACE_CONNECT((0,"phase 3: rc=%d is_async=%d in_progress=%d\n",rc,is_async,in_progress));
        if (rc == 0 && in_progress) /* connect completion pending */
        {
          TRACE_CONNECT((+1,"EINPROGRESS loop\n"));
          for (;;)
          {
            int revents = 0;
#if defined(LOOK_NOT_POLL)
            TRACE_CONNECT((0,"begin net_look\n"));
            rc = net_look( sock, ps_T_CONNECT, &revents, mssleep );
            TRACE_CONNECT((0,"end net_look\n"));
            if (rc != 0)
              break;
            if (revents == ps_T_CONNECT)
            {
              rc = 0;
              break;
            }
#else
            rc = net_poll1( sock, POLLOUT|POLLIN, &revents, mssleep );
            if (rc != 0)
              break;
            if ((revents & POLLERR)!=0) /* this shouldn't happen */
            {                           /* (rc should be non-zero) */
              rc = ps_stdneterr;
              break;
            }
            if ((revents & POLLIN)!=0)  /* we have an error */
            {
              char c;
              TRACE_CONNECT((+1,"recv(s,&ch,1,0)\n"));
              rc = recv(sock, &c, 1, 0);
              TRACE_CONNECT((-1,"recv(s,&ch,1,0) => %d%s\n", rc, trace_expand_api_rc(rc,sock) ));
              if (rc != 0) /* should always be the case here */
              {
                rc = ps_stdneterr;
                break;
              }
            }
            if ((revents & POLLOUT)!=0)
            {
              rc = 0;
              break;
            }
#endif
            #if ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32))
            {
              /* wsock32.dll bug: does not set wfds on successful connect */
              TRACE_CONNECT((+1,"2] connect(s, %s:%d)\n", net_ntoa(*that_address), *that_port));
              rc = connect(sock, (struct sockaddr *)&saddr, sizeof(saddr));
              TRACE_CONNECT((-1,"2] connect(s) => %d%s\n", rc, trace_expand_api_rc(rc,sock) ));
              if (rc == 0)
                break;
              rc = WSAGetLastError();
              if (rc == WSAEINVAL) /* Winsock 1.1 returns WSAEINVAL */
                rc = WSAEALREADY;  /* instead of WSAEALREADY */
              if (rc == WSAEISCONN)
              {
                rc = 0;
                break;
              }
              if (rc != WSAEINPROGRESS && rc != WSAEALREADY && rc != WSAEWOULDBLOCK)
              {
                rc = ps_stdneterr;
                break;
              }
            }
            #endif
            if (!break_pending && CheckExitRequestTriggerNoIO())
            {
              rc = ps_EINTR;
              break;
            }
            if ((++loopcount) >= maxloops)
            {
              rc = ps_ETIMEDOUT;
              break;
            }
            TRACE_CONNECT((0,"EINPROGRESS loop bottom\n"));
          } /* for (;;) */
          TRACE_CONNECT((-1,"EINPROGRESS loop =>%d%s\n",rc,trace_expand_ps_rc(rc,sock)));
        } /* connect completion pending */
        if (rc == 0)
        {
          int rc2;
          socklen_t addrsz;
          if (this_address || this_port)
          {
            addrsz = sizeof(saddr);
            memset((void *) &saddr, 0, sizeof(saddr));
            saddr.sin_family = AF_INET;

            TRACE_CONNECT((+1,"getsockname(...)\n"));
            rc2 = getsockname(sock, (struct sockaddr *)&saddr, &addrsz);
            TRACE_CONNECT((-1,"getsockname(...) => %d%s\n", rc2, trace_expand_api_rc(rc2,sock)));
            if (rc2 == 0)
            {
              if (this_address)
                *this_address = saddr.sin_addr.s_addr;
              if (this_port)
                *this_port = 0xffff & ((int)htons(saddr.sin_port));
            }
          }
          #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
          /* another freaking windows bug - winNT only:
           * getpeername() will freeze the machine (yes, super-duper-
           * "secure" NT can thus be frozen) in the half-blocking scenario
           * we have here.
          */
          if (winGetVersion() >= 2000) /* winnt (winsock2?) only */
          {
            that_address = 0; that_port = 0;
          }
          #endif
          if (that_address || that_port)
          {
            addrsz = sizeof(saddr);
            memset((void *) &saddr, 0, sizeof(saddr));
            saddr.sin_family = AF_INET;
            TRACE_CONNECT((+1,"getpeername(...)\n"));
            rc2 = getpeername(sock, (struct sockaddr *)&saddr, &addrsz);
            TRACE_CONNECT((-1,"getpeername(...) => %d%s\n", rc2, trace_expand_api_rc(rc2,sock)));
            if (rc2 == 0)
            {
              if (that_address)
                *that_address = saddr.sin_addr.s_addr;
              if (that_port)
                *that_port = 0xffff & ((int)htons(saddr.sin_port));
            }
          }
        }
      } /* bind was successful */
    } /* BSD sox */
#endif
  } /* (net_init_check_deinit(0,0)) */

  TRACE_CONNECT((-1, "net_connect(...) => %d%s\n", rc, trace_expand_ps_rc(rc,sock)));
  return rc;
}

/* --------------------------------------------------------------------- */

/* untested. for "demonstration" purposes only. :) */

int net_accept( SOCKET listen_fd, SOCKET *conn_fdP,
                u32 *that_address, int *that_port, int iotimeout )
{
  int rc = ps_ENETDOWN;

  TRACE_ACCEPT((+1, "net_accept(s,...,%d)\n", iotimeout ));

  if (listen_fd == INVALID_SOCKET)
    rc = ps_EBADF;
  else if (!conn_fdP || !that_address || !that_port)
    rc = ps_EINVAL;
  else if (!is_netapi_callable())
    rc = ps_ENOSYS;
  else if (CheckExitRequestTriggerNoIO())
    rc = ps_EINTR;
  else
  {
    int eintr_count = 0; /* only used by BSD sox */
    int loopcount = 0, maxloops = 0, mssleep = 0;
    __calc_timeout_metrics(iotimeout, /* millisecs */
                           &maxloops, /* max number of loops */
                           &mssleep ); /* sleep time per loop */
    for (;;)
    {
      #if defined(_TIUSER_)
      struct t_call sndcall;
      struct sockaddr_in saddr;

      memset((void *) &saddr, 0, sizeof(saddr));
      saddr.sin_family = AF_INET;
      saddr.sin_port = 0;
      saddr.sin_addr.s_addr = 0;

      memset((void *) &sndcall, 0, sizeof(sndcall));
      sndcall.addr.len = sizeof(saddr);
      sndcall.addr.maxlen = sizeof(saddr);
      sndcall.addr.buf = (char *)&saddr;

      /* Listen for any incoming indications */
      rc = t_listen(listen_fd, &sndcall );
      if (rc == -1)
      {
        rc = ps_stdneterr;
        if (t_errno != TNODATA) /* no connect indication pending */
          break;
      }
      else
      {
        int conn_fd = -1;
        rc = net_open( &conn_fd, 0,0); /* create a new client endpoint (bound) */
        if (rc == 0)
        {
          if (t_accept(listen_fd, conn_fd, &sndcall) == -1)
          {
            rc = t_errno;
            net_close(conn_fd);
            t_errno = rc;
            rc = ps_stdneterr;
          }
          else
          {
            *conn_fdP = conn_fd;
            *that_address = saddr.sin_addr.s_addr;
            *that_port = (0xffff & ((int)ntohs(saddr.sin_port)));
            break;
          }
        }
      }
      #elif defined(SOCK_STREAM) /* BSD sox */
#ifdef LOOK_NOT_POLL
      int revents = 0;
      rc = net_look( listen_fd, ps_T_LISTEN, &revents, mssleep );
      if (rc != 0)
        break;
      if (revents == ps_T_LISTEN)
        rc = 1;
#else
      fd_set fdr;
      struct timeval tv;
      FD_ZERO( &fdr );
      FD_SET( listen_fd, &fdr );
      tv.tv_sec = 0;
      tv.tv_usec = mssleep * 1000;

      TRACE_ACCEPT((+1,"select(, &fdr, 0, 0, 0:%d)\n",mssleep ));
      rc = select( listen_fd+1, &fdr, 0, 0, &tv );
      TRACE_ACCEPT((-1,"select(...) => %d%s\n", rc, trace_expand_api_rc(rc,listen_fd) ));
#endif
      if (rc > 0) /* connect pending */
      {
        SOCKET conn_fd;
        struct sockaddr_in saddr;
        socklen_t addrlen = sizeof(saddr);

        memset((void *) &saddr, 0, sizeof(saddr));
        saddr.sin_family = AF_INET;
        saddr.sin_port = 0;
        saddr.sin_addr.s_addr = 0;

        TRACE_ACCEPT((+1,"accept(...)\n"));
        conn_fd = accept( listen_fd, (struct sockaddr *)&saddr, &addrlen );
        TRACE_ACCEPT((-1,"accept(...) => %d%s\n", conn_fd, trace_expand_api_rc(conn_fd,listen_fd) ));

        if (conn_fd != INVALID_SOCKET)
        {
          rc = bsd_condition_new_socket(conn_fd, 0);
          if (rc != 0)
          {
            net_close(conn_fd);
            break;
          }
          *conn_fdP = conn_fd;
          *that_address = saddr.sin_addr.s_addr;
          *that_port = (0xffff & ((int)ntohs(saddr.sin_port)));
          TRACE_ACCEPT((0,"peer: %s:%d\n", *that_address, *that_port));
          rc = 0;
          break;
        }
        rc = -1;
        #if 0 /* cannot happen since ... */
        /* a) listeners are non-blocking b) select said conn waiting */
        if (net_match_errno(ps_EWOULDBLOCK))
          rc = 0; /* treat as timeout */
        #endif
      }
      if (rc == 0) /* time out */
      {
        eintr_count = 0;
      }
      else /* error */
      {
        rc = ps_stdneterr;
        if (!net_match_errno(ps_EINTR))
          break;
        eintr_count++;
        maxloops++;
      }
      #endif
      if (eintr_count > 5) /* stuck in EINTR */
      {
        rc = ps_stdneterr; /* ie EINTR */
        break;
      }
      if (CheckExitRequestTriggerNoIO())
      {
        rc = ps_EINTR;
        break;
      }
      if ((++loopcount) > maxloops)
      {
        rc = ps_ETIMEDOUT;
        break;
      }
    } /* for (;;) */
  } /* netapi_is_callable */

  TRACE_ACCEPT((-1, "net_accept(...) => %d%s\n", rc, trace_expand_ps_rc(rc,listen_fd)));
  return rc;
}

/* ======================================================================== */
/* READ/WRITE                                                               */
/* ======================================================================== */

/* read/write from an endpoint. read() will return as soon as any
*  data is available (or error or timeout). write() will return
*  as soon as the data has been queued completely (which may require
*  some of the data to be sent over the wire first). On timeout (no
*  data was sent/recvd), both functions returns zero and *bufsz will
*  be zero. This is believed to be more useful than returning a
*  'timedout' error code.
*/
int net_read( SOCKET sock, char *data, unsigned int *bufsz,
              u32 that_address, int that_port, int iotimeout /*millisecs*/ )
{
  unsigned int totaltodo = ((bufsz)?(*bufsz):(0));
  unsigned int totaldone = 0;
  int rc;

  TRACE_READ((+1,"net_read(s, %p, %d, %s:%d, %d)\n",
             data, totaltodo, net_ntoa(that_address), that_port, iotimeout));

  rc = ps_ENETDOWN;
  if ( sock == INVALID_SOCKET )
  {
    rc = ps_EBADF;
  }
  else if (!bufsz || !data || !that_address || !that_port)
  {
    rc = ps_EINVAL;
  }
  else if ((rc = net_init_check_deinit(0,0)) == 0) /* check */
  {
    int break_pending = CheckExitRequestTriggerNoIO();
    int tryloops = 0, maxloops = 0, mssleep = 0;
    unsigned int remainingtodo = totaltodo;
    unsigned int recvquota = 1500; /* how much to read per read() call */
    rc = ps_stdneterr;

    __calc_timeout_metrics(iotimeout, /* millisecs */
                           &maxloops, /* max number of loops */
                           &mssleep ); /* sleep time per loop */

    #if defined(_TIUSER_)
    recvquota = 512;
    struct t_info info;
    if ( t_getinfo( sock, &info ) < 0)
      info.tsdu = 0; /* assume tdsu not suppported */
    else if (info.tsdu > 0)
      recvquota = info.tsdu;
    else if (info.tsdu == -1) /* no limit */
      recvquota = totaltodo;
    else if (info.tsdu == 0) /* no boundaries */
      recvquota = 1500;
    #elif (CLIENT_OS == OS_WIN16)
    if (recvquota > 0x7FFF)  /* 16 bit OS but int is 32 bits */
      recvquota = 0x7FFF;
    #endif
    if (recvquota > INT_MAX)
      recvquota = INT_MAX;

    TRACE_READ((0,"recvquota = %u\n", recvquota));

    for (;;)
    {
      int revents = 0;
      TRACE_READ((0,"tryloops=%d, maxloops=%d, mssleep=%d\n", tryloops, maxloops, mssleep));

#if defined(LOOK_NOT_POLL)
      rc = net_look( sock, ps_T_DATA, &revents, mssleep );
      if (rc != 0)
        break;
      if (revents == ps_T_DATA)
#else
      rc = net_poll1( sock, POLLIN, &revents, mssleep );
      if (rc != 0)
        break;
      if ((revents & (POLLHUP|POLLERR)) != 0)
      {
        rc = ps_EDISCO;
        break;
      }
      if ((revents & POLLIN) != 0)
#endif
      {
        int didread = -1, toread = recvquota;
        if (((unsigned int)toread) > remainingtodo)
          toread = remainingtodo;

        #if defined(_TIUSER_)                              //TLI/XTI
        {
          /* we simulate the behaviour of BSD's recv() here */
          int flags = 0; /* T_MORE, T_EXPEDITED etc */

          #if 0 /* don't do connectionless yet */
          struct t_unitdata udata;
          struct sockaddr_in saddr;

          memset((void *) &saddr, 0, sizeof(saddr));
          saddr.sin_family = AF_INET;
          saddr.sin_port = htons(((u16)that_port));
          saddr.sin_addr.s_addr = that_address;

          udata.addr.maxlen = sizeof(saddr);
          udata.addr.len = sizeof(saddr);
          udata.addr.buf = (char *)&saddr;
          udata.opt.maxlen = udata.opt.len = 0;
          udata.opt.buf = (char *)0;
          udata.udata.maxlen = udata.udata.len = toread;
          udata.udata.buf = data;
          didread = t_rcvudata(sock, &udata, &flags );
          #else
          didread = t_rcv( sock, data, toread, &flags );
          #endif

          if (didread == 0) /* peer sent a zero byte message */
            didread = -1; /* treat as none waiting */
          else if (didread < 0)
          {
            int err = t_errno;
            didread = -1;
            if (err != TNODATA) /*an error other than "no data available yet"*/
            {
              didread = 0; /* treat as closed for all but TLOOK+T_[GO]DATA */
              debugtli("t_rcv", sock);
              if (err == TLOOK) /* not TSYSERR et al */
              {
                int look = t_look(sock);
                if (look == T_ORDREL) /* orderly disconnect */
                  t_rcvrel( sock ); /* ACK it */
                else if (look == T_DISCONNECT || look == T_ERROR)
                  ; /* disorderly disconnect or error - didread = 0; */
                else /* T_DATA (normal data ready) or T_GODATA and family */
                  didread = -1; /* recoverable. treat as none waiting */
              } /* TLOOK */
            } /* err != TNODATA */
          } /* rc < 0 */
        } /* _TIUSER */
        #elif defined(SOCK_STREAM)      /* BSD sox */
        {
          #if defined(FIONREAD)
          int read_ready = -1;
          TRACE_READ((+1,"ioctl(s, FIONREAD, ...)\n" ));
          rc = net_ioctl(sock, FIONREAD, &read_ready);
          TRACE_READ((-1,"ioctl(s, FIONREAD, ...) =>%d (read_ready=%d)\n", rc, read_ready ));
          if (rc != 0)
            break;
          if (read_ready < 0)
            toread = -1;
          else if (read_ready < toread)
            toread = read_ready;
          if (toread >= 0) /* note that we want to read if read_ready is */
          {                /* zero - its going to be an error */
            TRACE_READ((+1,"recv(s, data, %d)\n", toread ));
            #if 0
            struct sockaddr_in saddr;
            socklen_t sz = sizeof(saddr);
            memset((void *) &saddr, 0, sizeof(saddr));
            saddr.sin_family = AF_INET;
            saddr.sin_port = htons(((u16)that_port));
            saddr.sin_addr.s_addr = that_address;
            didread = recvfrom(sock, data, toread, 0, (struct sockaddr *)&saddr, &sz );
            #else
            didread = recv(sock, data, toread, 0 );
            #endif
            TRACE_READ((-1,"recv(s, data, %d) =%d%s\n", toread, didread, trace_expand_api_rc(didread,sock) ));
            if (didread < 0)
            {
              rc = ps_stdneterr;
              break;
            }
          }
          #else /* socket was opened non-blocking */
          TRACE_READ((+1,"recv(s, data, %d)\n", toread ));
          didread = recv(sock, data, toread, 0 );
          TRACE_READ((-1,"recv(s, data, %d) =%d%s\n", toread, didread, trace_expand_api_rc(didread,sock) ));
          #endif
        }
        #endif
        /* at this point, we have didread=-1==got-none, didread=0=connclosed */

        if (didread == 0) /* a zero here can only mean disconnect */
        {
          rc = ps_EDISCO;
          break;
        }
        if (didread > 0)
        {
          totaldone += didread;
          remainingtodo -= didread;
          data += didread;
          TRACE_READ((0,"didread = %d, remaining = %d\n", didread, (int)remainingtodo));
          /* if (remainingtodo == 0) */ /* we break if we read _anything_ */
          {
            rc = 0;
            break;
          }
        }
      } /* if ((revents & POLLIN) != 0) */
      if (!break_pending && CheckExitRequestTriggerNoIO())
      {
        rc = ps_EINTR;
        break;
      }
      if ((++tryloops) > maxloops ) /* timed out */
      {
        rc = 0; //ps_ETIMEDOUT;
        break;
      }
    } /* for (;;) */
  }  /* if (net_init_check_deinit(0,0) == 0) */

  if (bufsz)
    *bufsz = totaldone;
  if (rc != 0 && totaldone)
    rc = 0;
  TRACE_READ((-1,"net_read(...)=>%d%s (totaldone=%u)\n", rc, trace_expand_ps_rc(rc,sock), totaldone));
  return rc;
}

/* --------------------------------------------------------------------- */

int net_write( SOCKET sock, const char *__data, unsigned int *bufsz,
               u32 that_address, int that_port, int iotimeout /*millisecs*/ )
{
  unsigned int totaltodo = ((bufsz)?(*bufsz):(0));
  unsigned int totaldone = 0;
  char *data = (char *)0;
  int rc;

  if (__data)
    *((const char **)&data) = __data; /* drop const without compiler warning*/

  TRACE_WRITE((+1,"net_write(s,%p, %d,'%s:%d',%d)\n",
             data, totaltodo, net_ntoa(that_address),that_port,iotimeout ));

  rc = ps_ENETDOWN;
  if ( sock == INVALID_SOCKET )
    rc = ps_EBADF;
  else if (!data || !bufsz || !that_address || that_port<=0 || that_port>0xffff)
    rc = ps_EINVAL;
  else if ((rc = net_init_check_deinit(0,0)) == 0) /* check */
  {
    rc = 0;
    if (totaltodo)
    {
      int break_pending = CheckExitRequestTriggerNoIO();
      int tryloops = 0, maxloops = 0, mssleep = 0;
      unsigned int remainingtodo = totaltodo;
      unsigned int sendquota = 1500; /* how much to send per send() call */
      rc = ps_stdneterr;

      __calc_timeout_metrics(iotimeout, /* millisecs */
                             &maxloops, /* max number of loops */
                             &mssleep ); /* sleep time per loop */

      *((const char **)&data) = __data; /* drop const */
      #if defined(_TIUSER_)
      {
        struct t_info info;
        if ( t_getinfo( sock, &info ) == -1)
          sendquota = 512;
        else if (info.tsdu > 0)
          sendquota = info.tsdu;
        else if (info.tsdu == -1) /* no limit */
          sendquota = totaltodo;
        else if (info.tsdu == 0) /* no boundaries */
          sendquota = 1500;
        else //if (info.tsdu == -2) /* normal send not supp'd (ever happens?)*/
          sendquota = 512; /* error out later */
      }
      #elif (CLIENT_OS == OS_WIN16)
      if (sendquota > 0x7FFF)  /* 16 bit OS but int is 32 bits */
        sendquota = 0x7FFF;
      #endif
      if (sendquota > INT_MAX)
        sendquota = INT_MAX;

      for (;;)
      {
        int revents = 0;
#if defined(LOOK_NOT_POLL)
        rc = net_look( sock, ps_T_GODATA, &revents, mssleep );
        if (rc != 0)
          break;
        if (revents == ps_T_GODATA)
#else
        rc = 0;
        #if (!defined(_TIUSER_) && !defined(HAVE_POLL_SYSCALL))
        /* poor bastards that don't have poll() need separate checks */
        rc = net_poll1( sock, POLLIN, &revents, 0 );
        #endif
        if (rc == 0 && (revents & (POLLERR|POLLHUP))==0)
        {
          revents = 0; /* no disconnect indication so check POLLOUT now */
          rc = net_poll1( sock, POLLOUT, &revents, mssleep );
        }
        if (rc != 0)
          break;
        if ((revents & (POLLHUP|POLLERR)) != 0)
        {
          rc = ps_EDISCO;
          break;
        }
        if ((revents & POLLOUT) != 0)
#endif
        {
          int written, towrite;
          #if defined(AF_INET)
          struct sockaddr_in saddr;
          memset((void *) &saddr, 0, sizeof(saddr));
          saddr.sin_family = AF_INET;
          saddr.sin_port = htons(((u16)that_port));
          saddr.sin_addr.s_addr = that_address;
          #endif

          written = -1;
          towrite = sendquota;
          if (((unsigned int)towrite) > remainingtodo)
            towrite = remainingtodo;

          #if defined(_TIUSER_)                              //TLI/XTI
          {
            #if 0 /* we don't do connectionless */
            struct t_unitdata udata;
            udata.addr.maxlen = sizeof(saddr);
            udata.addr.len = sizeof(saddr);
            udata.addr.buf = (char *)&saddr;
            udata.opt.maxlen = udata.opt.len = 0;
            udata.opt.buf = (char *)0;
            udata.udata.maxlen = udata.udata.len = towrite;
            udata.udata.buf = data;
            written = t_sndudata(sock, &udata );
            #else
            int flag = 0; //(((remainingtodo - towrite)==0) ? (0) : (T_MORE));
            written = t_snd(sock, data, (unsigned int)towrite, flag );
            #endif

            if (written < 0) /* TBAD[DATA|F|FLAG],TFLOW,TLOOK,TNOSUPPORT,TSYSERR*/
            {
              written = -1;
              debugtli("t_snd", sock);
              if (t_errno != TFLOW) /* for TFLOW we just repeat the loop */
              {
                rc = ps_stdneterr;
                break;
              }
            }
          }
          #elif defined(SOCK_STREAM)      //BSD 4.3 sockets
          if (towrite > 0)
          {
            TRACE_WRITE((+1,"sendto(s,data,%d,0,'%s:%d',%d)\n",towrite,net_ntoa(that_address),that_port,sizeof(saddr) ));
            written = send(sock, data, towrite, 0);
            //written = sendto(sock, data, towrite, 0,
            //                   (struct sockaddr *)&saddr, sizeof(saddr) );
            TRACE_WRITE((-1,"sendto() => %d\n", rc ));
            if (written == 0) /* a zero here can only mean disconnect */
            {
              rc = ps_EDISCO;
              break;
            }
          }
          #endif
          if (written > 0)
          {
            totaldone += written;
            remainingtodo -= written;
            data += written;
            if (remainingtodo == 0)
            {
              rc = 0;
              break;
            }
          }
        } /* if ((revents & POLLOUT) != 0) */
        if (!break_pending && CheckExitRequestTriggerNoIO())
        {
          rc = ps_EINTR;
          break;
        }
        if ((++tryloops) > maxloops ) /* timed out */
        {
          rc = 0; //ps_ETIMEDOUT;
          break;
        }
      } /* for (;;) */
    } /* if (totaltodo) */
  } /* if (net_init_check_deinit(0,0) == 0) */

  *bufsz = totaldone;
  if (rc != 0 && totaldone == totaltodo)
    rc = 0;
  TRACE_WRITE((-1,"net_write() => %d%s, [written=%u]\n", rc,
                            trace_expand_ps_rc(rc,sock), totaldone));
  return rc;
}

/* ======================================================================== */
/* CONVERSION                                                               */
/* ======================================================================== */

// a) u32 -> in_addr.s_addr ->inet_ntoa is a hassle;
// b) BSD specs in_addr, but some implementations want a u_long;
// c) it works around context issues.
// d) it works everywhere, even when there is no net api loaded
const char *net_ntoa(u32 addr)
{
  static char buff[sizeof("255.255.255.255  ")];
  char *p = (char *)(&addr);
  sprintf( buff, "%d.%d.%d.%d",
      (p[0]&255), (p[1]&255), (p[2]&255), (p[3]&255) );
  return buff;
}

/* --------------------------------------------------------------------- */

/* courtesy of netware port :) */
static int _inet_atox( const char *cp, /* stringified address (source) */
                       u32 /*struct in_addr*/ *destp, /* destination buffer */
                       int minparts, /* must have at least this many parts */
                       int addrtype ) /* 'h':inet_aton(), 'n':inet_network()*/
{
  int err = 1;

  if (cp)
  {
    int parts = 0, bracket = 0;
    unsigned long maxval = 0xfffffffful;
    unsigned long buf[4];

    if ( *cp == '[' && addrtype != 'n')
    {
      bracket = 1;
      cp++;
    }

    buf[0] = 0; err = 0;
    while (*cp && !err)
    {
      register unsigned long val = 0;
      unsigned int radix = 10, len = 0;

      if ( addrtype == 'n' ) /* for inet_network() each component can */
        maxval = 0xff;       /* only be 8 bits */

      if ( *cp == '0' )
      {
        radix = 8;
        cp++;
        len = 1;
      }
      if ( *cp == 'X' || *cp == 'x')
      {
        radix = 16;
        cp++;
        len = 0;
      }
      while (!err)
      {
        register int c = (int)*cp;
        if (c >= '0' && c <= '9') /* ebcdic safe */
          c -= '0';
        else if (radix == 16 && c >= 'A' && c <= 'F') /* ebcdic safe */
          c = 10 + (c - 'A');
        else if (radix == 16 && c >= 'a' && c <= 'f') /* ebcdic safe */
          c = 10 + (c - 'a');
        else
          break; /* dot or bad char */
        if ( c < 0 || ((unsigned int)c) >= radix )
          err = 1;
        else
        {
          unsigned long newval = (val * radix) + c;
          if (newval < val) /* wrapped */
            err = 1;
          else if (newval > maxval)
            err = 1;
          else
          {
            val = newval;
            cp++;
            len++;
          }
        }
      } /* while (!err) */

      if (!err)
      {
        if (!( parts < 4 )) /* already did 4 */
          err = 1;
        else if (parts && buf[parts-1]>0xff) /* completed parts must be <=0xff */
          err = 1;
        else
        {
          buf[parts++] = val;
          maxval >>= 8; /* for next round */
          if ( *cp != '.' )
            break;
          cp++;
          if (!*cp)
            err = 1;
        }
      }
    } /* while (*cp && !err) */

    if (err || parts == 0 || parts < minparts)
      err = 1;
    else if ( bracket && *cp !=']' ) /* unmatched bracket */
      err = 1;
    else if (*cp && *cp!='\n' && *cp!='\r' && *cp!='\t' && *cp!=' ')
      err = 1;
    else if (destp)
    {
      register int n;
      u32 addr /*inp->s_addr*/ = 0;
      if (addrtype == 'n')                   /* inet_network() */
      {
        for ( n = 0; n < parts; n++ )
        {
          addr /*inp->s_addr*/ <<= 8;
          addr /*inp->s_addr*/ |= (buf[n] & 0xff);
        }
      }
      else                                   /* inet_aton() or inet_addr() */
      {
        char *p = (char *)(&(addr/*inp->s_addr*/));
        if (parts == 1)
          maxval = buf[0];
        else if (parts == 2)
          maxval = (buf[0]<<24) | buf[1];
        else if (parts == 3)
          maxval = (buf[0]<<24) | (buf[1]<<16) | buf[2];
        else if (parts == 4)
          maxval = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
        addr /*inp->s_addr*/ = 0;
        for ( n= 0; n < 4; n++ )
          *p++ = (char)((maxval >> ((3-n)<<3)) & 0xff);
      }
      *destp = addr;
    }
  } /* if (cp) */

  if (err)
    err = -1;
  return err; /* return 0 if valid, -1 if not */
}

/* unlike inet_aton() this _requires_ the address to have 4 parts */
int net_aton( const char *cp, u32 *addrp )
{
  if (_inet_atox( cp, addrp, 4, 'h' ) == 0)
    return 0;
  return ps_EINVAL;
}

/* ======================================================================== */
/* NETDB                                                                    */
/* ======================================================================== */

int net_gethostname(char *buffer, unsigned int len)
{
  int rc = ps_EINVAL;
  if (buffer && ((int)len) > 0)
  {
    rc = ps_ENOSYS;
    #if (defined(_TIUSER_) || defined(SOCK_STREAM))
    rc = ps_ENETDOWN;
    if (is_netapi_callable())
    {
      char scratch[256]; /* MAXHOSTNAMELEN */
      rc = ps_stdneterr;
      if (gethostname(scratch, sizeof(scratch)) == 0)
      {
        /* BSD man page for gethostname(3) sez: "The returned name is
           null-terminated, unless insufficient space is provided."
        */
        scratch[sizeof(scratch)-1] = '\0';
        #if (CLIENT_OS == OS_WIN32)|| (CLIENT_OS == OS_WIN16)
        {
          /* <quote> If no local host name has been configured gethostname
             must succeed and return a token host name that gethostbyname
             or WSAAsyncGetHostByName can resolve. </quote>
             How stupid can anymore get?
          */
          //hostname is REG_SZ in
          //HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Hostname
          #define MSCRAP_HOSTNAME_TOKEN "-" /* aforementioned "token" hostname */
          if (strcmp( scratch, MSCRAP_HOSTNAME_TOKEN ) == 0)
            scratch[0] = '\0';
        }
        #endif /* win32 */

        rc = ps_ENOENT;
        if (scratch[0])
        {
          strncpy( buffer, scratch, len );
          buffer[len-1] = '\0';
          rc = 0;
        }
      } /* if gethostname() == 0 */
    } /* if (is_netapi_callable()) */
    #endif /* (defined(_TIUSER_) || defined(SOCK_STREAM)) */
  } /* if (buffer && ((int)len) > 0) */
  return rc;
}

/* --------------------------------------------------------------------- */

int net_resolve( const char *hostname, u32 *addr_list, unsigned int *max_addrs)
{
  int rc;

  TRACE_NETDB((+1,"net_resolve('%s', ...)\n", ((hostname)?(hostname):("(null)")) ));

  rc = ps_EINVAL;
  if (hostname && addr_list && max_addrs && *max_addrs)
  {
    unsigned int len = 0, dot_count = 0, digit_count = 0;
    u32 addr; char buffer[128];

    rc = 0;
    while (*hostname == ' ' || *hostname == '\t')
    {
      hostname++;
    }
    while (*hostname && (*hostname != ' ' && *hostname != '\t'))
    {
      if (*hostname == '.')
        dot_count++;
      else if (*hostname >= '0' && *hostname <= '9')
        digit_count++;
      if (len >= (sizeof(buffer)-2))
      {
        rc = ps_EINVAL; /* EMSGSIZE */
        break;
      }
      buffer[len++] = (char)*hostname++;
    }
    buffer[len] = '\0';

    if (len == 0)
    {
      rc = ps_EINVAL; /* bad format */
    }
    else if (rc == 0)
    {
      TRACE_NETDB((0,"buff='%s' len=%d, dot_count=%d, digit_count=%d\n", buffer, len, dot_count, digit_count ));
      if (len == (dot_count+digit_count)) /*its an IP address*/
      {
        rc = ps_EINVAL;
        if (dot_count == 3) /* 3 dots */
        {
          if (net_aton(buffer, &addr) != 0)
            addr = 0xffffffff; /* can't resolve a broadcast addr, so this is ok */
          TRACE_NETDB((0,"1) inet_aton() => 0x%08x\n", addr ));
          if (addr != 0xffffffff)
          {
            addr_list[0] = addr;
            *max_addrs  = 1;
            rc = 0;
          }
        }
      }
      else if (len > 13 && strcmp(&buffer[len-13],".in-addr.arpa")==0)
      {
        rc = ps_EINVAL; /* assume mangled address */
        if (dot_count == 5 /* 3 in address plus 2 in ".in-addr.arpa" */
         && ((len-13) == (3+digit_count)) ) /* the rest is an IP address */
        {
          buffer[len-13] = '\0';
          if (net_aton(buffer, &addr) != 0)
            addr = 0xffffffff; /* can't resolve a broadcast addr, so this is ok */
          TRACE_NETDB((0,"2) inet_aton() => 0x%08x\n", addr ));
          if (addr != 0xffffffff)
          {
            addr_list[0] = ((addr>>24)&0xff)|((addr>>8)&0xff00)|
                           ((addr&0xff)<<24)|((addr&0xff00)<<8);
            *max_addrs  = 1;
            rc = 0;
          }
        }
      }
      else /* gethostbyname() */
      {
        rc = ps_ENETDOWN;
        if (is_netapi_callable())
        {
          #if (defined(_TIUSER_) || defined(SOCK_STREAM))
          struct hostent *hp;
          TRACE_NETDB((+1,"gethostbyname('%s')\n", buffer ));
          hp = gethostbyname(buffer);
          TRACE_NETDB((-1,"gethostbyname('%s') => %p\n", buffer, hp ));
          rc = ps_ENOENT; /* assume not found */
          if (hp)
          {
            unsigned int pos, foundcount = 0;
            for ( pos = 0; hp->h_addr_list[pos]; pos++)
            {
              unsigned int dupcheck = 0;
              addr = *((u32 *)(hp->h_addr_list[pos]));
              while (dupcheck < foundcount)
              {
                if (addr == addr_list[dupcheck])
                  break;
                dupcheck++;
              }
              if (!(dupcheck < foundcount))
              {
                addr_list[foundcount++] = addr;
                if (foundcount == *max_addrs)
                  break;
              }
            }
            if (foundcount)
            {
              *max_addrs = foundcount;
              rc = 0; /* success */
            }
            TRACE_NETDB((0,"found %u addresses\n", foundcount ));
          }
          else /* hp == NULL */
          {
            #if !defined(_TIUSER_) && defined(HOST_NOT_FOUND) && defined(NO_ADDRESS)
            int i = h_errno;
            if (i < 0) /* NETDB_INTERNAL (see errno). Not always defined */
              rc = ps_stdsyserr;
            else if (i == HOST_NOT_FOUND) /* Authoritative Answer Host not found */
              rc = ps_ENOENT;
            #if 0
            else if (i == TRY_AGAIN) /* Non-Authoritive Host not found, or SERVERFAIL */
              rc = ps_EAGAIN;
            else if (i == NO_RECOVERY) /* Non recoverable errors, FORMERR, REFUSED, NOTIMP */
              rc = ps_ENOENT;
            #endif
            else if (i == NO_ADDRESS) /* Valid name, no data record of requested type */
              rc = ps_ENODATA;
            /* default is ps_ENOENT */
            #endif
          }
          #else /* _TIUSER_ || SOCK_STREAM */
          rc = ps_ENOSYS;
          #endif /* _TIUSER_ || SOCK_STREAM */
        } /* if (is_netapi_callable()) */
      } /* name or address */
    } /* (rc == 0) */
  } /* if (hostname && addr_list && max_addrs && *max_addrs) */

  TRACE_NETDB((-1,"net_resolve(...) =>%d%s\n", rc, trace_expand_ps_rc(rc,-1) ));
  return rc;
}

/* --------------------------------------------------------------------- */

