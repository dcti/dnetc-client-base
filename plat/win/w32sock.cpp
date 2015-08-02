/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Shim layer between Winsock DLL and application.
 * The only functions in the Winsock 1.x spec that are not in this file are
 * the WSAAsyncXXX() calls (WSAxyzBlockingXYZ() functions are available though)
 *
 * While this shim is essential for Watcom 386 flat memory model extensions
 * for win16, is also used by win32 clients to allow the client to
 * run on systems without winsock.dll/wsock32.dll/ws2_32.dll. It also
 * compensates for bugs where appropriate (search for "BUG BUG").
 * [Winsock2 WSACleanup() BUG BUG not covered here (it affects
 * any application using winsock2 not just one using this module):
 * The first SendMessage(WM_USER ) to window after WSACleanup() will 
 * crash the application. (it appears WSACleanup() leaves a dangling
 * window message hook). Always make sure WSAStartup()/WSACleanup()
 * are called before/after windows are created/destroyed.]
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This module is known to work with 16bit Novell, Microsoft and Trumpet
 * stacks, as well as the 32bit Alpha and x86 Microsoft stacks.
 * [Novell stack BUG BUG: recv(,,MSG_PEEK) removes data from the queue] 
 *
 * Little endian byte format is assumed for HTON[LS]()/NTOH[LS]()
 *
 * ---------------------------------------------------------------------
 * When used with Watcom 386 flat model extensions, the winsock.h header
 * must be #included with the following wrapper:
 *   #include <windows.h> // use default shimming/redefs
 *   #ifdef __WINDOWS_386__
 *     #undef FAR
 *     #define FAR
 *   #endif
 *   #include <winsock.h> // use the 16 bit version
 *   #ifdef __WINDOWS_386__
 *     #undef FAR
 *     #define FAR far
 *     #undef FD_ISSET // macro applies a FAR cast that results in truncation
 *     #define FD_ISSET(fd, set) __WSAFDIsSet(fd, set) //...so discard the cast
 *   #endif
*/
const char *w32sock_cpp(void) {
return "@(#)$Id: w32sock.cpp,v 1.3 2002/10/09 22:22:15 andreasb Exp $"; }

#include <windows.h>
#include "w32sock.h" // <windows.h> and <winsock.h> as documented above.

#if defined(_M_ALPHA)
  #define WINSOCK_DLL_FILENAME "WS2_32.DLL"
  #undef PASCAL
  #define PASCAL
#elif defined(__WINDOWS_386__)
  #define WINSOCK_DLL_FILENAME "WINSOCK.DLL"
  #undef FAR
  #define FAR
#elif defined(__WIN32__) || defined(WIN32)
  static char *__GetWinsockDLLFilename(void)
  {
    static char * wslib = NULL;
    if (wslib == NULL)
    {
      char *libname;
      /* original win95, win95a, win95b, win95 with winsock2 either don't 
         have winsock2, or have a broken winsock2 (BUG BUG), that under 'some 
         circumstances' incorrectly fails socket() with WSAESOCKTNOSUPPORT 
         (specified socket type is not supported in address family)
      */
      OSVERSIONINFO osver;
      osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
      GetVersionEx(&osver);
      if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
        osver.dwMajorVersion += 20;
      else if (VER_PLATFORM_WIN32s == osver.dwPlatformId)
      {
        DWORD dwVersion = GetVersion();
        osver.dwMajorVersion = LOBYTE(LOWORD(dwVersion));
        osver.dwMinorVersion = HIBYTE(LOWORD(dwVersion));  
      }
      libname = "WSOCK32.DLL"; /* default for win95 */
      if (((osver.dwMajorVersion * 100)+osver.dwMinorVersion) > 400) 
      {
        libname = "WS2_32.DLL"; /* default for win98 and NT */
        if (GetModuleHandle(libname) == NULL)
        {
          OFSTRUCT ofstruct;
          ofstruct.cBytes = sizeof(ofstruct);
          #ifndef OF_SEARCH
          #define OF_SEARCH 0x0400
          #endif
          if ( OpenFile( libname, &ofstruct, OF_EXIST|OF_SEARCH) == HFILE_ERROR )
            libname = "WSOCK32.DLL"; /* no winsock2 installed */
        }
      }
      wslib = libname;
    }
    return wslib;
  }
  #define WINSOCK_DLL_FILENAME __GetWinsockDLLFilename()
#else
  #define WINSOCK_DLL_FILENAME "WINSOCK.DLL"
#endif


/* ==================================================================== */

static struct
{
  int initlevel;
  int lasterror;
  FARPROC blockinghook; /* windows386 use */
} w32sockstatics = {0,0,0};

//import winsock symbol should always be able to find the lib
//because WSAStartup calls __w32sockinitdeinit which loads the lib
static FARPROC ImportWinsockSymbol(LPCSTR lpszProcName)
{
  FARPROC proc = (FARPROC)0;
  if (w32sockstatics.initlevel > 0)
  {
    HMODULE ws = GetModuleHandle(WINSOCK_DLL_FILENAME /* "WINSOCK" */);
    if (ws)
      proc = GetProcAddress(ws, lpszProcName);
  }
  return proc;
}

static HINSTANCE __LoadLibrary( LPCSTR lpszLibName )
{
  /* Ensure that the library exists before trying to open it */
  HINSTANCE hinst = 0;
  HMODULE ws;
  OFSTRUCT ofstruct;
  ofstruct.szPathName[0] = '\0';

  /* [By user request] This next if is primarily for win16 that should 
     use the same winsock thats already loaded (if there is one loaded). 
  */
  if (( ws = GetModuleHandle(WINSOCK_DLL_FILENAME) ) != ((HMODULE)0))
  {
    unsigned int len = GetModuleFileName(ws, ofstruct.szPathName,
                       sizeof(ofstruct.szPathName));
    if (len >= sizeof(ofstruct.szPathName))
      len = sizeof(ofstruct.szPathName)-1;
    ofstruct.szPathName[len] = '\0';
  }
  else
  {
    ofstruct.cBytes = sizeof(ofstruct);
    #ifndef OF_SEARCH
    #define OF_SEARCH 0x0400
    #endif
    if ( OpenFile( lpszLibName, &ofstruct, OF_EXIST|OF_SEARCH) == HFILE_ERROR )
      ofstruct.szPathName[0] = '\0';
  }
  if (ofstruct.szPathName[0])
  {
    hinst = LoadLibrary( ofstruct.szPathName );
    if (hinst <= ((HINSTANCE)(32)))
      hinst = 0;
  }
  return hinst;
}

static int __w32sockInitDeinit(int init_or_deinit )
{
  HINSTANCE hinstWinsock = NULL;

  if (init_or_deinit == 0)   /* getstate */
  {
    if (w32sockstatics.initlevel <= 0)
      return 0;
    return (GetModuleHandle(WINSOCK_DLL_FILENAME /*"WINSOCK"*/) != NULL);
  }
  else if (init_or_deinit < 0)   /* deinitialize */
  {
    if ((--w32sockstatics.initlevel)==0)
    {
      if (hinstWinsock)
      {
        FreeLibrary(hinstWinsock);
        hinstWinsock = NULL;
      }
    }
  }
  else if (init_or_deinit > 0)   /* initialize */
  {
    if ((++w32sockstatics.initlevel)==1)
    {
      if (hinstWinsock == NULL)
      {
        hinstWinsock = __LoadLibrary(WINSOCK_DLL_FILENAME);
        if (hinstWinsock < ((HINSTANCE)32))
        {
          hinstWinsock = NULL;
          --w32sockstatics.initlevel;
          return 0;
        }
      }
    }
  }
  return 1;
}

/* ---------------------------------------- */

#if defined(__WATCOMC__) /* we can inline these suckers */
    unsigned short __ntohs( unsigned short );
    unsigned short __htons( unsigned short );
    #pragma aux __htons = 0x86 0xc4 /* xchg al,ah */ \
                        parm [ax] value[ax] modify exact[ax]
    #pragma aux (__htons) __ntohs;  /* ntohs is identical to htons */
    #ifdef __386__
    unsigned long  __ntohl( unsigned long );
    unsigned long  __htonl( unsigned long );
    #pragma aux __htonl = 0x86 0xc4 0xc1 0xc0 0x10 0x86 0xc4 \
                        parm [eax] value[eax] modify exact[eax]
                     /* xchg al,ah  rol eax,16   xchg al,ah */
    #pragma aux (__htonl) __ntohl;  /* ntohl is identical to htonl */
    #else
    #define __ntohl(x) ((((x)&0xff)<<24) | (((x)>>24)&0xff) | \
                      (((x)&0xff00)<<8) | (((x)>>8)&0xff00))
    #define __htonl(x) __ntohl(x)                      
    #endif
#else
    #define __ntohs(x) ((((x)&0xff)<<8) | (((x)>>8)&0xff))
    #define __htons(x) __ntohs(x)
    #define __ntohl(x) ((((x)&0xff)<<24) | (((x)>>24)&0xff) | \
                       (((x)&0xff00)<<8) | (((x)>>8)&0xff00))
    #define __htonl(x) __ntohl(x)                      
#endif

/* ---------------------------------------- */

u_short PASCAL FAR htons(u_short s)
{ return (u_short) ((((s)&0xff)<<8) | (((s)>>8)&0xff)); }
u_short PASCAL FAR ntohs(u_short s)
{ return (u_short) htons(s); }
u_long PASCAL FAR htonl(u_long l)
{ return (u_long) ((((l)&0xff)<<24) | (((l)>>24)&0xff) | (((l)&0xff00)<<8) | (((l)>>8)&0xff00)); }
u_long PASCAL FAR ntohl(u_long l)
{ return (u_long) htonl(l); }

/* ---------------------------------------- */

static char *inet_ntoa_r(char *buff, u_long inaddr )
{
  if ( buff )
  {
    unsigned int i;
    char *a = (char *)(&inaddr), *b = buff;

    for ( i=0; i<4; i++ )
    {
      register unsigned int r, c = (((unsigned int)(*a++)) & 255 );
      if ( i )
        *b++ = '.';
      if ( (r = c / 100) != 0 )
        *b++ = (char)( r + '0' );
      r <<= 8;
      if ( (r |= ((c % 100)/10)) != 0)
        *b++ = (char)( (r & 0xff) + '0' );
      *b++ = (char)((c%10) + '0');
    }
    *b = 0;
  }
  return buff;
}


static char *__inet_ultoa( u_long addr )
{
  static char buff[18];
  return inet_ntoa_r( buff, addr );
}

char FAR * PASCAL FAR inet_ntoa( struct in_addr addr )
{ return __inet_ultoa( *((u_long *)(&addr)) ); }

/* ---------------------------------------- */

void PASCAL FAR WSASetLastError(int iError)
{
  w32sockstatics.lasterror = iError;  
  if (iError == 0)
  {
    FARPROC __proc = ImportWinsockSymbol( "WSASetLastError" );
    if (__proc)
    {
      #if defined(__WINDOWS_386__)
      _Call16(__proc,"w",iError);
      #else
      (*((void (PASCAL FAR *)(int))(__proc)))(iError);
      #endif
    }
  }
  return;
}

/* ---------------------------------------- */

int PASCAL FAR WSAGetLastError(void)
{
  int rc = WSANOTINITIALISED;
  if (w32sockstatics.initlevel > 0)
  {
    rc = w32sockstatics.lasterror;
    if (rc == 0) /* w32sockstatics.lasterror is clear */
    {
      FARPROC __proc = ImportWinsockSymbol( "WSAGetLastError" );
      rc = WSAEOPNOTSUPP;
      if (__proc)
      {
        #if defined(__WINDOWS_386__)
        rc = (0xffff & (_Call16(__proc,"")));
        #else
        rc = (*((int (PASCAL FAR *)(void))(__proc)))();
        #endif
        /* contrary to expectations WSAGetLastError() _does_ clear the error */
        w32sockstatics.lasterror = rc;
      }
    }
  }
  return rc;
}

/* ---------------------------------------- */

int PASCAL FAR gethostname(char FAR * name, int namelen)
{
  FARPROC __gethostname = ImportWinsockSymbol( "gethostname" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (!name || namelen < 1)
  {
    w32sockstatics.lasterror = WSAEINVAL;
    return -1;
  }
  if (namelen < 2)
  {
    *name = '\0';
    return 0;
  }
  if (__gethostname)
  {
    #if defined(__WINDOWS_386__)
    /* gethostname() is 'magic' and tests both winsock and extender. */
    int rc = -1; char buf[256];
    DWORD alias = AllocAlias16( (void *)(&buf[0]) );
    if (!alias)
    {
      w32sockstatics.lasterror = WSAENOBUFS;
      return -1; 
    }
    buf[0]=0;
    if ((0xffff & (_Call16(__gethostname,"dw", alias, sizeof(buf))))==0)
      rc = 0;
    FreeAlias16( alias );
    if (rc == 0)
    {
      buf[sizeof(buf)-1]=0;
      for (rc=0;(buf[rc] && rc<(namelen-1));rc++)
       *name++ = buf[rc];
      *name++ = 0;
      rc = 0;
    }
    return rc;
    #else
    HINSTANCE hInst = GetModuleHandle("kernel32");
    if (hInst)
    {
      /* only on Win2k and WinME */ 
      typedef BOOL (WINAPI *_GetComputerNameExA_T)(DWORD, LPTSTR, LPDWORD);
      _GetComputerNameExA_T _GetComputerNameExA = (_GetComputerNameExA_T)
                            GetProcAddress(hInst,"GetComputerNameExA");
      if (_GetComputerNameExA)
      {
        char buffer[256+1]; DWORD maxlen = sizeof(buffer);
        if ((*_GetComputerNameExA)( 7 /*ComputerNamePhysicalDnsFullyQualified*/,
                                   buffer, &maxlen ))
        {
          int i;
          for (i = 0; buffer[i] && (DWORD)i < maxlen && i < (namelen-1); i++)
            *name++ = buffer[i];
          *name++ = '\0';
          return 0;
        }
        w32sockstatics.lasterror = WSAEOPNOTSUPP; /* what should we do? */
        return -1;
      }
    }
    return (*((int (PASCAL FAR *)(char FAR *,int))(__gethostname)))(name, namelen);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

SOCKET PASCAL FAR socket(int domain, int type, int protocol)
{
  FARPROC __socket = ImportWinsockSymbol( "socket" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__socket)
  {
    #if defined(__WINDOWS_386__)
    SOCKET ns = (SOCKET)(0xffff & (_Call16(__socket,"www",domain,type,protocol)));
    return ((ns == ((INVALID_SOCKET)&0xffff))?(INVALID_SOCKET):(ns));
    #else
    /* windows 95 BUG BUG: internally sockets are 16 bits, but because SOCKET
       is unsigned, the return value is not sign extended to the full
       32bit INVALID_SOCKET.
       (this is unrelated to the Win95+Winsock2+WSAESOCKTNOSUPPORT bug)
    */
    FARPROC __seterrproc = ImportWinsockSymbol( "WSASetLastError" );
    FARPROC __geterrproc = ImportWinsockSymbol( "WSAGetLastError" );
    if (__seterrproc && __geterrproc)
    {
      SOCKET sock;
      int preverr = ((*((int (PASCAL FAR *)(void))(__geterrproc)))());
      (*((void (PASCAL FAR *)(int))(__seterrproc)))(0);

      sock = (*(SOCKET (PASCAL FAR *)(int,int,int))(__socket))(domain,type,protocol);
      if (sock != INVALID_SOCKET) /* full 32 bit invalid socket */
      {
        if ((sock & 0xffff) == (INVALID_SOCKET & 0xffff))
        {
          if (((*((int (PASCAL FAR *)(void))(__geterrproc)))()) != 0)
            sock = INVALID_SOCKET;
        }
      }
      if (sock != INVALID_SOCKET)
        (*((void (PASCAL FAR *)(int))(__seterrproc)))(preverr);
      return sock;
    }
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return INVALID_SOCKET;
}

/* ---------------------------------------- */

int PASCAL FAR shutdown(SOCKET s, int how)
{
  FARPROC __shutdown = ImportWinsockSymbol( "shutdown" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__shutdown)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16(__shutdown,"ww",s,how) & 0xffff) == 0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(SOCKET,int))(__shutdown)))(s,how);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR closesocket(SOCKET s)
{
  FARPROC __closesocket = ImportWinsockSymbol( "closesocket" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__closesocket)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16(__closesocket,"w",s) & 0xffff) == 0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(SOCKET))(__closesocket)))(s);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

static int __getpeersockname( const char *procname, SOCKET s,
                              struct sockaddr FAR *name, int FAR * namelen )
{
  FARPROC __proc = ImportWinsockSymbol( procname );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    if (name && namelen)
    {
      WORD _namelen = (WORD)(*namelen);
      int rc = (((_Call16(__proc,"wpp",s,name,&_namelen)&0xffff)==0)?(0):(-1));
      *namelen = _namelen;
      return rc;
    }
    #else
      return (*((int (PASCAL FAR *)(SOCKET, struct sockaddr FAR *, int FAR *))
                                   (__proc)))( s, name, namelen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR getpeername(SOCKET s, struct sockaddr FAR *name, int FAR * namelen)
{
  return __getpeersockname( "getpeername", s, name, namelen );
}

/* ---------------------------------------- */

int PASCAL FAR getsockname(SOCKET s, struct sockaddr FAR *name, int FAR * namelen)
{
  return __getpeersockname( "getsockname", s, name, namelen );
}

/* ---------------------------------------- */

SOCKET PASCAL FAR accept(SOCKET s, struct sockaddr FAR *addr,
                          int FAR *addrlen)
{
  FARPROC __accept = ImportWinsockSymbol( "accept" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__accept)
  {
    #if defined(__WINDOWS_386__)
    if (addr && addrlen)
    {
      WORD _addrlen = (WORD)(*addrlen);
      SOCKET ns = (0xffff & _Call16(__accept, "wpp", s, addr, &_addrlen));
      *addrlen = _addrlen;
      return ((ns == ((INVALID_SOCKET)&0xffff))?(INVALID_SOCKET):(ns));
    }
    return -1;
    #else
    return (*((SOCKET (PASCAL FAR *)(SOCKET, struct sockaddr FAR *, int FAR *))
         (__accept)))( s, addr, addrlen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return INVALID_SOCKET;
}

/* ---------------------------------------- */

int PASCAL FAR bind(SOCKET s, const struct sockaddr FAR *addr, int addrlen)
{
  FARPROC __proc = ImportWinsockSymbol( "bind" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16(__proc, "wpw", s, addr, addrlen) & 0xffff)==0)?(0):(-1));
    #else
    return ((*((int (PASCAL FAR *)(SOCKET, const struct sockaddr FAR *, int))
                                 (__proc)))( s, addr, addrlen ));
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR connect(SOCKET s, const struct sockaddr FAR *addr, int addrlen)
{
  FARPROC __proc = ImportWinsockSymbol( "connect" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    int rc; WSASetLastError(0);
    #if defined(__WINDOWS_386__)
    rc = (((_Call16(__proc, "wpw", s, addr, addrlen) & 0xffff)==0)?(0):(-1));
    #else
    /* Winsock2 BUG BUG: non-blocking connect() may return -1 even on success*/
    /* First observed on alpha NT 4, but its in x86 Win2K as well.        */
    /* (also if using 16bit winsock on win2k since winsock.dll shims ws2) */
    rc = ((*((int (PASCAL FAR *)(SOCKET, const struct sockaddr FAR *, int))
                                 (__proc)))( s, addr, addrlen ));
    #endif
    if (rc != 0 && WSAGetLastError() == 0) /* catch the above mentioned bug */
      rc = 0;                              /* _here_. (ie, for win16 as well)*/
    return rc;
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR getsockopt(SOCKET s, int level, int optname,
                                           char FAR * optval, int FAR *optlen)
{
  FARPROC __proc = ImportWinsockSymbol( "getsockopt" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    short len = (short)(*optlen);
    if ((0xffff & _Call16(__proc, "wwwpp", s, level, optname, optval, &len))==0)
    {
      *optlen = (((int)len) & 0xffff);
      return 0;
    }
    return -1;
    #else
    return (*((int (PASCAL FAR *)(SOCKET, int, int, char FAR *, int FAR *))
                             (__proc)))( s, level, optname, optval, optlen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR setsockopt(SOCKET s, int level, int optname,
                             const char FAR * optval, int optlen)
{
  FARPROC __proc = ImportWinsockSymbol( "setsockopt" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return (((0xffff & _Call16(__proc, "wwwpw", s, level, 
                      optname, optval, optlen))==0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, int, int, const char FAR *, int ))
                             (__proc)))( s, level, optname, optval, optlen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR listen(SOCKET s, int backlog)
{
  FARPROC __proc = ImportWinsockSymbol( "listen" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return (((0xffff & _Call16(__proc, "ww", s, backlog ))==0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, int ))(__proc)))( s, backlog );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR recvfrom(SOCKET s, char FAR * buf, int len, int flags,
                         struct sockaddr FAR *from, int FAR * fromlen)
{
  FARPROC __proc = ImportWinsockSymbol( "recvfrom" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    WORD _fromlen = (WORD)(*fromlen);
    short rc;
    if (len > 0x7fff)
      len = 0x7fff;
    rc = (short)_Call16(__proc, "wpwwpp", s, buf, len, flags, from, &_fromlen);
    *fromlen = _fromlen;
    return ((rc<0)?(-1):(rc));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, char FAR *, int, int,
                 struct sockaddr FAR *, int FAR * ))(__proc)))
                 ( s, buf, len, flags, from, fromlen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR sendto(SOCKET s, const char FAR * buf, int len, int flags,
                       const struct sockaddr FAR *to, int tolen)
{
  FARPROC __proc = ImportWinsockSymbol( "sendto" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    short rc;
    if (len > 0x7fff)
      len = 0x7fff;
    rc = (short)_Call16(__proc, "wpwwpw", s, buf, len, flags, to, tolen );
    return ((rc<0)?(-1):(rc));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, const char FAR *, int, int,
                 const struct sockaddr FAR *, int ))(__proc)))
                 ( s, buf, len, flags, to, tolen );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

#if defined(__WINDOWS_386__)

struct fd_set_short {
  unsigned short fd_count;               /* how many are SET? */
  unsigned short fd_array[1];            /* an array of SOCKETs */
};

static void __map_int_to_short_fdset( fd_set FAR *argfds )
{
  if (argfds)
  {
    struct fd_set_short *shortfds;
    unsigned int i, count;

    shortfds = (struct fd_set_short *)argfds;
    count = (unsigned int)argfds->fd_count;
    shortfds->fd_count = (short)count;
    for (i=0;i<count;i++)
      shortfds->fd_array[i] = (short)argfds->fd_array[i];
  }
  return;
}

static void __map_short_to_int_fdset( fd_set FAR *argfds )
{
  if (argfds)
  {
    struct fd_set_short *shortfds;
    unsigned int i, count;

    shortfds = (struct fd_set_short *)argfds;
    count = (unsigned int)shortfds->fd_count;
    for (i=count;i>0;)
    {
      --i;
      argfds->fd_array[i] =
                (0xffff & ((unsigned int)(shortfds->fd_array[i])));
    }
    argfds->fd_count = count;
  }
  return;
}
#endif

/* ---------------------------------------- */

int PASCAL FAR __WSAFDIsSet(SOCKET s, fd_set FAR *fds)
{
  w32sockstatics.lasterror = 0; /* <- no override */
  if (fds && s != INVALID_SOCKET)
  {
    #if defined(__WINDOWS_386__)
    unsigned int i, count = (((unsigned int)(fds->fd_count)) & 0xffff);
    for (i=0;i<count;i++)
    {
      if (fds->fd_array[i] == s)
        return 1;
    }
    #else
    FARPROC __proc = ImportWinsockSymbol( "__WSAFDIsSet" );
    if (__proc)
    {
      return (*((int (PASCAL FAR *)(SOCKET, fd_set FAR *))(__proc)))( s, fds );
    }
    w32sockstatics.lasterror = WSAEOPNOTSUPP;
    #endif
  }
  return 0;
}

/* ---------------------------------------- */

int PASCAL FAR select(int nfds, fd_set FAR *readfds, fd_set FAR *writefds,
               fd_set FAR *exceptfds, const struct timeval FAR *timeout)
{
  FARPROC __proc = ImportWinsockSymbol( "select" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    short rc;
    __map_int_to_short_fdset( readfds );
    __map_int_to_short_fdset( writefds );
    __map_int_to_short_fdset( exceptfds );
    rc = (short)((0xFFFF) & (_Call16(__proc, "wpppp", nfds,
                        readfds, writefds, exceptfds, timeout )));
    __map_short_to_int_fdset( readfds );
    __map_short_to_int_fdset( writefds );
    __map_short_to_int_fdset( exceptfds );
    return ((rc<0)?(-1):(rc));
    #else
    /* WINSOCK2 select() BUG BUG for a _single_ (blocking only?) socket
       for timeout != NULL and timeout != 0.
       sleeps for as long as the timeout irrespective of whether there 
       is an event pending or not.
       Possible solutions:
       - temporarily create a second _connected_ socket, 
       - spin on a zero timeout
       - convert to non-blocking first, restore blocking afterwards.
         [how do we detect that it is not already non-blocking?]
    */
    return (*((int (PASCAL FAR *)(int, fd_set FAR *, fd_set FAR *,
         fd_set FAR *, const struct timeval FAR * ))(__proc)))
                            ( nfds, readfds, writefds, exceptfds, timeout );
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

unsigned long PASCAL FAR inet_addr(const char FAR * cp)
{
  FARPROC __inet_addr = ImportWinsockSymbol( "inet_addr" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__inet_addr)
  {
    #if defined(__WINDOWS_386__)
    return (unsigned long)_Call16(__inet_addr, "p", cp );
    #else
    return (*((int (PASCAL FAR *)(const char FAR *))(__inet_addr)))(cp);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return 0xfffffffful;
}

/* ---------------------------------------- */

#if defined(__WINDOWS_386__)
/* copy 16:16  FAR * FAR * (eg hostent.h_addr_list, h_aliases, etc) to local */
/* buffers. result array will always be NULL terminated */
static char **__copy_farpfarp( const void *__src16, /* char far * far * */
                               char *buffer, unsigned int buflen,
                               char *elemarray[], unsigned int maxelems,
                               int elemsize /* -1=asciiz */ ) 
{
  unsigned int index = 0;
  const char far *cfp = (const char far *)0;

  if (__src16 && maxelems > 1)
    cfp = (const char far *)MK_FP32((void *)__src16);

  if (cfp)
  {
    while (index < (maxelems-1))
    {
      unsigned int len;
      void *vp; char far *ufp;
      char *cp = (char *)&vp;
      for (len=0; len < sizeof(vp); len++)
        *cp++ = (char)(*cfp++);
      if (!vp) /* found NULL element (the terminator) */
        break;
      ufp = (char far *)MK_FP32(vp);
      if (!ufp) /* should not happen */
        break;
      len = elemsize;
      if (elemsize <= 0) /* asciiz. so figure out the length (including '\0') */
      {
        len = 0;
        while (ufp[len])
          len++;
        len++;
      }
      if (len <= buflen) /* enough space to copy element into buffer */
      {
        unsigned int pos;
        elemarray[index++] = buffer;
        for (pos = 0; pos < len; pos++)
          *buffer++ = *ufp++;
        buflen -= len;
      }
      if (elemsize > 0 && buflen < ((unsigned int)elemsize))
        break; /* element size is known and insufficient space for next elem */
    }
  }
  elemarray[index] = (char *)0;
  return &elemarray[0];
}
/* copy 16:16 char far * (eg hostent.h_name) to local buffer */
/* result will always be '\0' terminated */
static char *__copy_farp( void *__src16, char *buffer, unsigned int buflen)
{
  unsigned int pos = 0;
  const char far *cfp = (const char far *)0;

  if (__src16 && buflen > 1)
    cfp = (const char far *)MK_FP32( (void *)__src16 );

  if (cfp)
  {
    while (*cfp && pos < (buflen-1))
      buffer[pos++] = (char)(*cfp++);
  }
  buffer[pos] = '\0';
  return buffer;
}
#endif /* __WINDOWS_386__ */

/* ---------------------------------------- */

#if defined(__WINDOWS_386__)
static struct hostent FAR * PASCAL FAR __parsehostent(int dobyname, DWORD hpp )
{
  struct hostent far *hp = (struct hostent far *)0;

  if (hpp)
    hp = (struct hostent far *)MK_FP32( (void *)hpp );

  if (hp)
  {
    if (hp->h_length > 0 && hp->h_length < 10) /* size makes sense */
    {
      static struct
      {
        char hostname[256]; 
        char aliasbuf[256]; char *aliaslist[8];
        char  addrbuf[128]; char *addrlist[32]; 
        struct hostent hent;
      } ghn[2];
      dobyname = ((dobyname)?(1):(0));

      ghn[dobyname].hent.h_addrtype = hp->h_addrtype;
      ghn[dobyname].hent.h_length = hp->h_length;

      ghn[dobyname].hent.h_name = 
                 __copy_farp( (void *)hp->h_name, ghn[dobyname].hostname,
                              sizeof(ghn[dobyname].hostname) );

      ghn[dobyname].hent.h_addr_list = 
               __copy_farpfarp( (void *)hp->h_addr_list, 
                                ghn[dobyname].addrbuf, 
                                sizeof(ghn[dobyname].addrbuf),
                                ghn[dobyname].addrlist, 
                                (sizeof(ghn[dobyname].addrlist)/sizeof(ghn[dobyname].addrlist[0])),
                                ghn[dobyname].hent.h_length );

      ghn[dobyname].hent.h_aliases = 
               __copy_farpfarp( (void *)hp->h_aliases, 
                                ghn[dobyname].aliasbuf, 
                                sizeof(ghn[dobyname].aliasbuf),
                                ghn[dobyname].aliaslist, 
                                (sizeof(ghn[dobyname].aliaslist)/sizeof(ghn[dobyname].aliaslist[0])),
                                -1 /* asciiz */ );

      return &(ghn[dobyname].hent);
    } /* if (hp->h_length > 0 && hp->h_length < 10) */
  } /* if (hp) */

  return (struct hostent FAR *)0;
}
#endif /* if defined(__WINDOWS_386__) */

/* ---------------------------------------- */

struct hostent FAR * PASCAL FAR gethostbyaddr(const char FAR * name, int len, int type)
{
  FARPROC __gethostbyaddr = ImportWinsockSymbol( "gethostbyaddr" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__gethostbyaddr)
  {
    #if defined(__WINDOWS_386__)
    return __parsehostent(0, _Call16(__gethostbyaddr,"pww",name,len,type));
    #else
    return (*((struct hostent FAR * (PASCAL FAR *)(const char FAR *,int,int))
                                         (__gethostbyaddr)))(name,len,type);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (hostent FAR *)0;
}

/* ---------------------------------------- */

struct hostent FAR * PASCAL FAR gethostbyname(const char FAR * name)
{
  FARPROC __gethostbyname = ImportWinsockSymbol( "gethostbyname" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__gethostbyname)
  {
    #if defined(__WINDOWS_386__)
    return __parsehostent(1, _Call16(__gethostbyname,"p",name));
    #else
    return (*((struct hostent FAR * (PASCAL FAR *)(const char FAR *))
                                         (__gethostbyname)))(name);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (struct hostent FAR * )0;
}

/* ---------------------------------------- */

#if defined(__WINDOWS_386__)
static struct servent  FAR *__parse_protoservent(int isprotoent, DWORD __sep)
{
  struct servent far *sep = (struct servent far *)0;
  if (__sep)
    sep = (struct servent far *)MK_FP32( (void *)__sep );
  if (sep)
  {
    static struct {
      char s_name[32];
      char s_alias_buf[128];
      char *s_aliases[5];
      char s_proto[32];
      struct servent svent;
    } se[2];
    int selse = ((isprotoent)?(1):(0));

    se[selse].svent.s_name = __copy_farp( (void *)sep->s_name, 
                             se[selse].s_name, sizeof(se[selse].s_name));

    se[selse].svent.s_aliases = __copy_farpfarp( (void *)sep->s_aliases,
                             se[selse].s_alias_buf, 
                             sizeof(se[selse].s_alias_buf),
                             se[selse].s_aliases,
                             (sizeof(se[selse].s_aliases)/sizeof(se[selse].s_aliases[0])),
                             -1 /* asciiz */ );

    se[selse].svent.s_port = sep->s_port; /* SHORT! proto # if isprotent */

    if (!isprotoent) /* protoent only has 3 fields ('port' is proto number) */
    {
      se[selse].svent.s_proto = __copy_farp( (void *)sep->s_proto, 
                             se[selse].s_proto, sizeof(se[selse].s_proto));
    }

    return &(se[selse].svent);
  } /* if (sep) */
    
  return (struct servent FAR *)0;
}
#endif /* __WINDOWS_386__ */

/* ---------------------------------------- */

struct servent FAR * PASCAL FAR getservbyport (int port, const char FAR * proto)
{
  FARPROC __proc = ImportWinsockSymbol( "getservbyport" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    if (port > 0xffff) 
    {
      w32sockstatics.lasterror = WSANO_DATA;
      return (struct servent FAR *)0;
    }
    return __parse_protoservent(0, _Call16(__proc,"wp",port,proto));
    #else
    return (*((struct servent FAR * (PASCAL FAR *)(int, const char FAR *))
                                         (__proc)))(port, proto);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (struct servent FAR * )0;
}

/* ---------------------------------------- */

struct servent FAR * PASCAL FAR getservbyname (const char FAR * name, const char FAR * proto)
{
  FARPROC __proc = ImportWinsockSymbol( "getservbyname" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return __parse_protoservent(0, _Call16(__proc,"pp",name,proto));
    #else
    return (*((struct servent FAR * (PASCAL FAR *)(const char FAR *,const char FAR *))
                                         (__proc)))(name, proto);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (struct servent FAR * )0;
}

/* ---------------------------------------- */

struct protoent FAR * PASCAL FAR getprotobynumber (int proto)
{
  FARPROC __proc = ImportWinsockSymbol( "getprotobynumber" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return (struct protoent FAR *)__parse_protoservent(1, 
                                     _Call16(__proc,"w",proto));
    #else
    return (*((struct protoent FAR * (PASCAL FAR *)(int))
                                         (__proc)))(proto);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (struct protoent FAR * )0;
}

/* ---------------------------------------- */

struct protoent FAR * PASCAL FAR getprotobyname (const char FAR * name)
{
  FARPROC __proc = ImportWinsockSymbol( "getprotobyname" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__proc)
  {
    #if defined(__WINDOWS_386__)
    return (struct protoent FAR *)__parse_protoservent(1, 
                                     _Call16(__proc,"p",name));
    #else
    return (*((struct protoent FAR * (PASCAL FAR *)(const char FAR *))
                                         (__proc)))(name);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (struct protoent FAR * )0;
}

/* ---------------------------------------- */

int PASCAL FAR send(SOCKET s, const char FAR * buf, int len, int flags)
{
  FARPROC __send = ImportWinsockSymbol( "send" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__send)
  {
    #if defined(__WINDOWS_386__)
    short rc;
    if (len > 0x7fff)
      len = 0x7fff;
    rc = (short)(0xFFFF & (_Call16(__send, "wpww", s, buf, len, flags)));
    return ((rc<0)?(-1):(((int)(rc))&0xFFFF));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, const char FAR *, int, int))(__send)))
                                                         (s, buf, len, flags);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR recv (SOCKET s, char FAR * buf, int len, int flags)
{
  FARPROC __recv = ImportWinsockSymbol( "recv" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__recv)
  {
    #if defined(__WINDOWS_386__)
    short rc;
    if (len > 0x7fff)
      len = 0x7fff;
    rc = (short)(0xFFFF & (_Call16(__recv, "wpww", s, buf, len, flags)));
    return ((rc<0)?(-1):(((int)(rc))&0xFFFF));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, char FAR *, int, int))(__recv)))
                                                         (s, buf, len, flags);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

int PASCAL FAR ioctlsocket(SOCKET s, long cmd, u_long FAR *argp)
{
  FARPROC __ioctlsocket = ImportWinsockSymbol( "ioctlsocket" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__ioctlsocket)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16(__ioctlsocket, "wdp", s, cmd, argp )&0xFFFF)==0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(SOCKET, long, u_long FAR *))(__ioctlsocket)))
                                                          ( s, cmd, argp);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

/* ---------------------------------------- */

BOOL PASCAL FAR WSAIsBlocking(void)
{
  FARPROC __WSAIsBlocking = ImportWinsockSymbol( "WSAIsBlocking" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__WSAIsBlocking)
  {
    #if defined(__WINDOWS_386__)
    return (BOOL)(_Call16( __WSAIsBlocking, ""));
    #else
    return (*((BOOL (PASCAL FAR *)(void))(__WSAIsBlocking)))();
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return FALSE;
}

int PASCAL FAR WSAUnhookBlockingHook(void)
{
  FARPROC proc = ImportWinsockSymbol( "WSAUnhookBlockingHook" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (proc)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16( proc, "")&0xffff)==0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(void))(proc)))();
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

FARPROC PASCAL FAR WSASetBlockingHook(FARPROC lpBlockFunc)
{
  FARPROC proc;
  if (!lpBlockFunc)
  {
    w32sockstatics.lasterror = WSAEINVAL;
    return (FARPROC)0;
  }
  w32sockstatics.lasterror = 0; /* <- no override */
  proc = ImportWinsockSymbol( "WSASetBlockingHook" );
  if (proc)
  {
    #if defined(__WINDOWS_386__)
    return (FARPROC)_Call16(proc,"d",lpBlockFunc);
    #else
    return (*((FARPROC (PASCAL FAR *)(FARPROC))(proc)))(lpBlockFunc);
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return (FARPROC)0;
}

int PASCAL FAR WSACancelBlockingCall(void)
{
  FARPROC proc = ImportWinsockSymbol( "WSACancelBlockingCall" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (proc)
  {
    #if defined(__WINDOWS_386__)
    return (((_Call16( proc, "")&0xffff)==0)?(0):(-1));
    #else
    return (*((int (PASCAL FAR *)(void))(proc)))();
    #endif
  }
  w32sockstatics.lasterror = WSAEOPNOTSUPP;
  return -1;
}

#if !defined(_WINNT_)
/* Win32 does not install a default blocking hook (indeed, it doesn't
** even support blocking hooks), so for 16 bit apps running on win32, 
** we install a default blocking hook.
*/
void CALLBACK __default_blocking_hook(void)
{
  MSG msg;
  while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE ))
  {
    BOOL processed = FALSE;
    if (!GetMessage(&msg, NULL, 0, 0))
      break;
    if (msg.hwnd)
    {
      if (GetClassWord(msg.hwnd, GCW_ATOM) == 32770)
        processed = TRUE;
      else
      {
        HWND hwnd = GetParent(msg.hwnd);
        if (hwnd && GetClassWord(hwnd, GCW_ATOM) == 32770)
          processed = TRUE;
      }
      if (processed)
        processed = IsDialogMessage(msg.hwnd, &msg);
    }
    if (!processed)
    {  
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }
  return;
}
extern HINSTANCE winGetInstanceHandle(void); /* w32pre.cpp */
extern long winGetVersion(void);
#endif

/* ---------------------------------------- */

int PASCAL FAR WSAStartup(WORD wVerRequired, LPWSADATA lpWSAData)
{
  int ec = WSASYSNOTREADY;
  w32sockstatics.lasterror = WSANOTINITIALISED;

  if (!lpWSAData)
  {
    ec = WSAEFAULT;
  }
  else if (__w32sockInitDeinit(+1)) /* otherwise WSASYSNOTREADY */
  {
    FARPROC __WSAStartup = ImportWinsockSymbol( "WSAStartup" );
    if (__WSAStartup)
    {
      w32sockstatics.lasterror = 0; /* <- no override */
      #if defined(__WINDOWS_386__)
      lpWSAData->szDescription[0]=0;
      lpWSAData->szSystemStatus[0]=0;
      ec = (_Call16( __WSAStartup, "wp", wVerRequired, lpWSAData )&0xFFFF);
      #else
      ec = (*((int (PASCAL FAR *)(WORD, LPWSADATA))(__WSAStartup)))
                                           (wVerRequired, lpWSAData);
      #endif
      if (ec == 0)
      { 
        #if !defined(_WINNT_)
        if (w32sockstatics.initlevel == 1 && /* only required if */
            winGetVersion() >= 400)          /* running on 32bit OS */
        {                                 /* (no default blocking hook) */
          w32sockstatics.blockinghook = 
                  MakeProcInstance((FARPROC)__default_blocking_hook, 
                  winGetInstanceHandle());
          if (w32sockstatics.blockinghook)
            WSASetBlockingHook(w32sockstatics.blockinghook);
        }
        #endif
        return 0;
      }
    }
    __w32sockInitDeinit(-1);
  }
  return ec;
}

/* ---------------------------------------- */

int PASCAL FAR WSACleanup(void)
{
  int rc = -1; /* SOCKET_ERROR */
  FARPROC __WSACleanup = ImportWinsockSymbol( "WSACleanup" );
  w32sockstatics.lasterror = 0; /* <- no override */
  if (__WSACleanup) /* will be null if no previous init */
  {
    #if defined(__WINDOWS_386__)
    rc=(((_Call16(__WSACleanup,"") & 0xFFFF) == 0)?(0):(-1));
    #else
    rc=(*((int (PASCAL FAR *)(void))(__WSACleanup)))();
    #endif
    if (rc == 0)
    {
      __w32sockInitDeinit(-1);
      #if !defined(_WINNT_) /* not needed there */
      if (w32sockstatics.initlevel == 0 &&
         w32sockstatics.blockinghook)
      {
        (void)FreeProcInstance(w32sockstatics.blockinghook);
        w32sockstatics.blockinghook = 0; 
      }
      #endif
    }  
  }
  return rc;
}

/* ---------------------------------------- */

/*
 * IPV6 compatibility functions. Emulates getaddrinfo() and friends on system
 * without IPv6 libraries.
 * The emulation code is copied from Windows SDK (wspiapi.h) to allow building
 * with any SDK version (at least down to MSVC98/VS60).
 */
#ifdef HAVE_IPV6

#define NI_MAXHOST      1025  /* Max size of a fully-qualified domain name */

#define WspiapiMalloc(tSize)    calloc(1, (tSize))
#define WspiapiFree(p)          free(p)
#define WspiapiSwap(a, b, c)    { (c) = (a); (a) = (b); (b) = (c); }

#define _WSPIAPI_STRCPY_S(_Dst, _Size, _Src) strcpy((_Dst), (_Src))
#define _WSPIAPI_STRNCPY_S(_Dst, _Size, _Src, _Count) strncpy((_Dst), (_Src), (_Count)); (_Dst)[(_Size) - 1] = 0


static char *WspiapiStrdup(const char *pszString)
/*++

Routine Description
    allocates enough storage via calloc() for a copy of the string,
    copies the string into the new memory, and returns a pointer to it.

Arguments
    pszString       string to copy into new memory

Return Value
    a pointer to the newly allocated storage with the string in it.
    NULL if enough memory could not be allocated, or string was NULL.

--*/
{
    char    *pszMemory;
    size_t  cchMemory;

    if (!pszString)
        return(NULL);

    cchMemory = strlen(pszString) + 1;
    pszMemory = (char *) WspiapiMalloc(cchMemory);
    if (!pszMemory)
        return(NULL);

    _WSPIAPI_STRCPY_S(pszMemory, cchMemory, pszString);
    return pszMemory;
}

static struct addrinfo *WspiapiNewAddrInfo(int iSocketType, int iProtocol, WORD wPort, DWORD dwAddress)
/*++

Routine Description
    allocate an addrinfo structure and populate fields.
    IPv4 specific internal function, not exported.

Arguments
    iSocketType         SOCK_*.  can be wildcarded (zero).
    iProtocol           IPPROTO_*.  can be wildcarded (zero).
    wPort               port number of service (in network order).
    dwAddress           IPv4 address (in network order).

Return Value
    returns an addrinfo struct, or NULL if out of memory.

--*/
{
    struct addrinfo     *ptNew;
    struct sockaddr_in  *ptAddress;

    // allocate a new addrinfo structure.
    ptNew       =
        (struct addrinfo *) WspiapiMalloc(sizeof(struct addrinfo));
    if (!ptNew)
        return NULL;

    ptAddress   =
        (struct sockaddr_in *) WspiapiMalloc(sizeof(struct sockaddr_in));
    if (!ptAddress)
    {
        WspiapiFree(ptNew);
        return NULL;
    }
    ptAddress->sin_family       = AF_INET;
    ptAddress->sin_port         = wPort;
    ptAddress->sin_addr.s_addr  = dwAddress;

    // fill in the fields...
    ptNew->ai_family            = PF_INET;
    ptNew->ai_socktype          = iSocketType;
    ptNew->ai_protocol          = iProtocol;
    ptNew->ai_addrlen           = sizeof(struct sockaddr_in);
    ptNew->ai_addr              = (struct sockaddr *) ptAddress;

    return ptNew;
}

static int WspiapiClone(WORD wPort, struct addrinfo *ptResult)
/*++

Routine Description
    clone every addrinfo structure in ptResult for the UDP service.
    ptResult would need to be freed if an error is returned.

Arguments
    wPort               port number of UDP service.
    ptResult            list of addrinfo structures, each
                        of whose node needs to be cloned.

Return Value
    Returns 0 on success, an EAI_MEMORY on allocation failure.

--*/
{
    struct addrinfo *ptNext = NULL;
    struct addrinfo *ptNew  = NULL;

    for (ptNext = ptResult; ptNext != NULL; )
    {
        // create an addrinfo structure...
        ptNew = WspiapiNewAddrInfo(
            SOCK_DGRAM,
            ptNext->ai_protocol,
            wPort,
            ((struct sockaddr_in *) ptNext->ai_addr)->sin_addr.s_addr);
        if (!ptNew)
            break;

        // link the cloned addrinfo
        ptNew->ai_next  = ptNext->ai_next;
        ptNext->ai_next = ptNew;
        ptNext          = ptNew->ai_next;
    }

    if (ptNext != NULL)
        return EAI_MEMORY;

    return 0;
}

static BOOL WspiapiParseV4Address (const char *pszAddress, PDWORD pdwAddress)
/*++

Routine Description
    get the IPv4 address (in network byte order) from its string
    representation.  the syntax should be a.b.c.d.

Arguments
    pszArgument         string representation of the IPv4 address
    ptAddress           pointer to the resulting IPv4 address

Return Value
    Returns FALSE if there is an error, TRUE for success.

--*/
{
    DWORD       dwAddress   = 0;
    const char  *pcNext     = NULL;
    int         iCount      = 0;

    // ensure there are 3 '.' (periods)
    for (pcNext = pszAddress; *pcNext != '\0'; pcNext++)
        if (*pcNext == '.')
            iCount++;
    if (iCount != 3)
        return FALSE;

    // return an error if dwAddress is INADDR_NONE (255.255.255.255)
    // since this is never a valid argument to getaddrinfo.
    dwAddress = inet_addr(pszAddress);
    if (dwAddress == INADDR_NONE)
        return FALSE;

    *pdwAddress = dwAddress;
    return TRUE;
}

static int WspiapiQueryDNS(
          const char                      *pszNodeName,
          int                             iSocketType,
          int                             iProtocol,
          WORD                            wPort,
          char                            pszAlias[NI_MAXHOST],
          struct addrinfo                 **pptResult)
/*++

Routine Description
    helper routine for WspiapiLookupNode.
    performs name resolution by querying the DNS for A records.
    *pptResult would need to be freed if an error is returned.

Arguments
    pszNodeName         name of node to resolve.
    iSocketType         SOCK_*.  can be wildcarded (zero).
    iProtocol           IPPROTO_*.  can be wildcarded (zero).
    wPort               port number of service (in network order).
    pszAlias            where to return the alias.  must be of size NI_MAXHOST.
    pptResult           where to return the result.

Return Value
    Returns 0 on success, an EAI_* style error value otherwise.

--*/
{
    struct addrinfo **pptNext   = pptResult;
    struct hostent  *ptHost     = NULL;
    char            **ppAddresses;

    *pptNext    = NULL;
    pszAlias[0] = '\0';

    ptHost = gethostbyname(pszNodeName);
    if (ptHost)
    {
        if ((ptHost->h_addrtype == AF_INET)     &&
            (ptHost->h_length   == sizeof(struct in_addr)))
        {
            for (ppAddresses    = ptHost->h_addr_list;
                 *ppAddresses   != NULL;
                 ppAddresses++)
            {
                // create an addrinfo structure...
                *pptNext = WspiapiNewAddrInfo(
                    iSocketType,
                    iProtocol,
                    wPort,
                    ((struct in_addr *) *ppAddresses)->s_addr);
                if (!*pptNext)
                    return EAI_MEMORY;

                pptNext = &((*pptNext)->ai_next);
            }
        }

        // pick up the canonical name.
        _WSPIAPI_STRNCPY_S(pszAlias, NI_MAXHOST, ptHost->h_name, NI_MAXHOST - 1);

        return 0;
    }

    switch (WSAGetLastError())
    {
        case WSAHOST_NOT_FOUND: return EAI_NONAME;
        case WSATRY_AGAIN:      return EAI_AGAIN;
        case WSANO_RECOVERY:    return EAI_FAIL;
        case WSANO_DATA:        return EAI_NODATA;
        default:                return EAI_NONAME;
    }
}

static int WspiapiLookupNode(
          const char                      *pszNodeName,
          int                             iSocketType,
          int                             iProtocol,
          WORD                            wPort,
          BOOL                            bAI_CANONNAME,
          struct addrinfo                 **pptResult)
/*++

Routine Description
    resolve a nodename and return a list of addrinfo structures.
    IPv4 specific internal function, not exported.
    *pptResult would need to be freed if an error is returned.

    NOTE: if bAI_CANONNAME is true, the canonical name should be
          returned in the first addrinfo structure.

Arguments
    pszNodeName         name of node to resolve.
    iSocketType         SOCK_*.  can be wildcarded (zero).
    iProtocol           IPPROTO_*.  can be wildcarded (zero).
    wPort               port number of service (in network order).
    bAI_CANONNAME       whether the AI_CANONNAME flag is set.
    pptResult           where to return result.

Return Value
    Returns 0 on success, an EAI_* style error value otherwise.

--*/
{
    int     iError              = 0;
    int     iAliasCount         = 0;

    char    szFQDN1[NI_MAXHOST] = "";
    char    szFQDN2[NI_MAXHOST] = "";
    char    *pszName            = szFQDN1;
    char    *pszAlias           = szFQDN2;
    char    *pszScratch         = NULL;
    _WSPIAPI_STRNCPY_S(pszName, NI_MAXHOST, pszNodeName, NI_MAXHOST - 1);

    for (;;)
    {
        iError = WspiapiQueryDNS(pszNodeName,
                                 iSocketType,
                                 iProtocol,
                                 wPort,
                                 pszAlias,
                                 pptResult);
        if (iError)
            break;

        // if we found addresses, then we are done.
        if (*pptResult)
            break;

        // stop infinite loops due to DNS misconfiguration.  there appears
        // to be no particular recommended limit in RFCs 1034 and 1035.
        if ((!strlen(pszAlias))             ||
            (!strcmp(pszName, pszAlias))    ||
            (++iAliasCount == 16))
        {
            iError = EAI_FAIL;
            break;
        }

        // there was a new CNAME, look again.
        WspiapiSwap(pszName, pszAlias, pszScratch);
    }

    if (!iError && bAI_CANONNAME)
    {
        (*pptResult)->ai_canonname = WspiapiStrdup(pszAlias);
        if (!(*pptResult)->ai_canonname)
            iError = EAI_MEMORY;
    }

    return iError;
}

/* ---------------------------------------- */

static void WSAAPI WspiapiLegacyFreeAddrInfo (struct addrinfo *ptHead)
/*++

Routine Description
    Free an addrinfo structure (or chain of structures).
    As specified in RFC 2553, Section 6.4.

Arguments
    ptHead              structure (chain) to free

--*/
{
    struct addrinfo *ptNext;    // next strcture to free

    for (ptNext = ptHead; ptNext != NULL; ptNext = ptHead)
    {
        if (ptNext->ai_canonname)
            WspiapiFree(ptNext->ai_canonname);

        if (ptNext->ai_addr)
            WspiapiFree(ptNext->ai_addr);

        ptHead = ptNext->ai_next;
        WspiapiFree(ptNext);
    }
}

static int WSAAPI WspiapiLegacyGetAddrInfo(const char *pszNodeName, const char *pszServiceName, const struct addrinfo *ptHints, struct addrinfo **pptResult)
/*++

Routine Description
    Protocol-independent name-to-address translation.
    As specified in RFC 2553, Section 6.4.
    This is the hacked version that only supports IPv4.

Arguments
    pszNodeName         node name to lookup.
    pszServiceName      service name to lookup.
    ptHints             hints about how to process request.
    pptResult           where to return result.

Return Value
    returns zero if successful, an EAI_* error code if not.

--*/
{
    int                 iError      = 0;
    int                 iFlags      = 0;
    int                 iFamily     = PF_UNSPEC;
    int                 iSocketType = 0;
    int                 iProtocol   = 0;
    WORD                wPort       = 0;
    DWORD               dwAddress   = 0;

    struct servent      *ptService  = NULL;
    char                *pc         = NULL;
    BOOL                bClone      = FALSE;
    WORD                wTcpPort    = 0;
    WORD                wUdpPort    = 0;


    // initialize pptResult with default return value.
    *pptResult  = NULL;


    ////////////////////////////////////////
    // validate arguments...
    //

    // both the node name and the service name can't be NULL.
    if ((!pszNodeName) && (!pszServiceName))
        return EAI_NONAME;

    // validate hints.
    if (ptHints)
    {
        // all members other than ai_flags, ai_family, ai_socktype
        // and ai_protocol must be zero or a null pointer.
        if ((ptHints->ai_addrlen    != 0)       ||
            (ptHints->ai_canonname  != NULL)    ||
            (ptHints->ai_addr       != NULL)    ||
            (ptHints->ai_next       != NULL))
        {
            return EAI_FAIL;
        }

        // the spec has the "bad flags" error code, so presumably we
        // should check something here.  insisting that there aren't
        // any unspecified flags set would break forward compatibility,
        // however.  so we just check for non-sensical combinations.
        //
        // we cannot come up with a canonical name given a null node name.
        iFlags      = ptHints->ai_flags;
        if ((iFlags & AI_CANONNAME) && !pszNodeName)
            return EAI_BADFLAGS;

        // we only support a limited number of protocol families.
        iFamily     = ptHints->ai_family;
        if ((iFamily != PF_UNSPEC) && (iFamily != PF_INET))
            return EAI_FAMILY;

        // we only support only these socket types.
        iSocketType = ptHints->ai_socktype;
        if ((iSocketType != 0)                  &&
            (iSocketType != SOCK_STREAM)        &&
            (iSocketType != SOCK_DGRAM)         &&
            (iSocketType != SOCK_RAW))
            return EAI_SOCKTYPE;

        // REVIEW: What if ai_socktype and ai_protocol are at odds?
        iProtocol   = ptHints->ai_protocol;
    }


    ////////////////////////////////////////
    // do service lookup...

    if (pszServiceName)
    {
        wPort = (WORD) strtoul(pszServiceName, &pc, 10);
        if (*pc == '\0')        // numeric port string
        {
            wPort = wTcpPort = wUdpPort = htons(wPort);
            if (iSocketType == 0)
            {
                bClone      = TRUE;
                iSocketType = SOCK_STREAM;
            }
        }
        else                    // non numeric port string
        {
            if ((iSocketType == 0) || (iSocketType == SOCK_DGRAM))
            {
                ptService = getservbyname(pszServiceName, "udp");
                if (ptService)
                    wPort = wUdpPort = ptService->s_port;
            }

            if ((iSocketType == 0) || (iSocketType == SOCK_STREAM))
            {
                ptService = getservbyname(pszServiceName, "tcp");
                if (ptService)
                    wPort = wTcpPort = ptService->s_port;
            }

            // assumes 0 is an invalid service port...
            if (wPort == 0)     // no service exists
                return (iSocketType ? EAI_SERVICE : EAI_NONAME);

            if (iSocketType == 0)
            {
                // if both tcp and udp, process tcp now & clone udp later.
                iSocketType = (wTcpPort) ? SOCK_STREAM : SOCK_DGRAM;
                bClone      = (wTcpPort && wUdpPort);
            }
        }
    }



    ////////////////////////////////////////
    // do node name lookup...

    // if we weren't given a node name,
    // return the wildcard or loopback address (depending on AI_PASSIVE).
    //
    // if we have a numeric host address string,
    // return the binary address.
    //
    if ((!pszNodeName) || (WspiapiParseV4Address(pszNodeName, &dwAddress)))
    {
        if (!pszNodeName)
        {
            dwAddress = htonl((iFlags & AI_PASSIVE)
                              ? INADDR_ANY
                              : INADDR_LOOPBACK);
        }

        // create an addrinfo structure...
        *pptResult =
            WspiapiNewAddrInfo(iSocketType, iProtocol, wPort, dwAddress);
        if (!(*pptResult))
            iError = EAI_MEMORY;

        if (!iError && pszNodeName)
        {
            // implementation specific behavior: set AI_NUMERICHOST
            // to indicate that we got a numeric host address string.
            (*pptResult)->ai_flags |= AI_NUMERICHOST;

            // return the numeric address string as the canonical name
            if (iFlags & AI_CANONNAME)
            {
                (*pptResult)->ai_canonname =
                    WspiapiStrdup(inet_ntoa(*((struct in_addr *) &dwAddress)));
                if (!(*pptResult)->ai_canonname)
                    iError = EAI_MEMORY;
            }
        }
    }


    // if we do not have a numeric host address string and
    // AI_NUMERICHOST flag is set, return an error!
    else if (iFlags & AI_NUMERICHOST)
    {
        iError = EAI_NONAME;
    }


    // since we have a non-numeric node name,
    // we have to do a regular node name lookup.
    else
    {
        iError = WspiapiLookupNode(pszNodeName,
                                   iSocketType,
                                   iProtocol,
                                   wPort,
                                   (iFlags & AI_CANONNAME),
                                   pptResult);
    }

    if (!iError && bClone)
    {
        iError = WspiapiClone(wUdpPort, *pptResult);
    }

    if (iError)
    {
        WspiapiLegacyFreeAddrInfo(*pptResult);
        *pptResult  = NULL;
    }

    return (iError);
}

/* ---------------------------------------- */

typedef int  (WSAAPI *WSPIAPI_PGETADDRINFO)(PCSTR pNodeName, PCSTR pServiceName, const ADDRINFOA *pHints, PADDRINFOA *ppResult);
typedef void (WSAAPI *WSPIAPI_PFREEADDRINFO)(struct addrinfo *ai);

static WSPIAPI_PGETADDRINFO  p_getaddrinfo;
static WSPIAPI_PFREEADDRINFO p_freeaddrinfo;

/*
 * Load getaddrinfo and freeaddrinfo in pair to be sure that
 * _both_ functions have same style (real or emulated).
 * Note: not thread-safe.
 */
static void load_v6_functions(void)
{
  if (p_getaddrinfo == NULL)
  {
    if ( ( p_getaddrinfo  = (WSPIAPI_PGETADDRINFO)  ImportWinsockSymbol( "getaddrinfo"  ) ) == NULL ||
         ( p_freeaddrinfo = (WSPIAPI_PFREEADDRINFO) ImportWinsockSymbol( "freeaddrinfo" ) ) == NULL
       )
    {
      /* both are emulated */
      p_getaddrinfo  = WspiapiLegacyGetAddrInfo;
      p_freeaddrinfo = WspiapiLegacyFreeAddrInfo;
    }
  }
}

/* ---------------------------------------- */

int WSAAPI getaddrinfo(PCSTR pNodeName, PCSTR pServiceName, const ADDRINFOA *pHints, PADDRINFOA *ppResult)
{
  int ret;

  load_v6_functions();
  ret = p_getaddrinfo(pNodeName, pServiceName, pHints, ppResult);
  WSASetLastError(ret);
  return ret;
}

/* ---------------------------------------- */

void WSAAPI freeaddrinfo(struct addrinfo *ai)
{
  load_v6_functions();
  p_freeaddrinfo(ai);
}

/* ---------------------------------------- */

// WARNING: The gai_strerror inline functions below use static buffers,
// and hence are not thread-safe.  We'll use buffers long enough to hold
// 1k characters.  Any system error messages longer than this will be
// returned as empty strings.  However 1k should work for the error codes
// used by getaddrinfo().
#define GAI_STRERROR_BUFFER_SIZE 1024

char *gai_strerrorA(int ecode)
{
    DWORD dwMsgLen;
    static char buff[GAI_STRERROR_BUFFER_SIZE + 1];

    dwMsgLen = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM
                             |FORMAT_MESSAGE_IGNORE_INSERTS
                             |FORMAT_MESSAGE_MAX_WIDTH_MASK,
                              NULL,
                              ecode,
                              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                              (LPSTR)buff,
                              GAI_STRERROR_BUFFER_SIZE,
                              NULL);

    return buff;
}

#endif /* HAVE_IPV6 */

/* ---------------------------------------- */
