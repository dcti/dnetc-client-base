/* 
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
return "@(#)$Id: w32sock.cpp,v 1.1.2.3 2001/06/08 07:04:17 cyp Exp $"; }

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
          DWORD i;
          if (maxlen < (DWORD)namelen)
            namelen = maxlen;
          for (i = 0; i < (DWORD)(namelen-1); i++)
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
