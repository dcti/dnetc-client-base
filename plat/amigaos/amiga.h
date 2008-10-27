/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amiga.h,v 1.4 2008/10/27 09:49:33 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * Global includes, prototypes and defines for AmigaOS
 * ----------------------------------------------------------------------
*/

#ifndef _AMIGA_H_
#define _AMIGA_H_

#ifdef __cplusplus
extern "C" {
#endif

   #ifdef __amigaos4__
      #define __BEGIN_DECLS
      #define __END_DECLS
      #define __P(p) p
      #define __USE_BASETYPE__
      #define __USE_OLD_TIMEVAL__
      #define NO_MIAMI
   #endif

   #if defined(__PPC__) && !(defined(__amigaos4__) || defined(__MORPHOS__))
      #define __OS3PPC__
   #endif

   #ifdef __OS3PPC__
   #pragma pack(2)
   #endif

   #include <exec/exec.h>
   #include <exec/execbase.h>

   #if !defined(AFF_68060)
      #define AFF_68060 (1L<<7)
   #endif

   #ifdef __OS3PPC__
      #ifndef __POWERUP__
         #include <powerpc/powerpc.h>
         #include <powerpc/powerpc_protos.h>
      #else
         #include <powerup/ppcinline/macros.h>
         #include <powerup/ppclib/ppc.h>
         #include <powerup/ppclib/tasks.h>
         #include <powerup/ppclib/message.h>

         #ifndef PPC_BASE_NAME
         #define PPC_BASE_NAME PPCLibBase
         #endif /* !PPC_BASE_NAME */

         extern struct Library *PPCLibBase;

         #define PPCSetTaskAttrs(TaskObject, Tags) \
                 LP2(0xc0, ULONG, PPCSetTaskAttrs, void*, TaskObject, a0, struct TagItem*, Tags, a1, \
                 , PPC_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)

         #ifndef NO_PPCINLINE_STDARG
         #define PPCSetTaskAttrsTags(a0, tags...) \
                 ({ULONG _tags[] = { tags }; PPCSetTaskAttrs((a0), (struct TagItem*)_tags);})
         #endif /* !NO_PPCINLINE_STDARG */
      #endif
   #elif defined(__MORPHOS__)
      #include <exec/tasks.h>
      #include <emul/emulinterface.h>
   #endif

   #include <proto/exec.h>
   #include <proto/dos.h>
   #include <dos/dos.h>
   #include <dos/dostags.h>
   #include <devices/timer.h>

   #ifdef __OS3PPC__
   #pragma pack()
   #endif

   #include <sys/time.h>
   #include <sys/socket.h>
   #include <netinet/in.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <time.h>
   #include <signal.h>
   #include <stdarg.h>
   #include <assert.h>

   #ifdef __OS3PPC__
      #undef SetSignal
      #undef AllocVec
      #undef FreeVec
      #ifdef __POWERUP__
         #define SetSignal(x,y) PPCSetSignal(x,y)
         #define AllocVec(a,b) PPCAllocVec(a,b)
         #define FreeVec(a) PPCFreeVec(a)
      #else
         #define SetSignal(a,b) SetSignalPPC(a,b)
         #define AllocVec(a,b) AllocVecPPC(a,b,0)
         #define FreeVec(a) FreeVecPPC(a)
      #endif
   #endif

   #define DNETC_MSG_RESTART	0x01
   #define DNETC_MSG_SHUTDOWN	0x02
   #define DNETC_MSG_PAUSE	0x04
   #define DNETC_MSG_UNPAUSE	0x08
#ifdef __cplusplus
}
#endif

#ifndef __amigaos4__
struct	hostent {
	char	*h_name;	/* official name of host */
	char	**h_aliases;	/* alias list */
	int	h_addrtype;	/* host address type */
	int	h_length;	/* length of address */
	char	**h_addr_list;	/* list of addresses from name server */
#define	h_addr	h_addr_list[0]	/* address, for backward compatiblity */
};
#endif

struct ThreadArgsMsg {
   struct Message tp_ExecMessage;
   void *tp_Params;
};

__BEGIN_DECLS
// network.c
int amigaNetworkingInit(BOOL lurking);
void amigaNetworkingDeinit(void);
BOOL amigaOpenSocketLib(void);
void amigaCloseSocketLib(BOOL delayedclose);
int amigaIsNetworkingActive(void);
int amigaOnOffline(int online, char *ifacename);
BOOL amigaIsTermiteTCP(void);

// console.c
int amigaInitializeConsole(int runhidden, int runmodes);
int amigaConGetSize(int *width, int *height);
int amigaConGetPos(int *col, int *row);
int getch(void);
int amigaConIsScreen(void);
int amigaConIsGUI(void);
int amigaConOut(const char *msg);
int amigaConOutModal(const char *msg);
int amigaConOutErr(const char *msg);
BPTR amigaOpenNewConsole(char *conname);
void amigaCloseNewConsole(void);

// support.c
int amigaInit(int *argc, char **argv[]);
void amigaExit(void);
const char *amigaGetOSVersion(void);
int amigaThreadInit(void);
void amigaThreadExit(void);
ULONG amigaGetTriggerSigs(void);
int amigaPutTriggerSigs(ULONG trigs);

// memory.c
BOOL MemInit(VOID);
VOID MemDeinit(VOID);

// time.c
void amigaSleep(unsigned int secs, unsigned int usecs);
int amigaGetMonoClock(struct timeval *tp);

// gui.c
struct WBArg;
BOOL amigaGUIInit(char *programname, struct WBArg *iconname);
void amigaGUIDeinit(void);
void amigaGUIOut(char *msg);
#if !defined(__OS3PPC__)
void amigaHandleGUI(struct timerequest *tr);
#elif !defined(__POWERUP__)
void amigaHandleGUI(struct timeval *tr);
#else
void amigaHandleGUI(void *timer, ULONG timesig);
#endif

// install.c
int amigaInstall(int quiet, const char *progname);
int amigaUninstall(int quiet, const char *progname);

/* unistd prototypes - we can't include unistd.h as the prototypes interfere with
** with the Amiga socket prototype macros
*/
#ifndef __amigaos4__
int isatty __P((int));
int unlink __P((const char *));
int access __P((const char *, int));
int ftruncate(int file_descriptor, off_t length);
#endif

__END_DECLS

#endif
