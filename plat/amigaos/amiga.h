/*
** $Id: amiga.h,v 1.1.2.1 2001/01/21 15:10:27 cyp Exp $
*/

#ifndef _AMIGA_H_
#define _AMIGA_H_

extern "C" {
   #define AFF_68060 (1L<<7)

   #ifdef __PPC__
   #pragma pack(2)
   #endif

   #include <exec/types.h>
   #include <exec/execbase.h>
   #include <exec/libraries.h>

   #ifdef __PPC__
      #ifndef __POWERUP__
         #include <powerpc/warpup_macros.h>
         #include <powerpc/powerpc.h>
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
   #endif

   #include <proto/exec.h>
   #include <proto/dos.h>
   #include <dos/dos.h>
   #include <dos/dostags.h>

   #ifdef __PPC__
   #pragma pack()
   #endif

   #include <sys/ioctl.h>
   #include <sys/time.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <time.h>
   #include <signal.h>
   #include <stdarg.h>
   #include <assert.h>

   #ifdef __PPC__
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
}

struct	hostent {
	char	*h_name;	/* official name of host */
	char	**h_aliases;	/* alias list */
	int	h_addrtype;	/* host address type */
	int	h_length;	/* length of address */
	char	**h_addr_list;	/* list of addresses from name server */
#define	h_addr	h_addr_list[0]	/* address, for backward compatiblity */
};

struct ThreadArgsMsg {
   struct Message tp_ExecMessage;
   void *tp_Params;
};

__BEGIN_DECLS
// network.c
int amigaNetworkingInit(BOOL lurking);
void amigaNetworkingDeinit(void);
BOOL amigaOpenSocketLib(void);
void amigaCloseSocketLib(void);
int amigaIsNetworkingActive(void);
int amigaOnOffline(int online, char *ifacename);

// console.c
int amigaInitializeConsole(int runhidden, int runmodes);
int amigaConGetSize(int *width, int *height);
int amigaConGetPos(int *col, int *row);
int getch(void);
int amigaConIsScreen(void);
int MyzarIsRunning(void);

// support.c
int amigaInit(void);
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

/* unistd prototypes - we can't include unistd.h as the prototypes interfere with
** with the Amiga socket prototype macros
*/
int isatty __P((int));
int unlink __P((const char *));
int access __P((const char *, int));

__END_DECLS

#endif
