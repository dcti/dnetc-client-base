/* Created by Oliver Roberts <oliver@futaura.co.uk>
**
** $Id: amSupport.c,v 1.1.2.3 2001/03/19 19:18:19 oliver Exp $
**
** ----------------------------------------------------------------------
** This file contains general Amiga specific support code, including
** startup initializations.
** ----------------------------------------------------------------------
*/

/*
** Bump up the priority of the 68k mirror to 2?  Really not required anymore
** since we are running threaded (the cruncher task will always be bumped to
** pri 3), although it's enabled for PowerUp (stability problems otherwise)
*/
#ifdef __POWERUP__
#define CHANGE_MIRROR_TASK_PRI
#endif

#include "amiga.h"
#include "sleepdef.h"

#include <proto/locale.h>
#include <proto/timer.h>

#ifndef __PPC__ /* for 68K only */
#ifdef __SASC
long __near __stack  = 65536L;
#else
unsigned long __stack = 65536L;
#endif
#endif

#ifndef __PPC__
static struct MsgPort *TriggerPort;
#else
#ifdef CHANGE_MIRROR_TASK_PRI
static LONG Old68kMirrorPri; /* Used to restore shell priority on exit */
#endif
#ifdef __POWERUP__
struct Library *PPCLibBase;
static void *TriggerPort;
#else
static struct MsgPortPPC *TriggerPort;
#endif
#endif

struct TriggerMessage
{
   struct Message tm_ExecMsg;
   ULONG          tm_TriggerType;
};

extern BOOL GlobalTimerInit(VOID);
extern VOID GlobalTimerDeinit(VOID);
extern VOID CloseTimer(VOID);
extern struct Device *OpenTimer(VOID);
extern struct Library *SocketBase;

const char *amigaGetOSVersion(void)
{
   static const char *osver[11] = { "2.0","2.0x","2.1","3.0","3.1","3.2","3.3","3.4","3.5","3.9","4.0" };
   int ver = SysBase->LibNode.lib_Version;
   if (ver >= 40) {   // Detect OS 3.5/3.9
      struct Library *VersionBase;
      if ((VersionBase = OpenLibrary("version.library",0))) {
         ver = VersionBase->lib_Version;
         CloseLibrary(VersionBase);
      }
   }
   ver -= 36;
   if (ver > (46-36)) ver = (44-36);
   return osver[ver];
}

int amigaInit(void)
{
   int done = TRUE;

   #ifndef __POWERUP__
   if (!MemInit()) return FALSE;
   #endif

   #ifdef __PPC__
   /*
   ** PPC
   */
   //SetProgramName("dnetc_ppc");

   /* Set the priority of the PPC 68k mirror main task */
   #ifdef CHANGE_MIRROR_TASK_PRI
   Old68kMirrorPri = SetTaskPri(FindTask(NULL), 2);
   #endif

   #ifdef __POWERUP__
   /*
   ** PowerUp
   */
   /* Requires ppc.library 46.30 (or PPCLibEmu), due to timer/signal bugs */
   #define PPCINFOTAG_EMULATION (TAG_USER + 0x1f0ff)
   if (PPCGetAttr(PPCINFOTAG_EMULATION) == 'WARP') {
      if (!((PPCVersion() == 46 && PPCRevision() >= 29) || PPCVersion() > 46)) {
         printf("Requires PPCLibEmu 0.8a or higher!\n");
         done = FALSE;
      }
   }
   else {
      if (!((PPCVersion() == 46 && PPCRevision() >= 30) || PPCVersion() > 46)) {
         printf("Requires ppc.library 46.30 or higher!\n");
         done = FALSE;
      }
   }
   if (done) {
      if (PPCGetTaskAttr(PPCTASKTAG_STACKSIZE) < 200000) {
         printf("Please increase stack size to at least 200000 bytes!\n");
         done = FALSE;
      }
   }
   if (done) {
      done = ((PPCLibBase = OpenLibrary("ppc.library",46)) != NULL);
   }
   if (done) {
      struct TagItem tags[2] = { {PPCPORTTAG_NAME, (ULONG)"dnetc"}, {TAG_END,0} };
      void *port;
      if (!(port = PPCObtainPort(tags))) {
         done = FALSE;
         if ((TriggerPort = PPCCreatePort(tags))) {
            done = TRUE;
         }
      }
      else {
         PPCReleasePort(port);
      }
   }
   #else
   /*
   ** WarpOs
   */
   if (!FindPortPPC("dnetc")) {
      if ((TriggerPort = CreateMsgPortPPC())) {
         TriggerPort->mp_Port.mp_Node.ln_Name = "dnetc";
         TriggerPort->mp_Port.mp_Node.ln_Pri = 0;
         AddPortPPC(TriggerPort);
      }
      else {
         done = FALSE;
      }
   }
   #endif

   #else
   /*
   ** 68K
   */
   //SetProgramName("dnetc_68k");

   struct MsgPort *portexists;
   Forbid();
   portexists = FindPort("dnetc");
   Permit();

   if (!portexists) {
      if ((TriggerPort = CreateMsgPort())) {
         TriggerPort->mp_Node.ln_Name = "dnetc";
         TriggerPort->mp_Node.ln_Pri = 0;
         AddPort(TriggerPort);
      }
      else {
         done = FALSE;
      }
   }

   /*if (done) {
      if ((SysInfoBase = OpenLibrary("SysInfo.library",0))) {
         SysInfo = InitSysInfo();
      }
   }*/
   #endif

   if (!TimerBase && done) done = GlobalTimerInit();

   if (!done) amigaExit();

   return(done);
}

void amigaExit(void)
{
   GlobalTimerDeinit();
   CloseLibrary((struct Library *)LocaleBase);
   CloseLibrary(SocketBase); // ensure something hasn't left this open
   #ifdef __PPC__
   /*
   ** PPC
   */
   #ifdef CHANGE_MIRROR_TASK_PRI
   SetTaskPri(FindTask(NULL), Old68kMirrorPri);
   #endif
   #ifdef __POWERUP__
   /*
   ** PowerUp
   */
   if (TriggerPort) {
      while (!PPCDeletePort(TriggerPort)) usleep(250000);
   }
   CloseLibrary(PPCLibBase);
   #else
   /*
   ** WarpOS
   */
   if (TriggerPort) {
      RemPortPPC(TriggerPort);
      DeleteMsgPortPPC(TriggerPort);
   }
   #endif
   #else
   /*
   ** 68K
   */
   /*if (SysInfoBase) {
      if (SysInfo) FreeSysInfo(SysInfo);
      CloseLibrary(SysInfoBase);
   }*/
   if (TriggerPort) {
      RemPort(TriggerPort);
      DeleteMsgPort(TriggerPort);
   }
   #endif
}

int amigaThreadInit(void)
{
   return (OpenTimer() != NULL);
}

void amigaThreadExit(void)
{
   CloseTimer();
}

ULONG amigaGetTriggerSigs(void)
{
   ULONG trigs = 0;

   if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C ) trigs |= DNETC_MSG_SHUTDOWN;
   if (TriggerPort) {
      #ifndef __PPC__
      /*
      ** 68K
      */
      struct TriggerMessage *msg;
      while ((msg = (struct TriggerMessage *)GetMsg(TriggerPort))) {
         trigs |= msg->tm_TriggerType;
         ReplyMsg((struct Message *)msg);
      }
      #elif !defined(__POWERUP__)
      /*
      ** WarpOS
      */
      struct TriggerMessage *msg;
      while ((msg = (struct TriggerMessage *)GetMsgPPC(TriggerPort))) {
         trigs |= msg->tm_TriggerType;
         ReplyMsgPPC((struct Message *)msg);
      }
      #else
      /*
      ** PowerUp
      */
      void *msg;
      while ((msg = PPCGetMessage(TriggerPort))) {
         trigs |= PPCGetMessageAttr(msg,PPCMSGTAG_MSGID);
         PPCReplyMessage(msg);
      }
      #endif
   }

   return(trigs);
}

int amigaPutTriggerSigs(ULONG trigs)
{
   int done = -1;

   #ifndef __PPC__
   /*
   ** 68K
   */
   struct TriggerMessage msg;
   if ((msg.tm_ExecMsg.mn_ReplyPort = CreateMsgPort())) {
      struct MsgPort *port;
      msg.tm_TriggerType = trigs;
      msg.tm_ExecMsg.mn_Node.ln_Type = NT_MESSAGE;
      msg.tm_ExecMsg.mn_Length = sizeof(struct TriggerMessage);

      Forbid();
      if ((port = FindPort("dnetc")) && port != TriggerPort) {
         PutMsg(port,(struct Message *)&msg);
      }
      Permit();
      if (port && port != TriggerPort) {
         WaitPort(msg.tm_ExecMsg.mn_ReplyPort);
         GetMsg(msg.tm_ExecMsg.mn_ReplyPort);
         done = 1;
      }
      else {
         done = 0;
      }
      DeleteMsgPort(msg.tm_ExecMsg.mn_ReplyPort);
   }
   #else
   #ifndef __POWERUP__
   /*
   ** WarpOS
   */
   struct TriggerMessage msg;
   struct MsgPortPPC *replyport;
   if ((replyport = CreateMsgPortPPC())) {
      msg.tm_ExecMsg.mn_ReplyPort = (struct MsgPort *)replyport;
      msg.tm_ExecMsg.mn_Node.ln_Type = NT_MESSAGE;
      msg.tm_ExecMsg.mn_Length = sizeof(struct TriggerMessage);
      msg.tm_TriggerType = trigs;

      struct MsgPortPPC *port;
      if ((port = FindPortPPC("dnetc")) && port != TriggerPort) {
         PutMsgPPC(port,(struct Message *)&msg);
         WaitPortPPC(replyport);
         GetMsgPPC(replyport);
         done = 1;
      }
      else {
         done = 0;
      }
      DeleteMsgPortPPC(replyport);
   }
   #else
   /*
   ** PowerUp
   */
   struct TagItem tags[2] = { {PPCPORTTAG_NAME, (ULONG)"dnetc"}, {TAG_END,0} };
   void *msg, *replyport, *port;
   if ((replyport = PPCCreatePort(NULL))) {
      if ((msg = PPCCreateMessage(replyport,0))) {
         done = 0;
         if ((port = PPCObtainPort(tags))) {
            if (port != TriggerPort) {
               if (PPCSendMessage(port,msg,NULL,0,trigs)) {
                  PPCWaitPort(replyport);
                  PPCGetMessage(replyport);
                  done = 1;
	       }
	    }
            PPCReleasePort(port);
	 }
         PPCDeleteMessage(msg);
      }
      PPCDeletePort(replyport);
   }
   #endif
   #endif

   return(done);
}

extern unsigned long *__stdfiledes;	// libnix internal

int ftruncate(int fd, int newsize)
{
   int result = SetFileSize(__stdfiledes[fd],newsize,OFFSET_BEGINNING);
   if (result != -1) return 0;
   return(result);
}

#if defined(__PPC__) && defined(__POWERUP__)
/*
** libnix for PowerUp has broken strncpy().  Was causing NetResolve to trash
** addresses, confrwv.cpp to crash, and probably heaps of other problems!
*/
char *strncpy(char *d, const char *s, size_t n)
{
    char c;
    char *base = d;

    while(n && (c = *s)) {
	*d = c;
	++s;
	++d;
	--n;
    }
    if (n)
	*d = 0;
    return(base);
}

/*
** libnix for PowerUp is missing chmod()
*/
#include <sys/types.h>
#include <sys/stat.h>
#include <dos/dosextens.h>

__BEGIN_DECLS
extern void __seterrno(void);
extern char *__amigapath(const char *path);
__END_DECLS

int chmod(const char *name, mode_t mode)
{ int ret;

  if((name=__amigapath(name))==NULL)
    return -1;

  if ((ret=~(SetProtection((STRPTR)name,((mode&S_IRUSR?0:FIBF_READ)|
                                         (mode&S_IWUSR?0:FIBF_WRITE|FIBF_DELETE)|
                                         (mode&S_IXUSR?0:FIBF_EXECUTE)|
                                         (mode&S_IRGRP?FIBF_GRP_READ:0)|
                                         (mode&S_IWGRP?FIBF_GRP_WRITE|FIBF_GRP_DELETE:0)|
                                         (mode&S_IXGRP?FIBF_GRP_EXECUTE:0)|
                                         (mode&S_IROTH?FIBF_OTR_READ:0)|
                                         (mode&S_IWOTH?FIBF_OTR_WRITE|FIBF_OTR_DELETE:0)|
                                         (mode&S_IXOTH?FIBF_OTR_EXECUTE:0))))))
    __seterrno();
                              
  return ret;
}
#endif /* defined(__PPC__) && defined(__POWERUP__) */
