/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amNetwork.c,v 1.2.4.2 2004/01/08 21:00:48 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * This file contains Amiga specific networking code, including socket
 * lib init and dial-on-demand.
 * ----------------------------------------------------------------------
*/

#include "amiga.h"
#include <sys/socket.h>
#include <net/if.h>

#ifdef __OS3PPC__
#pragma pack(2)
#endif

#ifdef __amigaos4__
#include <proto/bsdsocket.h>
#else
#include <proto/socket.h>
#endif

#ifndef NO_MIAMI
#include <proto/miami.h>
#endif

#include <proto/rexxsyslib.h>

#ifdef __OS3PPC__
#pragma pack()
#endif

static int __CheckOnlineStatus(void);
BOOL GenesisIsOnline(LONG flags);

#ifdef __PPC__ /* MorphOS too! */
#define GenesisIsOnline(cmd) \
	LP1(0x78, BOOL, GenesisIsOnline, LONG, cmd, d0, \
	, GenesisBase, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)
#else
#define GenesisIsOnline(cmd) \
	LP1(0x78, BOOL, GenesisIsOnline, LONG, cmd, d0, \
	, GenesisBase)
#endif

#ifndef __amigaos4__
#define PREVENT_SOCKETLIB_DISKIO
#endif

struct Library *SocketBase = NULL;
#ifndef NO_MIAMI
struct Library *MiamiBase = NULL;
#endif
struct Library *GenesisBase = NULL;
static struct Library *TSocketBase = NULL;

#ifdef __amigaos4__
struct SocketIFace *ISocket = NULL;
int h_errno = 0;
#endif

static int SocketRefCnt = 0;

/*
** Open bsdsocket.library, if not already opened (never use OpenLibrary to
** open this lib directly!)
*/
BOOL amigaOpenSocketLib(void)
{
   BOOL success = TRUE;

   if (SocketRefCnt == 0 && !SocketBase) {
      BOOL libnotinmem = FALSE;

      #ifdef PREVENT_SOCKETLIB_DISKIO
      #ifndef __OS3PPC__
      Forbid();
      libnotinmem = (FindName(&SysBase->LibList,"bsdsocket.library") == NULL);
      Permit();
      #elif defined(__POWERUP__)
      /* disable for now - may have something to do with stability problems */
      //libnotinmem = (PPCFindNameSync(&SysBase->LibList,"bsdsocket.library") == NULL);
      #else
      ULONG skey = Super();
      libnotinmem = (FindNamePPC(&SysBase->LibList,"bsdsocket.library") == NULL);
      User(skey);
      #endif
      #endif /* PREVENT_SOCKETLIB_DISKIO */

      if (libnotinmem || !(SocketBase = OpenLibrary("bsdsocket.library",4UL)))
         success = FALSE;

      #ifdef __amigaos4__
      if (success) {
         if (!(ISocket = (struct SocketIFace *)GetInterface( SocketBase, "main", 1L, NULL ))) {
            CloseLibrary(SocketBase);
            SocketBase = NULL;
            success = FALSE;
	 }
      }
      #endif
   }

   if (success) SocketRefCnt++;

   return(success);
}

/*
** Close bsdsocket.library - must be paired with successful amigaOpenSocketLib()
*/
void amigaCloseSocketLib(BOOL delayedclose)
{
   if (SocketRefCnt > 0) {
      if ((--SocketRefCnt == 0) && !delayedclose) {
         #ifdef __amigaos4__
         DropInterface((struct Interface *)ISocket);
         ISocket = NULL;
         #endif
         CloseLibrary(SocketBase);
         SocketBase = NULL;
      }
   } 
}

/*
** This init routine tries to open miami or genesis libs if not lurking,
** which are used for simple online/offline checking.  These libs are not
** required if lurking is enabled since online/offline checks are done via
** bsdsocket.library instead.
*/
int amigaNetworkingInit(BOOL notlurking)
{
   int success = 1;

   // used to detect whether TermiteTCP is in use
   #ifndef __amigaos4__
   TSocketBase = OpenLibrary("tsocket.library",0UL);
   #endif

   if (notlurking) {
      #ifndef NO_MIAMI
      MiamiBase = OpenLibrary((unsigned char *)"miami.library",11UL);
      if (!MiamiBase) {
         GenesisBase = OpenLibrary("genesis.library",0UL);
      }
      #elif !defined(__amigaos4__)
      GenesisBase = OpenLibrary("genesis.library",0UL);
      #endif

      success = __CheckOnlineStatus();
   }

   if (success) success = amigaOpenSocketLib();

   if (!success) amigaNetworkingDeinit();

   return(success);
}

void amigaNetworkingDeinit(void)
{
   amigaCloseSocketLib(FALSE);

   if (GenesisBase) {
      CloseLibrary(GenesisBase);
      GenesisBase = NULL;
   }
#ifndef NO_MIAMI
   if (MiamiBase) {
      CloseLibrary(MiamiBase);
      MiamiBase = NULL;
   }
#endif
   if (TSocketBase) {
      CloseLibrary(TSocketBase);
      TSocketBase = NULL;
   }
}

BOOL amigaIsTermiteTCP(void)
{
   return(TSocketBase != NULL);
}

/*
** Simple check to see if any interfaces are "online" when lurking is disabled.
** When lurking is enabled, IsConnected() does a better job so we always return
** 1 in this case.
*/
int amigaIsNetworkingActive(void)
{
   int online = 0;

   if (SocketBase) {
      online = __CheckOnlineStatus();
   }

   return(online);
}

static int __CheckOnlineStatus(void)
{
   int online;

   #ifndef NO_MIAMI
   if (MiamiBase) {
      online = MiamiIsOnline(NULL);
      if (!online) {
         struct if_nameindex *name,*nameindex;
         if ((nameindex = if_nameindex())) {
            name = nameindex;
            while (!(name->if_index == 0 && name->if_name == NULL)) {
               if ((strncmp(name->if_name,"lo",2) != 0) && (MiamiIsOnline(name->if_name))) {
                  online = 1;
                  break;
               }
               name++;
            }
            if_freenameindex(nameindex);
         }
      }
   }
   else
   #elif !defined(__amigaos4__)
   if (GenesisBase) {
      online = GenesisIsOnline(NULL);
   }
   else
   #endif
   {
      online = 1;  // assume online (lurking, or no Miami / Genesis)
   }

   return(online);
}

/*
** Dial-on-Demand support - put Miami, MiamiDx or Genesis on/offline
**
** returns 0 for failure, 1 for success (synchronous), 2 for success (asynchronous)
*/
int amigaOnOffline(int online, char *ifacename)
{
   #ifndef __amigaos4__
   struct RxsLib *RexxSysBase;
   struct MsgPort *arexxreply,*port;
   struct RexxMsg *arexxmsg;
   int success = 0, genesis = 0;
   char cmdbuf[64];
 
   Forbid();

   #ifndef NO_MIAMI
   port = FindPort("MIAMI.1");
   if (!port) {
      port = FindPort("GENESIS");
      genesis = 1;
   }
   #elif !defined(__amigaos4__)
   port = FindPort("GENESIS");
   genesis = 1;
   #endif

   Permit();
 
   if (port) {
      if ((RexxSysBase = (struct RxsLib *)OpenLibrary("rexxsyslib.library",0))) {
         if ((arexxreply = CreateMsgPort())) {
            if ((arexxmsg = CreateRexxMsg(arexxreply,NULL,NULL))) {
               arexxmsg->rm_Action = RXCOMM;
               sprintf(cmdbuf,"%s%s%s", online ? "ONLINE" : "OFFLINE", (strlen(ifacename) > 0 && strcmp(ifacename,"mi0") != 0) ? " " : "", ifacename);
               if ((arexxmsg->rm_Args[0] = (char *)CreateArgstring(cmdbuf,strlen(cmdbuf)))) {
                  Forbid();
                  if ((port = FindPort((genesis ? "GENESIS" : "MIAMI.1")))) {
                     PutMsg(port,(struct Message *)arexxmsg);
                  }
                  Permit();
                  if (port) {
                     WaitPort(arexxreply);
                     while (GetMsg(arexxreply));
                     // Note Miami online/offline commands are synchronous, and
                     // Genesis' are asynchronous (so we have to wait)
                     success = genesis ? 2 : 1;
	          }
                  DeleteArgstring(arexxmsg->rm_Args[0]);
                  DeleteRexxMsg(arexxmsg);
	       }
               else {
                  DeleteRexxMsg(arexxmsg);
 	       }
 	    }
            DeleteMsgPort(arexxreply);
	 }
         CloseLibrary((struct Library *)RexxSysBase);
      }
   }

   return(success);
   #else
   return 0;
   #endif
}
