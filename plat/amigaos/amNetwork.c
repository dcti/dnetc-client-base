/* Created by Oliver Roberts <oliver@futaura.co.uk>
**
** $Id: amNetwork.c,v 1.1.2.2 2001/03/19 19:18:55 oliver Exp $
**
** ----------------------------------------------------------------------
** This file contains Amiga specific networking code, including socket
** lib init and dial-on-demand.
** ----------------------------------------------------------------------
*/

#include "amiga.h"
#include <net/if.h>

#ifdef __PPC__
#pragma pack(2)
#endif

#include <proto/socket.h>
#include <proto/miami.h>
#include <proto/rexxsyslib.h>

#ifdef __PPC__
#pragma pack()
#endif

static int __CheckOnlineStatus(void);
BOOL GenesisIsOnline(LONG flags);

#ifdef __PPC__
#define GenesisIsOnline(cmd) \
	LP1(0x78, BOOL, GenesisIsOnline, LONG, cmd, d0, \
	, GenesisBase, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)
#else
#define GenesisIsOnline(cmd) \
	LP1(0x78, BOOL, GenesisIsOnline, LONG, cmd, d0, \
	, GenesisBase)
#endif

#define PREVENT_SOCKETLIB_DISKIO

struct Library *SocketBase = NULL;
struct Library *MiamiBase = NULL;
struct Library *GenesisBase = NULL;

static int SocketRefCnt = 0;

/*
** Open bsdsocket.library, if not already opened (never use OpenLibrary to
** open this lib directly!)
*/
BOOL amigaOpenSocketLib(void)
{
   BOOL success = TRUE;

   if (SocketRefCnt == 0) {
      BOOL libnotinmem = FALSE;

      #ifdef PREVENT_SOCKETLIB_DISKIO
      #ifndef __PPC__
      Forbid();
      libnotinmem = (FindName(&SysBase->LibList,"bsdsocket.library") == NULL);
      Permit();
      #else
      #ifdef __POWERUP__
      /* disable for now - may have something to do with stability problems */
      //libnotinmem = (PPCFindNameSync(&SysBase->LibList,"bsdsocket.library") == NULL);
      #else
      ULONG skey = Super();
      libnotinmem = (FindNamePPC(&SysBase->LibList,"bsdsocket.library") == NULL);
      User(skey);
      #endif
      #endif
      #endif

      if (libnotinmem || !(SocketBase = OpenLibrary((unsigned char *)"bsdsocket.library",4UL)))
         success = FALSE;
   }

   if (success) SocketRefCnt++;

   return(success);
}

/*
** Close bsdsocket.library - must be paired with successful amigaOpenSocketLib()
*/
void amigaCloseSocketLib(void)
{
   if (SocketRefCnt > 0) {
      if (--SocketRefCnt == 0) {
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

   if (notlurking) {
      MiamiBase = OpenLibrary((unsigned char *)"miami.library",11UL);
      if (!MiamiBase) {
         GenesisBase = OpenLibrary((unsigned char *)"genesis.library",0UL);
      }
      success = __CheckOnlineStatus();
   }

   if (success) success = amigaOpenSocketLib();

   if (!success) amigaNetworkingDeinit();

   return(success);
}

void amigaNetworkingDeinit(void)
{
   amigaCloseSocketLib();

   if (GenesisBase) {
      CloseLibrary(GenesisBase);
      GenesisBase = NULL;
   }
   if (MiamiBase) {
      CloseLibrary(MiamiBase);
      MiamiBase = NULL;
   }
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
   else if (GenesisBase) {
      online = GenesisIsOnline(NULL);
   }
   else {
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
   struct RxsLib *RexxSysBase;
   struct MsgPort *arexxreply,*port;
   struct RexxMsg *arexxmsg;
   int success = 0, genesis = 0;
   char cmdbuf[64];
 
   Forbid();
   port = FindPort("MIAMI.1");
   if (!port) {
      port = FindPort("GENESIS");
      genesis = 1;
   }
   Permit();
 
   if (port) {
      if ((RexxSysBase = (struct RxsLib *)OpenLibrary((unsigned char *)"rexxsyslib.library",0))) {
         if ((arexxreply = CreateMsgPort())) {
            if ((arexxmsg = CreateRexxMsg(arexxreply,NULL,NULL))) {
               arexxmsg->rm_Action = RXCOMM;
               sprintf(cmdbuf,"%s%s%s", online ? "ONLINE" : "OFFLINE", (strlen(ifacename) > 0 && strcmp(ifacename,"mi0") != 0) ? " " : "", ifacename);
               if ((arexxmsg->rm_Args[0] = (char *)CreateArgstring((unsigned char *)cmdbuf,strlen(cmdbuf)))) {
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
}
