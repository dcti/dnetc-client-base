// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __LURK_H__
#define __LURK_H__ "@(#)$Id: lurk.h,v 1.21.2.7 2000/11/12 04:31:19 cyp Exp $"

/* lurk: fetch/flush if modem goes online but also go online if fetch/flush needed */
#define CONNECT_LURK         0x01 

/* lurkonly: connect only if modem goes online. equivalent to lurk+offlinemode? */
#define CONNECT_LURKONLY     0x02

/* ifacemask: limit the interfaces to watch for conn on */
#define CONNECT_IFACEMASK    0x04 

/* dodbyscript: run this script to initiate a dialup connection */
#define CONNECT_DODBYSCRIPT  0x08

/* dodbyprofile: use this profile when initiating a dialup connection */
#define CONNECT_DODBYPROFILE 0x10
#define CONNECT_DOD          (CONNECT_DODBYSCRIPT|CONNECT_DODBYPROFILE)

struct dialup_conf
{
  int lurkmode;            // 0 = disabled, 1=CONNECT_LURK, 2=CONNECT_LURKONLY
  int dialwhenneeded;      // 0 = we don't handle dial, !0 we dial/hangup
  char connprofile[64];    // Used by win32 for name of DUN connection to use.
  char connifacemask[64];  // a list of interfaces to monitor for online state
  char connstartcmd[64];   // name of script to call to start connection
  char connstopcmd[64];    // name of script to call to stop connection
};

// initialization/stop. -> 0=success, !0 = failure
int LurkStart(int nonetworking, struct dialup_conf *);  
int LurkStop(void);
  
// state info
int LurkIsWatching(void); //Start() was ok and CONNECT_LURK|LURKONLY|DOD */
int LurkIsWatcherPassive(void); //Start was ok and lurkmode is CONNECT_LURKONLY
int LurkIsConnected(void); // test (and say) connection state
const char **LurkGetConnectionProfileList(void); //get the list of conn profiles
int LurkGetCapabilityFlags(void);      //return supported CONNECT_* bits 

  // methods used for dialup initiation/hangup
int LurkDialIfNeeded(int ignore_lurkonly_flag); // -> 0=success, !0 = failure
int LurkHangupIfNeeded(void);          // -> 0=success, !0 = failure

#endif /* __LURK_H__ */

