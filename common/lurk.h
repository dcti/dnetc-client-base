// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __LURK_H__
#define __LURK_H__ "@(#)$Id: lurk.h,v 1.21.2.6 2000/10/06 00:40:15 mfeiri Exp $"

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


class Lurk
{
public:
  Lurk(); 
  ~Lurk();
  // initialization/stop. -> 0=success, !0 = failure
  int Start(int nonetworking, struct dialup_conf *);  
  int Stop(void);
  
  // state info
  int IsWatching(void); //Start() was ok and CONNECT_LURK|LURKONLY|DOD */
  int IsWatcherPassive(void); //Start was ok and lurkmode is CONNECT_LURKONLY
  int IsConnected(void); // test (and say) connection state
  const char **GetConnectionProfileList(void); //get the list of conn profiles
  int GetCapabilityFlags(void);      //return supported CONNECT_* bits 

  // methods used for dialup initiation/hangup
  int DialIfNeeded(int ignore_lurkonly_flag); // -> 0=success, !0 = failure
  int HangupIfNeeded(void);          // -> 0=success, !0 = failure

  #if 0 /* unused - IsConnected() does everything we need */
  // methods used for lurk
  int CheckIfConnectRequested(void); // -> 0=no, !0=yes
  int CheckForStatusChange(void);    // -> 0 = nochange, !0 connection dropped
  #endif
protected:
  int InternalIsConnected(void); /* workhorse */
  int islurkstarted;      //was lurk.Start() successful?
  struct dialup_conf conf; //local copy of config. Initialized by Start()

  int mask_include_all, mask_default_only; //what does the mask tell us?
  const char *ifacestowatch[(64/2)+1]; //(sizeof(connifacemask)/sizeof(char *))+1
  char ifacemaskcopy[64];            //sizeof(connifacemask)

  int showedconnectcount; //used by CheckIfConnectRequested()
  int dohangupcontrol;    //if we dialed, we're welcome to hangup

  #ifndef CLIENT_OS /* catch static struct problems _now_ */
  #error "CLIENT_OS isn't defined yet. cputypes.h must be #included before lurk.h"
  #endif
  #if (CLIENT_OS != OS_WIN16) && (CLIENT_OS != OS_MACOS)
  #define LURK_MULTIDEV_TRACK
  #endif
  
  #ifdef LURK_MULTIDEV_TRACK
  char conndevices[64*32];
  #else
  //name of the device a connection was detected on informational use only
  char conndevice[35];        
  char previous_conndevice[35]; //copy of last conndevice
  #endif
};

extern Lurk dialup;

#endif /* __LURK_H__ */

