// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Id: lurk.h,v 1.18 1999/04/05 13:51:10 jlawson Exp $

#ifndef __LURK_H__
#define __LURK_H__

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

class Lurk
{
public:

int lurkmode;
  // Mode of operation
  // 0 = disabled
  // 1 = CONNECT_LURK = fill buffers while online, try to connect when emptied
  // 2 = CONNECT_LURKONLY = fill buffers while online, never connect

int dialwhenneeded;      // 0 = we don't handle dial, !0 we dial/hangup
char connprofile[64];    // Used by win32 for name of DUN connection to use.
char connifacemask[64];  // a list of interfaces to monitor for online state
char connstartcmd[64];   // name of script to call to start connection
char connstopcmd[64];    // name of script to call to stop connection

const char **GetConnectionProfileList(void); //get the list of conn profiles
int GetCapabilityFlags(void);      //return supported CONNECT_* bits 

  //methods used for lurk
int CheckIfConnectRequested(void); // -> 0=no, !0=yes
int CheckForStatusChange(void);    // -> 0 = nochange, !0 connection dropped

  //methods used for dialup initiation/hangup
int DialIfNeeded(int ignore_lurkonly_flag); // -> 0=success, !0 = failure
int HangupIfNeeded(void);          // -> 0=success, !0 = failure

  //initialization/stop.
int Start(void);                   // Start -> 0=success, !0 = failure
int Stop(void);                    // Stop  -> 0=success, !0 = failure

int IsConnected(void);   // quietly! check if connected-> !0 = connected

Lurk(); 
~Lurk();

protected:

int mask_include_all, mask_default_only; //what does the mask tell us?
const char *ifacestowatch[(64/2)+1]; //(sizeof(connifacemask)/sizeof(char *))+1
char ifacemaskcopy[64];            //sizeof(connifacemask)

char conndevice[35];        //name of the device a connection was detected on
                            //informational use only
int islurkstarted;          //was lurk.Start() successful?
int lastcheckshowedconnect; //the connect state at the last CheckIfConnectRequested()
int dohangupcontrol;        //if we dialed, we're welcome to hangup

};

extern Lurk dialup;

#endif /* __LURK_H__ */

