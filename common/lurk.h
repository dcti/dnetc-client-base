// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: lurk.h,v $
// Revision 1.15  1999/04/01 02:59:12  cyp
// made IsConnected() public so that we can *quietly* check the state.
//
// Revision 1.14  1999/02/09 23:41:39  cyp
// Lurk iface mask changes: a) default iface mask no longer needs to be known
// outside lurk; b) iface mask now supports wildcards; c) redid help text.
//
// Revision 1.13  1999/02/09 03:17:59  remi
// Added Lurk::GetDefaultIFaceMask(void).
//
// Revision 1.12  1999/02/07 16:00:09  cyp
// Lurk changes: genericified variable names, made less OS-centric.
//
// Revision 1.11  1999/02/06 10:42:55  remi
// - the default for dialup.ifacestowatch is now 'ppp0:sl0'.
// - #ifdef'ed dialup.ifacestowatch (only Linux at the moment)
// - modified a bit the help text in confopt.cpp
//
// Revision 1.10  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the lookback one) will trigger the online
// status.
//
// Revision 1.9  1999/02/04 07:47:06  cyp
// Re-import from proxy base. Cleaned up. Added linux and win16 support.
//
// Revision 1.3  1999/01/28 19:53:11  trevorh
// Added lurkonly support for OS/2
//
// Revision 1.2  1999/01/24 23:14:58  dbaker
// copyright 1999 changes
//
// Revision 1.1  1998/12/30 01:45:11  jlawson
// added lurk code from client.
//
// Revision 1.6  1998/11/12 13:09:03  silby
// Added a stop function, made start and stop public.
//
// Revision 1.5  1998/10/03 03:24:36  cyp
// Fixed a broken #endif (had trailing comment without //). Added
// #ifndef __LURK_H__ /#define __LURK_H__ /#endif nesting.
//
#ifndef __LURK_H__
#define __LURK_H__

#define CONNECT_LURK         0x01
#define CONNECT_LURKONLY     0x02
#define CONNECT_IFACEMASK    0x04 /* limit the interfaces to watch for conn on */
#define CONNECT_DODBYSCRIPT  0x08
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

#endif //__LURK_H__
