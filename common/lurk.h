// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// These are config variables, everything needs access to them


class Lurk
  {
public:

s32 lurkmode;
  // Mode of operation
  // 0 = disabled
  // 1 = fill buffers while online, try to connect when emptied
  // 2 = fill buffers while online, never connect
s32 dialwhenneeded;
  // 0 = Don't dial, let autodial handle it or fail
  // 1 = Have the client manually dial/hangup when a flush happens.
char connectionname[100];
  // For win32, name of connection to use, perhaps useful for other lurkers.

s32 CheckIfConnectRequested(void);
  // Returns the possible values of connectrequested
s32 Start(void);
  // Initializes Lurk Mode -> 0=success, -1 = failed
s32 Status(void);
  // Checks status of connection -> !0 = connected
s32 InitiateConnection(void);
  // Initiates a dialup connection
  // 0 = already connected, 1 = connection started,
  // -1 = connection failed
s32 TerminateConnection(void);
  // -1 = connection did not terminate properly, 0 = connection
  // terminated
s32 oldlurkstatus;
  // Status of LurkStatus as of the last check.
s32 islurkstarted;
  // 0 = lurk functionality has not been initialized
  // 1 = lurk functionality has been initialized
protected:

friend void Log( const char *format, ...);
friend void LogScreenf( const char *format, ... );
#if defined(NEEDVIRTUALMETHODS)
friend virtual void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#else
friend void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#endif
};
