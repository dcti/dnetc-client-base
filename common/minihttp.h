// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __MINIHTTP_H__
#define __MINIHTTP_H__ "@(#)$Id: minihttp.h,v 1.1 2000/06/03 03:24:39 jlawson Exp $"

#include "cputypes.h"
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "netio.h"
#include "autobuff.h"


//response->SetHeader("Connection", (bKeepAliveHTTP ? "Keep-Alive" : "Close"));
//response->SetHeader("Content-Type", "application/binary-stream");


enum MiniHttpVersionType
{
  VERSION_UNKNOWN = 0,      // all other fields invalid
  VERSION_0_9   = 0x009,    // not used, unsupported.
  VERSION_1_0   = 0x100,
  VERSION_1_1   = 0x101
};

// --------------------------------------------------------------------------

struct MiniHttpRequest
{
  // request version/status.
  MiniHttpVersionType RequestVersion;

  // request line.
  char RequestMethod[16];
  AutoBuffer RequestUri;

  // headers.
  AutoBuffer Headers;       // includes trailing blank line.

  // content body.
  s32 ContentLength;        // copied from 'Headers'.
  AutoBuffer ContentBody;

  // methods.
  bool GetHeader(const char *headername, AutoBuffer *value);
  bool GetFormVariable(const char *variablename, AutoBuffer *value);
  bool GetDocumentUri(AutoBuffer *value);
};

// --------------------------------------------------------------------------

struct MiniHttpResponse
{
  // status line.
  unsigned int statuscode;

  // headers.
  AutoBuffer Headers;       // does not include trailing blank line.

  // content body.
  AutoBuffer ContentBody;

  // methods.
  MiniHttpResponse() : statuscode(200) {};
  void SetHeader(const char *headername, const char *value);
  const char *GetStatusText(void);

};

// --------------------------------------------------------------------------

class MiniHttpDaemonConnection
{
public:
  MiniHttpDaemonConnection(SOCKET fd, u32 addr);
  ~MiniHttpDaemonConnection(void);

  // raw network input and output called during network servicing.
  bool FetchIncoming(void);
  bool FlushOutgoing(void);

  // state information access.
  bool IsConnected(void) { return (fd != INVALID_SOCKET); }
  SOCKET GetSocket(void) { return fd; }
  u32 GetAddress(void) { return addr; }
  time_t GetLastActivity(void)
    {
      time_t curTime = time(NULL);
      if (curTime < activity) activity = curTime;
      return (time_t) (curTime - activity);
    }

  // incoming packet retrieval.
  bool IsComplete(void)
    { return haveStatusLine && haveHeaders &&
        (ContentLength <= 0 ||
         (int) (inwaiting.GetLength() - steplinepos) >= ContentLength); }

  bool FetchRequest(MiniHttpRequest *request);

  // outgoing packet transmission.
  bool HavePendingData(void)
    { return (outwaiting.GetLength() > 0); }

  bool QueueResponse(MiniHttpResponse *response);


protected:

  // internal helper to analyze buffered data that has been received.
  bool ReparseIncoming(void);


  // parsed incoming headers and body.
  bool haveStatusLine;
  bool haveHeaders;
  int ContentLength;                    // valid if >= 0
  unsigned int steplinepos;             // index into 'inwaiting'
  AutoBuffer inwaiting;
  AutoBuffer ReqMethod, ReqUri;         // valid if 'haveStatusLine'
  MiniHttpVersionType ReqVerType;       // valid if 'haveStatusLine'



  // encoded data waiting to be sent out.
  AutoBuffer outwaiting;


  // other internal information.
  u32 addr;                   // the client's IP information
  time_t activity;            // the timestamp of the last activity
  SOCKET fd;                  // socket handle of this connection
};

// --------------------------------------------------------------------------

#endif

