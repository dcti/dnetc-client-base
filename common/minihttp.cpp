// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

const char *minihttp_cpp(void) {
return "@(#)$Id: minihttp.cpp,v 1.2 2000/07/03 07:17:04 jlawson Exp $"; }


#include "cputypes.h"
#include "netio.h"        // netio_*(), SOCKET
#include "minihttp.h"
#include "sleepdef.h"     // time()
#include "cmpidefs.h"     // toupper()

//#if defined(DEBUG) && !defined(NDEBUG)
//#include <assert.h>
//#else
//#define assert(x)
//#endif



// Maximum incomplete network buffer size (bytes) for header or body each.
#define MAXINCOMPLETESIZE   4096

// Maximum amount to read per cycle when servicing a socket.
#define READAMOUNT    512

// Define to enable handling of multi-line split headers.
#define ALLOWMULTILINEHEADERS



// --------------------------------------------------------------------------

// If the specified linebuffer matches the desired header name, then a
// non-zero integer will be returned, otherwise 0 will be returned.
// When a non-zero value is returned, that number will represent the
// number of leading chars from linebuffer that should be skipped so
// that the header value can be parsed.

static int __strcmpheader(const char *linebuffer, const char *desired)
{
  assert(linebuffer != NULL && desired != NULL && desired[0] != '\0');
  assert(strpbrk(desired, ": \t\r\n") == NULL);
  const char *p = linebuffer;

  while (*p != '\0')
  {
    if (*desired == '\0')
    {
      if (*p++ == ':')
      {
        // great, we found the header.  skip some whitespace.
        while (*p == ' ' || *p == '\t') p++;
        return (p - linebuffer);
      }
      break;
    }
    else if (toupper(*p) != toupper(*desired))
    {
      break;
    }
    p++; desired++;
  }
  return 0;   // no match found.
}

// --------------------------------------------------------------------------

// Search through the received list of headers for a specific header name,
// and fill 'value' with the contents of that header.  Returns true when
// the requested header was found, otherwise false if it was not.

bool MiniHttpRequest::GetHeader(const char *headername, AutoBuffer *value)
{
  assert(headername != NULL && headername[0] != 0 && value != NULL);
  assert(strpbrk(headername, ": \t\r\n") == NULL);

  unsigned int lineoffset = 0;
  AutoBuffer workline;
  value->Clear();
  while (Headers.StepLine(&workline, &lineoffset))
  {
    int cmpresult = __strcmpheader(workline.GetHead(), headername);
    if (cmpresult > 0)
    {
      // This line matched the start of the header.  Remove the header
      // name and copy the rest of the line into the value buffer.
      workline.RemoveHead(cmpresult);
      *value = workline;

      // HTTP headers can potentially be split into multiple lines, with
      // subsequent lines marked with leading whitespace.
      #ifdef ALLOWMULTILINEHEADERS
      while (Headers.StepLine(&workline, &lineoffset))
      {
        const char *head = workline.GetHead();
        if (*head == ' ' || *head == '\t')
        {
          const char *p = head + 1;
          while (*p == ' ' || *p == '\t') p++;
          workline.RemoveHead(p - head);
          *value += workline;
        }
        else break;
      }
      #endif
      return true;    // header found.
    }
  }
  return false;   // header not found.
}

// --------------------------------------------------------------------------

static bool __FindFormVariable(const char *formbuffer,
    unsigned int formbufferlen, const char *variablename, AutoBuffer *value)
{
  const char *p = formbuffer;
  const char *pend = &formbuffer[formbufferlen];

  assert(formbuffer != NULL);
  assert(variablename != NULL && *variablename != '\0');
  assert(value != NULL);

  value->Clear();
  while (p < pend && *p)
  {
    bool matched = true;
    const char *varp = variablename;

    // See if this variable matches the one we're looking for.
    for (;;)
    {
      if (p >= pend || *p == '\0' || *p == '=' || *p == '&') {
        if (*varp != '\0') matched = false;
        if (*p == '=') p++;
        break;
      }
      else if (*varp == '\0' || toupper(*p) != toupper(*varp)) {
        matched = false;
        break;
      }
      p++; varp++;
    }

    // If the variable matched, then copy the value into the target.
    if (matched)
    {
      // determine the end of the value.
      const char *valend = p;
      while (valend < pend) {
        if (*valend == '\0' || *valend == '&') break;
        valend++;
      }

      // copy the value into the target buffer.
      if (valend > p) {
        value->Reserve(valend - p + 1);
        memcpy(value->GetTail(), p, valend - p);
        value->GetTail()[valend - p] = '\0';
        value->MarkUsed(valend - p);
      }
      return true;      // variable value found.
    }

    // Advance forward to the next separator, and then one beyond.
    while (p < pend && *p != '\0' && *p++ != '&') {};
  }
  return false;     // variable not found.
}

// --------------------------------------------------------------------------

bool MiniHttpRequest::GetFormVariable(const char *variablename, AutoBuffer *value)
{
  if (!value || !variablename || *variablename == '\0') return false;
  value->Clear();

  // Do checking of GET variables.  We allow GET variables
  // even on POST or other methods.
  const char *pend = RequestUri.GetTail();
  const char *p = RequestUri.GetHead();
  while ( p < pend && *p != '\0' && *p != '?' ) p++;
  if (__FindFormVariable(p, (unsigned int) (pend - p), variablename, value))
    return true;


  // If the request was actually a POST, then also check the document body.
  if (strcmp(RequestMethod, "POST") == 0)
  {
    AutoBuffer ContentType;
    if (GetHeader("Content-Type", &ContentType) &&
        strcmp(ContentType.GetHead(),
            "application/x-www-form-urlencoded") == 0)
    {
      return __FindFormVariable(ContentBody.GetHead(), ContentBody.GetLength(),
          variablename, value);
    }
  }
  return false;     // variable not found.
}

// --------------------------------------------------------------------------

// Copies the "document URI" portion of the URI (everything before the first
// question mark) into the specified buffer.

bool MiniHttpRequest::GetDocumentUri(AutoBuffer *value)
{
  if (!value) return false;
  value->Clear();

  // Determine the position of the question mark in the URI.
  const char *pend = RequestUri.GetTail();
  const char *phead = RequestUri.GetHead();
  const char *p = phead;
  while ( p < pend && *p && *p != '?' ) p++;
  if (p == phead) return false;

  // Copy the document portion.
  unsigned int urilen = p - phead;
  value->Reserve(urilen + 1);
  memcpy(value->GetTail(), phead, urilen);
  value->GetTail()[urilen] = '\0';
  value->MarkUsed(urilen);

  return true;
}

// --------------------------------------------------------------------------

MiniHttpDaemonConnection::MiniHttpDaemonConnection(SOCKET _fd, u32 _addr) :
  addr(_addr), fd(_fd), haveHeaders(false), haveStatusLine(false),
  ContentLength(-1), steplinepos(0)
{
  assert(fd != INVALID_SOCKET);
  activity = time(NULL);
}

// --------------------------------------------------------------------------

MiniHttpDaemonConnection::~MiniHttpDaemonConnection(void)
{
  if (fd != INVALID_SOCKET)
    netio_close(fd);
}

// --------------------------------------------------------------------------

// Immediately attempts to read data from the network into the
// incoming data buffers.  This function will block if there is
// no data waiting to be read from the socket, so it should only
// be called if the network socket functions have indicated that
// the socket is readable.
// returns true on success, or false on failure (socket closed)

bool MiniHttpDaemonConnection::FetchIncoming(void)
{
  if (IsConnected() &&
      inwaiting.GetLength() < MAXINCOMPLETESIZE)
  {
    // Read as much as we can into the autobuffer.
    inwaiting.Reserve(READAMOUNT);
    int result = netio_lrecv(fd, inwaiting.GetTail(),
                            inwaiting.GetTailSlack(), false);
    if (result > 0)
    {
      // The read was successful, so mark that data as present.
      inwaiting.MarkUsed(result);
      activity = time(NULL);

      // Scan the data and determine if a full request has been
      // received yet.  If we have not received a full request and
      // the pending size is already over the maximum size then
      // that is an error and success should not be returned.
      if (!ReparseIncoming())
        return false;         // parse error.
              // but we probably should send an HTTP error.

      if (IsComplete() ||
          inwaiting.GetLength() < MAXINCOMPLETESIZE)
        return true;      // no errors and still acceptable size.
    }
    else
    {
      // connection has just been closed.
      netio_close(fd);
      return false;     // network error.
    }
  }
  return false;
}

// --------------------------------------------------------------------------

// Scans the received data buffer and determines if a full HTTP request
// has been received yet.  This is done by looking for the blank line
// that separates the headers and the body in the request.  Additionally,
// while the headers are being scanned, the Content-Length header is looked
// for so that the size of the accompanying body can be known.
// This function does not return any value (use 'IsComplete' to determine
// if a complete request has been determined to now be ready).

bool MiniHttpDaemonConnection::ReparseIncoming(void)
{
  if (!haveHeaders)
  {
    AutoBuffer oneline;
    while (inwaiting.StepLine(&oneline, &steplinepos))
    {
      if (!haveStatusLine)
      {
        // this is the first line.
        AutoBuffer ReqVerString;
        const char *s, *p = oneline.GetHead();

        // Separate out the command method.
        ReqMethod.Clear();
        for (s = p; *p && !isspace(*p); p++) {};
        if (p > s) {
          ReqMethod.Reserve(p - s + 1);
          memcpy(ReqMethod.GetHead(), s, p - s);
          *(ReqMethod.GetHead() + (p - s)) = '\0';
          ReqMethod.MarkUsed(p - s + 1);
        }

        // Skip past the separating whitespace.
        while (*p && isspace(*p)) p++;

        // Separate out the target Uri.
        ReqUri.Clear();
        for (s = p; *p && !isspace(*p); p++) {};
        if (p > s) {
          ReqUri.Reserve(p - s + 1);
          memcpy(ReqUri.GetHead(), s, p - s);
          *(ReqUri.GetHead() + (p - s)) = '\0';
          ReqUri.MarkUsed(p - s + 1);
        }

        // Skip past the separating whitespace.
        while (*p && isspace(*p)) p++;

        // Separate out the request version.
        ReqVerString.Clear();
        for (s = p; *p && !isspace(*p); p++) {};
        if (p > s) {
          ReqVerString.Reserve(p - s + 1);
          memcpy(ReqVerString.GetHead(), s, p - s);
          *(ReqVerString.GetHead() + (p - s)) = '\0';
          ReqVerString.MarkUsed(p - s + 1);
        }

        // Perform some additional verifications on what we have read in.
        if (*p ||                           // must be at end of line.
            ReqMethod.GetLength() < 1 ||    // must have a command.
            ReqMethod.GetLength() >= 16 ||  // can't be too long.
            ReqUri.GetLength() < 1 ||       // must have a URI.
            ReqVerString.GetLength() < 1)   // must have a HTTP version.
        {
          steplinepos = 0;    // next ReparseIncoming will start over.
          return false;       // Bad request
        }
        if (strcmp(ReqVerString.GetHead(), "HTTP/1.0") == 0)
          ReqVerType = VERSION_1_0;
        else if (strcmp(ReqVerString.GetHead(), "HTTP/1.1") == 0)
          ReqVerType = VERSION_1_1;
        else
        {
          ReqVerType = VERSION_UNKNOWN;
          steplinepos = 0;    // next ReparseIncoming will start over.
          return false;       // Bad request
        }

        haveStatusLine = true;
        continue;
      }
      if (oneline.GetLength() == 0)
      {
        // found the blank line separator.
        haveHeaders = true;
        break;
      }
      if (ContentLength < 0)
      {
        int cmpresult = __strcmpheader(oneline.GetHead(), "Content-Length");
        if (cmpresult > 0)
        {
          oneline.RemoveHead(cmpresult);
          ContentLength = atoi(oneline.GetHead());
          if (ContentLength < 0) ContentLength = 0;   // bad length.
        }
      }
    }
  }
  return true;
}

// --------------------------------------------------------------------------

// Immediately flushes as much waiting data as possible over the
// network connection.  This function may block on network I/O if
// it was not safe to send.
// returns true on success, or false on failure (socket closed)

bool MiniHttpDaemonConnection::FlushOutgoing(void)
{
  if (IsConnected())
  {
    if (HavePendingData())
    {
      int result = netio_lsend(fd, outwaiting.GetHead(),
                              outwaiting.GetLength());
      if (result > 0)
      {
        // remove what was sent.
        outwaiting.RemoveHead(result);
        activity = time(NULL);
        return true;      // successful.
      }
      else
      {
        netio_close(fd);
        return false;     // network error.  disconnecting.
      }
    }
    return true;  // no pending data to send.
  }
  return false;   // not connected
}

// --------------------------------------------------------------------------

// Analyzes the pending data received from the client and parses and
// extracts the first request into the 'request' argument.  This function
// can assumed to be successful if 'IsComplete' returns true.  Once the
// request is extracted and copied into the 'request', it is removed
// from the outstanding data queue.  If the argument 'request' is NULL,
// then the first pending request is discarded without being copied.
// Returns true if a whole request was found, or false if a complete
// request has not yet been received from the client.

bool MiniHttpDaemonConnection::FetchRequest(MiniHttpRequest *request)
{
  assert(request != NULL);
  if (!IsComplete()) return false;

  // transfer the complete request into the target structure.
  if (request != NULL)
  {
    assert(haveStatusLine != false && haveHeaders != false);
    assert(ContentLength <= 0 ? steplinepos <= inwaiting.GetLength() :
           steplinepos + ContentLength <= inwaiting.GetLength() );

    // blank out the rest of the request buffers.
    request->RequestMethod[0] = '\0';
    request->RequestUri.Clear();
    request->Headers.Clear();
    request->ContentLength = -1;
    request->ContentBody.Clear();

    // determine the version.
    if (ReqVerType != VERSION_1_0 &&
        ReqVerType != VERSION_1_1)
    {
      request->RequestVersion = VERSION_UNKNOWN;
      goto LoadedRequest;
    }

    // copy the command method.
    unsigned int cmdlen = ReqMethod.GetLength();
    if (cmdlen >= sizeof(request->RequestUri))
      cmdlen = sizeof(request->RequestUri) - 1;
    strncpy(request->RequestMethod, ReqMethod.GetHead(), cmdlen);
    request->RequestMethod[cmdlen] = '\0';

    // copy the requested document URI.
    request->RequestUri.Reserve(ReqUri.GetLength() + 1);
    memcpy(request->RequestUri.GetHead(), ReqUri.GetHead(), ReqUri.GetLength());
    request->RequestUri.GetHead()[ReqUri.GetLength()] = '\0';
    request->RequestUri.MarkUsed(ReqUri.GetLength());

    // copy the headers.
    request->Headers.Reserve(steplinepos);
    memcpy(request->Headers.GetHead(),
            inwaiting.GetHead(),
            steplinepos);
    request->Headers.MarkUsed(steplinepos);

    // copy the content body.
    request->ContentLength = ContentLength;
    if (ContentLength > 0)
    {
      request->ContentBody.Reserve(ContentLength);
      memcpy(request->ContentBody.GetHead(),
              inwaiting.GetHead() + steplinepos,
              ContentLength);
      request->ContentBody.MarkUsed(ContentLength);
    }
  }

LoadedRequest:

  // remove this request from our buffers and prepare for another one.
  inwaiting.RemoveHead(steplinepos +
        (ContentLength > 0 ? ContentLength : 0));
  ContentLength = -1;
  haveHeaders = haveStatusLine = false;
  steplinepos = 0;
  ReqMethod.Clear();
  ReqUri.Clear();
  ReqVerType = VERSION_UNKNOWN;

  // scan for any secondary bodies that we already received.
  // (any ReparseIncoming errors can be caught the next time it is called.)
  ReparseIncoming();

  return true;
}

// --------------------------------------------------------------------------

// Updates a MiniHttpResponse structure to include the specified 'header'
// and 'value' in the set of headers that will be sent back.  These two
// arguments are copied and do not need to be preserved by the caller
// after invoking this function.  If a previous instance of 'header' already
// existed in the set of headers in the MiniHttpResponse structure, its
// old value is discarded and replaced with the new 'value'.  It is assumed
// that 'headername' and 'value' contain no unsafe characters (the caller
// should encode all unsafe characters according to the URL character
// escapement standards).  The 'value' argument may be NULL to specify that
// the named header should be deleted if it was found to be present already.

void MiniHttpResponse::SetHeader(const char *headername, const char *value)
{
  assert(headername != NULL && headername[0] != 0);
  assert(strpbrk(headername, ": \t\r\n") == NULL);

  // Look for any previous instances of this header and delete it.
  unsigned int lineoffset = 0;
  unsigned int lastlineoffset = 0;
  AutoBuffer workline;
  while (Headers.StepLine(&workline, &lineoffset))
  {
    int cmpresult = __strcmpheader(workline.GetHead(), headername);
    if (cmpresult > 0)
    {
      // This line matched the start of the header.  Remove this header
      // line, and we will add it back at the end with the new value.
      // HTTP headers can potentially be split into multiple lines, with
      // subsequent lines marked with leading whitespace.
      #ifdef ALLOWMULTILINEHEADERS
      unsigned int newoffset = lineoffset;
      while (Headers.StepLine(&workline, &newoffset))
      {
        const char *head = workline.GetHead();
        if (*head == ' ' || *head == '\t')
        {
          lineoffset = newoffset;
        }
        else break;
      }
      #endif

      // Actually remove this header and move the remaining
      // ones to fill in the introduced gap.
      memmove(Headers.GetHead() + lastlineoffset,
              Headers.GetHead() + lineoffset,
              Headers.GetLength() - lineoffset);
      Headers.RemoveTail(lineoffset - lastlineoffset);
      break;
    }
    lastlineoffset = lineoffset;
  }


  // Add the new header.
  if (value != NULL)
  {
    Headers.Reserve(strlen(headername) + 2 + strlen(value) + 3);
  #ifdef HAVE_SNPRINTF
    int usedsize = snprintf(Headers.GetTail(), Headers.GetTailSlack(),
  #else
    int usedsize = sprintf(Headers.GetTail(),
  #endif
          "%s: %s\r\n", headername, value);
    assert(usedsize >= 0 && usedsize < (int) Headers.GetTailSlack());
    Headers.MarkUsed(usedsize);
  }
}

// --------------------------------------------------------------------------

bool MiniHttpDaemonConnection::QueueResponse(MiniHttpResponse *response)
{
  assert(response != NULL);

  // Store the HTTP greeting and version line.
  int usedsize;
  const char *statustext = response->GetStatusText();
  outwaiting.Reserve(25 + strlen(statustext));
#ifdef HAVE_SNPRINTF
  usedsize = snprintf(outwaiting.GetTail(), outwaiting.GetTailSlack(),
#else
  usedsize = sprintf(outwaiting.GetTail(),
#endif
        "HTTP/1.0 %d %s\r\n", response->statuscode, statustext);
  assert(usedsize >= 0 && usedsize < (int) outwaiting.GetTailSlack());
  outwaiting.MarkUsed(usedsize);

  // Ensure the Content-Length header is set properly.
  char contentlength[20];
  unsigned long bodylen = response->ContentBody.GetLength();
  sprintf(contentlength, "%lu", (unsigned long) bodylen);
  response->SetHeader("Content-Length", contentlength);

  // Store the HTTP headers and the body.
  unsigned long headerlen = response->Headers.GetLength();
  outwaiting.Reserve(headerlen + 2 + bodylen + 2);
  memcpy(outwaiting.GetTail(), response->Headers.GetHead(), headerlen);
  memcpy(outwaiting.GetTail() + headerlen, "\r\n", 2);
  outwaiting.MarkUsed(headerlen + 2);
  memcpy(outwaiting.GetTail(), response->ContentBody.GetHead(), bodylen);
  memcpy(outwaiting.GetTail() + bodylen, "\r\n", 2);
  outwaiting.MarkUsed(bodylen + 2);

  return true;
}

// --------------------------------------------------------------------------

const char *MiniHttpResponse::GetStatusText(void)
{
  switch (statuscode)
  {
  case 200: return "OK";
  case 201: return "Created";
  case 202: return "Accepted";
  case 204: return "No Content";
  case 301: return "Moved Permanently";
  case 302: return "Moved Temporarily";
  case 304: return "Not Modified";
  case 400: return "Bad Request";
  case 401: return "Unauthorized";
  case 403: return "Forbidden";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 500: return "Internal Server Error";
  case 501: return "Not Implemented";
  case 502: return "Bad Gateway";
  case 503: return "Service Unavailable";
  case 504: return "Gateway Timeout";
  case 505: return "HTTP Version Not Supported";
  default: return "Unknown";
  };
}

// --------------------------------------------------------------------------

