// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//


#include "minihttp.h"

#include <time.h>
#include <string.h>
#include "client.h"

#define MAX_CONNECTIONS   5
#define CLIENTIDLETIMEOUT 30

bool ProcessClientPacket(MiniHttpDaemonConnection *httpcon, Client *client);

void __fdsetsockadd(SOCKET sock, fd_set *setname, int *sockmax, int *writecnt);

void initxmlserve(int address, int port);

void poll();

void deinitxmlserve();
