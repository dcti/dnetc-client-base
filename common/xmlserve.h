// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#include "minihttp.h"

#define MAX_CONNECTIONS   5
#define LISTENADDRESS     0
#define LISTENPORT        81
#define CLIENTIDLETIMEOUT 30

bool ProcessClientPacket(MiniHttpDaemonConnection *httpcon, Client *client);

void __fdsetsockadd(SOCKET sock, fd_set *setname, int *sockmax, int *writecnt);
