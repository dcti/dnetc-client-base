// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include <stdio.h>
#include "client.h"
#include "selcore.h"

// --------------------------------------------------------------------------

void Client::DisplayHelp( void )
{
  printf (
    "Actions:\n"
    "\t-test           tests for client errors\n"
    "\n"
    "\t-benchmark      tests the client speed\n"
    "\t-benchmarkrc5   tests only the client RC5 speed\n"
    "\t-benchmarkdes   tests only the client DES speed\n"
    "\n"
    "\t-benchmark2     quick RC5 & DES client speed test\n"
    "\t-benchmark2rc5  quick RC5 client speed test\n"
    "\t-benchmark2des  quick DES client speed test\n"
    "\n"
    "\t-cpuinfo        give the cpu ID tag of your processor\n"
    "\n"
    "Options:\n"
    "\t-c <cputype>    "
);

  int numcore = 0;
  const char *corename;
  do {
    corename = GetCoreNameFromCoreType( numcore++ );
    if (corename && *corename)
      printf ("%s%-2d= %s\n", numcore>1?"\t                ":"", numcore, corename);
  } while (corename && *corename);
  printf ("\n");
}

// --------------------------------------------------------------------------

