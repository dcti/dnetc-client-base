/*
 * Copyright distributed.net 1997-2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * Returns current date in format required for AmigaDOS version strings
 * ----------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
   time_t t;
   struct tm *tp;

   t = time(NULL);
   tp = localtime(&t);
   printf("\"%d.%d.%d\"",tp->tm_mday,tp->tm_mon+1,tp->tm_year+1900);

   return(0);
}
