/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *temperature_c(void) {
return "@(#)$Id: temperature.c,v 1.1.2.1 2003/07/29 23:57:20 bdragon Exp $"; }

/* ------------------------------------------------------------------------ */

/*
 *
 * int dunix_cputemp();
 *
 * Returns the CPUs temperature in Kelvin, else -1.
 * A test main() is at the end of this file.
 *
 * TODO:decode system calls and bypass sysconfig
 *
 */

#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"

#if 0
#define Log printf
#endif

#ifdef __cplusplus /* to ensure gcc -lang switches don't mess this up */
extern "C" {
#endif
    int dunix_cputemp();
#ifdef __cplusplus
}
#endif

int dunix_cputemp() {

int cputemp=-1;
FILE *file;
char buf[256];
char *buf2;
char envsup=0,found=0;
char searchstr[]="env_current_temp = ";

  if ((file = popen("/sbin/sysconfig -q envmon", "r"))
!= NULL) {
    while ((fgets(buf, sizeof(buf), file) != NULL) && ((!envsup) ||
          (cputemp==-1))) {
      if (strstr (buf, "env_supported = 1")) {
        //environmental system is loaded, and hardware supports it
        envsup=found=1;
      } else if (strstr (buf, "env_supported = 0")) {
        //environmental system is loaded, but hardware doesn't support it
        Log("Environmental monitoring not supported\n"
            "on this system, disabling temperature checking.\n");
        cputemp=-1;
        found=1;
        break;
      }
      if ((buf2=strstr (buf, searchstr)) != NULL) {
        //environmental system is loaded, and we have a temperature
        //reading, although it could be bogus, hence the support checks
        buf2+=strlen(searchstr); /*forward pointer past the search string*/
        if (*buf2) { /*just in case we've hit the end of the string*/
          cputemp=atoi(buf2);
          cputemp += 273/*.15*/; /* C -> K */
        }
      }
    }
    pclose(file);
    if (!found) {
      Log("Environmental monitoring subsystem not loaded,\n"
          "disabling temperature checking.\n");
      cputemp=-1;
    }
  }
  return cputemp;
}

#if 0
int main(int argc,char *argv[])
{
    while (1) { // test for leaks
        printf("Temp %d\n",dunix_cputemp());
    }
    
    return 0;
}
#endif

