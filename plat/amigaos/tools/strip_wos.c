/*
** Simple wrapper for Elf2Exe2 which acts in the same way as GCC strip
*/

#include <stdio.h>
#include <stdlib.h>
#include <proto/dos.h>

int main(int argc, char **argv)
{
   int ret = 20;
   UBYTE cmd[512];

   if (argc >= 2) {
      sprintf(cmd,"elf2exe2 %s dnetc.wostmp",argv[1]);
      ret = System(cmd,NULL);
      if (!ret) { // success
         DeleteFile(argv[1]);
         Rename("dnetc.wostmp",argv[1]);
      }
   }

   return(ret);
}
