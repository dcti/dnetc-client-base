/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: strip_pup.c,v 1.2 2002/09/02 00:35:50 andreasb Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * Strip symbols PowerUp relocatable ELF - requires strip and vlink (from
 * VBCC).  This is quite messy, but there is no other way with
 * gcc/PowerUp :(
 * ----------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <proto/dos.h>

int main(int argc, char **argv)
{
   int ret = 20;
   UBYTE cmd[512];

   if (argc >= 2) {
      /* First strip pass */
      sprintf(cmd,"ppc-amigaos-strip -S -x -X --strip-unneeded -R .comment %s",argv[1]);
      ret = System(cmd,NULL);
      if (!ret) { // success
         /* Run vlink to remove other unnecessary symbols, etc */
         sprintf(cmd,"vlink -r -S -b elf32powerup -o dnetc.puptmp %s",argv[1]);
         ret = System(cmd,NULL);
         if (!ret) { // success
            DeleteFile(argv[1]);
            Rename("dnetc.puptmp",argv[1]);
            sprintf(cmd,"ppc-amigaos-strip -S -x -X --strip-unneeded -R .comment %s",argv[1]);
            ret = System(cmd,NULL);
            SetProtection(argv[1],FIBF_OTR_READ | FIBF_OTR_WRITE | FIBF_OTR_EXECUTE | FIBF_OTR_DELETE);
	 }
      }
   }

   return(ret);
}
