/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amInstall.c,v 1.2 2002/09/02 00:35:49 andreasb Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * This file contains routines to install/deinstall an icon in the
 * WBStartup drawer
 * ----------------------------------------------------------------------
*/

#include "amiga.h"

#ifdef __PPC__
#pragma pack(2)
#endif

#include <proto/icon.h>

#ifdef __PPC__
#pragma pack()
#endif

#ifdef __PPC__
#define WBSTARTICONNAME "SYS:WBStartup/dnetc_ppc"
#define CLIENTSTACKSIZE 200000
#else
#define WBSTARTICONNAME "SYS:WBStartup/dnetc_68k"
#define CLIENTSTACKSIZE 64*1024
#endif

int amigaInstall(int quiet, const char *progname)
{
   BPTR progdir, olddir;
   struct DiskObject *icon;
   static const STRPTR ttypes[] = { "HIDE", "DONOTWAIT", NULL };
   UBYTE toolname[256];
   int rc = -1;
   struct Library *IconBase;

   if ((IconBase = OpenLibrary((UBYTE *)"icon.library",37))) {
      if ((progdir = GetProgramDir())) {
         olddir = CurrentDir(progdir);
         if ((icon = GetDiskObjectNew((UBYTE *)progname))) {
            if (NameFromLock(progdir,toolname,256)) {
               AddPart(toolname,(UBYTE *)progname,256);
               icon->do_Type = WBPROJECT;
               icon->do_CurrentX = NO_ICON_POSITION;
               icon->do_CurrentY = NO_ICON_POSITION;
               icon->do_ToolTypes = (STRPTR *)ttypes;
               icon->do_DefaultTool = (STRPTR)toolname;
               icon->do_StackSize = CLIENTSTACKSIZE;
               if (PutDiskObject(WBSTARTICONNAME,icon)) {
                  rc = 0;
                  if (!quiet) {
                     fprintf(stderr,
                             "%s: An icon to start the client from WBStartup has been successfully\n"
                             "installed so the client will automatically be started on system boot.\n"
                             "*** Please ensure that the client is configured ***\n",
                             progname);
                  }
	       }
               else if (!quiet) {
                  fprintf(stderr,
                          "%s: Unable to save icon in SYS:WBStartup\n",
                          progname);
	       }
 	    }
            FreeDiskObject(icon);
         }
         CurrentDir(olddir);
      }
      CloseLibrary(IconBase);
   }

   return(rc);
}

int amigaUninstall(int quiet, const char *progname)
{
   int rc = -1;
   struct Library *IconBase;
   BPTR lock;

   if ((lock = Lock(WBSTARTICONNAME ".info", ACCESS_READ))) {
      UnLock(lock);

      if ((IconBase = OpenLibrary((UBYTE *)"icon.library",37))) {
         if (DeleteDiskObject(WBSTARTICONNAME)) {
            rc = 0;
            if (!quiet) {
               fprintf(stderr,
                       "%s: The client has been sucessfully uninstalled and\n"
                       "will no longer be automatically started on system boot.\n",
                       progname);
            }
         }
         else if (!quiet) {
            fprintf(stderr,
                    "%s: Unable to delete '%s'\n",
                    progname,WBSTARTICONNAME);
         }
         CloseLibrary(IconBase);
      }
   }
   else {
      rc = 0;
      if (!quiet) {
         fprintf(stderr,
                 "%s: The client was not previously installed in WBStartup and\n"
                 "therefore cannot be uninstalled.\n",
                 progname);
      }
   }
   
   
   return(rc);
}
