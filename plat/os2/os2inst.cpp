/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------
 * The two functions in this module install/deinstall the client
 * to/from the startup folder
 * ----------------------------------------------------------------
 *
 * $Log: os2inst.cpp,v $
 * Revision 1.1.2.1  2001/01/21 15:10:27  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.7.2.1  2001/01/01 22:48:57  trevorh
 * Update for OS/2 465 build
 *
 * Revision 1.7  1999/04/11 14:47:14  cyp
 * a) simplified install by passing argv[0]; b) Added comment re:detached
 *
 *
*/
const char *os2inst_cpp(void) {
return "@(#)$Id: os2inst.cpp,v 1.1.2.1 2001/01/21 15:10:27 cyp Exp $"; }

#include <stdio.h>  /* sprintf() */
#include <string.h> /* strrchr(), strcat() */
#define  INCL_WIN
#define  INCL_WINHELP
#include <os2.h>


#define RC5DES_FOLDOBJ_NAME "<RC5DES-CLI>"
#define RC5DES_FULLNAME     "RC5DES Client for OS/2"


int os2CliUninstallClient(int do_the_uninstall_without_feedback)
{
  ULONG mbflag = MB_ERROR;
  HOBJECT hObject = WinQueryObject((const unsigned char*)RC5DES_FOLDOBJ_NAME);
  char *msg = "The RC5DES client was not found in the Startup Folder.\n";
  int rc = +1;

  if (hObject != NULLHANDLE)
  {
    rc = -1;
    msg = "The RC5DES client could not be removed from in the Startup Folder.";
    if (WinDestroyObject(hObject) == TRUE )
    {
      if (WinQueryObject((const unsigned char*)RC5DES_FOLDOBJ_NAME) == NULLHANDLE)
      {
        msg = "The RC5DES client was successfully removed from the Startup Folder.";
        mbflag = 0;
        rc = 0;
      }
    }
  }

  if (!do_the_uninstall_without_feedback)
    WinMessageBox(HWND_DESKTOP, HWND_DESKTOP, (PSZ)msg, (const unsigned char*)RC5DES_FULLNAME,
                 0 /* WindowId */, MB_OK | mbflag );
  return rc;
}



int os2CliInstallClient(int do_the_install_without_feedback, const char *exename)
{
  #define PSZSTRINGSIZE 4068
  int rc = -1;
  ULONG  ulFlags = 0,
         mbflag = MB_ERROR;
  int was_installed = 0;
  char   pszClassName[] = "WPProgram",
         pszTitle[] = "RC5-DES Cracking Client",
         pszLocation[] = "<WP_START>";    // Startup Folder
  char pszSetupString[PSZSTRINGSIZE] = "OBJECTID="RC5DES_FOLDOBJ_NAME";"
                                       "MINIMIZED=YES;"
                                       "PROGTYPE=WINDOWABLEVIO;"
				       "EXENAME=";
				
  if (!exename) /* should never happen */				
  {
    was_installed = 0;
    rc = -1;
  }				
  else if (((strlen(exename)<<1)+strlen(pszSetupString)+40)>=sizeof(pszSetupString))
  {
    was_installed = 0;
    rc = -1;
  }
  else if ((rc = os2CliUninstallClient(1)) < 0)
  { /* was there but couldn't be uninstalled */
    was_installed = 1;
    rc = -1;
  }
  else /* wasn't there or was removed */
  {
    char *p, *q;
    was_installed = (rc == 0);

    /*
       detached - this is really the wrong place to do this.
       The client should run detached only if either --hide or
       ini.runhidden=1, neither of which are known at the time this
       function is called. (btw: a "-quiet" found on the command line
       prior to calling this determines that *the installation* should
       be quiet, not the run)
       - EMX could just fork() [same code section in cmdline.cpp as all
       the other unix clients]
       - non-emx could choose to simply spawn
       "CMD.EXE PARAMETERS=/c detach RC5DES.EXE -svcrun ..."
       where "-svcrun" is simply a hack to ensure that it doesn't spawn
       itself in a loop. (of course, if we can detect if the client is
       already loaded, or already detached, then we don't need -svcrun)
    */
    strcat(pszSetupString, "CMD.EXE;");     // command processor
    strcat(pszSetupString, "PARAMETERS=/c detach ");   // detach

    strcat(pszSetupString, exename );
    strcat(pszSetupString, ";STARTUPDIR=");
    strcat(pszSetupString, exename );

    p = strrchr(pszSetupString, '\\');
    q = strrchr(pszSetupString, '/');
    if (q>p) p = q;
    if (p) *p='\0';
    //strcat( pszSetupString, ";");

    rc = -1;
    if ( WinCreateObject((const unsigned char*)pszClassName, (const unsigned char*)pszTitle, (const unsigned char*)pszSetupString,
                         (const unsigned char*)pszLocation, ulFlags) != NULLHANDLE )
    {		
      mbflag = 0;
      rc = 0;
    }
  }

  if (!do_the_install_without_feedback)
  {
    sprintf( pszSetupString,
            "The RC5DES client %s %sadded to the Startup Folder\n",
            ((rc == 0)?("has been"):("could not be")),
	    ((was_installed)?("re-"):("")) );
    WinMessageBox(HWND_DESKTOP, HWND_DESKTOP, (PSZ)pszSetupString,
                            (const unsigned char*)RC5DES_FULLNAME, 0, MB_OK | mbflag );
  }
  return rc;
}
