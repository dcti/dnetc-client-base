/* Created by Oliver Roberts <oliver@futaura.co.uk>
**
** $Id: amConsole.c,v 1.1.2.1 2001/01/21 15:10:27 cyp Exp $
**
** ----------------------------------------------------------------------
** This file contains the Amiga specific console code, and support for
** the Myzar GUI.
** ----------------------------------------------------------------------
*/

#ifdef __PPC__
#pragma pack(2)
#endif

#include <exec/types.h>
#include <exec/memory.h>
#include <libraries/dos.h>
#include <libraries/dosextens.h>
#include <proto/exec.h>
#include <proto/dos.h>

#ifdef __PPC__
#pragma pack()
#endif

#include <ctype.h>
#include "amiga.h"
#include "common/triggers.h"

#if (defined __PPC__) && (defined __POWERUP__)
#undef Read
#undef Write
#define Read(a,b,c) PPCRead(a,b,c)
#define Write(a,b,c) PPCWrite(a,b,c)
#endif

static struct {
   BPTR Input;
   BPTR Output;
   BOOL InputIsInteractive;
   BOOL OutputIsInteractive;
} ConStatics;

int amigaInitializeConsole(int runhidden, int runmodes)
{
   runhidden = runhidden;
   runmodes = runmodes;

   if ((ConStatics.Input = Input()))
      ConStatics.InputIsInteractive = (IsInteractive(ConStatics.Input) == DOSTRUE);
   if ((ConStatics.Output = Output()))
      ConStatics.OutputIsInteractive = (IsInteractive(ConStatics.Output) == DOSTRUE);

   // Workbench "Execute Command" output windows refuse to close without this!
   printf("\r");
   fflush(stdout);

   return 0;
}

int MyzarIsRunning(void)
{
   static int running = -1;

   // only run this code once (reduces overhead)
   if (running == -1) {
      Forbid();
      running = (FindPort((STRPTR)"MYZAR.1_mp") != NULL);
      Permit();
   }

   return(running);
}

int amigaConIsScreen(void)
{
   BPTR output = ConStatics.Output;
   int visible = 0;

   if (output) {
      visible = ConStatics.OutputIsInteractive;
   }

   if (!visible) {
      BOOL outputisfile = FALSE;

      /*
      ** Check to see if output has been redirected to a file
      */
      if (output) {
         char buffer[256];
         if (NameFromFH(output,buffer,256)) {
            if (IsFileSystem(buffer) == DOSTRUE) {
               struct FileInfoBlock *fib;
               if ((fib = (struct FileInfoBlock *)AllocDosObject(DOS_FIB,NULL))) {
                  if (ExamineFH(output,fib) == DOSTRUE) {
                     if (fib->fib_DirEntryType != ST_PIPEFILE) outputisfile = TRUE;
		  }
                  FreeDosObject(DOS_FIB,fib);
	       }
	    }
            else {
               if (strncmp(buffer,"NIL:",4) == 0) outputisfile = TRUE;
	    }
	 }
      }

      if (!outputisfile) {
         /*
         ** Check if Myzar is running - if so, our output needs to be fully
         ** visible and is not covered by the above since our output handle is
         ** actually a pipe (IsInteractive == FALSE)
         */
         visible = MyzarIsRunning();
      }
   }

   return(visible);
}

static int __ReadCtrlSeqResponse(char *cmd, int cmdlen, char matchchar1,
                                 char matchchar2, int *val1, int *val2)
{
   BPTR fh = ConStatics.Output;
   int success = -1; // ANSI console not available?

   if (fh && ConStatics.OutputIsInteractive) {
      SetMode(fh,1);

      Write(fh,(APTR)cmd,cmdlen);

      if (WaitForChar(fh,100000)) {  // protect against iconified KingCON windows
         char csibuf[256], *ptr;
         int buflen, val, cnt = 0;

         do {
            buflen = Read(fh,csibuf,256);
         } while (buflen == 1 && csibuf[0] == 0 && ++cnt < 3);

         SetMode(fh,0);

         // Look for end of the CSI sequence that we need
         ptr = &csibuf[buflen-1];
         while (--ptr > csibuf) {
            if (ptr[1] == matchchar1 && (!matchchar2 || ptr[0] == matchchar2)) {
               if (!matchchar2) ptr++;
               break;
	    }
	 }
         // Attempt to read the trailing 2 values given in the CSI sequence
         while (ptr > csibuf && *(--ptr) != ';');
         if (ptr > csibuf) {
            val = atoi(&ptr[1]);
            while (ptr > csibuf && *(--ptr) != ';' && *ptr != '\x9b');
            if (ptr > csibuf) {
               *val1 = val;
               *val2 = atoi(&ptr[1]);
               success = 0;
               //printf("%d,%d (%d)\n",*val1,*val2,buflen);
            }
	 }

         #if 0
         if (success) {
            int i;
            for (i=0;i<buflen;i++) printf("%02x",csibuf[i]);
            printf(" (%d)\n",buflen);
	 }
         #endif
      }
      else {
         SetMode(fh,0);
      }
   }

   return success;
}

int amigaConGetSize(int *width, int *height)
{
   // WINDOW STATUS REQUEST
   return __ReadCtrlSeqResponse("\x9b" "0 q",4,'r',' ',width,height);
}

int amigaConGetPos(int *col, int *row)
{
   // DEVICE STATUS REPORT
   return __ReadCtrlSeqResponse("\x9b" "6n",3,'R',0,col,row);
}

int getch(void) /* get a char from stdin, do not echo, block if none waiting */
{
   UBYTE c;

   if (ConStatics.InputIsInteractive) {
      BPTR fh = ConStatics.Input;
      SetMode(fh,1);
      Read(fh, &c, 1);

      if (c == 0x9B) {  /* Filter out ANSI CSI escape sequence */
         do {
            Read(fh, &c, 1);
         } while (!((c >= 0x40) && (c <= 0x7E)));
         c = 0;
      }

      SetMode(fh,0);

      if (c == 3) {  /* ^C in input */
         c = 0;
         #ifdef __PPC__
         #ifdef __POWERUP__
         void *task = PPCFindTask(NULL);
         PPCSignal(task, SIGBREAKF_CTRL_C);
         #else
         struct TaskPPC *task = FindTaskPPC(NULL);
         SignalPPC(task, SIGBREAKF_CTRL_C);
         #endif
         #else
         Signal(FindTask(NULL), SIGBREAKF_CTRL_C);
         #endif
         RaiseExitRequestTrigger();
      }
   }
   else {
      c = getc(stdin);
   }

   return(c);
}
