/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amConsole.c,v 1.2 2002/09/02 00:35:48 andreasb Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * This file contains the Amiga specific console code, and support for
 * the Myzar GUI.
 * ----------------------------------------------------------------------
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
#include "triggers.h"
#include "util.h"
#include "modereq.h"

#if defined(__PPC__) && defined(__POWERUP__)
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

   #ifndef NOGUI
   BPTR NewConsole;
   BPTR OldInput;
   BPTR OldOutput;
   #endif
} ConStatics;

extern struct Library *DnetcBase;

extern unsigned long *__stdfiledes;

#ifndef NOGUI
/* default console for WB startup (libnix) */
char __stdiowin[] = "NIL:";
#endif

int amigaInitializeConsole(int runhidden, int runmodes)
{
   runhidden = runhidden;
   runmodes = runmodes;

   if ((ConStatics.Input = Input()))
      ConStatics.InputIsInteractive = (IsInteractive(ConStatics.Input) == DOSTRUE);
   if ((ConStatics.Output = Output()))
      ConStatics.OutputIsInteractive = (IsInteractive(ConStatics.Output) == DOSTRUE);

   // Workbench "Execute Command" output windows refuse to close without this!
   #ifndef NOGUI
   if (!DnetcBase) {
      printf("\r");
      fflush(stdout);
   }
   #else
   printf("\r");
   fflush(stdout);
   #endif

   return 0;
}

#ifndef NOGUI
BPTR amigaOpenNewConsole(char *conname)
{
   if (!ConStatics.NewConsole) {
      BPTR con = Open(conname,MODE_OLDFILE);
      if (con) {
         ConStatics.OldInput = SelectInput(con);
         ConStatics.OldOutput = SelectOutput(con);
         __stdfiledes[0] = __stdfiledes[1] = con;
         ConStatics.NewConsole = con;
         amigaInitializeConsole(0,0);
      }
   }

   return(ConStatics.NewConsole);
}

void amigaCloseNewConsole(void)
{
   if (ConStatics.NewConsole) {
      Close(ConStatics.NewConsole);
      SelectInput(ConStatics.OldInput);
      SelectOutput(ConStatics.OldOutput);
      __stdfiledes[0] = ConStatics.OldInput;
      __stdfiledes[1] = ConStatics.OldOutput;
      ConStatics.NewConsole = NULL;
      amigaInitializeConsole(0,0);
   }
}
#endif

int amigaConOut(const char *msg)
{
   #ifndef NOGUI
   static int forcecon = 0;
   if (DnetcBase) {
      if (ModeReqIsSet(MODEREQ_CONFIG|MODEREQ_CONFRESTART)) {
         if (!forcecon) {
            amigaOpenNewConsole("CON://630/200/distributed.net client configuration");
            forcecon = 1;
	 }
      }
      else {
         if (forcecon) {
            amigaCloseNewConsole();
            forcecon = 0;
	 }
         amigaGUIOut((char *)msg);
         return 0;
      }
   }
   #endif

   fwrite( msg, sizeof(char), strlen(msg), stdout);
   fflush(stdout);

   return 0;
}

int amigaConOutModal(const char *msg)
{
   #ifndef NOGUI
   if (DnetcBase) {
      amigaGUIOut((char *)msg);
   }
   else
   #endif
   {
      fprintf( stderr, "%s\n", msg );
      fflush( stderr );
   }
   return 0;
}

int amigaConOutErr(const char *msg)
{
   #ifndef NOGUI
   if (DnetcBase) {
      amigaGUIOut((char *)msg);
   }
   else
   #endif
   {
      fprintf( stderr, "%s: %s\n", utilGetAppName(), msg );
      fflush( stderr );
   }
   return 0;
}

int amigaConIsGUI(void)
{
   static int running = -1;

   #ifndef NOGUI
   if (DnetcBase) return 1;
   #endif

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
         visible = amigaConIsGUI();
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
   if (amigaConIsGUI() && !ConStatics.NewConsole) {
      *width = 200;
      return 0;
   }
   else {
      // WINDOW STATUS REQUEST
      return __ReadCtrlSeqResponse("\x9b" "0 q",4,'r',' ',width,height);
   }
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
