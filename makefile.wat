## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 | w_h | wsv ]
##                       or anything else defined at the end of this makefile
##
## $Log: makefile.wat,v $
## Revision 1.18  1998/07/12 08:41:34  ziggyb
## oops, typed it wrong, it really puts cliident.obj in the LINKOBJs this time
##
## Revision 1.17  1998/07/12 08:29:25  ziggyb
## Added cliident.cpp to the path and MMX core changes
##
## Revision 1.16  1998/07/09 05:45:37  silby
## This neato file can build the nt service!
##
## Revision 1.15  1998/07/07 14:51:51  ziggyb
## Added the pathwork.cpp file to the link objs and the make process. Also moved
## the Cyrix core around, it seems to speed up the core a bit. Well at least it
## prevented it from being slowed down, in OS/2 at least.
##
## Revision 1.14  1998/06/21 17:10:20  cyruspatel
## Fixed some NetWare smp problems. Merged duplicate numcpu validation code
## in ::ReadConfig()/::ValidateConfig() into ::ValidateProcessorCount() and
## spun that off, together with what used to be ::x86id() or ::ArmId(), into
## cpucheck.cpp. Adjusted and cleaned up client.h accordingly.
##
## Revision 1.13  1998/06/18 08:38:10  cyruspatel
## Fixed Log and Id mixup
##
## Revision 1.12  1998/06/18 08:19:17  cyruspatel 
## Converted all names to lower case (ncpfs serves case sensitive names).
## Modified for use of the pre-assembled objs of the p?despro cores until a
## wasm version is available. Fixed use of %PATHOPS (got evaluated at
## assignment time, so %PATHOPS was always "/fr= /fo= /i"). Fixed misplaced
## colons that were causing unconditional makes for some files. Changed
## global /zp8 to /zp1 to work around alignment problems (no speed hit :).
## Compacted all those long dependency lists into .autodepend directives.
## Added suppression of wcpp/wasm 'whoami' banners. $Log and cleaned up a bit.
##
## Revision 1.11  1998/06/18 07:22:46  jlawson
## updated makefile with new paths to des/rc5 cores 
##
## Revision 1.10  1998/06/16 22:29:59  silby
## added p1bdespro.obj and p2bdespro.obj so that they'll get linked (unless 
## I did it wrong. <g>)
##
## Revision 1.9  1998/06/15 02:32:20  ziggyb
## added os2 dod.obj  Also moved the rc5 cores to the front of the link 
## list to speed up the keyrates a bit for os2
##
## Revision 1.8  1998/06/14 11:16:11  ziggyb
## Moved the /fr and /fo options to %PATHOPS so that I can change them
## easily. /fr doesn't work in Watcom 10 :( Also moved all the rc5/des 
## core files to the front of the list of object files since it does 
## seem to speed things up a bit.
##
## Revision 1.7  1998/06/14 01:43:16  cyruspatel
## important fix: uniform-ified /zp compiler switch for all modules &
## all platforms (yeah, I took the liberty). /zp must be uniform,
## or else indirect pointers to class members (as in clirate's use of
## prob->timelo) might end up being 'slightly' off.
##
## Revision 1.6  1998/06/14 01:43:16  cyruspatel
## added support for new disphelp.cpp; removed obsolete chdir cmds;
## added $Log
##          
## Revision 1.5  1998/06/09 08:54:12  jlawson
## NetWare patches - NetWare no longer uses watcom static clib
##
## Revision 1.4  1998/06/07 08:39:14  ziggyb
## Added a override to the stacksize for OS/2 builds. 32k stack size 
## slows down the rc5 cores.
##
## Revision 1.3  1998/05/26 23:02:02  ziggyb
## Another OS/2 Makefile change (taking out the default watcom/os2 directory)
##
## Revision 1.2  1998/05/26 21:02:16  ziggyb
## Added the new cli source file into the makefile. Also changed some of 
## the compile options under OS/2.
##
## Revision 1.1  1998/05/24 14:25:37  daa
## Import 5/23/98 client tree
## 

## $Id: makefile.wat,v 1.18 1998/07/12 08:41:34 ziggyb Exp $

CC=wpp386
CCASM=wasm
LINK=wlink

%VERSION  = 70.24          
%VERSTRING= v2.7024.409

%EXTOBJS =  des\brydmasm\p1bdespro.obj  des\brydmasm\p2bdespro.obj 
            des\mmx-bitslice\sboxes-mmx.obj
            des\mmx-bitslice\deseval-meggs3-mmx.obj
            #extra objs (made elsewhere) but need linking here

%LINKOBJS = output\rg-486.obj output\rg-k5.obj output\rg-k6.obj &
            output\rc5p5brf.obj output\rg-p6.obj output\rg-6x86.obj &
            output\bdeslow.obj output\bbdeslow.obj output\x86ident.obj &
            output\cliconfig.obj output\autobuff.obj output\buffwork.obj &
            output\mail.obj output\client.obj output\disphelp.obj &
            output\iniread.obj output\network.obj output\problem.obj &
            output\scram.obj output\des-x86.obj output\convdes.obj &
            output\clitime.obj output\clicdata.obj output\clirate.obj &
            output\clisrate.obj output\cpucheck.obj output\pathwork.obj &
            output\des-slice-meggs.obj output\cliident.obj
 
            # this list can be added to in the platform specific section


LNKbasename = rc5des       # for 'rc564'.err 'rc564'.lnk 'rc5des'.err etc
%STACKSIZE= 32767          #may be redefined in the platform specific section
%AFLAGS   = /5s /fp3 /mf   #may be defined in the platform specific section
%LFLAGS   =                #may be defined in the platform specific section
%CFLAGS   = /6s /fp3 /ei /mf #may be defined in the platform specific section
%OPT_SIZE = /s /os         #may be redefined in the platform specific section
%OPT_SPEED= /oneatx /oh /oi+ #redefine in platform specific section
%LIBPATH  =                #may be defined in the platform specific section
%LIBFILES =                #may be defined in the platform specific section
%MODULES  =                #may be defined in the platform specific section
%IMPORTS  =                #may be defined in the platform specific section
%BINNAME  =                #may be defined in the platform specific section
%COPYRIGHT=                #may be defined in the platform specific section
%FORMAT   =                #may be defined in the platform specific section

%OBJDIROP = /fo=$$^@       #Puts the .err/.objs in the right directories
%ERRDIROP = /fr=$$[:       #...redefine for older versions of Watcom
                           
#.silent
.nocheck

#-----------------------------------------------------------------------

noplatform: .symbolic
  @%write con: 
  @%write con:   Platform has to be specified. 
  @%write con:      eg: WMAKE [-f makefile] os2 
  @%write con:          WMAKE [-f makefile] netware 
  @%write con:  
  @%quit

#-----------------------------------------------------------------------

clean :
  #erase rc5des.*
  #erase rc5des*.*
  erase *.bak
  erase output\*.obj
  erase common\*.bak
  erase common\*.err
  erase des\*.bak
  erase des\*.err
  erase rc5\*.bak
  erase rc5\*.err

zip :
  *zip -r zip *

#-----------------------------------------------------------------------

output\cliconfig.obj : common\cliconfig.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\cpucheck.obj : common\cpucheck.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\client.obj : common\client.cpp  makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\convdes.obj : common\convdes.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clirate.obj : common\clirate.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clistime.obj : common\clistime.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\pathwork.obj : common\pathwork.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1


output\deseval-meggs2.obj : des\deseval-meggs2.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\deseval-meggs3.obj : des\deseval-meggs3.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\deseval-slice-meggs.obj : des\deseval-slice-meggs.cpp makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\des-x86.obj : des\des-x86.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\autobuff.obj : common\autobuff.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\network.obj : common\network.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\buffwork.obj : common\buffwork.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: 
  @set isused=1

output\cliident.obj : common\cliident.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-486.obj : rc5\rg-486.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-6x86.obj : rc5\rg-6x86.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rc5p5brf.obj : rc5\rc5p5brf.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-p5.obj : rc5\rg-p5.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-p6.obj : rc5\rg-p6.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-k5.obj : rc5\rg-k5.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-k6.obj : rc5\rg-k6.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\bdeslow.obj : des\brydwasm\bdeslow.asm 
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\bbdeslow.obj : des\brydwasm\bbdeslow.asm 
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\x86ident.obj : platforms\x86ident.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\netware.obj : platforms\netware\netware.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\hbyname.obj : platforms\netware\hbyname.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\chklocks.obj : platforms\dos\chklocks.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clearscr.obj : platforms\dos\clearscr.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\dod.obj : platforms\os2cli\dod.cpp makefile.wat .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

#-----------------------------------------------------------------------

platform: .symbolic
  @set CFLAGS    = $(%CFLAGS) /zp1            ## always pragma pack(1)
  ## /zp has to be same for all modules, otherwise we'll have big trouble.
  ## no problems with network/buffwork structures if packing on byte boundary

  @set CFLAGS    = $(%CFLAGS) /zq             ## compile quietly
  @set AFLAGS    = $(%AFLAGS) /q              ## assemble quietly
  
  @set isused=0
  @if not exist $(%BINNAME) @set isused=1
  @for %i in ($(%LINKOBJS)) do @%make %i
  @if $(%isused).==0. @%write con: All targets are up to date
  @if $(%isused).==0. @%quit

  @if exist  $(%BINNAME) @del $(%BINNAME)
  @if exist  $(LNKbasename).lnk @del $(LNKbasename).lnk
  @%append   $(LNKbasename).lnk Name $(%BINNAME)
  @for %i in ($(%STACKSIZE)) do @%append $(LNKbasename).lnk Op Stack=%i
  @for %i in ($(%LINKOBJS))   do @%append $(LNKbasename).lnk File %i
  @for %i in ($(%EXTOBJS))   do @%append $(LNKbasename).lnk File %i
  @for %i in ($(%LIBPATH))   do @%append $(LNKbasename).lnk Libpath %i
  @for %i in ($(%MODULES))   do @%append $(LNKbasename).lnk Module %i
  @for %i in ($(%IMPORTS))   do @%append $(LNKbasename).lnk Import %i
  @set isused=
  @for %i in ($(%VERSION))  do @set isused=1
  @if not $(%isused).==. @%append $(LNKbasename).lnk Op Version=$(%VERSION)
  @set isused=
  @for %i in ($(%LIBFILES))  do @set isused=1
  @if not $(%isused).==. @%append $(LNKbasename).lnk Library $(%LIBFILES)
  @set isused=
  @for %i in ($(%FORMAT))    do @set isused=1
  @if not $(%isused).==. @%append $(LNKbasename).lnk Format $(%FORMAT)
  @set isused=
  @for %i in ($(%COPYRIGHT)) do @set isused=1
  @if not $(%isused).==. @%append $(LNKbasename).lnk Op Copyright $(%COPYRIGHT)
  @set isused=
  *$(LINK) $(%LFLAGS) @$(LNKbasename).lnk > $(LNKbasename).err
  @if exist $(%BINNAME) @del $(LNKbasename).err
  @if exist $(LNKbasename).err @type $(LNKbasename).err

#---------------------- platform specific settings come here ----------

dos: .symbolic                                       # DOS/DOS4GW
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set LFLAGS    = sys dos4g op dosseg op eliminate op stub=platform/dos/d4GwStUb.CoM
     @set CFLAGS    = /6s /fp3 /fpc /zm /ei /mf /bt=dos /dDOS4G /DNONETWORK /I$(%watcom)\h
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).com
     @%make platform

os2: .symbolic                                       # OS/2
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /5s /fp5 /bm /mf /bt=os2 /DOS2 /DMULTITHREAD /DMMX_BITSLICER /DBIT_64 /DMEGGS /DKWAN
     @set OPT_SIZE  = /oantrlexi 
     @set OPT_SPEED = /oantrlexi 
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).exe
#     @set STACKSIZE = 16384                 #Will slow down client if it's 32k
     @set LINKOBJS  = $(%LINKOBJS) output\dod.obj
     @set OBJDIROP  = /fo=output\
     @set ERRDIROP  =                       # no /fr= option for Watcom 10.0
     @%make platform

w32: .symbolic                               # win95/winnt standard executable
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt
     @set CFLAGS    = /fpd /5s /fp5 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexih 
     @set OPT_SPEED = /oantrlexih /oi+ 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).exe
     @%make platform

w_h: .symbolic                               # win95 hidden executable
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt_win
     @set CFLAGS    = /fpd /6s /fp6 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexih 
     @set OPT_SPEED = /oantrlexih /oi+ 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename)h.exe
     @%make platform

wsv: .symbolic                               # winnt service
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt
     @set CFLAGS    = /fpd /6s /fp6 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD /DWINNTSERVICE="bovrc5nt"
     @set OPT_SIZE  = /oantrlexih 
     @set OPT_SPEED = /oantrlexih /oi+ 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename)srv.exe
     @%make platform

netware: .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM! (May 24 '98)
     @set STACKSIZE = 32767 #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set LFLAGS    = op multiload op nod op scr 'none' op map op osname='NetWare NLM' # symtrace systemConsoleScreen  #sys netware
     @set OPT_SIZE  = /os /s  
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /6s /fp3 /ei /ms /d__NETWARE__ /dMULTITHREAD /i$(inc_386) /we
                      #/fpc /bt=netware /i$(%watcom)\novh #/bm
     @set LIBFILES =  nwwatemu,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj output\hbyname.obj
     @set IMPORTS   = GetNestedInterruptLevel AllocateResourceTag &
                      GetCurrentTime OutputToScreenWithPointer OutputToScreen &
                      ActivateScreen ImportPublicSymbol UnImportPublicSymbol &
                      ScheduleNoSleepAESProcessEvent CancelNoSleepAESProcessEvent &
                      ScheduleSleepAESProcessEvent CancelSleepAESProcessEvent &
                      RingTheBell GetFileServerMajorVersionNumber Alloc &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = platform\netware\watavoid $(%watcom)\lib386 $(%watcom)\lib386\netware
     @set BINNAME   = $(LNKbasename).nlm
     @set COPYRIGHT ='Visit http://www.distibuted.net/ for more information'
     @set FORMAT    = Novell NLM 'RC5-DES $(%VERSTRING) Client'
     @%make platform
     @\develop\sdkcdall\nlmdump\nlm_dos.exe *$(LNKbasename).nlm /b:$(LNKbasename).map 
     #@\develop\sdkcd13\nwsdk\tools\nlmpackx $(LNKbasename).nlm $(LNKbasename).nlx
     #@del $(LNKbasename).nlm
     #@ren $(LNKbasename).nlx $(LNKbasename).nlm
