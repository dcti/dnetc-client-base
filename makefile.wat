## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 | w_h | wsv ]
##                       or anything else defined at the end of this makefile
##
## $Id: makefile.wat,v 1.6 1998/06/13 19:12:19 cyruspatel Exp $
##                     Added support for new disphelp.cpp, remv'd old 'cd's
## 

CC=wpp386
CCASM=wasm
LINK=wlink

%VERSION  = 70.24          
%VERSTRING= v2.7024.409

%LINKOBJS = output\cliconfig.obj output\autobuff.obj output\buffwork.obj &
           output\mail.obj output\client.obj output\disphelp.obj &
           output\iniread.obj output\network.obj output\problem.obj &
           output\scram.obj output\des-x86.obj output\convdes.obj &
           output\clitime.obj output\clicdata.obj output\clirate.obj &
           output\clisrate.obj output\BDESLOW.OBJ &
           output\RG-486.OBJ output\RG-6X86.OBJ &
           output\RG-K5.OBJ output\RG-K6.OBJ &
           output\RC5P5BRF.OBJ output\RG-P6.OBJ &
           output\BBDESLOW.OBJ output\X86IDENT.OBJ 
           # this list can be added to in the platform specific section

%STACKSIZE= 32767          #may be redefined in the platform specific section
%AFLAGS   = /5s /fp3 /mf   #may be defined in the platform specific section
%LFLAGS   =                #may be defined in the platform specific section
%CFLAGS   = /6s /fp3 /ei /mf #may be defined in the platform specific section
%OPT_SIZE = /s /os /zp1    #may be redefined in the platform specific section
%OPT_SPEED= /oneatx /oh /oi+ /zp8 #redefine in platform specific section
%LIBPATH  =                #may be defined in the platform specific section
%LIBFILES =                #may be defined in the platform specific section
%EXTOBJS  =                #extra objs (made elsewhere) but need linking here
%MODULES  =                #may be defined in the platform specific section
%IMPORTS  =                #may be defined in the platform specific section
%BINNAME  =                #may be defined in the platform specific section
%COPYRIGHT=                #may be defined in the platform specific section
%FORMAT   =                #may be defined in the platform specific section

!ifdef SILENT
.silent
!endif

#-----------------------------------------------------------------------

LNKbasename = rc5des     # for 'rc564'.err 'rc564'.lnk 'rc5des'.err etc
noplatform: .symbolic
  @echo .>CON:
  @echo Platform has to be specified. >CON:
  @echo      eg: WMAKE [-f makefile] os2 >CON:
  @echo          WMAKE [-f makefile] netware >CON:
  @echo .>CON:
  @%abort

#-----------------------------------------------------------------------

clean :
  erase rc5des.lnk
  erase rc5des*.exe
  erase *.bak
  erase output\*.obj
  erase common\*.bak
  erase common\*.err
  erase des\*.bak
  erase des\*.err
  erase rc5\*.bak
  erase rc5\*.err
  erase platforms\win32-os2\*.bak

zip :
  *zip -r zip *

output\cliconfig.obj : common\cliconfig.cpp common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[: 
  @set isused=1

output\client.obj : common\client.cpp common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp common\problem.h common\network.h common\cputypes.h common\autobuff.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\convdes.obj : common\convdes.cpp makefile.wat  
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp common\client.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp common\clitime.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp common\clicdata.h   makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clirate.obj : common\clirate.cpp common\clirate.h  makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp common\clisrate.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clistime.obj : common\clistime.cpp common\clistime.h makefile.wat  
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\deseval-meggs2.obj : des\deseval-meggs2.cpp makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\deseval-meggs3.obj : des\deseval-meggs3.cpp makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\deseval-slice-meggs.obj : des\deseval-slice-meggs.cpp makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\des-x86.obj : des\des-x86.cpp common\problem.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ /fo=$^@ /fr=$[: /i$[:;common
  @set isused=1

output\autobuff.obj : common\autobuff.cpp common\autobuff.h common\cputypes.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\network.obj : common\network.cpp common\network.h common\cputypes.h common\autobuff.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp common\iniread.h common\cputypes.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp common\scram.h common\cputypes.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp common\mail.h common\network.h common\client.h common\cputypes.h common\autobuff.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\buffwork.obj : common\buffwork.cpp common\buffwork.h common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[: 
  @set isused=1

output\rg-486.obj : platforms\win32-os2\rg-486.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rg-6x86.obj : platforms\win32-os2\rg-6x86.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rc5p5brf.obj : platforms\win32-os2\rc5p5brf.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rg-p5.obj : platforms\win32-os2\rg-p5.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rg-p6.obj : platforms\win32-os2\rg-p6.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rg-k5.obj : platforms\win32-os2\rg-k5.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\rg-k6.obj : platforms\win32-os2\rg-k6.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\bdeslow.obj: platforms\win32-os2\bdeslow.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\Bbdeslow.obj: platforms\win32-os2\Bbdeslow.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\bdeshgh.obj: platforms\win32-os2\bdeshgh.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\Bbdeshgh.obj: platforms\win32-os2\Bbdeshgh.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\x86ident.obj : platforms\win32-os2\x86ident.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\netware.obj : platforms\netware\netware.cpp common\client.h makefile.wat 
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:;common
  @set isused=1

output\hbyname.obj : platforms\netware\hbyname.cpp makefile.wat
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\chklocks.obj : platforms\win32-os2\chklocks.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

output\clearscr.obj : platforms\win32-os2\clearscr.asm makefile.wat
  *$(CCASM) $(%AFLAGS) $[@ /fo=$^@ /fr=$[: /i$[:
  @set isused=1

#-----------------------------------------------------------------------

platform: .symbolic
  @set isused=
  @for %i in ($(%LINKOBJS)) do @%make %i
  @if not exist $(%BINNAME) @set isused=1
  @if $(%isused).==. @echo All targets are up to date
  @if not $(%isused).==. @%make platlink

platlink: .symbolic
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
     @set OPT_SIZE  = /s /os /zp1
     @set OPT_SPEED = /oneatx /oh /oi+ /zp8
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj
     @set LIBFILES  =
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).com
     @%make platform

os2: .symbolic                                       # OS/2
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /5s /fp5 /bm /mf /bt=os2 /DOS2 /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexi /zp4
     @set OPT_SPEED = /oantrlexi /zp8
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).exe
     @set STACKSIZE = 16384                 #Will slow down client if it's 32k
     @%make platform

w32: .symbolic                               # win95/winnt standard executable
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt
     @set CFLAGS    = /fpd /5s /fp5 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexih /zp4
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     @set LIBFILES  =
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).exe
     @%make platform

w_h: .symbolic                               # win95 hidden executable
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt_win
     @set CFLAGS    = /fpd /6s /fp6 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexih /zp4
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     @set LIBFILES  =
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename)h.exe
     @%make platform

wsv: .symbolic                               # winnt service
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys nt
     @set CFLAGS    = /fpd /6s /fp6 /bm /mf /bt=nt /DWIN32 /DMULTITHREAD /DWINNTSERVICE="bovrc5nt"
     @set OPT_SIZE  = /oantrlexih /zp4
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     @set LIBFILES  =
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename)srv.exe
     @%make platform

netware: .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM! (May 24 '98)
     @set STACKSIZE = 32767 #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set LFLAGS    = op multiload op nod op scr 'none' op map op osname='NetWare NLM' # symtrace systemConsoleScreen  #sys netware
     @set OPT_SIZE  = /os /s /zp1 
     @set OPT_SPEED = /oneatx /oh /oi+ /zp8  
     @set CFLAGS    = /6s /fp3 /ei /ms /d__NETWARE__ /dMULTITHREAD /i$(inc_386) $(wcc386opt) #/fpc /bt=netware /i$(%watcom)\novh #/bm
     @set LIBFILES =  nwwatemu,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 # tcpip netdb
     @set EXTOBJS   =
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
