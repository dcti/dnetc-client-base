## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 | w_h | wsv ]
##                       or anything else defined at the end of this makefile
##

CC=wpp386
CCASM=wasm
LINK=wlink
TASM=tasm32.exe
TOPT=/ml /zi /m
%HAVE_TASM = 1              #set to 1 if you have it

%LINKOBJS = common\cliconfig.obj common\autobuff.obj common\buffwork.obj &
           common\mail.obj platforms\win32-os2\BDESLOW.OBJ common\client.obj &
           common\iniread.obj common\network.obj common\problem.obj &
           common\scram.obj des\des-x86.obj common\convdes.obj &
           platforms\win32-os2\RG-486.OBJ platforms\win32-os2\RG-6X86.OBJ &
           platforms\win32-os2\RG-K5.OBJ platforms\win32-os2\RG-K6.OBJ &
           platforms\win32-os2\RC5P5BRF.OBJ platforms\win32-os2\RG-P6.OBJ &
           platforms\win32-os2\BBDESLOW.OBJ platforms\win32-os2\X86IDENT.OBJ
           # this list can be added to in the platform specific section

%VERSION  = 70.10          #may be redefined in the platform specific section
%STACKSIZE= 32767          #may be redefined in the platform specific section
%AFLAGS   =                #may be defined in the platform specific section
%LFLAGS   =                #may be defined in the platform specific section
%CFLAGS   =                #may be defined in the platform specific section
%OPT_SIZE = /s /os /zp1    #may be redefined in the platform specific section
%OPT_SPEED= /oantrlexih /oi+ /zp8 #redefine in platform specific section
%LIBPATH  =                #may be defined in the platform specific section
%LIBFILES =                #may be defined in the platform specific section
%EXTOBJS  =                #extra objs (made elsewhere) but need linking here
%MODULES  =                #may be defined in the platform specific section
%IMPORTS  =                #may be defined in the platform specific section
%BINNAME  =                #may be defined in the platform specific section
%COPYRIGHT=                #may be defined in the platform specific section
%FORMAT   =                #may be defined in the platform specific section

!ifdef HAVE_TASM
%have_tasm=1
!endif
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
  *erase rc5des.lnk
  *erase rc5des*.exe
  *erase *.bak
  *erase common\*.bak
  *erase common\*.err
  *erase common\*.obj
  *erase des\*.bak
  *erase des\*.err
  *erase des\*.obj
  *erase rc5\*.bak
  *erase rc5\*.err
  *erase rc5\*.obj
  *erase platforms\win32-os2\*.bak
  @set isused=1

zip :
  *zip -r zip *
  @set isused=1

common\cliconfig.obj : common\cliconfig.cpp common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) cliconfig.cpp
  *cd ..
  @set isused=1

common\client.obj : common\client.cpp common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) client.cpp
  *cd ..
  @set isused=1

common\problem.obj : common\problem.cpp common\problem.h common\network.h common\cputypes.h common\autobuff.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) problem.cpp
  *cd ..
  @set isused=1

common\convdes.obj : common\convdes.cpp configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) convdes.cpp
  *cd ..
  @set isused=1

des\deseval-meggs2.obj : des\deseval-meggs2.cpp configure
  *cd des
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) deseval-meggs2.cpp
  *cd ..
  @set isused=1

des\deseval-meggs3.obj : des\deseval-meggs3.cpp configure
  *cd des
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) deseval-meggs3.cpp
  *cd ..
  @set isused=1

des\deseval-slice-meggs.obj : des\deseval-slice-meggs.cpp configure
  *cd des
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) deseval-slice-meggs.cpp
  *cd ..
  @set isused=1

des\des-x86.obj : des\des-x86.cpp common\problem.h configure
  *cd des
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) des-x86.cpp
  *cd ..
  @set isused=1

common\autobuff.obj : common\autobuff.cpp common\autobuff.h common\cputypes.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) autobuff.cpp
  *cd ..
  @set isused=1

common\network.obj : common\network.cpp common\network.h common\cputypes.h common\autobuff.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) network.cpp
  *cd ..
  @set isused=1

common\iniread.obj : common\iniread.cpp common\iniread.h common\cputypes.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) iniread.cpp
  *cd ..
  @set isused=1

common\scram.obj : common\scram.cpp common\scram.h common\cputypes.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) scram.cpp
  *cd ..
  @set isused=1

common\mail.obj : common\mail.cpp common\mail.h common\network.h common\client.h common\cputypes.h common\autobuff.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) mail.cpp
  *cd ..
  @set isused=1

common\buffwork.obj : common\buffwork.cpp common\client.h common\problem.h common\scram.h common\mail.h common\network.h common\iniread.h configure
  *cd common
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) buffwork.cpp
  *cd ..
  @set isused=1

platforms\win32-os2\rg-486.obj : platforms\win32-os2\rg-486.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-486.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rg-6x86.obj : platforms\win32-os2\rg-6x86.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-6x86.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rc5p5brf.obj : platforms\win32-os2\rc5p5brf.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rc5p5brf.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rg-p5.obj : platforms\win32-os2\rg-p5.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-p5.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rg-p6.obj : platforms\win32-os2\rg-p6.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-p6.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rg-k5.obj : platforms\win32-os2\rg-k5.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-k5.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\rg-k6.obj : platforms\win32-os2\rg-k6.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) rg-k6.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\bdeslow.obj: platforms\win32-os2\bdeslow.asm platforms\win32-os2\bdesmac.inc platforms\win32-os2\bdeschg.inc platforms\win32-os2\bdesdat.inc
  *cd platforms\win32-os2
  @if not $(%have_tasm).==. *$(TASM) $(TOPT) bdeslow.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\Bbdeslow.obj: platforms\win32-os2\Bbdeslow.asm platforms\win32-os2\Bbdesmac.inc platforms\win32-os2\Bbdeschg.inc platforms\win32-os2\Bbdesdat.inc
  *cd platforms\win32-os2
  @if not $(%have_tasm).==. *$(TASM) $(TOPT) Bbdeslow.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\bdeshgh.obj: platforms\win32-os2\bdeshgh.asm
  *cd platforms\win32-os2
  @if not $(%have_tasm).==. *$(TASM) $(TOPT) bdeshgh.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\Bbdeshgh.obj: platforms\win32-os2\Bbdeshgh.asm
  *cd platforms\win32-os2
  @if not $(%have_tasm).==. *$(TASM) $(TOPT) Bbdeshgh.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\x86ident.obj : platforms\win32-os2\x86ident.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) x86ident.asm
  *cd ..\..
  @set isused=1

platforms\win32-os2\chklocks.obj : platforms\win32-os2\chklocks.asm configure
  *cd platforms\win32-os2
  *$(CCASM) $(%AFLAGS) chklocks.asm
  *cd ..\..
  @set isused=1

platforms\netware\netware.obj : platforms\netware\netware.cpp common\client.h configure
  *cd platforms\netware
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) netware.cpp
  *cd ..\..
  @set isused=1

#-----------------------------------------------------------------------

platform: .symbolic
  @set isused=
  @if not ($(HAVE_TASM)).==. @set have_tasm=1
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
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys dos4g
     @set VERSION   = #dos4gw doesn't support version switch
     @set CFLAGS    = /5s /fp5 /mf /bt=dos /DDOS4G /DNONETWORK
     @set OPT_SIZE  = /oantrlexih /zp4
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     @set LIBFILES  =
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename)D.exe
     @%make platform

os2: .symbolic                                       # OS/2
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /5s /fp5 /bm /mf /bt=os2 /DOS2 /DMULTITHREAD -i$(%watcom)\h\os2
     @set OPT_SIZE  = /oantrlexih /zp4
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     @set LIBFILES  = libf so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set EXTOBJS   =
     @set IMPORTS   =
     @set BINNAME   = $(LNKbasename).exe
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

netware_all: .symbolic                       # NetWare NLM
     @set AFLAGS    = /5s /fp5 /bt=NETWARE /ms
     @set LFLAGS    = op nod op scr 'none' op map op osname='NetWare NLM' #symtrace xxx sys netware
     ##   CFLAGS    = /fpc /5 /fpc /ei /ms /bt=netware /i$(%watcom)\novh /bm /DMULTITHREAD
     @set OPT_SIZE  = /oantrlexih /zp4   #/s /os /zp1
     @set OPT_SPEED = /oantrlexih /oi+ /zp8
     ###  LIBFILES  = plibmt3s,clib3s,math387s,emu387#, #if compiled with /bm use 'plibmt3s' instead
     @set MODULES   = clib a3112 #tcpip #netdb
     @set EXTOBJS   =
     @set LINKOBJS  = $(%LINKOBJS) netware.obj
     @set IMPORTS   = NumberOfPollingLoops gethostname CRescheduleMyself &
                      ExternalPublicList InternalPublicList Alloc &
                      MaximumNumberOfPollingLoops GetNestedInterruptLevel &
                      AllocateResourceTag CurrentTime GetCurrentTime &
                      GetProcessorSpeedRating OutputToScreenWithPointer &
                      systemConsoleScreen DiskIOsPending activeScreen &
                      ActivateScreen ImportPublicSymbol UnImportPublicSymbol &
                      ScheduleNoSleepAESProcessEvent CancelNoSleepAESProcessEvent &
                      NDirtyBlocks CRescheduleLast RingTheBell &
                      OutputToScreenWithPointer &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\netware
     @set BINNAME   = $(LNKbasename).nlm
     @set COPYRIGHT ='Visit http://www.distibuted.net/ for more information'
     @set VERSION   = 2.70.10
     @set FORMAT    = Novell NLM 'RC5-DES v2.7021.404 Client [Project Monarch]'
     @%make platform

netware_smp: .symbolic                            # NetWare NLM (SMP)
     @set CFLAGS   = /fpc /5 /fpc /ei /ms /bt=netware /i$(%watcom)\novh &
                     /bm /DMULTITHREAD
     @set LIBFILES = plibmt3s,clib3s,math387s,emu387
     @%make netware_all
     @\develop\sdkcd13\nwsdk\tools\nlmpackx $(LNKbasename).nlm $(LNKbasename).nlx
     @del $(LNKbasename).nlm
     @ren $(LNKbasename).nlx $(LNKbasename).nlm

netware: .symbolic                            # NetWare NLM (non-SMP)
     @set CFLAGS   = /fpc /5 /fpc /ei /ms /bt=netware /i$(%watcom)\novh
     @set LIBFILES = plib3s,clib3s,math387s,emu387
     @%make netware_all
