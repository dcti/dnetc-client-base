## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 ]
##                or anything else with a section at the end of this file
##
## $Log: makefile.wat,v $
## Revision 1.26  1999/02/22 02:16:44  cyp
## adjusted for new rc5 core location
##
## Revision 1.25  1999/01/01 02:45:14  cramer
## Part 1 of 1999 Copyright updates...
##
## Revision 1.24  1998/11/16 22:42:21  cyp
## Brought up to date
##
## Revision 1.21  1998/07/19 20:06:11  cyruspatel
## Makefile now also gens the upload readme, eg rc5des-416-dos-x86-cli.readme
##
## Revision 1.20  1998/07/19 17:52:10  cyruspatel
## Automated tasm support. make will now also gen a zip if configured for it.
##
## Revision 1.19  1998/07/12 17:40:50  cyruspatel
## Different DES versions can now be made by changing a define.
##
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
## Added the pathwork.cpp file to the link objs and the make process. Also 
## moved the Cyrix core around, it seems to speed up the core a bit. Well 
## at least it prevented it from being slowed down, in OS/2 at least.
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

## $Id: makefile.wat,v 1.26 1999/02/22 02:16:44 cyp Exp $


%EXTOBJS  = #extra objs (made elsewhere) but need linking here
%DEFALL   = /DPIPELINE_COUNT=2 /DBRYD /D__showids__
            #defines used everywhere
%SYMALIAS = # symbols that need redefinition 
%LINKOBJS = output\p1bdespro.obj output\bdeslow.obj   &  
            output\des-x86.obj   output\convdes.obj   output\problem.obj  &
            output\rg486.obj     output\rc5-rgk5.obj  output\rc5-rgk6.obj &
            output\brfp5.obj     output\rc5-rgp6.obj  output\rg6x86.obj   &
            output\confrwv.obj   output\autobuff.obj  output\buffwork.obj &
            output\mail.obj      output\client.obj    output\disphelp.obj &
            output\iniread.obj   output\network.obj   output\scram.obj    &
            output\clitime.obj   output\clicdata.obj  output\clirate.obj  &
            output\clisrate.obj  output\cpucheck.obj  output\pathwork.obj &
            output\cliident.obj  output\checkpt.obj   output\x86ident.obj &
            output\logstuff.obj  output\triggers.obj  output\buffupd.obj  &
            output\selcore.obj   output\netinit.obj   output\cmdline.obj  &
            output\selftest.obj  output\pollsys.obj   output\probman.obj  &
            output\probfill.obj  output\clievent.obj  output\bench.obj    &
            output\clirun.obj    output\setprio.obj   output\console.obj  &
            output\modereq.obj   output\confmenu.obj  output\confopt.obj
            

            # this list can be added to in the platform specific section
            # 47 std OBJ's (+3 desmmx, +1 rc5mmx, +2 mt, +x plat specific)

#---
%rc5smc_LINKOBJS = output\rc5-486-smc-rg.o
#---
%rc5mmx_LINKOBJS = output\rc5mmx.obj 
%rc5mmx_DEFALL   = /DMMX_RC5 
#---
%desmmx_LINKOBJS = output\des-slice-meggs.obj output\deseval-mmx.obj
%desmmx_DEFALL   = /DMEGGS /DMMX_BITSLICER /DBIT_64 #/DBITSLICER_WITH_LESS_BITS
%desmmx_SYMALIAS = #
#---
%brydmt_LINKOBJS = output\p2bdespro.obj output\bbdeslow.obj 
%brydmt_DEFALL   = #
#---
%kwan_LINKOBJS   = output\des-slice.obj output\deseval.obj output\sboxes-kwan4.obj
%kwan_DEFALL     = /DKWAN #/DBIT_32
#---
%mt_LINKOBJS     = $(%brydmt_LINKOBJS) $(%kwan_LINKOBJS)
%mt_DEFALL       = /DMULTITHREAD $(%kwan_DEFALL) $(%brydmt_DEFALL) 
#---

#-----------------------------------------------------------------------

%CC=wpp386
%CCASM=wasm
%LINK=wlink #\develop\watcom\binnt\wlink.exe
LNKbasename = rc5des       # for 'rc564'.err 'rc564'.lnk 'rc5des'.err etc

%NASMEXE  = nasm           #point this to nasm (don't call the envvar 'NASM'!)
%NASMFLAGS= -f coff -s     #nothing special required #elf/win32/as86/aout
%TASMEXE  =                #point this to tasm in your section if you have it
%TFLAGS   = /ml /m9 /q /t  #if TASMEXE.==. then wasm will be executed
%STACKSIZE= 32K            #may be redefined in the platform specific section
%AFLAGS   = /5s /fp3 /mf   #may be defined in the platform specific section
%LFLAGS   =                #may be defined in the platform specific section
%CFLAGS   = /6s /fp3 /ei /mf #may be defined in the platform specific section
%OPT_SIZE = /s /os         #may be redefined in the platform specific section
%OPT_SPEED= /oneatx /oh /oi+ #redefine in platform specific section
%LIBPATH  =                #may be defined in the platform specific section
%DEBUG    =                #@%make debug to enable debugging
%LIBFILES =                #may be defined in the platform specific section
%MODULES  =                #may be defined in the platform specific section
%IMPORTS  =                #may be defined in the platform specific section
%BINNAME  =                #must be defined in the platform specific section
%COPYRIGHT=                #may be defined in the platform specific section
%FORMAT   =                #may be defined in the platform specific section
%WLINKOPS = map            #one word wlink OP-tions. no spaces but '=' is ok
%ERRDIROP = /fr=$$[:       #Puts the .err files in the right directories
%dependall= # makefile.wat common/version.h  # remake everything if these change

%ZIPFILE  = # eg -dos-x86-cli or blank for auto
%DOCFILES =                #list of files in ./docs to include in the zip
%ZIPPER   = zip.exe        # a zip file won't be made if not defined
%ZIPOPTS  = #-u -9 -o -i -v 
                           
#.silent
#.nocheck

#-----------------------------------------------------------------------

noplatform: .symbolic
  @%write con 
  @%write con   Platform has to be specified. 
  @%write con      eg: WMAKE [-f makefile] os2 
  @%write con          WMAKE [-f makefile] netware 
  @%write con  
  @%quit

#-----------------------------------------------------------------------

clean :  .symbolic
  @set dirlist = output common des rc5
  @for %i in ($(%dirlist)) do @if exist %i\*.obj @del %i\*.obj 
  @for %i in ($(%dirlist)) do @if exist %i\*.bak @del %i\*.bak 
  @for %i in ($(%dirlist)) do @if exist %i\*.~?? @del %i\*.~??
  @for %i in ($(%dirlist)) do @if exist %i\*.err @del %i\*.err 
  @if not exist .\$(LNKbasename)*.* @%quit
  @for %i in (.\$(LNKbasename)*.*) do @del %i

zip :  .symbolic
  @if $(%ZIPPER).==.  @echo Error(E02): ZIPPER is not defined
  @if $(%ZIPPER).==.  @%abort
  @if $(%ZIPFILE).==. @set ZIPFILE=$(LNKbasename)-$(OSNAME)-x86-cli
  
  @if exist $(%ZIPFILE).zip @del $(%ZIPFILE).zip >nul:
  @%write con 
  @echo Generating $(%ZIPFILE).zip...
  @$(%ZIPPER) $(%ZIPOPTS) $(%ZIPFILE).zip $(%BINNAME) $(%DOCFILES)

debug : .symbolic
  @set DEBUG=1

#-----------------------------------------------------------------------

declare_for_desmmx : .symbolic
  @set LINKOBJS = $(%desmmx_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%desmmx_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%desmmx_SYMALIAS) $(%SYMALIAS) 

declare_for_rc5mmx : .symbolic
  @set LINKOBJS = $(%rc5mmx_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%rc5mmx_DEFALL) $(%DEFALL) 

declare_for_multithread : .symbolic
  @set LINKOBJS = $(%mt_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%mt_DEFALL) $(%DEFALL) 

declare_for_rc5smc : .symbolic
  @set LINKOBJS = $(%rc5smc_LINKOBJS) $(%LINKOBJS)
  #@set DEFALL   = $(%rc5smc_DEFALL) $(%DEFALL) 

#-----------------------------------------------------------------------

output\rc5-486-smc-rg.o : rc5\x86\rc5-486-smc-rg.o
  @if not exist $[@ @%write con File not found: $[@
  @if not exist $[@ @%quit
  copy $[@ $^@ >nul: 
  wtouch $^@
  @echo Updated $^@ from $[@
  @set isused=1

output\rg486.obj : rc5\x86\nasm\rg486.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rc5-rgk5.obj : rc5\x86\nasm\rc5-rgk5.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rc5-rgk6.obj : rc5\x86\nasm\rc5-rgk6.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\brfp5.obj : rc5\x86\nasm\brfp5.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rc5-rgp6.obj : rc5\x86\nasm\rc5-rgp6.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rg6x86.obj : rc5\x86\nasm\rg6x86.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

#output\rg-p5.obj : rc5\rg-p5.asm $(%dependall)
#  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
#  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
#  @set isused=1

output\rc5mmx.obj : rc5\x86\nasm\rc5mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\des-slice-meggs.obj : des\des-slice-meggs.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\deseval-mmx.obj : des\mmx-bitslice\deseval-mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

#output\des-mmx.obj : des\mmx-bitslice\des-mmx.asm $(%dependall) 
#  *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
#  @set isused=1

#-----------------------------------------

output\des-slice.obj : des\des-slice.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\x86-slic.obj : des\x86-slic.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\deseval.obj : des\deseval.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-kwan4.obj : des\sboxes-kwan4.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-k.obj : des\sboxes-k.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-kwan3.obj : des\sboxes-kwan3.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

#-----------------------------------------

output\p1bdespro.obj : des\brydmasm\p1bdespro.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%write con File not found: $(%x) 
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\p2bdespro.obj : des\brydmasm\p2bdespro.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bdeslow.obj : des\brydmasm\bdeslow.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bbdeslow.obj : des\brydmasm\bbdeslow.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\x86ident.obj : platforms\x86ident.asm $(%dependall)
  *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confrwv.obj : common\confrwv.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confopt.obj : common\confopt.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confmenu.obj : common\confmenu.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pollsys.obj : common\pollsys.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probman.obj : common\probman.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probfill.obj : common\probfill.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\bench.obj : common\bench.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clirun.obj : common\clirun.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\checkpt.obj : common\checkpt.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\setprio.obj : common\setprio.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\console.obj : common\console.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cpucheck.obj : common\cpucheck.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\client.obj : common\client.cpp  $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cmdline.obj : common\cmdline.cpp  $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffupd.obj : common\buffupd.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selcore.obj : common\selcore.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selftest.obj : common\selftest.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\convdes.obj : common\convdes.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clirate.obj : common\clirate.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clistime.obj : common\clistime.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pathwork.obj : common\pathwork.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\autobuff.obj : common\autobuff.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\network.obj : common\network.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\netinit.obj : common\netinit.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffwork.obj : common\buffwork.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: 
  @set isused=1

output\cliident.obj : common\cliident.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clievent.obj : common\clievent.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\triggers.obj : common\triggers.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\modereq.obj : common\modereq.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\logstuff.obj : common\logstuff.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\des-x86.obj : des\des-x86.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\netware.obj : platforms\netware\netware.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\hbyname.obj : platforms\netware\hbyname.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clidos.obj : platforms\dos\clidos.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32svc.obj : platforms\win32cli\w32svc.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32cons.obj : platforms\win32cli\w32cons.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32pre.obj : platforms\win32cli\w32pre.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32sock.obj : platforms\win32cli\w32sock.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\os2inst.obj : platforms\os2cli\os2inst.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\lurk.obj : common\lurk.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1


#-----------------------------------------------------------------------

platform: .symbolic
  @set CFLAGS    = $(%CFLAGS) /zq             ## compile quietly
  @set AFLAGS    = $(%AFLAGS) /q              ## assemble quietly
  @set CFLAGS    = $(%CFLAGS) $(%DEFALL)      ## tack on global defines
  @set isused=0
  @if not exist $(%BINNAME) @set isused=1
  @for %i in ($(%LINKOBJS)) do @%make %i
  @if $(%isused).==0. @%write con All targets are up to date
  @if $(%isused).==0. @%quit
  @%make dolink
  @if not exist $(%BINNAME) @%quit
  @%make postlink
  @if not exist $(%BINNAME) @%quit
  @%make zip

dolink : .symbolic
  @if exist  $(%BINNAME) @del $(%BINNAME)
  @if exist  $(LNKbasename).lnk @del $(LNKbasename).lnk
  @%append   $(LNKbasename).lnk Name $(%BINNAME)
  @for %i in ($(%STACKSIZE)) do @%append $(LNKbasename).lnk Op Stack=%i
  @for %i in ($(%LINKOBJS))  do @%append $(LNKbasename).lnk File %i
  @for %i in ($(%EXTOBJS))   do @%append $(LNKbasename).lnk File %i
  @for %i in ($(%LIBPATH))   do @%append $(LNKbasename).lnk Libpath %i
  @for %i in ($(%MODULES))   do @%append $(LNKbasename).lnk Module %i
  @for %i in ($(%IMPORTS))   do @%append $(LNKbasename).lnk Import %i
  @for %i in ($(%SYMALIAS))  do @%append $(LNKbasename).lnk Alias %i
  @for %i in ($(%WLINKOPS))  do @%append $(LNKbasename).lnk Op %i
  @if not $(%DEBUG).==. @%append $(LNKbasename).lnk debug all
  @if not $(%DEBUG).==. @%append $(LNKbasename).lnk Op map
  @if not $(%DEBUG).==. @%append $(LNKbasename).lnk Op verbose
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
  @if exist  $(%BINNAME) @del $(%BINNAME)
  *$(%LINK) $(%LFLAGS) @$(LNKbasename).lnk > $(LNKbasename).err
  @if exist $(%BINNAME) @del $(LNKbasename).err
  @if exist $(LNKbasename).err @type $(LNKbasename).err

.ERROR
  @if $(%BINNAME).==. @%quit
  @if not exist $(%BINNAME) @%quit
  @del $(%BINNAME)
  @echo Target '$(%BINNAME)' deleted.

postlink: .symbolic
  @if $(%OSNAME).==netware.  @\develop\sdkcdall\nlmdump\nlm_dos.exe *$(LNKbasename).nlm /b:$(LNKbasename).map 
  #@if $(%OSNAME).==netware. @\develop\sdkcd13\nwsdk\tools\nlmpackx $(LNKbasename).nlm $(LNKbasename).nlx
  #@if $(%OSNAME).==netware. @del $(LNKbasename).nlm
  #@if $(%OSNAME).==netware. @ren $(LNKbasename).nlx $(LNKbasename).nlm

  @if $(%OSNAME).==dos.      @\develop\pmodew\pmwlite.exe /C4 /S\develop\pmodew\pmodew.exe $(%BINNAME)
  @if $(%OSNAME).==dos.      @\develop\pmodew\pmwsetup.exe /b0 /q $(%BINNAME)

  @if exist $(LNKbasename).rex @del $(LNKbasename).rex
  @if $(%OSNAME).==win16.    @ren $(LNKbasename).exe $(LNKbasename).rex 
  #@if $(%OSNAME).==win16.   @wbind rc5des -n
  #@if $(%OSNAME).==win16.   @echo "1 ICON platforms\win32gui\cowhead.ico" > $(LNKbasename).rc
  @if $(%OSNAME).==win16.    @wrc -r -i=. -fo=.\$(LNKbasename).res -bt=windows -30 platforms\win32cli\w32cons.rc
  @if $(%OSNAME).==win16.    @wbind $(LNKbasename) -R $(LNKbasename).res
  @if exist $(LNKbasename).rex @del $(LNKbasename).rex

  #@if $(%OSNAME).==win32.   @echo "1 ICON platforms\win32gui\cowhead.ico" > $(LNKbasename).rc
  @if $(%OSNAME).==win32.    @wrc /bt=nt platforms\win32cli\w32cons.rc $(%BINNAME)

  #@if exist $(LNKbasename).rc @del $(LNKbasename).rc >nul:
  #@if exist $(LNKbasename).res @del $(LNKbasename).res >nul:

#---------------------- platform specific settings come here ----------

dos: .symbolic                                       # DOS/PMODE
     @set OSNAME    = dos
     @set AFLAGS    = /5s /fp3 /bt=netware /ms # no such thing as /bt=dos4g
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\dos 
     @set WLINKOPS  = dosseg eliminate map # stub=\develop\pmodew\pmodew.exe
                                           #stub=platforms/dos/d4GwStUb.CoM 
     @set LFLAGS    = symtrace printf symtrace whack16 
     @set FORMAT    = os2 le
     @set CFLAGS    = /zp8 /wx /we /6s /fp3 /fpc /zm /ei /mf &
                      /wcd=604 /wcd=594 /wcd=7 /bt=dos /d__MSDOS__ &
                      /iplatforms/dos /I$(%watcom)\h #/Iplatforms\dos\libtcp 
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clidos.obj
     @set LIBFILES  = #platforms\dos\libtcp\libtcp.a
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\readme.dos docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-dos-x86-cli
     @set BINNAME   = $(LNKbasename).com
     #@%make declare_for_rc5smc
     @%make declare_for_rc5mmx
     @%make declare_for_desmmx
     @%make platform

d16: .symbolic                                       # DOS/large model
     @set OSNAME    = d16
     @set CC        = wpp
     @set AFLAGS    = /3s /bt=dos /ml
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set LIBPATH   = $(%watcom)\lib286 $(%watcom)\lib286\dos 
     @set WLINKOPS  = dosseg eliminate map # stub=\develop\pmodew\pmodew.exe
                                           #stub=platforms/dos/d4GwStUb.CoM 
     @set LFLAGS    = symtrace printf symtrace whack16 
     @set FORMAT    = dos
     @set CFLAGS    = /zp1 /3 /wx /we /fpc /zc /zt1 /ei /ml /d__MSDOS__ &
                      /iplatforms/dos /i$(%watcom)\h
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj output\clidos.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\readme.dos docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-dos-x86-cli
     @set BINNAME   = $(LNKbasename).com
     #@%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

os2: .symbolic                                       # OS/2
     @set OSNAME    = os2
     @set AFLAGS    = /5s /fp5 /bt=OS2 /mf
     @set TASMEXE   = tasm32.exe
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /zp8 /5s /fp5 /bm /mf /bt=os2 /DOS2 /DLURK
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oantrlexi 
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\readme.os2 docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-os2-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @set STACKSIZE = 32K  # 16384        #Will slow down client if it's 32k
     @set LINKOBJS  = output\os2inst.obj  output\lurk.obj
     @set OBJDIROP  = /fo=output\
     @set ERRDIROP  =                      # no /fr= option for Watcom 10.0
     @%make declare_for_multithread
     @%make declare_for_rc5mmx
     @%make declare_for_desmmx
     @%make platform

w16: .symbolic                                       # Windows/16
     @set OSNAME    = win16
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set LFLAGS    = system win386 #debug all op de 'RC5DES Client for Windows'
     @set CFLAGS    = /3s /w4 /zW /bt=windows /d_Windows &
                      /i$(%watcom)\h;$(%watcom)\h\win /iplatforms/win32cli &
                      /DBITSLICER_WITH_LESS_BITS 
                      #/d2
                      #/zp8 /6s /fp3 /fpc /zm /ei /mf /bt=dos /d_Windows &
                      #/d_ENABLE_AUTODEPEND /d__WINDOWS_386__ &
                      #/bw (bw causes default windowing lib to be linked)
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oaxt #/oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) &
                      output\w32cons.obj output\w32pre.obj output\w32sock.obj
                      #output\clearscr.obj output\clidos.obj &
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-win16-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @if exist $(LNKbasename).rex @del $(LNKbasename).rex
     #@%make declare_for_multithread
     @%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

w32: .symbolic                               # win32
     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set WLINKOPS  = map #alignment=16 objalign=16
     @set LFLAGS    = sys nt_win op de 'RC5DES Client for Windows' #nt
     @set CFLAGS    = /zp8 /fpd /6s /fp3 /bm /mf /bt=nt /DWIN32 /DLURK &
                      /iplatforms/win32cli /i$(%watcom)\h;$(%watcom)\h\nt
     @set OPT_SIZE  = /oneatx /oh /oi+  #/s /os
     @set OPT_SPEED = /oneatx /oh /oi+ /ei #/oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\lurk.obj &
                      output\w32svc.obj output\w32cons.obj &
                      output\w32pre.obj output\w32sock.obj 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set ZIPFILE   = $(LNKbasename)-win32-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     #@%make debug
     @%make declare_for_desmmx
     @%make declare_for_rc5mmx
     @%make declare_for_multithread
     @%make platform

netware : .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM! (May 24 '98)
     @set OSNAME    = netware
     @set STACKSIZE = 32K #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set WLINKOPS  = xdcdata=platforms/netware/rc5des.xdc &
                      multiload nod map #osdomain
     @set LFLAGS    = op scr 'none' op version=0.0 op osname='NetWare NLM' #symtrace systemConsoleScreen #sys netware
     @set OPT_SIZE  = /os /s  
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /zp1 /zm /6s /fp3 /ei /ms /d__NETWARE__ &
                      /DBITSLICER_WITH_LESS_BITS /bt=netware &
                      /i$(inc_386) #/fpc /bt=netware /i$(%watcom)\novh #/bm
     @set LIBFILES  = nwwatemu,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 tli # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj output\hbyname.obj 
     @set EXTOBJS   = $(%EXTOBJS) platforms\netware\watavoid\i8s.obj
     @set IMPORTS   = GetNestedInterruptLevel AllocateResourceTag UngetKey &
                      GetCurrentTime OutputToScreenWithPointer OutputToScreen &
                      ActivateScreen ImportPublicSymbol UnImportPublicSymbol &
                      ScheduleSleepAESProcessEvent CancelSleepAESProcessEvent &
                      RingTheBell GetFileServerMajorVersionNumber Alloc &
                      GetSuperHighResolutionTimer ConvertTicksToSeconds &
                      ConvertSecondsToTicks NWSMPThreadToMP &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\tli.imp
                      # @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = platforms\netware\watavoid $(%watcom)\lib386 $(%watcom)\lib386\netware
     @set ZIPFILE   = $(LNKbasename)-netware-x86-cli
     @set ZIPPER    = c:\util\pkzip
     @set DOCFILES  = docs\readme.nw docs\rc5des.txt docs\readme.txt
     @set BINNAME   = $(LNKbasename).nlm
     @set COPYRIGHT = 'Copyright 1997-1999 distributed.net\r\n  Visit http://www.distibuted.net/ for more information'
     @set FORMAT    = Novell NLM 'RC5DES Client for NetWare'
     @set %dependall=
     #@%make declare_for_desmmx
     @%make declare_for_rc5mmx
     @%make declare_for_multithread
     @%make platform

