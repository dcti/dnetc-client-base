## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 | w_h | wsv ]
##                       or anything else defined at the end of this makefile
##
## $Log: makefile.wat,v $
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

## $Id: makefile.wat,v 1.25 1999/01/01 02:45:14 cramer Exp $

%VERMINOR = 418b      # for zip - fixit if not the same as version.h
%VERMAJOR = 7100      # for NetWare copyright: v2.$(%VERMAJOR).$(%VERMINOR)
%FILEVER  = 71.00     # for when the binary has an embedded version #

%EXTOBJS  = #extra objs (made elsewhere) but need linking here
%DEFALL   = /DPIPELINE_COUNT=2 /DBRYD /D__showids__
            #defines used everywhere
%SYMALIAS = # symbols that need redefinition 
%LINKOBJS = output\p1bdespro.obj output\bdeslow.obj   &  
            output\des-x86.obj   output\convdes.obj   output\problem.obj  &
            output\rg-486.obj    output\rg-k5.obj     output\rg-k6.obj    &
            output\rc5p5brf.obj  output\rg-p6.obj     output\rg-6x86.obj  &
            output\cliconfig.obj output\autobuff.obj  output\buffwork.obj &
            output\mail.obj      output\client.obj    output\disphelp.obj &
            output\iniread.obj   output\network.obj   output\scram.obj    &
            output\clitime.obj   output\clicdata.obj  output\clirate.obj  &
            output\clisrate.obj  output\cpucheck.obj  output\pathwork.obj &
            output\cliident.obj  output\threadcd.obj  output\x86ident.obj &
            output\logstuff.obj  output\triggers.obj  output\buffupd.obj  &
            output\selcore.obj   output\netinit.obj   output\cmdline.obj  &
            output\selftest.obj  output\pollsys.obj   output\probman.obj  &
            output\probfill.obj  output\guistuff.obj  output\bench.obj    &
            output\clirun.obj    output\setprio.obj   output\console.obj  &
            output\modereq.obj
            # this list can be added to in the platform specific section

            # 45 std OBJ's (+3 desmmx, +1 rc5mmx, +2 mt, +x plat specific)

%rc5mmx_LINKOBJS = output\rc5mmx.obj 
%rc5mmx_DEFALL   = /DMMX_RC5 
%desmmx_LINKOBJS = output\sboxmmx.o output\desm3mmx.o &
                   output\des-slice-meggs.obj 
#%desmmx_LINKOBJS = output\sboxes-mmx.obj output\deseval-meggs3-mmx.obj &
#                   output\des-slice-meggs.obj 
%desmmx_DEFALL   = /DMMX_BITSLICER /DBIT_64 /DBITSLICER_WITH_LESS_BITS 
%desmmx_SYMALIAS = whack16=_whack16 _malloc=malloc _free=free
%mt_LINKOBJS     = output\p2bdespro.obj output\bbdeslow.obj 
%mt_DEFALL       = /DMULTITHREAD

#-----------------------------------------------------------------------

%CC=wpp386
%CCASM=wasm
%LINK=wlink #\develop\watcom\binnt\wlink.exe
LNKbasename = rc5des       # for 'rc564'.err 'rc564'.lnk 'rc5des'.err etc

%NASMEXE  = nasm           #point this to nasm (don't call the envvar 'NASM'!)
%NASMFLAGS= -f coff -s     #nothing special required
%TASMEXE  =                #point this to tasm in your section if you have it
%TFLAGS   = /ml /m9 /q /t  #if TASMEXE.==. then wasm will be executed
%STACKSIZE= 32K            #may be redefined in the platform specific section
%AFLAGS   = /5s /fp3 /mf   #may be defined in the platform specific section
%LFLAGS   =                #may be defined in the platform specific section
%CFLAGS   = /6s /fp3 /ei /mf #may be defined in the platform specific section
%OPT_SIZE = /s /os         #may be redefined in the platform specific section
%OPT_SPEED= /oneatx /oh /oi+ #redefine in platform specific section
%LIBPATH  =                #may be defined in the platform specific section
%LIBFILES =                #may be defined in the platform specific section
%MODULES  =                #may be defined in the platform specific section
%IMPORTS  =                #may be defined in the platform specific section
%BINNAME  =                #must be defined in the platform specific section
%COPYRIGHT=                #may be defined in the platform specific section
%FORMAT   =                #may be defined in the platform specific section
%WLINKOPS = map            #one word wlink OP-tions. no spaces but '=' is ok
%ERRDIROP = /fr=$$[:       #Puts the .err files in the right directories
%dependall= # makefile.wat common/version.h  # remake everything if these change

%ZIPFILE  = # eg $(LNKbasename)-$(%VERMINOR)-dos-x86-cli or blank for auto
%DOCFILES =                #list of files in ./docs to include in the zip
%PORTER   =                # your name and email address
%ZIPPER   = zip.exe        # a zip file won't be made if not defined
%ZIPOPTS  = #-u -9 -o -i -v 
                           
#.silent
.nocheck

#-----------------------------------------------------------------------

noplatform: .symbolic
  @%write con 
  @%write con   Platform has to be specified. 
  @%write con      eg: WMAKE [-f makefile] os2 
  @%write con          WMAKE [-f makefile] netware 
  @%write con  
  @%quit

#-----------------------------------------------------------------------

clean : 
  @set dirlist = output common des rc5
  @for %i in ($(%dirlist)) do @if exist %i\*.obj @del %i\*.obj 
  @for %i in ($(%dirlist)) do @if exist %i\*.bak @del %i\*.bak 
  @for %i in ($(%dirlist)) do @if exist %i\*.err @del %i\*.err 
  @if not exist .\$(LNKbasename)*.* @%quit
  @for %i in (.\$(LNKbasename)*.*) do @del %i

zip :
  @if $(%ZIPPER).==.  @echo Error(E02): ZIPPER is not defined
  @if $(%ZIPPER).==.  @%abort
  @if $(%ZIPFILE).==. @set ZIPFILE=$(LNKbasename)-$(%VERMINOR)-$(OSNAME)-x86-cli
  
  @if exist $(%ZIPFILE).zip @del $(%ZIPFILE).zip >nul:
  @%write con 
  @echo Generating $(%ZIPFILE).zip...
  @$(%ZIPPER) $(%ZIPOPTS) $(%ZIPFILE).zip $(%BINNAME) $(%DOCFILES)
  @if $(%PORTER).==. @set PORTER= 
  @echo Generating $(%ZIPFILE).readme... 
                        @echo Uploaded...>$(%ZIPFILE).readme 
  @%append $(%ZIPFILE).readme 
  @%append $(%ZIPFILE).readme     $(%ZIPFILE).zip 
  @%append $(%ZIPFILE).readme     $(%ZIPFILE).readme (this file) 
  @%append $(%ZIPFILE).readme
  @%append $(%ZIPFILE).readme $(%PORTER)
  @%append $(%ZIPFILE).readme

#-----------------------------------------------------------------------

declare_for_desmmx : 
  @set LINKOBJS = $(%desmmx_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%desmmx_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%desmmx_SYMALIAS) $(%SYMALIAS) 
  
declare_for_rc5mmx :  
  @set LINKOBJS = $(%rc5mmx_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%rc5mmx_DEFALL) $(%DEFALL) 

declare_for_multithread : 
  @set LINKOBJS = $(%mt_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%mt_DEFALL) $(%DEFALL) 

#-----------------------------------------------------------------------

output\rg-486.obj : rc5\rg-486.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-6x86.obj : rc5\rg-6x86.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rc5p5brf.obj : rc5\rc5p5brf.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-p5.obj : rc5\rg-p5.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-p6.obj : rc5\rg-p6.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-k5.obj : rc5\rg-k5.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-k6.obj : rc5\rg-k6.asm $(%dependall)
  @if $(%TASMEXE).==. *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @if not $(%TASMEXE).==. $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rc5mmx.obj : rc5\nasm\rc5mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\desm3mmx.o : .symbolic #des\desm3mmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bits\desm3mmx.o
  @if not exist $(%x) @%write con File not found: $(%x) 
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  #wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\sboxmmx.o : .symbolic #des\sboxmmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bits\sboxmmx.o 
  @if not exist $(%x) @%write con File not found: $(%x) 
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  #wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\deseval-meggs3-mmx.obj : des\deseval-meggs3-mmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bitslice\$[&.obj
  @if not exist $(%x) @%write con File not found: $(%x) 
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\sboxes-mmx.obj : des\sboxes-mmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bitslice\$[&.obj
  @if not exist $(%x) @%write con File not found: $(%x) 
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\des-slice-meggs.obj : des\des-slice-meggs.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

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

output\cliconfig.obj : common\cliconfig.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pollsys.obj : common\pollsys.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probman.obj : common\probman.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probfill.obj : common\probfill.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\guistuff.obj : common\guistuff.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\bench.obj : common\bench.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clirun.obj : common\clirun.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\setprio.obj : common\setprio.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\console.obj : common\console.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cpucheck.obj : common\cpucheck.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\client.obj : common\client.cpp  $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cmdline.obj : common\cmdline.cpp  $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffupd.obj : common\buffupd.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selcore.obj : common\selcore.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selftest.obj : common\selftest.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\convdes.obj : common\convdes.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clirate.obj : common\clirate.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clistime.obj : common\clistime.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pathwork.obj : common\pathwork.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\autobuff.obj : common\autobuff.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\network.obj : common\network.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\netinit.obj : common\netinit.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffwork.obj : common\buffwork.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: 
  @set isused=1

output\cliident.obj : common\cliident.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\threadcd.obj : common\threadcd.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\triggers.obj : common\triggers.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\modereq.obj : common\modereq.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\logstuff.obj : common\logstuff.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\des-x86.obj : des\des-x86.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\netware.obj : platforms\netware\netware.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\hbyname.obj : platforms\netware\hbyname.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clidos.obj : platforms\dos\clidos.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32svc.obj : platforms\win32cli\w32svc.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32cons.obj : platforms\win32cli\w32cons.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32pre.obj : platforms\win32cli\w32pre.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32sock.obj : platforms\win32cli\w32sock.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\dod.obj : platforms\os2cli\dod.cpp $(%dependall) .autodepend
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\clearscr.obj : platforms\dos\clearscr.asm $(%dependall)
  *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\lurk.obj : common\lurk.cpp $(%dependall) .autodepend
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
  @for %i in ($(%WLINKOPS))  do @%append $(LNKbasename).lnk Op %i
  @for %i in ($(%SYMALIAS))  do @%append $(LNKbasename).lnk Alias %i
  @set isused=
  @for %i in ($(%FILEVER))  do @set isused=1
  @if not $(%isused).==. @%append $(LNKbasename).lnk Op Version=$(%FILEVER)
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

postlink:
  @if $(%OSNAME).==netware.  @\develop\sdkcdall\nlmdump\nlm_dos.exe *$(LNKbasename).nlm /b:$(LNKbasename).map 
  #@if $(%OSNAME).==netware. @\develop\sdkcd13\nwsdk\tools\nlmpackx $(LNKbasename).nlm $(LNKbasename).nlx
  #@if $(%OSNAME).==netware. @del $(LNKbasename).nlm
  #@if $(%OSNAME).==netware. @ren $(LNKbasename).nlx $(LNKbasename).nlm
  @if $(%OSNAME).==dos.      @\develop\pmodew\pmwlite.exe /C4 /S\develop\pmodew\pmodew.exe $(%BINNAME)
  @if $(%OSNAME).==dos.      @\develop\pmodew\pmwsetup.exe /b0 /q $(%BINNAME)
  @if $(%OSNAME).==win16.    copy $(%BINNAME) *.rex #>nul:
  @if $(%OSNAME).==win16.    @wrc -r -i=. -fo=.\$(LNKbasename).res -bt=windows -30 platforms\win32cli\w32cons.rc
  @if $(%OSNAME).==win16.    @wbind $(LNKbasename) -R $(LNKbasename).res
  @if $(%OSNAME).==win32.    @wrc /bt=nt platforms\win32cli\w32cons.rc $(%BINNAME)

#---------------------- platform specific settings come here ----------

dos: .symbolic                                       # DOS/PMODE
     @set OSNAME    = dos
     @set AFLAGS    = /5s /fp3 /bt=netware /ms # no such thing as /bt=dos4g
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\dos 
     @set WLINKOPS  = dosseg eliminate map # stub=\develop\pmodew\pmodew.exe
                                           #stub=platform/dos/d4GwStUb.CoM 
     @set LFLAGS    = symtrace printf # symtrace whack16 
     @set FORMAT    = os2 le
     @set CFLAGS    = /zp8 /wx /we /wcd=604 /wcd=594 /6s /fp3 /fpc /zm /ei /mf /bt=dos /d__MSDOS__ /I$(%watcom)\h #/Iplatforms\dos\libtcp 
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj output\clidos.obj
     @set LIBFILES  = #platforms\dos\libtcp\libtcp.a
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\readme.dos docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-dos-x86-cli
     @set BINNAME   = $(LNKbasename).com
     @%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

d16: .symbolic                                       # DOS/large model
     @set OSNAME    = d16
     @set CC        = wpp
     @set AFLAGS    = /3s /bt=dos /ml
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set LIBPATH   = $(%watcom)\lib286 $(%watcom)\lib286\dos 
     @set WLINKOPS  = dosseg eliminate map # stub=\develop\pmodew\pmodew.exe
                                           #stub=platform/dos/d4GwStUb.CoM 
     @set LFLAGS    = symtrace printf # symtrace whack16 
     @set FORMAT    = dos
     @set CFLAGS    = /zp1 /3 /wx /we /fpc /zc /zt1 /ei /ml /d__MSDOS__ /i$(%watcom)\h
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj output\clidos.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\readme.dos docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-dos-x86-cli
     @set BINNAME   = $(LNKbasename).com
     #@%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

os2: .symbolic                                       # OS/2
     @set OSNAME    = os2
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASMEXE   = tasm32.exe
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /zp8 /5s /fp5 /bm /mf /bt=os2 /DOS2
     @set OPT_SIZE  = /oantrlexi 
     @set OPT_SPEED = /oantrlexi 
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set IMPORTS   =
     @set PORTER    = Oscar 'ZiggyB' Chang (oscar@divideby0.com)
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-os2-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @set STACKSIZE = 32K  # 16384        #Will slow down client if it's 32k
     @set LINKOBJS  = $(%LINKOBJS) output\dod.obj output\lurk.obj
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
                      /i$(%watcom)\h;$(%watcom)\h\win /iplatforms/win32cli 
                      #/d2
                      #/zp8 /6s /fp3 /fpc /zm /ei /mf /bt=dos /d_Windows &
                      #/d_ENABLE_AUTODEPEND /d__WINDOWS_386__ &
                      #/bw (bw causes default windowing lib to be linked)
     @set OPT_SIZE  = /oaxt #/s /os 
     @set OPT_SPEED = /oaxt #/oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) &
                      output\w32cons.obj output\w32pre.obj output\w32sock.obj
                      #output\clearscr.obj output\clidos.obj &
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set FILEVER   = #Version not recognized for win16 executable
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-win16-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     #@%make declare_for_multithread
     @%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

w32: .symbolic                               # win32
     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set LFLAGS    = sys nt_win op de 'RC5DES Client for Windows' #nt
     @set CFLAGS    = /zp8 /fpd /5s /fp3 /bm /mf /bt=nt /DWIN32 &
                      /iplatforms/win32cli /i$(%watcom)\h;$(%watcom)\h\nt
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oneatx /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\lurk.obj &
                      output\w32svc.obj output\w32cons.obj output\w32pre.obj 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-win32-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @%make declare_for_multithread
     @%make declare_for_rc5mmx
     #@%make declare_for_desmmx
     @%make platform

netware : .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM! (May 24 '98)
     @set OSNAME    = netware
     @set STACKSIZE = 32K #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasm.exe
     @set WLINKOPS  = xdcdata=platforms/netware/rc5des.xdc &
                      multiload nod map #osdomain
     @set LFLAGS    = op scr 'none' op osname='NetWare NLM' #symtrace systemConsoleScreen #sys netware
     @set OPT_SIZE  = /os /s  
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /zp1 /zm /6s /fp3 /ei /ms /d__NETWARE__ /i$(inc_386) #/fpc /bt=netware /i$(%watcom)\novh #/bm
     @set LIBFILES  = nwwatemu,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 tli # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj output\hbyname.obj 
     @set EXTOBJS   = $(%EXTOBJS) platform\netware\watavoid\i8s.obj
     @set IMPORTS   = GetNestedInterruptLevel AllocateResourceTag UngetKey &
                      GetCurrentTime OutputToScreenWithPointer OutputToScreen &
                      ActivateScreen ImportPublicSymbol UnImportPublicSymbol &
                      ScheduleSleepAESProcessEvent CancelSleepAESProcessEvent &
                      RingTheBell GetFileServerMajorVersionNumber Alloc &
                      GetSuperHighResolutionTimer ConvertTicksToSeconds &
                      ConvertSecondsToTicks NWSMPThreadToMP &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\tli.imp
                      # @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = platform\netware\watavoid $(%watcom)\lib386 $(%watcom)\lib386\netware
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-netware-x86-cli
     @set ZIPPER    = c:\util\pkzip
     @set DOCFILES  = docs\readme.nw docs\rc5des.txt docs\readme.txt
     @set BINNAME   = $(LNKbasename).nlm
     @set COPYRIGHT = 'Copyright 1997-1999 distributed.net\r\n  Visit http://www.distibuted.net/ for more information'
     @set FILEVER   = 0.0   # don't tag with version #
     @set FORMAT    = Novell NLM 'RC5DES Client for NetWare' #'RC5DES v2.$(%VERMAJOR).$(%VERMINOR) Client for NetWare'

     @set %dependall=
     @%make declare_for_multithread
     @%make declare_for_rc5mmx
     #@%make declare_for_desmmx 
     @%make platform

