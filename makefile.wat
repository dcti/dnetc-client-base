## Compiler, linker, and lib stuff
## Makefile for use with *ALL* Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of [dos | netware | os2 | w32 | w_h | wsv ]
##                       or anything else defined at the end of this makefile
##
## $Log: makefile.wat,v $
## Revision 1.23  1998/08/10 22:36:47  cyruspatel
## Added support for triggers.cpp and buffupd.cpp
##
## Revision 1.22  1998/08/02 16:17:28  cyruspatel
## Completed support for logging.
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

## $Id: makefile.wat,v 1.23 1998/08/10 22:36:47 cyruspatel Exp $

%VERMINOR = 416       # for zip - fixit if not the same as version.h
%VERMAJOR = 7100      # for NetWare copyright: v2.$(%VERMAJOR).$(%VERMINOR)
%FILEVER  = 71.00     # for when the binary has an embedded version #

%EXTOBJS  = #extra objs (made elsewhere) but need linking here
%DEFALL   = /DBRYD /D__showids__ #/DCLIENT_BUILD_FRAC=$(%VERMINOR) 
            #defines used everywhere
%SYMALIAS = # symbols that need redefinition 
%LINKOBJS = output\p1bdespro.obj output\bdeslow.obj &
            output\des-x86.obj   output\convdes.obj   output\problem.obj  &
            output\rg-486.obj    output\rg-k5.obj     output\rg-k6.obj    &
            output\rc5p5brf.obj  output\rg-p6.obj     output\rg-6x86.obj  &
            output\cliconfig.obj output\autobuff.obj  output\buffwork.obj &
            output\mail.obj      output\client.obj    output\disphelp.obj &
            output\iniread.obj   output\network.obj   output\scram.obj    &
            output\clitime.obj   output\clicdata.obj  output\clirate.obj  &
            output\clisrate.obj  output\cpucheck.obj  output\pathwork.obj &
            output\cliident.obj  output\threadcd.obj  output\x86ident.obj &
            output\logstuff.obj  output\triggers.obj  output\buffupd.obj
            # this list can be added to in the platform specific section

            # 30 std OBJ's (+3 mmx, +2 mt) - platform specific stuff extra

%mmx_LINKOBJS = output\sboxes-mmx.obj output\deseval-meggs3-mmx.obj &
                output\des-slice-meggs.obj 
%mmx_DEFALL   = /DMMX_BITSLICER /DBIT_64 /DMEGGS /DKWAN &
                /DBITSLICER_WITH_LESS_BITS 
%mmx_SYMALIAS = whack16=_whack16 _malloc=malloc _free=free

%mt_LINKOBJS  = output\p2bdespro.obj output\bbdeslow.obj 
%mt_DEFALL    = /DMULTITHREAD

#-----------------------------------------------------------------------

CC=wpp386
CCASM=wasm
LINK=wlink
LNKbasename = rc5des       # for 'rc564'.err 'rc564'.lnk 'rc5des'.err etc

%TASM     =                #point this to tasm in your section if you have it
%TFLAGS   = /ml /m9 /q /t  #if TASM.==. then wasm will be executed
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
%OBJDIROP = /fo=$$^@       #Puts the .err/.objs in the right directories
%ERRDIROP = /fr=$$[:       #...redefine for older versions of Watcom
%dependall= makefile.wat common/version.h  # remake everything if these change

%ZIPFILE  = # eg $(LNKbasename)-$(%VERMINOR)-dos-x86-cli or blank for auto
%DOCFILES =                #list of files in ./docs to include in the zip
%PORTER   =                # your name and email address
%ZIPPER   = zip.exe        # a zip file won't be made if not defined
%ZIPOPTS  = -u -9 -o -i -v 
                           
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
  @set dirlist = output common des rc5
  @for %i in ($(%dirlist)) do @if exist %i\*.obj @erase %i\*.obj 
  @for %i in ($(%dirlist)) do @if exist %i\*.bak @erase %i\*.bak 
  @for %i in ($(%dirlist)) do @if exist %i\*.err @erase %i\*.err 
  @if not exist .\$(LNKbasename)*.* @%quit
  @for %i in (.\$(LNKbasename)*.*) do @erase %i

internal_zip :
  @if $(%ZIPPER).==.  @echo Error(E02): ZIPPER is not defined
  @if $(%ZIPPER).==.  @%abort
  @if $(%ZIPFILE).==. @set ZIPFILE=$(LNKbasename)-$(%VERMINOR)-$(OSNAME)-x86-cli
  @if exist $(%ZIPFILE).readme @erase $(%ZIPFILE).readme 
  @if exist $(%ZIPFILE).zip @erase $(%ZIPFILE).zip
  @%write con: 
  @echo Generating $(%ZIPFILE).zip...
  @$(%ZIPPER) $(%ZIPOPTS) $(%ZIPFILE).zip $(%BINNAME) $(%DOCFILES) >nul:
  @if $(%PORTER).==. @set PORTER=($$(%PORTER) is still undefined)
  @echo Generating $(%ZIPFILE).readme... 
  @%append $(%ZIPFILE).readme
  @%append $(%ZIPFILE).readme Two files uploaded:
  @%append $(%ZIPFILE).readme     $(%ZIPFILE).zip 
  @%append $(%ZIPFILE).readme     $(%ZIPFILE).readme (this file) 
  @%append $(%ZIPFILE).readme
  @%append $(%ZIPFILE).readme $(%PORTER)
  @%append $(%ZIPFILE).readme

zip :
  @if $(%ZIPPER).==.  @echo ZIPPER is not defined. zipfile will not be made.
  @if not $(%ZIPPER).==. @%make internal_zip

#-----------------------------------------------------------------------

declare_for_mmx : 
  @set LINKOBJS = $(%mmx_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%mmx_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%mmx_SYMALIAS) $(%SYMALIAS) 

declare_for_multithread : 
  @set LINKOBJS = $(%mt_LINKOBJS) $(%LINKOBJS)
  @set DEFALL   = $(%mt_DEFALL) $(%DEFALL) 

#-----------------------------------------------------------------------

output\cliconfig.obj : common\cliconfig.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\cpucheck.obj : common\cpucheck.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\logstuff.obj : common\logstuff.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\client.obj : common\client.cpp  $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\convdes.obj : common\convdes.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clirate.obj : common\clirate.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clistime.obj : common\clistime.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\pathwork.obj : common\pathwork.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\autobuff.obj : common\autobuff.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\network.obj : common\network.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\buffupd.obj : common\buffupd.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\triggers.obj : common\triggers.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\buffwork.obj : common\buffwork.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: 
  @set isused=1

output\cliident.obj : common\cliident.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\threadcd.obj : common\threadcd.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\rg-486.obj : rc5\rg-486.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-6x86.obj : rc5\rg-6x86.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rc5p5brf.obj : rc5\rc5p5brf.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-p5.obj : rc5\rg-p5.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-p6.obj : rc5\rg-p6.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-k5.obj : rc5\rg-k5.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\rg-k6.obj : rc5\rg-k6.asm $(%dependall)
  @if $(%TASM).==. *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @if not $(%TASM).==. $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\deseval-meggs3-mmx.obj : des\deseval-meggs3-mmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bitslice\$[&.obj
  @if not exist $(%x) @echo $(%x) not found
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\sboxes-mmx.obj : des\sboxes-mmx.cpp $(%dependall) .autodepend
  @set x=des\mmx-bitslice\$[&.obj
  @if not exist $(%x) @echo $(%x) not found
  @if not exist $(%x) @%quit
  @copy $(%x) $^@ >nul: 
  wtouch $^@
  @echo Updated $^@ from $(%x)
  @set isused=1

output\des-slice-meggs.obj : des\des-slice-meggs.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\p1bdespro.obj : des\brydmasm\p1bdespro.asm $(%dependall)
  @if "$(%TASM)"=="" @set x=$[*.obj
  @if "$(%TASM)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASM)"=="" @if not exist $(%x) @%quit
  @if "$(%TASM)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASM)"=="" wtouch $^@
  @if "$(%TASM)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASM)"=="" $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\p2bdespro.obj : des\brydmasm\p2bdespro.asm $(%dependall)
  @if "$(%TASM)"=="" @set x=$[*.obj
  @if "$(%TASM)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASM)"=="" @if not exist $(%x) @%quit
  @if "$(%TASM)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASM)"=="" wtouch $^@
  @if "$(%TASM)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASM)"=="" $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bdeslow.obj : des\brydmasm\bdeslow.asm $(%dependall)
  @if "$(%TASM)"=="" @set x=$[*.obj
  @if "$(%TASM)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASM)"=="" @if not exist $(%x) @%quit
  @if "$(%TASM)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASM)"=="" wtouch $^@
  @if "$(%TASM)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASM)"=="" $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bbdeslow.obj : des\brydmasm\bbdeslow.asm $(%dependall)
  @if "$(%TASM)"=="" @set x=$[*.obj
  @if "$(%TASM)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASM)"=="" @if not exist $(%x) @%quit
  @if "$(%TASM)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASM)"=="" wtouch $^@
  @if "$(%TASM)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASM)"=="" $(%TASM) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\des-x86.obj : des\des-x86.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\x86ident.obj : platforms\x86ident.asm $(%dependall)
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\netware.obj : platforms\netware\netware.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\hbyname.obj : platforms\netware\hbyname.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\chklocks.obj : platforms\dos\chklocks.asm $(%dependall)
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clearscr.obj : platforms\dos\clearscr.asm $(%dependall)
  *$(CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[:
  @set isused=1

output\clidos.obj : platforms\dos\clidos.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

output\dod.obj : platforms\os2cli\dod.cpp $(%dependall) .autodepend
  *$(CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) $(%OBJDIROP) /i$[: /icommon
  @set isused=1

#-----------------------------------------------------------------------

platform: .symbolic
  @set CFLAGS    = $(%CFLAGS) /zq             ## compile quietly
  @set AFLAGS    = $(%AFLAGS) /q              ## assemble quietly
  @set CFLAGS    = $(%CFLAGS) $(%DEFALL)      ## tack on global defines
  @set isused=0
  @if not exist $(%BINNAME) @set isused=1
  @for %i in ($(%LINKOBJS)) do @%make %i
  @if $(%isused).==0. @%write con: All targets are up to date
  @if $(%isused).==0. @%quit
  @%make dolink
  @%make postlink
  @%make zip

postlink:
  @if $(%OSNAME).==netware. @\develop\sdkcdall\nlmdump\nlm_dos.exe *$(LNKbasename).nlm /b:$(LNKbasename).map 
  #@if $(%OSNAME).==netware. @\develop\sdkcd13\nwsdk\tools\nlmpackx $(LNKbasename).nlm $(LNKbasename).nlx
  #@if $(%OSNAME).==netware. @del $(LNKbasename).nlm
  #@if $(%OSNAME).==netware. @ren $(LNKbasename).nlx $(LNKbasename).nlm
  @if $(%OSNAME).==dos. @\develop\pmodew\pmwlite.exe /C4 /S\develop\pmodew\pmodew.exe $(%BINNAME)
  @if $(%OSNAME).==dos. @\develop\pmodew\pmwsetup.exe /b0 $(%BINNAME) >nul:
  @if $(%OSNAME).==w16. @copy $(LNKbasename).exe @erase $(LNKbasename).rex >nul:
  @if $(%OSNAME).==w16. @wbind rc5des -n

dolink : .symbolic
  @if exist  $(%BINNAME) @del $(%BINNAME)
  @if exist  $(LNKbasename).lnk @del $(LNKbasename).lnk
  @%append   $(LNKbasename).lnk Name $(%BINNAME)
  @for %i in ($(%STACKSIZE)) do @%append $(LNKbasename).lnk Op Stack=%i
  @for %i in ($(%LINKOBJS))   do @%append $(LNKbasename).lnk File %i
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
  *$(LINK) $(%LFLAGS) @$(LNKbasename).lnk > $(LNKbasename).err
  @if exist $(%BINNAME) @del $(LNKbasename).err
  @if exist $(LNKbasename).err @type $(LNKbasename).err


#---------------------- platform specific settings come here ----------

dos: .symbolic                                       # DOS/DOS4GW
     @set OSNAME    = dos
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set TASM      = \develop\tasm32\tasm32.exe
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\dos 
     @set WLINKOPS  = dosseg eliminate map # stub=\develop\pmodew\pmodew.exe
                                           #stub=platform/dos/d4GwStUb.CoM 
     @set LFLAGS    = # symtrace whack16 
     @set FORMAT    = os2 le
     @set CFLAGS    = /zp8 /wx /we /6s /fp3 /fpc /zm /ei /mf /bt=dos /dDOS4G /DNONETWORK /I$(%watcom)\h
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj output\clidos.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-dos-x86-cli
     @set BINNAME   = $(LNKbasename).com
     #@%make declare_for_mmx
     @%make platform

w16: .symbolic                                       # Windows/16
     @set OSNAME    = win16
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set TASM      = \develop\tasm32\tasm32.exe
     @set LFLAGS    = sys win386 
     @set CFLAGS    = /zp8 /wx /we /6s /fp3 /fpc /zm /ei /mf /bw /bt=windows /dDOS4G /DNONETWORK /I$(%watcom)\h
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\clearscr.obj output\clidos.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set VERSION   =
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-win16-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @%make platform

os2: .symbolic                                       # OS/2
     @set OSNAME    = os2
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASM      = tasm32.exe
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
     @set LINKOBJS  = $(%LINKOBJS) output\dod.obj
     @set OBJDIROP  = /fo=output\
     @set ERRDIROP  =                      # no /fr= option for Watcom 10.0
     @%make declare_for_multithread
     @%make declare_for_mmx
     @%make platform

w32: .symbolic                               # win32
     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASM      = tasm32.exe
     @set LFLAGS    = sys nt
     @set CFLAGS    = /zp8 /fpd /5s /fp5 /bm /mf /bt=nt /DWIN32
     @set OPT_SIZE  = /oantrlexih 
     @set OPT_SPEED = /oantrlexih /oi+ 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set PORTER    = #your name here
     @set DOCFILES  = docs\rc5des.txt docs\readme.txt
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-win32-x86-cli
     @set BINNAME   = $(LNKbasename).exe
     @%make declare_for_multithread
     @%make declare_for_mmx
     @%make platform

netware : .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM! (May 24 '98)
     @set OSNAME    = netware
     @set STACKSIZE = 32K #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set TASM      = \develop\tasm32\tasm32.exe
     @set WLINKOPS  = xdcdata=platforms/netware/rc5des.xdc multiload nod map osdomain
     @set LFLAGS    = op scr 'none' op osname='NetWare NLM' #symtrace systemConsoleScreen #sys netware
     @set OPT_SIZE  = /os /s  
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /zp1 /zm /6s /fp3 /ei /ms /d__NETWARE__ /i$(inc_386) #/fpc /bt=netware /i$(%watcom)\novh #/bm
     @set LIBFILES  = nwwatemu,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj output\hbyname.obj 
     @set EXTOBJS   = $(%EXTOBJS) platform\netware\watavoid\i8s.obj
     @set IMPORTS   = GetNestedInterruptLevel AllocateResourceTag UngetKey &
                      GetCurrentTime OutputToScreenWithPointer OutputToScreen &
                      ActivateScreen ImportPublicSymbol UnImportPublicSymbol &
                      ScheduleSleepAESProcessEvent CancelSleepAESProcessEvent &
                      RingTheBell GetFileServerMajorVersionNumber Alloc &
                      @$(%watcom)\novi\clib.imp # @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = platform\netware\watavoid $(%watcom)\lib386 $(%watcom)\lib386\netware
     @set PORTER    = Cyrus 'cyp' Patel (cyp@fb14.uni-mainz.de)
     @set ZIPFILE   = $(LNKbasename)-$(%VERMINOR)-netware-x86-cli
     @set ZIPPER    = c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\readme.nw docs\rc5des.txt docs\readme.txt
     @set BINNAME   = $(LNKbasename).nlm
     @set COPYRIGHT = 'Copyright 1997-1998 distributed.net\r\n  Visit http://www.distibuted.net/ for more information'
     @set FILEVER   = 0.0   # don't tag with version #
     @set FORMAT    = Novell NLM 'RC5DES Client for NetWare' #'RC5DES v2.$(%VERMAJOR).$(%VERMINOR) Client for NetWare'

     @set %dependall=
     @%make declare_for_multithread
     #@%make declare_for_mmx 
     @%make platform

