## Compiler, linker, and lib stuff
## Makefile for use with all Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of 
##               [dos | netware | os2 | w32 | w16]
##               or anything else with a section at the end of this file
##
## $Id: makefile.wat,v 1.27.2.9 2000/03/10 03:10:39 jlawson Exp $

BASENAME = dnetc

%EXTOBJS  = #extra objs (made elsewhere) but need linking here
%DEFALL   = /DDYN_TIMESLICE /D__showids__ /IOGR
            #defines used everywhere
%SYMALIAS = # symbols that need redefinition 
%COREOBJS = output\rg486.obj     output\rc5-rgk5.obj  output\rg6x86.obj   &
            output\rc5-rgk6.obj  output\brfp5.obj     output\rc5-rgp6.obj
%LINKOBJS = output\problem.obj  &
            output\confrwv.obj   output\autobuff.obj  output\buffbase.obj &
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
            output\modereq.obj   output\confmenu.obj  output\confopt.obj  &
            output\util.obj      output\base64.obj    output\random.obj   &
            output\netres.obj    output\buffpriv.obj
            # this list can be added to in the platform specific section
            # 49 std OBJ's (+3 desmmx, +1 rc5mmx, +2 mt, +x plat specific)

%PRELINKDEPS = # dependancies that need 'compiling' but not linking (eg .RC's)
%POSTLINKTGTS= # targets to make after linking (bind etc)

#---
%cscstd_LINKOBJS = output\csc-1k-i.obj output\csc-1k.obj &
                   output\csc-6b-i.obj output\csc-6b.obj &
                   output\convcsc.obj output\csc-common.obj &
                   output\csc-mmx.obj
%cscstd_DEFALL   = -DHAVE_CSC_CORES -Icsc -DMMX_CSC
%cscstd_SYMALIAS = 
#                  csc_unit_func_1k=_csc_unit_func_1k
#                  csc_unit_func_6b_i=_csc_unit_func_6b_i &
#                  csc_unit_func_1k=_csc_unit_func_1k &
#                  csc_unit_func_6b=_csc_unit_func_6b
#                  csc_unit_func_1k_i=_csc_unit_func_1k_i &                   
#                  
#---
%ogrstd_LINKOBJS = output\ogr.obj output\choosedat.obj output\crc32.obj 
%ogrstd_DEFALL   = -DHAVE_OGR_CORES -Iogr
%ogrstd_SYMALIAS = #
#---
%desstd_LINKOBJS = output\des-x86.obj output\convdes.obj &
                   output\p1bdespro.obj output\bdeslow.obj
%desstd_DEFALL   = /DBRYD -DHAVE_DES_CORES
%desstd_SYMALIAS = #
#---
%rc5std_LINKOBJS = output\rg486.obj output\rc5-rgk5.obj output\rg6x86.obj &
                   output\rc5-rgk6.obj output\brfp5.obj output\rc5-rgp6.obj
%rc5std_DEFALL   = #
%rc5std_SYMALIAS = #
#---
%rc5smc_LINKOBJS = output\rc5-486-smc-rg.o
%rc5smc_DEFALL   = #
%rc5smc_SYMALIAS = #
#---
%rc5mmx_LINKOBJS = output\rc5mmx.obj 
%rc5mmx_DEFALL   = /DMMX_RC5 
%rc5mmx_SYMALIAS = #
#---
%rc5mmxamd_LINKOBJS = output\rc5mmx-k6-2.obj
%rc5mmxamd_DEFALL   = /DMMX_RC5_AMD
%rc5mmxamd_SYMALIAS = #
#---
%desmmx_LINKOBJS = output\des-slice-meggs.obj output\deseval-mmx.obj
%desmmx_DEFALL   = /DMEGGS /DMMX_BITSLICER #/DBITSLICER_WITH_LESS_BITS
%desmmx_SYMALIAS = #
#---
%des_mt_LINKOBJS = output\p2bdespro.obj output\bbdeslow.obj &
                   output\des-slice.obj output\deseval.obj output\sboxes-kwan4.obj 
%des_mt_DEFALL   = /DKWAN 
%des_mt_SYMALIAS = #
#---

#-----------------------------------------------------------------------

%CC=wpp386
%CCASM=wasm
%LINK=wlink #\develop\watcom\binnt\wlink.exe

%NASMEXE  = nasm           #point this to nasm (don't call the envvar 'NASM'!)
%NASMFLAGS= -f win32 -s #-f obj -D__OMF__ -DOS2 -s
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
.ERASE

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
  @if not exist .\$(BASENAME)*.* @%quit
  @for %i in (.\$(BASENAME)*.*) do @del %i

zip: .symbolic  
  #that $(%ZIPPER) is not "" should already have been done
  @if $(%ZIPPER).==.  @echo Error(E02): ZIPPER is not defined
  @if $(%ZIPPER).==.  @%quit
  @if $(%ZIPFILE).==. @set ZIPFILE=$(BASENAME)-$(%OSNAME)-x86-cli
  
  @if exist $(%ZIPFILE).zip @del $(%ZIPFILE).zip >nul:
  @%write con 
  @echo Generating $(%ZIPFILE).zip...
  @$(%ZIPPER) $(%ZIPOPTS) $(%ZIPFILE).zip $(%BINNAME) $(%DOCFILES)

debug : .symbolic
  @set DEBUG=1

#-----------------------------------------------------------------------

declare_for_csc : .symbolic
  @set COREOBJS = $(%cscstd_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%cscstd_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%cscstd_SYMALIAS) $(%SYMALIAS) 

declare_for_ogr : .symbolic
  @set COREOBJS = $(%ogrstd_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%ogrstd_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%ogrstd_SYMALIAS) $(%SYMALIAS) 

declare_for_des : .symbolic
  @set COREOBJS = $(%desstd_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%desstd_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%desstd_SYMALIAS) $(%SYMALIAS) 

declare_for_desmt : .symbolic
  @set COREOBJS = $(%des_mt_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%des_mt_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%des_mt_SYMALIAS) $(%SYMALIAS) 

declare_for_desmmx : .symbolic
  @set COREOBJS = $(%desmmx_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%desmmx_DEFALL) $(%DEFALL) 
  @set SYMALIAS = $(%desmmx_SYMALIAS) $(%SYMALIAS) 

declare_for_rc5mmx : .symbolic
  @set COREOBJS = $(%rc5mmx_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%rc5mmx_DEFALL) $(%DEFALL) 

declare_for_rc5mmxamd : .symbolic
  @set COREOBJS = $(%rc5mmxamd_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%rc5mmxamd_DEFALL) $(%DEFALL) 

declare_for_rc5smc : .symbolic
  @set COREOBJS = $(%COREOBJS) $(%rc5smc_LINKOBJS) 
  @set DEFALL   = $(%rc5smc_DEFALL) $(%DEFALL) 

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

output\rc5mmx-k6-2.obj : rc5\x86\nasm\rc5mmx-k6-2.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

# ----------------------------------------------------------------

output\x86ident.obj : platforms\x86ident.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  #*$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
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

output\util.obj : common\util.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\base64.obj : common\base64.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\random.obj : common\random.cpp $(%dependall) .AUTODEPEND
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

output\netres.obj : common\netres.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\netinit.obj : common\netinit.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\lurk.obj : common\lurk.cpp $(%dependall) .AUTODEPEND
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

output\buffbase.obj : common\buffbase.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: 
  @set isused=1

output\buffpriv.obj : common\buffpriv.cpp $(%dependall) .AUTODEPEND
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

# ----------------------------------------------------------------

output\des-slice-meggs.obj : des\des-slice-meggs.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\deseval-mmx.obj : des\mmx-bitslice\deseval-mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

#output\des-mmx.obj : des\mmx-bitslice\des-mmx.asm $(%dependall) 
#  *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
#  @set isused=1

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

output\des-x86.obj : des\des-x86.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\convdes.obj : common\convdes.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

# ----------------------------------------------------------------

output\convcsc.obj : csc\x86\convcsc.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\csc-mmx.obj : csc\x86\mmx\csc-mmx.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[:..\ $[@
  @set isused=1

output\csc-common.obj : csc\x86\csc-comm.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\csc-1k.obj : csc\x86\csc-1k.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -l $[*.lst -i $[: $[@ 
  @set isused=1

output\csc-1k-i.obj : csc\x86\csc-1k-i.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\csc-6b.obj : csc\x86\csc-6b.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\csc-6b-i.obj : csc\x86\csc-6b-i.asm $(%dependall) .AUTODEPEND
  @if exist $[*.obj copy $[*.obj $^@ >nul: 
  @if exist $[*.obj wtouch $^@
  @if exist $[*.obj @echo Updated $^@ from $[*.obj
  @if not exist $[*.obj $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

# ----------------------------------------------------------------

output\ogr.obj : ogr\ogr.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\choosedat.obj : ogr\choosedat.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\crc32.obj : ogr\crc32.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

output\netware.obj : platforms\netware\netware.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

output\cdoscon.obj : platforms\dos\cdoscon.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosidle.obj : platforms\dos\cdosidle.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdostime.obj : platforms\dos\cdostime.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosemu.obj : platforms\dos\cdosemu.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosinet.obj : platforms\dos\cdosinet.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdoskeyb.obj : platforms\dos\cdoskeyb.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdospmeh.obj : platforms\dos\cdospmeh.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

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

output\w32ras.obj : platforms\win32cli\w32ras.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32x86.obj : platforms\win32cli\w32x86.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32util.obj : platforms\win32cli\w32util.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32ss.obj : platforms\win32cli\w32ss.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32cuis.obj : platforms\win32cli\w32cuis.c $(%LINKOBJS) .AUTODEPEND
   @set include=$(%WATCOM)\h;$(%WATCOM)\h\nt
   @wcl386 /zl /s /3s /os /mf /l=nt /fe=$(BASENAME).com &
           /fo=$^@ /"lib $(%LIBFILES) op start=main" $[@
   @if not $(%EXECOMPRESSOR).==. @$(%EXECOMPRESSOR) $(BASENAME).com

output\w32ssb.obj : platforms\win32cli\w32ssb.cpp platforms\win32cli\w32cons.rc &
                    output\w32util.obj output\w32ss.obj
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) platforms\win32cli\w32ssb.cpp $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1
  @if $(%OSNAME).==win16. @wlink $(%LFLAGS) name $(BASENAME).scr &
                file output\w32ssb.obj,output\w32ss.obj,output\w32util.obj
  @if $(%OSNAME).==win16. @if exist $(BASENAME).rex @del $(BASENAME).rex
  @if $(%OSNAME).==win16. @ren $(BASENAME).scr $(BASENAME).rex
  @if $(%OSNAME).==win16. *@wbind $(BASENAME).rex &
                -D "SCRNSAVE : distributed.net client" &
                -R -q -30 -bt=windows -i$(%WATCOM)\h;$(%WATCOM)\h\win &
                   -fo=output\$(BASENAME).res &
                   platforms\win32cli\w32cons.rc $(BASENAME).exe
  @if $(%OSNAME).==win16. @if exist $(BASENAME).scr @del $(BASENAME).scr
  @if $(%OSNAME).==win16. @ren $(BASENAME).exe $(BASENAME).scr
  @if $(%OSNAME).==win16. @if exist $(BASENAME).rex @del $(BASENAME).rex
  @if $(%OSNAME).==win32. @wlink $(%LFLAGS) name $(BASENAME).scr &
                lib $(%LIBFILES) &
                file output\w32ss.obj,output\w32util.obj,output\w32ssb.obj
  @if $(%OSNAME).==win32. @wrc -31 -bt=nt &
                -i$(%WATCOM)\h;$(%WATCOM)\h\win -fo=output\$(BASENAME).res &
                platforms\win32cli\w32cons.rc $(BASENAME).scr
  @if not $(%EXECOMPRESSOR).==. @$(%EXECOMPRESSOR) $(BASENAME).scr

# ----------------------------------------------------------------

output\os2inst.obj : platforms\os2cli\os2inst.cpp $(%dependall) .AUTODEPEND
  *$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

#-----------------------------------------------------------------------

platform: .symbolic
  @set CFLAGS    = $(%CFLAGS) /zq #-DBETA      ## compile quietly
  @set AFLAGS    = $(%AFLAGS) /q              ## assemble quietly
  @set CFLAGS    = $(%CFLAGS) $(%DEFALL)      ## tack on global defines
  @set isused=0
  @set LINKOBJS= $(%COREOBJS) $(%LINKOBJS) 
  @if not exist $(%BINNAME) @set isused=1
  @for %i in ($(%LINKOBJS)) do @%make %i
  @for %i in ($(%PRELINKDEPS)) do @%make %i
  @if $(%isused).==0. @%write con All targets are up to date
  @if not $(%isused).==0. @%make dolink
  @for %i in ($(%POSTLINKTGTS)) do @%make %i

dolink : .symbolic
  @if exist  $(%BINNAME) @del $(%BINNAME)
  @if exist  $(BASENAME).lnk @del $(BASENAME).lnk
  @%append   $(BASENAME).lnk Name $(%BINNAME)
  @for %i in ($(%STACKSIZE)) do @%append $(BASENAME).lnk Op Stack=%i
  @for %i in ($(%LINKOBJS))  do @%append $(BASENAME).lnk File %i
  @for %i in ($(%EXTOBJS))   do @%append $(BASENAME).lnk File %i
  @for %i in ($(%LIBPATH))   do @%append $(BASENAME).lnk Libpath %i
  @for %i in ($(%MODULES))   do @%append $(BASENAME).lnk Module %i
  @for %i in ($(%IMPORTS))   do @%append $(BASENAME).lnk Import %i
  @for %i in ($(%SYMALIAS))  do @%append $(BASENAME).lnk Alias %i
  @for %i in ($(%WLINKOPS))  do @%append $(BASENAME).lnk Op %i
  @if not $(%DEBUG).==. @%append $(BASENAME).lnk debug all
  @if not $(%DEBUG).==. @%append $(BASENAME).lnk Op map
  @if not $(%DEBUG).==. @%append $(BASENAME).lnk Op verbose
  @set isused=
  @for %i in ($(%LIBFILES))  do @set isused=1
  @if not $(%isused).==. @%append $(BASENAME).lnk Library $(%LIBFILES)
  @set isused=
  @for %i in ($(%FORMAT))    do @set isused=1
  @if not $(%isused).==. @%append $(BASENAME).lnk Format $(%FORMAT)
  @set isused=
  @for %i in ($(%COPYRIGHT)) do @set isused=1
  @if not $(%isused).==. @%append $(BASENAME).lnk Op Copyright $(%COPYRIGHT)
  @set isused=
  @if exist  $(%BINNAME) @del $(%BINNAME)
  *$(%LINK) $(%LFLAGS) @$(BASENAME).lnk > $(BASENAME).err
  @if exist $(%BINNAME) @del $(BASENAME).err
  @if exist $(BASENAME).err @type $(BASENAME).err
  @if exist $(BASENAME).err @%quit
  
.ERROR
  @if $(%BINNAME).==. @%quit
  @if not exist $(%BINNAME) @%quit
  @del $(%BINNAME)
  @echo Target '$(%BINNAME)' deleted.

# =======================================================================
#---------------------- platform specific settings come here ----------

dos: .symbolic                                    # DOS-PMODE/W or DOS/4GW
     @if not $(%OSNAME).==dos4g. @set OSNAME=dos
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\dos 
     @set WLINKOPS  = map dosseg
     @if $(%OSNAME).==dos4g. @set WLINKOPS=$(%WLINKOPS) stub=platforms/dos/d4GwStUb.CoM
                                                      # stub=\develop\pmodew\pmodew.exe
     @set LFLAGS    = symtrace usleep  #symtrace printf symtrace whack16 
     @set FORMAT    = os2 le
     @set CFLAGS    = /zp8 /wx /we /6s /fp3 /fpc /zm /ei /mf &
                      /bt=dos /d__MSDOS__ /wcd=604 /wcd=594 /wcd=7 &
                      /DINIT_TIMESLICE=0x40000 /DDYN_TIMESLICE &
                      /iplatforms/dos /I$(%watcom)\h #/Iplatforms\dos\libtcp
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\cdostime.obj output\cdosidle.obj &
                      output\cdoscon.obj output\cdosemu.obj output\cdosinet.obj &
                      output\cdospmeh.obj output\cdoskeyb.obj
     @set LIBFILES  = #platforms\dos\libtcp\libtcp.a
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = #c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\readme.dos docs\$(BASENAME).txt docs\readme.txt
     @set ZIPFILE   = $(BASENAME)-dos-x86-cli
     @set BINNAME   = $(BASENAME).com
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_rc5mmx
     #@%make declare_for_rc5smc
#    @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #-------------------------
     @\develop\pmodew\pmwlite.exe /C4 /S\develop\pmodew\pmodew.exe $(%BINNAME)
     @\develop\pmodew\pmwsetup.exe /b0 /q $(%BINNAME)

dos4g: .symbolic
     @set OSNAME    = dos4g
     @%make dos

os2: .symbolic                                       # OS/2
     @set OSNAME    = os2
     @set AFLAGS    = /5s /fp5 /bt=OS2 /mf
     @set TASMEXE   = tasm32.exe
     @set NASMEXE   = nasm.exe
     @set LFLAGS    = sys os2v2
     @set CFLAGS    = /zp8 /5s /fp5 /bm /mf /bt=os2 
                      /DOS2 /DLURK /DMULTITHREAD
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oantrlexi 
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\readme.os2 docs\$(BASENAME).txt docs\readme.txt
     @set ZIPFILE   = $(BASENAME)-os2-x86-cli
     @set BINNAME   = $(BASENAME).exe
     @set STACKSIZE = 32K  # 16384        #Will slow down client if it's 32k
     @set LINKOBJS  = output\os2inst.obj  output\lurk.obj
     @set OBJDIROP  = /fo=output\
     @set ERRDIROP  =                      # no /fr= option for Watcom 10.0
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_rc5mmx
     #@%make declare_for_rc5smc
#    @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform

w16: .symbolic                                       # Windows/16
     @set OSNAME    = win16
     @set AFLAGS    = /5s /fp3 /bt=dos /mf # no such thing as /bt=dos4g
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set LFLAGS    = system win386 symtrace open #debug all op de 'SCRSAVE : distributed.net client for Windows'
     @set CFLAGS    = /3s /w4 /zW /bt=windows /d_Windows &
                      /i$(%watcom)\h;$(%watcom)\h\win /iplatforms/win32cli &
                      /DBITSLICER_WITH_LESS_BITS /DDYN_TIMESLICE 
                      #/d2
                      #/zp8 /6s /fp3 /fpc /zm /ei /mf /bt=dos /d_Windows &
                      #/d_ENABLE_AUTODEPEND /d__WINDOWS_386__ &
                      #/bw (bw causes default windowing lib to be linked)
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oaxt #/oneatx /oh /oi+ 
     @set LINKOBJS  = output\w32pre.obj output\w32ss.obj output\w32cons.obj &
                      output\w32sock.obj output\w32svc.obj output\w32x86.obj &
                      output\w32util.obj $(%LINKOBJS)
     @set PRELINKDEPS = output\w32ssb.obj
     @set POSTLINKTGTS = 
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = #c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set ZIPFILE   = $(BASENAME)-win16-x86-cli
     @set BINNAME   = $(BASENAME).exe
     @if exist $(BASENAME).rex @del $(BASENAME).rex
##   @%make declare_for_des
##   @%make declare_for_desmt
##   #@%make declare_for_desmmx
     @%make declare_for_rc5mmx
     #@%make declare_for_rc5smc
#    @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #---------------------------
     @if exist $(BASENAME).rex @del $(BASENAME).rex
     @ren $(BASENAME).exe $(BASENAME).rex
     *@wbind $(BASENAME).rex &
                -D "distributed.net client" &
                -R -q -30 -bt=windows -i$(%WATCOM)\h;$(%WATCOM)\h\win &
                   -fo=output\$(BASENAME).res &
                   platforms\win32cli\w32cons.rc $(BASENAME).exe
     @if exist $(BASENAME).rex @del $(BASENAME).rex
              
w32: .symbolic                               # win32
     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp5 /mf
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set WLINKOPS  = alignment=64 map
     @set LFLAGS    = sys nt_win op de 'distributed.net client for Windows' #nt
     @set CFLAGS    = /zp8 /s /fpd /6s /bm /fp3 /mf /bt=nt /DWIN32 /DLURK &
                      /iplatforms/win32cli /i$(%watcom)\h;$(%watcom)\h\nt &
                      /DINIT_TIMESLICE=0x40000 /DDYN_TIMESLICE
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oneatx /oh /oi+ /ei #/oneatx /oh /oi+ 
     @set LINKOBJS  = output\w32pre.obj output\w32ss.obj output\w32svc.obj &
                      output\w32cons.obj output\w32sock.obj output\w32ras.obj &
                      output\w32util.obj output\lurk.obj $(%LINKOBJS)
     @set PRELINKDEPS = output\w32ssb.obj output\w32cuis.obj
     @set POSTLINKTGTS = 
     @set LIBFILES  = user32,kernel32,advapi32,gdi32
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set ZIPPER    = #c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set ZIPFILE   = #$(BASENAME)-win32-x86-cli
     @set BINNAME   = $(BASENAME).exe
     @set EXECOMPRESSOR=\develop\upx\upxw.exe -9 --compress-resources=0
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_rc5mmx
     #@%make declare_for_rc5smc
#    @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #---------------------------------
     @wrc -31 -bt=nt &
          -i$(%WATCOM)\h;$(%WATCOM)\h\win -fo=output\$(BASENAME).res &
          platforms\win32cli\w32cons.rc $(BASENAME).exe
     @if not $(%EXECOMPRESSOR).==. @$(%EXECOMPRESSOR) $(BASENAME).exe

w32ss: .symbolic                               # win32 screen saver
     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp5 /bt=DOS4GW /mf
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set WLINKOPS  = map #alignment=16 objalign=16
     @set LFLAGS    = sys nt_win op de 'distributed.net client for Windows' #nt
     @set CFLAGS    = /zp8 /s /fpd /6s /fp3 /bm /mf /bt=nt /DWIN32 /DLURK &
                      /iplatforms/win32cli /i$(%watcom)\h;$(%watcom)\h\nt &
                      /DSSSTANDALONE
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oneatx /oh /oi+ /ei #/oneatx /oh /oi+ 
     @set LINKOBJS  = output\w32ssb.obj output\w32ss.obj output\w32util.obj
     @set COREOBJS  =
     @set LIBFILES  = user32,kernel32,advapi32,gdi32
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set ZIPPER    = #c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set ZIPFILE   = 
     @set BINSUFFIX = scr
     @set BINNAME   = $(BASENAME).scr
     @%make platform

w16ss: .symbolic                    # Windows/16 screen saver
     @set OSNAME    = win16
     @set AFLAGS    = /5s /fp3 /bt=dos /mf
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set LFLAGS    = system win386 #symtrace open #debug all
     @set CFLAGS    = /3s /w4 /zW /bt=windows /d_Windows /DSSSTANDALONE &
                      /i$(%watcom)\h;$(%watcom)\h\win;platforms/win32cli
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oaxt 
     @set LINKOBJS  = output\w32ss.obj output\w32util.obj
     @set DEFALL    =
     @set COREOBJS  =
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set ZIPPER    = #c:\util\pkzip
     @set ZIPOPTS   = -exo
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set ZIPFILE   = $(BASENAME)-win16-x86-cli
     @set BINSUFFIX = scr
     @set BINNAME   = $(BASENAME).scr
     @if exist $(BASENAME).rex @del $(BASENAME).rex
     @%make platform

netware : .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM-gunk! (May 24 '98)
     @set OSNAME    = netware
     @set STACKSIZE = 32K #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set TASMEXE   = \develop\tasm32\tasm32.exe
     @set NASMEXE   = \develop\nasm\nasmw.exe
     @set WLINKOPS  = xdcdata=platforms/netware/client.xdc &
                      version=0.0 multiload nod map #osdomain
     @set LFLAGS    = op scr 'none' op osname='NetWare NLM' symtrace spawnlp #sys netware
     @set OPT_SIZE  = /os /s  
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /zp1 /wx /we /6s /fp3 /fpc /zm /ei /ms &
                      /bt=dos /d__NETWARE__ /wcd=604 /wcd=594 /wcd=7 &
                      /DBITSLICER_WITH_LESS_BITS /bt=netware &
                      /DNO_DES_SUPPORT /DMULTITHREAD &
                      /i$(inc_386) #/fpc /bt=netware /i$(%watcom)\novh #/bm
                      #/zp1 /zm /6s /fp3 /ei /ms /d__NETWARE__ &
     @set LIBFILES  = nwwatemu,inetlib,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 tli # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj 
     #@set EXTOBJS   = $(%EXTOBJS) platforms\netware\watavoid\i8s.obj
     @set IMPORTS   = ImportPublicSymbol UnImportPublicSymbol &
                      GetCurrentTime OutputToScreen &
                      GetServerConfigurationInfo Abend &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\tli.imp
                      # @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = platforms\netware\misc platforms\netware\inet &
                      $(%watcom)\lib386 #$(%watcom)\lib386\netware
     @set ZIPFILE   = $(BASENAME)-netware-x86-cli
     @set ZIPPER    = #c:\util\pkzip
     @set DOCFILES  = docs\readme.nw docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).nlm
     @set COPYRIGHT = 'Copyright 1997-1999 distributed.net\r\n  Visit http://www.distibuted.net/ for more information'
     @set FORMAT    = Novell NLM 'distributed.net client for NetWare'
     @set %dependall=
##   #@%make declare_for_des
##   #@%make declare_for_desmt
##   #@%make declare_for_desmmx
     @%make declare_for_rc5mmx
     #@%make declare_for_rc5smc
#    @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #
     @\develop\sdkcdall\nlmdump\nlm_dos.exe *$(BASENAME).nlm /b:$(BASENAME).map 
     #@\develop\sdkcd13\nwsdk\tools\nlmpackx $(BASENAME).nlm $(BASENAME).nlx
     #@del $(BASENAME).nlm
     #@ren $(BASENAME).nlx $(BASENAME).nlm
