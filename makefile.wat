## Makefile for use with all Watcom platforms.
##
##   Run as WMAKE <platform>
##   where <platform> is one of 
##               [dos | netware | os2 | win32 | win16]
##               or anything else with a section at the end of this file
##               (adjust $(known_tgts) if you add a new section)
##
## $Id: makefile.wat,v 1.27.2.28 2001/04/16 17:58:15 cyp Exp $
##
## - This makefile *requires* nasm (http://www.web-sites.co.uk/nasm/)
## - if building a DES-capable client, then it also requires either
##   a) the Microsoft Assembler (masm) OR b) Borland Turbo Assembler (tasm32)
##   OR c) pre-assembled object files to be located in the same directory
##   as the .asm source. If using masm, the makefile has to be adjusted.
## - There are some non-critical hard coded paths used for utilitities 
##   that are not part of the standard watcom distribution (for example,
##   pmodew, as used by the DOS build, or upx as used by the win32 build)
##   Search for "\develop" to see which filenames/paths are affected.
##   Since those paths/filenames are canonical, the makefile can
##   automatically detect if those utilities are available, and not use
##   them if they are not.
## - This makefile expects the standard watcom utilities to be in 
##   the search path, and, (for many targets) requires the WATCOM= 
##   environment variable to point to the base of the watcom distribution,
##   (as is required for wlink anyway), so that the INCLUDE= and library 
##   paths can be adjusted for each target individually.
##

BASENAME  =dnetc
known_tgts=netware dos win16 win32 os2# list of known (possible) builds

%EXTOBJS  = #extra objs (made elsewhere) but need linking here
%DEFALL   = /DDYN_TIMESLICE #defines used everywhere
%SYMALIAS = # symbols that need redefinition 
%COREOBJS = # constructed at runtime
# LINKOBJS is (somewhat) sorted by coherence - speed sentitive stuff first
%LINKOBJS = output\problem.obj  &
            output\bench.obj    &
            output\clirun.obj   &
            output\pollsys.obj  &
            output\selftest.obj &
            output\probman.obj  &
            output\probfill.obj &
            output\checkpt.obj  &
            output\coremem.obj  &
            output\random.obj   &
            output\clicdata.obj &
            output\base64.obj   &
            output\netbase.obj  &
            output\netconn.obj  &
            output\mail.obj     &
            output\logstuff.obj &
            output\cpucheck.obj &
            output\selcore.obj  &
            output\x86ident.obj &
            output\util.obj     &
            output\cliident.obj &
            output\modereq.obj  &
            output\client.obj   &
            output\cmdline.obj  &
            output\iniread.obj  &
            output\confrwv.obj  &
            output\confmenu.obj &
            output\confopt.obj  &
            output\console.obj  &
            output\disphelp.obj &
            output\triggers.obj &
            output\clitime.obj  &
            output\clievent.obj &
            output\setprio.obj  &
            output\pathwork.obj &
            output\buffbase.obj
%PRIVMODS = common\buffpriv.cpp  common\buffupd.cpp   common\scram.cpp
%PRIVOBJS = output\buffpriv.obj  output\buffupd.obj   output\scram.obj
%PUBOBJS =  output\buffpub.obj
            # this list can be added to in the platform specific section
            # (+3 desmmx, +2 mt, +x plat specific)

%PRELINKDEPS = # dependancies that need 'compiling' but not linking (eg .RC's)
%POSTLINKTGTS= # targets to make after linking (bind etc)

#---
%cscstd_LINKOBJS = output\csc-1k-i.obj output\csc-1k.obj &
                   output\csc-6b-i.obj output\csc-6b.obj &
                   output\convcsc.obj output\csc-common.obj &
                   output\csc-mmx.obj
%cscstd_DEFALL   = -DHAVE_CSC_CORES -Icsc
%cscstd_SYMALIAS = 
#---
%ogrstd_LINKOBJS = output\ogr-a.obj output\ogr-b.obj output\ogr_dat.obj output\ogr_sup.obj
%ogrstd_DEFALL   = -DHAVE_OGR_CORES -Iogr
%ogrstd_SYMALIAS = #
#---
%desstd_LINKOBJS = output\des-x86.obj output\convdes.obj &
                   output\p1bdespro.obj output\bdeslow.obj
%desstd_DEFALL   = -DHAVE_DES_CORES /DBRYD 
%desstd_SYMALIAS = #
#---
%rc5std_LINKOBJS = output\rg-486.obj output\rg-k5.obj output\brf-p5.obj &
                   output\rg-k6.obj output\rg-p6.obj  output\rg-6x86.obj &
                   output\hb-k7.obj output\jp-mmx.obj output\nb-p7.obj
%rc5std_DEFALL   = /DHAVE_RC5_CORES
%rc5std_SYMALIAS = #
#---
%rc5smc_LINKOBJS = output\brf-smc.obj
%rc5smc_DEFALL   = /DSMC
#---
#%rc5mmxamd_LINKOBJS = output\rc5mmx-k6-2.obj
#%rc5mmxamd_DEFALL   = /DMMX_RC5_AMD
#%rc5mmxamd_SYMALIAS = #
#---
%desmmx_LINKOBJS = output\des-slice-meggs.obj output\deseval-mmx.obj
%desmmx_DEFALL   = /DMEGGS /DMMX_BITSLICER #/DBITSLICER_WITH_LESS_BITS
%desmmx_SYMALIAS = #
#---
%des_mt_LINKOBJS = output\p2bdespro.obj output\bbdeslow.obj &
                   output\des-slice.obj output\deseval.obj &
                   output\sboxes-kwan4.obj
%des_mt_DEFALL   = /DKWAN 
%des_mt_SYMALIAS = #
#---

#-----------------------------------------------------------------------

# -oa   Alias checking is relaxed. (assumes global variables are not 
#       indirectly referenced through pointers)
# -ob   order the blocks of code emitted such that the "expected" execution 
#       path will be straight through.
# -oc   *disable* conversion of 'call followed by ret' to 'jmp'
# -od   *disable* all optimization (generate debuggable code)
# -oe=N Certain user functions are expanded in-line. (if number of quads <=N)
# -oh   enable repeated optimizations
# -oi   all intrinsifyable functions are generated inline
# -oi+  -oi but sets inline depth to max (255)
# -ok   enables flowing of register save (from prologue) down into the 
#       function's flow graph
# -ol   enable loop optimization (including moving loop-invariant code out)
# -ol+  -ol and perform loop unrolling
# -om   generate inline code for atan,cos,fabs,log10,log,sin,sqrt,tan
# -on   replace floating point divisions with multiplications by the reciprocal
# -oo   continue compilation even if low on memory
# -op   emit code to store intermediate floating-point results into memory
# -or   enable instruction scheduling for pipelined architectures
# -os   favour small code
# -ot   favour fast code
# -ou   forces the compiler to make sure that all function labels are unique
# -ox   "/obiklmr" and "s" (no stack overflow checking) options are selected.

%CCPP     =wpp386
%CC       =wcc386
%CCASM    =wasm
%LINK     =wlink
%NASMEXE  =nasm           #point this to nasm (don't call the envvar 'NASM'!)
%NASMFLAGS=-f obj -D__OMF__ -s
%TASMEXE  =                #point this to tasm in your section if you have it
%TFLAGS   =/ml /m9 /q /t  #if TASMEXE.==. then wasm will be executed
%STACKSIZE=48K            #may be redefined in the platform specific section
%AFLAGS   =/5s /fp3 /mf   #may be defined in the platform specific section
%LFLAGS   =               #may be defined in the platform specific section
%CFLAGS   =/6s /fp3 /ei /mf #may be defined in the platform specific section
%CWARNLEV =/wx /we /wcd=604 /wcd=594
%OPT_SIZE =/s /os         #may be redefined in the platform specific section
%OPT_SPEED=/s /os /oa /oe=4096 /oi+ /ol+ #yes. this is really fastest.
%LIBPATH  =               #may be defined in the platform specific section
%DEBUG    =               #@%make debug to enable debugging
%LIBFILES =               #may be defined in the platform specific section
%MODULES  =               #may be defined in the platform specific section
%IMPORTS  =               #may be defined in the platform specific section
%BINNAME  =               #must be defined in the platform specific section
%COPYRIGHT=               #may be defined in the platform specific section
%FORMAT   =               #may be defined in the platform specific section
%WLINKOPS =map            #one word wlink OP-tions. no spaces but '=' is ok
%ERRDIROP =               #Puts the .err files in the right directories
!ifeq __VERSION__ 11      #actually greater than or equal to 11
%ERRDIROP = /fr=$$[:
!endif
%dependall=# makefile.wat common/version.h  # remake everything if these change
%ZIPFILE  =# blank for auto
%DOCFILES =               #list of files in ./docs to include in the zip
%ZIPPER   =zip.exe        # a zip file won't be made if not defined
%ZIPOPTS  =#-u -9 -o -i -v 
 
#.silent
#.nocheck
.ERASE

!ifdef __LOADDLL__
!  loaddll wcc      wccd
!  loaddll wccaxp   wccdaxp
!  loaddll wcc386   wccd386
!  loaddll wpp      wppdi86
!  loaddll wppaxp   wppdaxp
!  loaddll wpp386   wppd386
!  loaddll wlink    wlink
!  loaddll wlib     wlibd
!endif

#-----------------------------------------------------------------------

default : .symbolic
  @set autosel=#
  @for %i in ($(known_tgts)) do @if exist output\%i.out @set autosel=%i#
  @if not $(%autosel).==. @%make $(%autosel)
  @if not $(%autosel).==. @%quit
  #
  @if $(%autosel).==. @if exist output\w32cuis.obj @set autosel=win32#
  @if $(%autosel).==. @if exist output\w32*.obj @set autosel=win16#
  @if $(%autosel).==. @if exist output\netware.obj @set autosel=netware#
  @if $(%autosel).==. @if exist output\cdos*.obj @set autosel=dos#
  @if $(%autosel).==. @if exist output\os2*.obj @set autosel=os2#
  @if not $(%autosel).==. @%make $(%autosel)
  @if not $(%autosel).==. @%quit
  #
  @set possibles=#
  @if exist $(%watcom)\lib386\netware\clib3s.lib @set possibles=netware#  
  @if exist $(%watcom)\lib386\dos\clib3s.lib @set possibles=$(%possibles)dos#
  @if exist $(%watcom)\lib386\nt\clib3s.lib @set possibles=$(%possibles)win32#
  @if exist $(%watcom)\lib386\win\clib3s.lib @set possibles=$(%possibles)win16#
  @if exist $(%watcom)\lib386\os2\clib3s.lib @set possibles=$(%possibles)os2#
  @for %i in ($(known_tgts)) do @if $(%possibles).==%i. @set autosel=%i
  @set possibles=#
!ifdef __OS2__  # we can default for OS/2 since its extremely unlikely that
  @if $(%autosel).==. @set autosel=os2# 
!endif          # a make under OS/2 wants a target that won't run under OS/2
  @if not $(%autosel).==. @%make $(%autosel)
  @if not $(%autosel).==. @%quit
  #
  @%write con 
  @%write con Unable to automatically select target build.
  @%write con Please specify target manually.
  @%write con   eg: WMAKE [-f makefile] {target}
  @%write con   where {target} is one of: $(known_tgts)
  @%quit

#-----------------------------------------------------------------------

clean :  .symbolic
  @set alist = output . common
  @for %i in ($(%alist)) do @if exist %i\*.obj del %i\*.obj
  @for %i in ($(%alist)) do @if exist %i\*.res del %i\*.res
  #@for %i in ($(%alist)) do @if exist %i\*.bak del %i\*.bak
  #@for %i in ($(%alist)) do @if exist %i\*.~?? del %i\*.~??
  @for %i in ($(%alist)) do @if exist %i\*.err del %i\*.err 
  @set alist = exe com scr nlm lnk map err res
  @for %i in ($(%alist)) do @if exist $(BASENAME).%i del $(BASENAME).%i
  @for %i in ($(known_tgts)) do @if exist $(BASENAME)???-%i-x86.zip del $(BASENAME)???-%i-x86.zip
  @for %i in ($(known_tgts)) do @if exist $(BASENAME)-%i-x86.zip del $(BASENAME)-%i-x86.zip
  @%quit

zip : .symbolic  
  @if $(%ZIPFILE).==. @set ZIPFILE=$(BASENAME)-$(%OSNAME)-x86
  @set ZIPPER=#
  #
  @set zipexe=#
  @set pxxx=$(%PATH:;= )
  @for %i in ($(%pxxx)) do @if exist %i\pkzip.exe @set zipexe=%i\pkzip.exe -exo
  @if not $(%zipexe).==. set ZIPPER=$(%zipexe)
  @echo gotit=$(%ZIPPER)
  @set zipexe=#
  @for %i in ($(%pxxx)) do @if exist %i\zip.exe @set zipexe=%i\zip.exe -u -9 -o -i -v
  @if not $(%zipexe).==. set ZIPPER=$(%zipexe)
  @echo gotit=$(%ZIPPER)
  @set zipexe=
  @set pxxx=
  #
  #that $(%ZIPPER) is not "" should already have been done
  #@if $(%ZIPPER).==.  @echo Error(E02): ZIPPER is not defined
  #@if $(%ZIPPER).==.  @%quit
  #
  @if exist $(%ZIPFILE).zip @del $(%ZIPFILE).zip >nul:
  @%write con 
  @if not $(%ZIPPER).==. echo Generating $(%ZIPFILE).zip...
  @if not $(%ZIPPER).==. $(%ZIPPER) $(%ZIPOPTS) $(%ZIPFILE).zip $(%BINNAME) $(%DOCFILES)

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

declare_for_rc5 : .symbolic
  @set COREOBJS = $(%rc5std_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%rc5std_DEFALL) $(%DEFALL) 

declare_for_rc5smc : .symbolic
  @set COREOBJS = $(%rc5smc_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%rc5smc_DEFALL) $(%DEFALL) 

declare_for_rc5mmxamd : .symbolic
  @set COREOBJS = $(%rc5mmxamd_LINKOBJS) $(%COREOBJS)
  @set DEFALL   = $(%rc5mmxamd_DEFALL) $(%DEFALL) 

#-----------------------------------------------------------------------

output\rg-486.obj : rc5\x86\rg-486.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rg-k5.obj : rc5\x86\rg-k5.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rg-k6.obj : rc5\x86\rg-k6.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\brf-p5.obj : rc5\x86\brf-p5.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rg-p6.obj : rc5\x86\rg-p6.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\nb-p7.obj : rc5\x86\nb-p7.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\rg-6x86.obj : rc5\x86\rg-6x86.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\hb-k7.obj : rc5\x86\hb-k7.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@
  @set isused=1

output\jp-mmx.obj : rc5\x86\jp-mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

output\brf-smc.obj : rc5\x86\brf-smc.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

# ----------------------------------------------------------------

output\x86ident.obj : plat\x86\x86ident.asm $(%dependall)
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  #*$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confrwv.obj : common\confrwv.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confopt.obj : common\confopt.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\confmenu.obj : common\confmenu.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\util.obj : common\util.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\base64.obj : common\base64.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\random.obj : common\random.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pollsys.obj : common\pollsys.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probman.obj : common\probman.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\probfill.obj : common\probfill.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\bench.obj : common\bench.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clirun.obj : common\clirun.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\checkpt.obj : common\checkpt.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\coremem.obj : common\coremem.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\setprio.obj : common\setprio.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\console.obj : common\console.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cpucheck.obj : common\cpucheck.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\client.obj : common\client.cpp  $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\cmdline.obj : common\cmdline.cpp  $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffupd.obj : common\buffupd.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selcore.obj : common\selcore.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\selftest.obj : common\selftest.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\problem.obj : common\problem.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\disphelp.obj : common\disphelp.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clitime.obj : common\clitime.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clicdata.obj : common\clicdata.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clisrate.obj : common\clisrate.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\pathwork.obj : common\pathwork.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\netconn.obj : common\netconn.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\netbase.obj : common\netbase.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\lurk.obj : common\lurk.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\iniread.obj : common\iniread.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\scram.obj : common\scram.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\mail.obj : common\mail.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffbase.obj : common\buffbase.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: 
  @set isused=1

output\buffpub.obj : common\buffpub.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\buffpriv.obj : common\buffpriv.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: 
  @set isused=1

output\cliident.obj : common\cliident.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\clievent.obj : common\clievent.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\triggers.obj : common\triggers.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\modereq.obj : common\modereq.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

output\logstuff.obj : common\logstuff.cpp $(%dependall) .AUTODEPEND
  #with /wcd=7 to suppress "&array" may not produce intended result" warning
  #would otherwise require 'va_list *x=((va_list *)(&(__va_list[0])))'
  *$(%CCPP) $(%CFLAGS) /wcd=7 $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
  @set isused=1

# ----------------------------------------------------------------

output\des-slice-meggs.obj : des\des-slice-meggs.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\deseval-mmx.obj : des\x86\deseval-mmx.asm $(%dependall) 
  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
  @set isused=1

#output\des-mmx.obj : des\x86\des-mmx.asm $(%dependall) 
#  *$(%CCASM) $(%AFLAGS) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
#  @set isused=1

output\des-slice.obj : des\des-slice.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\x86-slic.obj : des\x86-slic.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\deseval.obj : des\deseval.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-kwan4.obj : des\sboxes-kwan4.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-k.obj : des\sboxes-k.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\sboxes-kwan3.obj : des\sboxes-kwan3.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\p1bdespro.obj : des\x86\p1bdespro.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%write con File not found: $(%x) 
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\p2bdespro.obj : des\x86\p2bdespro.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bdeslow.obj : des\x86\bdeslow.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\bbdeslow.obj : des\x86\bbdeslow.asm $(%dependall)
  @if "$(%TASMEXE)"=="" @set x=$[*.obj
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @echo $(%x) not found
  @if "$(%TASMEXE)"=="" @if not exist $(%x) @%quit
  @if "$(%TASMEXE)"=="" copy $(%x) $^@ >nul: 
  @if "$(%TASMEXE)"=="" wtouch $^@
  @if "$(%TASMEXE)"=="" @echo Updated $^@ from $(%x)
  @if not "$(%TASMEXE)"=="" $(%TASMEXE) $(%TFLAGS) /i$[: $[@,$^@
  @set isused=1

output\des-x86.obj : des\x86\des-x86.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\convdes.obj : common\convdes.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[:
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

output\csc-common.obj : csc\x86\csc-common.asm $(%dependall) .AUTODEPEND
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

FASTEST_OGR=/3s /s /os /oa /oe=4096 /oi+ /ol+
#it took me 5 hours to determine this.
# #/ox /oa /oe=512 /oh /oi+ /ol+

output\ogr-a.obj : ogr\x86\ogr-a.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(FASTEST_OGR) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\ogr-b.obj : ogr\x86\ogr-b.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(FASTEST_OGR) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

#output\ogr.obj : ogr\x86\ogr.asm $(%dependall) 
#  $(%NASMEXE) $(%NASMFLAGS) -o $^@ -i $[: $[@ 
#  @set isused=1

output\ogr_dat.obj : ogr\ansi\ogr_dat.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\ogr_sup.obj : ogr\ansi\ogr_sup.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SPEED) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

output\netware.obj : plat\netware\netware.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

output\cdoscon.obj : plat\dos\cdoscon.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosidle.obj : plat\dos\cdosidle.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdostime.obj : plat\dos\cdostime.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosemu.obj : plat\dos\cdosemu.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdosinet.obj : plat\dos\cdosinet.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdoskeyb.obj : plat\dos\cdoskeyb.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\cdospmeh.obj : plat\dos\cdospmeh.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

# ----------------------------------------------------------------

output\w32svc.obj : plat\win\w32svc.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32cons.obj : plat\win\w32cons.cpp $(%dependall) .AUTODEPEND
  #WCD=716 to suppress truncation warnings
  *$(%CCPP) $(%CFLAGS) /wcd=716 $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32pre.obj : plat\win\w32pre.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32sock.obj : plat\win\w32sock.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32ras.obj : plat\win\w32ras.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32x86.obj : plat\win\w32x86.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

#output\w32pid.obj : plat\win\w32pid.cpp $(%dependall) .AUTODEPEND
#  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
#  @set isused=1

output\w32exe.obj : plat\win\w32exe.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32ini.obj : plat\win\w32ini.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32util.obj : plat\win\w32util.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32ss.obj : plat\win\w32ss.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32snapp.obj : plat\win\w32snapp.c $(%LINKOBJS) .AUTODEPEND
  #*$(%CC) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

output\w32cuis.obj $(BASENAME).com : plat\win\w32cuis.c
   @set include=$(%include);$(%WATCOM)\h;$(%WATCOM)\h\nt
   wcl386 /zl /s /3s /os /mf /l=nt /fe=$(BASENAME).com &
           /fo=$^@ /"lib $(%LIBFILES) op start=main op map" $[@
   #win32_binpack will have been validated in make platform
   @if not $(%WIN32_BINPACK).==. @-$(%WIN32_BINPACK) $(BASENAME).com

output\w32ssb.obj $(BASENAME).scr : plat\win\w32ssb.cpp &
     plat\win\w32cons.rc output\w32util.obj output\w32ss.obj &
     output\w32ini.obj output\w32exe.obj $(%LINKOBJS)
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) plat\win\w32ssb.cpp &
                                    $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1
  @if $(%OSNAME).==win16. @wlink $(%LFLAGS) name $(BASENAME).scr &
                file output\w32ssb.obj,output\w32ss.obj,output\w32util.obj &
                file output\w32ini.obj,output\w32exe.obj
  @if $(%OSNAME).==win16. @if exist $(BASENAME).rex @del $(BASENAME).rex
  @if $(%OSNAME).==win16. @ren $(BASENAME).scr $(BASENAME).rex
  @if $(%OSNAME).==win16. *@wbind $(BASENAME).rex &
                -D "SCRNSAVE : distributed.net client" &
                -R -q -30 -bt=windows -i$(%WATCOM)\h;$(%WATCOM)\h\win &
                   -fo=output\$(BASENAME).res &
                   plat\win\w32cons.rc $(BASENAME).exe
  @if $(%OSNAME).==win16. @if exist $(BASENAME).scr @del $(BASENAME).scr
  @if $(%OSNAME).==win16. @ren $(BASENAME).exe $(BASENAME).scr
  @if $(%OSNAME).==win16. @if exist $(BASENAME).rex @del $(BASENAME).rex
  @if $(%OSNAME).==win32. @wlink $(%LFLAGS) name $(BASENAME).scr &
                lib $(%LIBFILES) &
                file output\w32ssb.obj,output\w32ss.obj,output\w32util.obj &
                file output\w32ini.obj,output\w32exe.obj
  @if $(%OSNAME).==win32. @wrc -31 -bt=nt -q &
                -i$(%WATCOM)\h;$(%WATCOM)\h\win -fo=output\$(BASENAME).res &
                plat\win\w32cons.rc $(BASENAME).scr
  #win32_binpack will have been validated in make platforms
  @if not $(%WIN32_BINPACK).==. @-$(%WIN32_BINPACK) $(BASENAME).scr

# ----------------------------------------------------------------

output\os2inst.obj : plat\os2\os2inst.cpp $(%dependall) .AUTODEPEND
  *$(%CCPP) $(%CFLAGS) $(%OPT_SIZE) $[@ $(%ERRDIROP) /fo=$^@ /i$[: /icommon
  @set isused=1

#-----------------------------------------------------------------------

cleanup_critical_names : .symbolic  #cleanup/validate some critical names
  @set cleanup=#
  @for %i in ($(%OSNAME)) do @set cleanup=%i#
  @set OSNAME=$(%cleanup)#
  @if $(%OSNAME).==. echo OSNAME not defined
  @if $(%OSNAME).==. %@quit
  @set cleanup=#
  #
  #
  @set cleanup=#
  @for %i in ($(%NASMEXE)) do @set cleanup=%i#
  @set NASMEXE=$(%cleanup)
!ifdef __NT__ #make running under winnt
  @if "$(%NASMEXE)"=="nasm" @set NASMEXE=nasmw.exe
!else         #make running under os/2 or dos
  @if "$(%NASMEXE)"=="nasm" @set NASMEXE=nasm.exe  
!endif
  @set cleanup=#
  #
  #
  @for %i in ($(%WIN32_BINPACK)) do @if exist %i @set cleanup=%i#
  @if $(%cleanup).==. @set WIN32_BINPACK=#
  @set cleanup=#

#-----------------------------------------------------------------------

resolve_target_conflicts : .symbolic
   #
   # remove conflicts in two things:
   # 1) executable file conflicts 
   #              between win32 and dos builds (both have xxx.com)
   #              and between win32 and win16 builds (both have xxx.scr)
   #              by ensuring all and only the expected targets exist. 
   #              This is so that make for different platforms will have the 
   #              correct targets in the end, and not, say, a win32 build 
   #              with a .com from from a previous dos build.
   # 2) obj conflicts
   #              that the output directory contains the link set
   #              for the target in question and not something left over.
   #
   ### ++++ first resolution +++
   @set explist=#
   #** caution: ** order must be com, then exe, then scr (note leading '_')
   @if $(%OSNAME).==win32. @set explist=_com_exe_scr
   @if $(%OSNAME).==win16. @set explist=_exe_scr
   @if $(%OSNAME).==dos.   @set explist=_com
   @set dellist=#
   #** caution: ** order must be com, then exe, then scr
   #we need two lists, one for == test, and one for del
   @if not $(%explist).==. @if exist $(BASENAME).com @set dellist=$(%dellist) com
   @if not $(%explist).==. @if exist $(BASENAME).com @set cmplist=$(%cmplist)_com
   @if not $(%explist).==. @if exist $(BASENAME).exe @set dellist=$(%dellist) exe
   @if not $(%explist).==. @if exist $(BASENAME).exe @set cmplist=$(%cmplist)_exe
   @if not $(%explist).==. @if exist $(BASENAME).scr @set dellist=$(%dellist) scr
   @if not $(%explist).==. @if exist $(BASENAME).scr @set cmplist=$(%cmplist)_scr
   @if $(%explist).==$(%cmplist). @set dellist=#
   @for %i in ($(%dellist)) do @del $(BASENAME).%i
   @if not $(%dellist).==. @set dellist=$(%PRELINKDEPS)
   @for %i in ($(%dellist)) do @if exist %i @del %i
   @set explist=#
   @set dellist=#
   @set cmplist=#
   ### ++++ second resolution - check output dir is watcom (not vc/bcc etc) ++++
   @set dellist=#
   @set cmplist=output\bltwith.wat
   @if not exist $(%cmplist) @set dellist=out # fall into third resolution
   @for %i in ($(%dellist)) do @if exist output\bltwith.* @del output\bltwith.*
   @for %i in ($(%dellist)) do @if exist output\*.%i @del output\*.%i
   @%create $(%cmplist)
   @set dellist=#
   @set cmplist=#
   ### ++++ third resolution - check output dir is for target OS ++++
   @set dellist=#
   @set cmplist=output\$(%OSNAME).out
   @if not exist $(%cmplist) @set dellist=out obj res
   @for %i in ($(%dellist)) do @if exist output\*.%i @del output\*.%i
   @%create $(%cmplist)
   @set dellist=#
   @set cmplist=#

#-----------------------------------------------------------------------

#.ERROR
#  @if $(%BINNAME).==. @%quit #make hasn't done anything yet
#  @set dellist=#
#  @if not $(%BINNAME).==. @if exist $(%BINNAME) @set dellist=$(%BINNAME)#
#  @if $(%OSNAME).==win32. @if exist $(BASENAME).com @set dellist=$(%BINNAME) $(%dellist)
#  @if $(%OSNAME).==win32. @if exist $(BASENAME).scr @set dellist=$(BASENAME).com $(%dellist)
#  @if $(%OSNAME).==win16. @if exist $(BASENAME).scr @set dellist=$(BASENAME).scr $(%dellist)
#  @if not $(%dellist).==. @del $(%dellist)
#  @if not $(%dellist).==. @echo Target(s) '$(%dellist)' deleted.
#  @set dellist=#
#  @%quit

#-----------------------------------------------------------------------

platform: .symbolic
  @set CFLAGS    = $(%CFLAGS) $(%CWARNLEV)    ## add warning level
  @set CFLAGS    = $(%CFLAGS) /zq #-DBETA      ## compile quietly
  @set AFLAGS    = $(%AFLAGS) /q              ## assemble quietly
  @set CFLAGS    = $(%CFLAGS) $(%DEFALL)      ## tack on global defines

  @%make cleanup_critical_names
  @%make resolve_target_conflicts

  @set isused=0
  @for %i in ($(%PRIVMODS)) do @if not exist %i @set isused=1
  @if not $(%isused).==0. @set LINKOBJS=$(%LINKOBJS) $(%PUBOBJS)
  @if $(%isused).==0. @set LINKOBJS=$(%LINKOBJS) $(%PRIVOBJS)
  @set LINKOBJS= $(%COREOBJS) $(%LINKOBJS) 
  @set isused=0
  @if not exist $(%BINNAME) @set isused=1
  @for %i in ($(%LINKOBJS)) do @%make %i
  @for %i in ($(%PRELINKDEPS)) do @%make %i
  @if $(%isused).==0. @echo All objects are up to date
  @if $(%isused).==0. @%quit
  @if not $(%isused).==0. @%make dolink

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
  @if not $(%DEBUG).==. @%append $(BASENAME).lnk Op statics
  @if $(%DEBUG).==. @%append $(BASENAME).lnk Op quiet
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
  @for %i in ($(%EXTRATGTS)) do @%make %i

# =======================================================================
#---------------------- platform specific settings come here ----------

dos: .symbolic                                    # DOS-PMODE/W or DOS/4GW
     #one of the following must be valid or the make will stop
     @set DOS4GW_STUB=$(%watcom)\binw\wstubq.exe
     @if exist plat\dos\d4GwStUb.CoM @set DOS4GW_STUB=plat\dos\d4GwStUb.CoM
     @set PMODEW_PATH=\develop\pmodew

     #automatically select dos/4gw or pmode/w build
     @set OSNAME=
     @if exist $(%DOS4GW_STUB) @set OSNAME=dos4g
     @if exist $(%PMODEW_PATH)\pmodew.exe @set OSNAME=dos
     @if $(%OSNAME).==. @echo Unable to find either the dos4gw stub or pmode/w
     @if $(%OSNAME).==. @%quit
     @if $(%OSNAME).==dos4g. @set PMODEW_PATH=#
     @if $(%OSNAME).==dos. @set DOS4GW_STUB=#
     @set OSNAME=dos

     @set NASMFLAGS = $(%NASMFLAGS) -DUSE_DPMI
     @set AFLAGS    = /5s /fp3 /bt=dos /mf
     @set TASMEXE   = tasm32.exe
     @set LIBPATH   = $(%watcom)\lib386 $(%watcom)\lib386\dos 
     @set WLINKOPS  = map dosseg
     @if not $(%DOS4GW_STUB).==. @set WLINKOPS=$(%WLINKOPS) stub=$(%DOS4GW_STUB)
     @set LFLAGS    = symtrace usleep  #symtrace printf symtrace whack16 
     @set FORMAT    = os2 le
     @set CWARNLEV  = $(%CWARNLEV)
     @set CFLAGS    = /zp4 /6s /fp3 /fpc /zm /ei /mf &
                      /bt=dos /d__MSDOS__ &
                      /DDYN_TIMESLICE &
                      /DUSE_DPMI &
                      /Iplat\dos /I$(%watcom)\h #;plat\dos\libtcp
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oneatx /oh /oi+ 
     @set LINKOBJS  = $(%LINKOBJS) output\cdostime.obj output\cdosidle.obj &
                      output\cdoscon.obj output\cdosemu.obj output\cdosinet.obj &
                      output\cdospmeh.obj output\cdoskeyb.obj
     @set LIBFILES  = #plat\dos\libtcp\libtcp.a
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\readme.dos docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).com
     
     @%make declare_for_rc5
     @%make declare_for_rc5smc
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #-------------------------
     @if not $(%PMODEW_PATH).==. @$(%PMODEW_PATH)\pmwlite.exe /C4 /S$(%PMODEW_PATH)\pmodew.exe $(%BINNAME)
     @if not $(%PMODEW_PATH).==. @$(%PMODEW_PATH)\pmwsetup.exe /b0 /q $(%BINNAME)


os2: .symbolic                                       # OS/2
     @set OSNAME    = os2
     @set AFLAGS    = /5s /fp5 /bt=OS2 /mf
     @set TASMEXE   = 
     @set LFLAGS    = sys os2v2
     @set CWARNLEV  = $(%CWARNLEV)
     @set CFLAGS    = /zp4 /5s /fp5 /bm /mf /zm /bt=os2 /DOS2 /DLURK &
                      /iplat\os2
     @set OPT_SIZE  = /s /os
     @set OPT_SPEED = /oantrlexi 
     @set LIBFILES  = so32dll.lib,tcp32dll.lib
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\readme.os2 docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).exe
     @set STACKSIZE = 48K  # 16384        #Will slow down client if it's 32k
     @set LINKOBJS  = output\os2inst.obj output\lurk.obj $(%LINKOBJS)
     @if not $(%watcom).==. @set include=$(%include);$(%WATCOM)\h;$(%WATCOM)\os2
     #@if not $(%watcom).==. @set LIBPATH=$(%watcom)\lib386 $(%watcom)\lib386\os2
     @%make declare_for_rc5
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform

w16: .symbolic
     @%make win16

win16: .symbolic                                       # Windows/16
     @set OSNAME    = win16
     @set AFLAGS    = /5s /fp3 /bt=dos /mf
     @set TASMEXE   = tasm32.exe
     @set NASMFLAGS = $(%NASMFLAGS) -DUSE_DPMI
     @set LFLAGS    = system win386 symtrace open #debug all op de 'SCRSAVE : distributed.net client for Windows'
     @set CWARNLEV  = $(%CWARNLEV)
     @set CFLAGS    = /3s /zW /bt=windows /d_Windows /DUSE_DPMI &
                      /i$(%watcom)\h;$(%watcom)\h\win /iplat\win &
                      /DBITSLICER_WITH_LESS_BITS /DDYN_TIMESLICE 
                      #/d2
                      #/zp4 /6s /fp3 /fpc /zm /ei /mf /bt=dos /d_Windows &
                      #/d_ENABLE_AUTODEPEND /d__WINDOWS_386__ &
                      #/bw (bw causes default windowing lib to be linked)
     @set OPT_SIZE  = /s /os 
     @set OPT_SPEED = /oaxt #/oneatx /oh /oi+ 
     @set LINKOBJS  = output\w32pre.obj output\w32ss.obj output\w32cons.obj &
                      output\w32sock.obj output\w32svc.obj output\w32x86.obj &
                      output\w32util.obj output\w32exe.obj output\w32ini.obj &
                      $(%LINKOBJS)                      
     @set PRELINKDEPS = output\w32ssb.obj
     @set LIBFILES  =
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).exe
     @%make declare_for_rc5
     @%make declare_for_rc5smc
##   @%make declare_for_des
##   @%make declare_for_desmt
##   #@%make declare_for_desmmx
     @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #---------------------------
     @if exist $(BASENAME).rex @del $(BASENAME).rex
     @ren $(BASENAME).exe $(BASENAME).rex
     *@wbind $(BASENAME).rex &
                -D "distributed.net client" &
                -R -q -30 -bt=windows -i$(%WATCOM)\h;$(%WATCOM)\h\win &
                   -fo=output\$(BASENAME).res &
                   plat\win\w32cons.rc $(BASENAME).exe
     @if exist $(BASENAME).rex @del $(BASENAME).rex

w32: .symbolic
     @%make win32
              
win32: .symbolic                               # win32
     #if WIN32_BINPACK exists, then win32 targets will be compressed
     @set WIN32_BINPACK=\develop\upx\upxw.exe -qq -9 --compress-resources=0

     @set OSNAME    = win32
     @set AFLAGS    = /5s /fp6 /mf
     @set TASMEXE   = tasm32.exe
     @set NASMEXE   = nasmw.exe
     @set WLINKOPS  = alignment=64 map
     @set LFLAGS    = sys nt_win op de 'distributed.net client for Windows'
     @set CWARNLEV  = $(%CWARNLEV)
     @set CFLAGS    = /3s /zp4 /s /fpi87 /fp6 /bm /mf /zmf /zc /bt=nt /DWIN32 /DLURK &
                      /iplat\win /i$(%watcom)\h;$(%watcom)\h\nt &
                      /DDYN_TIMESLICE
     @set OPT_SPEED = $(%OPT_SPEED)
     @set OPT_SIZE  = $(%OPT_SIZE)
     @set LINKOBJS  = output\w32pre.obj output\w32ss.obj output\w32svc.obj &
                      output\w32cons.obj output\w32sock.obj output\w32ras.obj &
                      output\w32util.obj output\w32exe.obj output\w32ini.obj &
                      output\w32snapp.obj output\lurk.obj $(%LINKOBJS)
     @set PRELINKDEPS = output\w32cuis.obj output\w32ssb.obj
     @set LIBFILES  = user32,kernel32,advapi32,gdi32
     @set MODULES   =
     @set IMPORTS   =
     @set DOCFILES  = docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).exe
     @%make declare_for_rc5
     @%make declare_for_rc5smc
##   @%make declare_for_des
##   @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #---------------------------------
     @wrc -31 -bt=nt -q &
          -i$(%WATCOM)\h;$(%WATCOM)\h\win -fo=output\$(BASENAME).res &
          plat\win\w32cons.rc $(BASENAME).exe
     #win32_binpack will have been validated in make platform
     @if not $(%WIN32_BINPACK).==. @-$(%WIN32_BINPACK) $(BASENAME).exe

netware : .symbolic   # NetWare NLM unified SMP/non-SMP, !NOWATCOM-gunk! (May 24 '98)
     @set NLMDUMPEXE=\develop\sdkcdall\nlmdump\nlm_w32.exe

     @set OSNAME    = netware
     @set STACKSIZE = 48K #16384
     @set AFLAGS    = /5s /fp3 /bt=netware /ms
     @set TASMEXE   = tasm32.exe
     @set NASMEXE   = nasmw.exe
     @set WLINKOPS  = version=2.80 multiload nod map &
                      xdcdata=plat/netware/client.xdc osdomain
     @set LFLAGS    = op scr 'none' op osname='NetWare NLM' symtrace spawnlp #sys netware
     @set OPT_SIZE  = /os /s  
     @set CWARNLEV  = $(%CWARNLEV)
     @set OPT_SPEED = /oneatx /oh /oi+  
     @set CFLAGS    = /zp1 /6s /fp3 /fpc /zm /ei /ms &
                      /bt=dos /d__NETWARE__ &
                      /DBITSLICER_WITH_LESS_BITS /bt=netware &
                      /i$(inc_386) #/fpc /bt=netware /i$(%watcom)\novh #/bm
                      #/zp1 /zm /6s /fp3 /ei /ms /d__NETWARE__ &
     @set LIBFILES  = nwwatemu,inetlib,plib3s #plibmt3s,clib3s,math387s,emu387
     @set MODULES   = clib a3112 tli # tcpip netdb
     @set LINKOBJS  = $(%LINKOBJS) output\netware.obj 
     #@set EXTOBJS   = $(%EXTOBJS) plat\netware\watavoid\i8s.obj
     @set IMPORTS   = ImportPublicSymbol UnImportPublicSymbol &
                      GetCurrentTime OutputToScreen fmod &
                      GetServerConfigurationInfo Abend FEGetOpenFileInfo &
                      @$(%watcom)\novi\clib.imp @$(%watcom)\novi\tli.imp
                      # @$(%watcom)\novi\mathlib.imp
     @set LIBPATH   = plat\netware\misc plat\netware\inet &
                      $(%watcom)\lib386 #$(%watcom)\lib386\netware
     @set DOCFILES  = docs\readme.nw docs\$(BASENAME).txt docs\readme.txt
     @set BINNAME   = $(BASENAME).nlm
     @set COPYRIGHT = 'Copyright 1997-2000 Distributed Computing Technologies, Inc.\r\n  Visit http://www.distibuted.net/ for more information'
     @set FORMAT    = Novell NLM 'distributed.net client for NetWare'
     @set %dependall=
     @%make declare_for_rc5
     @%make declare_for_rc5smc
#    @%make declare_for_des
#    @%make declare_for_desmt
##   @%make declare_for_desmmx
     @%make declare_for_ogr
#    @%make declare_for_csc
     @%make platform
     #
     @if exist $(NLMDUMPEXE) @$(NLMDUMPEXE) *$(BASENAME).nlm /b:$(BASENAME).map 
     #@\develop\sdkcd13\nwsdk\tools\nlmpackx $(BASENAME).nlm $(BASENAME).nlx
     #@del $(BASENAME).nlm
     #@ren $(BASENAME).nlx $(BASENAME).nlm

