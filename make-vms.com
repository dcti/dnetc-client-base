$ v = f$verify(0)
$!===========================================================================
$! Procedure:  make-vms.com
$! Created:    15-Oct-2003
$! Author:     Jason Brady
$!
$! Purpose:    A utility for compiling and linking DNETC client modules on 
$!             the OpenVMS platform.  See instructions in file readme.vms.
$!
$! Modifications:
$!   10/15/2003    Jason Brady    Re-wrote David Sowder's RC5 procedure for
$!                                the current dnetc client.
$!===========================================================================
$!
$! Note:  Before executing this command procedure, modify the following 
$!        statements (application and source module directories, client
$!        executable name) as well as the compile and link statements 
$!        (modules, defines) if necessary.
$!
$   appdir     = "SYS$SYSDEVICE:[DNETC]"
$   moddir     = "SYS$SYSDEVICE:[DNETC.MODULES]"
$   execname   = "DNETC.EXE"
$!
$! Initialization
$!
$   on control_y then goto error_exit
$   on error then goto error_exit
$   on warning then continue
$!
$   display = "WRITE SYS$OUTPUT"
$   set default 'appdir
$!
$   type/page nl:
$   display "DNETC Client Module Compile and Link Procedure"
$   show time
$   display " "
$!
$! Compile Modules
$!
$   set default 'moddir
$   display "Compiling modules..."
$   cxx/standard=ms/prefix=all -
       /define=(HAVE_RC5_72_CORES,HAVE_OGR_CORES) -
       /optimize=(level=4,inline=all,unroll=50,tune=host) -
       base64.cpp,bench.cpp,buffbase.cpp,buffpriv.cpp, -
       buffupd.cpp,checkpt.cpp,clicdata.cpp,client.cpp, -
       clievent.cpp,cliident.cpp,clirun.cpp,clitime.cpp, -
       cmdline.cpp,confmenu.cpp,confopt.cpp,confrwv.cpp, -
       console.cpp,coremem.cpp, -
       core_ogr.cpp,core_r72.cpp, -
       cpucheck.cpp,disphelp.cpp,iniread.cpp,logstuff.cpp, -
       lurk.cpp,mail.cpp,memfile.cpp,modereq.cpp,netbase.cpp, -
       netconn.cpp,ogr.cpp,ogr_dat.cpp,ogr_sup.cpp,pathwork.cpp, -
       pollsys.cpp,probfill.cpp,problem.cpp,probman.cpp, -
       projdata.cpp,random.cpp,r72ansi1.cpp,r72ansi2.cpp, -
       r72ansi4.cpp,r72-ref.cpp,scram.cpp,selcore.cpp, -
       selftest.cpp,setprio.cpp,triggers.cpp, -
       util.cpp
$!
$! Link Modules
$!
$   display " "
$   display "Linking modules..."
$   cxxlink/executable='appdir''execname -
       base64.obj,bench.obj,buffbase.obj,buffpriv.obj, -
       buffupd.obj,checkpt.obj,clicdata.obj,client.obj, -
       clievent.obj,cliident.obj,clirun.obj,clitime.obj, -
       cmdline.obj,confmenu.obj,confopt.obj,confrwv.obj, -
       console.obj,coremem.obj, -
       core_ogr.obj,core_r72.obj, -
       cpucheck.obj,disphelp.obj,iniread.obj,logstuff.obj, -
       lurk.obj,mail.obj,memfile.obj,modereq.obj,netbase.obj, -
       netconn.obj,ogr.obj,ogr_dat.obj,ogr_sup.obj,pathwork.obj, -
       pollsys.obj,probfill.obj,problem.obj,probman.obj, -
       projdata.obj,random.obj,r72ansi1.obj,r72ansi2.obj, -
       r72ansi4.obj,r72-ref.obj,scram.obj,selcore.obj, -
       selftest.obj,setprio.obj,triggers.obj, -
       util.obj
$!
$! Purge old file versions
$!
$    display " "
$    display "Purging old file versions..."
$    purge *.cpp
$    purge *.obj
$!
$! Exit Routines
$!
$ success_exit:
$    exit_status = 1
$    goto exit_proc
$!
$ error_exit:
$    display " "
$    display "Procedure error occurred."
$!
$ exit_proc:
$    set default 'appdir
$    display " "
$    display "End of compile and link procedure."
$    display " "
$    show time
$    set noon
$    v = f$verify(v)
$    exit exit_status
$!
