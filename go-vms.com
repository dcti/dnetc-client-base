$ v = f$verify(0)
$!===========================================================================
$! Procedure:  go-vms.com
$! Created:    15-Oct-2003
$! Author:     Jason Brady
$!
$! Purpose:    Procedure for running the DNETC client program on the OpenVMS
$!             platform.  See instructions in file readme.vms.
$!
$! Modifications:
$!   10/15/2003    Jason Brady    Re-wrote David Sowder's RC5 DCL wrapper
$!                                procedure for the current dnetc client.
$!===========================================================================
$!
$! Note:  Before executing this command procedure, modify the following
$!        three statements (application directory, client executable,
$!        batch queue name, batch log file name) if necessary.
$!
$   appdir     = "SYS$SYSDEVICE:[DNETC]"
$   execname   = "DNETC.EXE"
$   logname    = "DNETC.LOG"
$   queuename  = "DNETC_BATCH"
$   jobname    = "DNETC_CLIENT"
$!
$! Command procedure parameters are as follows:
$!     P1 - INTER for interactive, BATCH for batch.  Default is INTER.
$!     P2 - Command line options for the client enclosed in double quotes.
$!
$! Initialization
$!
$   on control_y then goto error_exit
$   on error then goto error_exit
$   on warning then continue
$!
$   thisproc = f$environment("procedure")
$   client :== $'appdir''execname
$   display  = "WRITE SYS$OUTPUT"
$!
$! Set directory and branch to appropriate routine
$!
$   set default 'appdir
$!
$   if "''p1'" .eqs. "" then goto inter_proc
$   if "''p1'" .eqs. "RUN" then goto run_proc
$   if "''p1'" .eqs. "BATCH" then goto batch_proc
$   if "''p1'" .eqs. "INTER" then goto inter_proc
$   display " "
$   display "Invalid option.  Must be INTER or BATCH."
$   goto success_exit
$!
$! Routine to run client in batch queue
$!
$batch_proc:
$   if f$search("''logname'") .nes. "" then purge/keep=2 'logname
$   submit/queue='queuename' 'thisproc' -
          /param=("RUN","''p2'") -
          /log='appdir''logname' -
          /name='jobname -
          /norestart
$   goto success_exit
$!
$! Routine to run the client interactively
$!
$inter_proc:
$   define sys$input sys$command
$   client 'p2
$   goto success_exit
$!
$! Routine to run the client executable inside batch job
$!
$run_proc:
$   set process/name='jobname
$   client 'p2
$   goto success_exit
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
$    set noon
$    v = f$verify(v)
$    exit exit_status
$!
