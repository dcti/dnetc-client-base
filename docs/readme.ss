This is a simple screen saver adaption of the standard Win32 GUI client.
It should run on Win95 and WinNT.  Running the normal clients are
preferred over the screen saver version since they continue working
in idle cycles even while the machine is actively in use.  However,
this screen saver version might be necessary in environments where
issues of utilizing *any* cpu time while a user is actively trying to
use the machine is a problem.

Installation involves simply copying the .scr file into one of the
following locations:

on Win95:  your \WINDOWS\ or \WINDOWS\SYSTEM\ directory
on WinNT:  your \WINNT\ or \WINNT\SYSTEM32\ directory

Once you have copied it, go to the Display properties in the Control
Panel and select the RC5 screen saver as the active screen saver and
click on the Configure button there.  Configuration should be mostly
identical to the regular GUI client.

Notice that the RC5 screen saver needs its INI file to be located in
the same directory as its SCR file itself.  Additionally, the buffer
files are by default created in that directory as well.

On WindowsNT, this means that you should be sure that you put the client
in a location where people using the screen saver will have file write
access to the buffer files.  You can additionally set the [options]/in=
and [options]/out= items in the ini file to point to the full pathname
and filename of the buffer files it should use.

Additionally, when the screen saver is interrupted, it needs to wait for
the next save point to occur before it can shut down.  If you have a slow
machine, this may be a couple of seconds.  If you wish to reduce this
delay, you can edit the ini file and change the [options]/timeslice=
value to a number lower than the default value of "65536".  However, note
that smaller values will slightly degrade overall client efficiency.

The text strings displayed by the screen saver were gathered from the
"slogan" page on the RC5 home page and are hard coded into the screen
saver.  They are not currently customizable, though I might provide means
for adjusting them in the future.



Jeff Lawson (BovineOne)
bovine@distributed.net / jlawson@hmc.edu

