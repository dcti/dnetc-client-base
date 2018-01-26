[![Build Status](https://travis-ci.org/dcti/dnetc-client-base.svg?branch=master)](https://travis-ci.org/dcti/dnetc-client-base)

disributed.net client (dnetc) public source framework
===================================================
This source will only allow you to compile a harness for testing and benchmarking the performance of cores.
It does not have the network or file buffer capabilities necessary to create a full client.

Building
--------
Ok, I just downloaded it, now what must I do?
If you are on a Unix box, type "./configure list". It will show a list of supported targets. If your
hardware/software isn't listed, look at how the configure script is made and write your own entry,
it's not very hard. Once configure has created a Makefile for you, just type "make" and the whole
thing will compile.

If you are unlucky enough to not have access to the wonderfulness of Unix, you will have to look for a Makefile / Project file suitable for your machine. Look in makefile.vc (MS Visual C++), makefile.wat (Watcom for DOS, NetWare, Win16, OS/2, Win32cli), smakefile (Amiga), Platforms/beos/* (BeOS), make-vms.com (VMS). 
