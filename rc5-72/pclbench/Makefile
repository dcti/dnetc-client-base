# $Id: Makefile,v 1.1.2.1 2003/03/03 01:45:59 andreasb Exp $

all: pclbench

BASEDIR		= ../../..
COREOBJFILES	= $(wildcard $(BASEDIR)/output/r72*.o)

pclbench: pclbench.mk
	$(MAKE) -f pclbench.mk

pclbench.mk: pclbench.mk.in $(COREOBJFILES)
	cp $@.in $@
	for core in $(COREOBJFILES) ; do \
		funcname=`nm $$core | grep rc5_72_unit_func | sed -e 's/.*rc5_72_unit_func_//;s/_$$//' | head -1` ; \
		test -n "$$funcname" && targets="$$targets pclbench-$$funcname" ; \
		test -n "$$funcname" && echo "pclbench-$$funcname: pclbench.cpp $$core" >> $@ ; \
		test -n "$$funcname" && echo "	\$$(CC) \$$(CFLAGS) -DCOREFUNC=$$funcname pclbench.cpp $$core \$$(LDLIBS) -o \$$@" >> $@ ; \
		test -n "$$funcname" && echo "" >> $@ ; \
	    done ; \
	    echo "pclbench-all: $$targets" >> $@
