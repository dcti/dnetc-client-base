# $Id: Makefile,v 1.1.2.4 2003/04/01 22:35:36 andreasb Exp $

all: pclbench

BASEDIR		= ../..
COREOBJFILES	= $(wildcard $(BASEDIR)/output/r72*.o)

pclbench: pclbench.mk
	$(MAKE) -f pclbench.mk

pclbench.mk: pclbench.mk.*.in $(COREOBJFILES)
	-rm -f $@.tmp
	for core in $(COREOBJFILES) ; do \
		funcname=`nm $$core | grep rc5_72_unit_func | sed -e 's/.*rc5_72_unit_func_//;s/_$$//' | head -1` ; \
		test -n "$$funcname" && targets="$$targets pclbench-$$funcname" ; \
		test -n "$$funcname" && echo "pclbench-$$funcname: pclbench.cpp config.h $$core" >> $@.tmp ; \
		test -n "$$funcname" && echo "	\$$(CC) \$$(CFLAGS) -DCOREFUNC=$$funcname pclbench.cpp $$core \$$(LDLIBS) -o \$$@" >> $@.tmp ; \
		test -n "$$funcname" && echo "" >> $@.tmp ; \
	    done ; \
	    sed -e "s/@TARGETS@/$$targets/" $@.`uname -s`.in > $@
	cat $@.tmp >> $@
	-rm -f $@.tmp

clean:
	-test -f pclbench.mk && make -f pclbench.mk clean
	-rm -f pclbench.mk
