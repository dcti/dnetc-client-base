# Buffutil makefile for building under Cygwin [andreasb]
# modify LDFLAGS and LIBS for your system

# $Id: Makefile,v 1.1.2.1 2000/02/24 13:21:29 andreasb Exp $

CXX = g++
LD = g++
CXXFLAGS = -I../common -I../des -I../rc5 -I../csc -I../ogr
LDFLAGS = -mconsole -mno-cygwin
LIBS = -lwsock32 -lmingw32 -lbfd -liberty
STRIP = strip
ALLOBJS =  ../output/scram.o buffutil.o

buffutil.exe:	${ALLOBJS}
	${LD} ${LDFLAGS} ${ALLOBJS} ${LIBS} -o $@
	${STRIP} $@

clean:
	rm -f ${ALLOBJS}  buffutil.exe

../output/scram.o:	../common/scram.cpp ../common/cputypes.h ../common/network.h ../common/scram.h
	${CXX} -c ${CXXFLAGS} ../common/scram.cpp -o ../output/scram.o

buffutil.o:	buffutil.cpp ../common/client.h ../common/packets.h ../common/scram.h ../ogr/ogr.h ../common/clisrate.h
	${CXX} -c ${CXXFLAGS} buffutil.cpp -o buffutil.o

