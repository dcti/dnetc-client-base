CFLAGS = -O2 -g

all: choose test-ogr test-stubsplit

test-ogr: test-ogr.o ogr.o choosedat.o crc32.o

test-stubsplit: test-stubsplit.o stubsplit.o

ogr.o: ogr.c client2.h crc32.h stub.h

stubsplit.o: stubsplit.c stubsplit.h stub.h

test-stubsplit.o: test-stubsplit.c stubsplit.h stub.h

test-ogr.o: test-ogr.c client2.h stub.h

choosedat.c: choose.dat bin2c.pl
	perl bin2c.pl choose.dat choosedat.c choose_dat

choose.dat: choose.c
	./choose
	rm -f choose.dat
	mv choose.new choose.dat

choose: choose.o

dist:
	rm -f ogr.zip
	zip -9 ogr Makefile client2.h ogr.c crc32.h crc32.c test-ogr.c stub.h stubsplit.h stubsplit.c test-stubsplit.c choose.c bin2c.pl

clean:
	rm -f *.o
	rm -f choosedat.c test-ogr test-stubsplit choose ogr.zip

chooseclean:
	rm -f *.o
	rm -f choose.dat
	rm -f choosedat.c test-ogr test-stubsplit choose ogr.zip
