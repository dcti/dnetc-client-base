CFLAGS = -O2 -g -Wall -Du16=short

all: choose test-ogr test-stubsplit

test-ogr: test-ogr.o ogr.o choosedat.o crc32.o

test-stubsplit: test-stubsplit.o stubsplit.o

ogr.o: ogr.cpp client2.h crc32.h ogr.h
	g++ $(CFLAGS) -c $<

stubsplit.o: stubsplit.cpp stubsplit.h ogr.h
	g++ $(CFLAGS) -c $<

test-stubsplit.o: test-stubsplit.cpp stubsplit.h ogr.h
	g++ $(CFLAGS) -c $<

test-ogr.o: test-ogr.cpp client2.h ogr.h
	g++ $(CFLAGS) -c $<

choosedat.o: choosedat.cpp
	g++ $(CFLAGS) -c $<

crc32.o: crc32.cpp
	g++ $(CFLAGS) -c $<

choosedat.cpp: choose.dat bin2c.pl
	perl bin2c.pl choose.dat choosedat.cpp choose_dat

choose.dat: choose.c
	./choose
	rm -f choose.dat
	mv choose.new choose.dat

choose: choose.o

dist:
	rm -f ogr.zip
	zip -9 ogr Makefile client2.h ogr.c crc32.h crc32.c test-ogr.c ogr.h stubsplit.h stubsplit.c test-stubsplit.c choose.c bin2c.pl

clean:
	rm -f *.o
	rm -f choosedat.cpp test-ogr test-stubsplit choose ogr.zip

chooseclean:
	rm -f *.o
	rm -f choose.dat
	rm -f choosedat.cpp test-ogr test-stubsplit choose ogr.zip
