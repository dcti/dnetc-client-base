OPTIMIZE=-O2 -fno-schedule-insns

all:
	@echo "no target specified. use:"
	@echo "make desc"
	@echo "for the C-Language version and"
	@echo "make desa"
	@echo "for the asm-version."
	@echo "s-boxes based on the work of Matthew Kwan."
	@echo "Asm had some help from Richard Henderson."
	@echo "Everything else by Christoph Dworzak"

kwan:	kwan.o deseval.o des.o
	gcc -o kwan des.o kwan.o deseval.o
	./kwan <destest

desa:	des.o cka.o
	gcc -o desa des.o cka.o
	time ./desa <desshorttest

desc:	des.o ck.o
	gcc -o desc des.o ck.o
	time ./desc <desshorttest

clean:
	rm -f core *.o des

.S.o:
	gcc -c $<

.c.o:
	gcc $(OPTIMIZE) -c $<

