#
# Makefile for dnetcgui.library
#

TARGET	= ../../../dnetcgui.library

#
##
#

CC	= gcc
RM	= delete
CFLAGS	= -m68000 -msmall-code -O2 -fomit-frame-pointer -fpermissive -fno-exceptions -fstrict-aliasing -I../../../plat/amigaos
LDFLAGS	= -noixemul -nostartfiles -s -L.
LDLIBS	= -lreaction
OBJDIR  = ../../../output
REACTOR_OBJS = gui.o $(OBJDIR)/guistrings.o
#OBJS	= $(OBJDIR)/LibBase.o $(OBJDIR)/main.o $(REACTOR_OBJS)
OBJS	= $(OBJDIR)/LibBase.o $(OBJDIR)/main.o $(OBJDIR)/prefs.o

#
# target
#

$(TARGET): $(OBJS) $(OBJDIR)/endcode.o
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS) -lgcc -lnixmain -lnix -lamiga $(OBJDIR)/endcode.o

$(OBJDIR)/LibBase.o: LibBase.c common.h dnetcgui_rev.h
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/main.o: main.c common.h main.h prefs.h
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/prefs.o: prefs.c common.h main.h prefs.h
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/endcode.o: endcode.c
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/guistrings.o: $(OBJDIR)/guistrings.s gui.cd
	phxass $< to $@ sc sd=a4 INCPATH sc:include/ SET CATCOMP_ARRAY

$(OBJDIR)/guistrings.s: gui.cd
	catcomp gui.cd cfile guistrings.h asmfile $(OBJDIR)/guistrings.s xdef

includes: fd/dnetcgui_lib.fd clib/dnetcgui_protos.h
	fd2inline fd/dnetcgui_lib.fd clib/dnetcgui_protos.h > inline/dnetcgui.h
#	fd2inline -proto fd/dnetcgui_lib.fd clib/dnetcgui_protos.h > proto/dnetcgui.h
	fd2inline_ -powerup fd/dnetcgui_lib.fd clib/dnetcgui_protos.h > inline/dnetcguippc.h
#	fd2inline -powerup -proto fd/dnetcgui_lib.fd clib/dnetcgui_protos.h > proto/dnetcguippc.h
	mkdir -p morphos/clib morphos/proto morphos/ppcinline
	cvinclude.pl --fd fd/dnetcgui_lib.fd --clib clib/dnetcgui_protos.h --outclib morphos/clib/cnetcgui_protos.h
	cvinclude.pl --fd fd/dnetcgui_lib.fd --clib clib/dnetcgui_protos.h --proto morphos/proto/dnetcgui.h
	cvinclude.pl --fd fd/dnetcgui_lib.fd --clib clib/dnetcgui_protos.h --inlines morphos/ppcinline/dnetcgui.h
