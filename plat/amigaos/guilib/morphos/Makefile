#
# Makefile for MorphOS dnetcgui.library
#

TARGET	= ../../../../dnetcgui.library

#
##
#

# If you use another compiler change the following line:
CC	= ppc-morphos-gcc
CDEFS	= -DUSE_INLINE_STDARG
CWARN	= -Wall -Wno-parentheses -Wno-missing-braces
CFLAGS	= $(CDEFS) $(CWARN) -O2 -fomit-frame-pointer -mcpu=603 -finline-functions -mstring -mmultiple
LDFLAGS	= -s -noixemul

OBJS	= LibHeader.o LibInit.o AppClass.o MUIGUI.o CreateGUI.o

#
# target
#

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET).db

$(TARGET): $(OBJS)
		  $(CC) $(LDFLAGS) -nostartfiles -o $@.db $(OBJS)
		  ppc-morphos-strip --strip-unneeded --strip-debug --remove-section .comment -o $@ $@.db

GLOBAL		= CreateGui.h ../dnetcgui.h
LibHeader.o	: LibHeader.c $(GLOBAL) LibHeader.h guilib_version.h
LibInit.o	: LibInit.c  $(GLOBAL) LibHeader.h AppClass.h
AppClass.o	: AppClass.c $(GLOBAL) AppClass.h
CreateGUI.o	: CreateGUI.c $(GLOBAL) guilib_version.h
MUIGUI.o	: MUIGUI.c $(GLOBAL)


includes: ../fd/dnetcgui_lib.fd ../clib/dnetcgui_protos.h
	mkdir -p clib proto ppcinline
	cvinclude.pl --fd ../fd/dnetcgui_lib.fd --clib ../clib/dnetcgui_protos.h --outclib clib/dnetcgui_protos.h
	cvinclude.pl --fd ../fd/dnetcgui_lib.fd --clib ../clib/dnetcgui_protos.h --proto proto/dnetcgui.h
	cvinclude.pl --fd ../fd/dnetcgui_lib.fd --clib ../clib/dnetcgui_protos.h --inlines ppcinline/dnetcgui.h
