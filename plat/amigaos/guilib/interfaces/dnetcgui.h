#ifndef DNETCGUI_INTERFACE_DEF_H
#define DNETCGUI_INTERFACE_DEF_H
/*
** This file is machine generated from idltool
** Do not edit
*/ 

#include <exec/types.h>
#include <exec/exec.h>
#include <exec/interfaces.h>

#include <dnetcgui.h>

struct DnetcIFace
{
	struct InterfaceData Data;

	ULONG APICALL (*Obtain)(struct DnetcIFace *Self);
	ULONG APICALL (*Release)(struct DnetcIFace *Self);
	void APICALL (*Expunge)(struct DnetcIFace *Self);
	struct Interface * APICALL (*Clone)(struct DnetcIFace *Self);
	ULONG APICALL (*dnetcguiOpen)(struct DnetcIFace *Self, ULONG cpu, UBYTE * programname, struct WBArg * iconname, const char * vstring);
	BOOL APICALL (*dnetcguiClose)(struct DnetcIFace *Self, struct ClientGUIParams * params);
	ULONG APICALL (*dnetcguiHandleMsgs)(struct DnetcIFace *Self, ULONG signals);
	ULONG APICALL (*dnetcguiConsoleOut)(struct DnetcIFace *Self, ULONG cpu, UBYTE * output, BOOL overwrite);
};

#endif
