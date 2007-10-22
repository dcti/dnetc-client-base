#ifndef INLINE4_DNETCGUI_H
#define INLINE4_DNETCGUI_H
/*
** This file is machine generated from idltool
** Do not edit
*/ 

#include <exec/types.h>
#include <exec/exec.h>
#include <exec/interfaces.h>

#include <dnetcgui.h>

/*
 * Inline functions for Interface "main"
 */
#define dnetcguiOpen(cpu, programname, iconname, vstring) IDnetc->dnetcguiOpen(cpu, programname, iconname, vstring) 
#define dnetcguiClose(params) IDnetc->dnetcguiClose(params) 
#define dnetcguiHandleMsgs(signals) IDnetc->dnetcguiHandleMsgs(signals) 
#define dnetcguiConsoleOut(cpu, output, overwrite) IDnetc->dnetcguiConsoleOut(cpu, output, overwrite) 

#endif
