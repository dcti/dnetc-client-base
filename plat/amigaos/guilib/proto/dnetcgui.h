#ifndef PROTO_DNETCGUI_H
#define PROTO_DNETCGUI

#include <clib/dnetcgui_protos.h>
#ifndef __NOLIBBASE__
extern struct Library *DnetcBase;
#endif /* __NOLIBBASE__ */

#ifdef __amigaos4__
#ifdef __USE_INLINE__
#include <inline4/dnetcgui.h>
#endif /* __USE_INLINE__ */
#include <interfaces/dnetcgui.h>

#ifndef __NOGLOBALIFACE__
extern struct DnetcIFace *IDnetc;
#endif /* __NOGLOBALIFACE__*/

#else /* __amigaos4__ */
#if defined(__GNUC__)
#if defined(__PPC__)
#include <inline/dnetcguippc.h>
#else
#include <inline/dnetcgui.h>
#endif
#elif defined(__VBCC__)
#ifndef __PPC__
#include <inline/dnetcgui_protos.h>
#endif
#else
#include <pragmas/dnetcgui_pragmas.h>
#endif /* __GNUC__ */
#endif /* __amigaos4__ */
#endif /* PROTO_DNETCGUI_H */
