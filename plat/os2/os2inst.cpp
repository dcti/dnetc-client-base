/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------
 * The two functions in this module install/deinstall the client
 * to/from the startup folder
 * ----------------------------------------------------------------
 *
 * $Log: os2inst.cpp,v $
 * Revision 1.2.4.1  2002/11/10 12:15:44  pfeffi
 * OS/2:
 * a) rewritten installation;
 * b) added signal processing (kill, restart, etc);
 *     (only 'kill' is currently supported).
 *     c) added os/2 version of os2GetPIDList() - now 'pause-when-running' etc.
 *         are supported.
 *
 * code was supplied by Roman Trunov, proxyma@tula.net
 *
 * Revision 1.2  2002/09/02 00:35:51  andreasb
 * sync: merged changes from release-2-7111 branch between
 *       2000-07-11 and release-2-7111_20020901 into trunk
 *
 * Revision 1.1.2.2  2002/04/12 23:56:50  andreasb
 * 2002 copyright update - round 2
 *
 * Revision 1.1.2.1  2001/01/21 15:10:27  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.7.2.1  2001/01/01 22:48:57  trevorh
 * Update for OS/2 465 build
 *
 * Revision 1.7  1999/04/11 14:47:14  cyp
 * a) simplified install by passing argv[0]; b) Added comment re:detached
 *
*/
const char *os2inst_cpp(void) {
return "@(#)$Id: os2inst.cpp,v 1.2.4.1 2002/11/10 12:15:44 pfeffi Exp $"; }

//#define TRACE

#include "baseincs.h"
#include "console.h"  // ConOutErr()
#include "util.h"     // trace

#include "dosqss.h"   // DosQProcStatus()

static char RC5DES_FOLDOBJ_NAME[] = "<RC5DES-CLI>";

int os2CliUninstallClient(int do_the_uninstall_without_feedback)
{
	HOBJECT hObject = WinQueryObject((const unsigned char*)RC5DES_FOLDOBJ_NAME);
	char *msg = "The distributed.net client was not found in the Startup Folder.";
	int rc = +1;

	if (hObject != NULLHANDLE)
	{
		rc = -1;
		msg = "The distributed.net client could not be removed from in the Startup Folder.";
		if (WinDestroyObject(hObject) == TRUE)
		{
			if (WinQueryObject((const unsigned char*)RC5DES_FOLDOBJ_NAME) == NULLHANDLE)
			{
				msg = "The distributed.net client was successfully removed from the Startup Folder.";
				rc = 0;
			}
		}
	}

	if (!do_the_uninstall_without_feedback)
		if (rc == 0)
			ConOutModal(msg);
		else
			ConOutErr(msg);
	return rc;
}

int os2CliInstallClient(int do_the_install_without_feedback, const char *exename)
{
#define PSZSTRINGSIZE 4068
	static char
		pszClassName[] = "WPProgram",
		pszTitle[]     = "distributed.net Client",
		pszLocation[]  = "<WP_START>";    // Startup Folder

	int rc, was_installed;
	char pszSetupString[PSZSTRINGSIZE];
	char *execopy = strdup(exename);

	if (execopy == NULL || (rc = os2CliUninstallClient(1)) < 0)
	{ /* was there but couldn't be uninstalled */
		was_installed = 1;
		rc = -1;
	}
	else /* wasn't there or was removed */
	{
		char *p, *q;
		was_installed = (rc == 0);

		for (p = execopy, q = NULL; *p; p++)
			if (*p == ':' || *p == '\\' || *p == ':')
				q = p+1; /* where to split path */
		if (q)
			*q = 0;
		else
			q = "";

		sprintf(pszSetupString,
				"OBJECTID=%s;"
				"MINIMIZED=YES;"
				"PROGTYPE=WINDOWABLEVIO;"
				"EXENAME=%s;"
				"PARAMETERS=-svcstart;"
				"STARTUPDIR=%s",
				RC5DES_FOLDOBJ_NAME,
				exename,
				execopy
			   );

		rc = -1;
		if (WinCreateObject((const unsigned char*)pszClassName, (const unsigned char*)pszTitle, (const unsigned char*)pszSetupString,
							(const unsigned char*)pszLocation, 0) != NULLHANDLE)
		{
			rc = 0;
		}
	}

	if (execopy)
		free(execopy);

	if (!do_the_install_without_feedback)
	{
		sprintf(pszSetupString,
				"The distributed.net client %s %sadded to the Startup Folder.",
				(rc == 0 ? "has been":"could not be"),
				(was_installed ? "re-":"")
			   );
		if (rc == 0)
			ConOutModal(pszSetupString);
		else
			ConOutErr(pszSetupString);
	}
	return rc;
}

// structures for filename splitting

struct elementinfo
{
	const char *pElement;
	int  cbElement;
};

struct partinfo
{
	struct elementinfo drive, path, name;
};

static void split_fileparts(const char *procname, struct partinfo *info)
{
	const char *s, *lastslash;

	memset(info, 0, sizeof(*info));

	// is drive specified ?

	if (procname[0] && procname[1] == ':')
	{
		info->drive.pElement  = procname;
		info->drive.cbElement = 2;
		procname += 2;
	}

	// everything after drive up to last (back)slash is path

	for (s = procname, lastslash = NULL; *s; s++)
	{
		if (*s == '/' || *s == '\\')
			lastslash = s;
	}
	if (lastslash)
	{
		info->path.pElement  = procname;
		info->path.cbElement = lastslash - procname + 1;
		procname = lastslash + 1;
	}

	// only filename left in string

	info->name.pElement  = procname;
	info->name.cbElement = strlen(procname);
}

// match element of filename, if present
// ignore case and slash types (back/forward)

static int match_fileparts(struct elementinfo *a, struct elementinfo *b)
{
	const char *pa = a->pElement;
	const char *pb = b->pElement;
	int len = a->cbElement;

	if (pa == NULL || pb == NULL)
		return 1;
	if (len != b->cbElement)
		return 0;
	for (; len; len--, pa++, pb++)
	{
		char ca = (char) toupper(*pa);
		char cb = (char) toupper(*pb);

		if (ca == '\\')
			ca = '/';
		if (cb == '\\')
			cb = '/';
		if (ca != cb)
			return 0;
	}
	return 1;
}

#define MAX_CLIENT_PROGS 16
/* 0=notfound, 1=found+ok, -1=error */
int os2CliSendSignal(int action, const char *exename)
{
	long pids[MAX_CLIENT_PROGS];
	struct partinfo exeparts;
	int ret, nClients, i;

	TRACE_OUT((0, "os2CliSendSignal: %02X to '%s'\n", action, exename));

	// we will search for programs with same name, even if path will differ. Ok?

	split_fileparts(exename, &exeparts);

	nClients = utilGetPIDList(exeparts.name.pElement, pids, MAX_CLIENT_PROGS);
	if (nClients <= 0)
		return 0;  // error or no clients -- return 'no clients found'

	ret = 1; // by default, return 'ok'
	for (i = 0; i < nClients; i++)
	{
		APIRET rc;
		TRACE_OUT((0, "os2CliSendSignal: action %02x to pid %d\n", action, pids[i]));
		switch(action)
		{
		case DNETC_MSG_SHUTDOWN:
			rc = DosKillProcess(DKP_PROCESS, pids[i]);
			break;
		default: // others are not supported. yet.
			rc = 0;
			ret = -1; // error
			break;

//		case DNETC_MSG_RESTART:  rc = 0; break;
//		case DNETC_MSG_PAUSE:	 rc = 0; break;
//		case DNETC_MSG_UNPAUSE:  rc = 0; break;
		}
		if (rc)
		{
			TRACE_OUT((0, "? os2CliSendSignal: action %02X failed with rc=%d\n", action, rc));
			ret = -1; // an error occured
		}
	}
    return ret;
}

/*
 get list of pid's for procname. if pidlist is NULL or maxnumpids is 0,
 then return found count, else return number of pids now in list.
 On error return < 0.
 */
int os2GetPIDList( const char *procname, long *pidlist, int maxnumpids )
{
	static time_t lasttime;
	time_t curtime;

	static PVOID infobuf;

	APIRET rc = 0;
	int num_found = -1;

	TRACE_OUT((+1, "os2GetPIDList: '%s' to %p (max %d)\n", procname, pidlist, maxnumpids));

	// allocate buffer on first call

	if (infobuf == NULL)
	{
		rc = DosAllocMem(&infobuf, 0x10000, fALLOC);
		if (rc)
		{
			TRACE_OUT((0, "? DosAllocMem() error %d\n", rc));
			infobuf = NULL;
		}
	}

	// do not update process info too often, 1 second should be enough

	// we must use 16-bit DosQProcStatus() because DosQuerySysState() is
	// buggy (race conditions inside kernel) and can cause kernel traps

	curtime = time(NULL);
	if (rc == 0 && lasttime != curtime)
	{
		lasttime = curtime;

		TRACE_OUT((0, "Time to update info\n"));
		rc = DosQProcStatus(infobuf, 0xFFFF);
		if (rc != 0)
		{
			TRACE_OUT((0, "? DosQProcStatus() failed with rc=%d", rc));
		}
	}

	if (rc == 0)  // successefully retrieved data
	{
		struct partinfo baseinfo;
		PQPROCESS pProcess;
		unsigned ownpid = getpid();

		split_fileparts(procname, &baseinfo);

		// main loop along processes

		for (pProcess = ((PQTOPLEVEL)infobuf)->procdata;
			 pProcess && pProcess->rectype == 1;
			 pProcess = (PQPROCESS) (pProcess->threads+pProcess->threadcnt)
			)
		{
			PQMODULE pModule;

			if (pProcess->pid == ownpid)
				continue; /* hey, it's me! */

			for (pModule = ((PQTOPLEVEL)infobuf)->moddata; pModule; pModule = pModule->next)
			{
				if (pProcess->hndmod == pModule->hndmod)
					break;
			}
			if (pModule)
			{
				struct partinfo curmodinfo;

				TRACE_OUT((0, "module: %s\n", pModule->name));

				split_fileparts(pModule->name, &curmodinfo);

				if (match_fileparts(&baseinfo.drive, &curmodinfo.drive) &&  // drive matches/absent
					match_fileparts(&baseinfo.path,  &curmodinfo.path ) &&  // path matches/absent
					(match_fileparts(&baseinfo.name, &curmodinfo.name) || // filename matches completely
					 // or base part of filename matches and extension is '.EXE'
					 (memicmp(curmodinfo.name.pElement, baseinfo.name.pElement, baseinfo.name.cbElement) == 0 &&
					  stricmp(curmodinfo.name.pElement+baseinfo.name.cbElement, ".EXE") == 0
					 )
					)
				   )
				{
					TRACE_OUT((0, "compare of '%s' and '%s': match\n", procname, pModule->name));
					num_found++;

					if (pidlist)
					{
						if (num_found >= maxnumpids)
							break; // out of process loop
						pidlist[num_found] = pProcess->pid;
					}
				}
			} // module found
		} // process loop
		num_found++;  // return count of pid's found
	}

	TRACE_OUT((-1, "os2GetPIDList => %d\n", num_found));
	return num_found;
}
