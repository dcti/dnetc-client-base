/* Cobbled together from various sources (NtQuerySystemInformation() from
 * EnTeHandle from Fred Forester http://www.cyberenet.net/~fforest/, and
 * NT Performance counters from various msdn support documents), by
 * Cyrus Patel <cyp@fb14.uni-mainz.de> for use by the distributed.net client.
 * The crazy stuff with mailslots is my doing.
 *
 * This module contains stubs for toolhelp32's CreateToolhelp32Snapshot(),
 * Process32First() and Process32Next(), and emulation routines thereof
 * for NT3/NT4 which don't have these functions in the kernel. Selection
 * of emulation method (NtQuerySystemInformation() or Performance Counters)
 * is automatic when real toolhelp is not available.
 *
 * For emulation, we use a mailslot because handles returned from
 * CreateToolhelp32Snapshot() need to be closable with CloseHandle(), and
 * because mailslots are convenient. :)
 * caveats: a) only does TH32CS_SNAPPROCESS, b) Process32First will not
 * 'rewind' the snap, it gets the next. (this could be fixed by making the
 * mailslot a closed loop, or reading/rewriting a single message, but both
 * solutions would be computationally expensive.)
 *
 * On NT, irrespective of which method was used to obtain the data [real
 * toolhelp, performance counters, NtQuerySystemInformation], the pe.cntUsage
 * and pe.th32DefaultHeapID are always zero and pe.szExeFile has no dirspec.
 * In addition, if using performance counters, the pe.cntThreads,
 * pe.th32ParentProcessID and pe.pcPriClassBase fields are zero as well,
 * and pe.szExeFile usually (I've seen a few ".com"s, but not all .com's
 * appear with ".com") does not have an extension.
 *
 * $Id: w32snapp.c,v 1.1.2.3 2002/03/12 22:55:13 jlawson Exp $
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winperf.h>  /* performance counter data stuff */
#include <tlhelp32.h> /* toolhlp32 structures and function prototypes */

//#define FORCE_TOOLHELP /* the real toolhelp snapshot functions */
//#define FORCE_NTQUERY  /* the NtQuery snapshot emulation routines */
//#define FORCE_PERFCAPS /* the perfcaps snapshot emulation routines */
//#define TEST_WITH_MAIN /* make this file standalone for testing */

#if defined(TEST_WITH_MAIN)
  #include <stdio.h>
#endif

/* ---------------------------------------------------- */

#pragma pack(1)
typedef struct _SYSTEM_PROCESS_INFORMATION
{
  ULONG NextEntryOffset;
  ULONG NumberOfThreads;
  LARGE_INTEGER SpareLi1;
  LARGE_INTEGER SpareLi2;
  LARGE_INTEGER SpareLi3;
  LARGE_INTEGER CreateTime;
  LARGE_INTEGER UserTime;
  LARGE_INTEGER KernelTime;
  struct { USHORT Length; USHORT MaximumLength; PWSTR Buffer; } ImageName;
                                /* UNICODE_STRING ImageName; */
  LONG BasePriority;
  HANDLE UniqueProcessId;
  HANDLE InheritedFromUniqueProcessId;
  ULONG HandleCount;
  ULONG SpareUl2;
  ULONG SpareUl3;
  ULONG PeakVirtualSize;
  ULONG VirtualSize;
  ULONG PageFaultCount;
  ULONG PeakWorkingSetSize;
  ULONG WorkingSetSize;
  ULONG QuotaPeakPagedPoolUsage;
  ULONG QuotaPagedPoolUsage;
  ULONG QuotaPeakNonPagedPoolUsage;
  ULONG QuotaNonPagedPoolUsage;
  ULONG PagefileUsage;
  ULONG PeakPagefileUsage;
  ULONG PrivatePageCount;
} SYSTEM_PROCESS_INFORMATION, *PSYSTEM_PROCESS_INFORMATION;
#pragma pack()

typedef DWORD (WINAPI *NtQuerySystemInformationT) (
        DWORD SystemInformationClass, PVOID SystemInformation,
        ULONG SystemInformationLength, PULONG ReturnLength);


static HANDLE WINAPI NtQuery_CreateToolhelp32Snapshot(DWORD dwFlags,
                                                      DWORD th32ProcessID)
{
  /* emulate toolhelp32's CreateToolhelp32Snapshot() function using
     the undocumented NtQuerySystemInformation.

     Yes, it does work on NT5 as well, but toolhelp is available there.
     Whether its available on NT3 is unknown.

     NtQuerySystemInformation is an undocumented internal function in
     ntdll.dll that is not intended for external-MS use since they those
     are subject to change between system releases. It should be safe on NT4
     though given that that it hasn't changed between 4 and 5.
     Moreover, there do exist other external-MS utilities that use this call,
     such as http://www.sysinternals.com/listdlls.htm, and Fred Forester's
     EnTeHandle http://www.cyberenet.net/~fforest/handle.html
  */
  char mailslotname[128];
  FILETIME ftUnique, ftJunk;
  HANDLE hSnapshot = (HANDLE)0; /* assume failure */

  GetProcessTimes(GetCurrentProcess(), &ftJunk, &ftJunk, &ftJunk, &ftUnique );
  wsprintf(mailslotname,"\\\\.\\mailslot\\Thelp32.emu\\%lu\\%lu\\%lu",
     GetCurrentProcessId(), ftUnique.dwHighDateTime, ftUnique.dwLowDateTime );

  if ((dwFlags & TH32CS_SNAPPROCESS)!=0) /* the only flag we currently support */
  {
    static int issupported = -1;
    static NtQuerySystemInformationT fnNtQuerySystemInformation = 0;

    if (issupported == -1)
    {
      const char *loadname = "ntdll.dll";
      HINSTANCE hWeLoadedItInst = NULL;
      HMODULE hNtdll = GetModuleHandle( loadname );
      if (hNtdll == NULL) /* shouldn't be needed, but doesn't hurt */
      {
        HINSTANCE hWeLoadedItInst = LoadLibrary( loadname );
        if (hWeLoadedItInst)
          hNtdll = GetModuleHandle( loadname );
      }
      if (hNtdll == NULL)
        fnNtQuerySystemInformation = (NtQuerySystemInformationT)0;
      else
        fnNtQuerySystemInformation = (NtQuerySystemInformationT)
                GetProcAddress(hNtdll, "NtQuerySystemInformation");
      issupported = ((fnNtQuerySystemInformation)?(+1):(0));
      if (!issupported && hWeLoadedItInst)
        FreeLibrary( (HMODULE)hWeLoadedItInst );
      /* otherwise keep it loaded for speed */
    }
    if (issupported && fnNtQuerySystemInformation)
    {
      HANDLE hProcessHeap = GetProcessHeap();
      if (hProcessHeap)
      {
        static DWORD dwStartingBytes = 8192;
        DWORD dwNumBytesRet = 0;
        PSYSTEM_PROCESS_INFORMATION pProcessList;

        if (dwStartingBytes < 8192)
          dwStartingBytes = 8192;

        /* Ask the system for the list of running processes */
        pProcessList = (PSYSTEM_PROCESS_INFORMATION)
            HeapAlloc( hProcessHeap, HEAP_ZERO_MEMORY, dwStartingBytes );
        while (pProcessList)
        {
          DWORD rc = (*fnNtQuerySystemInformation)(5, pProcessList,
                            dwStartingBytes, &dwNumBytesRet);
//printf("getinfo %d -> result %d\n", (int) dwStartingBytes, rc);
          if (rc == 0 && dwNumBytesRet != 0)
          {
            //assert(dwNumBytesRet <= dwStartingBytes);
            break; /* found it! */
          }
          else
          {
            dwStartingBytes += 4096; /* make it easier for HeapAlloc */
            HeapFree( hProcessHeap, 0, pProcessList);
            pProcessList = (PSYSTEM_PROCESS_INFORMATION)
              HeapAlloc( hProcessHeap, HEAP_ZERO_MEMORY, dwStartingBytes );
          }
        }
        dwStartingBytes -= 1024; /*gradually diminish the size for next time.*/

        /* Now walk the process list, convert each to a PROCESSENTRY32 entry
           and stuff those into the mailslot.
        */
        if (pProcessList != NULL)
        {
          BOOL bWeCreatedSnapHandle = FALSE;
          if (hSnapshot == NULL)
          {
            hSnapshot = CreateMailslot( mailslotname, 0, 0, NULL );
//printf("createmailslot=%p\n", hSnapshot);
            bWeCreatedSnapHandle = TRUE;
            if (hSnapshot == INVALID_HANDLE_VALUE)
            {
              hSnapshot = NULL;
              bWeCreatedSnapHandle = FALSE;
            }
          }
          if (hSnapshot != NULL)
          {
            DWORD num_added = 0;
            HANDLE hSpool = CreateFile( mailslotname, GENERIC_WRITE,
                                        FILE_SHARE_READ, NULL,
                                        OPEN_EXISTING,
                                        FILE_ATTRIBUTE_NORMAL,
                                        (HANDLE)NULL );
//printf("hspool=%p\n", hSpool);
            if (hSpool)
            {
              PSYSTEM_PROCESS_INFORMATION pWalk = pProcessList;
              BOOL bDonePidZero = FALSE;

              while ((BYTE*)pWalk < ((BYTE*)pProcessList) + dwNumBytesRet)
              {
                if (pWalk->ImageName.Buffer != NULL)
                {
                  PROCESSENTRY32 pe;
                  DWORD dwBytes = WideCharToMultiByte( CP_ACP, 0,
                                  pWalk->ImageName.Buffer,
                                  pWalk->ImageName.Length / sizeof(WCHAR),
                                  pe.szExeFile, sizeof(pe.szExeFile),
                                  NULL, NULL);
                  if (dwBytes != 0 && dwBytes < sizeof(pe.szExeFile))
                  {
                    pe.szExeFile[dwBytes] = '\0';
                    pe.dwSize = sizeof(pe);
                    pe.cntUsage = 0; /* "refs to pid". real thelp sets zero */
                    pe.th32ProcessID = (DWORD)pWalk->UniqueProcessId;
                    pe.th32DefaultHeapID = 0; /* zero from real toolhelp */
                    pe.th32ModuleID = 0; /* real toolhelp's reference */
                    pe.cntThreads = pWalk->NumberOfThreads;
                    pe.th32ParentProcessID = (DWORD)pWalk->InheritedFromUniqueProcessId;
                    pe.pcPriClassBase = pWalk->BasePriority;
                    pe.dwFlags = 0; /* toolhelp "reserved" */
//printf("process %p -> %s\n", pWalk->UniqueProcessId, pe.szExeFile);

                    if (!bDonePidZero)
                    {
                      bDonePidZero = TRUE;
                      if (pe.th32ProcessID == 0)
                        lstrcpy(pe.szExeFile,"[System Process]");
                      else
                      {
                        /* synthesize a record that matches the first one
                           returned by toolhelp and perfcaps
                        */
                        PROCESSENTRY32 syspe;
                        lstrcpy(syspe.szExeFile,"[System Process]");
                        syspe.dwSize = sizeof(pe);
                        syspe.cntUsage = 0;
                        syspe.th32ProcessID = 0;
                        syspe.th32DefaultHeapID = 0;
                        syspe.th32ModuleID = 0;
                        syspe.cntThreads = 1;
                        syspe.th32ParentProcessID = 0;
                        syspe.pcPriClassBase = 0;
                        syspe.dwFlags = 0;
                        if (!WriteFile(hSpool,&syspe,sizeof(syspe),&dwBytes,0))
                          break;
                        if (dwBytes != sizeof(syspe))
                          break;
                        num_added++;
                      }
                    }
                    else if (pe.th32ProcessID == 0)
                    {
                      pe.szExeFile[0] = '\0';
                    }

                    if (pe.szExeFile[0])
                    {
                      if (!WriteFile( hSpool, &pe, sizeof(pe), &dwBytes, NULL))
                        break;
                      if (dwBytes != sizeof(pe))
                        break;
                      num_added++;
                    }
                  }
                }
                /* move onto the next record. */
                if (pWalk->NextEntryOffset <
                           sizeof(SYSTEM_PROCESS_INFORMATION))
                  break;
                pWalk = (PSYSTEM_PROCESS_INFORMATION)
                          ( ((BYTE*)pWalk) + pWalk->NextEntryOffset);
              }  /* while pWalk ... */
              CloseHandle( hSpool );
            } /* if (hSpool) */
            if (num_added == 0 && bWeCreatedSnapHandle)
            {
              CloseHandle(hSnapshot);
              hSnapshot = NULL;
            }
          } /* if (hSnapshot != NULL) */
          HeapFree( hProcessHeap, 0, pProcessList);
        } /* if (pProcessList != NULL) */
      } /* if (hProcessHeap) */
    } /* if (issupported && fnNtQuerySystemInformation) */
  } /* if ((dwFlags & THCS_SNAPPROCESS) != 0) */

  th32ProcessID = th32ProcessID; /* shaddup compiler */
  return hSnapshot;
}

/* ---------------------------------------------------- */

static DWORD my_atoi(const char *str)
{
  DWORD num = 0;
  for (;;)
  {
    char c = (char)(*str++);
    if (c < '0' || c >'9')
      break;
    num = (num * 10) + (c - '0');
  }
  return num;
}

static HANDLE WINAPI perfCaps_CreateToolhelp32Snapshot(DWORD dwFlags,
                                                      DWORD th32ProcessID)
{
  /*
    This method uses Windows NT Performance Counters, as seen in
    http://msdn.microsoft.com/library/psdk/pdh/perfdata_9feb.htm
    http://support.microsoft.com/support/kb/articles/Q119/1/63.asp
    The following code will work on all versions of NT (3x, 4x, win2k)

    However, the perfcap specification allows third-party libraries
    to extend standard system counters, which means that these custom
    third-party perfcap libraries may also be indirectly loaded when this is
    used, which in turn may impose a large memory footprint. This is
    particularly noticable in conjunction with an Oracle (server) installation
    which adds *6*MB overhead.

    For NT4 and above, there is a PSAPI.DLL as shipped with the platform sdk
    (and consequently not a standard component), that provides a fairly simple
    method of obtaining a list of process IDs. Although PSAPI.DLL uses the
    undocumented NtQuerySystemInformation() call, it requires the caller
    to use OpenProcess() to obtain anything (even the basename) other than
    the pid, which makes its usage doggone slow. PSAPI is documented at
    http://msdn.microsoft.com/library/psdk/winbase/psapi_1ulh.htm
  */
  char mailslotname[128];
  FILETIME ftUnique, ftJunk;
  HANDLE hSnapshot = (HANDLE)0; /* assume failure */

  GetProcessTimes(GetCurrentProcess(), &ftJunk, &ftJunk, &ftJunk, &ftUnique );
  wsprintf(mailslotname,"\\\\.\\mailslot\\Thelp32.emu\\%lu\\%lu\\%lu",
     GetCurrentProcessId(), ftUnique.dwHighDateTime, ftUnique.dwLowDateTime );

  if ((dwFlags & TH32CS_SNAPPROCESS)!=0) /* the only flag we currently support */
  {
    HANDLE hProcessHeap = GetProcessHeap();
    if (hProcessHeap)
    {
      static DWORD dwIndex_PROCESS = (DWORD) -1;
      static DWORD dwIndex_IDPROCESS = (DWORD) -1;
      /*
       On the first time through, we need to identify the name/index of the
       performance counters that we are interested in later querying.
       Since this information should not change until next reboot, we can
       keep this information in static variables and reuse them next time.
      */
      if (dwIndex_PROCESS == (DWORD) -1 || dwIndex_IDPROCESS == (DWORD) -1)
      {
        HKEY hKeyIndex;
        if (RegOpenKeyEx( HKEY_LOCAL_MACHINE,
             "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Perflib\\009",
              0, KEY_READ, &hKeyIndex ) == ERROR_SUCCESS)
        {
          /* Get the size of the counter. */
          DWORD dwBytes = 0;
          if (RegQueryValueEx( hKeyIndex, "Counter",
                       NULL, NULL, NULL,  &dwBytes ) == ERROR_SUCCESS)
          {
            char *pszBuffer = NULL;
            if (dwBytes != 0)
              pszBuffer = (char *)HeapAlloc( hProcessHeap,
                                           HEAP_ZERO_MEMORY, dwBytes );
            if (pszBuffer != NULL)
            {
              /* Get the titles and counters. (REG_MULTI_SZ).  This registry
                 key is an alternating set of names and corresponding integers
                 that represent them.  We look for the ones we want by name.
              */
              if (RegQueryValueEx( hKeyIndex, "Counter", NULL, NULL,
                        (LPBYTE)pszBuffer, &dwBytes ) == ERROR_SUCCESS )
              {
                DWORD pos = 0;

                while( dwIndex_PROCESS == (DWORD) -1 ||
                       dwIndex_IDPROCESS == (DWORD) -1 )
                {
                  /* save the current position in "valpos", and skip to the
                     end of the first string (the numeric value), while also
                     counting the length of it in "vallen".
                  */
                  DWORD valpos = pos, vallen = 0, cmppos;
                  while (pos < dwBytes && pszBuffer[pos] != '\0')
                  {
                    pos++;
                    vallen++;
                  }
                  if (pos >= dwBytes)
                    break;
                  /*
                    save the position of the start of the name in "cmpppos"
                    and then skip to the end of the string.
                  */
                  cmppos = (++pos); /* skip the '\0' */
                  while (pos < dwBytes && pszBuffer[pos] != '\0')
                    pos++;
                  if (pos >= dwBytes)
                    break;
                  pos++; /* skip the '\0' */

                  /* See if this value is one that we are looking for. */
//printf("Counter='%s', value=%u\n", &pszBuffer[cmppos], my_atoi(&pszBuffer[valpos]));
                  if (lstrcmpi( &pszBuffer[cmppos], "Process") == 0 )
                  {
                    if (dwIndex_PROCESS == (DWORD)-1)
                      dwIndex_PROCESS = (DWORD)my_atoi(&pszBuffer[valpos]);
                  }
                  else if (lstrcmpi( &pszBuffer[cmppos],"ID Process") == 0)
                  {
                    if (dwIndex_IDPROCESS == (DWORD)-1)
                      dwIndex_IDPROCESS = (DWORD)my_atoi(&pszBuffer[valpos]);
                  }

                }
              }
              HeapFree( hProcessHeap, 0, (LPVOID)pszBuffer );
            }
          }
          RegCloseKey( hKeyIndex );
        }
      }

      /*
         Now that we have identified the performance counter that we are
         interested in, actually do the query to retrieve the data
         associated with it so that we can get the list of processes.
      */
      if (dwIndex_PROCESS != (DWORD)-1 && dwIndex_IDPROCESS != (DWORD)-1)
      {
        static DWORD dwStartingBytes = 8192;
        PPERF_DATA_BLOCK pdb; LONG lResult;
        char szIndexName[sizeof(DWORD)*3];
        DWORD dwBytes;

        /* Read in all object table/counters/instrecords for "Process".
           We call RegQueryValueEx in a loop because "If hKey specifies
           HKEY_PERFORMANCE_DATA and the lpData buffer is too small,
           RegQueryValueEx returns ERROR_MORE_DATA but lpcbData does not
           return the required buffer size. This is because the size of
           the performance data can change from one call to the next. In
           this case, you must increase the buffer size and call
           RegQueryValueEx again passing the updated buffer size in the
           lpcbData parameter. Repeat this until the function succeeds"
        */
        dwBytes = dwStartingBytes;
        if (dwBytes < 8192)
          dwBytes = 8192;
        pdb = (PPERF_DATA_BLOCK)HeapAlloc( hProcessHeap,
                                           HEAP_ZERO_MEMORY, dwBytes );
        lResult = ERROR_MORE_DATA;
        wsprintf(szIndexName, "%u", dwIndex_PROCESS);
        while (pdb != NULL && lResult == ERROR_MORE_DATA)
        {
          /*
            The SDK says "lpcbData does not return the required buffer
            size" (see above), but NT4 has a bug, and returns lpcbData
            set to HALF of what it was before. See...
            http://support.microsoft.com/support/kb/articles/Q226/3/71.ASP
            So, we remember the last used size, and ignore what the OS
            tells us what we should use.
          */
          lResult = RegQueryValueEx(HKEY_PERFORMANCE_DATA,
                                (LPTSTR)szIndexName, NULL, NULL,
                                (LPBYTE)pdb, &dwBytes);
          if (lResult == ERROR_SUCCESS)
          {
            /* save the sizes for the next time we run this function */
            dwStartingBytes = dwBytes - 128;
            break;
          }
          if (lResult == ERROR_MORE_DATA)
          {
            LPVOID newmem;
            dwBytes = dwStartingBytes + 4096;
            if (dwBytes < dwStartingBytes) /* overflow */
              break;
            newmem = HeapReAlloc( hProcessHeap, HEAP_ZERO_MEMORY,
                                           (LPVOID)pdb, dwBytes );
            if (newmem == NULL) /* couldn't realloc.  abort. */
              break;
            pdb = (PPERF_DATA_BLOCK)newmem;
            dwStartingBytes = dwBytes;
          }
        }

        if (pdb != NULL)  /* got perfdata? */
        {
          if (lResult == ERROR_SUCCESS) /* is that perfdata valid? */
          {
            BOOL bWeCreatedSnapHandle = FALSE;
            if (hSnapshot == NULL)
            {
              hSnapshot = CreateMailslot( mailslotname, 0, 0, NULL );
//printf("createmailslot=%p\n", hSnapshot);
              bWeCreatedSnapHandle = TRUE;
              if (hSnapshot == INVALID_HANDLE_VALUE)
              {
                hSnapshot = NULL;
                bWeCreatedSnapHandle = FALSE;
              }
            }
            if (hSnapshot != NULL)
            {
              DWORD num_added = 0;
              HANDLE hSpool = CreateFile( mailslotname, GENERIC_WRITE,
                                          FILE_SHARE_READ, NULL,
                                          OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL,
                                          (HANDLE)NULL );
//printf("hspool=%p\n", hSpool);
              if (hSpool)
              {
                LONG i, totalcount;
                PPERF_OBJECT_TYPE         pot;
                PPERF_COUNTER_DEFINITION  pcd;
                PPERF_INSTANCE_DEFINITION piddef;
                DWORD dwProcessIdOffset = 0;
                BOOL bDonePidZero = FALSE;

                /* Get the PERF_OBJECT_TYPE. */
                pot = (PPERF_OBJECT_TYPE)(((PBYTE)pdb) + pdb->HeaderLength);

                /* Get the first counter definition. */
                pcd = (PPERF_COUNTER_DEFINITION)(((PBYTE)pot) + pot->HeaderLength);

                /* walk the counters to find the offset to the ProcessID */
                totalcount = pot->NumCounters;
                for ( i=0; i < totalcount; i++ )
                {
                  /* get offset of the processID in the PERF_COUNTER_BLOCKs */
                  if (pcd->CounterNameTitleIndex == dwIndex_IDPROCESS)
                  {
                    dwProcessIdOffset = pcd->CounterOffset;
                    break;
                  }
                  pcd = ((PPERF_COUNTER_DEFINITION)(((PBYTE)pcd) + pcd->ByteLength));
                }

                /* Get the first process instance definition */
                piddef = (PPERF_INSTANCE_DEFINITION)(((PBYTE)pot) + pot->DefinitionLength);

    //printf("getpidlist 3: numpids = %u\n", pot->NumInstances );

                /* now walk the process definitions */
                totalcount = pot->NumInstances;
                for ( i = 0; i < totalcount; i++ )
                {
                  PPERF_COUNTER_BLOCK pcb;
                  PWSTR foundname;
                  DWORD namelen;
                  DWORD thatpid;

                  pcb = (PPERF_COUNTER_BLOCK) (((PBYTE)piddef) + piddef->ByteLength);
                  thatpid = *((DWORD *) (((PBYTE)pcb) + dwProcessIdOffset));
                  namelen = piddef->NameLength;
                  foundname = (PWSTR)0;
                  if (piddef->NameOffset)
                    foundname = (PWSTR)(((PBYTE)piddef) + piddef->NameOffset);

                  /* we have all the data we need, skip to the next pid. */
                  piddef = (PPERF_INSTANCE_DEFINITION) (((PBYTE)pcb) + pcb->ByteLength);

                  /* if (thatpid != 0) */
                  {
                    PROCESSENTRY32 pe;
                    dwBytes = 1;
                    if (thatpid == 0) /* "Idle", "_Total" */
                    {
                      dwBytes = 0;
                      if (!bDonePidZero)
                      {
                        /* convert the first instance of pid 0 to what
                           both NtQuerySystemInformation and toolhelp
                           give us: "[System Process]"
                        */
                        lstrcpy(pe.szExeFile,"[System Process]");
                        dwBytes = lstrlen(pe.szExeFile)+1;
                        bDonePidZero = TRUE;
                      }
                    }
                    else if (foundname && namelen > 1)
                    {
                      /* Namelen is "Length in bytes of name; 0 = none;
                         this length includes the characters in the string
                         plus the size of the terminating NULL char."
                      */
                      dwBytes = WideCharToMultiByte( CP_ACP, 0,
                                   foundname,  namelen/sizeof(WCHAR),
                                   pe.szExeFile, sizeof(pe.szExeFile),
                                   NULL, NULL);
                    }
                    /* WCTMB return value includes trailing null */
                    if (dwBytes > 0 && dwBytes < sizeof(pe.szExeFile))
                    {
                      pe.szExeFile[dwBytes-1] = '\0';
//printf("process=0x%x, namelen=%d, '%s'\n", thatpid, namelen, pe.szExeFile );
                      pe.dwSize = sizeof(pe);
                      pe.cntUsage = 0; /* "#refs to pid", real thelp sets 0 */
                      pe.th32ProcessID = thatpid;
                      pe.th32DefaultHeapID = 0; /* real toolhelp sets zero */
                      pe.th32ModuleID = 0; /* real toolhelp's reference */
                      pe.cntThreads = 0; /* we don't know */
                      pe.th32ParentProcessID = 0; /* unknown */
                      pe.pcPriClassBase = 0; /* unknown */
                      pe.dwFlags = 0; /* toolhelp "reserved" */

                      if (!WriteFile( hSpool, &pe, sizeof(pe), &dwBytes, NULL))
                        break;
                      if (dwBytes != sizeof(pe))
                        break;
                      num_added++;
                    }
                  } /* if (foundname && thatpid != 0) */
                } /* for ( i = 0; i < totalcount; i++ ) */
                CloseHandle(hSpool);
              } /* if (hSpool) */
              if (num_added == 0 && bWeCreatedSnapHandle)
              {
                CloseHandle(hSnapshot);
                hSnapshot = NULL;
              }
            } /* if (hSnapshot != NULL) */
          } /* if RegQueryValueEx == ERROR_SUCCESS */

          HeapFree(hProcessHeap, 0, (LPVOID)pdb );
        } /* if (pdb != NULL) */

      } /* if (dwIndex_PROCESS && dwIndex_IDPROCESS) */
    } /* if (hProcessHeap) */
  } /* if ((dwFlags & TH32CS_SNAPPROCESS)!=0) */

  th32ProcessID = th32ProcessID; /* shaddup compiler */
  return hSnapshot;
}

/* ---------------------------------------------------- */

static BOOL WINAPI msrw_Process32Next(HANDLE hSnapshot, LPPROCESSENTRY32 lppe)
{
  if (hSnapshot)
  {
    DWORD msgSize, msgCount;
    if (GetMailslotInfo( hSnapshot, NULL, &msgSize, &msgCount, NULL))
    {
      PROCESSENTRY32 pe;
      if (msgSize != sizeof(pe) || msgCount == 0)
      {
        SetLastError(ERROR_NO_MORE_FILES);
      }
      else if ( ReadFile( hSnapshot, &pe, sizeof(pe), &msgSize, NULL ))
      {
        if (sizeof(pe) == msgSize)
        {
          if (lppe)
          {
            if (msgSize > lppe->dwSize)
              msgSize = lppe->dwSize;
            CopyMemory( lppe, &pe, msgSize );
            lppe->dwSize = msgSize;
          }
          return TRUE;
        }
      }
    }
  }
  return FALSE;
}

static BOOL WINAPI msrw_Process32First(HANDLE hSnap, LPPROCESSENTRY32 lppe)
{
  return msrw_Process32Next(hSnap, lppe);
}

/* ---------------------------------------------------- */

typedef HANDLE (WINAPI *CreateToolhelp32SnapshotT)(DWORD dwFlags, DWORD pID);
typedef BOOL (WINAPI *Process32FirstT)(HANDLE hSnapshot,LPPROCESSENTRY32 lppe);
typedef BOOL (WINAPI *Process32NextT)(HANDLE hSnapshot, LPPROCESSENTRY32 lppe);

static int __getstubptrs( CreateToolhelp32SnapshotT *create,
                           Process32FirstT *procfirst,
                           Process32NextT *procnext )
{
  static CreateToolhelp32SnapshotT fnCreateToolhelp32Snapshot = 0;
  static Process32FirstT fnProcess32First = 0;
  static Process32NextT fnProcess32Next = 0;
  static long winver = -1L;

  if (winver == -1L)
  {
#undef FORCE_SOMETHING
#if defined(FORCE_TOOLHELP) || defined(FORCE_NTQUERY) || defined(FORCE_PERFCAPS)
  #define FORCE_SOMETHING
#endif
#if !defined(FORCE_SOMETHING) || defined(FORCE_TOOLHELP)
    HMODULE hKernel32 = GetModuleHandle( "kernel32.dll" );
    if (hKernel32 != NULL)
    {
      fnCreateToolhelp32Snapshot = (CreateToolhelp32SnapshotT)
                GetProcAddress(hKernel32, "CreateToolhelp32Snapshot");
      fnProcess32First = (Process32FirstT)
                GetProcAddress(hKernel32, "Process32First");
      fnProcess32Next = (Process32NextT)
                GetProcAddress(hKernel32, "Process32Next");
      if (fnCreateToolhelp32Snapshot &&
          fnProcess32First && fnProcess32Next)
      {
        winver = 2500; /* good for win2k (and win9x) */
        #if defined(TEST_WITH_MAIN) || defined(FORCE_SOMETHING)
        printf("Using real toolhelp\n");
        #endif
      }
    }
#endif
#if !defined(FORCE_SOMETHING) || defined(FORCE_NTQUERY) || defined(FORCE_PERFCAPS)
    if (winver == -1L) /* still not there */
    {
      OSVERSIONINFO osver;
      osver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
      GetVersionEx(&osver);
      if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
      {
        fnCreateToolhelp32Snapshot = NtQuery_CreateToolhelp32Snapshot;
        #if (!defined(FORCE_PERFCAPS))
        if (osver.dwMajorVersion < 4)
        #endif
        {
          fnCreateToolhelp32Snapshot = perfCaps_CreateToolhelp32Snapshot;
        }
        #if defined(TEST_WITH_MAIN) || defined(FORCE_SOMETHING)
        printf("Using %s\n", ((fnCreateToolhelp32Snapshot ==
          NtQuery_CreateToolhelp32Snapshot)?("NtQuerySystemInformation"):
                                            ("Perf Counters")) );
        #endif
        fnProcess32First = msrw_Process32First;
        fnProcess32Next = msrw_Process32Next;
        winver = 2000+(osver.dwMajorVersion*100); /* 2300, 2400 */
      }
    }
#endif
    if (winver == -1L) /* still not there */
    {
      winver = 0; /* not supported */
    }
  }
  if (winver)
  {
    if (create) *create = fnCreateToolhelp32Snapshot;
    if (procfirst) *procfirst = fnProcess32First;
    if (procnext) *procnext = fnProcess32Next;
  }
  return winver;
}

/* ---------------------------------------------------- */

HANDLE WINAPI CreateToolhelp32Snapshot(DWORD dwFlags, DWORD th32ProcessID)
{
  CreateToolhelp32SnapshotT create;
  if (__getstubptrs(&create, 0, 0))
    return (*create)(dwFlags, th32ProcessID);
  return (HANDLE)0;
}

/* ---------------------------------------------------- */

BOOL WINAPI Process32First(HANDLE hSnapshot, LPPROCESSENTRY32 lppe)
{
  if (hSnapshot)
  {
    Process32FirstT procfirst;
    if (__getstubptrs(0, &procfirst, 0))
      return (*procfirst)( hSnapshot, lppe);
  }
  return FALSE;
}

/* ---------------------------------------------------- */

BOOL WINAPI Process32Next(HANDLE hSnapshot, LPPROCESSENTRY32 lppe)
{
  if (hSnapshot)
  {
    Process32NextT procnext;
    if (__getstubptrs(0, 0, &procnext))
      return (*procnext)( hSnapshot, lppe);
  }
  return FALSE;
}

/* ---------------------------------------------------- */

#if defined(TEST_WITH_MAIN)
int main(int argc, char **argv)
{
  /* display list of found processes. if an argument is provided, show
     matches with an '='. Show our own process with an '@'.

     Name matching: if any component (path,name,extension) of the provided
     name OR the found name is not available, those match anything.
  */

  HANDLE hSnapshot = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
  if (!hSnapshot)
  {
    printf("Unable to get snapshot\n");
  }
  else
  {
    DWORD ourpid = GetCurrentProcessId();
    unsigned int num_found = 0, num_matched = 0;
    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(pe);

    if (Process32First(hSnapshot, &pe))
    {
      unsigned int basenamepos = 0, basenamelen = 0, suffixlen = 0;
      if (argc > 1)
      {
        char *procname = argv[1];
        basenamelen = basenamepos = strlen(procname);
        while (basenamepos > 0)
        {
          basenamepos--;
          if (procname[basenamepos] == '\\' ||
              procname[basenamepos] == '/' ||
              procname[basenamepos] == ':')
          {
            basenamepos++;
            basenamelen-=basenamepos;
            break;
          }
        }
        if (basenamelen > 3)
        {
          if (lstrcmpi( &procname[(basenamepos+basenamelen)-4],".com" )==0 ||
              lstrcmpi( &procname[(basenamepos+basenamelen)-4],".exe" )==0 )
          {
            suffixlen = 3;
            basenamelen -=4;
          }
        }
      }

      do /* while Process32[First|Next] */
      {
        //if (pe.szExeFile[0])
        {
          int cmpresult = -1;

          if (argc > 1 && pe.szExeFile[0])
          {
            char *foundname = pe.szExeFile;
            char *procname = argv[1];
            unsigned int len = strlen( foundname );
            unsigned int fbasenamelen = len;

            while (len > 0)
            {
              len--;
              if (foundname[len]=='\\' ||
                  foundname[len]=='/' ||
                  foundname[len]==':')
              {
                len++;
                fbasenamelen-=len;
                break;
              }
            }

            /*if no path was provided, then allow match if the rest is equal*/
            if (basenamepos == 0) /* no path in template */
            {
              foundname += len; /* then skip dir in foundname */
            }
            else if (len == 0) /*dir in template, but no dir in foundname */
            {
              procname += basenamepos; /* then skip dir in template */
            }
            cmpresult = lstrcmpi( procname, foundname );

            if ( cmpresult )
            {
              /* if either template OR foundname have no suffix, (but
                 not both, which will have been checked above) then
                 allow a match if the basenames (sans-suffix) are equal.
              */
              int fsuffixlen = 0;
              if (fbasenamelen > 3)
              {
                if ( lstrcmpi( &foundname[fbasenamelen-4], ".exe" ) == 0
                  || lstrcmpi( &foundname[fbasenamelen-4], ".com" ) == 0 )
                {
                  fsuffixlen = 3;
                  fbasenamelen -= 4;
                }
              }
              if (suffixlen != fsuffixlen && basenamelen == fbasenamelen)
              {
                cmpresult = memicmp( foundname, procname, basenamelen );
              }
            }
            if (cmpresult == 0)
              num_matched++;
          } /* argc > 1 */

          num_found++;

          /*
            On NT, irrespective of which method was used to obtain the data
            [real toolhelp, performance counters, NtQuerySystemInformation],
            the pe.cntUsage and pe.th32DefaultHeapID are always zero and
            pe.szExeFile has no dirspec. In addition, if using performance
            counters, the pe.cntThreads, pe.th32ParentProcessID and
            pe.pcPriClassBase fields are zero as well and pe.szExeFile
            usually does not have an extension.
          */
          printf("%c %c %08x '%s'\n"
          "             cnt=%u, heap=%p, #thr=%u, ppid=%p, priobase=0x%x\n",
                  ((cmpresult == 0)?('='):(' ')),
                  ((pe.th32ProcessID == ourpid)?('@'):(' ')),
                  pe.th32ProcessID,
                  pe.szExeFile,
                  pe.cntUsage,
                  pe.th32DefaultHeapID,
                  pe.cntThreads,
                  pe.th32ParentProcessID,
                  pe.pcPriClassBase );

        } /* if (pe.szExeFile[0]) */
      } while (Process32Next(hSnapshot, &pe));
    } /* if (Process32First(hSnapshot, &pe)) */
    CloseHandle( hSnapshot );

    printf("\nTotal found: %u\n", num_found );
    if (argc > 1)
      printf("Matched: %u\n", num_matched );
  }
  return 0;
}
#endif

