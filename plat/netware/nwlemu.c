/* 
 * Emulation functions/stubs for portability across NetWare versions.
 * All functions here are CLIB safe (don't require context). 
 * References to GetNLMHandle() are to a CLIB safe version therof in
 * prelude [NetWare 3x's GetNLMHandle() is otherwise not CLIB safe].
 *
 * If using #PATCH_SELF, care has to be taken to ensure thread safety:
 * where possible, stubs are initialized by 'family', so a call to
 * any member of that family initializes all.
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwlemu.c,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
*/

//#define DEMO_TIMEFUNC
//#define TIMEFUNC_FORCE_PIT

/* #define PATCH_SELF */ /* patch the stubs with 'jmp xxxx' when possible */
#include <process.h> /* GetNLMHandle(), ThreadSwitch() */
#include "nwlemu.h" /* ourselves */

/* ===================================================================== */

#ifdef __cplusplus
extern "C" {
#endif
extern void *ImportPublicSymbol( int nlmHandle, const char *lname ); /* krnl/a3112 */
extern int UnImportPublicSymbol( int nlmHandle, const char *lname ); /* krnl/a3112 */
extern unsigned int GetNLMHandleFromPrelude(void);                   /* nwpre3x.c */
extern int GetFileServerName( int conn, const char *buffer /* min49 chars */ );
extern void OutputToScreen( int scrhandle, const char *fmt, ... );
extern void ConsolePrintf( /* const */ char *fmt, ... ); /* avoid nw*.h */
extern void ThreadSwitch( void );
extern unsigned long GetCurrentTicks(void);
extern unsigned long time( unsigned long* ); /* doesn't need ctx as used (3.x) */
#ifdef __cplusplus
}
#endif

/* ===================================================================== */

#if defined(PATCH_SELF)
static void __patchme( void *fxn_to_patch, void *abs_jump_to_addr)
{
  register char *fxn = (char *)fxn_to_patch;
  #if 0 /* don't do this! results in 'jmp 0000:offset' */
  register char *offs = (char *)abs_jump_to_addr;
  *fxn++ = 0xEA; /* jmp (absolute, long) */
  *((char **)fxn) = offs;
  #else
  long offs = (((char *)abs_jump_to_addr)-(fxn+5));
  *fxn++ = 0xE9; /* jmp (relative, long) */
  *((long *)fxn) = offs;
  #endif
  return;  
}
#endif

/* ===================================================================== */

unsigned int GetNLMHandle(void)      
{
  /* CLIB doesn't read the NLM handle from the PCB (thread structure),
     but instead, uses RunningProcess->threadGroupID->NLMID->nlmHandle.
     So, to stay clean of context issues, we get it from our custom prelude.
     (bug alert: CLIB 3x does not check SCGT (threadgroup) or SCN (NLMID) 
     validity when retrieving the NLM handle.)
  */
  return GetNLMHandleFromPrelude();
}                           

/* ===================================================================== */

static void *__ImUnimportSymbol(int NLMHandle, char *cname,register doimport)
{
  if (cname)
  {
    char lname[64]; 
    unsigned int cnamelen = 0;
    //if (NLMHandle == 0)
    //  NLMHandle = GetNLMHandle();
    while (cname[cnamelen] && (cnamelen+1) < sizeof(lname))
    {
      register char c = cname[cnamelen];
      lname[++cnamelen] = c;
    }
    if (cnamelen > 0 && (cnamelen+1) < sizeof(lname))
    {
      lname[0] = (char)cnamelen;
      if (doimport)
        return ImportPublicSymbol( NLMHandle, lname );
      return (void *)UnImportPublicSymbol( NLMHandle, lname );
    }
  }
  return (void *)0;    
}

void *ImportSymbol(int NLMHandle, char *cname) /* all important. */
{                                              /* CLIB.NLM >=4.x */
  return __ImUnimportSymbol( NLMHandle, cname, 1 );
}  

int UnimportSymbol(int NLMHandle, char *cname) /* CLIB.NLM >=4.x */
{
  return (int)__ImUnimportSymbol( NLMHandle, cname, 0 );
}  

/* ===================================================================== */

unsigned long GetSystemConsoleScreen(void)
{
  static unsigned long (*_systemConsoleScreen);
  static unsigned long (*_GetSystemConsoleScreen)(void) = 
                                            ((unsigned long (*)(void))0x01);
  if (_GetSystemConsoleScreen == ((unsigned long (*)(void))0x01))
  {
    _systemConsoleScreen = (unsigned long (*))
                       ImportSymbol(GetNLMHandle(),"systemConsoleScreen");
    _GetSystemConsoleScreen = (unsigned long (*)(void))
                       ImportSymbol(GetNLMHandle(),"GetSystemConsoleScreen");
    #if defined(PATCH_SELF)
    if (_GetSystemConsoleScreen)
      __patchme( GetSystemConsoleScreen, _GetSystemConsoleScreen);
    #endif
  }
  if (_GetSystemConsoleScreen)
    return (*_GetSystemConsoleScreen)();
  if (_systemConsoleScreen)
    return (*_systemConsoleScreen);
  return 0;
}

/* ===================================================================== */

void RingTheBell(void)  /* >=312 */
{
  static void (*_RingTheBell)(void) = ((void (*)(void))0x01);
  /* static void (*_StartAndStopBell)(void); static void (*_StopBell)(void); */
  
  if (_RingTheBell == ((void (*)(void))0x01))
  {
    _RingTheBell = (void (*)(void))ImportSymbol(GetNLMHandle(),"RingTheBell");
    #if defined(PATCH_SELF)
    if (_RingTheBell)
      __patchme( RingTheBell, _RingTheBell);
    #endif
  }
  if (_RingTheBell)
    (*_RingTheBell)();
  else
  {
    unsigned long sysScreen = GetSystemConsoleScreen();
    if (sysScreen) OutputToScreen( sysScreen, "\a" );
  }
  return;
}

/* ===================================================================== */

unsigned long GetCurrentTime(void)
{
  static unsigned long (*_GetCurrentTime)(void) = 
                                      ((unsigned long (*)(void))0x01);
  static unsigned long (*_currentTime);

  if (_GetCurrentTime == ((unsigned long (*)(void))0x01))
  {
    unsigned int nlmHandle = GetNLMHandle();
    void *lvect = 0, *fvect = ImportSymbol( nlmHandle, "GetCurrentTime" );
    if (!fvect) 
      fvect = GetCurrentTicks;
    if (GetNumberOfRegisteredProcessors() < 2)
      lvect = ImportSymbol( nlmHandle, "currentTime" );
    _currentTime = (unsigned long (*))lvect;
    _GetCurrentTime = (unsigned long (*)(void))fvect;
    #if defined(PATCH_SELF)
    __patchme( GetCurrentTime, _GetCurrentTime);
    #endif
  }
  if (_currentTime)
    return (*_currentTime);
  return (*_GetCurrentTime)();
}  

/* ===================================================================== */

static int __get_cpu_and_flags(void) /* [0..3]=step,[4..7]=model,[8..11]=family */
{                                    /* [12..15(13?)]=type(Overdrive etc), */
  static int cpu_info = -1;          /* [16..32]=low 16 bits of flags */

  if (cpu_info == -1)
  {
    _asm 
    {
      ; an interrupt could change the AC bit, so loop 'n'
      ; times, incrementing a state counter for each state
      pushfd                  ; save flags from damage
      xor     edx,edx         ; our counter and state
      pushfd                  ; copy EFLAGS
      pop     ecx             ; ... into ecx
      _386next:
      mov     eax, ecx        ; copy original EFLAGS
      xor     eax, 40000h     ; toggle AC bit (1<<18) in eflags
      push    eax             ; copy modified eflags
      popfd                   ; ... to EFLAGS
      pushfd                  ; copy (possible modified) EFLAGS
      pop     eax             ; ... back to eax
      cmp     eax, ecx        ; will be 386 if no change
      setz    al              ; set to one if 386, else zero
      setnz   ah              ; set to zero if 386, else one
      add     dx,ax           ; add to totals
      mov     al,dl           ; copy our 'is386' count
      add     al,dh           ; add the 'not386' count
      cmp     al,31           ; done 31 checks?
      jb      _386next        ; continue looping if not
      popfd                   ; restore saved flags
      cmp     dl,dh           ; 'is386' count less than 'not386'?
      mov     eax,300h        ; assume its a 386
      jnb     _end            ; proceed if is386 >= not386

      pushfd                  ; save flags from damage
      pushfd                  ; copy EFLAGS
      pop     ecx             ; ... into ecx
      mov     eax, ecx        ; copy original flags
      xor     eax, 200000h    ; try to toggle ID bit
      push    eax             ; copy modified eflags
      popfd                   ; ... to EFLAGS
      pushfd                  ; push back flags
      pop     eax             ; ... into eax
      popfd                   ; restore saved flags
      and     ecx, 200000h    ; clear all but the ID bit
      and     eax, 200000h    ; clear all but the ID bit
      cmp     eax, ecx        ; was it changed?
      mov     eax, 400h       ; assume no change
      jz      _end            ; proceed if no change

      mov     eax,1           ; cpuid 1
      push    ebx             ; cpuid trashes ebx 
      .586                    ;
      cpuid                   ; db 0x0f, 0xa2
      pop     ebx             ; restore ebx
      shl     edx,16          ; features
      and     eax,0fffh       ; keep only family/model/stepping
      or      eax,edx         ; add low 16 bits of feature flags
      _end:
      mov     cpu_info,eax
    } /* _asm */
    #ifdef DEMO_TIMEFUNC
    ConsolePrintf("\r__get_cpu_and_flags() => 0x%08x\r\n", cpu_info );
    #endif
  }
  return cpu_info;
}
//#define __have_tsc() (!__get_cpu_and_flags()) 
#define __have_tsc() ((__get_cpu_and_flags() & 0x00040000ul)!=0)

/* ------------------------------------------------------------------ */

#if defined(__WATCOMC__)
  #pragma pack(push,8)
  typedef struct { volatile unsigned int lock; } spinlock_t;
  #pragma pack(pop)
  #define SPIN_LOCK_UNLOCKED { 0 }
  void spinlock_unlock( spinlock_t * );
  int spinlock_trylock( spinlock_t * ); /* returns -1 if lock already set, else zero */

  #pragma aux spinlock_unlock =  "xor eax,eax"    \
                                 "xchg [edx],eax" \
                                 "xor eax,eax"    \
                                 parm [edx] modify exact [eax];
  #pragma aux spinlock_trylock = "xor eax,eax"    \
                                 "inc eax"        \
                                 "xchg [edx],eax" \
                                 "cmp eax,1"      \
                                 "cmc"            \
                                 "sbb eax,eax"    \
                                 parm [edx] value [eax] modify exact [eax];
#elif defined(__GNUC__)
  typedef struct { volatile unsigned int lock; } spinlock_t;
  #define SPIN_LOCK_UNLOCKED { 0 }
  typedef struct { unsigned long a[100]; } __dummy_lock_t;
  #define __dummy_lock(lock) (*(__dummy_lock_t *)(lock))
  #define spin_unlock(spinlock_t *lock) \
     __asm__ __volatile__( "lock ; btrl $0,%0" :"=m" (__dummy_lock(lock)))
  extern __inline__ int spin_trylock( volatile spinlock_t *lock) 
  {
    register int result; /* -1 if lock already set, else zero */
    __asm__ __volatile__( "lock ; btsl $0,%1 ; sbbl %0,%0" 
                         :"=r" (result), "=m" (__dummy_lock(lock)));  
    return result;
  }
#endif

/* ------------------------------------------------------------------ */

static unsigned long __3x_get_ticks_and_pit(unsigned long *ticksptr,
                                            unsigned long *clksptr)
{
  /* Things I know about what the 3.x (and 4.11) loader does with the PIT:
     - During init it sets timer to mode 2 (rate generator) [since 
       mode 3 (square wave) doesn't count linearly].
     - rate is set to default 0x0000 (65536) (18.2 ticks per sec).
     - time and date fields are filled in FROM CMOS.
       fields are assumed to be BCD. timer edge is checked before 
       inp'ing time/date, but *not* between each port. Skerrie.
     - tick int handler is in the loader. Increments DWORD CurrentTime;
     - *BUG* 'CurrentTime' may go backwards (and even become negative) if 
       the time-of-day is changed. This was fixed in an early fixpack. 
       We don't check or compensate for it here. 
     - GetHighResolutionTimer() disables interrupts on entry and then 
       re-enables interrupts (on return) without checking if interrupts
       had been previously disabled by the caller. Appears to be by design.
     - GetSuperHighResolutionTimer() doesn't guard against rollover. 
       the PIT's count can be significantly offset from interrupt time,
       (obviously, the PIT never waits for the interrupt to be handled)
       so its possible to end up with sequential reads that look 
       like this: 0:FFFA, 0:0005, 1:000A (all 5 clks apart)
       the latency on my 486/66 is 219. ***** DONT USE GetSuperHi *****
  */
  static spinlock_t spl = SPIN_LOCK_UNLOCKED;
  static unsigned long lastticks = 0, lastclks = 0, lastwrap = 0;
  unsigned long thisclks, thisticks, thiswrap; 

  while (spinlock_trylock(&spl))
    ;
  thisticks = GetCurrentTime();
  thisclks = lastclks;
  if (clksptr)
  {
    unsigned long prevticks = 0;
    do    
    {
      prevticks = thisticks;
      _asm xor  eax, eax
      _asm mov  dx, 43h 
      _asm out  dx, al  /* mode control, counter 0, read */
      _asm in   al, 40h  /* read lsb */
      _asm mov  ah, al
      _asm in   al, 40h /* read msb */
      _asm xchg ah, al
      _asm mov  thisclks,eax
      thisticks = GetCurrentTime();
    } while (prevticks != thisticks);

    /* PITRATE 0x10000 is the assumption that the pit is 
       programmed for the default rate of 0x10000. 
       3.12 and below, 4.11: yes, confirmed from loader source.
       3.2 or 4.0-4.10 or >=4.2: unknown
    */
    thisclks = (0x10000 - thisclks); /* counter counts down */
  }

  thiswrap = lastwrap;
  if (thisticks < lastticks) /* Potential bug on original 3.1x here */
    lastwrap = ++thiswrap;   /* See comments above */
  else if (thisticks == lastticks && thisclks < lastclks)
    thisclks = lastclks;
  lastclks = thisclks;
  lastticks = thisticks;
  spinlock_unlock(&spl);

  if (clksptr) *clksptr = thisclks;
  if (ticksptr) *ticksptr = thisticks;
  return thiswrap;
}  

/* ------------------------------------------------------------------ */

static void __3x_convert_ticks_and_pit(unsigned long wrap_count,
                                       unsigned long ticks,
                                       unsigned long pit,
                                       unsigned long *secsptr,
                                       unsigned long *nsecsptr )
{
  #define TICKS_IN_90_MIN (0x1800B0ul >> 4) /* 0x1800B (98315) */
  #define SECS_IN_90_MIN  (86400ul >> 4)  /* 5400 (0x1518) */
  /* 
  ** the reason nothing else is defined is because the asm code will
  ** need adjustment even if one of the defines changes. The assumptions
  ** in effect are as follows:
  */
  /* 1 tick = 0.054,925,494,583,735,950,770,482,632,355,185 secs (exactly) */
  /*          (obtained from 86,400,000,000,000ns/day / 1800B0 ticks/day)  */
  /* 1 pit  = 0.000,000,838,096,536,006,713,116,004,678,838 secs (enough!) */
  /*          (obtained from [default rate] 1 tick / 0x10000)              */
  /*          if the rate is not the default, then scale it, eg:           */
  /*          if rate is twice the default then pass (pitcount * 2)        */
  /* Unlike ticks which we compute exactly, we only compute the pitcount   */
  /* to 0.000,000,838,096,536,006th/sec. The rest is worthless anyway.     */

  unsigned long days, secs, nsecs;

  days = ticks / 0x1800B0ul;
  ticks %= 0x1800B0ul;
  days += wrap_count * 0xAAAul;     /* 0x100000000 / 0x1800B0 */
  ticks += wrap_count * 0x8AB20ul;  /* 0x100000000 % 0x1800B0 */
  ticks += pit / 0x10000ul;
  pit %= 0x10000ul;
  nsecs = 0;
  secs = (86400ul * days) + (SECS_IN_90_MIN * (ticks / TICKS_IN_90_MIN));
  ticks %= TICKS_IN_90_MIN;

  /* +++++++++++++++++++++++++ */

  _asm mov  eax,pit       /* get pit count out of the way. Don't need ... */
  _asm xor  edx,edx       /* ... higher res than e-7 (think 'clock jitter')*/
  _asm mov  ecx,05C105C6h /* 0.000,000,838,[096,536,006],713,116,004,678,838*/
  _asm mul  ecx           /* 0x10000 => 6,326,583,689,216 (5C1:05C6000) */
  _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
  _asm div  ecx           /* yes, truncate, don't round */
  _asm push eax           /* 0x10000 => 6326 */
  _asm mov  eax,pit
  _asm xor  edx,edx
  _asm mov  ecx,346h      /* 0.[000,000,838],096,536,006,713,116,004,678,838*/
  _asm mul  ecx           /* 0x1000 => 54,919,168 (0:3460000) */
  _asm pop  ecx
  _asm add  eax,ecx       /* 54,919,168 + 6,326 => 54,925,494 */
  _asm adc  edx,0
  _asm add  nsecs,eax     /* 0x1000 => 0.054,925,494 secs */

  /* +++++++++++++++++++++++++ */

  _asm mov  eax,ticks     /* now do long multiplication with (ticks%0x1800B)*/
  _asm xor  edx,edx
  _asm mov  ecx,56b71h    /* 0.054,925,494,583,735,950,770,482,632,[355,185]*/
  _asm mul  ecx           /* 0x1800B => 34,920,013,275 (8:21651DDB) */
  _asm mov  ecx,0F4240h   /* 1,000,000 */
  _asm div  ecx           /* 0x1800B => 34920.013275 (33DB:8868) */
                          /* 0x1800B0 => 558720.2124 */
  _asm push eax
  _asm mov  eax,ticks
  _asm xor  edx,edx
  _asm mov  ecx,2deca1c8h /* 0.054,925,494,583,735,950,[770,482,632],355,185*/
  _asm mul  ecx           /* 0x1800B => 75,749,999,965,080 (44E4:EBD6F398) */
  _asm pop  ecx
  _asm add  eax,ecx       /* 75,749,999,965,080 + 34920 => 75750000000000 */
  _asm adc  edx,0
  _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
  _asm div  ecx           /* 0x1800B => 75750.000000000 (0:127E6) */
                          /* 0x1800B0 => 1212000.000000000 */
  _asm push eax
  _asm mov  eax,ticks
  _asm xor  edx,edx
  _asm mov  ecx,22CB1A8Eh /* 0.054,925,494,[583,735,950],770,482,632,355,185*/
  _asm mul  ecx           /* 0x1800B => 57,389,999,924,250 (3432:268F241A) */
  _asm pop  ecx
  _asm add  eax,ecx       /* 57,389,999,924,250 + 75750 => 57390000000000 */
  _asm adc  edx,0
  _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
  _asm div  ecx           /* 0x1800B => 57390.000000000 (0:E02E) */
                          /* 0x1800B0 => 918240.000000000 */
  _asm push eax
  _asm mov  eax,ticks    
  _asm xor  edx,edx
  _asm mov  ecx,34618b6h  /* 0.[054,925,494],583,735,950,770,482,632,355,185*/
  _asm mul  ecx           /* 0x1800B => 5,399,999,942,610 (4E9:49140FD2) */
  _asm pop  ecx
  _asm add  eax,ecx       /* 5,399,999,942,610 + 57390 => 5400000000000 */
  _asm adc  edx,0
  _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
  _asm div  ecx           /* 0x1800B => 5400.000000000 (0:1518) */
                          /* 0x1800B0 => 86399.[99908176+918240=100826416] */
  _asm add  secs,eax
  _asm add  nsecs,edx

  /* +++++++++++++++++++++++++ */

  /* adjust secs with nanosec overflow ala 'if (nsecs>=x){secs++;nsecs-=x};'*/
  _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
  _asm cmp  nsecs,ecx
  _asm cmc
  _asm sbb  eax,eax
  _asm mov  edx,eax
  _asm and  eax,1
  _asm and  edx,ecx
  _asm add  secs,eax
  _asm sub  nsecs,edx

  /* +++++++++++++++++++++++++ */

  if (secsptr) *secsptr = secs;
  if (nsecsptr) *nsecsptr = nsecs;
  return;
}                                    

/* ------------------------------------------------------------------ */

unsigned int read_cmos_index(unsigned int index);
/* return value of port in al, if had to wait, ah is != 0 */
#pragma aux read_cmos_index = \
               "xor  eax,eax" \
               "_chk1:"       \
               "inc  ah"      /* ah=ah+1 */                               \  
               "cmp  ah,1"    /* set carry if it wrapped (ah is now 0) */ \
               "sbb  al,al"   /* make al=0xff if it wrapped */            \
               "and  al,2"    /* make al=2    if it wrapped */            \
               "or   ah,al"   /* make ah=2    if it wrapped */            \ 
               "mov  al,0Ah"  /* CMOS 0Ah - RTC - STATUS REGISTER A */    \
               "out  70h,al"  /* request to read from register A 0Ah */   \
               "db 0xEB,0x00" /* jmp short $+2 */                         \
               "in   al,71h"  /* if bit 7 is 1 then "time update ... */   \
               "test al,80h"  /* ... cycle in progress", and ... */       \
               "jnz  _chk1"   /* ... data ouputs are undefined. */        \
               "mov  al,dl"   /* index to read */                         \
               "out  70h,al"  /* request read */                          \
               "db 0xEB,0x00" /* jmp short $+2 */                         \
               "in   al,71h"  /* read the value */                        \
               "cmp  ah,2"    /* number of times we had to wait > 1? */   \ 
               "cmc"          /* set carry if so */                       \
               "sbb  ah,ah"   /* ah=0xff if we had to wait, else zero */  \
               parm [edx] value [eax] modify exact [eax] nomemory;

typedef union {
        struct { unsigned long lo,hi; } l; 
        struct { unsigned long quot,rem; } div;
        unsigned __int64       i;
} quad_t;        

void read_tsc( quad_t *result );
#pragma aux read_tsc =            \
               ".586"             \
               "rdtsc"            \
               "mov  [ebx+0],eax" \
               "mov  [ebx+4],edx" \
               parm [ebx] modify [eax edx];
void sub64( quad_t *lo, quad_t *hi, quad_t *result );
#pragma aux sub64 =               \
              "push  dword ptr [eax+4]" \
              "mov   eax,[eax+0]" \
              "sub   eax,[edx+0]" \
              "mov   [ebx+0],eax" \
              "pop   eax"         \
              "sbb   eax,[edx+4]" \
              "mov   [ebx+4],eax" \
              parm [edx] [eax] [ebx] modify [eax];
void add64( quad_t *, quad_t *, quad_t *result );
#pragma aux add64 =               \
              "push  dword ptr [eax+4]" \
              "mov   eax,[eax+0]" \
              "add   eax,[edx+0]" \
              "mov   [ebx+0],eax" \
              "pop   eax"         \
              "adc   eax,[edx+4]" \
              "mov   [ebx+4],eax" \
              parm [edx] [eax] [ebx] modify [eax];
void mul32( unsigned long plicand, unsigned long plier, quad_t *result );
#pragma aux mul32 =               \
              "push  ebx        " \
              "mov   ebx,edx    " \
              "xor   edx,edx    " \
              "mul   ebx        " \
              "pop   ebx        " \
              "mov   [ebx+0],eax" \
              "mov   [ebx+4],edx" \
              parm [eax] [edx] [ebx] modify [eax edx];
void div64( quad_t *, unsigned long isor, 
            unsigned long *quot, unsigned long *rem);
#pragma aux div64 =               \
              "push edx         " \
              "mov  edx,[eax+0] " \
              "mov  eax,[eax+4] " \
              "xchg eax,edx     " \
              "div  ecx         " \
              "mov  [ebx],eax   " \
              "mov  eax,edx     " \
              "pop  edx         " \
              "mov  [edx],eax   " \
              parm [eax] [ecx] [ebx] [edx] modify [eax];

/* ------------------------------------------------------------------ */

static int __is_showing_timer = 0;

/* get TSC count per second/per tick. not avail => tsc_res will be zero */
static void __get_tsc_per_X(quad_t *tsc_res, int per_what)
{                  
  quad_t tsc_lo, tsc_hi;
  unsigned long per1, per2, per3;
  tsc_res->l.hi = tsc_res->l.lo = 0;

  if (per_what == 's') /* per second */
  {
    static unsigned long saved_hi = 0xfffffffful, saved_lo = 0xfffffffful;
    tsc_res->l.hi = saved_hi; tsc_res->l.lo = saved_lo;

    if (tsc_res->l.hi == 0xfffffffful && tsc_res->l.lo == 0xfffffffful)
    {
      tsc_res->l.hi = tsc_res->l.lo = 0;
      /* [0..3]=step,[4..7]=model,[8..11]=family,[16..32]=feature flags */
      if (__have_tsc()) /* have TSC? */
      {          
        ThreadSwitch();
        per1 = (read_cmos_index(0 /* seconds index */) & 0xff);
        while (per1 == (per2 = (read_cmos_index(0) & 0xff)))
          ;
        per3 = GetCurrentTime();
        read_tsc(&tsc_lo);  
        while (per2 == (per1 = (read_cmos_index(0) & 0xff)))
          ;
        read_tsc(&tsc_hi);
        per3 = GetCurrentTime() - per3;
        ThreadSwitch();
        sub64( &tsc_lo, &tsc_hi, tsc_res );
      } /* have TSC */
      saved_hi = tsc_res->l.hi; saved_lo = tsc_res->l.lo;
      if (__is_showing_timer)
      {
        ConsolePrintf("\r__get_tsc_per_X(): secs_rate=%u.%u (%u %u %u)\r\n", tsc_res->l.hi, tsc_res->l.lo, per2, per1, per3 );
        ThreadSwitch();
      }
    } /* (tsc_res->l.hi == 0xfffffffful && tsc_res->l.lo == 0xfffffffful) */
  } /* if (per_what == 's') */
  else if (per_what == 't') /* per tick */
  {
    static unsigned long saved_hi = 0xfffffffful, saved_lo = 0xfffffffful;
    tsc_res->l.hi = saved_hi; tsc_res->l.lo = saved_lo;

    if (tsc_res->l.hi == 0xfffffffful && tsc_res->l.lo == 0xfffffffful)
    {
      tsc_res->l.hi = tsc_res->l.lo = 0;
      /* [0..3]=step,[4..7]=model,[8..11]=family,[16..32]=feature flags */
      if (__have_tsc()) /* have TSC? */
      {          
        unsigned int count;
        GetCurrentTime(); /* arm the function if it hasn't been done yet */
        for (count = 1; count < 9; count++)
        {
          per3 = 0; per1 = 1;
          while (per3 < per1)
          {
            ThreadSwitch();
            per2 = per1 = GetCurrentTime();
            while (per1 == per2)
              per1 = GetCurrentTime();
            read_tsc(&tsc_lo); 
            per2 = count+per1;
            per3 = per1; 
            while (per3 < per2)
              per3 = GetCurrentTime();
            read_tsc(&tsc_hi); 
          }
          ThreadSwitch();
          per3 -= per1;
          if (per3 != 0)
          {
            sub64( &tsc_lo, &tsc_hi, &tsc_hi );
            if (__is_showing_timer)
            {
              ConsolePrintf("\r__get_tsc_per_X(): rate in %u ticks: %u.%u\r\n", per3, tsc_hi.l.hi, tsc_hi.l.lo);
              ThreadSwitch();
            }
            if (tsc_hi.l.hi >= per3) /* would result in a divide overflow? */
              break;
            per1 = tsc_hi.l.lo;
            if (per3 != 1)
              div64( &tsc_hi, per3, &per1, &per2 );
            tsc_res->l.lo = per1;
            tsc_res->l.hi = 0;
            if (per3 > count)
              count = per3;
          }
        } /* for (;;) */
      } /* have TSC */
      saved_hi = tsc_res->l.hi; saved_lo = tsc_res->l.lo;
      if (__is_showing_timer)
      {
        ConsolePrintf("\r__get_tsc_per_X(): rate in 1 tick: %u.%u\r\n", tsc_res->l.hi, tsc_res->l.lo);
        ThreadSwitch();
      }
    } /* (tsc_res->l.hi == 0xfffffffful && tsc_res->l.lo == 0xfffffffful) */
  } /* 's' or 't' */
  return;
}

/* ------------------------------------------------------------------ */

//extern "C" double fmod(double,double);

/* get secs and nanosecs either from the TSC or from the clock chip */
static unsigned long __get_secs_and_nanosecs(unsigned long *nanosecs)
{
  static unsigned long tsc_tick_rate = -1, tsc_secs_rate = -1;
  unsigned long secs; quad_t tsc;

  /* determine source */
  if (tsc_tick_rate == -1)
  {
    __get_tsc_per_X( &tsc, 's');

    /* < 4.2Ghz? won't overflow on secs calc? */ 
    if (tsc.l.hi == 0 && tsc.l.lo != 0)
    {
      tsc_secs_rate = tsc.l.lo;
      tsc_tick_rate = 0;
      #ifdef DEMO_TIMEFUNC
      ConsolePrintf("\r__get_secs_and_nanosecs(): tsc_per_sec=%u (0x%08x)\r\n", tsc.l.lo, tsc.l.lo );
      #endif
    }
    else /* try ticks, limit is ~76.44Ghz */
    {
      __get_tsc_per_X( &tsc, 't');
      if (tsc.l.hi == 0 && tsc.l.lo != 0)
      {
        tsc_secs_rate = 0;
        tsc_tick_rate = tsc.l.lo;
        #ifdef DEMO_TIMEFUNC
        ConsolePrintf("\r__get_secs_and_nanosecs(): tsc_per_tick=%u (0x%08x)\r\n", tsc.l.lo, tsc.l.lo );
        #endif
      }
      else  /* have to use PIT */
      {
        #ifdef DEMO_TIMEFUNC
        ConsolePrintf("\runable to use TSC. Using PIT instead\r\n");
        #endif
        tsc_secs_rate = 0;
        tsc_tick_rate = 0;
      }
    }
    #ifdef TIMEFUNC_FORCE_PIT
    ConsolePrintf("\rForcing PIT...\r\n");
    tsc_secs_rate = 0;
    tsc_tick_rate = 0;
    #endif
  }

  /* ++++++++++ source determined +++++++ */

  if (tsc_secs_rate != 0) /* use TSC straight to secs/nanosecs */
  {                       
    unsigned long nsecs = 0;
    read_tsc(&tsc); 

    if (__is_showing_timer)
      ConsolePrintf("\r1) hi:lo = %u %u (%08x %08x)\r\n",tsc.l.hi,tsc.l.lo,tsc.l.hi,tsc.l.lo);

    if (tsc_tick_rate != 0) /* TSC with clock rate > 4.2Ghz */
    {
      double r = (((double)tsc_secs_rate)*4294967296.0)+((double)tsc_tick_rate);
      double d = (((double)tsc.l.hi)*4294967296.0)+((double)tsc.l.lo);
      secs = (unsigned long)(d / r);
      if (nanosecs)
      {
        d -= ((double)secs) * r;
        d *= (double)1000000000ul;
        nsecs = (unsigned long)(d / r);
        *nanosecs = nsecs;
      }
    }
    else /* TSC with clock rate <= 4.2Ghz */
    {
      /* 0x1:00000000 secs == tsc_secs_rate << 32 */
      tsc.l.hi %= tsc_secs_rate; /* 'tsc %= (0x1:00000000 secs * tsc_rate)' */

      div64( &tsc, tsc_secs_rate, &secs, &nsecs );

      if (__is_showing_timer)
        ConsolePrintf("\r2) secs:lo = %u %u (%08x %08x)\r\n",secs,tsc.l.lo,secs,tsc.l.lo);

      if (nanosecs)
      {
        mul32( nsecs, 1000000000ul, &tsc ); 
        div64( &tsc, tsc_secs_rate, nanosecs, &nsecs /* dummy */ );
        //ConsolePrintf("\r3) %u.%08u\r\n",secs,*nanosecs);
      }
    }
  }
  else if (tsc_tick_rate != 0) /* TSC: > 4.2Ghz */ 
  {                            /* get as ticks/clks, then convert */
    unsigned long ticks_hi32, ticks_lo32, clks;
    read_tsc(&tsc); 

    //ConsolePrintf("\r1) hi:lo = %u:%u, years=%u\r\n",tsc.l.hi,tsc.l.lo,ticks_hi32);
    /* 0x1:00000000 ticks == tsc_tick_rate << 32 */
    ticks_hi32 = tsc.l.hi / tsc_tick_rate;
    tsc.l.hi %= tsc_tick_rate;
    //ConsolePrintf("\r2) hi:lo = %u:%u, years=%u\r\n",tsc.l.hi,tsc.l.lo,ticks_hi32);
    div64( &tsc, tsc_tick_rate, &ticks_lo32, &tsc.l.lo);
    tsc.l.hi = tsc.l.lo >> 16;  /* \__ tsc <<= 16 */
    tsc.l.lo <<= 16;            /* /              */
    div64( &tsc, tsc_tick_rate, &clks, &tsc.l.lo /* dummy */);

    __3x_convert_ticks_and_pit(ticks_hi32, ticks_lo32, clks, &secs, nanosecs);
  }
  else /* use the PIT */
  {
    unsigned long ticks_hi32, ticks_lo32, clks;
    ticks_hi32 = __3x_get_ticks_and_pit(&ticks_lo32, &clks);
    //ConsolePrintf("\r3) hi:lo = %u:%u.%u\r\n", ticks_hi32, ticks_lo32, clks );
    __3x_convert_ticks_and_pit(ticks_hi32, ticks_lo32, clks, &secs, nanosecs);
  }
  return secs;
}


/* ------------------------------------------------------------------ */

unsigned long read_tsc_lo(void);
#pragma aux read_tsc_lo = ".586p" \
                          "rdtsc" \
                       value [eax] modify exact [eax edx] nomemory;

#if 0
static void __calibrate_tsc(void)
{
  unsigned long lo1, lo2, res, tsc_hi1, tsc_lo1, tsc_hi2, tsc_lo2;
  register unsigned long ticks, newticks;

  ThreadSwitch();
  ticks = GetCurrentTime();
  while (ticks == (newticks = GetCurrentTime()))
    ;
  while (newticks == (ticks = GetCurrentTime()))
    ;
  lo1 = read_tsc_lo();
  while (ticks == GetCurrentTime())
    ;
  lo2 = read_tsc_lo();
  if (lo2 < lo1)
  {
    lo2 += (0xfffffffful - lo1)+1;
    lo1 = 0;
  }
  lo2 -= lo1;
  _asm mov eax,lo2
  _asm xor edx,edx
  _asm mov ecx,18206
  _asm mul ecx
  _asm add eax, 500000000
  _asm adc edx, 0
  _asm mov ecx, 1000000000
  _asm div ecx
  _asm mov res, eax
  ConsolePrintf("\rcalibarate_tsc(): tsc: %uMhz\r\n", res);

  ConsolePrintf("\rcalibarate_tsc(): begin tsc check\r\n");
  ThreadSwitch();
  /* get secs and nanosecs either from the TSC or from the clock chip */
  tsc_hi1 = __get_secs_and_nanosecs(&tsc_lo1);
  tsc_hi2 = __get_secs_and_nanosecs(&tsc_lo2);

  ticks = GetCurrentTime();
  while (ticks == (newticks = GetCurrentTime()))
    ;
  tsc_hi1 = __get_secs_and_nanosecs(&tsc_lo1);
  while (newticks == GetCurrentTime())
    ;
  tsc_hi2 = __get_secs_and_nanosecs(&tsc_lo2);
  ticks = GetCurrentTime();
  ThreadSwitch();
 
  _asm mov eax,tsc_lo2
  _asm mov edx,tsc_hi2
  _asm sub eax,tsc_lo1
  _asm sbb edx,tsc_hi1
  _asm mov lo1,eax
  _asm mov lo2,edx
  ConsolePrintf("\rcalibarate_tsc(): elapsed: %u.%09u (%d) (should be ~0.054,925,494)\n", lo2,lo1,ticks-newticks);
  return;
}
#endif

#ifdef DEMO_TIMEFUNC
static void __demonstrate_time_func(unsigned long (*proc)(void), const char *name)
{
  if (proc && name)
  {
    unsigned long ctr_1, ctr_2, time_start, time_end, elapsed;

    __is_showing_timer++;
    ThreadSwitchWithDelay();

    time_start = time_end = 0;
    ctr_1 = ctr_2 = (read_cmos_index(0 /* seconds index */) & 0xff);
    while (ctr_2 == ctr_1)
    {
      ctr_1 = (read_cmos_index(0 /* seconds index */) & 0xff);
    }
    time_start = (*proc)();
    ThreadSwitchWithDelay();
    ctr_2 = ctr_1;
    while (ctr_1 == ctr_2)
    {
      ctr_2 = (read_cmos_index(0 /* seconds index */) & 0xff);
    }
    time_end = (*proc)();
    ThreadSwitchWithDelay();
    elapsed = ((long)time_end) - ((long)time_start);
    if (ctr_2 != ctr_1)
      elapsed /= (ctr_2-ctr_1);

    ConsolePrintf("\r%s: diff=%d per_sec = %u\r\n",
                   name, ctr_2-ctr_1, elapsed);  
    ThreadSwitchWithDelay();

    ctr_1 = ctr_2 = GetCurrentTime();
    while (ctr_1 == ctr_2)
    {
      ctr_1 = GetCurrentTime();
    }
    time_start = (*proc)();
    ThreadSwitch();
    ctr_2 = ctr_1;
    while (ctr_1 == ctr_2)
    {
      ctr_2 = GetCurrentTime();
    }
    time_end = (*proc)();
    ThreadSwitchWithDelay();
    elapsed = ((long)time_end) - ((long)time_start);
    if (ctr_2 != ctr_1)
      elapsed /= (ctr_2-ctr_1);

    ConsolePrintf("\r%s: diff=%d per_tick = %u\r\n",
                   name, ctr_2 - ctr_1, elapsed);  
    ThreadSwitchWithDelay();

    __is_showing_timer--;
  }
  return;
}
#endif


/* ===================================================================== */
/* GetClockStatus()                                                      */
/* ===================================================================== */

static void __3x_GetClockStatus(unsigned long casptr[3])
{
  static unsigned long splbuf[2] = {0,0};
  static unsigned long epoch_delta = 0, last_secs = 0, last_nsecs = 0;
  unsigned long ticks_lo32, ticks_hi32, clks, secs, nsecs, fracsecs, utctime;

  int lacquired = 0;
  char *splp = (char *)&splbuf[0];
  splp += (sizeof(long)-(((unsigned long)splp)&(sizeof(long)-1)));
  while (!lacquired)
  {
    _asm mov ecx,splp;
    _asm mov eax,1
    _asm lock xchg eax,[ecx]
    _asm xor eax,1
    _asm mov lacquired,eax
  }

  ticks_hi32 = __3x_get_ticks_and_pit(&ticks_lo32, &clks);
  __3x_convert_ticks_and_pit(ticks_hi32, ticks_lo32, clks, &secs, &nsecs);

  secs += epoch_delta;
  utctime = (unsigned long)time(0);

  if (secs < (utctime-1) || secs > (utctime+1)) /* time-of-day changed */
  {
    //ConsolePrintf("\rsecs != utctime (%u != %u)\r\n", secs, utctime);
    secs -= epoch_delta;
    epoch_delta = utctime - secs;
    secs += epoch_delta;
    if (secs == last_secs && nsecs < last_nsecs)
    {
      epoch_delta--;
      secs--;
    }
    //ConsolePrintf("\r%u: new secs = %u\r\n", GetThreadID(), secs);
  }
  last_secs = secs;
  last_nsecs = nsecs;
  *((long *)splp) = 0;

  if (nsecs >= 1000000000ul) /* should never happen */
  {
    secs += nsecs / 1000000000ul;
    nsecs %= 1000000000ul;
  }
    
  fracsecs = 0;
  _asm mov eax, nsecs     /* nsecs *should* never be > 1000000000 */
  _asm xor edx, edx       /* but make sure to guarantee no overflow */
  _asm mov ecx, 3B9ACA00h /* 1,000,000,000 */
  _asm div ecx
  _asm add secs, eax      /* overflowed secs */
  _asm mov eax, edx       /* remaining nanosecs */
  _asm xor edx, edx
  _asm xor ecx, ecx
  _asm dec ecx
  _asm mul ecx
  _asm mov ecx, 3B9ACA00h /* 1,000,000,000 */
  _asm div ecx
  _asm mov fracsecs,eax   /* 999,999,999ns => 0xFFFFFFFB */
    
  casptr[0] = secs;
  casptr[1] = fracsecs;
  casptr[2] = 0; /* not CLOCK_SYNCHRONIZATION_IS_ACTIVE */
  return;
}

typedef unsigned long clockAndStatus[3];
void GetClockStatus( clockAndStatus casptr )
{
  static void (*_GetClockStatusT)(clockAndStatus) = 
                                         ((void (*)(clockAndStatus))0x01);
  register void (*_GetClockStatus)(clockAndStatus) = _GetClockStatusT;

  if (_GetClockStatus == ((void (*)(clockAndStatus))0x01))
  {
    _GetClockStatus = (void (*)(clockAndStatus))
                          ImportSymbol(GetNLMHandle(), "GetClockStatus" );
    if (!_GetClockStatus)
    {
      if (GetFileServerMajorVersionNumber() >= 4)
      {
        /* we can't use the emulation functions with nw4x/5x because the
           clock rate is 'unknown'. (and we shouldn't be doing inp/outp anyway)
        */
        ConsolePrintf("\rDNETC *FATAL*: Unable to get GetClockStatus vector\r\n");
        RingTheBell();
        casptr[0] = time(0); /* <= time() calls GetClockStatus on 4x */
        casptr[1] = 0;
        casptr[2] = 0;
        while (casptr[0] == time(0))
          ThreadSwitchWithDelay(); /* not delay()! */
        return;
      }
      /* netware 3 */
      _GetClockStatus = __3x_GetClockStatus;
      (*_GetClockStatus)( casptr ); /* prime it */
    }  
    _GetClockStatusT = _GetClockStatus;
    #if defined(PATCH_SELF)
    __patchme( GetClockStatus, _GetClockStatus );
    #endif
  }  
  (*_GetClockStatus)( casptr );
  return;
}

/* ===================================================================== */
/* NWGetSuperHighResolutionTimer(void) [1us count regardless of OS ver]  */
/* ===================================================================== */

/* this returns the equivalent of the netware 4.11 1us timer, and
   and is named after the NW* style wrapper in NetWare 4.11's NLMLIB
   rather than the kernel function because the underlying kernel function 
   is not consistant with docs. [see Issue a)]

   NLMLIB's NWGetSuperHighResolutionTimer() is simply a wrapper around
   the 4.11+ kernel's GetSuperHighResolutionTimer().

   ISSUES:
   a) The 4.11+ GetSuperHighResolutionTimer() returns a 1us count.
      The older GetSuperHighResolutionTimer() returns a 838ns count.
      Resolution: DONT CARE. We don't need to care about that here. 
      The NW* version always returns a 1us count. (the NW* version 
      first appeared on 4.11)
   b) The real [>=nw411] GetSuperHighResolutionTimer() returns a 
      64bit value, NLMLIB's NWGetSuperHighResolutionTimer() only 
      returns the low 32bits. 
      Resolution: DONT CARE. We don't need to care about that here. 
      We emulate what the NLMLIB version does.
   c) (Lan Dispatch Newsletter: October 14, 1996)
      <quote>
        First, for 486 processors the OS calculates the time by 
        concatinating a free running timer with current time. The problem 
        is that the two times are not always in sync, causing the microsecond 
        count to take jumps backwards from time to time. This is due to the 
        free running timer rolling over well before the current time is 
        incremented. 
        Second, for Pentium processors the microsecond time is retrieved 
        from the processor as a 32 bit value and there is no rollover problem. 
        However, on SMP machines all Pentiums will not necessarily be in sync 
        with their timers. Each call to GetSuperHighResolutionTime will give 
        you the time for the processor that you are currently running on.
      </quote>
      ARRRRRRRRRRRRGGGGGGGGGGGGGGHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH!
      These idiots apparently haven't discovered rdmsr/wrmsr 0x10
      Resolution: use the kernel's GetHighResolutionTimer() only *if*
        we have a TSC (otherwise the kernel would use the pit, in
        which case we should use our own pit reader which won't wrap)
        AND either
            a) we have more than one CPU. I don't know what clock rate
               smp.nlm re-configures the CPU for.
            b) we have NetWare 5. I hope they will fix ISSUE c) above.
      If the above 'if' is not true, then emulate it ourselves.
*/

static unsigned long __emu_NWGetSuperHighResolutionTimer(void)
{
  #if 0
  unsigned long ticks_hi32, ticks_lo32, clks, secs, nsecs, hi, lo;
  ticks_hi32 = __3x_get_ticks_and_pit(&ticks_lo32, &clks);
  __3x_convert_ticks_and_pit(ticks_hi32, ticks_lo32, clks, &secs, &nsecs);

  hi = secs;  
  lo = nsecs /= 1000;
  _asm mov eax,hi
  _asm xor edx,edx
  _asm mov ecx,0F4240h
  _asm mul ecx
  _asm add eax,lo
  _asm adc edx,0
  _asm mov lo, eax
  _asm mov hi, edx
  return (lo);
  #elif 0
  unsigned long usec, sec;
  clockAndStatus cas;  
  GetClockStatus( cas );

  sec  = cas[0]; /* full secs */
  usec = cas[1]; /* frac secs */

  _asm mov eax, usec
  _asm xor edx, edx
  _asm mov ecx, 1000000
  _asm mul ecx
  _asm mov usec, edx

  return (sec * 1000000)+usec;
  #else
  unsigned long nsecs, secs;
  /* get secs and nanosecs either from the TSC or from the clock chip */
  secs = __get_secs_and_nanosecs(&nsecs);
  if (__is_showing_timer)
    ConsolePrintf("\r__emu_NWGetSuperHighResolutionTimer:%u.%08u\r\n", secs,nsecs);
  return ((nsecs / 1000ul)+(secs * 1000000ul));
  #endif
}

unsigned long NWGetSuperHighResolutionTimer(void)
{                         /* 1us count regardless of OS version */
  /* This function, unlike GetSuperHighResolutionTimer(), returns a
   1us count regardless of OS version.

   It has been named/modelled after the NetWare 4.11 NLMLIB.NLM function
   of the same name. Note that although the real [>=NetWare 411] 
   GetSuperHighResolutionTimer() returns a 64bit value, NLMLIB's 
   NWGetSuperHighResolutionTimer() only returns the low 32bits.

   NW4.11's NWGetSuperHighResolutionTimer() disables/re-enables interrupts!
  */
  static unsigned long (*_NWGetSuperHighResolutionTimer_T)(void) =
                                         ((unsigned long  (*)(void))0x01);
  register unsigned long (*_NWGetSuperHighResolutionTimer)(void) =
                                         _NWGetSuperHighResolutionTimer_T;
  if (_NWGetSuperHighResolutionTimer == ((unsigned long (*)(void))0x01))
  { 
    _NWGetSuperHighResolutionTimer = (unsigned long (*)(void))0;

    if (__have_tsc() && (GetFileServerMajorVersionNumber() >=5 ||
                         GetNumberOfRegisteredProcessors() > 1))
    {
      /* we only want the OS version if ...
         - we have a TSC (our PIT version is better, doesn't wrap) AND
         - when using netware 5 (because I don't know how its loader works),
         - when we have more than one cpu (because I *hope* that
           either tsc will be synchronized [it isn't in 4.11, is it in 5.x?] 
           or the function will use the APIC or PCI clock
      */
      _NWGetSuperHighResolutionTimer = (unsigned long (*)(void))
              ImportSymbol( GetNLMHandle(), "_NWGetSuperHighResolutionTimer" );
      #ifdef DEMO_TIMEFUNC
      if (_NWGetSuperHighResolutionTimer)
        ConsolePrintf("\rUsing native NWGetSuperHighResolutionTimer()\r\n");
      #endif
    }
    if (!_NWGetSuperHighResolutionTimer)
    {
      _NWGetSuperHighResolutionTimer = 
               (unsigned long (*)(void))__emu_NWGetSuperHighResolutionTimer;
      (*_NWGetSuperHighResolutionTimer)(); /* prime it */
      #ifdef DEMO_TIMEFUNC
      ConsolePrintf("\rUsing emulated NWGetSuperHighResolutionTimer()\r\n");
      #endif
    }
    _NWGetSuperHighResolutionTimer_T = _NWGetSuperHighResolutionTimer;

    #ifdef DEMO_TIMEFUNC
    __demonstrate_time_func(_NWGetSuperHighResolutionTimer,"NWGetSuperHighResolutionTimer");
    #endif
  }
  return (*_NWGetSuperHighResolutionTimer)();
}

/* ===================================================================== */
/* GetSuperHighResolutionTimer() [1us on 4.11+, 838ns below 4.11]        */
/* ===================================================================== */

#if 0 /* -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- */

static unsigned long __3x_GetSuperHighResolutionTimer(void)
{
  /* this returns the NetWare 3x equivalent of GetSuperHighResolutionTimer(),
     ie 838nanosec count, and not the the NetWare 411 1us equivalent
  */
  unsigned long ticks_lo32, clks;
  __3x_get_ticks_and_pit(&ticks_lo32, &clks);
  if (__is_showing_timer)
    ConsolePrintf("\r__3x_GetSuperHighResolutionTimer:%lu\r\n", (ticks_lo32<<16)+clks);
  return (ticks_lo32<<16)+clks;
}  

static unsigned long __4x_GetSuperHighResolutionTimer(void)
{
  /* this returns the NetWare 4x equivalent of GetSuperHighResolutionTimer(),
     ie 1us count, and not the the NetWare 3x 838ns equivalent
  */
  unsigned long nsecs, secs;
  /* get secs and nanosecs either from the TSC or from the clock chip */
  secs = __get_secs_and_nanosecs(&nsecs);
  if (__is_showing_timer)
    ConsolePrintf("\r__4x_GetSuperHighResolutionTimer:%u.%08u\r\n", secs,nsecs);
  return ((nsecs / 1000ul)+(secs * 1000000ul));
}

unsigned long GetSuperHighResolutionTimer(void)
{
  /* http://developer.novell.com/ndk/doc/storarch/nwpa_enu/data/hct8zfuy.htm
     SDK: "This is a high resolution timer that combines the lowest 16bits
     of Current Time with the timer register to give a timer resolution
     of approximately 838 nanoseconds per count. This call does not allow
     for possible tick count rollover, so the programmer must take into
     consideration a "negative" time count."
     (838ns == 1000000000/1193181.666...)

     This is totally WRONG for Pentium Aware (ie use TSC) NetWare 4.11 and 
     above. 

     To get around all this crappiness, I've created an equivalent of
     NetWare 4.11's NWGetSuperHighResolutionTimer() - 1us always.
  */
  static unsigned long (*_GetSuperHighResolutionTimer_T)(void) =
                                         ((unsigned long (*)(void))0x01);
  register unsigned long (*_GetSuperHighResolutionTimer)(void) =
                                         _GetSuperHighResolutionTimer_T;
  if (_GetSuperHighResolutionTimer == ((unsigned long (*)(void))0x01))
  {  
    _GetSuperHighResolutionTimer = (unsigned long (*)(void))0;
    if (__have_tsc() && GetFileServerMajorVersionNumber() >= 4)
    {
      _GetSuperHighResolutionTimer = (unsigned long (*)(void))
              ImportSymbol( GetNLMHandle(), "GetSuperHighResolutionTimer" );
      #ifdef DEMO_TIMEFUNC
      if (_GetSuperHighResolutionTimer)
        ConsolePrintf("\rUsing kernel's GetSuperHighResolutionTimer\r\n");
      #endif
    }
    if (!_GetSuperHighResolutionTimer)
    {
      _GetSuperHighResolutionTimer = __3x_GetSuperHighResolutionTimer;
      if (GetFileServerMajorVersionNumber() >= 4)
        _GetSuperHighResolutionTimer = __4x_GetSuperHighResolutionTimer;
      (*_GetSuperHighResolutionTimer)(); /* prime it */
      #ifdef DEMO_TIMEFUNC
        ConsolePrintf("\rUsing emulated GetSuperHighResolutionTimer\r\n");
      #endif
    }
    _GetSuperHighResolutionTimer_T = _GetSuperHighResolutionTimer;

    #ifdef DEMO_TIMEFUNC
    __demonstrate_time_func(_GetSuperHighResolutionTimer,"GetSuperHighResolutionTimer");
    #endif
  }
  return (*_GetSuperHighResolutionTimer)();
}
#endif /* -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- UNUSED -- */

/* ===================================================================== */
/* GetHighResolutionTimer() [100us on all OSs]                           */
/* ===================================================================== */

static unsigned long __emu_GetHighResolutionTimer(void)
{
  /* SDK: "This timer combines the Current Time with the timer register 
     to create a return value that has a resolution of approximately 
     100 microseconds per count."

     That sounds wrong on 3.x because...
     - Judging from 3.12 kernel histogram stuff:
         call GetHighResolutionTimer
         shl  eax, 17h      <- huh? 
         mov  [ebp-8], eax
         mov  edx, dword ptr CurrentTime
         not  edx
         and  edx, 7FFFFFh  <- huh? 
         mov  ecx, [ebp-8]
         or   ecx, edx
         mov  [eax+0Ch], ecx
     - NKS support for 3.12 has something like this: 
         proc _312microseconds
         call GetHighResolutionTimer
         mul  eax, 1000       <- 1000? should be 100!
         endp
     BUT we don't care, and simply do what the API says _today_ 
     (and which is correct for 4.11, 5.0 and 5.1)
  */
  unsigned long nsecs, secs;
  /* get secs and nanosecs either from the TSC or from the clock chip */
  secs = __get_secs_and_nanosecs(&nsecs);
  if (__is_showing_timer)
    ConsolePrintf("\r__emu_GetHighResolutionTimer:%u.%08u\r\n", secs,nsecs);
  return ((nsecs / 100000ul)+(secs * 10000ul));
}


unsigned long GetHighResolutionTimer(void)
{
  /* SDK: "This timer combines the Current Time with the timer register 
     to create a return value that has a resolution of approximately 
     100 microseconds per count."
  */   
  static unsigned long (*_GetHighResolutionTimer_T)(void) = 
                                         ((unsigned long (*)(void))0x01);
  register unsigned long (*_GetHighResolutionTimer)(void) =
                                         _GetHighResolutionTimer_T;
  if (_GetHighResolutionTimer == ((unsigned long (*)(void))0x01))
  {
    _GetHighResolutionTimer = (unsigned long (*)(void))0;
    if (__have_tsc() && (GetFileServerMajorVersionNumber() >=5 ||
                         GetNumberOfRegisteredProcessors() > 1))
    {
      /* we only want the OS version if ...
         - we have a TSC (out PIT version is better, doesn't wrap) AND
         - when using netware 5 (because I don't know how its loader works),
         - when we have more than one cpu (because I *hope* tsc will
           be synchronized [it isn't in 4.11, is it in 5.x?] 
      */
      _GetHighResolutionTimer = (unsigned long (*)(void))
              ImportSymbol( GetNLMHandle(), "GetHighResolutionTimer" );
      #ifdef DEMO_TIMEFUNC
      if (_GetHighResolutionTimer) 
        ConsolePrintf("\rSelected kernel's GetHighResolutionTimer().\r\n");
      #endif
    }
    if (!_GetHighResolutionTimer)
    {
      _GetHighResolutionTimer = __emu_GetHighResolutionTimer;
      (*_GetHighResolutionTimer)(); /* prime it */
      #ifdef DEMO_TIMEFUNC
      ConsolePrintf("\rWill get GetHighResolutionTimer() from TSC or PIT.\r\n");
      #endif
    }
    _GetHighResolutionTimer_T = _GetHighResolutionTimer;

    #ifdef DEMO_TIMEFUNC
    __demonstrate_time_func(_GetHighResolutionTimer,"GetHighResolutionTimer");
    #endif
  }
  return (*_GetHighResolutionTimer)();
}

/* ===================================================================== */

int ReturnFileServerName(char *buffer) /* min 64+1 */
{                                      /* to get len call with buffer==NULL */
  static char fsname[64+1];
  static int fsnamelen = -1;
  int pos;
  
  if (fsnamelen < 0)
  {
    char *lname;
    unsigned int nlmHandle = GetNLMHandle();
    char *symname = "ReturnFileServerName";
    void *vect = ImportSymbol( nlmHandle, symname );
    fsnamelen = 0;
    if (vect)
    {
      char scratch[128+1];
      fsnamelen = ((int (*)(char *))vect)(scratch);
      if (fsnamelen)
      {
        pos = 0;
        lname = scratch;
        while (pos < fsnamelen && pos < (sizeof(fsname)-1))
          fsname[pos++] = *lname++;
        fsnamelen = pos;
      }
      fsname[fsnamelen] = '\0';
    }
    else
    {
      vect = ImportSymbol( nlmHandle, symname+6 );
      lname = (vect) ? (*((char **)vect)) : ((char *)0);
      if (lname)
      {
        fsnamelen = *((unsigned char *)lname); 
        lname++;
      }
      if (fsnamelen)
      {
        pos = 0;
        while (pos < fsnamelen && pos < (sizeof(fsname)-1))
          fsname[pos++] = *lname++;
        fsnamelen = pos;
      }
      fsname[fsnamelen] = '\0';
      UnimportSymbol( nlmHandle, symname+6 );
    }
  }
  if (buffer)
  {
    int i;
    for (i=0; i<fsnamelen; i++)
      buffer[i] = fsname[i];
    buffer[fsnamelen]='\0';
  }
  return fsnamelen;
}

/* ===================================================================== */

static int __GetSetSetableParameterValue( register int doGet, 
       unsigned long connum, unsigned char *setableParamName, void *val )
{
  static int (*_GetSetableParameterValue)( unsigned long, 
       unsigned char *, void * ) = 
       (int (*)(unsigned long, unsigned char *, void *))(0x01);
  static int (*_SetSetableParameterValue)( unsigned long, 
       unsigned char *, void * ) = 
       (int (*)(unsigned long, unsigned char *, void *))(0x01);

  if (_GetSetableParameterValue == ((int (*)(unsigned long, 
                                     unsigned char *, void *))(0x01)) )
  {
    _GetSetableParameterValue = 
        (int (*)(unsigned long, unsigned char *, void *))
        ImportSymbol( GetNLMHandle(), "GetSetableParameterValue" );
    _SetSetableParameterValue = 
        (int (*)(unsigned long, unsigned char *, void *))
        ImportSymbol( GetNLMHandle(), "SetSetableParameterValue" );
    #if defined(PATCH_SELF)
    if (_GetSetableParameterValue && _SetSetableParameterValue)
    {
      __patchme( GetSetableParameterValue, _GetSetableParameterValue);
      __patchme( SetSetableParameterValue, _SetSetableParameterValue);
    }
    #endif
  }
  if (_GetSetableParameterValue && _SetSetableParameterValue)
  {
    if (doGet)
      return (*_GetSetableParameterValue)(connum,setableParamName, val);
    return (*_SetSetableParameterValue)(connum,setableParamName, val);
  } 
  return -1;    
}

unsigned long GetSetableParameterValue( unsigned long connum,
                             unsigned char *setableParamName, void *val )
{                                                  /* CLIB.NLM>=4.x */
  return __GetSetSetableParameterValue( 1, connum, setableParamName, val );
}  
               
unsigned long SetSetableParameterValue( unsigned long connum, 
                             unsigned char *setableParamName, void *val )
{                                                  /* CLIB.NLM>=4.x */
  return __GetSetSetableParameterValue( 0, connum, setableParamName, val );
}  

/* ===================================================================== */
               
static int __GetFileServerMajorMinorRevisionVersionNumber(register int which)
{
  /* although A3112 exports this on 3.x, it always returns 3.11 */
  static int minor, revision, major = -1;
  if (major == -1)
  {
    unsigned int thisthat, nlmHandle = GetNLMHandle();
    struct { char *symname; int vernum; } minmajrev[3] = {
           { "GetFileServerMajorVersionNumber", 3 },
           { "GetFileServerMinorVersionNumber", 0 },
           { "GetFileServerRevisionNumber",     0 } };
    for (thisthat = 0; thisthat < 3; thisthat++)
    {
      void *vect; char *symname = minmajrev[thisthat].symname;
      /* although A3112 exports functions on 3.x, it always returns 3.11,
         so try to use the unsigned char data pointers first.
      */
      if ((vect = ImportSymbol( nlmHandle, symname+3 ))!=((void *)0))
      {
        minmajrev[thisthat].vernum = (*((unsigned char *)(vect)));
        UnimportSymbol( nlmHandle, symname+3 );
      }
      else if ((vect = ImportSymbol( nlmHandle, symname ))!=((void *)0))
      {
        minmajrev[thisthat].vernum = (*((int (*)(void))vect))(); 
        UnimportSymbol( nlmHandle, symname );
      }
    }
    revision = minmajrev[2].vernum;
    minor = minmajrev[1].vernum;
    major = minmajrev[0].vernum;
  }
  if (which == 'maj') return major;
  if (which == 'min') return minor;
  if (which == 'rev') return revision;
  return 0;
}  
  
unsigned int GetFileServerMajorVersionNumber(void) /* A3112/SERVER.NLM >=4.x */
{
  return __GetFileServerMajorMinorRevisionVersionNumber('maj');
}

unsigned int GetFileServerMinorVersionNumber(void) /* A3112/SERVER.NLM >=4.x */
{
  return __GetFileServerMajorMinorRevisionVersionNumber('min');
}

unsigned int GetFileServerRevisionNumber(void) /* A3112/SERVER.NLM >=4.x */
{
  return __GetFileServerMajorMinorRevisionVersionNumber('rev');
}

/* ===================================================================== */

static unsigned int __GetCurrMaxNumberOfPollingLoops(int getMax)
{
  static int (*_MaximumNumberOfPollingLoops) = ((int *)0x01);
  static int (*_NumberOfPollingLoops);
  if (_MaximumNumberOfPollingLoops == ((int *)0x01))
  {
    _MaximumNumberOfPollingLoops = 
       (int *)ImportSymbol(GetNLMHandle(), "MaximumNumberOfPollingLoops" );
    _NumberOfPollingLoops = 
       (int *)ImportSymbol(GetNLMHandle(), "NumberOfPollingLoops" );
  }
  if (_MaximumNumberOfPollingLoops && _NumberOfPollingLoops)
  {
    if (getMax)
      return (unsigned int)(*_MaximumNumberOfPollingLoops);
    return (unsigned int)(*_NumberOfPollingLoops);
  }
  return ((getMax)?(+1):(0));
}

unsigned int GetMaximumNumberOfPollingLoops(void) /* SERVER.NLM >=4.x */
{
  return __GetCurrMaxNumberOfPollingLoops(1);
}

unsigned int GetNumberOfPollingLoops(void)  /* SERVER.NLM >=4.x */
{
  return __GetCurrMaxNumberOfPollingLoops(0);
}

/* ===================================================================== */

#define __yield_ThreadSwitch                 1
#define __yield_ThreadSwitchWithDelay        2
#define __yield_ThreadSwitchLowPriority      3
#define __yield_CYieldIfNeeded            0x11
#define __yield_CYieldWithDelay           0x12
#define __yield_CYieldUntilIdle           0x13

static void __ThreadSwitchXXX(int mode)
{
  /* ThreadSwitch/thr_yield/CRescheduleMyself/CYieldIfNeeded */
  static void (*_ThreadSwitch)(void) = (void (*)(void))0x01; /* control */
  static void (*_ThreadSwitchWithDelay)(void);
  static void (*_ThreadSwitchLowPriority)(void);
  static void (*_CYieldIfNeeded)(void);
  static void (*_CYieldWithDelay)(void);
  static void (*_CYieldUntilIdle)(void);

  if (_ThreadSwitch == (void (*)(void))0x01)
  {
    void (*proc)(void);
    unsigned int nlmHandle = GetNLMHandle();
    
    _CYieldIfNeeded = _ThreadSwitch = ThreadSwitch;
    _CYieldWithDelay = _ThreadSwitchWithDelay = (void (*)(void))0;
    _CYieldUntilIdle = _ThreadSwitchLowPriority = (void (*)(void))0;

    proc = (void (*)(void))ImportSymbol(nlmHandle, "CYieldIfNeeded"); /* 4.11 */
    if (!proc)
      proc = (void (*)(void))ImportSymbol(nlmHandle, "CRescheduleMyself");
    if (!proc) 
      proc = (void (*)(void))ImportSymbol(nlmHandle, "CRescheduleLast"); /*3.11*/
    if (proc)               
      _CYieldIfNeeded = proc; /* phew! */

    proc = (void (*)(void))ImportSymbol(nlmHandle, "CYieldUntilIdle");
    if (!proc)
      proc = (void (*)(void))ImportSymbol(nlmHandle, "CRescheduleLastLowPriority");
    if (proc)
    {
      _ThreadSwitchLowPriority = _CYieldUntilIdle = proc;
      proc = (void (*)(void))ImportSymbol(nlmHandle, "ThreadSwitchLowPriority");
      if (proc)
      {
        _ThreadSwitchLowPriority = proc;
        #if defined(PATCH_SELF)
        __patchme( ThreadSwitchLowPriority, _ThreadSwitchLowPriority );
        #endif
      }
    }
    
    proc = (void (*)(void))ImportSymbol(nlmHandle, "CYieldWithDelay");
    if (!proc)
      proc = (void (*)(void))ImportSymbol(nlmHandle, "CRescheduleLastWithDelay");
    if (proc)
    {
      _ThreadSwitchWithDelay = _CYieldWithDelay = proc;
      proc = (void (*)(void))ImportSymbol(nlmHandle, "ThreadSwitchWithDelay");
      if (proc)
      {
        _ThreadSwitchWithDelay = proc;
        #if defined(PATCH_SELF)
        __patchme( ThreadSwitchWithDelay, _ThreadSwitchWithDelay );
        #endif
      }
    }
  } 
   
  if (mode == __yield_CYieldUntilIdle)
  {
    if (_CYieldUntilIdle)
      (*_CYieldUntilIdle)();
    else
      (*_CYieldIfNeeded)();
  }
  else if (mode == __yield_ThreadSwitchLowPriority)
  {
    if (_ThreadSwitchLowPriority)
      (*_ThreadSwitchLowPriority)();
    //else if (_ThreadSwitchWithDelay)
    //  (*_ThreadSwitchWithDelay)();
    else
      ThreadSwitch();
  }
  else if (mode == __yield_CYieldWithDelay)
  {
    if (_CYieldWithDelay)
      (*_CYieldWithDelay)();
    else
    { /* should we do a CSchedule|CancelInterruptTimeCallBack thingie here? */
      int i=0; 
      while ((i++)<=10)
        (*_CYieldIfNeeded)();
    }
  }
  else if (mode == __yield_ThreadSwitchWithDelay)
  {
    if (_ThreadSwitchWithDelay)
      (*_ThreadSwitchWithDelay)();
    else
    {
      int i=0;
      while ((i++)<=10)
        ThreadSwitch();      
      /* while (GetDiskIOsPending() || GetNestedInterruptLevel()) 
        ThreadSwitch();
      */
    }
  }   
  else /* if (mode == __yield_CYieldIfNeeded) */
  {
    (*_CYieldIfNeeded)();
  }
  return;
}

void ThreadSwitchWithDelay(void) /* CLIB >=3.12 */
{  __ThreadSwitchXXX(__yield_ThreadSwitchWithDelay); }
void ThreadSwitchLowPriority(void) /* CLIB >=4.x */
{  __ThreadSwitchXXX(__yield_ThreadSwitchLowPriority); }

void CRescheduleLastWithDelay(void) /* kernel >=4.x, 3.12 */
{  __ThreadSwitchXXX(__yield_CYieldWithDelay); }
void CRescheduleWithDelay(void) /* kernel >=4.x, 3.12 */
{  __ThreadSwitchXXX(__yield_CYieldWithDelay); }
void CYieldWithDelay(void) /* kernel >=4.x, 3.12 */
{  __ThreadSwitchXXX(__yield_CYieldWithDelay); }

void CRescheduleLastLowPriority(void) /* kernel >=4.x */
{  __ThreadSwitchXXX(__yield_CYieldUntilIdle); }
void CYieldUntilIdle(void) /* kernel >=4.x */
{  __ThreadSwitchXXX(__yield_CYieldUntilIdle); }

void CRescheduleMyself(void) /* kernet >=3.0? */
{  __ThreadSwitchXXX(__yield_CYieldIfNeeded); }
void CRescheduleLast(void) /* kernet >=3.0? */
{  __ThreadSwitchXXX(__yield_CYieldIfNeeded); }
void CYieldIfNeeded(void) 
{  __ThreadSwitchXXX(__yield_CYieldIfNeeded); }

/* ===================================================================== */

/* what is this function called? its not CalculateProcessorUtilization either*/

unsigned int GetProcessorUtilization(void) /* SMP.NLM or kernel */
{
  static int (*_CPU_Combined) = ((int *)0x01);
  unsigned int m, n;
  if (_CPU_Combined == ((int *)0x01))
    _CPU_Combined = (int *)ImportSymbol( GetNLMHandle(), "CPU_Combined" );
  if (_CPU_Combined)
    return (unsigned int)(*_CPU_Combined);
  m = GetMaximumNumberOfPollingLoops();
  n = GetNumberOfPollingLoops();
  if (n > ((0xFFFFFFFFul)/150))
  {
    n>>=8;
    m>>=8;
  }
  if ((m == 0) || (n > m))
    return 0;
  return (100-(((n*100)+(m>>1))/m));
}  

/* ===================================================================== */

unsigned int GetNumberOfRegisteredProcessors(void) /* Kernel >= 4.x */
{
  static int count = -1;
  if (count < 0)
  {
    unsigned int nlmHandle = GetNLMHandle();
    char *fname = "GetNumberOfRegisteredProcessors";
    int (*proc)(void) = (int (*)(void))ImportSymbol( nlmHandle, fname );
    if (proc)
    {
      count = ((*proc)());
      UnimportSymbol( nlmHandle, fname );
    }
    if (count < 1)
      count = 1;
  }
  return count;
}

/* ===================================================================== */
/* 
   NWSMPIsLoaded() and NWSMPIsAvailable() have to be the stupidest, most 
   misleading pieces of code ever to cross the CLIB arena.
*/
static int __NWSMPIsXXX(int which)
{
  static int _isavail, _isloaded = -1;
  if (_isloaded == -1)
  {
    int isloaded = 0, isavail = 0;
    int ver = (GetFileServerMajorVersionNumber() * 100) + 
               GetFileServerMinorVersionNumber();
    if ( ver >= 411 )
    {
      unsigned int nlmHandle = GetNLMHandle();
      char *fname; int (*proc)(void); 

      fname = "NWSMPIsAvailable";
      proc = (int (*)(void))ImportSymbol( nlmHandle, fname );
      if (proc)
      {
        if ((*proc)())
          isavail++;
        UnimportSymbol( nlmHandle, fname );
      }
      isloaded = isavail; /* must be loaded if available */
      if (!isloaded)
      {
        fname = "NWSMPIsLoaded";
        proc = (int (*)(void))ImportSymbol( nlmHandle, fname );
        if (proc)
        {
          if ((*proc)())
            isloaded++;
          UnimportSymbol( nlmHandle, fname );
        }
      }

      if (!isavail && !isloaded) /* cannot trust the test just done */
      {
        char scratch[64];
        fname = "SMP NetWare Kernel Mode";
        if ( GetSetableParameterValue( 0, (unsigned char *)fname, 
                                            (void *)&scratch[0] ) == 0 )
        {
          isloaded = 1;
          if ( ver >= 500 )
            isavail = 1;
          else 
          {
            unsigned int (*_spin_alloc)(const char *name);
            _spin_alloc = (unsigned int (*)(const char *name))
                      ImportSymbol( nlmHandle, "NWSMPSpinAlloc" );
            if (_spin_alloc)
            {
              int (*_spin_destroy)( unsigned int ) = (int (*)(unsigned int))
                   ImportSymbol( nlmHandle, "NWSMPSpinDestroy" );
              if (_spin_destroy)
              {
                unsigned int spin = (*_spin_alloc)("distributed.net");
                if (spin)
                {
                  isavail = 1;
                  (*_spin_destroy)(spin);
                }
                UnimportSymbol( nlmHandle, "NWSMPSpinDestroy" );
              }
              UnimportSymbol( nlmHandle, "NWSMPSpinAlloc" );
            }
          } /* NW411 */
        } /* GetSetableParameterValue */
      } /* (!isavail && !isloaded) */
    } /* Ver >= 411 */
    _isavail = isavail;
    _isloaded = isloaded;
  }
  if (_isavail)
    return 1;
  if (which == 'l')
    return _isloaded;
  return _isavail;
}      
 
int NWSMPIsLoaded(void) /* "has CLIB CLIB'ified SMP.NLM exports?" */
{ return __NWSMPIsXXX('l'); }
int NWSMPIsAvailable(void) /* test of SMP.NLM's "extern int *mdisable;" */
{ return __NWSMPIsXXX('a'); }

/* ===================================================================== */

static void NWSMPThreadToXXX(int which) /* 'MP' or 'NW' */
{
  static void (*_NWSMPThreadToMP)(void) = (void (*)(void))0x01;
  static void (*_NWSMPThreadToNW)(void) = (void (*)(void))0x01;
  if (_NWSMPThreadToMP == ((void (*)(void))0x01))
  {
    void *procMP, *procNW;
    int nlmhandle = GetNLMHandle();
    procMP = ImportSymbol( nlmhandle, "kExitNetWare" );
    if (!procMP) procMP = ImportSymbol( nlmhandle, "NWSMPThreadToMP" );
    if (!procMP) procMP = ImportSymbol( nlmhandle, "NWThreadToMP" );
    procNW = ImportSymbol( nlmhandle, "kEnterNetWare" );
    if (procNW) procNW = ImportSymbol( nlmhandle, "NWSMPThreadToNetWare" );
    if (procNW) procNW = ImportSymbol( nlmhandle, "NWThreadToNetWare" );
    #if defined(PATCH_SELF)
    if (procMP) __patchme(NWSMPThreadToMP,procMP);
    if (procNW) __patchme(NWSMPThreadToNetWare,procNW);
    #endif
    _NWSMPThreadToNW = (void (*)(void))procNW;
    _NWSMPThreadToMP = (void (*)(void))procMP;
  }
  if (which == 'MP')
  {
    if (_NWSMPThreadToMP)
      (*_NWSMPThreadToMP)();
  }  
  else if (_NWSMPThreadToNW)
    (*_NWSMPThreadToNW)();
  return;
}

void NWSMPThreadToMP(void) { NWSMPThreadToXXX('MP'); }
void NWThreadToMP(void) { NWSMPThreadToMP(); }
void NWSMPThreadToNetWare(void) { NWSMPThreadToXXX('NW'); }
void NWThreadToNetWare(void) { NWSMPThreadToNetWare(); }

/* ===================================================================== */

int GetServerConfigurationInfo(int *servType, int *ldrType) /* >=3.12/4.1 */
{ 
  static int ldr = -1, serv = -1;
  if (serv == -1)
  {
    int nlmHandle = GetNLMHandle();
    char *symname = "GetServerConfigurationInfo";
    void *proc = ImportSymbol( nlmHandle, symname );
    if (proc)
    {
      (*((int (*)(int *,int *))proc))(&serv, &ldr);
      UnimportSymbol( nlmHandle, symname );
    }
    else
    {
      symname = "GetServerConfigurationType";
      proc = ImportSymbol( nlmHandle, symname );
      if (proc)
      {
        serv = (*((int (*)(void))proc))();
        UnimportSymbol( nlmHandle, symname );
      }
      else 
      {
        symname = "serverConfigurationType";
        proc = ImportSymbol( nlmHandle, symname );
        if (proc)
        {
          serv = (*((int *)proc));
          UnimportSymbol( nlmHandle, symname );
        }
      }
    }
    if (serv == -1)
      serv = 0;
  }
  if (servType)
    *servType = serv;
  if (ldrType)
    *ldrType = ((ldr == -1)?(1):(ldr));
  return 0;
}  

/* ===================================================================== */

static unsigned int __GetUnsignedXXX(int which)
{
  static unsigned int (*_GetNestedInterruptLevel)(void) = (unsigned int (*)(void))0x01;
  static unsigned int (*_GetDiskIOsPending)(void) = (unsigned int (*)(void))0x01;
  static unsigned int (*_NestedInterruptCount) = (unsigned int *)0;
  static unsigned int (*_DiskIOsPending) = (unsigned int *)0;

  /* the Get... versions cause a migration. */
  /* The unsigned int *'s don't, so use the */
  /* unsigned int's whenever possible. */

  if (_GetNestedInterruptLevel == ((unsigned int (*)(void))0x01))
  {
    unsigned int nlmHandle = GetNLMHandle();

    _DiskIOsPending = (unsigned int *)
                 ImportSymbol( nlmHandle, "DiskIOsPending" );
    _GetDiskIOsPending = (unsigned int (*)(void))
                 ImportSymbol( nlmHandle, "GetDiskIOsPending" );
    _NestedInterruptCount = (unsigned int *)
                 ImportSymbol( nlmHandle, "NestedInterruptCount" );
    _GetNestedInterruptLevel = (unsigned int (*)(void))
                 ImportSymbol( nlmHandle, "GetNestedInterruptLevel" );
  }
  if (which == 'q')
  {
    if (_NestedInterruptCount)
      return (*_NestedInterruptCount);
    if (_GetNestedInterruptLevel)
      return (*_GetNestedInterruptLevel)();
  }
  else if (which == 'd')
  {
    if (_DiskIOsPending)
      return (*_DiskIOsPending);
    if (_GetDiskIOsPending)
      return (*_GetDiskIOsPending)();
  }
  return 0;
}

int GetDiskIOsPending(void) { return __GetUnsignedXXX('q'); }
int GetNestedInterruptLevel(void) { return __GetUnsignedXXX('d'); }

/* ===================================================================== */

static int __addremovepollingproc( void (*proc)(void), unsigned long rTag )
{
  static void (*_RemovePollingProcedure)(void (*)(void))
               = (void (*)(void (*)(void)))0x01;
  static int (*_AddPollingProcedureRTag)(void (*)(void), unsigned long)
               = ((int (*)(void (*)(void),unsigned long ))0x01);
  if (_RemovePollingProcedure == ((void (*)(void))0x01))
  {
    unsigned int nlmHandle = GetNLMHandle();
    void *add = ImportSymbol( nlmHandle, "AddPollingProcedureRTag" );
    void *rem = ImportSymbol( nlmHandle, "RemovePollingProcedure" );
    if (!add || !rem)
    {
      if (add)
        UnimportSymbol( nlmHandle, "AddPollingProcedureRTag" );
      if (rem)
        UnimportSymbol( nlmHandle, "RemovePollingProcedure" );
      add = rem = (void *)0;
    }
    _AddPollingProcedureRTag = (int (*)(void (*)(void),unsigned long))add;
    _RemovePollingProcedure = (void (*)(void (*)(void)))rem;

    #if defined(PATCH_SELF)
    if (_AddPollingProcedureRTag && _RemovePollingProcedure)
    {
      __patchme(AddPollingProcedureRTag,_AddPollingProcedureRTag);
      __patchme(RemovePollingProcedure,_RemovePollingProcedure);
    }
    #endif
  }
  if (_AddPollingProcedureRTag && _RemovePollingProcedure)
  {
    if (rTag)
      return (*_AddPollingProcedureRTag)( proc, rTag );
    (*_RemovePollingProcedure)( proc );
    return 0;
  }
  return -1;
}

int AddPollingProcedureRTag( void (*proc)(void), unsigned long rTag )
{
  return (!rTag) ? -1 : __addremovepollingproc( proc, rTag );
}

void RemovePollingProcedure( void (*proc)(void) )
{
  __addremovepollingproc( proc, 0 ); return;
}

/* ===================================================================== */

