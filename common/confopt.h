// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confopt.h,v $
// Revision 1.12  1999/03/18 03:59:09  cyp
// new "Project priority" option.
//
// Revision 1.11  1999/02/20 03:07:17  gregh
// Add OGR options to configuration data.
//
// Revision 1.10  1999/02/07 16:00:09  cyp
// Lurk changes: genericified variable names, made less OS-centric.
//
// Revision 1.9  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the lookback one) will trigger the online
// status.
//
// Revision 1.8  1999/02/04 10:44:19  cyp
// Added support for script-driven dialup. (currently linux only)
//
// Revision 1.7  1999/01/04 02:47:30  cyp
// Cleaned up menu options and handling.
//
// Revision 1.5  1998/12/23 00:41:45  silby
// descontestclosed and scheduledupdatetime now read from the .ini file.
//
// Revision 1.3  1998/12/21 00:21:01  silby
// Universally scheduled update time is now retrieved from the proxy,
// and stored in the .ini file.  Not yet used, however.
//
// Revision 1.2  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.1  1998/11/22 15:16:19  cyp
// Split from cliconfig.cpp; Changed runoffline/runbuffers/blockcount handling
// (runbuffers is now synonymous with blockcount=-1; offlinemode is always
// 0/1); changed 'frequent' context to describe what it does better: check
// buffers frequently and not connect frequently. Removed outthreshold[0] as
// well as both DES thresholds from the menu. Removed 'processdes' from the menu.
// Fixed various bugs. Range validation is always based on the min/max values in
// the option table.
//
// 

#ifndef __CONFOPT_H__
#define __CONFOPT_H__

#define CONF_ID                    0 /* CONF_MENU_BUFF */
#define CONF_NODISK                1 /* CONF_MENU_BUFF */
#define CONF_FREQUENT              2 /* CONF_MENU_BUFF */
#define CONF_PREFERREDBLOCKSIZE    3 /* CONF_MENU_BUFF */
#define CONF_THRESHOLDI            4 /* CONF_MENU_BUFF */
#define CONF_RC5IN                 5 /* CONF_MENU_BUFF */
#define CONF_RC5OUT                6 /* CONF_MENU_BUFF */
#define CONF_DESIN                 7 /* CONF_MENU_BUFF */
#define CONF_DESOUT                8 /* CONF_MENU_BUFF */
#define CONF_OGRIN                 9 /* CONF_MENU_BUFF */
#define CONF_OGROUT               10 /* CONF_MENU_BUFF */
#define CONF_CHECKPOINT           11 /* CONF_MENU_BUFF */

#define CONF_COUNT                12 /* CONF_MENU_MISC -1 == runbuffers */
#define CONF_HOURS                13 /* CONF_MENU_MISC */
#define CONF_PAUSEFILE            14 /* CONF_MENU_MISC */
#define CONF_QUIETMODE            15 /* CONF_MENU_MISC */
#define CONF_NOEXITFILECHECK      16 /* CONF_MENU_MISC */
#define CONF_PERCENTOFF           17 /* CONF_MENU_MISC */
#define CONF_CONTESTPRIORITY      18 /* CONF_MENU_BUFF "DES,OGR,RC5" */

#define CONF_CPUTYPE              19 /* CONF_MENU_PERF */
#define CONF_NUMCPU               20 /* CONF_MENU_PERF 0 ... */
#define CONF_NICENESS             21 /* CONF_MENU_PERF priority */

#define CONF_LOGNAME              22 /* CONF_MENU_LOG */
#define CONF_MESSAGELEN           23 /* CONF_MENU_LOG */
#define CONF_SMTPSRVR             24 /* CONF_MENU_LOG */
#define CONF_SMTPPORT             25 /* CONF_MENU_LOG */
#define CONF_SMTPFROM             26 /* CONF_MENU_LOG */
#define CONF_SMTPDEST             27 /* CONF_MENU_LOG */

#define CONF_OFFLINEMODE          28 /* CONF_MENU_NET runoffline not runbuffers */
#define CONF_NETTIMEOUT           29 /* CONF_MENU_NET */
#define CONF_UUEHTTPMODE          30 /* CONF_MENU_NET */
#define CONF_AUTOFINDKS           31 /* CONF_MENU_NET autofindkeyserver */
#define CONF_KEYSERVNAME          32 /* CONF_MENU_NET (CONF_KEYPROXY) */
#define CONF_KEYSERVPORT          33 /* CONF_MENU_NET (CONF_HTTPPORT) */
#define CONF_NOFALLBACK           34 /* CONF_MENU_NET */
#define CONF_FWALLHOSTNAME        35 /* CONF_MENU_NET (CONF_HTTPPROXY) */
#define CONF_FWALLHOSTPORT        36 /* CONF_MENU_NET (CONF_HTTPPPORT) */
#define CONF_FWALLUSERNAME        37 /* CONF_MENU_NET (CONF_HTTPID)*/
#define CONF_FWALLPASSWORD        38 /* CONF_MENU_NET */

#define CONF_LURKMODE             39 /* CONF_MENU_NET */
#define CONF_CONNIFACEMASK        40 /* CONF_MENU_NET */
#define CONF_DIALWHENNEEDED       41 /* CONF_MENU_NET */
#define CONF_CONNPROFILE          42 /* CONF_MENU_NET */
#define CONF_CONNSTARTCMD         43 /* CONF_MENU_NET */
#define CONF_CONNSTOPCMD          44 /* CONF_MENU_NET */

#define CONF_OPTION_COUNT         45

#define CONF_MENU_UNDEF            0
#define CONF_MENU_BUFF             1
#define CONF_MENU_LOG              2
#define CONF_MENU_NET              3
#define CONF_MENU_PERF             4
#define CONF_MENU_MISC             5

#define CONF_TYPE_UNDEF            0
#define CONF_TYPE_ASCIIZ           1
#define CONF_TYPE_INT              2
#define CONF_TYPE_BOOL             3
#define CONF_TYPE_TIMESTR          4
#define CONF_TYPE_PASSWORD         5

/* ---------------------------------------------------------------- */

struct optionstruct
  {
  const char *description;     //description of the option
  const char *defaultsetting;  //default setting
  const char *comments;        //additional comments
  int optionscreen;            //screen to appear on
  int type;                    //type: 0=other, 1=string, 2=integer, 3=bool (yes/no)
  int menuposition;            //number on that menu to appear as
  void *thevariable;           //pointer to the variable
  const char **choicelist;     //pointer to the char* array of choices
                               //(used for numeric responses)
  s32 choicemin;               //minimum choice number
  s32 choicemax;               //maximum choice number
  };

/* ---------------------------------------------------------------- */

extern struct optionstruct conf_options[CONF_OPTION_COUNT];
extern int                 confopt_IsHostnameDNetHost( const char * hostname );
extern int                 confopt_isstringblank( const char *string );
extern void                confopt_killwhitespace( char *string );
extern const char *        confopt_build_projectmap( char *buf, const char *strtomap );
extern const char *        confopt_expand_projectmap( const char *map );

/* ---------------------------------------------------------------- */

#endif /* #ifndef __CONFOPT_H__ */
