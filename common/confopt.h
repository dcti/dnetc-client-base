/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 

#ifndef __CONFOPT_H__
#define __CONFOPT_H__ "@(#)$Id: confopt.h,v 1.13.2.1 1999/04/13 19:45:20 jlawson Exp $"

#define CONF_ID                    0 /* CONF_MENU_BUFF */
#define CONF_NODISK                1 /* CONF_MENU_BUFF */
#define CONF_FREQUENT              2 /* CONF_MENU_BUFF */
#define CONF_PREFERREDBLOCKSIZE    3 /* CONF_MENU_BUFF */
#define CONF_THRESHOLDI            4 /* CONF_MENU_BUFF */
#define CONF_INBUFFERBASENAME      5 /* CONF_MENU_BUFF */
#define CONF_OUTBUFFERBASENAME     6 /* CONF_MENU_BUFF */
#define CONF_CHECKPOINT            7 /* CONF_MENU_BUFF */
#define CONF_REMOTEUPDATEDIR       8 /* CONF_MENU_BUFF */

#define CONF_COUNT                 9 /* CONF_MENU_MISC -1 == runbuffers */
#define CONF_HOURS                10 /* CONF_MENU_MISC */
#define CONF_PAUSEFILE            11 /* CONF_MENU_MISC */
#define CONF_QUIETMODE            12 /* CONF_MENU_MISC */
#define CONF_NOEXITFILECHECK      13 /* CONF_MENU_MISC */
#define CONF_PERCENTOFF           14 /* CONF_MENU_MISC */
#define CONF_CONTESTPRIORITY      15 /* CONF_MENU_BUFF "DES,OGR,RC5" */

#define CONF_CPUTYPE              16 /* CONF_MENU_PERF */
#define CONF_NUMCPU               17 /* CONF_MENU_PERF 0 ... */
#define CONF_NICENESS             18 /* CONF_MENU_PERF priority */

#define CONF_LOGNAME              19 /* CONF_MENU_LOG */
#define CONF_MESSAGELEN           20 /* CONF_MENU_LOG */
#define CONF_SMTPSRVR             21 /* CONF_MENU_LOG */
#define CONF_SMTPPORT             22 /* CONF_MENU_LOG */
#define CONF_SMTPFROM             23 /* CONF_MENU_LOG */
#define CONF_SMTPDEST             24 /* CONF_MENU_LOG */

#define CONF_OFFLINEMODE          25 /* CONF_MENU_NET runoffline not runbuffers */
#define CONF_NETTIMEOUT           26 /* CONF_MENU_NET */
#define CONF_UUEHTTPMODE          27 /* CONF_MENU_NET */
#define CONF_AUTOFINDKS           28 /* CONF_MENU_NET autofindkeyserver */
#define CONF_KEYSERVNAME          29 /* CONF_MENU_NET */
#define CONF_KEYSERVPORT          30 /* CONF_MENU_NET */
#define CONF_NOFALLBACK           31 /* CONF_MENU_NET */
#define CONF_FWALLHOSTNAME        32 /* CONF_MENU_NET */
#define CONF_FWALLHOSTPORT        33 /* CONF_MENU_NET */
#define CONF_FWALLUSERNAME        34 /* CONF_MENU_NET */
#define CONF_FWALLPASSWORD        35 /* CONF_MENU_NET */

#define CONF_LURKMODE             36 /* CONF_MENU_NET */
#define CONF_CONNIFACEMASK        37 /* CONF_MENU_NET */
#define CONF_DIALWHENNEEDED       38 /* CONF_MENU_NET */
#define CONF_CONNPROFILE          39 /* CONF_MENU_NET */
#define CONF_CONNSTARTCMD         40 /* CONF_MENU_NET */
#define CONF_CONNSTOPCMD          41 /* CONF_MENU_NET */

#define CONF_OPTION_COUNT         42

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
  int optionscreen;            //screen to appear on CONF_MENU_*
  int type;                    //type: CONF_TYPE_*
  void *thevariable;           //pointer to the variable
  const char **choicelist;     //pointer to the char* array of choices
                               //(used for numeric responses)
  s32 choicemin;               //minimum choice number
  s32 choicemax;               //maximum choice number
  const char *disabledtext;    //is NULL if not disabled
  };
extern struct optionstruct conf_options[CONF_OPTION_COUNT];

/* ---------------------------------------------------------------- */

#endif /* __CONFOPT_H__ */

