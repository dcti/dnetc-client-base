/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 

#ifndef __CONFOPT_H__
#define __CONFOPT_H__ "@(#)$Id: confopt.h,v 1.13.2.2 1999/04/24 07:34:59 jlawson Exp $"

#define CONF_MENU_MISC_PLACEHOLDER  0
#define CONF_ID                     1 /* CONF_MENU_MISC */
#define CONF_COUNT                  2 /* CONF_MENU_MISC -1 == runbuffers */
#define CONF_HOURS                  3 /* CONF_MENU_MISC */
#define CONF_PAUSEFILE              4 /* CONF_MENU_MISC */
#define CONF_QUIETMODE              5 /* CONF_MENU_MISC */
#define CONF_NOEXITFILECHECK        6 /* CONF_MENU_MISC */
#define CONF_PERCENTOFF             7 /* CONF_MENU_MISC */
#define CONF_CONTESTPRIORITY        8 /* CONF_MENU_BUFF "DES,OGR,RC5" */

#define CONF_MENU_BUFF_PLACEHOLDER  9
#define CONF_NODISK                10  /* CONF_MENU_BUFF */
#define CONF_INBUFFERBASENAME      11  /* CONF_MENU_BUFF */
#define CONF_OUTBUFFERBASENAME     12  /* CONF_MENU_BUFF */
#define CONF_CHECKPOINT            13  /* CONF_MENU_BUFF */

#define CONF_OFFLINEMODE           14  /* CONF_MENU_BUFF */
#define CONF_MENU_NET_PLACEHOLDER  15  /* CONF_MENU_BUFF */
#define CONF_REMOTEUPDATEDISABLED  16  /* CONF_MENU_BUFF */
#define CONF_REMOTEUPDATEDIR       17  /* CONF_MENU_BUFF */
#define CONF_FREQUENT              18  /* CONF_MENU_BUFF */
#define CONF_PREFERREDBLOCKSIZE    19  /* CONF_MENU_BUFF */
#define CONF_THRESHOLDI            20  /* CONF_MENU_BUFF */

#define CONF_MENU_PERF_PLACEHOLDER 21
#define CONF_CPUTYPE               22 /* CONF_MENU_PERF */
#define CONF_NUMCPU                23 /* CONF_MENU_PERF 0 ... */
#define CONF_NICENESS              24 /* CONF_MENU_PERF priority */

#define CONF_MENU_LOG_PLACEHOLDER  25
#define CONF_LOGTYPE               26 /* CONF_MENU_LOG */
#define CONF_LOGNAME               27 /* CONF_MENU_LOG */
#define CONF_LOGLIMIT              28 /* CONF_MENU_LOG */
#define CONF_MESSAGELEN            29 /* CONF_MENU_LOG */
#define CONF_SMTPSRVR              30 /* CONF_MENU_LOG */
#define CONF_SMTPPORT              31 /* CONF_MENU_LOG */
#define CONF_SMTPFROM              32 /* CONF_MENU_LOG */
#define CONF_SMTPDEST              33 /* CONF_MENU_LOG */

#define CONF_NETTIMEOUT            34 /* CONF_MENU_NET */
#define CONF_AUTOFINDKS            35 /* CONF_MENU_NET */
#define CONF_KEYSERVNAME           36 /* CONF_MENU_NET */
#define CONF_KEYSERVPORT           37 /* CONF_MENU_NET */
#define CONF_NOFALLBACK            38 /* CONF_MENU_NET */
#define CONF_FWALLTYPE             39 /* CONF_MENU_NET */
#define CONF_FWALLHOSTNAME         40 /* CONF_MENU_NET */
#define CONF_FWALLHOSTPORT         41 /* CONF_MENU_NET */
#define CONF_FWALLUSERNAME         42 /* CONF_MENU_NET */
#define CONF_FWALLPASSWORD         43 /* CONF_MENU_NET */
#define CONF_FORCEHTTP             44 /* CONF_MENU_NET */
#define CONF_FORCEUUE              45 /* CONF_MENU_NET */

#define CONF_LURKMODE              46 /* CONF_MENU_NET */
#define CONF_CONNIFACEMASK         47 /* CONF_MENU_NET */
#define CONF_DIALWHENNEEDED        48 /* CONF_MENU_NET */
#define CONF_CONNPROFILE           49 /* CONF_MENU_NET */
#define CONF_CONNSTARTCMD          50 /* CONF_MENU_NET */
#define CONF_CONNSTOPCMD           51 /* CONF_MENU_NET */

#define CONF_OPTION_COUNT          52

#define CONF_MENU_MAIN             0
#define CONF_MENU_BUFF             1
#define CONF_MENU_LOG              2
#define CONF_MENU_NET              3
#define CONF_MENU_PERF             4
#define CONF_MENU_MISC             5

#define CONF_TYPE_MENU             0
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

