/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 

#ifndef __CONFOPT_H__
#define __CONFOPT_H__ "@(#)$Id: confopt.h,v 1.22 2000/06/02 06:24:55 jlawson Exp $"

enum /* anonymous */
{
  CONF_MENU_MISC_PLACEHOLDER =0, /* 0 */
  CONF_ID                      , /* CONF_MENU_MISC */
  CONF_COUNT                   , /* CONF_MENU_MISC -1 == runbuffers */
  CONF_HOURS                   , /* CONF_MENU_MISC */
  CONF_PAUSEFILE               , /* CONF_MENU_MISC */
  CONF_EXITFILE                , /* CONF_MENU_MISC */
  CONF_RESTARTONINICHANGE      , /* CONF_MENU_MISC */
  CONF_PAUSEPLIST              , /* CONF_MENU_MISC */
  CONF_PAUSEIFCPUTEMPHIGH      , /* CONF_MENU_MISC */ 
  CONF_CPUTEMPTHRESHOLDS       , /* CONF_MENU_MISC */
  CONF_PAUSEIFBATTERY          , /* CONF_MENU_MISC */
  CONF_QUIETMODE               , /* CONF_MENU_MISC */
  CONF_PERCENTOFF              , /* CONF_MENU_MISC */
  CONF_COMPLETIONSOUNDON       , /* CONF_MENU_MISC */

  CONF_MENU_BUFF_PLACEHOLDER   , /* 15 */
  CONF_NODISK                  , /* CONF_MENU_BUFF */
  CONF_INBUFFERBASENAME        , /* CONF_MENU_BUFF */
  CONF_OUTBUFFERBASENAME       , /* CONF_MENU_BUFF */
  CONF_CHECKPOINT              , /* CONF_MENU_BUFF */

  CONF_OFFLINEMODE             , /* CONF_MENU_BUFF */
  CONF_MENU_NET_PLACEHOLDER    , /* CONF_MENU_BUFF */
  CONF_REMOTEUPDATEDISABLED    , /* CONF_MENU_BUFF */
  CONF_REMOTEUPDATEDIR         , /* CONF_MENU_BUFF */
  CONF_LOADORDER               , /* CONF_MENU_BUFF "DES,OGR,RC5" */
  CONF_FREQUENT                , /* CONF_MENU_BUFF */
  CONF_PREFERREDBLOCKSIZE      , /* CONF_MENU_BUFF */
  CONF_THRESHOLDI              , /* CONF_MENU_BUFF */
  CONF_THRESHOLDT              , /* CONF_MENU_BUFF */

  CONF_MENU_PERF_PLACEHOLDER   , /* 28 */
  CONF_CPUTYPE                 , /* CONF_MENU_PERF */
  CONF_NUMCPU                  , /* CONF_MENU_PERF 0 ... */
  CONF_NICENESS                , /* CONF_MENU_PERF priority */

  CONF_MENU_LOG_PLACEHOLDER    , /* 32 */
  CONF_LOGTYPE                 , /* CONF_MENU_LOG */
  CONF_LOGNAME                 , /* CONF_MENU_LOG */
  CONF_LOGLIMIT                , /* CONF_MENU_LOG */
  CONF_MESSAGELEN              , /* CONF_MENU_LOG */
  CONF_SMTPSRVR                , /* CONF_MENU_LOG */
  CONF_SMTPPORT                , /* CONF_MENU_LOG */
  CONF_SMTPFROM                , /* CONF_MENU_LOG */
  CONF_SMTPDEST                , /* CONF_MENU_LOG */

  CONF_NETTIMEOUT              , /* CONF_MENU_NET */
  CONF_AUTOFINDKS              , /* CONF_MENU_NET */
  CONF_KEYSERVNAME             , /* CONF_MENU_NET */
  CONF_KEYSERVPORT             , /* CONF_MENU_NET */
  CONF_NOFALLBACK              , /* CONF_MENU_NET */
  CONF_FWALLTYPE               , /* CONF_MENU_NET */
  CONF_FWALLHOSTNAME           , /* CONF_MENU_NET */
  CONF_FWALLHOSTPORT           , /* CONF_MENU_NET */
  CONF_FWALLUSERNAME           , /* CONF_MENU_NET */
  CONF_FWALLPASSWORD           , /* CONF_MENU_NET */
  CONF_FORCEHTTP               , /* CONF_MENU_NET */
  CONF_FORCEUUE                , /* CONF_MENU_NET */

  CONF_LURKMODE                , /* CONF_MENU_NET */
  CONF_CONNIFACEMASK           , /* CONF_MENU_NET */
  CONF_DIALWHENNEEDED          , /* CONF_MENU_NET */
  CONF_CONNPROFILE             , /* CONF_MENU_NET */
  CONF_CONNSTARTCMD            , /* CONF_MENU_NET */
  CONF_CONNSTOPCMD             , /* CONF_MENU_NET */

  CONF_OPTION_COUNT            
};

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
#define CONF_TYPE_IARRAY           6

/* ---------------------------------------------------------------- */

struct optionstruct
{
  int index;                   // CONF_x
  const char *description;     //description of the option
  const char *defaultsetting;  //default setting
  const char *comments;        //additional comments
  int optionscreen;            //screen to appear on CONF_MENU_*
  int type;                    //type: CONF_TYPE_*
  void *thevariable;           //pointer to the variable
  const char **choicelist;     //pointer to the char* array of choices
                               //(used for numeric responses)
  int choicemin;               //minimum choice number
  int choicemax;               //maximum choice number
  const char *disabledtext;    //is NULL if not disabled
  int *iarray;                 //int array, one int per contest, in cont order
};
extern struct optionstruct conf_options[CONF_OPTION_COUNT];

/* ---------------------------------------------------------------- */

#endif /* __CONFOPT_H__ */

