// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confopt.h,v $
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

#define CONF_ID                    0
#define CONF_THRESHOLDI            1
#define CONF_THRESHOLDO            2 /* obsolete in menu */
#define CONF_THRESHOLDI2           3 /* obsolete in menu */
#define CONF_THRESHOLDO2           4 /* obsolete in menu */
#define CONF_COUNT                 5 /* now allows -1 == runbuffers */
#define CONF_HOURS                 6
#define CONF_TIMESLICE             7 /* obsolete */
#define CONF_NICENESS              8 /* priority */
#define CONF_LOGNAME               9
#define CONF_UUEHTTPMODE          10
#define CONF_KEYPROXY             11
#define CONF_KEYPORT              12
#define CONF_HTTPPROXY            13
#define CONF_HTTPPORT             14
#define CONF_HTTPID               15
#define CONF_CPUTYPE              16
#define CONF_MESSAGELEN           17
#define CONF_SMTPSRVR             18
#define CONF_SMTPPORT             19
#define CONF_SMTPFROM             20
#define CONF_SMTPDEST             21
#define CONF_NUMCPU               22 /* 0 ... */
#define CONF_CHECKPOINT           23
#define CONF_CHECKPOINT2          24 /* obsolete */
#define CONF_RANDOMPREFIX         25 /* obsolete in menu */
#define CONF_PREFERREDBLOCKSIZE   26 
#define CONF_PROCESSDES           27 /* obsolete */
#define CONF_QUIETMODE            28
#define CONF_NOEXITFILECHECK      29
#define CONF_PERCENTOFF           30
#define CONF_FREQUENT             31
#define CONF_NODISK               32
#define CONF_NOFALLBACK           33
#define CONF_CKTIME               34 /* obsolete */
#define CONF_NETTIMEOUT           35
#define CONF_EXITFILECHECKTIME    36 /* obsolete */
#define CONF_OFFLINEMODE          37 /* runoffline not runbuffers */
#define CONF_LURKMODE             38
#define CONF_RC5IN                39
#define CONF_RC5OUT               40
#define CONF_DESIN                41
#define CONF_DESOUT               42
#define CONF_PAUSEFILE            43
#define CONF_DIALWHENNEEDED       44
#define CONF_CONNECTNAME          45
#define OPTION_COUNT              46

/* ---------------------------------------------------------------- */

struct optionstruct
  {
  const char *name;            //name of the option in the .ini file
  const char *description;     //description of the option
  const char *defaultsetting;  //default setting
  const char *comments;        //additional comments
  s32 optionscreen;            //screen to appear on
  s32 type;                    //type: 0=other, 1=string, 2=integer, 3=bool (yes/no)
  s32 menuposition;            //number on that menu to appear as
  void *thevariable;           //pointer to the variable
  const char **choicelist;     //pointer to the char* array of choices
                               //(used for numeric responses)
  s32 choicemin;               //minimum choice number
  s32 choicemax;               //maximum choice number
  };

/* ---------------------------------------------------------------- */

extern struct optionstruct conf_options[OPTION_COUNT];
extern int    confopt_IsHostnameDNetHost( const char * hostname );
extern int    confopt_isstringblank( const char *string );
extern void   confopt_killwhitespace( char *string );

/* ---------------------------------------------------------------- */

#endif /* #ifndef __CONFOPT_H__ */
