/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confmenu_cpp(void) {
return "@(#)$Id: confmenu.cpp,v 1.37 1999/04/08 19:04:50 cyp Exp $"; }

/* ----------------------------------------------------------------------- */

#include "cputypes.h" // CLIENT_OS, s32
#include "console.h"  // ConOutErr()
#include "client.h"   // Client class
#include "baseincs.h" // strlen() etc
#include "cmpidefs.h" // strcmpi()
#include "logstuff.h" // LogScreenRaw()
#include "selcore.h"  // GetCoreNameFromCoreType()
#include "util.h"     // projectmap_*()
#include "base64.h"   // base64_[en|de]code()
#include "lurk.h"     // lurk stuff
#include "triggers.h" // CheckExitRequestTriggerNoIO()
#include "network.h"  // base64_encode()
#include "confopt.h"  // the option table

/* ----------------------------------------------------------------------- */

#define MAX_MENUENTRIESPERSCREEN 18 /* max menu entries per screen */
static const char *CONFMENU_CAPTION="RC5DES Client Configuration: %s\n"
"-----------------------------------------------------------------------\n";

/* ----------------------------------------------------------------------- */
       
static int findmenuoption( int menu, unsigned int menuposition )
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
  unsigned int tpos, mpos = 0;

  for (tpos=0; tpos < CONF_OPTION_COUNT; tpos++)
  {
    if (conf_options[tpos].optionscreen==menu)
    {
      if ((++mpos) == menuposition)
      {      
#ifndef GREGH /* this is only for greg! :) */
        if (conf_options[tpos].disabledtext != NULL ||
            conf_options[tpos].thevariable == NULL)
          return -1;
#endif          
        return (int)tpos;
      }
    }
  }
  return -1;
}

/* ------------------------------------------------------------------------ */

static int confopt_isstringblank( const char *string )
{
  register int len = ( string ? ( strlen( string )+1 ) : 0 );

  while (len)
    {
    len--;
    if ( isprint( string[len] ) && !isspace( string[len] ) )
      return 0;
    }
  return 1;
}

/* ----------------------------------------------------------------------- */

static void confopt_killwhitespace( char *string )
{
  char *opos, *ipos;
  ipos = opos = string;
  while ( *ipos )
    {
    if ( !isspace( *ipos ) )
      *opos++ = *ipos;
    ipos++;
    }
  *opos = 0;
  if ( strcmpi(string, "none") == 0 )
    string[0]=0;
  return;
}

/* ------------------------------------------------------------------------ */

static const char *menutable[] =
{
  "Project and Buffer Options",
  "Logging Options",
  "Network and Communication Options",
  "Performance and Processor Options",
  "Miscellaneous Options"
};

/* ----------------------------------------------------------------------- */

int Client::Configure( void )
//A return of 1 indicates to save the changed configuration
//A return of -1 indicates to NOT save the changed configuration
{
  int whichmenu;
  int returnvalue = 0;

  if (!ConIsScreen())
  {
    ConOutErr("Can't configure when stdin or stdout is redirected.\n");
    return -1;
  }
    
  // ---- Set all stuff that doesn't change during config ----   
  // note that some options rely on others, so watch the init order

  /* ------------------- CONF_MENU_BUFF ------------------ */  
        
  s32 threshold = (s32)inthreshold[0];
  if (strcmpi(id,"rc5@distributed.net") == 0)
    id[0] = 0; /*is later converted back to 'rc5@distributed.net' */
  conf_options[CONF_ID].thevariable=(char *)(&id[0]);
  conf_options[CONF_THRESHOLDI].thevariable=&threshold;
  conf_options[CONF_FREQUENT].thevariable=&connectoften;
  conf_options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
  conf_options[CONF_NODISK].thevariable=&nodiskbuffers;
  conf_options[CONF_INBUFFERBASENAME].thevariable=(char *)(&in_buffer_basename[0]);
  conf_options[CONF_OUTBUFFERBASENAME].thevariable=(char *)(&out_buffer_basename[0]);
  conf_options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0]);
  char loadorder[64];
  strcpy(loadorder, projectmap_expand( loadorder_map ) );
  conf_options[CONF_CONTESTPRIORITY].thevariable=(char *)(&loadorder[0]);
  conf_options[CONF_REMOTEUPDATEDIR].thevariable=(char *)(&remote_update_dir[0]);

  /* ------------------- CONF_MENU_LOG  ------------------ */  
  
  conf_options[CONF_LOGNAME].thevariable=(char *)(&logname[0]);
  conf_options[CONF_MESSAGELEN].thevariable=&messagelen;
  conf_options[CONF_SMTPSRVR].thevariable=(char *)(&smtpsrvr[0]);
  conf_options[CONF_SMTPPORT].thevariable=&smtpport;
  conf_options[CONF_SMTPFROM].thevariable=(char *)(&smtpfrom[0]);
  conf_options[CONF_SMTPDEST].thevariable=(char *)(&smtpdest[0]);
  conf_options[CONF_SMTPFROM].defaultsetting=
  conf_options[CONF_SMTPDEST].defaultsetting=(char *)conf_options[CONF_ID].thevariable;

  /* ------------------- CONF_MENU_NET  ------------------ */  

  s32 autofindks = (autofindkeyserver!=0);
  conf_options[CONF_OFFLINEMODE].thevariable=&offlinemode;
  conf_options[CONF_NETTIMEOUT].thevariable=&nettimeout;
  conf_options[CONF_UUEHTTPMODE].thevariable=&uuehttpmode;
  conf_options[CONF_KEYSERVNAME].thevariable=(char *)(&keyproxy[0]);
  conf_options[CONF_KEYSERVPORT].thevariable=&keyport;
  conf_options[CONF_AUTOFINDKS].thevariable=&autofindks;
  conf_options[CONF_NOFALLBACK].thevariable=&nofallback;

  conf_options[CONF_LURKMODE].thevariable=
  conf_options[CONF_CONNIFACEMASK].thevariable=
  conf_options[CONF_DIALWHENNEEDED].thevariable=
  conf_options[CONF_CONNPROFILE].thevariable=
  conf_options[CONF_CONNSTARTCMD].thevariable=
  conf_options[CONF_CONNSTOPCMD].thevariable=NULL;

  #if defined(LURK)
  int dupcap = dialup.GetCapabilityFlags();
  if ((dupcap & (CONNECT_LURK|CONNECT_LURKONLY))!=0)
  {
    conf_options[CONF_LURKMODE].thevariable=&dialup.lurkmode;
  }
  if ((dupcap & CONNECT_IFACEMASK)!=0)
  {
    conf_options[CONF_CONNIFACEMASK].thevariable=&dialup.connifacemask;
  }
  if ((dupcap & CONNECT_DOD)!=0)
  {
    conf_options[CONF_DIALWHENNEEDED].thevariable=&dialup.dialwhenneeded;
    if ((dupcap & CONNECT_DODBYSCRIPT)!=0)
    {
      conf_options[CONF_CONNSTARTCMD].thevariable=&dialup.connstartcmd;
      conf_options[CONF_CONNSTOPCMD].thevariable=&dialup.connstopcmd;
    }
    if ((dupcap & CONNECT_DODBYPROFILE)!=0)
    {
      conf_options[CONF_CONNPROFILE].thevariable=&dialup.connprofile;
      conf_options[CONF_CONNPROFILE].choicemin = 
      conf_options[CONF_CONNPROFILE].choicemax = 0;
      unsigned int maxconn = 0;
      const char **connectnames = dialup.GetConnectionProfileList();
      if (connectnames) {
        while (connectnames[maxconn])
          maxconn++;
        }
      if (maxconn > 1) /* the first option is "", ie default */
      {
        connectnames[0] = "<Use Control Panel Setting>";
        conf_options[CONF_CONNPROFILE].choicemax = (s32)(maxconn-1);
        conf_options[CONF_CONNPROFILE].choicelist = connectnames;
      }
    }
  }
  #endif // if(LURK)
  conf_options[CONF_FWALLHOSTNAME].thevariable=(char *)(&httpproxy[0]);
  conf_options[CONF_FWALLHOSTPORT].thevariable=&httpport;
  struct { char username[128], password[128]; } userpass;
  userpass.username[0] = userpass.password[0] = 0;
  if (httpid[0] == 0)
    ; //nothing
  else if (uuehttpmode == 2 || uuehttpmode == 3)
  {
    char *p;
    if (strlen( httpid ) > 80) /* not rfc compliant (max 76) */
      userpass.username[0]=0;
    else if (base64_decode(userpass.username, httpid )!=0) 
      userpass.username[0]=0;                         /* bit errors */
    else if ((p = strchr( userpass.username, ':' )) == NULL)
      userpass.username[0]=0;                         /* wrong format */
    else
    {
      *p++ = 0;
      strcpy(userpass.password,p);
    }
  }
  else if (uuehttpmode == 4) 
    strcpy(userpass.username,httpid);
  else if (uuehttpmode == 5)
  {
    strcpy( userpass.username, httpid );
    char *p = strchr( userpass.username,':');
    if (p != NULL) 
    {
      *p++ = 0;
      strcpy( userpass.password, p );
    }
  }
  conf_options[CONF_FWALLUSERNAME].thevariable = (char *)(&userpass.username[0]);
  conf_options[CONF_FWALLPASSWORD].thevariable = (char *)(&userpass.password[0]);

  /* ------------------- CONF_MENU_PERF ------------------ */  

  conf_options[CONF_CPUTYPE].thevariable=NULL;
  conf_options[CONF_CPUTYPE].choicemax=0;
  conf_options[CONF_CPUTYPE].choicemin=0;
  const char *corename = GetCoreNameFromCoreType(0);
  if (corename && *corename)
  {
    static const char *cputypetable[10];
    unsigned int tablesize = 2;
    cputypetable[0]="Autodetect";
    cputypetable[1]=corename;
    do
    {
      corename = GetCoreNameFromCoreType(tablesize-1);
      if (!corename || !*corename)
        break;
      cputypetable[tablesize++]=corename;
    } while (tablesize<((sizeof(cputypetable)/sizeof(cputypetable[0]))));
    conf_options[CONF_CPUTYPE].choicelist=&cputypetable[1];
    conf_options[CONF_CPUTYPE].choicemin=-1;
    conf_options[CONF_CPUTYPE].choicemax=tablesize-2;
    conf_options[CONF_CPUTYPE].thevariable=&cputype;
  }
  conf_options[CONF_NICENESS].thevariable = &priority;
  conf_options[CONF_NUMCPU].thevariable = &numcpu;

  /* ------------------- CONF_MENU_MISC ------------------ */  

  conf_options[CONF_COUNT].thevariable=&blockcount;
  conf_options[CONF_HOURS].thevariable=&minutes;
  conf_options[CONF_QUIETMODE].thevariable=&quietmode;
  conf_options[CONF_NOEXITFILECHECK].thevariable=&noexitfilecheck;
  conf_options[CONF_PERCENTOFF].thevariable=&percentprintingoff;
  conf_options[CONF_PAUSEFILE].thevariable=(char *)(&pausefile[0]);

  /* --------------------------------------------------------- */
  
  while (returnvalue == 0)
  {
    ConClear();
    LogScreenRaw(CONFMENU_CAPTION, "");
    for (whichmenu = 1;
        whichmenu <= (int)(sizeof(menutable)/sizeof(menutable[0]));
        whichmenu++)
      LogScreenRaw(" %u) %s\n",whichmenu, menutable[whichmenu-1]);
    LogScreenRaw("\n 9) Discard settings and exit"
                 "\n 0) Save settings and exit\n\n");

    if (confopt_isstringblank(id) || strcmpi(id,"rc5@distributed.net")==0)
      LogScreenRaw("Note: You have not yet provided a distributed.net ID.\n"
              "       Please go to the '%s' and set it.\n\n",menutable[0]);

    LogScreenRaw("Choice --> ");
    char chbuf[6];
    ConInStr(chbuf, 2, 0);
    whichmenu = ((strlen(chbuf)==1 && isdigit(chbuf[0]))?(atoi(chbuf)):(-1));
    
    if (CheckExitRequestTriggerNoIO() || whichmenu==9)
      returnvalue = -1; //Breaks and tells it NOT to save
    else if (whichmenu < 0) 
      ; /* nothing - ignore it */
    else if (whichmenu == 0)
      returnvalue = 1; //Breaks and tells it to save
    else if (whichmenu<=(int)(sizeof(menutable)/sizeof(menutable[0])))
    {
      int userselection = 0;
      int redoselection = -1;

      // note: don't return or break from inside
      // the loop. Let it fall through instead. - cyp
      while (userselection >= 0)
      {
        char parm[128];
        char *p;

        // there are two ways to deal with the keyport validation "problematik".
        // (Port redirection/mapping per datapipe or whatever is dealt with
        //  by the client as if the target host were a personal proxy, 
        //  so they are NOT an issue from the client's perspective.)
        //
        // - either we totally uncouple firewall and keyserver parameters
        //   and don't do any validation of the keyport based on the 
        //   firewall method,
        // - or we bind them tightly under the assumption that anyone using
        //   behind a firewall is not going to be connecting to a personal
        //   proxy outside the firewall, ie keyproxy is _always_ a dnet host.
        //
        // I opted for the former:
        //   a) The network layer will use a default port # if the port # is 
        //      zero.  So.... why not leave it at zero? 
        //   b) Validation should not be a config issue. Anything can be
        //      modified in the ini itself and subsystems do their own 
        //      validation anyway. If users want to play, let them.
        //   c) If implementing a forced d.net host is preferred, it should 
        //      not be done here. Network::Open is better suited for that.
        //                                                            - cyp

        /* --------------- drop/pickup menu options ---------------- */

        if (whichmenu == CONF_MENU_BUFF)
        {
          conf_options[CONF_FREQUENT].disabledtext= 
                      ((!nodiskbuffers && !offlinemode)?(NULL):
                      ("n/a [no disk buffers/no net connectivity]"));

          conf_options[CONF_THRESHOLDI].disabledtext= 
                      ((!offlinemode)?(NULL):
                      ("n/a [network connectivity is disabled]") );

          conf_options[CONF_INBUFFERBASENAME].disabledtext=
          conf_options[CONF_OUTBUFFERBASENAME].disabledtext=
          conf_options[CONF_CHECKPOINT].disabledtext=
          conf_options[CONF_PREFERREDBLOCKSIZE].disabledtext=
                      ((!nodiskbuffers)?(NULL):
                      ("n/a [disk buffers are disabled]"));
        }
        else if (whichmenu == CONF_MENU_LOG)
        {
          conf_options[CONF_SMTPSRVR].disabledtext=
          conf_options[CONF_SMTPPORT].disabledtext=
          conf_options[CONF_SMTPDEST].disabledtext=
          conf_options[CONF_SMTPFROM].disabledtext=
                      ((!offlinemode && messagelen > 0)?(NULL):
                      ("n/a [no networking or message length is zero]"));
        }
        else if (whichmenu == CONF_MENU_NET)
        {
          unsigned int n;
          p = NULL;

          if (offlinemode)
            p = "n/a [requires networking]";
          for (n = 0; n < CONF_OPTION_COUNT; n++)
          {
            if (conf_options[n].thevariable != NULL &&
                conf_options[n].optionscreen == CONF_MENU_NET &&
                conf_options[n].thevariable != ((void *)&offlinemode))
              conf_options[n].disabledtext = (const char *)p;
          }
          if (!offlinemode)
          {
            if (uuehttpmode<2)
            {
              conf_options[CONF_FWALLHOSTNAME].disabledtext=
              conf_options[CONF_FWALLHOSTPORT].disabledtext=
              conf_options[CONF_FWALLUSERNAME].disabledtext=
              conf_options[CONF_FWALLPASSWORD].disabledtext=
                    "n/a [inappropriate for encoding method]";
            }
            else if (uuehttpmode!=2 && uuehttpmode!=3 && uuehttpmode!=5)
            {
              conf_options[CONF_FWALLPASSWORD].disabledtext=
                      "n/a [inappropriate for encoding method]";
            }
            if (autofindks)
            {
              conf_options[CONF_NOFALLBACK].disabledtext= //can't fallback to self
              conf_options[CONF_KEYSERVNAME].disabledtext = 
                      "n/a [requires non-distributed.net host]";
            }
            #ifdef LURK
            if (!dialup.dialwhenneeded || conf_options[CONF_DIALWHENNEEDED].thevariable==NULL)
            {
              conf_options[CONF_CONNPROFILE].disabledtext=
              conf_options[CONF_CONNSTARTCMD].disabledtext=
              conf_options[CONF_CONNSTOPCMD].disabledtext=
              "n/a [requires dial-on-demand support]";
            }
            #endif
          }
        }

         
        /* -------------------- display menu -------------------------- */
        
        ConClear();
        LogScreenRaw(CONFMENU_CAPTION, menutable[whichmenu-1]);
        
        unsigned int menuoption;
        for (menuoption=1;menuoption<MAX_MENUENTRIESPERSCREEN;menuoption++)
        {
          int seloption = findmenuoption( whichmenu, menuoption );
          if (seloption >= 0)
          {
            p = NULL;

            if (conf_options[seloption].thevariable == NULL)
            {
              p = "n/a [not available on this platform]";
            }
            else if (conf_options[seloption].disabledtext != NULL)
            {
              p = (char *)conf_options[seloption].disabledtext;
            }
            else if (conf_options[seloption].type==CONF_TYPE_ASCIIZ)
            {
              p = (char *)conf_options[seloption].thevariable;
            }
            else if (conf_options[seloption].type==CONF_TYPE_PASSWORD)
            {
              int i = strlen((char *)conf_options[seloption].thevariable);
              memset(parm, '*', i);
              parm[i] = 0;
              p = parm;
            }
            else if (conf_options[seloption].type==CONF_TYPE_TIMESTR)
            {
              long t = (long)*((s32 *)conf_options[seloption].thevariable);
              sprintf(parm, "%ld:%02u", (t/60), 
                             (unsigned int)(((t<0)?(-t):(t))%60) );
              p = parm;
            }
            else if (conf_options[seloption].type==CONF_TYPE_INT)
            {
              long thevar = (long)*(s32 *)conf_options[seloption].thevariable;
              if ((conf_options[seloption].choicelist != NULL) &&
                   (thevar >= (long)(conf_options[seloption].choicemin)) &&
                   (thevar <= (long)(conf_options[seloption].choicemax)) )
              {
                p = (char *)conf_options[seloption].choicelist[thevar];
              }
              else if (thevar == (long)(atoi(conf_options[seloption].defaultsetting)))
              {
                p = (char *)conf_options[seloption].defaultsetting;
              }
              else
              {
                sprintf(parm, "%li", thevar );
                p = parm;
              }
            }
            else if (conf_options[seloption].type == CONF_TYPE_BOOL)
            {
              p = (char *)(((*(s32 *)conf_options[seloption].thevariable)?("yes"):("no")));
            }
            if (p)
            {
              char parm2[128];
              unsigned int optlen = sprintf(parm2, "%2u) %s ==> ",
                  menuoption, conf_options[seloption].description );
              strncpy( &parm2[optlen], p, (80-optlen) );
              parm2[79]=0;
              LogScreenRaw( "%s\n", parm2 );
            }
          }
        }
    
        /* -------------------- get user selection -------------------- */

        if (redoselection >= 0)
        {
          userselection = redoselection;
          redoselection = -1;
        }
        else
        {
          LogScreenRaw("\n 0) Return to main menu\n\nChoice --> ");
          ConInStr( parm, 4, 0 );
          userselection = atoi( parm );
          
          if (userselection == 0 || CheckExitRequestTriggerNoIO())
            userselection = -2;
          else if (userselection > 0 )
          {
            userselection = findmenuoption(whichmenu,userselection); //-1 if !found
            if (userselection >= 0)
            {
              if (conf_options[userselection].disabledtext != NULL ||
                  conf_options[userselection].thevariable == NULL)
                userselection = -1;
            }
          }
          else
          {
            userselection = -1;
          }
        }
        
        /* -- display user selection in detail and get new value --- */
    
        long newval_d = 0;
        char *newval_z = "";
        int newval_isok = 1;
    
        if ( userselection >= 0 )
        {
          ConClear(); 
          LogScreenRaw(CONFMENU_CAPTION, menutable[whichmenu-1]);
          LogScreenRaw("\n%s:\n\n", conf_options[userselection].description );

          p = (char *)conf_options[userselection].comments;
          while (strlen(p) > (sizeof(parm)-1))
          {
            strncpy(parm, p, (sizeof(parm)-1));
            parm[(sizeof(parm)-1)] = 0;
            LogScreenRaw("%s", parm);
            p += (sizeof(parm)-1);
          }
          LogScreenRaw("%s\n",p);
    
          if ( conf_options[userselection].type==CONF_TYPE_ASCIIZ || 
               conf_options[userselection].type==CONF_TYPE_INT ||
               conf_options[userselection].type==CONF_TYPE_PASSWORD )
          {
            p = "";
            char defaultbuff[30];
            int coninstrmode = CONINSTR_BYEXAMPLE;
            
            if (conf_options[userselection].choicelist !=NULL)
            {
              long selmin = conf_options[userselection].choicemin;
              long selmax = conf_options[userselection].choicemax;
              long listpos;
              for ( listpos = selmin; listpos <= selmax; listpos++)
                LogScreenRaw("  %2ld) %s\n", listpos, 
                      conf_options[userselection].choicelist[listpos]);
            }
            if (conf_options[userselection].type==CONF_TYPE_PASSWORD)
            {
              int i = strlen((char *)conf_options[userselection].thevariable);
              memset(parm, '*', i);
              parm[i] = 0;
              coninstrmode = CONINSTR_ASPASSWORD;
            }
            else if (conf_options[userselection].type==CONF_TYPE_ASCIIZ)
            {
              strcpy(parm, (char *)conf_options[userselection].thevariable);
              p = (char *)(conf_options[userselection].defaultsetting);
            }
            else //if (conf_options[userselection].type==CONF_TYPE_INT)
            {
              sprintf(parm, "%li", (long)*(s32 *)conf_options[userselection].thevariable);
              sprintf(defaultbuff, "%li", atol(conf_options[userselection].defaultsetting));
              p = defaultbuff;
            }
            LogScreenRaw("Default Setting: %s\n"
                         "Current Setting: %s\n"
                         "New Setting --> ", p, parm );

            ConInStr( parm, 64 /*sizeof(parm)*/, coninstrmode );

            if (CheckExitRequestTriggerNoIO())
              userselection = -2;
            else if (conf_options[userselection].type==CONF_TYPE_INT)
            {
              p = parm;
              int i = 0;
              while (isspace(*p))
                p++;
              while (*p)
              {
                if ((i==0 && (*p=='-' || *p == '+')) || isdigit(*p))
                  parm[i++]=*p++;
                else
                {
                  newval_isok = 0;
                  break;
                }
              }
              parm[i] = 0;
              if (newval_isok)
              {
                newval_d = atol(parm);
                long selmin = conf_options[userselection].choicemin;
                long selmax = conf_options[userselection].choicemax;
                if ((selmin != 0 || selmax != 0) && 
                  (newval_d < selmin || newval_d > selmax))
                newval_isok = 0;
              }
            }
            else //if (conf_options[userselection].type==CONF_TYPE_ASCIIZ)
            {
              if (parm[0]!=0)
              {
                p = &parm[strlen(parm)-1];
                while (p >= &parm[0] && isspace(*p))
                  *p-- = 0;
                p = parm;
                while (*p && isspace(*p))
                  p++;
                if (p > &parm[0])
                  strcpy( parm, p );
              }
              newval_z = parm;
              newval_isok = 1;

              if (parm[0] != 0 && conf_options[userselection].choicemax != 0 && 
                  conf_options[userselection].choicelist) /* int *and* asciiz */
              {
                newval_d=atoi(parm);
                if ( ((newval_d > 0) || (parm[0] == '0')) &&
                   (newval_d <= conf_options[userselection].choicemax) )
                {
                  strncpy(parm, conf_options[userselection].choicelist[newval_d], sizeof(parm));
                  parm[sizeof(parm)-1]=0; 
                  if (newval_d == 0 && userselection == CONF_CONNPROFILE)
                    parm[0]=0;
                }
              }
            }
          }
          else if (conf_options[userselection].type == CONF_TYPE_TIMESTR)
          {
            long t = (long)*((s32 *)conf_options[userselection].thevariable);
            sprintf(parm,"%ld:%02u", (t/60), 
                             (unsigned int)(((t<0)?(-t):(t))%60) );
            LogScreenRaw("Default Setting: %s\n"
                         "Current Setting: %s\n"
                         "New Setting --> ",
                         conf_options[userselection].defaultsetting, parm );
            
            ConInStr( parm, 10, CONINSTR_BYEXAMPLE );
            
            if (CheckExitRequestTriggerNoIO())
              userselection = -2;
            else 
            {
              if (parm[0]!=0)
              {
                p = &parm[strlen(parm)-1];
                while (p >= &parm[0] && isspace(*p))
                  *p-- = 0;
                p = parm;
                while (*p && isspace(*p))
                  p++;
                if (p > &parm[0])
                  strcpy( parm, p );
              }
              if (parm[0]!=0)
              {
                int h=0, m=0, pos, isok = 0, dotpos=0;
                if (isdigit(parm[0]))
                {
                  isok = 1;
                  for (pos = 0; parm[pos] != 0; pos++)
                  {
                    if (!isdigit(parm[pos]))
                    {
                      if (dotpos != 0 || (parm[pos] != ':' && parm[pos] != '.'))
                      {
                        isok = 0;
                        break;
                      }
                      dotpos = pos;
                    }
                  }
                  if (isok)
                  {
                    if ((h = atoi( parm )) < 0)
                      isok = 0;
                    //else if (h > 23)
                    //  isok = 0;
                    else if (dotpos == 0)
                      isok = 0;
                    else if (strlen(&parm[dotpos+1]) != 2)
                      isok = 0;
                    else if (((m = atoi(&parm[dotpos+1])) > 59))
                      isok = 0;
                  }
                } //if (isdigit(parm[0]))
                if (isok)
                  newval_d = ((h*60)+m);
                else
                  newval_isok = 0;
              } //if (parm[0]!=0)
            } //if (CheckExitRequestTriggerNoIO()) else ...
          }
          else if (conf_options[userselection].type==CONF_TYPE_BOOL)
          {
            sprintf(parm, "%s", *(s32 *)conf_options[userselection].thevariable?"yes":"no");
            LogScreenRaw("Default Setting: %s\n"
                         "Current Setting: %s\n"
                         "New Setting --> ",
                         *(conf_options[userselection].defaultsetting)=='0'?"no":"yes", 
                         parm );
            parm[1] = 0;
            ConInStr( parm, 2, CONINSTR_BYEXAMPLE|CONINSTR_ASBOOLEAN );
            if (CheckExitRequestTriggerNoIO())
              userselection = -2;
            else if (parm[0]=='y' || parm[0]=='Y')
              newval_d = 1;
            else if (parm[0]=='n' || parm[0]=='N') 
              newval_d = 0;
            else
              newval_isok = 0;
          }
          else
          {
            userselection = -1;
          }
        }
          
        /* --------------- have modified value, so assign -------------- */

        if (userselection >= 0 && !newval_isok)
        {
          ConBeep();
          redoselection = userselection;
        }
        else if (userselection >= 0 && newval_isok)
        {
          // DO NOT TOUCH ANY VARIABLE EXCEPT THE SELECTED ONE
          // (unless those variables are not menu options)
          // DO IT AFTER ALL MENU DRIVEN CONFIG IS FINISHED (see end)
          
          if (userselection == CONF_ID || userselection == CONF_KEYSERVNAME ||
            userselection == CONF_SMTPFROM || userselection == CONF_SMTPSRVR ||
            userselection == CONF_FWALLHOSTNAME)
          {
            confopt_killwhitespace(parm);
          }
          if (conf_options[userselection].type==CONF_TYPE_ASCIIZ ||
              conf_options[userselection].type==CONF_TYPE_PASSWORD)
          {
            if (confopt_isstringblank(parm)) 
              parm[0] = 0;
            strncpy( (char *)conf_options[userselection].thevariable, parm, 
                     64 - 1 );
            ((char *)conf_options[userselection].thevariable)[64-1]=0;
          }
          else //bool or int types
          {
            *(s32 *)conf_options[userselection].thevariable = (s32)newval_d;
            if ( userselection == CONF_COUNT && newval_d < 0)
              blockcount = -1;
            else if (userselection == CONF_THRESHOLDI)
              inthreshold[0]=outthreshold[0]=inthreshold[1]=outthreshold[1]=newval_d;
            else if (userselection == CONF_NETTIMEOUT)
              nettimeout = ((newval_d<0)?(-1):((newval_d<5)?(5):(newval_d)));
          }
        } // if (userselection >= 0)
      } // while (userselection >= 0)
  
      /* ------------- */
    
      if (CheckExitRequestTriggerNoIO())
        returnvalue = -1;
    } // if (whichmenu<=(int)(sizeof(menutable)/sizeof(menutable[0])))
  } // while (returnvalue == 0)
    
  /* -- massage mapped options and dependancies back into place -- */

  if (returnvalue != -1)
  {
    if (id[0] == 0)
      strcpy(id, "rc5@distributed.net");

    autofindkeyserver = (autofindks!=0);

    if (nettimeout < 0)
      nettimeout = -1;
    else if (nettimeout < 5)
      nettimeout = 5;

    #ifdef LURK
    if (dialup.lurkmode != 1)
      connectoften=0;
    #endif

    projectmap_build(loadorder_map, loadorder );
    
    if (strlen(userpass.username) == 0 && strlen(userpass.password) == 0)
      httpid[0] = 0;
    else if (uuehttpmode == 2 || uuehttpmode == 3)
    {
      if (((strlen(userpass.username)+strlen(userpass.password)+4)*4/3) >
        sizeof(httpid)) /* too big. what should we do? */
        httpid[0] = 0; 
      else
      {
        strcat(userpass.username,":");
        strcat(userpass.username,userpass.password);
        base64_encode(httpid,userpass.username);
      }
    }
    else if (uuehttpmode == 4)
    {
      strcpy( httpid, userpass.username );
    }
    else if (uuehttpmode == 5)
    {
      strcat(userpass.username, ":");
      strcat(userpass.username, userpass.password);
      strncpy( httpid, userpass.username, sizeof( httpid )-1);
      httpid[sizeof( httpid )-1] = 0;
    }
  }

  //fini
    
  return returnvalue;
}

