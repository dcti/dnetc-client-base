/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confmenu_cpp(void) {
return "@(#)$Id: confmenu.cpp,v 1.41 1999/04/22 01:51:44 cyp Exp $"; }

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
static const char *CONFMENU_CAPTION="RC5DES Client Configuration: %s\n"
"--------------------------------------------------------------------------\n";

int Client::Configure( void ) /* returns >1==save, <1==DON'T save */
{
  if (!ConIsScreen())
  {
    ConOutErr("Can't configure when stdin or stdout is redirected.\n");
    return -1;
  }
    
  // ---- Set all stuff that doesn't change during config ----   
  // note that some options rely on others, so watch the init order

  /* ------------------- CONF_MENU_MISC ------------------ */  

  if (strcmpi(id,"rc5@distributed.net") == 0)
    id[0] = 0; /*is later converted back to 'rc5@distributed.net' */
  conf_options[CONF_ID].thevariable=(char *)(&id[0]);
  conf_options[CONF_COUNT].thevariable=&blockcount;
  conf_options[CONF_HOURS].thevariable=&minutes;
  conf_options[CONF_QUIETMODE].thevariable=&quietmode;
  conf_options[CONF_NOEXITFILECHECK].thevariable=&noexitfilecheck;
  conf_options[CONF_PERCENTOFF].thevariable=&percentprintingoff;
  conf_options[CONF_PAUSEFILE].thevariable=(char *)(&pausefile[0]);
  char loadorder[64];
  strcpy(loadorder, projectmap_expand( loadorder_map ) );
  conf_options[CONF_CONTESTPRIORITY].thevariable=(char *)(&loadorder[0]);

  /* ------------------- CONF_MENU_BUFF ------------------ */  

  conf_options[CONF_NODISK].thevariable=&nodiskbuffers;
  conf_options[CONF_INBUFFERBASENAME].thevariable=(char *)(&in_buffer_basename[0]);
  conf_options[CONF_OUTBUFFERBASENAME].thevariable=(char *)(&out_buffer_basename[0]);
  conf_options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0]);
  conf_options[CONF_OFFLINEMODE].thevariable=&offlinemode;
  conf_options[CONF_REMOTEUPDATEDISABLED].thevariable=&noupdatefromfile;
  conf_options[CONF_REMOTEUPDATEDIR].thevariable=(char *)(&remote_update_dir[0]);
  conf_options[CONF_FREQUENT].thevariable=&connectoften;
  conf_options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
  conf_options[CONF_THRESHOLDI].thevariable=(s32 *)&inthreshold[0];

  /* ------------------- CONF_MENU_LOG  ------------------ */  

  static const char *logtypes[] = {"none","no limit","restart","fifo","rotate"};
  char logkblimit[sizeof(logfilelimit)], logrotlimit[sizeof(logfilelimit)];
  s32 logtype = LOGFILETYPE_NOLIMIT;
  logkblimit[0] = logrotlimit[0] = '\0';
  
  if ( strcmp( logfiletype, "rotate" ) == 0)
  {
    logtype = LOGFILETYPE_ROTATE;
    strcpy( logrotlimit, logfilelimit );
  }
  else 
  {
    strcpy( logkblimit, logfilelimit );
    if ( logname[0] == '\0' || strcmp( logfiletype, "none" ) == 0 )
      logtype = LOGFILETYPE_NONE;
    else if (strcmp( logfiletype, "restart" ) == 0)
      logtype = LOGFILETYPE_RESTART;
    else if (strcmp( logfiletype, "fifo" ) == 0)
      logtype = LOGFILETYPE_FIFO;
  }
  
  conf_options[CONF_LOGTYPE].thevariable=&logtype;
  conf_options[CONF_LOGTYPE].choicelist=&logtypes[0];
  conf_options[CONF_LOGTYPE].choicemax=(s32)((sizeof(logtypes)/sizeof(logtypes[0]))-1);
  conf_options[CONF_LOGNAME].thevariable=(char *)(&logname[0]);
  conf_options[CONF_LOGLIMIT].thevariable=(char *)(&logkblimit[0]);
  conf_options[CONF_MESSAGELEN].thevariable=&messagelen;
  conf_options[CONF_SMTPSRVR].thevariable=(char *)(&smtpsrvr[0]);
  conf_options[CONF_SMTPPORT].thevariable=&smtpport;
  conf_options[CONF_SMTPFROM].thevariable=(char *)(&smtpfrom[0]);
  conf_options[CONF_SMTPDEST].thevariable=(char *)(&smtpdest[0]);
  conf_options[CONF_SMTPFROM].defaultsetting=(char *)conf_options[CONF_ID].thevariable;
  conf_options[CONF_SMTPDEST].defaultsetting=(char *)conf_options[CONF_ID].thevariable;

  /* ------------------- CONF_MENU_NET  ------------------ */  

  conf_options[CONF_NETTIMEOUT].thevariable=&nettimeout;
  s32 autofindks = (autofindkeyserver!=0);
  conf_options[CONF_AUTOFINDKS].thevariable=&autofindks;
  conf_options[CONF_KEYSERVNAME].thevariable=(char *)(&keyproxy[0]);
  conf_options[CONF_KEYSERVPORT].thevariable=&keyport;
  conf_options[CONF_NOFALLBACK].thevariable=&nofallback;

  #define UUEHTTPMODE_UUE      1
  #define UUEHTTPMODE_HTTP     2
  #define UUEHTTPMODE_UUEHTTP  3
  #define UUEHTTPMODE_SOCKS4   4
  #define UUEHTTPMODE_SOCKS5   5
  static const char *fwall_types[] = { "none/transparent/mapped",
                                       "HTTP", "SOCKS4", "SOCKS5" };
  #define FWALL_TYPE_NONE      0
  #define FWALL_TYPE_HTTP      1
  #define FWALL_TYPE_SOCKS4    2
  #define FWALL_TYPE_SOCKS5    3
  s32 fwall_type = FWALL_TYPE_NONE;
  s32 use_http_regardless = (( uuehttpmode == UUEHTTPMODE_HTTP
                            || uuehttpmode == UUEHTTPMODE_UUEHTTP)
                            && httpproxy[0] == '\0');
  s32 use_uue_regardless =  (  uuehttpmode == UUEHTTPMODE_UUE 
                            || uuehttpmode == UUEHTTPMODE_UUEHTTP);
  if (httpproxy[0])
  {                           
    if (uuehttpmode == UUEHTTPMODE_SOCKS4) 
      fwall_type = FWALL_TYPE_SOCKS4;
    else if (uuehttpmode == UUEHTTPMODE_SOCKS5) 
      fwall_type = FWALL_TYPE_SOCKS5;
    else if (uuehttpmode==UUEHTTPMODE_HTTP || uuehttpmode==UUEHTTPMODE_UUEHTTP)
      fwall_type = FWALL_TYPE_HTTP;
  }
  conf_options[CONF_FORCEHTTP].thevariable=&use_http_regardless;
  conf_options[CONF_FORCEUUE].thevariable=&use_uue_regardless;
  conf_options[CONF_FWALLTYPE].thevariable=&fwall_type;
  conf_options[CONF_FWALLTYPE].choicelist=&fwall_types[0];
  conf_options[CONF_FWALLTYPE].choicemax=(s32)((sizeof(fwall_types)/sizeof(fwall_types[0]))-1);
  
  conf_options[CONF_FWALLHOSTNAME].thevariable=(char *)(&httpproxy[0]);
  conf_options[CONF_FWALLHOSTPORT].thevariable=&httpport;
  struct { char username[128], password[128]; } userpass;
  userpass.username[0] = userpass.password[0] = 0;
  
  if (httpid[0] == 0)
    ; //nothing
  else if (uuehttpmode==UUEHTTPMODE_UUEHTTP || uuehttpmode==UUEHTTPMODE_HTTP)
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
  else if (uuehttpmode == UUEHTTPMODE_SOCKS4) 
    strcpy(userpass.username,httpid);
  else if (uuehttpmode == UUEHTTPMODE_SOCKS5)
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

  conf_options[CONF_LURKMODE].thevariable=
  conf_options[CONF_CONNIFACEMASK].thevariable=
  conf_options[CONF_DIALWHENNEEDED].thevariable=
  conf_options[CONF_CONNPROFILE].thevariable=
  conf_options[CONF_CONNSTARTCMD].thevariable=
  conf_options[CONF_CONNSTOPCMD].thevariable=NULL;

  #if defined(LURK)
  int dupcap = dialup.GetCapabilityFlags();
  s32 lurkmode = dialup.lurkmode;
  s32 dialwhenneeded = dialup.dialwhenneeded;
  char connifacemask[sizeof(dialup.connifacemask)];
  char connstartcmd[sizeof(dialup.connstartcmd)];
  char connstopcmd[sizeof(dialup.connstopcmd)];
  char connprofile[sizeof(dialup.connprofile)];
  strcpy(connifacemask, dialup.connifacemask);
  strcpy(connstartcmd, dialup.connstartcmd);
  strcpy(connstopcmd, dialup.connstopcmd);
  strcpy(connprofile, dialup.connprofile);
  if ((dupcap & (CONNECT_LURK|CONNECT_LURKONLY))!=0)
  {
    conf_options[CONF_LURKMODE].thevariable=&lurkmode;
  }
  if ((dupcap & CONNECT_IFACEMASK)!=0)
  {
    conf_options[CONF_CONNIFACEMASK].thevariable=&connifacemask[0];
  }
  if ((dupcap & CONNECT_DOD)!=0)
  {
    conf_options[CONF_DIALWHENNEEDED].thevariable=&dialwhenneeded;
    if ((dupcap & CONNECT_DODBYSCRIPT)!=0)
    {
      conf_options[CONF_CONNSTARTCMD].thevariable=&connstartcmd[0];
      conf_options[CONF_CONNSTOPCMD].thevariable=&connstopcmd[0];
    }
    if ((dupcap & CONNECT_DODBYPROFILE)!=0)
    {
      const char **connectnames = dialup.GetConnectionProfileList();
      conf_options[CONF_CONNPROFILE].thevariable=&connprofile[0];
      conf_options[CONF_CONNPROFILE].choicemin = 
      conf_options[CONF_CONNPROFILE].choicemax = 0;
      if (connectnames) 
      {
        unsigned int maxconn = 0;
        while (connectnames[maxconn])
          maxconn++;
        if (maxconn > 1) /* the first option is "", ie default */
        {
          connectnames[0] = "<Use Control Panel Setting>";
          conf_options[CONF_CONNPROFILE].choicemax = (s32)(maxconn-1);
          conf_options[CONF_CONNPROFILE].choicelist = connectnames;
        }
      }
    }
  }
  #endif // if(LURK)

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

  /* --------------------------------------------------------- */

  int returnvalue = 0;
  int editthis = -1; /* in a menu */
  int whichmenu = CONF_MENU_MAIN; /* main menu */
  const char *menuname = "";
  while (returnvalue == 0)
  {
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
      const char *na = "n/a";
      int noremotedir = 0;

      conf_options[CONF_INBUFFERBASENAME].disabledtext=
      conf_options[CONF_OUTBUFFERBASENAME].disabledtext=
      conf_options[CONF_CHECKPOINT].disabledtext=
                  ((!nodiskbuffers)?(NULL):
                  ("n/a [disk buffers are disabled]"));

      noremotedir = (noupdatefromfile || remote_update_dir[0]=='\0');

      conf_options[CONF_MENU_NET_PLACEHOLDER].disabledtext = 
                  (offlinemode ? na : NULL );
      conf_options[CONF_REMOTEUPDATEDIR].disabledtext = 
                  (noupdatefromfile ? na : NULL );
      conf_options[CONF_THRESHOLDI].disabledtext= 
                  (offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_FREQUENT].disabledtext= 
                  (offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_PREFERREDBLOCKSIZE].disabledtext= 
                  (offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_THRESHOLDI].disabledtext= 
                  (offlinemode && noremotedir ? na : NULL );
 
    }
    else if (whichmenu == CONF_MENU_LOG)
    {
      conf_options[CONF_LOGLIMIT].thevariable=(char *)(&logkblimit[0]);
      if (logtype == LOGFILETYPE_ROTATE)
        conf_options[CONF_LOGLIMIT].thevariable=(char *)(&logrotlimit[0]);
      conf_options[CONF_LOGNAME].disabledtext=
                  ((logtype != LOGFILETYPE_NONE) ? (NULL) : 
                  ("n/a [file log disabled]"));
      conf_options[CONF_LOGLIMIT].disabledtext=
                  ((logtype != LOGFILETYPE_NONE && 
                    logtype != LOGFILETYPE_NOLIMIT) ? (NULL) : 
                  ("n/a [inappropriate for log type]"));
      conf_options[CONF_SMTPSRVR].disabledtext=
      conf_options[CONF_SMTPPORT].disabledtext=
      conf_options[CONF_SMTPDEST].disabledtext=
      conf_options[CONF_SMTPFROM].disabledtext=
                  ((messagelen > 0)?(NULL):
                  ("n/a [mail log disabled]"));
    }
    else if (whichmenu == CONF_MENU_NET)
    {
      unsigned int x;
      for (x=0; x < CONF_OPTION_COUNT; x++)
      {
        if (conf_options[x].optionscreen == CONF_MENU_NET)
          conf_options[x].disabledtext= NULL;
      }
      if (fwall_type == FWALL_TYPE_NONE)
      {
        conf_options[CONF_FWALLHOSTNAME].disabledtext=
        conf_options[CONF_FWALLHOSTPORT].disabledtext=
        conf_options[CONF_FWALLUSERNAME].disabledtext=
        conf_options[CONF_FWALLPASSWORD].disabledtext=
              "n/a [firewall support disabled]";
      }
      else
      {
        conf_options[CONF_FORCEHTTP].disabledtext= "n/a";
        if ( fwall_type != FWALL_TYPE_HTTP )
        {
          conf_options[CONF_FORCEUUE].disabledtext=
                "n/a [not available for this proxy method]";
        }
        if (httpproxy[0] == 0)
        {
          conf_options[CONF_FWALLHOSTPORT].disabledtext=
          conf_options[CONF_FWALLUSERNAME].disabledtext=
          conf_options[CONF_FWALLPASSWORD].disabledtext=
                "n/a [firewall hostname missing]";
        }
        else if (fwall_type!=FWALL_TYPE_HTTP && fwall_type!=FWALL_TYPE_SOCKS5)
        {
          conf_options[CONF_FWALLPASSWORD].disabledtext=
                "n/a [proxy method does not support passwords]";
        }
      }     
      
      if (autofindks)
      {
        conf_options[CONF_NOFALLBACK].disabledtext= //can't fallback to self
        conf_options[CONF_KEYSERVNAME].disabledtext = 
                "n/a [requires non-distributed.net host]";
      }
      #ifdef LURK
      if (lurkmode!=CONNECT_LURK && lurkmode!=CONNECT_LURKONLY)
      {
        conf_options[CONF_CONNIFACEMASK].disabledtext=
        conf_options[CONF_DIALWHENNEEDED].disabledtext=
        conf_options[CONF_CONNPROFILE].disabledtext=
        conf_options[CONF_CONNSTARTCMD].disabledtext=
        conf_options[CONF_CONNSTOPCMD].disabledtext=
        "n/a [requires dial-on-demand support]";
      }
      else if (!dialwhenneeded || conf_options[CONF_DIALWHENNEEDED].thevariable==NULL)
      {
        conf_options[CONF_CONNPROFILE].disabledtext=
        conf_options[CONF_CONNSTARTCMD].disabledtext=
        conf_options[CONF_CONNSTOPCMD].disabledtext=
        "n/a [requires dial-on-demand support]";
      }
      #endif
    }

    /* -------------------- display menu -------------------------- */

    if (editthis < 0) /* menu */
    {
      int optionlist[18]; /*18==maxperpage*/
      unsigned int optioncount = 0;
      int menuoption;

      if (whichmenu == CONF_MENU_MAIN) /* top level */
      {
        // we handle the main menu separately because of the ID prompt
        // and because we want to enforce a definitive selection.
        // Other than that, its a simplified version of a non-main menu.
        
        int id_menu = (int)conf_options[CONF_ID].optionscreen;
        const char *id_menuname = NULL;
        menuname = "mainmenu";
        optioncount = 0;
          
        for (menuoption=0; menuoption < CONF_OPTION_COUNT; menuoption++)
        {
          if (conf_options[menuoption].disabledtext != NULL)
          {
            /* ignore it */
          }
          else if (conf_options[menuoption].type==CONF_TYPE_MENU &&
              ((int)conf_options[menuoption].optionscreen) == whichmenu)
          {
            optionlist[optioncount++] = menuoption;
            if (id_menu == conf_options[menuoption].choicemin)
              id_menuname = conf_options[menuoption].description;
            if (optioncount == 8) /* max 8 submenus on main menu */
              break;
          }
        }
        
        if (optioncount == 0)
          returnvalue = -1;
        
        while (whichmenu == CONF_MENU_MAIN && returnvalue == 0)
        {
          char chbuf[6];
          ConClear();
          LogScreenRaw(CONFMENU_CAPTION, "");
         
          for (menuoption=0; menuoption < ((int)(optioncount)); menuoption++)
            LogScreenRaw(" %u) %s\n", menuoption+1,
                       conf_options[optionlist[menuoption]].description);
          LogScreenRaw("\n 9) Discard settings and exit"
                       "\n 0) Save settings and exit\n\n");
          if (id_menuname && id[0] == '\0')
            LogScreenRaw("Note: You have not yet provided a distributed.net ID.\n"
            "      Please go to the '%s' and set it.\n", id_menuname);
          LogScreenRaw("\nChoice --> " );
          ConInStr(chbuf, 2, 0);
          menuoption = ((strlen(chbuf)==1 && isdigit(chbuf[0]))?(atoi(chbuf)):(-1));
          if (CheckExitRequestTriggerNoIO() || menuoption==9)
          {
            whichmenu = -1;
            returnvalue = -1; //Breaks and tells it NOT to save
          }
          else if (menuoption == 0)
          {
            whichmenu = -1;
            returnvalue = 1; //Breaks and tells it to save
          }
          else if (menuoption>0 && menuoption<=((int)(optioncount)))
          {
            menuoption = optionlist[menuoption-1];
            whichmenu = conf_options[menuoption].choicemin;
            menuname = conf_options[menuoption].description;
          }
          ConClear();
        } 
      }
      else /* non-main menu */
      {
        editthis = -1;
        while (editthis == -1)
        {
          int parentmenu = CONF_MENU_MAIN;
          const char *parentmenuname = "main menu";
          optioncount = 0;
          
          for (menuoption=0; menuoption < CONF_OPTION_COUNT; menuoption++)
          {
            if (conf_options[menuoption].type==CONF_TYPE_MENU &&
                ((int)conf_options[menuoption].choicemin) == whichmenu)
            {                     /* we are in the sub-menu of this menu */
              unsigned parpar;
              parentmenu = conf_options[menuoption].optionscreen;
              parentmenuname = "main menu"; //conf_options[menuoption].description;
              for (parpar=0; parpar < CONF_OPTION_COUNT; parpar++)
              {
                if (conf_options[parpar].type==CONF_TYPE_MENU &&
                  ((int)conf_options[parpar].choicemin) == parentmenu)
                {
                  parentmenuname = conf_options[parpar].description;
                  break;
                }
              }
            }
            else if (conf_options[menuoption].optionscreen == whichmenu &&
                   optioncount<((sizeof(optionlist)/sizeof(optionlist[0])-1)))
            {
              char parm[128];
              const char *descr = NULL;
              
              if (conf_options[menuoption].disabledtext != NULL)
              {
                #ifdef GREGH /* this is only for greg! :) */
                descr = (char *)conf_options[menuoption].disabledtext;
                #endif /* othewise ignore it */
              }
              else if (conf_options[menuoption].type==CONF_TYPE_MENU)
              {
                descr = "";
              }
              else if (conf_options[menuoption].thevariable == NULL)
              {
                #ifdef GREGH /* this is only for greg! :) */
                descr = "n/a [not available on this platform]";
                #endif /* othewise ignore it */
              }
              else if (conf_options[menuoption].type==CONF_TYPE_ASCIIZ)
              {
                descr = (char *)conf_options[menuoption].thevariable;
                if (!*descr)
                  descr = (char *)conf_options[menuoption].defaultsetting;
              }
              else if (conf_options[menuoption].type==CONF_TYPE_PASSWORD)
              {
                int i = strlen((char *)conf_options[menuoption].thevariable);
                memset(parm, '*', i);
                parm[i] = 0;
                descr = parm;
              }
              else if (conf_options[menuoption].type==CONF_TYPE_TIMESTR)
              {
                long t = (long)*((s32 *)conf_options[menuoption].thevariable);
                sprintf(parm, "%ld:%02u", (t/60), 
                               (unsigned int)(((t<0)?(-t):(t))%60) );
                descr = parm;
              }
              else if (conf_options[menuoption].type==CONF_TYPE_INT)
              {
                long thevar = (long)*(s32 *)conf_options[menuoption].thevariable;
                if ((conf_options[menuoption].choicelist != NULL) &&
                     (thevar >= (long)(conf_options[menuoption].choicemin)) &&
                     (thevar <= (long)(conf_options[menuoption].choicemax)) )
                {
                  descr = (char *)conf_options[menuoption].choicelist[thevar];
                }
                else if (thevar == (long)(atoi(conf_options[menuoption].defaultsetting)))
                {
                  descr = (char *)conf_options[menuoption].defaultsetting;
                }
                else
                {
                  sprintf(parm, "%li", thevar );
                  descr = parm;
                }
              }
              else if (conf_options[menuoption].type == CONF_TYPE_BOOL)
              {
                descr = (char *)(((*(s32 *)conf_options[menuoption].thevariable)?("yes"):("no")));
              }

              if (descr)
              {
                char parm2[128];
                unsigned int optlen;
                optionlist[optioncount++] = menuoption;
                optlen = sprintf(parm2, "%2u) %s%s", optioncount, 
                     conf_options[menuoption].description,
                     (conf_options[menuoption].type == CONF_TYPE_MENU ? "" :
                                                       " ==> " ));
                if (descr)
                {
                  strncpy( &parm2[optlen], descr, (80-optlen) );
                  parm2[79]=0;
                }
                if (optioncount == 1)
                {
                  ConClear();
                  LogScreenRaw(CONFMENU_CAPTION, menuname);
                }
                LogScreenRaw( "%s\n", parm2 );
              }
            }
          }

          menuoption = 0;
          if (optioncount > 0)
          { 
            char chbuf[sizeof(long)*3];
            menuoption = 0; //-1;
            LogScreenRaw("\n 0) Return to %s\n\nChoice --> ",parentmenuname);
            if (ConInStr( chbuf, sprintf(chbuf,"%d",optioncount)+1, 0 )!=0)
            {
              if (!CheckExitRequestTriggerNoIO())
              {
                menuoption = atoi( chbuf );
                if (menuoption<0 || menuoption>((int)(optioncount)))
                  menuoption = -1;
                else if (menuoption != 0)
                {
                  menuoption = optionlist[menuoption-1];
                  if ((conf_options[menuoption].disabledtext != NULL) ||
                    ((conf_options[menuoption].type != CONF_TYPE_MENU) &&
                    conf_options[menuoption].thevariable == NULL))
                  {
                    menuoption = -1;
                  }
                }
              }
            } 
          }
          
          if (CheckExitRequestTriggerNoIO())
          {
            returnvalue = -1;
            editthis = -3;
            whichmenu = -2;
          }
          else if (menuoption == 0)
          {
            whichmenu = parentmenu;
            menuname = parentmenuname;
            editthis = ((parentmenu == CONF_MENU_MAIN)?(-2):(-1));
          }
          else if (menuoption > 0)
          {
            if (conf_options[menuoption].disabledtext != NULL)
            {
              editthis = -1;
            }
            else if (conf_options[menuoption].type == CONF_TYPE_MENU)
            {
              parentmenu = whichmenu;
              parentmenuname = menuname;
              whichmenu = (int)conf_options[menuoption].choicemin;
              menuname = conf_options[menuoption].description;
              editthis = -2; //-1;
            }
            else if (conf_options[menuoption].thevariable != NULL)
            {
              editthis = menuoption;
            }
          }
        } /* while (editthis == -1) */
      } /* non-main menu */
    }  /* editthis < 0 */
    else
    {
      int newval_isok = 0;
      long newval_d = 0;
      char parm[128];

      /* -- display user selection in detail and get new value --- */
      while ( editthis >= 0 && !newval_isok) 
      {
        char *p;

        ConClear(); 
        LogScreenRaw(CONFMENU_CAPTION, menuname);
        LogScreenRaw("\n%s:\n\n", conf_options[editthis].description );

        newval_isok = 1;
        if ((p = (char *)conf_options[editthis].comments) != NULL)
        {
          while (strlen(p) > (sizeof(parm)-1))
          {
            strncpy(parm, p, sizeof(parm));
            parm[(sizeof(parm)-1)] = 0;
            LogScreenRaw("%s", parm);
            p += (sizeof(parm)-1);
          }
          LogScreenRaw("%s\n",p);
        }
    
        if ( conf_options[editthis].type == CONF_TYPE_ASCIIZ || 
             conf_options[editthis].type == CONF_TYPE_INT ||
             conf_options[editthis].type == CONF_TYPE_PASSWORD )
        {
          p = "";
          char defaultbuff[30];
          int coninstrmode = CONINSTR_BYEXAMPLE;
          
          if (conf_options[editthis].choicelist !=NULL)
          {
            const char *ppp;
            long selmin = (long)(conf_options[editthis].choicemin);
            long selmax = (long)(conf_options[editthis].choicemax);
            sprintf(defaultbuff,"%ld) ", selmax );
            ppp = strstr( conf_options[editthis].comments, defaultbuff );
            if (ppp != NULL)
            {
              sprintf(defaultbuff,"%ld) ", selmin );
              ppp = strstr( conf_options[editthis].comments, defaultbuff );
            }
            if (ppp == NULL)
            {
              long listpos;
              for ( listpos = selmin; listpos <= selmax; listpos++)
                LogScreenRaw("  %2ld) %s\n", listpos, 
                      conf_options[editthis].choicelist[listpos]);
            }
          }
          if (conf_options[editthis].type==CONF_TYPE_PASSWORD)
          {
            int i = strlen((char *)conf_options[editthis].thevariable);
            memset(parm, '*', i);
            parm[i] = 0;
            coninstrmode = CONINSTR_ASPASSWORD;
          }
          else if (conf_options[editthis].type==CONF_TYPE_ASCIIZ)
          {
            strcpy(parm, (char *)conf_options[editthis].thevariable);
            p = (char *)(conf_options[editthis].defaultsetting);
          }
          else //if (conf_options[editthis].type==CONF_TYPE_INT)
          {
            sprintf(parm, "%li", (long)*(s32 *)conf_options[editthis].thevariable);
            sprintf(defaultbuff, "%li", atol(conf_options[editthis].defaultsetting));
            p = defaultbuff;
          }
          LogScreenRaw("Default Setting: %s\n"
                       "Current Setting: %s\n"
                       "New Setting --> ", p, parm );

          ConInStr( parm, 64 /*sizeof(parm)*/, coninstrmode );

          if (CheckExitRequestTriggerNoIO())
          {
            editthis = -2;
            returnvalue = -1;
          }
          else if (conf_options[editthis].type==CONF_TYPE_INT)
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
              long selmin = conf_options[editthis].choicemin;
              long selmax = conf_options[editthis].choicemax;
              if ((selmin != 0 || selmax != 0) && 
                (newval_d < selmin || newval_d > selmax))
              newval_isok = 0;
            }
          }
          else //if (conf_options[editthis].type==CONF_TYPE_ASCIIZ)
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
            newval_isok = 1;

            if (parm[0] != 0 && conf_options[editthis].choicemax != 0 && 
                conf_options[editthis].choicelist) /* int *and* asciiz */
            {
              newval_d=atoi(parm);
              if ( ((newval_d > 0) || (parm[0] == '0')) &&
                 (newval_d <= conf_options[editthis].choicemax) )
              {
                strncpy(parm, conf_options[editthis].choicelist[newval_d], sizeof(parm));
                parm[sizeof(parm)-1]=0; 
                if (newval_d == 0 && editthis == CONF_CONNPROFILE)
                  parm[0]=0;
              }
            }
          }
        }
        else if (conf_options[editthis].type == CONF_TYPE_TIMESTR)
        {
          long t = (long)*((s32 *)conf_options[editthis].thevariable);
          sprintf(parm,"%ld:%02u", (t/60), 
                           (unsigned int)(((t<0)?(-t):(t))%60) );
          LogScreenRaw("Default Setting: %s\n"
                       "Current Setting: %s\n"
                       "New Setting --> ",
                       conf_options[editthis].defaultsetting, parm );
          
          ConInStr( parm, 10, CONINSTR_BYEXAMPLE );
          
          if (CheckExitRequestTriggerNoIO())
          {
            editthis = -2;
            returnvalue = -1;
          }
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
        else if (conf_options[editthis].type==CONF_TYPE_BOOL)
        {
          sprintf(parm, "%s", *(s32 *)conf_options[editthis].thevariable?"yes":"no");
          LogScreenRaw("Default Setting: %s\n"
                       "Current Setting: %s\n"
                       "New Setting --> ",
                       *(conf_options[editthis].defaultsetting)=='0'?"no":"yes", 
                       parm );
          parm[1] = 0;
          ConInStr( parm, 2, CONINSTR_BYEXAMPLE|CONINSTR_ASBOOLEAN );
          if (CheckExitRequestTriggerNoIO())
          {
            editthis = -2;
            returnvalue = -1;
          }
          else if (parm[0]=='y' || parm[0]=='Y')
            newval_d = 1;
          else if (parm[0]=='n' || parm[0]=='N') 
            newval_d = 0;
          else
            newval_isok = 0;
        }
        else
        {
          editthis = -1;
        }
        if (editthis >= 0 && !newval_isok)
        {
          ConBeep();
        }
      }
        
      /* --------------- have modified value, so assign -------------- */

      if (editthis >= 0 && newval_isok)
      {
        // DO NOT TOUCH ANY VARIABLE EXCEPT THE SELECTED ONE
        // (unless those variables are not menu options)
        // DO IT AFTER ALL MENU DRIVEN CONFIG IS FINISHED (see end)
        
        if (editthis == CONF_ID || editthis == CONF_KEYSERVNAME ||
          editthis == CONF_SMTPFROM || editthis == CONF_SMTPSRVR ||
          editthis == CONF_FWALLHOSTNAME)
        {
          char *opos, *ipos;
          ipos = opos = &parm[0];
          while ( *ipos )
          {
            if ( !isspace( *ipos ) )
              *opos++ = *ipos;
            ipos++;
          }
          *opos = '\0';
          if ( strcmp( parm, "none" ) == 0 )
            parm[0]='\0';
        }
        if (conf_options[editthis].type==CONF_TYPE_ASCIIZ ||
            conf_options[editthis].type==CONF_TYPE_PASSWORD)
        {
          strncpy( (char *)conf_options[editthis].thevariable, parm, 
                   64 - 1 );
          ((char *)conf_options[editthis].thevariable)[64-1]=0;
        }
        else //bool or int types
        {
          *(s32 *)conf_options[editthis].thevariable = (s32)newval_d;
          if ( editthis == CONF_COUNT && newval_d < 0)
            blockcount = -1;
          else if (editthis == CONF_THRESHOLDI)
            inthreshold[0]=outthreshold[0]=inthreshold[1]=outthreshold[1]=newval_d;
          else if (editthis == CONF_NETTIMEOUT)
            nettimeout = ((newval_d<0)?(-1):((newval_d<5)?(5):(newval_d)));
        }
      } /* if (editthis >= 0 && newval_isok) */
      editthis = -1; /* no longer an editable option */
    } /* not a menu */
  } /* while (returnvalue == 0) */

  if (CheckExitRequestTriggerNoIO())
    returnvalue = -1;

  /* -- massage mapped options and dependancies back into place -- */

  if (returnvalue != -1)
  {
    if (id[0] == 0)
      strcpy(id, "rc5@distributed.net");

    if (logtype >=0 && logtype < (s32)(sizeof(logtypes)/sizeof(logtypes[0])))
    {
      if (logtype == LOGFILETYPE_ROTATE)
        strcpy( logfilelimit, logrotlimit );
      else 
      {
        if (logname[0] == '\0')
          logtype = LOGFILETYPE_NONE;
        strcpy( logfilelimit, logkblimit );
      }
      strcpy( logfiletype, logtypes[logtype] );
    }

    autofindkeyserver = (autofindks!=0);

    if (nettimeout < 0)
      nettimeout = -1;
    else if (nettimeout < 5)
      nettimeout = 5;

    #ifdef LURK
    dialup.lurkmode = lurkmode;
    dialup.dialwhenneeded = dialwhenneeded;
    strcpy(dialup.connifacemask, connifacemask);
    strcpy(dialup.connstartcmd, connstartcmd);
    strcpy(dialup.connstopcmd, connstopcmd);
    strcpy(dialup.connprofile, connprofile);
    #endif

    projectmap_build(loadorder_map, loadorder );

    uuehttpmode = 0;
    if (fwall_type == FWALL_TYPE_SOCKS4)
      uuehttpmode = UUEHTTPMODE_SOCKS4;
    else if (fwall_type == FWALL_TYPE_SOCKS5)
      uuehttpmode = UUEHTTPMODE_SOCKS5;
    else if (fwall_type == FWALL_TYPE_HTTP || use_http_regardless)
      uuehttpmode = (use_uue_regardless?UUEHTTPMODE_UUEHTTP:UUEHTTPMODE_HTTP);
    else if (use_uue_regardless)
      uuehttpmode = UUEHTTPMODE_UUE;
    
    if (strlen(userpass.username) == 0 && strlen(userpass.password) == 0)
      httpid[0] = 0;
    else if (uuehttpmode==UUEHTTPMODE_UUEHTTP || uuehttpmode==UUEHTTPMODE_HTTP)
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
    else if (uuehttpmode == UUEHTTPMODE_SOCKS4)
    {
      strcpy( httpid, userpass.username );
    }
    else if (uuehttpmode == UUEHTTPMODE_SOCKS5)
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

