/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------------
 * Real programmers don't bring brown-bag lunches.  If the vending machine
 * doesn't sell it, they don't eat it.  Vending machines don't sell quiche.
 * ---------------------------------------------------------------------
*/
const char *confmenu_cpp(void) {
return "@(#)$Id: confmenu.cpp,v 1.41.2.12 1999/11/23 19:15:15 cyp Exp $"; }

/* ----------------------------------------------------------------------- */

//#define TRACE
//#define PLAINTEXT_PW

//#include "cputypes.h" // CLIENT_OS
#include "console.h"  // ConOutErr()
#include "client.h"   // client->members, MINCLIENTOPTSTRLEN
#include "baseincs.h" // strlen() etc
#include "cmpidefs.h" // strcmpi()
#include "logstuff.h" // LogScreenRaw()
#include "selcore.h"  // GetCoreNameFromCoreType()
#include "clicdata.h" // GetContestNameFromID()
#include "util.h"     // projectmap_*()
#include "lurk.h"     // lurk stuff
#include "triggers.h" // CheckExitRequestTriggerNoIO()
#include "confopt.h"  // the option table
#include "confmenu.h" // ourselves

//#define REVEAL_DISABLED /* this is for gregh :) */

/* ----------------------------------------------------------------------- */
static const char *CONFMENU_CAPTION="distributed.net client configuration: %s\n"
"--------------------------------------------------------------------------\n";

static int __enumcorenames(const char **corenames, int index, void * /*unused*/)
{
  unsigned int cont_i;
  char linebuff[CONTEST_COUNT][32];
  if (index == 0)
  {
    const char *uline = "------------------------";
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      if (cont_i != OGR)
      {
        const char *contname = CliGetContestNameFromID(cont_i);
        linebuff[cont_i][0] = '\0';
        if (contname)
        {
          strncpy(&(linebuff[cont_i][0]),contname,sizeof(linebuff[cont_i]));
          linebuff[cont_i][sizeof(linebuff[cont_i])-1] = '\0';
        }
      }
    }
    LogScreenRaw(" %-25.25s %-25.25s %-25.25s\n", &(linebuff[RC5][0]), 
                  &(linebuff[DES][0]),   &(linebuff[CSC][0]) );
    LogScreenRaw(" %-25.25s %-25.25s %-25.25s\n",uline,uline,uline);
    uline = "-1) Auto-select";
    LogScreenRaw(" %-25.25s %-25.25s %-25.25s\n",uline,uline,uline);
    
  }
  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    linebuff[cont_i][0] = '\0';
    if (cont_i != OGR && corenames[cont_i])
    {
      sprintf(&(linebuff[cont_i][0]),"%2d) ", index );
      strncpy(&(linebuff[cont_i][4]),corenames[cont_i],sizeof(linebuff[cont_i])-4);
      linebuff[cont_i][sizeof(linebuff[cont_i])-1] = '\0';
    }  
  } 
  LogScreenRaw(" %-25.25s %-25.25s %-25.25s\n", &(linebuff[RC5][0]), 
                &(linebuff[DES][0]),   &(linebuff[CSC][0]) );
  return +1; /* keep going */
}      

int Configure( Client *client ) /* returns >0==success, <0==cancelled */
{
  struct __userpass { 
    char username[MINCLIENTOPTSTRLEN*2]; 
    char password[MINCLIENTOPTSTRLEN*2]; 
  } userpass;
  unsigned int cont_i;
  char loadorder[MINCLIENTOPTSTRLEN];

  // ---- Set all stuff that doesn't change during config ----   
  // note that some options rely on others, so watch the init order

  /* ------------------- CONF_MENU_MISC ------------------ */  

  if (strcmpi(client->id,"rc5@distributed.net") == 0)
    client->id[0] = 0; /*is later converted back to 'rc5@distributed.net' */
  conf_options[CONF_ID].thevariable=&(client->id[0]);
  conf_options[CONF_COUNT].thevariable=&(client->blockcount);
  conf_options[CONF_HOURS].thevariable=&(client->minutes);
  conf_options[CONF_QUIETMODE].thevariable=&(client->quietmode);
  conf_options[CONF_NOEXITFILECHECK].thevariable=&(client->noexitfilecheck);
  conf_options[CONF_PERCENTOFF].thevariable=&(client->percentprintingoff);
  conf_options[CONF_PAUSEFILE].thevariable=&(client->pausefile[0]);
  conf_options[CONF_CONTESTPRIORITY].thevariable = 
       strcpy(loadorder, projectmap_expand( client->loadorder_map ) );

  /* ------------------- CONF_MENU_BUFF ------------------ */  

  conf_options[CONF_NODISK].thevariable=&(client->nodiskbuffers);
  conf_options[CONF_INBUFFERBASENAME].thevariable=&(client->in_buffer_basename[0]);
  conf_options[CONF_OUTBUFFERBASENAME].thevariable=&(client->out_buffer_basename[0]);
  conf_options[CONF_CHECKPOINT].thevariable=&(client->checkpoint_file[0]);
  conf_options[CONF_OFFLINEMODE].thevariable=&(client->offlinemode);
  conf_options[CONF_REMOTEUPDATEDISABLED].thevariable=&(client->noupdatefromfile);
  conf_options[CONF_REMOTEUPDATEDIR].thevariable=&(client->remote_update_dir[0]);
  conf_options[CONF_FREQUENT].thevariable=&(client->connectoften);
  conf_options[CONF_PREFERREDBLOCKSIZE].thevariable=&(client->preferred_blocksize[0]);
  conf_options[CONF_THRESHOLDI].thevariable=&(client->inthreshold[0]);

  /* ------------------- CONF_MENU_LOG  ------------------ */  

  static const char *logtypes[] = {"none","no limit","restart","fifo","rotate"};
  char logkblimit[sizeof(client->logfilelimit)], logrotlimit[sizeof(client->logfilelimit)];
  int logtype = LOGFILETYPE_NOLIMIT;
  logkblimit[0] = logrotlimit[0] = '\0';
  
  if ( strcmp( client->logfiletype, "rotate" ) == 0)
  {
    logtype = LOGFILETYPE_ROTATE;
    strcpy( logrotlimit, (client->logfilelimit) );
  }
  else 
  {
    strcpy( logkblimit, client->logfilelimit );
    if ( client->logname[0] == '\0' || strcmp( client->logfiletype, "none" ) == 0 )
      logtype = LOGFILETYPE_NONE;
    else if (strcmp( client->logfiletype, "restart" ) == 0)
      logtype = LOGFILETYPE_RESTART;
    else if (strcmp( client->logfiletype, "fifo" ) == 0)
      logtype = LOGFILETYPE_FIFO;
  }
  
  conf_options[CONF_LOGTYPE].thevariable=&logtype;
  conf_options[CONF_LOGTYPE].choicelist=&logtypes[0];
  conf_options[CONF_LOGTYPE].choicemax=(int)((sizeof(logtypes)/sizeof(logtypes[0]))-1);
  conf_options[CONF_LOGNAME].thevariable=&(client->logname[0]);
  conf_options[CONF_LOGLIMIT].thevariable=&logkblimit[0];
  conf_options[CONF_MESSAGELEN].thevariable=&(client->messagelen);
  conf_options[CONF_SMTPSRVR].thevariable=&(client->smtpsrvr[0]);
  conf_options[CONF_SMTPPORT].thevariable=&(client->smtpport);
  conf_options[CONF_SMTPFROM].thevariable=&(client->smtpfrom[0]);
  conf_options[CONF_SMTPDEST].thevariable=&(client->smtpdest[0]);
  conf_options[CONF_SMTPFROM].defaultsetting=(char *)conf_options[CONF_ID].thevariable;
  conf_options[CONF_SMTPDEST].defaultsetting=(char *)conf_options[CONF_ID].thevariable;

  /* ------------------- CONF_MENU_NET  ------------------ */  

  conf_options[CONF_NETTIMEOUT].thevariable=&(client->nettimeout);
  conf_options[CONF_AUTOFINDKS].thevariable=&(client->autofindkeyserver);
  conf_options[CONF_KEYSERVNAME].thevariable=&(client->keyproxy[0]);
  conf_options[CONF_KEYSERVPORT].thevariable=&(client->keyport);
  conf_options[CONF_NOFALLBACK].thevariable=&(client->nofallback);

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
  int fwall_type = FWALL_TYPE_NONE;
  int use_http_regardless = (( client->uuehttpmode == UUEHTTPMODE_HTTP
                            || client->uuehttpmode == UUEHTTPMODE_UUEHTTP)
                            && client->httpproxy[0] == '\0');
  int use_uue_regardless =  (  client->uuehttpmode == UUEHTTPMODE_UUE 
                            || client->uuehttpmode == UUEHTTPMODE_UUEHTTP);
  if (client->httpproxy[0])
  {                           
    if (client->uuehttpmode == UUEHTTPMODE_SOCKS4) 
      fwall_type = FWALL_TYPE_SOCKS4;
    else if (client->uuehttpmode == UUEHTTPMODE_SOCKS5) 
      fwall_type = FWALL_TYPE_SOCKS5;
    else if (client->uuehttpmode==UUEHTTPMODE_HTTP || 
             client->uuehttpmode==UUEHTTPMODE_UUEHTTP)
      fwall_type = FWALL_TYPE_HTTP;
  }
  conf_options[CONF_FORCEHTTP].thevariable=&use_http_regardless;
  conf_options[CONF_FORCEUUE].thevariable=&use_uue_regardless;
  conf_options[CONF_FWALLTYPE].thevariable=&fwall_type;
  conf_options[CONF_FWALLTYPE].choicelist=&fwall_types[0];
  conf_options[CONF_FWALLTYPE].choicemax=(int)((sizeof(fwall_types)/sizeof(fwall_types[0]))-1);
  
  conf_options[CONF_FWALLHOSTNAME].thevariable=&(client->httpproxy[0]);
  conf_options[CONF_FWALLHOSTPORT].thevariable=&(client->httpport);
  userpass.username[0] = userpass.password[0] = 0;
  
  if (client->httpid[0])
  {
    strcpy( userpass.username, client->httpid );
    char *p = strchr( userpass.username,':');
    if (p) 
    {
      *p++ = 0;
      strcpy( userpass.password, p );
    }
  }
  conf_options[CONF_FWALLUSERNAME].thevariable = (&userpass.username[0]);
  conf_options[CONF_FWALLPASSWORD].thevariable = (&userpass.password[0]);

  conf_options[CONF_LURKMODE].thevariable=
  conf_options[CONF_CONNIFACEMASK].thevariable=
  conf_options[CONF_DIALWHENNEEDED].thevariable=
  conf_options[CONF_CONNPROFILE].thevariable=
  conf_options[CONF_CONNSTARTCMD].thevariable=
  conf_options[CONF_CONNSTOPCMD].thevariable=NULL;

  #if defined(LURK)
  int dupcap = dialup.GetCapabilityFlags();
  int lurkmode = dialup.lurkmode;
  int dialwhenneeded = dialup.dialwhenneeded;
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
          conf_options[CONF_CONNPROFILE].choicemax = (int)(maxconn-1);
          conf_options[CONF_CONNPROFILE].choicelist = connectnames;
        }
      }
    }
  }
  #endif // if(LURK)

  /* ------------------- CONF_MENU_PERF ------------------ */  

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
     client->coretypes[cont_i] = selcoreValidateCoreIndex(cont_i,client->coretypes[cont_i]);
  conf_options[CONF_CPUTYPE].thevariable = &(client->coretypes[0]);
  conf_options[CONF_NICENESS].thevariable = &(client->priority);
  conf_options[CONF_NUMCPU].thevariable = &(client->numcpu);

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
      const char *na = "n/a [no net & no remote dir]";
      int noremotedir = 0;

      conf_options[CONF_INBUFFERBASENAME].disabledtext=
      conf_options[CONF_OUTBUFFERBASENAME].disabledtext=
      conf_options[CONF_CHECKPOINT].disabledtext=
                  ((!client->nodiskbuffers)?(NULL):
                  ("n/a [disk buffers are disabled]"));

      noremotedir = (client->noupdatefromfile || 
                     client->remote_update_dir[0]=='\0');

      conf_options[CONF_MENU_NET_PLACEHOLDER].disabledtext = 
                  (client->offlinemode ? " ==> n/a [no net & no remote dir]" : NULL );
      conf_options[CONF_REMOTEUPDATEDIR].disabledtext = 
                  (client->noupdatefromfile ? na : NULL );
      conf_options[CONF_THRESHOLDI].disabledtext= 
                  (client->offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_FREQUENT].disabledtext= 
                  (client->offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_PREFERREDBLOCKSIZE].disabledtext= 
                  (client->offlinemode && noremotedir ? na : NULL );
      conf_options[CONF_THRESHOLDI].disabledtext= 
                  (client->offlinemode && noremotedir ? na : NULL );
 
    }
    else if (whichmenu == CONF_MENU_LOG)
    {
      conf_options[CONF_LOGLIMIT].thevariable=(&logkblimit[0]);
      if (logtype == LOGFILETYPE_ROTATE)
        conf_options[CONF_LOGLIMIT].thevariable=(&logrotlimit[0]);
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
                  ((client->messagelen > 0)?(NULL):
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
        if (client->httpproxy[0] == 0)
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
      
      if (client->autofindkeyserver)
      {
        conf_options[CONF_NOFALLBACK].disabledtext= "n/a"; //can't fallback to self
        conf_options[CONF_KEYSERVNAME].disabledtext = "n/a [autoselected]";
      }
      #ifdef LURK
      if (lurkmode!=CONNECT_LURK && lurkmode!=CONNECT_LURKONLY)
      {
        conf_options[CONF_CONNIFACEMASK].disabledtext=
        conf_options[CONF_DIALWHENNEEDED].disabledtext=
        conf_options[CONF_CONNPROFILE].disabledtext=
        conf_options[CONF_CONNSTARTCMD].disabledtext=
        conf_options[CONF_CONNSTOPCMD].disabledtext=
        "n/a [Dialup detection is off]";
      }
      else if (!dialwhenneeded || conf_options[CONF_DIALWHENNEEDED].thevariable==NULL)
      {
        conf_options[CONF_CONNPROFILE].disabledtext=
        conf_options[CONF_CONNSTARTCMD].disabledtext=
        conf_options[CONF_CONNSTOPCMD].disabledtext=
        "n/a [Demand-dial is off/not supported]";
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
          if (id_menuname && client->id[0] == '\0')
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
                #ifdef REVEAL_DISABLED /* this is only for greg! :) */
                descr = (char *)conf_options[menuoption].disabledtext;
                #endif /* othewise ignore it */
              }
              else if (conf_options[menuoption].type==CONF_TYPE_MENU)
              {
                descr = "";
              }
              else if (conf_options[menuoption].thevariable == NULL)
              {
                #ifdef REVEAL_DISABLED /* this is only for greg! :) */
                descr = "n/a [not available on this platform]";
                #endif /* othewise ignore it */
              }
              else if (conf_options[menuoption].type==CONF_TYPE_IARRAY)
              {
                int *vectb = NULL;
                if ( menuoption == CONF_THRESHOLDI )  // don't have a
                  vectb = &(client->outthreshold[0]); // THRESHOLDO any more
                utilGatherOptionArraysToList( parm, sizeof(parm),
                    (int *)conf_options[menuoption].thevariable, vectb ); 
                descr = parm;
              }
              else if (conf_options[menuoption].type==CONF_TYPE_ASCIIZ
#ifdef PLAINTEXT_PW
                || conf_options[menuoption].type==CONF_TYPE_PASSWORD
#endif                
              )
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
                int t = *((int *)conf_options[menuoption].thevariable);
                sprintf(parm, "%d:%02u", (t/60), 
                               (unsigned int)(((t<0)?(-t):(t))%60) );
                descr = parm;
              }
              else if (conf_options[menuoption].type==CONF_TYPE_INT)
              {
                int thevar = *((int *)conf_options[menuoption].thevariable);
                if ((conf_options[menuoption].choicelist != NULL) &&
                     (thevar >= conf_options[menuoption].choicemin) &&
                     (thevar <= conf_options[menuoption].choicemax) )
                {
                  descr = (char *)conf_options[menuoption].choicelist[thevar];
                }
                else if (thevar == atoi(conf_options[menuoption].defaultsetting))
                {
                  descr = (char *)conf_options[menuoption].defaultsetting;
                }
                else
                {
                  sprintf(parm, "%d", thevar );
                  descr = parm;
                }
              }
              else if (conf_options[menuoption].type == CONF_TYPE_BOOL)
              {
                descr = "no";
                if (*((int *)conf_options[menuoption].thevariable))
                  descr = "yes";
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
              whichmenu = conf_options[menuoption].choicemin;
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
             conf_options[editthis].type == CONF_TYPE_PASSWORD ||
             conf_options[editthis].type==CONF_TYPE_IARRAY )
        {
          p = "";
          char defaultbuff[30];
          int coninstrmode = CONINSTR_BYEXAMPLE;
          
          if (editthis == CONF_CPUTYPE) /* ugh! */
          {
            selcoreEnumerateWide( __enumcorenames, NULL ); 
            LogScreenRaw("\n");
          }
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
          
          if (conf_options[editthis].type==CONF_TYPE_ASCIIZ
#ifdef PLAINTEXT_PW
           || conf_options[editthis].type==CONF_TYPE_PASSWORD
#endif           
          )
          {
            strcpy(parm, (char *)conf_options[editthis].thevariable);
            p = (char *)(conf_options[editthis].defaultsetting);
          }
          else if (conf_options[editthis].type==CONF_TYPE_PASSWORD)
          {
            int i = strlen((char *)conf_options[editthis].thevariable);
            memset(parm, '*', i);
            parm[i] = 0;
            coninstrmode = CONINSTR_ASPASSWORD;
          }
          else if (conf_options[editthis].type==CONF_TYPE_IARRAY)
          {
            int *vectb = NULL;
            if ( editthis == CONF_THRESHOLDI )  // don't have a
              vectb = &(client->outthreshold[0]); // THRESHOLDO any more
            utilGatherOptionArraysToList( parm, sizeof(parm),
                    (int *)conf_options[editthis].thevariable, vectb ); 
            p = (char *)(conf_options[editthis].defaultsetting);
          }
          else //if (conf_options[editthis].type==CONF_TYPE_INT)
          {
            sprintf(parm, "%d", *((int *)conf_options[editthis].thevariable) );
            sprintf(defaultbuff, "%d", atoi(conf_options[editthis].defaultsetting));
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
              long selmin = (long)conf_options[editthis].choicemin;
              long selmax = (long)conf_options[editthis].choicemax;
              if ((selmin != 0 || selmax != 0) && 
                (newval_d < selmin || newval_d > selmax))
              newval_isok = 0;
            }
          }
          else if (conf_options[editthis].type == CONF_TYPE_IARRAY)
          {
            newval_isok = 1; /* never rejected */
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
              newval_d = atol(parm);
              if ( ((newval_d > 0) || (parm[0] == '0')) &&
                 (newval_d <= ((long)conf_options[editthis].choicemax)) )
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
          int t = *((int *)conf_options[editthis].thevariable);
          sprintf(parm,"%d:%02u", (t/60), 
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
          strcpy(parm, (*((int *)conf_options[editthis].thevariable))?"yes":"no");
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
                   MINCLIENTOPTSTRLEN );
          ((char *)conf_options[editthis].thevariable)[MINCLIENTOPTSTRLEN-1]=0;
        }
        else if ( conf_options[editthis].type == CONF_TYPE_IARRAY)
        {
          int *vecta = (int *)conf_options[editthis].thevariable;
          int *vectb = NULL;
          unsigned int cont_i;
          if ( editthis == CONF_THRESHOLDI )  // don't have a
            vectb = &(client->outthreshold[0]); // THRESHOLDO any more
          utilScatterOptionListToArraysEx(parm,vecta, vectb,NULL, NULL );
          if (editthis == CONF_CPUTYPE) 
          {
            for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
              vecta[cont_i] = selcoreValidateCoreIndex(cont_i,vecta[cont_i]);
          }
          else 
          {
            int mmin = conf_options[editthis].choicemin;
            int mmax = conf_options[editthis].choicemax;
            if (mmin && mmax)
            {
              for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
              {
                if (vecta[cont_i] < mmin)
                  vecta[cont_i] = mmin;
                else if (vecta[cont_i] > mmax)
                  vecta[cont_i] = mmax;
                if (vectb)
                {
                  if (vectb[cont_i] < mmin)
                    vectb[cont_i] = mmin;
                  else if (vectb[cont_i] > mmax)
                    vectb[cont_i] = mmax;
                }
              }
            }
          } 
        }
        else //bool or int types
        {
          *((int *)conf_options[editthis].thevariable) = (int)newval_d;
          if ( editthis == CONF_COUNT && newval_d < 0)
            *((int *)conf_options[editthis].thevariable) = -1;
          else if (editthis == CONF_NETTIMEOUT)
            *((int *)conf_options[editthis].thevariable) = ((newval_d<0)?(-1):((newval_d<5)?(5):(newval_d)));
        }
      } /* if (editthis >= 0 && newval_isok) */
      editthis = -1; /* no longer an editable option */
    } /* not a menu */
  } /* while (returnvalue == 0) */

  if (CheckExitRequestTriggerNoIO())
  {
    LogScreenRaw("\n"); 
    returnvalue = -1;
  }

  /* -- massage mapped options and dependancies back into place -- */

  if (returnvalue != -1)
  {
    //if (client->id[0] == 0)
    //  strcpy(client->id, "rc5@distributed.net");

    if (logtype >=0 && logtype < (int)(sizeof(logtypes)/sizeof(logtypes[0])))
    {
      if (logtype == LOGFILETYPE_ROTATE)
        strcpy( client->logfilelimit, logrotlimit );
      else 
      {
        if (client->logname[0] == '\0')
          logtype = LOGFILETYPE_NONE;
        strcpy( client->logfilelimit, logkblimit );
      }
      strcpy( client->logfiletype, logtypes[logtype] );
    }

    if (client->nettimeout < 0)
      client->nettimeout = -1;
    else if (client->nettimeout < 5)
      client->nettimeout = 5;

    #ifdef LURK
    dialup.lurkmode = lurkmode;
    dialup.dialwhenneeded = dialwhenneeded;
    strcpy(dialup.connifacemask, connifacemask);
    strcpy(dialup.connstartcmd, connstartcmd);
    strcpy(dialup.connstopcmd, connstopcmd);
    strcpy(dialup.connprofile, connprofile);
    #endif

    projectmap_build(client->loadorder_map, loadorder );

    client->uuehttpmode = 0;
    if (fwall_type == FWALL_TYPE_SOCKS4)
      client->uuehttpmode = UUEHTTPMODE_SOCKS4;
    else if (fwall_type == FWALL_TYPE_SOCKS5)
      client->uuehttpmode = UUEHTTPMODE_SOCKS5;
    else if (fwall_type == FWALL_TYPE_HTTP || use_http_regardless)
      client->uuehttpmode = (use_uue_regardless?UUEHTTPMODE_UUEHTTP:UUEHTTPMODE_HTTP);
    else if (use_uue_regardless)
      client->uuehttpmode = UUEHTTPMODE_UUE;
    
    TRACE_OUT((0,"precomp: u:p=\"%s:%s\"\n",userpass.username,userpass.password));
    client->httpid[0] = 0;
    if (strlen(userpass.username) || strlen(userpass.password))
    {
      if (strlen(userpass.password))
        strcat(strcat(userpass.username, ":"), userpass.password);
      strncpy( client->httpid, userpass.username, sizeof( client->httpid ));
      client->httpid[sizeof( client->httpid )-1] = 0;
    }
    TRACE_OUT((0,"postcomp: u:p=\"%s\"\n",client->httpid));
  }

  //fini
    
  return returnvalue;
}
