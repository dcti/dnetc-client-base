// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confmenu.cpp,v $
// Revision 1.3  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.2  1998/11/26 22:30:36  cyp
// Fixed an implied signed/unsigned comparison in ::Configure.
//
// Revision 1.1  1998/11/22 15:16:17  cyp
// Split from cliconfig.cpp; Changed runoffline/runbuffers/blockcount handling
// (runbuffers is now synonymous with blockcount=-1; offlinemode is always
// 0/1); changed 'frequent' context to describe what it does better: check
// buffers frequently and not connect frequently. Removed outthreshold[0] as
// well as both DES thresholds from the menu. Removed 'processdes' from the menu.
// Fixed various bugs. Range validation is always based on the min/max values in
// the option table.
//
//


#if (!defined(lint) && defined(__showids__))
const char *confmenu_cpp(void) {
return "@(#)$Id: confmenu.cpp,v 1.3 1998/12/20 23:00:35 silby Exp $"; }
#endif

#include "cputypes.h" // CLIENT_OS, s32
#include "console.h"  // ConOutErr()
#include "client.h"   // Client class
#include "baseincs.h" // strlen() etc
#include "cmpidefs.h" // strcmpi()
#include "logstuff.h" // LogScreenRaw()
#include "selcore.h"  // GetCoreNameFromCoreType()
#include "lurk.h"     // lurk stuff
#include "triggers.h" // CheckExitRequestTriggerNoIO()
#include "network.h"  // base64_encode()
#include "confopt.h"  // the option table

// --------------------------------------------------------------------------

#define MAXMENUENTRIES 18 /* max menu entries per screen */

static const char *CONFMENU_CAPTION="RC5DES Client Configuration: %s\n"
"-------------------------------------------------------------------\n";
       
static const char *menutable[]=
  {
  "Block and Buffer Options",
  "Logging Options",
  "Network and Communication Options",
  "Performance and Processor Options",
  "Miscellaneous Options"
  };

// --------------------------------------------------------------------------

static s32 findmenuoption( s32 menu, s32 option)
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
  int tpos;

  for (tpos=0; tpos < OPTION_COUNT; tpos++)
    {
    if ((conf_options[tpos].optionscreen==menu) &&
        (conf_options[tpos].menuposition==option))
      return (s32)tpos;
    }
  return -1;
}

// --------------------------------------------------------------------------

// checks for user to type yes or no. Returns 1=yes, 0=no, -1=unknown
static int yesno(char *str)
{
  if (strcmpi(str, "yes")==0 || strcmpi(str, "y")==0 || 
      strcmpi(str, "true")==0 || strcmpi(str, "t")==0 || 
      strcmpi(str, "1")==0) 
    return 1;
  if (strcmpi(str, "no")==0 || strcmpi(str, "n")==0 || 
      strcmpi(str, "false")==0 || strcmpi(str, "f")==0 || 
      strcmpi(str, "0")==0) 
    return 0;
  return -1;
}

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
s32 Client::ConfigureGeneral( s32 currentmenu )
{
  char parm[128],parm2[128];
  s32 choice=1;
  s32 temp;
  s32 temp2;
  char *p;

  do                    //note: don't return or break from inside 
    {                   //the loop. Let it fall through instead. - cyp

    /* ------ Sets all the pointers/etc for optionstruct options ------- */

    if (strcmpi(id,"rc5@distributed.net") == 0)
      id[0]=0; /*is later converted back to 'rc5@distributed.net' */
    
    conf_options[CONF_ID].thevariable=(char *)(&id[0]);
    conf_options[CONF_THRESHOLDI].thevariable=&inthreshold[0];

    conf_options[CONF_THRESHOLDI].choicemax=MAXBLOCKSPERBUFFER; /* client.h */

    #if 0 /* obsolete */
    conf_options[CONF_THRESHOLDO].thevariable=&outthreshold[0];
    conf_options[CONF_THRESHOLDI2].thevariable=&inthreshold[1];
    conf_options[CONF_THRESHOLDO2].thevariable=&outthreshold[1];
    conf_options[CONF_THRESHOLDO].comments=conf_options[CONF_THRESHOLDI2].comments=
    conf_options[CONF_THRESHOLDO2].comments=conf_options[CONF_THRESHOLDI].comments;
    #endif
    
    char hours[64];
    sprintf(hours,"%u:%02u", (unsigned)(minutes/60), (unsigned)(minutes%60)); 
    conf_options[CONF_HOURS].thevariable=(char *)(&hours[0]);
    
    #if 0 /* obsolete */
    conf_options[CONF_TIMESLICE].thevariable=&timeslice;
    conf_options[CONF_TIMESLICE].optionscreen=0;
    #endif

    #ifdef OLDNICENESS
    conf_options[CONF_NICENESS].thevariable=&niceness;
    #else
    conf_options[CONF_NICENESS].thevariable=&priority;
    #endif
    
    conf_options[CONF_UUEHTTPMODE].thevariable=&uuehttpmode;
    conf_options[CONF_KEYPROXY].thevariable=(char *)(&keyproxy[0]);
    conf_options[CONF_KEYPROXY].optionscreen=((uuehttpmode < 2)?(3):(0));
    conf_options[CONF_KEYPORT].thevariable=&keyport;
    conf_options[CONF_KEYPORT].optionscreen=((uuehttpmode < 2)?(3):(0));
    conf_options[CONF_HTTPPROXY].thevariable=(char *)(&httpproxy[0]);
    conf_options[CONF_HTTPPROXY].optionscreen=((uuehttpmode < 2)?(0):(3));
    conf_options[CONF_HTTPPORT].thevariable=&httpport;
    conf_options[CONF_HTTPPORT].optionscreen=((uuehttpmode < 2)?(0):(3));
    conf_options[CONF_HTTPID].thevariable=(char *)(&httpid[0]);
    conf_options[CONF_HTTPID].optionscreen=((uuehttpmode < 2)?(0):(3));

    conf_options[CONF_MESSAGELEN].thevariable=&messagelen;
    conf_options[CONF_SMTPSRVR].thevariable=(char *)(&smtpsrvr[0]);
    conf_options[CONF_SMTPPORT].thevariable=&smtpport;
    conf_options[CONF_SMTPFROM].thevariable=(char *)(&smtpfrom[0]);
    conf_options[CONF_SMTPDEST].thevariable=(char *)(&smtpdest[0]);
    conf_options[CONF_SMTPFROM].defaultsetting=
    conf_options[CONF_SMTPDEST].defaultsetting=(char *)conf_options[CONF_ID].thevariable;

    conf_options[CONF_NUMCPU].thevariable=&numcpu;
    conf_options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
    conf_options[CONF_QUIETMODE].thevariable=&quietmode;
    conf_options[CONF_NOEXITFILECHECK].thevariable=&noexitfilecheck;
    conf_options[CONF_PERCENTOFF].thevariable=&percentprintingoff;
    conf_options[CONF_FREQUENT].thevariable=&connectoften;
    conf_options[CONF_NODISK].thevariable=&nodiskbuffers;
    conf_options[CONF_NOFALLBACK].thevariable=&nofallback;
    conf_options[CONF_NETTIMEOUT].thevariable=&nettimeout;

    #if 0 /* obsolete */
    conf_options[CONF_RANDOMPREFIX].thevariable=&randomprefix;
    conf_options[CONF_PROCESSDES].thevariable=&preferred_contest_id;
    #endif
    
    conf_options[CONF_LOGNAME].thevariable=(char *)(&logname[0]);
    conf_options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0][0]);
    
    #if 0 /* obsolete */
    conf_options[CONF_CHECKPOINT2].thevariable=(char *)(&checkpoint_file[1][0]);
    #endif
    
    conf_options[CONF_RC5IN].thevariable=(char *)(&in_buffer_file[0][0]);
    conf_options[CONF_RC5OUT].thevariable=(char *)(&out_buffer_file[0][0]);
    conf_options[CONF_DESIN].thevariable=(char *)(&in_buffer_file[1][0]);
    conf_options[CONF_DESOUT].thevariable=(char *)(&out_buffer_file[1][0]);
    conf_options[CONF_PAUSEFILE].thevariable=(char *)(&pausefile[0]);
    
    conf_options[CONF_CPUTYPE].optionscreen=0; // no config screen if only one core
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
      conf_options[CONF_CPUTYPE].name="cputype";
      conf_options[CONF_CPUTYPE].defaultsetting="-1";
      conf_options[CONF_CPUTYPE].thevariable=&cputype;
      conf_options[CONF_CPUTYPE].choicelist=&cputypetable[1];
      conf_options[CONF_CPUTYPE].choicemin=-1;
      conf_options[CONF_CPUTYPE].choicemax=tablesize-2;
      conf_options[CONF_CPUTYPE].optionscreen=4;
      }
    
    if (offlinemode == 2)
      {
      offlinemode = 0;
      blockcount = -1; 
      }
    conf_options[CONF_OFFLINEMODE].thevariable=&offlinemode;
    conf_options[CONF_COUNT].thevariable=&blockcount;
    
    
    #if (!defined(LURK))
    conf_options[CONF_LURKMODE].optionscreen=0;
    conf_options[CONF_DIALWHENNEEDED].optionscreen=0;
    conf_options[CONF_CONNECTNAME].optionscreen=0;
    #else
    conf_options[CONF_LURKMODE].thevariable=&dialup.lurkmode;
    conf_options[CONF_DIALWHENNEEDED].thevariable=&dialup.dialwhenneeded;
    conf_options[CONF_CONNECTNAME].thevariable=&dialup.connectionname;
    
    char *connectnames = dialup.GetEntryList(&conf_options[CONF_CONNECTNAME].choicemax);
    
    if (conf_options[CONF_CONNECTNAME].choicemax < 1)
      {
      conf_options[CONF_CONNECTNAME].optionscreen=0;
      conf_options[CONF_CONNECTNAME].choicelist=NULL;
      }
    else
      {
      static char *connectnamelist[10];
      if (conf_options[CONF_CONNECTNAME].choicemax>10)
        conf_options[CONF_CONNECTNAME].choicemax=10;
      for (int i=0;i<((int)(conf_options[CONF_CONNECTNAME].choicemax));i++)
        connectnamelist[i]=&(connectnames[i*60]);
      conf_options[CONF_CONNECTNAME].choicelist=(const char **)(&connectnamelist[0]);
      conf_options[CONF_CONNECTNAME].choicemin=0;
      conf_options[CONF_CONNECTNAME].choicemax--;
      };
    #endif

    if (messagelen != 0)
      {
      conf_options[CONF_SMTPSRVR].optionscreen=2;
      conf_options[CONF_SMTPPORT].optionscreen=2;
      conf_options[CONF_SMTPDEST].optionscreen=2;
      conf_options[CONF_SMTPFROM].optionscreen=2;
      }
      else
      {
      conf_options[CONF_SMTPSRVR].optionscreen=0;
      conf_options[CONF_SMTPPORT].optionscreen=0;
      conf_options[CONF_SMTPDEST].optionscreen=0;
      conf_options[CONF_SMTPFROM].optionscreen=0;
      };

conf_options[CONF_DESCONTESTCLOSED].thevariable=(s32*)&descontestclosed;
      
    /* -------------------- end setup options --------------------- */
    
    int lkg_autofind = (autofindkeyserver != 0);
    char lkg_keyproxy[sizeof(keyproxy)];
    strcpy( lkg_keyproxy, keyproxy );
    if ( confopt_isstringblank( keyproxy) || 
      ( autofindkeyserver && confopt_IsHostnameDNetHost( keyproxy ) ))
      {
      autofindkeyserver = 1;
      strcpy( keyproxy, "(auto)" );
      }

    // display menu

    do   //while invalid CONF_xxx option selected
      {
      ConClear(); //in logstuff.cpp
      LogScreenRaw(CONFMENU_CAPTION, menutable[currentmenu-1]);

      for ( temp2=1; temp2 < MAXMENUENTRIES; temp2++ )
        {
        choice=findmenuoption(currentmenu,temp2);
        if (choice >= 0)
          {
          LogScreenRaw("%2d) %s ==> ",
                 (int)conf_options[choice].menuposition,
                 conf_options[choice].description);

          if (conf_options[choice].type==1)
            {
            if (conf_options[choice].thevariable != NULL)
              LogScreenRaw("%s\n",(char *)conf_options[choice].thevariable);
            }
          else if (conf_options[choice].type==2)
            {
            if (conf_options[choice].choicelist != NULL)
              strcpy(parm,conf_options[choice].choicelist[
                ((long)*(s32 *)conf_options[choice].thevariable)]);
            else if ((long)*(s32 *)conf_options[choice].thevariable == 
                (long)(atoi(conf_options[choice].defaultsetting)))
              strcpy(parm,conf_options[choice].defaultsetting);
            else
              sprintf(parm,"%li",(long)*(s32 *)conf_options[choice].thevariable);
            LogScreenRaw("%s\n",parm);
            }
          else if (conf_options[choice].type==3)
            {
            sprintf(parm, "%s", *(s32 *)conf_options[choice].thevariable?"yes":"no");
            LogScreenRaw("%s\n",parm);
            }
          }
        }
      LogScreenRaw("\n 0) Return to main menu\n");

      // get choice from user
      LogScreenRaw("\nChoice --> ");
      ConInStr( parm, 4, 0 );
      choice = atoi( parm );

      if ( choice == 0 || CheckExitRequestTriggerNoIO())
        choice = -2; //quit request
      else if ( choice > 0 )
        choice = findmenuoption(currentmenu,choice); // returns -1 if !found
      else
        choice = -1;
      } while ( choice == -1 ); //while findmenuoption() says this is illegal

    if ( choice >= 0 ) //if valid CONF_xxx option
      {
      ConClear(); //in logstuff.cpp
      LogScreenRaw(CONFMENU_CAPTION, menutable[currentmenu-1]);
      LogScreenRaw("\n%s:\n\n", conf_options[choice].description );
      p = (char *)conf_options[choice].comments;
      while (strlen(p) > (sizeof(parm)-1))
        {
        strncpy(parm,p,(sizeof(parm)-1));
        parm[(sizeof(parm)-1)]=0;
        LogScreenRaw("%s",parm);
        p+=(sizeof(parm)-1);
        }
      LogScreenRaw("%s\n",p);

      if ( conf_options[choice].type==1 || conf_options[choice].type==2 )
        {
        if (conf_options[choice].choicelist !=NULL)
          {
          for ( temp = conf_options[choice].choicemin; temp < conf_options[choice].choicemax+1; temp++)
            LogScreenRaw("  %2d) %s\n", (int) temp,conf_options[choice].choicelist[temp]);
          }
        if (conf_options[choice].type==1)
          strcpy(parm,(char *)conf_options[choice].thevariable);
        else 
          sprintf(parm,"%li",(long)*(s32 *)conf_options[choice].thevariable);
        LogScreenRaw("Default Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
                      conf_options[choice].defaultsetting, parm );
        ConInStr( parm, 64 /*sizeof(parm)*/, 0 /*CONINSTR_BYEXAMPLE*/ );


        for ( p = parm; *p; p++ )
          {
          if ( !isprint(*p) )
            {
            *p = 0;
            break;
            }
          }
        if (strlen(parm) != 0)
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
        }
      else if (conf_options[choice].type==3)
        {
        sprintf(parm, "%s", *(s32 *)conf_options[choice].thevariable?"yes":"no");
        LogScreenRaw("Default Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
               *(conf_options[choice].defaultsetting)=='0'?"no":"yes", parm );
        parm[1]=0;
        ConInStr( parm, 2, CONINSTR_BYEXAMPLE );
        if (parm[0] == 'y' || parm[0] == 'Y')
          strcpy(parm,"yes");
        else if (parm[0] == 'n' || parm[0] == 'N')
          strcpy(parm,"no");
        else parm[0]=0;
        }
      else
        choice = -1;

      if (CheckExitRequestTriggerNoIO())
        choice = -2;
      } //if ( choice >= 0 ) //if valid CONF_xxx option

    if ( choice != CONF_KEYPROXY )
      {
      strcpy( keyproxy, lkg_keyproxy ); //copy whatever they had back
      autofindkeyserver = (lkg_autofind != 0);
      }
    else
      {
      autofindkeyserver = 0; //OFF unless the user left it at auto
      if (!parm[0] || strcmpi(parm,"(auto)")==0 || strcmpi(parm,"auto")==0)
        {
        autofindkeyserver = 1;
        strcpy( parm, lkg_keyproxy ); //copy back what was in there before
        if (confopt_isstringblank( parm )) //dummy value so that we don't fall into
          strcpy( parm, "rc5proxy.distributed.net" ); //the null string trap.
        }
      }

    
    if ( choice >= 0 ) 
      {
      switch ( choice )
        {
        case CONF_ID:
          strncpy( id, parm, sizeof(id) - 1 );
          id[sizeof(id)-1]=0;
          confopt_killwhitespace(id);
          break;
        case CONF_THRESHOLDI:
          choice=atoi(parm);
          if (choice > 0 && choice < 1000) 
            inthreshold[0]=outthreshold[0]=
            inthreshold[1]=outthreshold[1]=choice;
          break;
        case CONF_COUNT:
          choice = atoi(parm);
          blockcount = ((choice < 0)?(-1):(choice));
          break;
        case CONF_HOURS:
          choice = atoi(parm);
          if (parm[0] != 0 && choice>=0)
            {
            choice *= 60;
            if ((p = strchr( parm, ':')) == NULL)
              minutes = choice;
            else if (strlen(p)==3 && isdigit(p[1]) && isdigit(p[2]) 
              && atoi(p+1)>=0 && atoi(p+1) <=59)
              minutes = choice + atoi(p+1);
            }
          break;
        case CONF_TIMESLICE:  
          #if 0 /* obsolete */      
          if ((choice = atoi(parm)) >= 1)
            timeslice = choice;
          #endif
          break;
        case CONF_NICENESS:
          choice = atoi(parm);
          if ( choice >= conf_options[CONF_NICENESS].choicemin && 
               choice <= conf_options[CONF_NICENESS].choicemax )
            #ifdef OLDNICENESS
            niceness = choice;
            #else
            priority = choice;
            #endif
          break;
        case CONF_LOGNAME:
          strncpy( logname, parm, sizeof(logname) - 1 );
          logname[sizeof(logname)-1]=0;
          confopt_killwhitespace(logname);
          if (confopt_isstringblank(logname)) 
            logname[0]=0;
          break;
        case CONF_KEYPROXY:
          strncpy( keyproxy, parm, sizeof(keyproxy) - 1 );
          keyproxy[sizeof(keyproxy)-1]=0;
          confopt_killwhitespace(keyproxy);
          p = strchr( keyproxy, '.' );
          if (p != NULL && confopt_IsHostnameDNetHost( keyproxy ))
            {
            do{ --p; 
              if (p < (&keyproxy[0]))
                { p++; break; }
              if (!isdigit(*p))
                { p++; break; }
              } while (isdigit(*p));
            choice = ((!isdigit(*p))?(2064):(atoi(p)));
            strcpy( p, ".v27.distributed.net");
            if (choice == 2064 || choice == 80 || choice == 23)
              keyport = choice;
            else if (keyport != 3064 || keyport != 2064 || choice != 80 || choice != 23)
              keyport = 2064;
            }
          break;
        case CONF_KEYPORT:
          choice = atoi(parm);
          if (confopt_IsHostnameDNetHost( keyproxy ) && 
             (keyport!=3064 || keyport!=2064 || choice!=80 || choice!=23))
            keyport = 2064;
          else if (choice > 0 && choice <= 65535)
            keyport = choice;
          break;
        case CONF_HTTPPROXY:
          strncpy( httpproxy, parm, sizeof(httpproxy) - 1);
          httpproxy[sizeof(httpproxy)-1]=0;
          confopt_killwhitespace(httpproxy);
          break;
        case CONF_HTTPPORT:
          choice = atoi(parm); 
          if (choice > 0 && choice <= 65535)
            httpport = choice;
          break;
        case CONF_HTTPID:
          if ( parm[0] == 0 || strcmp(parm,".") == 0)
            httpid[0]=0;
          else if (uuehttpmode == 4) // socks4
            strcpy(httpid, parm);
          else
            {             // http & socks5
            LogScreenRaw("Enter password--> ");

            ConInStr( parm2, sizeof(parm2), 0 /* CONINSTR_ASPASSWORD */ );
            for ( p = parm2; *p; p++ )
              {
              if ( !isprint(*p) )
                {
                *p = 0;
                break;
                }
              }
            if (uuehttpmode == 5)   // socks5
              sprintf(httpid, "%s:%s", parm, parm2);
            else                    // http
              strcpy(httpid,Network::base64_encode(parm, parm2));
            }
          break;
        case CONF_UUEHTTPMODE:
          choice = atoi(parm);
          if (choice >= conf_options[CONF_UUEHTTPMODE].choicemin && 
              choice <= conf_options[CONF_UUEHTTPMODE].choicemax)
            {
            uuehttpmode = choice;
            if ( choice > 0 )
              {
              autofindkeyserver=1; //we are using a default, so turn it back on
              switch (uuehttpmode)
                {
                case 1: /* UUE mode (telnet) */ p = "23"; keyport = 23; break;   
                case 2: /* HTTP mode */
                case 3: /* HTTP+UUE mode */     p = "80"; keyport = 80; break;
                case 4: /* SOCKS4 */
                case 5: /* SOCKS5 */
                default:/* normal */            p = ""; keyport = 2064; break;
                }
              strcpy(keyproxy,"rc5proxy.distributed.net");
              //sprintf(keyproxy,"us%s.v27.distributed.net", p );
              }
            }
          break;
        case CONF_CPUTYPE:
          choice = ((parm[0] == 0)?(-1):(atoi(parm)));
          if (choice >= conf_options[CONF_CPUTYPE].choicemin && 
              choice <= conf_options[CONF_CPUTYPE].choicemax)
            cputype = choice;
          break;
        case CONF_MESSAGELEN:
          choice = atoi(parm);
          if (choice == 0 || 
             (choice >= conf_options[CONF_MESSAGELEN].choicemin && 
              choice <= conf_options[CONF_MESSAGELEN].choicemax))
            messagelen = choice;
          break;
        case CONF_SMTPPORT:
          choice = atoi(parm);
          if (choice > 0 && choice <= 65535 ) 
            smtpport = choice;
          break;
        case CONF_SMTPSRVR:
          strncpy( smtpsrvr, parm, sizeof(smtpsrvr) - 1 );
          smtpsrvr[sizeof(smtpsrvr) - 1]=0;
          confopt_killwhitespace(smtpsrvr);
          break;
        case CONF_SMTPFROM:
          strncpy( smtpfrom, parm, sizeof(smtpfrom) - 1 );
          smtpfrom[sizeof(smtpfrom) - 1]=0;
          confopt_killwhitespace(smtpfrom);
          if (smtpfrom[0]==0) 
            strcpy(smtpfrom,id);
          break;
        case CONF_SMTPDEST:
          strncpy( smtpdest, parm, sizeof(smtpdest) - 1 );
          smtpdest[sizeof(smtpdest) - 1]=0;
          if (confopt_isstringblank(smtpdest))
            strcpy(smtpdest,id);
          break;
        case CONF_NUMCPU:
          numcpu = atoi(parm);
          break; //validation is done in SelectCore() 1998/06/21 cyrus
        case CONF_CHECKPOINT:
          strncpy( checkpoint_file[0] , parm, sizeof(checkpoint_file[1]) -1 );
          checkpoint_file[0][sizeof(checkpoint_file[0]) - 1]=0;
          confopt_killwhitespace(checkpoint_file[0]);
          break;
        case CONF_CHECKPOINT2:
          #if 0 /* obsolete */
          strncpy( checkpoint_file[1] , parm, sizeof(checkpoint_file[1]) -1 );
          checkpoint_file[1][sizeof(checkpoint_file[1]) - 1]=0;
          confopt_killwhitespace(checkpoint_file[1]);
          #endif
          break;
        case CONF_PREFERREDBLOCKSIZE:
          choice = atoi(parm);
          if (choice >= conf_options[CONF_PREFERREDBLOCKSIZE].choicemin && 
              choice <= conf_options[CONF_PREFERREDBLOCKSIZE].choicemax)
            preferred_blocksize = choice;
          break;
        case CONF_PROCESSDES:
          #if 0 /* obsolete */
          choice = yesno(parm);
          if ((choice >= 0) && (choice <= 1))
             preferred_contest_id = choice;
          #endif
          break;
        case CONF_QUIETMODE:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_QUIETMODE].thevariable=choice;
            }
          break;
        case CONF_NOEXITFILECHECK:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_NOEXITFILECHECK].thevariable=choice;
            }
          break;
        case CONF_PERCENTOFF:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_PERCENTOFF].thevariable=choice;
            }
          break;
        case CONF_FREQUENT:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_FREQUENT].thevariable=choice;
            }
          break;
        case CONF_NODISK:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_NODISK].thevariable=choice;
            }
          break;
        case CONF_NOFALLBACK:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) 
              *(s32 *)conf_options[CONF_NOFALLBACK].thevariable=choice;
            }
          break;
        case CONF_NETTIMEOUT:
          choice=atoi(parm);
          if (choice >= conf_options[CONF_NETTIMEOUT].choicemin && 
              choice <= conf_options[CONF_NETTIMEOUT].choicemax)
            *(s32 *)conf_options[CONF_NETTIMEOUT].thevariable=choice;
          break;
        case CONF_OFFLINEMODE:
          if (parm[0] != 0)
            {
            choice=atoi(parm);
            if (choice >= conf_options[CONF_OFFLINEMODE].choicemin && 
                choice <= conf_options[CONF_OFFLINEMODE].choicemax)
              *(s32 *)conf_options[CONF_OFFLINEMODE].thevariable=choice;
            }
          break;
        case CONF_RC5IN:
          strncpy( in_buffer_file[0] , parm, sizeof(in_buffer_file[0]) -1 );
          in_buffer_file[0][sizeof(in_buffer_file[0]) - 1]=0;
          confopt_killwhitespace(in_buffer_file[0]);
          if (in_buffer_file[0][0]==0)
            strcpy(in_buffer_file[0],conf_options[CONF_RC5IN].defaultsetting);
          break;
        case CONF_RC5OUT:
          strncpy( out_buffer_file[0] , parm, sizeof(out_buffer_file[0]) -1 );
          out_buffer_file[0][sizeof(out_buffer_file[0]) - 1]=0;
          confopt_killwhitespace(out_buffer_file[0]);
          if (out_buffer_file[0][0]==0)
            strcpy(out_buffer_file[0],conf_options[CONF_RC5OUT].defaultsetting);
          break;
        case CONF_DESIN:
          strncpy( in_buffer_file[1] , parm, sizeof(in_buffer_file[1]) -1 );
          in_buffer_file[1][sizeof(in_buffer_file[1]) - 1]=0;
          confopt_killwhitespace(in_buffer_file[1]);
          if (in_buffer_file[1][0]==0)
            strcpy(in_buffer_file[1],conf_options[CONF_DESIN].defaultsetting);
          break;
        case CONF_DESOUT:
          strncpy( out_buffer_file[1] , parm, sizeof(out_buffer_file[1]) -1 );
          out_buffer_file[1][sizeof(out_buffer_file[1]) - 1]=0;
          confopt_killwhitespace(out_buffer_file[1]);
          if (out_buffer_file[1][0]==0)
            strcpy(out_buffer_file[1],conf_options[CONF_DESOUT].defaultsetting);
          break;
        case CONF_PAUSEFILE:
          strncpy( pausefile, parm, sizeof(pausefile) -1 );
          pausefile[sizeof(pausefile) - 1]=0;
          confopt_killwhitespace(pausefile);
          if (confopt_isstringblank(pausefile)) 
            pausefile[0]=0;
          break;
        #ifdef LURK
        case CONF_LURKMODE:
          if (parm[0] != 0)
            {
            choice=atoi(parm);
            if (choice>=0 && choice<=2)
              {
              dialup.lurkmode=choice;
              if (choice!=1)
                connectoften=0;
              }
            }
          break;
        case CONF_DIALWHENNEEDED:
          if (parm[0] != 0)
            {
            choice=yesno(parm);
            if (choice >= 0) *(s32 *)conf_options[CONF_DIALWHENNEEDED].thevariable=choice;
            }
          break;
        case CONF_CONNECTNAME:
          if (parm[0] != 0)
            {
            choice=atoi(parm);
            if ( ((choice > 0) || (parm[0]=='0')) &&
                 (choice <= conf_options[CONF_CONNECTNAME].choicemax) )
              {
              strcpy( (char *)conf_options[CONF_CONNECTNAME].thevariable,
                       conf_options[CONF_CONNECTNAME].choicelist[choice]);
              }
            else strncpy( (char *)conf_options[CONF_CONNECTNAME].thevariable,
                        parm, sizeof(dialup.connectionname)-1);
            }
          break;
        #endif
        default:
          break;
        }
      choice = 1; // continue with menu
      }
    } while ( choice >= 0 ); //while we have a valid CONF_xxx to work with


  return 0;
}
#endif

//----------------------------------------------------------------------------

s32 Client::Configure( void )
//A return of 1 indicates to save the changed configuration
//A return of -1 indicates to NOT save the changed configuration
{
  int returnvalue = 1;

#if !defined(NOCONFIG)
  int pos;
  returnvalue=0;

  if (!ConIsScreen())
    {
    ConOutErr("Can't configure when stdin or stdout is redirected.\n");
    returnvalue = -1;
    }
  
  while (returnvalue == 0)
    {
    ConClear();
    LogScreenRaw(CONFMENU_CAPTION, "");
    for (pos=1;pos<=(int)(sizeof(menutable)/sizeof(menutable[0]));pos++)
      LogScreenRaw(" %u) %s\n",pos, menutable[pos-1]);
    LogScreenRaw("\n 9) Discard settings and exit"
                 "\n 0) Save settings and exit\n\n");

    if (confopt_isstringblank(id) || strcmpi(id,"rc5@distributed.net")==0)
      LogScreenRaw("Note: You have not yet provided a distributed.net ID.\n"
              "       Please go to the '%s' and set it.\n\n",menutable[0]);

    LogScreenRaw("Choice --> ");
    char chbuff[10];    
    ConInStr(chbuff, 2, 0);
    pos = ((strlen(chbuff)==1 && isdigit(chbuff[0]))?(atoi(chbuff)):(-1));
    
    if (CheckExitRequestTriggerNoIO() || pos==9)
      returnvalue = -1; //Breaks and tells it NOT to save
    else if (pos < 0) 
      ; /* nothing - ignore it */
    else if (pos == 0)
      returnvalue=1; //Breaks and tells it to save
    else if (pos<=(int)(sizeof(menutable)/sizeof(menutable[0])))
      {
      ConfigureGeneral(pos);
      if (CheckExitRequestTriggerNoIO())
        returnvalue = -1;
      }
    }
#endif
  return returnvalue;
}

//----------------------------------------------------------------------------

