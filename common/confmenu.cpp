// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confmenu.cpp,v $
// Revision 1.18  1999/01/17 13:50:11  cyp
// buffer thresholds must be volatile.
//
// Revision 1.17  1999/01/12 14:57:35  cyp
// -1 is a legal nettimeout value (force blocking net i/o).
//
// Revision 1.16  1999/01/12 14:37:42  cyp
// re-prompt the user on range error.
//
// Revision 1.15  1999/01/07 02:20:35  cyp
// Greatly improved "yes"/"no" editing with my rad ConInStr() bool mode. :)
//
// Revision 1.14  1999/01/04 19:02:25  chrisb
// fixed an erroneous implicit cast
//
// Revision 1.13  1999/01/04 04:49:17  cyp
// Cleared a 'potential integer truncation' warning.
//
// Revision 1.12  1999/01/04 02:47:30  cyp
// Cleaned up menu options and handling.
//
// Revision 1.11  1999/01/03 02:28:27  cyp
// Added bounds check before displaying an optionlist.
//
// Revision 1.10  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.9  1998/12/30 19:19:28  cyp
// Fixed broken offlinemode (not being able to change).
//
// Revision 1.8  1998/12/28 03:03:39  silby
// Fixed problem with filenames having whitespace stripped from them.
//
// Revision 1.7  1998/12/21 18:40:12  cyp
// Removed 'unused'/'unimplemented' sil[l|b]yness committed in 1.3/1.4
//
// Revision 1.6  1998/12/21 01:21:39  remi
// Recommitted to get the right modification time.
//
// Revision 1.5  1998/12/21 14:23:58  remi
// Fixed the weirdness of proxy, keyport, uuehttpmode etc... handling :
// - if keyproxy ends in .distributed.net, keyport and uuehttpmode are
//   kept in sync in ::ValidateConfig()
// - if uuehttpmode == 2|3, keyport = 80 and can't be changed
//   (port 80 is hardwired in the http code somewhere in network.cpp)
// - do not delete uuehttpmode from the .ini file when it's > 1 (!!)
//   this bug makes client <= 422 difficult to use with http or socks...
// - write keyport in the .ini only if it differs from the default
//   (2064 | 80 | 23) for a given uuehttmode
// - fixed bugs in ::ConfigureGeneral() related to autofindkeyserver,
//   uuehttpmode, keyproxy, and keyport.
//
// Revision 1.4  1998/12/21 00:21:01  silby
// Universally scheduled update time is now retrieved from the proxy,
// and stored in the .ini file.  Not yet used, however.
//
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
// well as both DES thresholds from the menu. Removed 'processdes' from the 
// menu. Fixed various bugs. Range validation is always based on the min/max 
// values in the option table.
//


#if (!defined(lint) && defined(__showids__))
const char *confmenu_cpp(void) {
return "@(#)$Id: confmenu.cpp,v 1.18 1999/01/17 13:50:11 cyp Exp $"; }
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

static int findmenuoption( int menu, int option )
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
  unsigned int tpos;

  for (tpos=0; tpos < CONF_OPTION_COUNT; tpos++)
    {
    if ((conf_options[tpos].optionscreen==menu) &&
        (conf_options[tpos].menuposition==option))
      {
      return ((conf_options[tpos].thevariable==NULL)?(-1):((int)(tpos)));
      }
    }
  return -1;
}

// --------------------------------------------------------------------------

#if 0
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
#endif

// --------------------------------------------------------------------------

static unsigned char base64table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
                                     "ghijklmnopqrstuvwxyz0123456789+/";

int base64_encode(char *outbuf, const char *inbuf) //outbuff must be at least
{                                                 //(strlen(inbuf))*4/3) bytes  
  unsigned int length = strlen(inbuf)+1;           
  
  #define B64_ENC(Ch) (char) (base64table[(char)(Ch) & 63])

  for (; length > 2; length -= 3, inbuf += 3)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
    *outbuf++ = B64_ENC(((inbuf[1] << 2) & 074) | ((inbuf[2] >> 6) & 03));
    *outbuf++ = B64_ENC(inbuf[2] & 077);
    }
  if (length == 1)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC((inbuf[0] << 4) & 060);
    *outbuf++ = '=';
    *outbuf++ = '=';
    }
  else if (length == 2)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
    *outbuf++ = B64_ENC((inbuf[1] << 2) & 074);
    *outbuf++ = '=';
    }
  *outbuf = 0;

  return 0;
}


int base64_decode(char *outbuf, const char *inbuf )
{
  static char inalphabet[256], decoder[256];
  int i, bits, c, char_count, errors = 0;

   for (i = (64/*sizeof(base64table)*/-1); i >= 0 ; i--) 
     {
     inalphabet[base64table[i]] = 1;
     decoder[base64table[i]] = ((unsigned char)(i));
     }
  char_count = 0;
  bits = 0;
  while ((c = *inbuf++) != 0) 
    {
    if (c == '=')
      {
      switch (char_count) 
        {
        case 1:
          //base64 encoding incomplete: at least 2 bits missing
          errors++;
          break;
        case 2:
          *outbuf++ = (char)((bits >> 10));
          break;
        case 3:
          *outbuf++ = (char)((bits >> 16));
          *outbuf++ = (char)(((bits >> 8) & 0xff));
          break;
        }
      break;
      }
    if (c > 255 || ! inalphabet[c])
      continue;
    bits += decoder[c];
    char_count++;
    if (char_count == 4) 
      {
      *outbuf++ = (char)((bits >> 16));
      *outbuf++ = (char)(((bits >> 8) & 0xff));
      *outbuf++ = (char)((bits & 0xff));
      bits = 0;
      char_count = 0;
      } 
    else 
      {
      bits <<= 6;
      }
    }
  if (c == 0 && char_count) 
    {
    //base64 encoding incomplete: at least ((4 - char_count) * 6)) bits truncated
    errors++;
    }
  return ((errors)?(-1):(0));
}

// --------------------------------------------------------------------------

int Client::Configure( void )
//A return of 1 indicates to save the changed configuration
//A return of -1 indicates to NOT save the changed configuration
{
  int whichmenu;
  int returnvalue=0;

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
    id[0]=0; /*is later converted back to 'rc5@distributed.net' */
  conf_options[CONF_ID].thevariable=(char *)(&id[0]);
  conf_options[CONF_THRESHOLDI].thevariable=&threshold;
  conf_options[CONF_THRESHOLDI].choicemax=MAXBLOCKSPERBUFFER; /* client.h */
  conf_options[CONF_FREQUENT].thevariable=&connectoften;
  conf_options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
  conf_options[CONF_NODISK].thevariable=&nodiskbuffers;
  conf_options[CONF_RC5IN].thevariable=(char *)(&in_buffer_file[0][0]);
  conf_options[CONF_RC5OUT].thevariable=(char *)(&out_buffer_file[0][0]);
  conf_options[CONF_DESIN].thevariable=(char *)(&in_buffer_file[1][0]);
  conf_options[CONF_DESOUT].thevariable=(char *)(&out_buffer_file[1][0]);
  conf_options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0]);

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

  if (offlinemode == 2) { offlinemode = 0; blockcount = -1; }
  s32 autofindks = (autofindkeyserver!=0);
  conf_options[CONF_OFFLINEMODE].thevariable=&offlinemode;
  conf_options[CONF_NETTIMEOUT].thevariable=&nettimeout;
  conf_options[CONF_UUEHTTPMODE].thevariable=&uuehttpmode;
  conf_options[CONF_KEYSERVNAME].thevariable=(char *)(&keyproxy[0]);
  conf_options[CONF_KEYSERVPORT].thevariable=&keyport;
  conf_options[CONF_AUTOFINDKS].thevariable=&autofindks;
  conf_options[CONF_NOFALLBACK].thevariable=&nofallback;

  conf_options[CONF_LURKMODE].thevariable=
  conf_options[CONF_DIALWHENNEEDED].thevariable=
  conf_options[CONF_CONNECTNAME].thevariable=NULL;

  #if defined(LURK)
  conf_options[CONF_LURKMODE].optionscreen=
  conf_options[CONF_DIALWHENNEEDED].optionscreen=
  conf_options[CONF_CONNECTNAME].optionscreen=CONF_MENU_NET;
  conf_options[CONF_LURKMODE].thevariable=&dialup.lurkmode;
  conf_options[CONF_DIALWHENNEEDED].thevariable=&dialup.dialwhenneeded;
  conf_options[CONF_CONNECTNAME].thevariable=&dialup.connectionname;
  char *connectnames = dialup.GetEntryList(&conf_options[CONF_CONNECTNAME].choicemax);
  if (conf_options[CONF_CONNECTNAME].choicemax < 1)
    {
    conf_options[CONF_CONNECTNAME].optionscreen=CONF_MENU_NET;
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
    }
  #endif
  conf_options[CONF_FWALLHOSTNAME].thevariable=(char *)(&httpproxy[0]);
  conf_options[CONF_FWALLHOSTPORT].thevariable=&httpport;
  struct { char username[128], password[128]; } userpass;
  userpass.username[0]=userpass.password[0]=0;
  if (httpid[0]==0)
    ; //nothing
  else if (uuehttpmode == 2 || uuehttpmode == 3)
    {
    char *p;
    if (strlen( userpass.username ) > 80) /* not rfc compliant (max 76) */
      userpass.username[0]=0;
    if (base64_decode(userpass.username, httpid )!=0) 
      userpass.username[0]=0;                         /* bit errors */
    else if (userpass.username[strlen(userpass.username)-1]!='\n')
      userpass.username[0]=0;                         /* wrong format */
    else if ((p = strchr( userpass.username, ':' )) == NULL)
      userpass.username[0]=0;                         /* wrong format */
    else
      {
      *p++=0;
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

  conf_options[CONF_CPUTYPE].thevariable=&cputype;
  conf_options[CONF_CPUTYPE].optionscreen=0;
  conf_options[CONF_CPUTYPE].choicemax=0;
  conf_options[CONF_CPUTYPE].choicemin=0;
  const char *corename = GetCoreNameFromCoreType(0);
  if (corename && *corename)
    {
    static const char *cputypetable[10];
    unsigned int tablesize = 2;
    cputypetable[0]="Autodetect";
    cputypetable[1]=corename;
    do{
      corename = GetCoreNameFromCoreType(tablesize-1);
      if (!corename || !*corename)
        break;
      cputypetable[tablesize++]=corename;
      } while (tablesize<((sizeof(cputypetable)/sizeof(cputypetable[0]))));
    conf_options[CONF_CPUTYPE].name="cputype";
    conf_options[CONF_CPUTYPE].choicelist=&cputypetable[1];
    conf_options[CONF_CPUTYPE].choicemin=-1;
    conf_options[CONF_CPUTYPE].choicemax=tablesize-2;
    conf_options[CONF_CPUTYPE].optionscreen=CONF_MENU_PERF;
    }
  conf_options[CONF_NICENESS].thevariable=&priority;
  conf_options[CONF_NUMCPU].thevariable=&numcpu;

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
    for (whichmenu=1;whichmenu<=(int)(sizeof(menutable)/sizeof(menutable[0]));whichmenu++)
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
      returnvalue=1; //Breaks and tells it to save
    else if (whichmenu<=(int)(sizeof(menutable)/sizeof(menutable[0])))
      {
      int userselection = 0;
      int redoselection = -1;
                                 //note: don't return or break from inside 
      while (userselection >= 0) //the loop. Let it fall through instead. - cyp
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


        /* --------------- drop/pickup menu options ---------------- */

        if (whichmenu == CONF_MENU_BUFF)
          {
          conf_options[CONF_FREQUENT].optionscreen= 
                      ((nodiskbuffers || offlinemode)?(0):(CONF_MENU_BUFF));

          conf_options[CONF_RC5IN].optionscreen=
          conf_options[CONF_DESIN].optionscreen=
          conf_options[CONF_RC5OUT].optionscreen=
          conf_options[CONF_DESOUT].optionscreen=
          conf_options[CONF_CHECKPOINT].optionscreen=
                      ((nodiskbuffers != 0)?(0):(CONF_MENU_BUFF));

          conf_options[CONF_PREFERREDBLOCKSIZE].optionscreen=
          conf_options[CONF_THRESHOLDI].optionscreen= 
                      ((offlinemode)?(0):(CONF_MENU_BUFF));
          }
        else if (whichmenu == CONF_MENU_LOG)
          {
          conf_options[CONF_SMTPSRVR].optionscreen=
          conf_options[CONF_SMTPPORT].optionscreen=
          conf_options[CONF_SMTPDEST].optionscreen=
          conf_options[CONF_SMTPFROM].optionscreen=
                      ((offlinemode || messagelen <= 0)?(0):(CONF_MENU_LOG));
          }
        else if (whichmenu == CONF_MENU_NET)
          {
          conf_options[CONF_NOFALLBACK].optionscreen= //can't fallback to self
          conf_options[CONF_NETTIMEOUT].optionscreen = /* 60 if auto */
          conf_options[CONF_KEYSERVNAME].optionscreen = 
                      ((autofindks || offlinemode)?(0):(CONF_MENU_NET));

          conf_options[CONF_FWALLHOSTNAME].optionscreen=
          conf_options[CONF_FWALLHOSTPORT].optionscreen=
          conf_options[CONF_FWALLUSERNAME].optionscreen=
                      ((uuehttpmode>=2 && !offlinemode)?(CONF_MENU_NET):(0));

          conf_options[CONF_FWALLPASSWORD].optionscreen=
                      (((uuehttpmode==2 || uuehttpmode==3   
                      || uuehttpmode==5) && !offlinemode)?(CONF_MENU_NET):(0));

          conf_options[CONF_UUEHTTPMODE].optionscreen=
          conf_options[CONF_AUTOFINDKS].optionscreen=
          conf_options[CONF_KEYSERVPORT].optionscreen=
          conf_options[CONF_LURKMODE].optionscreen=
          conf_options[CONF_DIALWHENNEEDED].optionscreen=
          conf_options[CONF_CONNECTNAME].optionscreen=
                      ((offlinemode)?(0):(CONF_MENU_NET));
          }

          
        /* -------------------- display menu -------------------------- */
        
        ConClear();
        LogScreenRaw(CONFMENU_CAPTION, menutable[whichmenu-1]);
        
        unsigned int menuoption;
        for (menuoption = 1; menuoption < MAXMENUENTRIES; menuoption++)
          {
          int seloption = findmenuoption( whichmenu, menuoption );
          if (seloption >= 0 && conf_options[seloption].thevariable != NULL)
            {
            p = NULL;

            if (conf_options[seloption].type==CONF_TYPE_ASCIIZ)
              {
              p = (char *)conf_options[seloption].thevariable;
              }
            else if (conf_options[seloption].type==CONF_TYPE_PASSWORD)
              {
              int i=strlen((char *)conf_options[seloption].thevariable);
              memset(parm,'*',i);
              parm[i]=0;
              p = parm;
              }
            else if (conf_options[seloption].type==CONF_TYPE_TIMESTR)
              {
              long t=(long)*((s32 *)conf_options[seloption].thevariable);
              sprintf(parm,"%ld:%02u", (t/60), 
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
                sprintf(parm,"%li", thevar );
                p = parm;
                }
              }
            else if (conf_options[seloption].type==CONF_TYPE_BOOL)
              {
              p = (char *)(((*(s32 *)conf_options[seloption].thevariable)?("yes"):("no")));
              }
            if (p)
              {
              char parm2[128];
              unsigned int optlen = sprintf(parm2, "%2d) %s ==> ",
                                  (int)conf_options[seloption].menuposition,
                                  conf_options[seloption].description);
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
            userselection = findmenuoption(whichmenu,userselection); //-1 if !found
          else
            userselection = -1;
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
            strncpy(parm,p,(sizeof(parm)-1));
            parm[(sizeof(parm)-1)]=0;
            LogScreenRaw("%s",parm);
            p+=(sizeof(parm)-1);
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
              memset(parm,'*',i);
              parm[i]=0;
              coninstrmode = CONINSTR_ASPASSWORD;
              }
            else if (conf_options[userselection].type==CONF_TYPE_ASCIIZ)
              {
              strcpy(parm,(char *)conf_options[userselection].thevariable);
              p = (char *)(conf_options[userselection].defaultsetting);
              }
            else //if (conf_options[userselection].type==CONF_TYPE_INT)
              {
              sprintf(parm,"%li",(long)*(s32 *)conf_options[userselection].thevariable);
              sprintf(defaultbuff,"%li",atol(conf_options[userselection].defaultsetting));
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
              parm[i]=0;
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
              }
            }
          else if (conf_options[userselection].type==CONF_TYPE_TIMESTR)
            {
            long t =(long)*((s32 *)conf_options[userselection].thevariable);
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
                      if (dotpos!=0 || (parm[pos]!=':' && parm[pos]!='.'))
                        {
                        isok = 0;
                        break;
                        }
                      dotpos=pos;
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
            parm[1]=0;
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
          //DO NOT TOUCH ANY VARIABLE EXCEPT THE SELECTED ONE
          //(unless those variables are not menu options)
          //DO IT AFTER ALL MENU DRIVEN CONFIG IS FINISHED (see end)
          
          if (userselection == CONF_ID || userselection == CONF_KEYSERVNAME ||
            userselection == CONF_SMTPFROM || userselection == CONF_SMTPSRVR ||
            userselection == CONF_FWALLHOSTNAME)
            {
            confopt_killwhitespace(parm);
            }
          #ifdef LURK
          if (userselection == CONF_CONNECTNAME) /* wierdo */
            {
            if (*newval_z != 0)
              {
              newval_d=atoi(newval_z);
              if ( ((newval_d > 0) || (newval_z[0]=='0')) &&
                   (newval_d <= conf_options[CONF_CONNECTNAME].choicemax) )
                {
                strcpy( (char *)conf_options[CONF_CONNECTNAME].thevariable,
                      conf_options[CONF_CONNECTNAME].choicelist[newval_d]);
                }
              else strncpy( (char *)conf_options[CONF_CONNECTNAME].thevariable,
                          newval_z, sizeof(dialup.connectionname)-1);
              }
            }
          else
          #endif
          if (conf_options[userselection].type==CONF_TYPE_ASCIIZ ||
              conf_options[userselection].type==CONF_TYPE_PASSWORD)
            {
            if (confopt_isstringblank(parm)) 
              parm[0]=0;
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
    if (id[0]==0)
      strcpy(id,"rc5@distributed.net");

    autofindkeyserver = (autofindks!=0);

    if (nettimeout < 0)
      nettimeout = -1;
    else if (nettimeout < 5)
      nettimeout = 5;

    #ifdef LURK
    if (dialup.lurkmode != 1)
      connectoften=0;
    #endif

    if (strlen(userpass.username)==0 && strlen(userpass.password)==0)
      httpid[0]=0;
    else if (uuehttpmode == 2 || uuehttpmode == 3)
      {
      if (((strlen(userpass.username)+strlen(userpass.username)+4)*4/3) >
        sizeof(httpid)) /* too big. what should we do? */
        httpid[0]=0; 
      else
        {
        strcat(userpass.username,":");
        strcat(userpass.username,userpass.password);
        strcat(userpass.username,"\n");
        base64_encode(httpid,userpass.username);
        }
      }
    else if (uuehttpmode == 4)
      {
      strcpy( httpid, userpass.username );
      }
    else if (uuehttpmode == 5)
      {
      strcat(userpass.username,":");
      strcat(userpass.username,userpass.password);
      strncpy( httpid, userpass.username, sizeof( httpid )-1);
      httpid[sizeof( httpid )-1]=0;
      }
    }

  //fini
    
  return returnvalue;
}

