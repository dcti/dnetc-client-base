/*
 * -install and -uninstall support for Linux (maybe all SysV style init?)
 *
 * Created Aug 23 2000, by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * $Id: li_inst.c,v 1.1.2.2 2001/02/18 23:58:40 cyp Exp $
 *
*/
#define __NO_STRING_INLINES /* work around bugs in glibc bits/string2.h */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <errno.h>

#ifdef __cplusplus /* to ensure gcc -lang switches don't mess this up */
extern "C" {
#endif
int linux_uninstall(const char *basename, int quietly);
int linux_install(const char *basename, int argc, const char *argv[], 
                  int quietly); /* argv[1..(argc-1)] are boot options */
#ifdef __cplusplus
}
#endif

#define get_KS_for_runlevel(run_level) \
        ((run_level>=2 && run_level<6)?("S90"):("K09"))

static const char *get_etc_base(void) 
{ 
  /* where to look for init.d and rc?.d subdirectories */
  static const char *poss_bases[] = { "/etc/rc.d/", 
				      "/etc/" };
  unsigned int i;
  for (i=0; i < (sizeof(poss_bases)/sizeof(poss_bases[0])); i++)
  {
    char path[64];
    struct stat statblk;
    strcat(strcpy(path,poss_bases[i]),"init.d");
    if (stat(path,&statblk)==0)
      return poss_bases[i];
  }
  return (const char *)0;    
}

int linux_uninstall(const char *basename, int quietly)
{
  const char *etc_base = get_etc_base();
  if (!etc_base)
  {
    if (!quietly)
      fprintf(stderr,
      "%s:\tUnable to determine where init scripts (/etc[/rc.d]/init.d/* and\n"
      "\t/etc[/rc.d]/rc?.d/*) are located.\n", basename );
  }
  else
  {
    char fullpath[NAME_MAX+64];
    int rcd, errors = 0, numfound = 0;
    for (rcd = 0; rcd <= 6; rcd++)
    {
      const char *kstype = get_KS_for_runlevel(rcd);
      sprintf(fullpath,"%src%d.d/%s%s",etc_base,rcd,kstype,basename);
      if (access(fullpath,0)==0)
      { 
        numfound++;
        if (unlink(fullpath)!=0)
        {
          int rcd2; char srcpath[64];
          if (!quietly) 
	    fprintf(stderr,"%s: Unable to unlink %s: %s\n", basename,
                              fullpath, strerror(errno));
          strcat(strcpy(srcpath,"../init.d/"),basename);
          for (rcd2 = 0; rcd2 < rcd; rcd2++)
          {
            kstype = get_KS_for_runlevel(rcd2);
            sprintf(fullpath,"%src%d.d/%s%s",etc_base,rcd2,kstype,basename);
            if (symlink(srcpath,fullpath)!=0)
      	    {  
  	      if (!quietly)
                fprintf(stderr,"%s: Unable to re-link %s: %s\n", basename,
                                fullpath, strerror(errno));
            }			    
   	  }  
          errors++;
  	  break;
        }    
      } /* if exists */
    } /* for rcd = 0; rcd <= 6; rcd++ */   
    if (errors == 0)
    {
      strcat(strcat(strcpy(fullpath,etc_base),"init.d/"),basename);
      if (access(fullpath,0)==0)
      {
        numfound++;
        if (remove(fullpath)!=0)
        {
          errors++;
          if (!quietly)
            fprintf(stderr,"%s: Unable to remove %s: %s\n", basename,
                                fullpath, strerror(errno));
        }	
      }	
    }
    if (errors == 0)
    {
      if (!quietly)
      {
        if (numfound > 0)
          fprintf(stderr,
          "%s:\tThe client has been sucessfully removed from\n" 
          "\t%sinit.d and %src?.d. If you wish to ensure\n"
          "\tthat no clients are running (now) use '%s -shutdown'\n", 
          basename, etc_base, etc_base, basename );
        else
          fprintf(stderr,
  	  "%s:\tThe client does not appear to be installed in\n"
	  "\t%sinit.d or %src?d.\n", basename, etc_base, etc_base );
      }
      return numfound;
    }
  }    
  return -1;
}



#ifndef MAXPATHLEN
  #ifdef PATH_MAX
    #define MAXPATHLEN PATH_MAX+1
  #else
    #define MAXPATHLEN 1024+1
  #endif     
#endif

static int determine_appname(char *buffer,unsigned int bufsize)
{
  char fullpath[MAXPATHLEN];
  char *app = getenv("_");
  if (app)
  {
    app = realpath(app,fullpath);
  }
  else
  {
    FILE *file;
    char procstat[256];
    sprintf(procstat,"/proc/%lu/status",(unsigned long)getpid());
    file = fopen(procstat,"r");
    if (file)
    {
      while (fgets(procstat,sizeof(procstat),file))
      {
        procstat[sizeof(procstat)-1]='\0';
	if (memcmp(procstat,"Name:\t",6)==0)
	{
	  struct stat st; char *p;
	  unsigned int i, n = strlen(procstat);
	  if (n == sizeof(procstat))
	    break;
	  procstat[n-1] = '\0';
	  if (!getcwd(fullpath,sizeof(fullpath)))
	    break;
	  i = strlen(fullpath);    
	  if ((i+n)>(sizeof(fullpath)-2))
	    break;
	  if (i>0 && fullpath[i-1]!='/')
	    fullpath[i++]='/';
	  strcpy(&fullpath[i],&procstat[6]);
	  if (stat(fullpath,&st)!=0)
	    break;
          sprintf(procstat,"/proc/%lu/exe",(unsigned long)getpid());
	  i = readlink(procstat,procstat,sizeof(procstat));
	  if (((int)i) < 0 || i == sizeof(procstat))
	    break;
	  procstat[i] = '\0';    
	  p = strchr(procstat,':');
	  if (p)
	  {
	    dev_t dev = 0; n = 1;
	    if (((ino_t)atol(p+1)) != st.st_ino)
	      break;
	    if (st.st_dev == 0 || procstat[0]!='[')
	      app = fullpath;
	    while (!app)
	    {
	      if (procstat[n] >= '0' && procstat[n] <= '9')
	        dev = (dev*16)+(procstat[n]-'0');
	      else if (procstat[n] >= 'A' && procstat[n] <= 'F')
	        dev = (dev*16)+(procstat[n]-'A')+10;
	      else if (procstat[n] >= 'a' && procstat[n] <= 'f')
	        dev = (dev*16)+(procstat[n]-'a')+10;
	      else if (procstat[n] == ']' && st.st_dev == dev)
	        app = fullpath;
	      else	
	        break;	
	      n++;	
	    }  	
	  }                       
          break;
        } /* if (memcmp(procstat,"Name:\t",6)==0) */
      } /* while (fgets(procstat,sizeof(procstat),file)) */
      fclose(file);
    } /* if (file) */
  } /* if (!app) */
  if (app)
  {
    if ((strlen(app)+1) > bufsize)
      return -1;
    return strlen(strcpy(buffer, app));
  }
  buffer[0] = '\0';
  return 0;
}  

static int create_init_d_script(const char *basename,const char *script_fullpath,
            const char *appname,int argc, const char *argv[])
{
  int i, n; FILE *file;
  int got_ini_opt = 0;
  int (*my_fchmode)(int,int);

  file = fopen(script_fullpath,"w");
  if (!file)
    return -1;
    
  fprintf(file,
  "#! /bin/sh\n"
  "#\n"
  "# distributed.net client startup/shutdown script generated by '%s -install'\n"
  "# Use '%s -uninstall' to stop the client from being started automatically.\n"
  "#\n"
  "# Don't forget to change buffer/.ini file perms if you wish to run suid'd.\n"
  "#\n"
  "\n"
  "CLIENT=%s\n"
  "STARTOPTS=\"", basename, basename, appname );
  n = 0;
  for (i=1;i<argc;i++)
  {
    const char *p = argv[i];
    if (*p == '-' && p[1] == '-') p++; 
    if (strcmp(p,"-quiet")==0 || strcmp(p,"-hide")==0)
      continue;
    if (strcmp(p,"-ini")==0)
    {
      if ((++i)==argc)
        break;
      got_ini_opt = 1;
      fprintf(file,"%s-ini",(((++n)==1)?(""):(" ")) );
    }
    if (strchr(argv[i],' ') || strchr(argv[i],'\t'))
      fprintf(file,"%s\'%s\'",(((++n)==1)?(""):(" ")), argv[i]);
    else  
      fprintf(file,"%s%s",(((++n)==1)?(""):(" ")), argv[i]);
  }    
  if (!got_ini_opt)
    fprintf(file,"%s-ini ${CLIENT}.ini",(((++n)==1)?(""):(" ")) );
  fprintf( file,"\"\n"
  "#STARTOPTS will need to be in quotes if it has white space in it\n"
  "\n"
  "test -f $CLIENT || exit 0\n"
  "\n"
  #if 0
  "#set -e\n"
  "\n"
  #endif
  "case \"$1\" in\n"
  "\t*start)\n"
  "\t\t$CLIENT -quiet -shutdown  # only allow one instance to run.\n"
  "\t\t$CLIENT -quiet $STARTOPTS # -quiet is 'mandatory' here.\n"
  "\t\t;;\n"
  "\t*stop)\t# sends SIGTERM to all running clients.\n"
  "\t\t$CLIENT -quiet -shutdown  # remove '-quiet' to see activity.\n"
  "\t\t;;\n"
  "\t*reload)\t# sends SIGHUP to all running clients.\n"
  "\t\t$CLIENT -quiet -restart   # remove '-quiet' to see activity.\n"
  "\t\t;;\n"
  "\t*)\n"
  "\t\techo \"Usage: $0 {[-]start|[-]stop|[-]reload}\"\n"
  "\t\texit 1\n"
  "\t\t;;\n"
  "esac\n"
  "\n"
  "exit 0\n" );
  my_fchmode = (int (*)(int,int))fchmod; /* work around broken .h */
  (*my_fchmode)(fileno(file),0755);
  fclose(file);
  return 0;
}

int linux_install(const char *basename, int argc, const char *argv[], 
                  int quietly)
{
  char appname[MAXPATHLEN];
  char srcpath[64];
  int i, rcd, numfound=0;
  const char *etc_base = get_etc_base();
  if (!etc_base)
  {
    if (!quietly)
      fprintf(stderr,
      "%s:\tUnable to determine where init scripts (/etc[/rc.d]/init.d/* and\n"
      "\t/etc[/rc.d]/rc?.d/*) are located.\n", basename );
    return -1;		     
  }
  i = determine_appname(appname,sizeof(appname));
  if (i < 0 || 1 > (128-5)) /* 128 is max ini path len */
  {
    if (!quietly)
      fprintf(stderr,"%s: Unable to install. Path too long.\n",basename);
    return -1;
  }
  if (i == 0)
  {
    if (!quietly)
      fprintf(stderr,"%s: Unable to obtain canonical path to executable.\n",basename);
    return -1;
  }
  strcat(strcat(strcpy(srcpath,etc_base),"init.d/"),basename);
  if (access(srcpath,0)==0)
    numfound++;
  if (create_init_d_script(basename,srcpath,appname,argc,argv) != 0)
  {
    if (!quietly)
      fprintf(stderr,"%s: Unable to create/write %s: %s\n", 
                          basename, srcpath, strerror(errno));
    linux_uninstall(basename,1);
    return -1;			  
  } 
  strcat(strcpy(srcpath,"../init.d/"),basename);
  for (rcd = 0; rcd <= 6; rcd++)
  {
    char fullpath[64];
    const char *kstype = get_KS_for_runlevel(rcd);
    sprintf(fullpath,"%src%d.d/%s%s",etc_base,rcd,kstype,basename);
    if (access(fullpath,0)==0)
    {
      numfound++;
      unlink(fullpath);
    }  
    if (symlink(srcpath,fullpath)!=0)
    {  
      if (!quietly)
        fprintf(stderr,"%s: Unable to symlink %s->%s: %s\n", basename,
                              fullpath, srcpath, strerror(errno));
      linux_uninstall(basename,1);			      
      return -1;			      
    }			    
  }    
  if (!quietly)
    fprintf(stderr,
    "%s:\tThe client has been sucessfully %sinstalled in %sinit.d\n"
    "\tand will be automatically started on entry to run-level 2/3\n"
    "\t(multi-user mode). It will be stopped automatically on reboot,\n"
    "\tsystem shutdown, halt, or on a switch to single-user mode.\n"
    "\t*** Please ensure that the client is configured ***\n", basename,
    ((numfound==0)?(""):("re-")), etc_base );
  return 0;
}       

#if 0
int main(int argc,char *argv[])
{
  if (argc > 1 && strcmp(argv[1],"-uninstall")==0)
    return linux_uninstall("dnetc",0);
  else if (argc == 1)    
    return linux_install("dnetc", argc, (const char **)argv, 0);
  return 0;
}
#endif
