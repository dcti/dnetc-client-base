#!/bin/sh
# shell script versions of gethostbyname() and gethostbyaddr()
#
# Created July 27 2000, by Cyrus Patel <cyp@fb14.uni-mainz.de>
# $Id: resolv.sh,v 1.1.2.1 2001/01/21 15:10:24 cyp Exp $
#
# this script does these things: 
#   - determine if called to process a hostname or hostaddress
#   - parse /etc/hosts (optionally considering domain name)
#   - process 'builtins' (localhost etc)
#   - parse/process the equivalent of nslookup -t=<A|PTR> <name|addr>
# prints results in the same format as newer nslookup
# hex/octal/.in-addr.arpa forms of addresses are supported
# depends on grep, tr, cut, bc, (expr), nslookup

# trace=1   # <- show what its doing to file trace.out
trace_out() {
  if [ "${trace}X" != "X" ]; then
    if [ "${trace}" != "0" ]; then
      if [ "${trace}X" != "." ]; then
        if [ -f trace.out ]; then
	  echo "" > trace.out
	  trace="." 
	fi
        echo "$1">> trace.out
      fi 	 
    fi    
  fi
}

# ---------------------------------------------------------

inet_ntoa=""
inet_ntoa() {     # returns "" if invalid, else 4 component IP address
  inet_ntoa=""
  if [ "${1}X" != "X" ]; then
    source_string=`echo "$1"|tr -dc 0-9`
    if [ "${1}X" = "${source_string}X" ]; then
      comp_list="1 2 3 4"
      for comp in $comp_list; do
        comp=`echo "${source_string}%256"|bc`
	#comp=`expr ${source_string} % 256`
	source_string=`echo "${source_string}/256"|bc|cut -d. -f1`
        #source_string=`expr ${source_string} / 256`
        if [ "${inet_ntoa}X" != "X" ]; then
	  inet_ntoa="${inet_ntoa}.$comp";
	else
	  inet_ntoa="$comp"
	fi
      done
      if [ "${source_string}X" != "0X" ]; then
        inet_ntoa=""
      fi
    fi      	
  fi	
}

# ---------------------------------------------------------

inet_aton=""
inet_aton() {   # returns "" if invalid, else long number
  inet_aton=""  # (an address with less than 4 components is considered bad)

  #comp_numlist=`echo "$1"|tr -dc .`   
  comp_numlist=`echo "$1"|tr -d 0-9A-Fa-fxX`   
  if [ "${comp_numlist}X" = "...X" ]; then
    comp_numlist="4 3 2 1"
    inet_aton=0
    for comp_num in $comp_numlist; do
      if [ "${inet_aton}X" != "X" ]; then
        digit_list=`echo "$1"|cut -d. -f${comp_num}`
        base=10
        startpos=1
        dig=`echo "$digit_list"|cut -c 1`

        longval=""
        if [ "${dig}X" = "0X" ]; then
          base=8
          startpos=2
	  longval=0
        fi    
        dig=`echo "$digit_list"|cut -c $startpos`
        if [ "${dig}X" = "XX" -o "${dig}X" = "xX" ]; then
          base=16
          if [ $startpos -eq 1 ]; then
            startpos=2
          else
            startpos=3
          fi  
        fi    
        if [ $startpos -gt 1 ]; then
          digit_list=`echo "$digit_list"|cut -c ${startpos}-`
        fi    
	if [ "${digit_list}X" != "X" ]; then #not all done
	  longval=0
          if [ $base -eq 10 ]; then
            dig2=`echo "$digit_list"|tr -dc 0-9`
  	    if [ "${dig2}X" = "${digit_list}X" ]; then
              longval=$digit_list
  	    else
	      longval=""
	    fi
	  else # non-base 10
	    while [ "${digit_list}X" != "X" ]; do
              dig=`echo "$digit_list"|cut -c1`
	      digit_list=`echo "$digit_list"|cut -c2-`
	      if [ "${longval}X" != "X" ]; then
                dig2=`echo "$dig"|tr -dc 0-9`
                if [ "${dig2}X" != "X" ]; then # decimal digit
                  if [ $base -eq 8 ]; then
		    if [ $dig2 -gt 7 ]; then     
		      dig2=""
	              longval=""
	              break  
		    fi
		  fi      
                else # non-decimal digit
                  dig2=`echo "$dig"|tr -dc \[:xdigit:\]`
                  if [ "${dig2}X" != "X" ]; then
  	            dig2=`echo "$dig"|tr [:lower:] [:upper:]|tr A-F 0-5`
  	            dig2="1${dig2}"
	          else
	            dig2=""
	            longval=""
	            break  
	          fi  
                fi
                if [ "${dig2}X" != "X" ]; then
                  if [ $longval -ne 0 ]; then
                    #longval=`echo "(${longval}*${base})+${dig2}"|bc`
		    longval=`expr \( ${longval} \* ${base} \) + ${dig2}`
                  else
                    longval=$dig2
	          fi  
		  if [ $longval -gt 255 ]; then
		    longval=""
		    break
		  fi    
                fi  	
	      fi # "${longval}X" != "X"
            done # while [ "${digit_list}X" != "X" ]; do
	  fi # not base 10    
        fi # if not all done
        if [ "${longval}X" = "X" ]; then #something invalid
          inet_aton=""
	  break
        else
          if [ $inet_aton -eq 0 ]; then
	    inet_aton=$longval
	  else  # the order has already been reversed
            inet_aton=`echo "(${inet_aton}*256)+${longval}"|bc`
	    #inet_aton=`expr \( ${inet_aton} \* 256 \) + ${longval}`
          fi  
        fi
      fi  # if "${inet_aton}X" != "X" 
    done # for component in $comp_list; do  
  fi # have 4 components
}

# ---------------------------------------------------------

resolv_conf_domainname=""    # not yet parsed
parse_resolv_conf() {
  if [ "${resolv_conf_domainname}X" = "X" ]; then
    resolv_conf_domainname=""
    if [ -f /etc/resolv.conf ]; then
      lines=`cat /etc/resolv.conf|tr -s '\t' ' '|tr -s ' ' '^'|tr -s '\n' ' '`
      for line in $lines; do
        if [ "${resolv_conf_domainname}X" = "X" ]; then
          line=`echo "$line"|cut -d'#' -f1` #strip comments
          if [ "${line}X" != "X" ]; then
            resolv_conf_domainname=`echo "$line"|cut -d'^' -f1`
	    if [ "${resolv_conf_domainname}X" = "domainX" ]; then
	      resolv_conf_domainname=`echo "$line"|cut -d'^' -f2`
	      lines=""
	      break
	    else  
	      resolv_conf_domainname=""
	    fi
	  fi
	fi
      done	        
    fi  # -f /etc/resolv.conf
    if [ "${resolv_conf_domainname}X" = "X" ]; then
      resolv_conf_domainname=`echo "$HOSTNAME"|cut -d. -f2-`
    fi	
    if [ "${resolv_conf_domainname}X" = "X" ]; then
      resolv_conf_domainname="."
    fi	      
  fi  # "${resolv_conf_domainname}X" = "X"
}  

# ---------------------------------------------------------

compare_hostnames=""
compare_hostnames() { # compare two hostnames (while taking domainname into 
                      # consideration. See note below) If not matched, 
		      # return "", else return "most appropriate form"
  do_non_compatible=0 # <- see note below for this
  
  compare_hostnames=""
  host1=`echo "$1"|tr [:upper:] [:lower:]`
  host2=`echo "$2"|tr [:upper:] [:lower:]`
  if [ "${host1}X" = "${host2}X" ]; then
    compare_hostnames="$2"
  else
    # note that gethostent() does not do anything with the domainname 
    # [and consequently, neither does gethostby[name|addr]() when they
    # checks /etc/hosts]. We do some basic matching here to avoid the
    # call to nslookup.
    if [ $do_non_compatible -ne 0 ]; then 
      parse_resolv_conf
      domname=`echo "$resolv_conf_domainname"|tr [:upper:] [:lower:]`
      if [ "${domname}X" = ".X" ]; then
        domname=""
      fi  
      if [ "${domname}X" != "X" ]; then
        #numparts1=`echo "$host1"|tr '.' '\n'|grep -c \"\"`
        #numparts2=`echo "$host2"|tr '.' '\n'|grep -c \"\"`
        numparts1=0
        comp_list=`echo "${host1}X"|tr '.' ' '`
        for comp in $comp_list; do
          numparts1=`expr $numparts1 + 1`
        done	
        iscanon1=0
        if [ $numparts1 -gt 1 ]; then
          comp=`echo "${host1}X"|cut -d. -f$numparts1`
          if [ "${comp}X" = "XX" ]; then
            iscanon1=1
	  fi  
        fi	
        numparts2=0
        comp_list=`echo "{$host2}X"|tr '.' ' '`
        for comp in $comp_list; do
          numparts2=`expr $numparts2 + 1`
        done	
        iscanon2=0
        if [ $numparts2 -gt 1 ]; then
          comp=`echo "${host2}X"|cut -d. -f$numparts2`
          if [ "${comp}X" = "XX" ]; then
            iscanon2=1
          fi	
        fi
        if [ $iscanon1 -eq 1 -a $iscanon2 -eq 1 ]; then
          numparts1=$numparts2
        fi	
        if [ $numparts1 -ne $numparts2 ]; then
          compare_hostnames=""
	  numparts=0
          if [ $numparts1 -lt $numparts2 ]; then
	    if [ $iscanon1 -eq 0 ]; then
	      compare_hostnames="$2"
              numparts=`expr $numparts2 - $numparts1`
	      needed_parts=`echo "$domname"|cut -d. -f1-$numparts`
	      host1="$host1.$needed_parts"
	    fi    
	  else
	    if [ $iscanon2 -eq 0 ]; then
	      compare_hostnames="$1"
              numparts=`expr $numparts1 - $numparts2`
	      needed_parts=`echo "$domname"|cut -d. -f1-$numparts`
	      host2="$host2.$needed_parts"
	    fi    
	  fi
          if [ "${host1}X" != "${host2}X" ]; then
	    compare_hostnames=""
	  else  
	    numparts=`expr $numparts + 1`
	    needed_parts=`echo "$resolv_conf_domainname"|cut -d. -f${numparts}-`
	    if [ "${needed_parts}X" != "X" ]; then
              compare_hostnames="$compare_hostnames.$needed_parts"
	    fi    
          fi
	fi  
      fi	
    fi  
  fi    
}

# ******************************************************

h_name=""
h_addr_list=""
h_aliases=""

do_etc_hosts() { 
  trace_out "do_etc_hosts: ltype=$1, hostname=$2, numaddr=$3"
  h_name=""
  h_addr_list=""
  h_aliases=""
  ltype="$1"         # $1="PTR" or "A"
  host="$2"          # $2=hostname or stringified address
  numeric_address=$3 # $3=numeric address (if $1 is "PTR")

  if [ "${host}X" != "X" ]; then
    if [ -f /etc/hosts ]; then
      lines=`cat /etc/hosts|tr -s '\t' ' '|tr -s ' ' '^'|tr -s '\n' ' '`
      for line in $lines; do
        line=`echo "$line"|cut -d'#' -f1` #strip comments        
        if [ "${line}X" != "X" ]; then
          addr=`echo "$line"|cut -d'^' -f1`
	  aliases=`echo "$line"|cut -d'^' -f2-|tr '^' ' '`
	  if [ "${aliases}X" != "X" ]; then
	    found=0
	    name=""
	    if [ "$ltype" = "PTR" ]; then
	      if [ "${addr}X" = "${host}X" ]; then
	        found=1
	      else
	        inet_aton "$addr"
		if [ "${inet_aton}X" = "${numeric_address}X" ]; then
		  addr="$host" # we want the decimal form
	      	  found=1
		fi
	      fi	  
	      if [ $found -eq 1 ]; then
		name=`echo "$aliases"|cut -d' ' -f1`
		aliases=`echo "$aliases"|cut -d' ' -f2-`
	      fi	  
	    else
	      otheraliases=""
	      for alias in $aliases; do
		if [ "${name}X" = "X" ]; then
		  name="$alias"
		fi    
		xfound=0
	        if [ $found -eq 0 ]; then
	          compare_hostnames $host $alias                 
                  if [ "${compare_hostnames}X" != "X" ]; then
    		    name="$compare_hostnames"
		    found=1
		    xfound=1
		  fi
		fi  
		if [ $xfound -eq 0 ]; then
		  if [ "${otheraliases}X" != "X" ]; then
		    otheraliases="${otheraliases} ${alias}"
		  else
		    otheraliases="${alias}"
		  fi
		fi      
	      done
	      aliases="$otheraliases"
	    fi    # if ltype = PTR|A	
	    if [ $found -eq 1 ]; then	  
 	      h_addr_list="$addr"
              h_name="$name"
	      h_aliases="$aliases"
	      lines=""
	      break
	    fi
	  fi    # if $aliases != ""
	fi    # if $line != ""  
      done  # for lines
    fi  # if -f /etc/hosts
  fi  # if $host != ""
}		  

# ---------------------------------------------------------

do_builtins() {
  trace_out "do_builtins: ltype=$1, hostname=$2, numaddr=$3"
  ltype="$1"         # $1="PTR" or "A"
  host="$2"          # $2=hostname or stringified address
  numeric_address=$3 # $3=numeric address (if $1 is "PTR")
  h_name=""
  h_addr_list=""
  h_aliases=""
  if [ "$ltype" = "A" ]; then
    host=`echo "$host"|tr [:upper:] [:lower:]`
  fi    

  # our builtins table (decimal addresses, lower case hostnames)
  # lines begin with a '^', tokens are comma-separated, space is ignored
  builtins="^127.0.0.1,      localhost,  lb \
            ^127.0.0.1,      localhost,  lb"

  lines=`echo "$builtins"|tr '\t' ' '|tr -d ' '|tr '^' ' '`
  for line in $lines; do
    if [ "${line}X" != "X" ]; then
      addr=`echo "$line"|cut -d, -f1`
      aliases=`echo "$line"|cut -d, -f2-|tr ',' ' '`
      if [ "${aliases}X" != "X" ]; then
        found=0
	name=""
        if [ "$ltype" = "PTR" ]; then
          if [ "${addr}X" = "${host}X" ]; then
	    name=`echo "$aliases"|cut -d ' ' -f1` 
	    aliases=`echo "$aliases"|cut -d ' ' -f2-` 
	    found=1
	  fi
	else
          otheraliases=""
	  for alias in $aliases; do
	    if [ "${name}X" = "X" ]; then
	       name="$alias"
	    fi    
	    xfound=0
	    if [ $found -eq 0 ]; then
	      if [ "${alias}X" = "${host}X" ]; then
	        found=1
		xfound=1
	      fi
	    fi
	    if [ $xfound -eq 0 ]; then
	      if [ "${otheraliases}X" != "X" ]; then
	        otheraliases="${otheraliases} ${alias}"
	      else
	        otheraliases="${alias}"
	      fi	
	    fi
	  done
	  aliases="$otheraliases"
	fi  
	if [ $found -eq 1 ]; then	  
          h_name="$name"
 	  h_addr_list="$addr"
	  h_aliases="$aliases"
	  lines=""
	  break
	fi
      fi  # if aliases != ""
    fi  # if line != ""
  done # for line in $lines
}        

# ---------------------------------------------------------

do_nslookup() {
  trace_out "do_nslookup: ltype=$1, hostname=$2, numaddr=$3"
  ltype="$1"         # $1="PTR" or "A"
  host="$2"          # $2=hostname or stringified address
  numeric_address=$3 # $3=numeric address (if $1 is "PTR")
  h_name=""
  h_addr_list=""
  h_aliases=""

  if [ "${host}X" != "X" ]; then
    #do it the interactive way for backwards compatibility
    lookup=`echo "set type=${ltype}#$host#quit"|tr '#' '\n'|nslookup 2>/dev/null`
    #lookup=`cat out`
    #echo -n "$lookup">out

    if [ "${lookup}X" != "X" ]; then
      found=0
      if [ "$ltype" = "PTR" ]; then
        line=`echo "$lookup"|tr '\t' '#'|grep ".in-addr.arpa#name = "|head -n 1`
        if [ "${line}X" != "X" ]; then                  # new style "PTR"
	  h_name=`echo "$line"|tr -d ' '|cut -d'=' -f2`
	  h_addr_list="$host"
	  found=1
        fi	
      else                                              # old style "A"
        lines=`echo "$lookup"|tr '\t' '#'|tr -s ' ' '^'|tr '\n' ' '`
        if [ "${lines}X" != "X" ]; then
	  for xline in $lines; do
	    gotalias=0
	    line=`echo "$xline"|grep "#A#"`
	    if [ "${line}X" = "X" ]; then
 	      if [ "${h_addr_list}X" = "X" ]; then
  	        line=`echo "$xline"|grep "#CNAME#"`
		if [ "${line}X" != "X" ]; then
		  gotalias=1
		fi  
	      fi
	      if [ "${line}X" = "X" ]; then
	        lines=""
	        break
	      fi  	
	    fi  	
	    if [ "${line}X" != "X" ]; then  
	      if [ "${h_name}X" = "X" ]; then
	        fields=`echo "$line"|cut -d'#' -f1|tr '>' ' '|tr '^' ' '`
		for field in $fields; do
		  if [ "${field}X" != "X" ]; then
		    if [ "${field}" != ">" ]; then
		      h_name="$field"
		      fields=""
		      break
		    fi
		  fi
		done
	      fi	        
	      if [ "${h_name}X" = "X" ]; then
	        lines=""
	        break
	      else
	        if [ $gotalias -eq 1 ]; then
 		  h_aliases="$h_name"
		  h_name=""
		else
	          addr=`echo "$line"|cut -d'#' -f3`
	          if [ "${h_addr_list}X" = "X" ]; then
	            h_addr_list="$addr"
	          else
	       	    h_addr_list="$h_addr_list $addr"
		  fi    
	          found=1
	        fi	
	      fi      
	    fi  
	  done    
        fi
      fi  
      if [ $found -eq 0 ]; then          # new style A, old style PTR
        lines=`echo "$lookup"|grep -A 10 "Name:"`
	if [ "${lines}X" != "X" ]; then
	  name=`echo "$lines"|head -n 1|cut -d':' -f2|tr -d ' '`
	  if [ "${name}X" != "X" ]; then
            if [ "$ltype" = "PTR" ]; then    
	      h_name="$name"
	      h_addr_list="$host"
	      found=1
	    else  
	      h_name="$name"
	      h_addr_list=`echo "$lines"|grep "Address"|head -n 1|\
	             cut -d':' -f2|tr -d ' '|tr ',' ' '`
	      h_aliases=`echo "$lines"|grep "Alias"|head -n 1|\
	             cut -d':' -f2|tr -d ' '|tr ',' ' '`
	      found=1
	    fi
	  fi  # if name != ""
	fi  # if found "Name:"    
      fi  # if $found = 0
    fi  # if $lookup != ""  
  fi   # if $host != ""
}

# ******************************************************

hostnameoraddr=$1
if [ "${hostnameoraddr}X" = "X" ]; then
  hostnameoraddr=$1 # dummy line to squelch old bash error
  #------------------- test suite ----------------------
  #hostnameoraddr="ftp.uni-mainz.de"                  # CNAME test
  #hostnameoraddr="us.v27.distributed.net"            # multiple-A
  #hostnameoraddr="fb14.uni-mainz.de"                 # single-A
  #hostnameoraddr="119.246.93.134.in-addr.arpa"       # reverse
  #hostnameoraddr="134.0135.xF6.0x77"                 # (134.93.246.119)
  #hostnameoraddr="0x01.0x00.0x00.0177.in-addr.arpa"  # hex/octal test
  #hostnameoraddr="127.0.0.1"                         # rev /etc/hosts test
  #hostnameoraddr="localhost"                         # /etc/hosts test
fi  

if [ "${hostnameoraddr}X" = "X" ]; then
  echo "Syntax: $0 <hostname|hostaddr>" >&2
else
  ltype="A"
  numeric_address=0

  #
  # determine if we have to do a reverse lookup, 
  # if so, convert and reformat the hostname, and set ltype=PTR
  #
  numeric_address=0
  byaddr=`echo "$hostnameoraddr"|tr -dc .`
  if [ "${byaddr}X" = "...X" ]; then # four parts
    inet_aton $hostnameoraddr
    if [ "${inet_aton}X" != "X" ]; then
      numeric_address="$inet_aton"
    fi
  else
    if [ "${byaddr}X" = ".....X" ]; then # six parts
      byaddr=`echo "$hostnameoraddr"|cut -d. -f5-`
      if [ "${byaddr}X" = "in-addr.arpaX" ]; then
        comp_list=`echo "$hostnameoraddr"|cut -d. -f1-4|tr '.' ' '`
        byaddr=""
        for comp in $comp_list; do
          if [ "${byaddr}X" = "X" ]; then
            byaddr="$comp"
	  else
	    byaddr="${comp}.${byaddr}"
	  fi
        done	  
        inet_aton $byaddr
        if [ "${inet_aton}X" = "X" ]; then
          numeric_address=-1
        else
	  numeric_address="$inet_aton"
        fi	
      fi
    fi #six parts
  fi #four parts    
  if [ $numeric_address -ne 0 ]; then   # !=0 means its an address
    if [ $numeric_address -lt 0 ]; then # < 0 means its a bad addr
      numeric_address=0
      hostnameoraddr=""
    else                                # > 0 means its a valid addr
      inet_ntoa $numeric_address
      hostnameoraddr="$inet_ntoa"
      ltype="PTR"
    fi
  fi

  trace_out "params=>ltype=\"$ltype\",nameoraddr=\"$hostnameoraddr\",numaddr=$numeric_address"
  #
  # call the various subroutines
  #    ltype=A or PTR
  #    hostnameoraddr=hostname or stringified IP address
  #    numeric_address=long number of IP address
  #
  resolv_order_list="/etc/hosts builtins nslookup"
  for resolv_order in $resolv_order_list; do
    if [ "${hostnameoraddr}X" != "X" ]; then
      if [ "${resolv_order}X" = "/etc/hostsX" ]; then
        do_etc_hosts $ltype $hostnameoraddr $numeric_address
      fi
      if [ "${resolv_order}X" = "builtinsX" ]; then
        do_builtins $ltype $hostnameoraddr $numeric_address
      fi
      if [ "${resolv_order}X" = "nslookupX" ]; then
        do_nslookup $ltype $hostnameoraddr $numeric_address
      fi
      if [ "${h_addr_list}X" != "X" ]; then
        hostnameoraddr=""
	break
      fi
    fi
  done
fi
      
#
# print the results
#

if [ "${h_addr_list}X" != "X" ]; then
  echo "Name: $h_name"
  plural=`echo "$h_addr_list"|tr -dc ' '`
  if [ "${plural}X" != "X" ]; then
    plural="es"
  fi	
  echo "Address${plural}: $h_addr_list"
  if [ "${h_aliases}X" != "X" ]; then
    plural=`echo "$h_aliases"|tr -dc ' '`
    if [ "${plural}X" != "X" ]; then
      plural="es"
    fi	
    echo "Alias${plural}: $h_aliases"
  fi  
  exit 0
fi  
exit 1


