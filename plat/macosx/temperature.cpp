/*
 * Copyright distributed.net 2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * int macosx_cputemp();
 *
 * Returns the CPUs temperature in Kelvin (in fixed-point format, with two digits
 * after the decimal point), else -1.
 * For multiple cpus, gets one with highest temperature.
 * A test main() is at the end of this file.
 *
 * Currently we try 3 different sources:
 *   - host_processor_info(), which uses CPUs built-in Thermal Assist Unit
 *   - AppleCPUThermo, an IOKit Object that provides "temp-monitor" from a
 *     dedicated sensor near the CPU (PowerMac MDD and XServe only)
 *   - IOHWSensor, an IOKit Object that provides "temp-sensor" data from a
 *     dedicated sensor
 *
 * NOTES: Apple uses various sensors in its models
 *
 *  Computer     | Chipset                | Accuracy | Resolution
 *  -------------+------------------------+----------+-----------
 *  PowerPC G3   | Built-in TAU           | +/- 16.0 |     4
 *  PowerMac MDD | Dallas DS1775          |  +/- 2.0 |     0.0625
 *  XServe       | Dallas DS1775 (?)      |          |
 *  PowerMac G5  | Analog Devices AD7417  |  +/- 1.0 |     0.25
 *  AluBook 15"  | Analog Devices ADT7460 |  +/- 1.0 |   N/A
 *  AluBook 17"  | Analog Devices ADM1031 |  +/- 1.0 |     0.25
 *
 * FIXES:
 *   - #3338 : kIOMasterPortDefault doesn't exist prior Mac OS 10.2 (2.9006.485)
 *   - #3343 : The object filled by CFNumberGetValue shall not be released (2.9006.485)
 *
 *  $Id: temperature.cpp,v 1.2 2007/10/22 16:48:31 jlawson Exp $
 */

#include <string.h>
#include <stdio.h>

#include <mach/mach.h>
#include <mach/mach_error.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CFNumber.h>
#include "logstuff.h"
#include "cpucheck.h"

SInt32 macosx_cputemp(void);

typedef struct {
  const char *location;   /* Location string ("CPU TOPSIDE", etc) */
  SInt32      divisor;    /* Scale */
} SensorInfos;


static SInt32 _readTAU(void) {

    SInt32                 cputemp = -1;
    kern_return_t          ret;
    natural_t              processorCount;
    processor_info_array_t processorInfoList;
    mach_msg_type_number_t processorInfoCount;

    // pass a message to the kernel that we need some info
    ret = host_processor_info( mach_host_self(), // get info from this host
                               PROCESSOR_TEMPERATURE, // want temperature
                               &processorCount,	// get processor count
                               &processorInfoList,
                               &processorInfoCount);

    if (ret==KERN_SUCCESS) {
        // get temperature for 1st processor, -1 on failure
        cputemp = ((int*)processorInfoList)[0];
        if (vm_deallocate(mach_task_self(),
                          (vm_address_t)processorInfoList,
                          processorInfoCount)!=KERN_SUCCESS) {
            //deallocation failed?
            cputemp = -1;
        }
        
        if (cputemp!=-1) {
            cputemp = cputemp * 100 + 27315 /*273.15*/; /* C -> K */
        }
    }

    return cputemp;
}


/*
** className := IO Class name (AppleCPUThermo / IOHWSensor)
** sensors   := Sensor(s) informations. If NULL, the function matches any
**              location and the temperature returned is converted to the
**              fixed point format but not scaled.
** dataKey   := Key that provides the raw temperature value.
** Returns the largest raw temperature value.
*/

static SInt32 _readTemperature(const char *className, SensorInfos *sensors, CFStringRef dataKey) 
{
  SInt32 temp, divisor = 1, temperature = -1;
  char strbuf[64];
  CFNumberRef number = NULL;
  CFStringRef string = NULL;
  io_object_t handle;
  io_iterator_t objectIterator;
  CFMutableDictionaryRef properties = NULL;
  mach_port_t master_port;

  /* Bug #3338 : kIOMasterPortDefault doesn't exist prior Mac OS 10.2,
  ** so we obtain the default port the old way */
  
  if (kIOReturnSuccess == IOMasterPort(MACH_PORT_NULL, &master_port)) {
    if (kIOReturnSuccess == IOServiceGetMatchingServices(master_port /*kIOMasterPortDefault*/, 
                                    IOServiceNameMatching(className), 
                                    &objectIterator)) {
		
      while ( (handle = IOIteratorNext(objectIterator)) ) {
        int matched = 0;

        if (kIOReturnSuccess == IORegistryEntryCreateCFProperties(handle, &properties, 
                                      kCFAllocatorDefault, kNilOptions)) {
				
          /*
          ** Check whether the "location" key matches the string pointed to by location.
          ** If the IO Class doesn't have a "location" key, location must be NULL.
          */
          if(sensors == NULL) {
            matched = 1;      /* match all instances */
          }
          else if (CFDictionaryGetValueIfPresent(properties, CFSTR("location"), (const void **) &string)) {
            if (CFStringGetTypeID() == CFGetTypeID(string) 
                  && CFStringGetCString(string, strbuf, sizeof(strbuf), CFStringGetSystemEncoding())) {
              SensorInfos *p;
              for (p = sensors; p->location != NULL; p++) {
                if (strcmp(strbuf, p->location) == 0) {
                  matched = 1;
                  divisor = p->divisor;   /* Scale */
                  break;
                }
              }
            }
          }
				
          /*
          ** Obtain raw temperature data. When multiple instances are matched, we remember the
          ** largest temperature value.
          */
          if(matched && CFDictionaryGetValueIfPresent(properties, dataKey, (const void **) &number)) {
            if (CFNumberGetTypeID() == CFGetTypeID(number) 
                      && CFNumberGetValue(number, kCFNumberSInt32Type, &temp)) {
						
              temp = (temp * 100) / divisor;
              if (temp > temperature)	
                temperature = temp;
            }
          }

          CFRelease(properties);
        }	/* if IORegistryEntryCreateCFProperties() */
        IOObjectRelease(handle);
      }	/* while */
      IOObjectRelease(objectIterator);
    }	/* if IOServiceGetMatchingServices() */
    mach_port_deallocate(mach_task_self(), master_port);
  }
	
  return temperature;         /* -1 := Error / No sensor */
}


/*
** PowerMac MDD and XServe
*/
static SInt32 _readAppleCPUThermo(void)
{
    SInt32 rawT;
    SInt32 k = -1;
    
    if ( (rawT = _readTemperature("AppleCPUThermo", NULL, CFSTR("temperature"))) >= 0)
      k = rawT / 256 + 27315;
    
    return k;    /* Temperature (* 100, in kelvin) or -1 (error / no sensor) */
}


/*
** PowerBook Alu (12", 15" and 17"), PowerMac G5 and iMacG5
*/

static SensorInfos _sensors[] = {
  {"CPU/INTREPID BOTTOMSIDE", 65536},     // AluBook 15"
  {"CPU BOTTOMSIDE",          65536},     // AluBook 15" and 17"
  {"CPU TOPSIDE",             65536},     // AluBook 12"
  {"CPU A AD7417 AD1",           10},     // PowerMac G5
  {"CPU B AD7417 AD1",           10},     // PowerMac G5 (2nd CPU)
  {"CPU T-Diode",                10},     // iMac G5
  {NULL,                          1}      // LAST ENTRY
};

static SInt32 _readIOHWSensor(void)
{
    SInt32 rawT;
    SInt32 k = -1;
    
    if ( (rawT = _readTemperature("IOHWSensor", _sensors, CFSTR("current-value"))) >= 0)
      k = rawT + 27315;
    
    return k;    /* Temperature (* 100, in kelvin) or -1 (error / no sensor) */
}


SInt32 macosx_cputemp(void) {
    static int source = -1;                 /* No source defined */
    SInt32 temp = -1;
    long CPUid = 0;
    
    switch (source) {
        case 0:
            break;
        case 1:
            return _readTAU();              /* G3 */
        case 2:
            return _readAppleCPUThermo();   /* PowerMac MDD, XServe */
        case 3:
            return _readIOHWSensor();       /* PowerBook Alu, PowerMac G5*/
            
        default:
            CPUid = GetProcessorType(-1);
            if ((GetProcessorFeatureFlags() & CPU_F_ALTIVEC) == 0
                || CPUid == 0x000C || CPUid == 0x800C) {
                /*
                ** Don't read the TAU if AltiVec units are detected
                ** (CPU = G4+ or G5) since this unit is disabled and we'd
                ** get random values.
                */
                temp = _readTAU();
                if (temp >= 0) {source = 1; break;}
            }

            temp = _readAppleCPUThermo();
            if (temp >= 0) {source = 2; break;}
            temp = _readIOHWSensor();
            if (temp >= 0) {source = 3; break;}
    }

    if (source > 0) {
      float k = temp / 100.0;
      Log("Current CPU temperature : %5.2fK (%2.2fC)\n", k, k-273.15);
    }
    else {
      Log("Temperature monitoring disabled (no sensor found)\n");
      source = 0;
    }

    return temp;
}


#if 0
int main(int argc,char *argv[])
{
    printf("TAU %d Kelvin %d Celsius\n",_readTAU(),_readTAU()-27315);
    printf("AppleCPUThermo: %d/100 Kelvin (%d/100 Celsius)\n",_readAppleCPUThermo(),_readAppleCPUThermo()-27315);
    printf("IOHWSensor: %d/100 Kelvin (%d/100 Celsius)\n",_readIOHWSensor(),_readIOHWSensor()-27315);

    while (1) { // test for leaks
        printf("Temp %d/100\n",macosx_cputemp());
    }
    
    return 0;
}
#endif
