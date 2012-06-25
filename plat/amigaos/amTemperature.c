/*
 * Sensor polling for MorphOS by Harry Sintonen <sintonen@iki.fi>
 * Copyright distributed.net 2012 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * LONG morphos_cputemp(void);
 * int morphos_isrunningonbattery(void);
 *
 * Returns the CPUs temperature in Kelvin (in fixed-point format, with two digits
 * after the decimal point), else -1.
 * For multiple cpus, gets one with highest temperature.
 *
 *  $Id: amTemperature.c,v 1.2 2012/06/25 00:30:00 piru Exp $
 */

#include <stdlib.h>

#include <libraries/sensors.h>
#include <proto/exec.h>
#include <proto/sensors.h>

#include "logstuff.h"

static BOOL sensorsinitialized;
struct Library *SensorsBase;
static APTR batterysensors;
static APTR acpowersensors;
static APTR cputemperaturesensors;

static void cleanupsensors(void)
{
  if (SensorsBase)
  {
    if (batterysensors)
      ReleaseSensorsList(batterysensors, NULL);

    if (acpowersensors)
      ReleaseSensorsList(acpowersensors, NULL);

    if (cputemperaturesensors)
      ReleaseSensorsList(cputemperaturesensors, NULL);

    CloseLibrary(SensorsBase);
  }
}

static int initsensors(void)
{
  if (!sensorsinitialized)
  {
    if (atexit(cleanupsensors) == -1)
      return FALSE;

    SensorsBase = OpenLibrary("sensors.library", 50);

    sensorsinitialized = TRUE;
  }

  if (!SensorsBase)
    return FALSE;

  return TRUE;
}

LONG morphos_cputemp(void)
{
  DOUBLE maxtemperature = -1000.0;

  if (initsensors())
  {
    static const struct TagItem sensortags[] =
    {
      {SENSORS_Type, SensorType_Temperature},
      {SENSORS_SensorPlacement, SensorPlacement_Processor},
      {TAG_DONE, 0}
    };
    DOUBLE temperature = 0.0;
    struct TagItem ttemperature[] =
    {
      {SENSORS_Temperature_Temperature, (IPTR)&temperature},
      {TAG_DONE, 0}
    };
    APTR sensor;

    if (!cputemperaturesensors)
      cputemperaturesensors = ObtainSensorsList((struct TagItem *) sensortags);

    sensor = NULL;
    while ((sensor = NextSensor(sensor, cputemperaturesensors, NULL)) != NULL)
    {
      if (GetSensorAttr(sensor, ttemperature) > 0 &&
          temperature > maxtemperature)
      {
        maxtemperature = temperature;
      }
    }
  }

  if (maxtemperature > -1000.0)
  {
    //Log("Current CPU temperature : %5.2fK (%2.2fC)\n", maxtemperature + 273.15, maxtemperature);
    return (LONG) (maxtemperature * 100.0) + 27315;
  }
  else
  {
    Log("Temperature monitoring disabled (no sensor found)\n");
  }
  return -1;
}

int morphos_isrunningonbattery(void)
{
  int disableme = 2; /* assume further checking should be disabled */
  LONG pluggedin = 0;
  LONG charging = 0;

  if (initsensors())
  {
    static const struct TagItem acpowersensortags[] =
    {
      {SENSORS_Type, SensorType_ACPower},
      {TAG_DONE, 0}
    };
    static const struct TagItem batterysensortags[] =
    {
      {SENSORS_Type, SensorType_Battery},
      {TAG_DONE, 0}
    };
    struct TagItem tpluggedin[] =
    {
      {SENSORS_ACPower_PluggedIn, (IPTR)&pluggedin},
      {TAG_DONE, 0}
    };
    struct TagItem tcharging[] =
    {
      {SENSORS_Battery_Charging, (IPTR)&charging},
      {TAG_DONE, 0}
    };
    APTR sensor;

    /* Check if AC mains power is connected - if not, pause crunching */
    if (!acpowersensors)
      acpowersensors = ObtainSensorsList((struct TagItem *) acpowersensortags);

    sensor = NULL;
    while ((sensor = NextSensor(sensor, acpowersensors, NULL)) != NULL)
    {
      if (GetSensorAttr(sensor, tpluggedin) > 0)
      {
        disableme--;
        break;
      }
    }

    /* Also check if we're currently charging - if so, pause crunching */
    /* Full CPU load while the battery is charging can shorten the lifespan
       of the battery. */
    if (!batterysensors)
      batterysensors = ObtainSensorsList((struct TagItem *) batterysensortags);

    sensor = NULL;
    while ((sensor = NextSensor(sensor, batterysensors, NULL)) != NULL)
    {
      if (GetSensorAttr(sensor, tcharging) > 0)
      {
        disableme--;
        break;
      }
    }
  }

  return disableme ? -1 : (pluggedin && !charging) ? 0 : 1;
}
