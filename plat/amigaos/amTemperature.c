/*
 * CPU Temperature polling for MorphOS by Harry Sintonen <sintonen@iki.fi>
 * Copyright distributed.net 2012 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * LONG morphos_cputemp(void);
 *
 * Returns the CPUs temperature in Kelvin (in fixed-point format, with two digits
 * after the decimal point), else -1.
 * For multiple cpus, gets one with highest temperature.
 *
 *  $Id: amTemperature.c,v 1.1 2012/06/24 18:26:57 piru Exp $
 */

#include <libraries/sensors.h>
#include <proto/exec.h>
#include <proto/sensors.h>

#include "logstuff.h"

LONG morphos_cputemp(void)
{
  DOUBLE maxtemperature = -1000.0;
  struct Library *SensorsBase;

  SensorsBase = OpenLibrary("sensors.library", 50);
  if (SensorsBase)
  {
    struct TagItem sensortags[] =
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
    APTR sensors, sensor;

    sensor = NULL;
    sensors = ObtainSensorsList(sensortags);

    while ((sensor = NextSensor(sensor, sensors, NULL)) != NULL)
    {
      if (GetSensorAttr(sensor, ttemperature) > 0 &&
          temperature > maxtemperature)
      {
        maxtemperature = temperature;
      }
    }
    ReleaseSensorsList(sensors, NULL);

    CloseLibrary(SensorsBase);
  }

  if (maxtemperature > -1000.0)
  {
    Log("Current CPU temperature : %5.2fK (%2.2fC)\n", maxtemperature + 273.15, maxtemperature);
    return (LONG) (maxtemperature * 100.0) + 27315;
  }
  else
  {
    Log("Temperature monitoring disabled (no sensor found)\n");
  }
  return -1;
}
