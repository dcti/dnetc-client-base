//Get device ID based on cl_amd_predefined_macros extension (if available)
uint devID()
{
#if defined(__WinterPark__)
  return 5;
#elif defined(__BeaverCreek__)
  return 6;
#elif defined(__Turks__)
  return 7;
#elif defined(__Caicos__)
  return 8;
#elif defined(__Tahiti__)
  return 9;
#elif defined(__Pitcairn__)
  return 10;
#elif defined(__Capeverde__)
  return 11;
#elif defined(__Cayman__)
  return 12;
#elif defined(__Barts__)
  return 13;
#elif defined(__Cypress__)
  return 14;
#elif defined(__Juniper__)
  return 15;
#elif defined(__Redwood__)
  return 16;
#elif defined(__Cedar__)
  return 17;
#elif defined(__ATI_RV770__)
  return 18;
#elif defined(__ATI_RV730__)
  return 19;
#elif defined(__ATI_RV710__)
  return 20;
#elif defined(__Loveland__)
  return 21;
#elif defined(__GPU__)
  return 1;
#elif defined(__X86__)
  return 2;
#elif defined(__X86_64__)
  return 3;
#elif defined(__CPU__)
  return 4;
#endif

  return 0;
}

__kernel void deviceID(__global uint *outbuf)
{
  outbuf[0] = devID();
}