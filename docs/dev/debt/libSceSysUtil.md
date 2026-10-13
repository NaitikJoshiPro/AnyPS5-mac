# libSceSysUtil technical debt

### Unknown function info

- [sceSysUtilSendSystemNotificationWithText](../../../core/libs/prx/libSceSysUtil/Export.cpp) (libSceSysUtil) - `(int type, const char* text)` as SuperTuxKart's oops-sdk calls it (type 0). The text goes to stderr as a log line with no host notification. The console's types and return values are unknown, and a null text throws
