# libSceAppInstUtil technical debt

### Unknown function info

- [sceAppInstUtilAppExists, sceAppInstUtilAppInstallPkg, sceAppInstUtilAppUnInstall, sceAppInstUtilGetInstallProgress, sceAppInstUtilGetInstallProgressInfo](../../../core/libs/prx/libSceAppInstUtil/Export.cpp) (libSceAppInstUtil) and [sceGnmSubmitCommandBuffers, sceGnmSubmitDone, sceGnmDrawIndex, sceGnmDispatchDirect, sceGnmDispatchInitDefaultHardwareState](../../../core/libs/prx/libSceGnmDriver/Export.cpp) (libSceGnmDriver) - SuperTuxKart (STKT00001) needs these modules to load but imports none of these functions. The names come from oops-apps `common/symbols.txt`. They have no implementation because their signatures are unknown, and there is no PS4 Gnm front end
