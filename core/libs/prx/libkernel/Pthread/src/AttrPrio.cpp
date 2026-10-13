#include "prx/libkernel/Pthread/include/Pthread.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/KernelErrors.hpp"

extern "C" {

int APS5_VABI scePthreadAttrSetschedparam(PthreadAttr* attr, const KernelSchedParam* param);

int APS5_VABI scePthreadAttrSetprio(PthreadAttr* attr, int priority) {
    KernelSchedParam param{};
    param.sched_priority = priority;
    return scePthreadAttrSetschedparam(attr, &param);
}

}
