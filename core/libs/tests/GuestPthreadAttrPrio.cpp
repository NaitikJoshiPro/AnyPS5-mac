#include "SceTypes.hpp"
#include <cstdlib>
#include <initializer_list>

extern "C" {
int APS5_VABI scePthreadAttrInit(PthreadAttr* attr);
int APS5_VABI scePthreadAttrDestroy(PthreadAttr* attr);
int APS5_VABI scePthreadAttrSetprio(PthreadAttr* attr, int priority);
int APS5_VABI scePthreadAttrGetschedparam(const PthreadAttr* attr, KernelSchedParam* param);
}

static constexpr int SCE_OK = 0;
static constexpr int SCE_KERNEL_ERROR_EINVAL = static_cast<int>(0x80020016u);
static constexpr int SCE_KERNEL_ERROR_EOPNOTSUPP = static_cast<int>(0x8002002Du);

static void Require(bool value) { if (!value) std::abort(); }

int main() {
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrSetprio(nullptr, 700) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadAttrSetprio(&attr, 700) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadAttrInit(&attr) == SCE_OK);
    KernelSchedParam read{};
    for (int priority : {-1, 0, 255, 768, 960}) {
        Require(scePthreadAttrSetprio(&attr, priority) == SCE_KERNEL_ERROR_EOPNOTSUPP);
        Require(scePthreadAttrGetschedparam(&attr, &read) == SCE_OK && read.sched_priority == 700);
    }
    for (int priority : {256, 479, 767}) {
        Require(scePthreadAttrSetprio(&attr, priority) == SCE_OK);
        Require(scePthreadAttrGetschedparam(&attr, &read) == SCE_OK && read.sched_priority == priority);
    }
    Require(scePthreadAttrDestroy(&attr) == SCE_OK);
}
