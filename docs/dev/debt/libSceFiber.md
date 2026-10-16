# libSceFiber technical debt

### Unknown function info

- [_sceFiberGetThreadFramePointerAddress](../../../core/libs/prx/libSceFiber/Export.cpp) (libSceFiber, NID `0dy4JtMUcMQ`) forwards to `sceFiberGetThreadFramePointerAddress`; that the argument list matches is inferred from the export without the underscore, as for the underscore Ult exports, not measured

### Functional

- [Fiber switches on arm64 macOS](../../../core/libs/prx/libSceFiber/Export.cpp) (libSceFiber) keep each fiber's and the thread's MXCSR and x87 control word, as the x86-64 stack switch does: a resume to a context with other control words goes through the runner's whole-machine-state hooks (`ContinueWithControl`) instead of the register-only resume, which costs a full state copy (fixture `fibers-control`). Without the runner's hooks that case throws. The host stack switch (`Aps5FiberSwitchStack`, `Aps5FiberTrampoline`) exists for x86-64 hosts only and throws elsewhere; arm64 macOS never reaches it
