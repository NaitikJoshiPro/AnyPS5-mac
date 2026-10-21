// Metal twin of tests/execution/PipelinedDispatch.cpp (agc_driver_pipelined_dispatch and
// agc_driver_pipelined_dispatch_item; upstream ccfee8e260, dispatch-cache recipe hits as ordered commit items),
// swapped in by twins/UPSYNC3-tests.cmake. The upstream file is compiled here with its main renamed and runs as
// written. With APS5_PIPELINE_DISPATCH set (the _item registration) a dispatch becomes a commit item only when its
// dispatch-cache variant has a recipe: the Metal device attaches one to every recorded direct dispatch (it binds by
// guest address, so nothing needs completion; MetalDevice.mm Dispatch, TechnicalDebt metal-dispatch-recipes).
// APS5_NO_DISPATCH_RECIPE turns recipes off, and then the item registration has nothing to commit and skips.

#define main AgcDriverPipelinedDispatchVulkanMain
#include "../../../tests/execution/PipelinedDispatch.cpp"
#undef main

int main() {
    const char* dispatchSwitch = std::getenv("APS5_PIPELINE_DISPATCH");
    const bool pipelinedDispatch = dispatchSwitch != nullptr && dispatchSwitch[0] != '\0' && dispatchSwitch[0] != '0';
    if (pipelinedDispatch && !AgcDriver::VulkanDevice::DispatchRecipes()) {
        std::puts("skipped, dispatch recipes are off (APS5_NO_DISPATCH_RECIPE), so APS5_PIPELINE_DISPATCH has no recipe hits to commit as items");
        return VulkanTestSkipped;
    }
    return AgcDriverPipelinedDispatchVulkanMain();
}
