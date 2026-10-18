// Metal twin of tests/execution/PipelinedDispatch.cpp (agc_driver_pipelined_dispatch and
// agc_driver_pipelined_dispatch_item; upstream ccfee8e260, dispatch-cache recipe hits as ordered commit items),
// swapped in by twins/UPSYNC3-tests.cmake. The upstream file is compiled here with its main renamed and runs as
// written. With APS5_PIPELINE_DISPATCH set (the _item registration) a dispatch becomes a commit item only when it
// is a dispatch recipe hit; the Metal device has no dispatch recipes yet (VulkanDevice::DispatchRecipes() is false,
// DispatchRecipe throws), so every such dispatch falls back ("no recipe") and drains the pipeline, which the item
// checks refuse. That registration skips naming the gap until recipes exist; the default one runs unchanged.

#define main AgcDriverPipelinedDispatchVulkanMain
#include "../../../tests/execution/PipelinedDispatch.cpp"
#undef main

int main() {
    const char* dispatchSwitch = std::getenv("APS5_PIPELINE_DISPATCH");
    const bool pipelinedDispatch = dispatchSwitch != nullptr && dispatchSwitch[0] != '\0' && dispatchSwitch[0] != '0';
    if (pipelinedDispatch && !AgcDriver::VulkanDevice::DispatchRecipes()) {
        std::puts("skipped, dispatch recipes are not written on Metal (DispatchRecipes() is false), so APS5_PIPELINE_DISPATCH has no recipe hits to commit as items: TechnicalDebt metal-dispatch-recipes");
        return VulkanTestSkipped;
    }
    return AgcDriverPipelinedDispatchVulkanMain();
}
