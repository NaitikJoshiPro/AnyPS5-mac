# libSceContentExport technical debt

### Functional

- [Content export sessions](../../../core/libs/prx/libSceContentExport/Export.cpp): the host has no media library, so `sceContentExportFromDataWithThumbnail` throws for a valid session without invoking data callbacks. Session allocation does not reproduce guest allocator calls, worker creation or their failure paths. `sceContentExportGetProgress` throws because an idle session returns either zero or an invalid-state error according to the caller's SDK compatibility mode, which is not available here. Initialization and session lookup still throw for some invalid inputs instead of reproducing the console's error precedence and legacy SDK behavior. Session IDs now use the lowest free slot in 0–9; exhaustion, termination with live sessions and cancellation before an export return the confirmed errors.
- [sceContentExportFromData](../../../core/libs/prx/libSceContentExport/Export.cpp) (libSceContentExport): signature `int32_t(int32_t expId, const SceContentExportParam* param, size_t contentLength, provide function, void* userdata, char* path, size_t pathBufferLength)` from Orbital (as above). Like `sceContentExportFromDataWithThumbnail` it throws after the session check, before the data callback runs and before `path` is written. Writing the exported file into a host folder under the title's data directory was considered and not done: it needs the guest's data-provide callback called from the library, and no console error code for "export not available" or "cancelled" is documented anywhere we could find (searched 2026-10-14), so returning one would be invented. ASTRO BOT (PPSA21564) imports it

### Unknown function info

- [sceContentExportDataProvideFunction, sceContentExportFree, sceContentExportFrom and sceContentExportMalloc](../../../core/libs/prx/libSceContentExport/Export.cpp): their signatures remain unresolved; these names have no matching exports in the supplied module's C or assembly listing.

- [Content export arguments](../../../core/libs/prx/libSceContentExport/Export.cpp): file/data entry points and content-ID operations have restored argument lists. The Share Factory parameter contains a size, metadata string, content-ID list, count and option; metadata format and option values remain unresolved. Protected export bodies are unchanged.
