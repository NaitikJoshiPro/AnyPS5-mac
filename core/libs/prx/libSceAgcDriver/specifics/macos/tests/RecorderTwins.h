#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_RECORDERTWINS_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_RECORDERTWINS_H

// Metal ports of the tests/Recorder.cpp modes (agc_pipeline_cache, agc_host_import_unmap, ...). RecorderTwin.mm
// replaces tests/Recorder.cpp in agc_driver_recorder_tests and runs the port registered for the mode on its
// command line. A work package ports a mode in a source of its own that registers it, for example
//     static const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--host-import-unmap-only", &Run);
// and adds that source to the executable as one more twin of tests/Recorder.cpp in its twins/<package>.cmake:
//     "tests/Recorder.cpp=specifics/macos/tests/<Package>RecorderModes.mm"
// Run returns the exit code (0 passed, 77 skipped with a "skipped, ..." line, anything else failed).

namespace AgcDriver::MetalTwins {

using RecorderMode = int (*)();

bool RegisterRecorderMode(const char* flag, RecorderMode run);

}

#endif
