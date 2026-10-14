// The guest-thread sampler (APS5_FEX_SAMPLE) maps a spinning thread's host PC back to the guest RIP of its loop and
// reports the guest registers there: run_fex_fixtures.py checks that the hottest RIP lies in sample_poll and that
// rdi holds the address of sampleFlag, the word it polls.
extern "C" [[noreturn]] void exit(int);

// A data relocation: the relinker wants DT_RELA.
void (*volatile exitImport)(int) = exit;

extern "C" {
volatile unsigned sampleFlag;

__attribute__((noinline)) unsigned long sample_poll(volatile unsigned* flag, unsigned long budget) {
    while (*flag == 0)
        if (--budget == 0) *flag = 1;
    return budget;
}
}

extern "C" [[noreturn]] void _start(void*) {
    // About a second of JIT code; the sampler runs every millisecond.
    if (sample_poll(&sampleFlag, 600000000ul) != 0) exit(1);
    exitImport(43);
    exit(2);
}
