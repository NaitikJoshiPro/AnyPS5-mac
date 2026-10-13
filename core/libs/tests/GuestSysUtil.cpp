#include "prx/libc/include/general/VabiMacros.hpp"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

extern "C" {
int APS5_VABI sceSysUtilSendSystemNotificationWithText(int type, const char* text);
}

static void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stdout, "failed: %s\n", what);
    std::exit(1);
}

int main() {
    char path[] = "/tmp/aps5-sysutil-XXXXXX";
    const int file = mkstemp(path);
    Require(file >= 0, "mkstemp");
    std::fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    Require(saved >= 0 && dup2(file, STDERR_FILENO) >= 0, "redirect stderr");
    const int result = sceSysUtilSendSystemNotificationWithText(222, "Trophy unlocked: first lap");
    bool threw = false;
    try {
        sceSysUtilSendSystemNotificationWithText(0, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    std::string logged;
    char buffer[512];
    lseek(file, 0, SEEK_SET);
    for (ssize_t read = 0; (read = ::read(file, buffer, sizeof(buffer))) > 0;) logged.append(buffer, static_cast<std::size_t>(read));
    close(file);
    unlink(path);
    Require(result == 0, "notification returns 0");
    Require(threw, "a null text throws invalid_argument");
    Require(logged.find("(type 222): Trophy unlocked: first lap\n") != std::string::npos, "the text reaches the log");
    return 0;
}
