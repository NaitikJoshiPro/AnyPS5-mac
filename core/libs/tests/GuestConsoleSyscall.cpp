#include "SceTypes.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>

extern "C" std::int64_t APS5_VABI Aps5ConsoleSyscall_nid_no_patch(const std::uint64_t* frame);

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "failed: %s\n", what);
    std::exit(1);
}

std::int64_t Call(std::uint64_t number, std::initializer_list<std::uint64_t> arguments) {
    std::uint64_t frame[7] {number};
    std::size_t index = 1;
    for (const auto argument : arguments) frame[index++] = argument;
    return Aps5ConsoleSyscall_nid_no_patch(frame);
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uint64_t>(pointer); }

constexpr int OpenWriteCreate = 0x0001 | 0x0200 | 0x0400;
constexpr int Enoent = 2;
constexpr int Ebadf = 9;

}

int main() {
    const std::string root = "anyps5-console-syscall-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string file = root + "/log.txt";
    const std::string renamed = root + "/renamed.txt";
    Require(Call(5, {Address("anyps5-console-syscall-missing/none"), 0, 0}) == -Enoent, "open of a missing path gives -ENOENT");
    Require(Call(6, {0x7fffff}) == -Ebadf, "close of a bad descriptor gives -EBADF");
    Require(Call(20, {}) > 0, "getpid");
    Require(Call(136, {Address(root.c_str()), 0755}) == 0, "mkdir");
    const std::int64_t descriptor = Call(5, {Address(file.c_str()), OpenWriteCreate, 0644});
    Require(descriptor >= 0, "open creates a file");
    Require(Call(4, {static_cast<std::uint64_t>(descriptor), Address("lap 1"), 5}) == 5, "write");
    Require(Call(478, {static_cast<std::uint64_t>(descriptor), 0, 1}) == 5, "lseek reports the position");
    Require(Call(6, {static_cast<std::uint64_t>(descriptor)}) == 0, "close");
    Require(Call(128, {Address(file.c_str()), Address(renamed.c_str())}) == 0, "rename");
    const std::int64_t reader = Call(5, {Address(renamed.c_str()), 0, 0});
    Require(reader >= 0, "open for reading");
    char buffer[8] {};
    Require(Call(3, {static_cast<std::uint64_t>(reader), Address(buffer), sizeof(buffer)}) == 5 && std::memcmp(buffer, "lap 1", 5) == 0, "read");
    Require(Call(6, {static_cast<std::uint64_t>(reader)}) == 0, "close the reader");
    Require(Call(15, {Address(renamed.c_str()), 0600}) == 0, "chmod");
    Require(Call(10, {Address(renamed.c_str())}) == 0, "unlink");
    Require(Call(137, {Address(root.c_str())}) == 0, "rmdir");
    Require(!std::filesystem::exists(root), "the directory is gone");
    KernelTimespec now {};
    Require(Call(232, {0, Address(&now)}) == 0 && (now.tv_sec != 0 || now.tv_nsec != 0), "clock_gettime");
    Require(Call(601, {7, Address("klog line from the console syscall test\n")}) == 0, "klog");
    bool threw = false;
    try {
        Call(2, {});
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "console system call 2 is not supported";
    }
    Require(threw, "an unsupported system call throws");
    return 0;
}
