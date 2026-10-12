#include "SceTypes.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>

extern "C" {
int APS5_VABI kqueue_nid_postfix();
int APS5_VABI kevent_nid_postfix(int queue, const KernelEvent* changes, int changeCount, KernelEvent* events, int eventCount, const KernelTimespec* timeout);
int APS5_VABI close_nid_postfix(int d);
int* APS5_VABI __error_nid_postfix();
}

static constexpr std::int16_t EvfiltVnode = -4;
static constexpr std::int16_t EvfiltUser = -11;
static constexpr std::uint16_t EvAdd = 0x1;
static constexpr std::uint16_t EvDelete = 0x2;
static constexpr std::uint16_t EvEnable = 0x4;
static constexpr std::uint16_t EvClear = 0x20;
static constexpr std::uint16_t EvForceOneshot = 0x100;
static constexpr std::uint32_t NoteDelete = 0x1;
static constexpr std::uint32_t NoteWrite = 0x2;
static constexpr std::uint32_t NoteExtend = 0x4;
static constexpr std::uint32_t NoteLink = 0x10;
static constexpr std::uint32_t NoteRename = 0x20;
static constexpr std::uint32_t NoteOpen = 0x80;
static constexpr int GuestEbadf = 9;
static constexpr int GuestEfault = 14;

static void Require(bool value) { if (!value) std::abort(); }

static KernelEvent Change(int descriptor, std::int16_t filter, std::uint16_t flags, std::uint32_t notes) {
    KernelEvent change;
    change.ident = static_cast<std::uintptr_t>(descriptor);
    change.filter = filter;
    change.flags = flags;
    change.fflags = notes;
    return change;
}

static bool Rejects(int queue, const KernelEvent& change) {
    try {
        kevent_nid_postfix(queue, &change, 1, nullptr, 0, nullptr);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

int main() {
    char path[] = "/tmp/anyps5-kqueue-XXXXXX";
    const int file = ::mkstemp(path);
    Require(file >= 0);
    const int queue = kqueue_nid_postfix();
    Require(queue >= 0);

    const KernelTimespec poll{0, 0};
    const KernelTimespec second{1, 0};
    KernelEvent events[4];
    const KernelEvent watch = Change(file, EvfiltVnode, EvAdd | EvEnable | EvClear, NoteDelete | NoteWrite | NoteExtend | NoteLink | NoteRename);
    Require(kevent_nid_postfix(queue, &watch, 1, nullptr, 0, nullptr) == 0);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &poll) == 0);

    Require(::write(file, "x", 1) == 1);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &second) == 1);
    Require(events[0].ident == static_cast<std::uintptr_t>(file));
    Require(events[0].filter == EvfiltVnode);
    Require((events[0].fflags & NoteWrite) != 0);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &poll) == 0);

    Require(::unlink(path) == 0);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &second) == 1);
    Require((events[0].fflags & NoteDelete) != 0);

    const KernelEvent remove = Change(file, EvfiltVnode, EvDelete, 0);
    Require(kevent_nid_postfix(queue, &remove, 1, nullptr, 0, nullptr) == 0);
    Require(::write(file, "y", 1) == 1);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &poll) == 0);

    Require(Rejects(queue, Change(1, EvfiltUser, EvAdd, 0)));
    Require(Rejects(queue, Change(file, EvfiltVnode, EvAdd | EvForceOneshot, NoteWrite)));
    Require(Rejects(queue, Change(file, EvfiltVnode, EvAdd, NoteOpen)));
    Require(Rejects(queue, Change(0x10000000, EvfiltVnode, EvAdd, NoteWrite)));

    Require(kevent_nid_postfix(queue, nullptr, 0, nullptr, 4, &poll) == -1);
    Require(*__error_nid_postfix() == GuestEfault);
    Require(close_nid_postfix(queue) == 0);
    Require(kevent_nid_postfix(queue, nullptr, 0, events, 4, &poll) == -1);
    Require(*__error_nid_postfix() == GuestEbadf);
    Require(::close(file) == 0);
    return 0;
}
