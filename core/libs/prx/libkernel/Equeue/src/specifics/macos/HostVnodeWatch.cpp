#include "prx/libkernel/Equeue/src/specifics/macos/HostVnodeWatch.hpp"

#include <cerrno>
#include <climits>
#include <cstddef>
#include <ctime>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/event.h>
#include <sys/fsgetpath.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

// The host kqueue reports what changed on a vnode but not which directory entry changed or how, so
// directory watches keep a listing and diff it on every host event to rebuild what inotify reports on
// Linux: entries created or deleted, renamed inside the directory (paired move) or moved across it.

namespace HostVnode {

namespace {

constexpr int GUEST_EBADF = 9;
constexpr int GUEST_EINVAL = 22;
constexpr std::uint32_t GUEST_NOTE_DELETE = 0x0001;
constexpr std::uint32_t GUEST_NOTE_WRITE = 0x0002;
constexpr std::uint32_t GUEST_NOTE_EXTEND = 0x0004;
constexpr std::uint32_t GUEST_NOTE_ATTRIB = 0x0008;
constexpr std::uint32_t GUEST_NOTE_LINK = 0x0010;
constexpr std::uint32_t GUEST_NOTE_RENAME = 0x0020;
constexpr std::uint32_t HOST_NOTES = NOTE_DELETE | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK | NOTE_RENAME | NOTE_REVOKE;
constexpr std::size_t HOST_EVENT_BATCH = 16;
constexpr int MAX_RESCANS = 8;
constexpr std::size_t MAX_DEPARTED = 4096;

struct Entry {
    ino_t inode;
    bool directory;
};

using Listing = std::unordered_map<std::string, Entry>;

enum class Location { Gone, Inside, Elsewhere };

[[noreturn]] void Fail(const char* what, int error) {
    throw std::runtime_error(std::string("EVFILT_VNODE: ") + what + " failed, errno=" + std::to_string(error));
}

std::int64_t Nanoseconds(const timespec& time) {
    return static_cast<std::int64_t>(time.tv_sec) * 1000000000 + time.tv_nsec;
}

std::int64_t RealtimeNanoseconds() {
    timespec now{};
    ::clock_gettime(CLOCK_REALTIME, &now);
    return Nanoseconds(now);
}

bool SameNode(const struct stat& left, const struct stat& right) {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

void CloseDescriptor(int& descriptor) {
    if (descriptor >= 0) {
        ::close(descriptor);
        descriptor = -1;
    }
}

}

class Watch {
public:
    Watch() = default;
    ~Watch() {
        CloseDescriptor(queue);
        CloseDescriptor(entries);
        CloseDescriptor(target);
    }

    Watch(const Watch&) = delete;
    Watch& operator=(const Watch&) = delete;

    int queue = -1;
    int target = -1;
    int entries = -1;
    bool directory = false;
    struct stat last {};
    fsid_t volume{};
    Listing listing;
    std::int64_t scannedAt = 0;
    std::unordered_set<ino_t> departed;
};

namespace {

// Opens a host descriptor of our own on the guest's file: by path with O_EVTONLY when the path still
// names the same vnode, or as a duplicate of the guest descriptor (an unlinked file, for instance).
int OpenOwn(int descriptor, const struct stat& status, int flags) {
    char path[PATH_MAX];
    if (::fcntl(descriptor, F_GETPATH, path) == 0) {
        const int opened = ::open(path, flags | O_CLOEXEC);
        struct stat reopened {};
        if (opened >= 0 && ::fstat(opened, &reopened) == 0 && SameNode(reopened, status)) {
            return opened;
        }
        if (opened >= 0) {
            ::close(opened);
        }
    }
    return ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
}

void Scan(const Watch& watch, Listing& listing, std::int64_t& startedAt) {
    listing.clear();
    startedAt = RealtimeNanoseconds();
    const int descriptor = ::openat(watch.entries, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0) {
        if (errno == ENOENT) {
            return;
        }
        Fail("opening the watched directory", errno);
    }
    DIR* directory = ::fdopendir(descriptor);
    if (directory == nullptr) {
        const int error = errno;
        ::close(descriptor);
        Fail("fdopendir", error);
    }
    for (;;) {
        errno = 0;
        const dirent* entry = ::readdir(directory);
        if (entry == nullptr) {
            const int error = errno;
            ::closedir(directory);
            if (error != 0 && error != ENOENT) {
                Fail("reading the watched directory", error);
            }
            return;
        }
        const std::string name(entry->d_name, entry->d_namlen);
        if (name == "." || name == "..") {
            continue;
        }
        bool isDirectory = entry->d_type == DT_DIR;
        if (entry->d_type == DT_UNKNOWN) {
            struct stat status {};
            if (::fstatat(watch.entries, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
                continue;
            }
            isDirectory = S_ISDIR(status.st_mode);
        }
        listing[name] = Entry{static_cast<ino_t>(entry->d_ino), isDirectory};
    }
}

// Where an entry that left the listing went: deleted, still in this directory (the listing raced a
// rename), or into another directory of the same volume.
Location Locate(Watch& watch, ino_t inode) {
    char path[PATH_MAX];
    const auto length = ::fsgetpath(path, sizeof(path), &watch.volume, static_cast<std::uint64_t>(inode));
    if (length <= 0) {
        return Location::Gone;
    }
    std::string parent(path);
    const auto separator = parent.find_last_of('/');
    if (separator == std::string::npos) {
        return Location::Elsewhere;
    }
    parent.resize(separator == 0 ? 1 : separator);
    struct stat status {};
    if (::lstat(parent.c_str(), &status) == 0 && SameNode(status, watch.last)) {
        return Location::Inside;
    }
    return Location::Elsewhere;
}

// An entry that appeared but was born before the previous listing was taken came from another
// directory. Hard links (more than one link to a file) count as created, as IN_CREATE does on Linux.
bool MovedIn(Watch& watch, const std::string& name, const Entry& entry) {
    if (watch.departed.erase(entry.inode) != 0) {
        return true;
    }
    struct stat status {};
    if (::fstatat(watch.entries, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        return false;
    }
    return Nanoseconds(status.st_birthtimespec) < watch.scannedAt && (S_ISDIR(status.st_mode) || status.st_nlink == 1);
}

void Remember(Watch& watch, ino_t inode) {
    if (watch.departed.size() >= MAX_DEPARTED) {
        watch.departed.clear();
    }
    watch.departed.insert(inode);
}

void DirectoryEntryHints(Watch& watch, std::vector<std::uint32_t>& hints) {
    Listing current;
    std::int64_t startedAt = 0;
    std::vector<ino_t> movedOut;
    for (int attempt = 0;; ++attempt) {
        Scan(watch, current, startedAt);
        hints.clear();
        movedOut.clear();
        std::unordered_set<ino_t> previousInodes;
        std::unordered_set<ino_t> currentInodes;
        for (const auto& [name, entry] : watch.listing) {
            previousInodes.insert(entry.inode);
        }
        for (const auto& [name, entry] : current) {
            currentInodes.insert(entry.inode);
        }
        bool raced = false;
        for (const auto& [name, entry] : watch.listing) {
            const auto kept = current.find(name);
            if (kept != current.end() && kept->second.inode == entry.inode) {
                continue;
            }
            if (currentInodes.count(entry.inode) != 0) {
                hints.push_back(GUEST_NOTE_WRITE);
                continue;
            }
            const std::uint32_t link = entry.directory ? GUEST_NOTE_LINK : 0;
            switch (Locate(watch, entry.inode)) {
            case Location::Gone:
                hints.push_back(GUEST_NOTE_WRITE | link);
                break;
            case Location::Elsewhere:
                hints.push_back(GUEST_NOTE_WRITE | GUEST_NOTE_EXTEND | link);
                movedOut.push_back(entry.inode);
                break;
            case Location::Inside:
                raced = true;
                hints.push_back(GUEST_NOTE_WRITE);
                break;
            }
        }
        if (raced && attempt < MAX_RESCANS) {
            continue;
        }
        for (const auto& [name, entry] : current) {
            const auto kept = watch.listing.find(name);
            if ((kept != watch.listing.end() && kept->second.inode == entry.inode) || previousInodes.count(entry.inode) != 0) {
                continue;
            }
            const std::uint32_t link = entry.directory ? GUEST_NOTE_LINK : 0;
            hints.push_back(MovedIn(watch, name, entry) ? GUEST_NOTE_WRITE | GUEST_NOTE_EXTEND | link : GUEST_NOTE_WRITE | link);
        }
        break;
    }
    for (const auto inode : movedOut) {
        Remember(watch, inode);
    }
    watch.listing = std::move(current);
    watch.scannedAt = startedAt;
}

std::vector<std::uint32_t> DirectoryHints(Watch& watch, std::uint32_t host) {
    std::vector<std::uint32_t> hints;
    if ((host & (NOTE_WRITE | NOTE_EXTEND | NOTE_LINK)) != 0) {
        DirectoryEntryHints(watch, hints);
        // The status is read after the listing, so a change the listing missed shows up in the next diff.
        struct stat current {};
        if (::fstat(watch.target, &current) == 0) {
            const bool modified = Nanoseconds(current.st_mtimespec) != Nanoseconds(watch.last.st_mtimespec);
            if (hints.empty() && modified) {
                // Changes that cancelled out between two listings (create then delete, rename and back).
                hints.push_back(GUEST_NOTE_WRITE | ((host & NOTE_LINK) != 0 ? GUEST_NOTE_LINK : 0));
            }
            watch.last = current;
        }
    }
    if ((host & NOTE_ATTRIB) != 0) {
        hints.push_back(GUEST_NOTE_ATTRIB);
    }
    if ((host & NOTE_RENAME) != 0) {
        hints.push_back(GUEST_NOTE_RENAME);
    }
    if ((host & NOTE_DELETE) != 0) {
        hints.push_back(GUEST_NOTE_DELETE);
    }
    return hints;
}

std::vector<std::uint32_t> FileHints(Watch& watch, std::uint32_t host) {
    std::vector<std::uint32_t> hints;
    if ((host & (NOTE_WRITE | NOTE_EXTEND)) != 0) {
        struct stat current {};
        if (::fstat(watch.target, &current) != 0) {
            hints.push_back(GUEST_NOTE_WRITE);
        } else {
            std::uint32_t hint = GUEST_NOTE_WRITE;
            if (current.st_size > watch.last.st_size) {
                hint = GUEST_NOTE_WRITE | GUEST_NOTE_EXTEND;
            } else if (current.st_size < watch.last.st_size) {
                hint = GUEST_NOTE_ATTRIB;
            }
            watch.last = current;
            hints.push_back(hint);
        }
    }
    if ((host & (NOTE_ATTRIB | NOTE_LINK | NOTE_DELETE)) != 0) {
        struct stat current {};
        std::uint32_t hint = (host & NOTE_DELETE) != 0 ? GUEST_NOTE_DELETE : (host & NOTE_ATTRIB) != 0 ? GUEST_NOTE_ATTRIB : GUEST_NOTE_LINK;
        if (::fstat(watch.target, &current) == 0) {
            if (current.st_nlink > watch.last.st_nlink) {
                hint = GUEST_NOTE_LINK;
            } else if (current.st_nlink < watch.last.st_nlink) {
                hint = GUEST_NOTE_DELETE;
            }
            watch.last = current;
        }
        hints.push_back(hint);
    }
    if ((host & NOTE_RENAME) != 0) {
        hints.push_back(GUEST_NOTE_RENAME);
    }
    return hints;
}

}

int Open(int descriptor, std::shared_ptr<Watch>& watch) {
    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        return GUEST_EBADF;
    }
    if (S_ISFIFO(status.st_mode) || S_ISSOCK(status.st_mode)) {
        return GUEST_EINVAL;
    }
    if (!S_ISREG(status.st_mode) && !S_ISDIR(status.st_mode)) {
        return UNSUPPORTED_DESCRIPTOR;
    }
    auto opened = std::make_shared<Watch>();
    opened->directory = S_ISDIR(status.st_mode);
    opened->target = OpenOwn(descriptor, status, O_EVTONLY);
    if (opened->target < 0 || ::fstat(opened->target, &opened->last) != 0) {
        Fail("opening the watched file", errno);
    }
    if (opened->directory) {
        opened->entries = OpenOwn(descriptor, status, O_RDONLY | O_DIRECTORY);
        if (opened->entries < 0) {
            Fail("opening the watched directory", errno);
        }
        struct statfs volume {};
        if (::fstatfs(opened->target, &volume) != 0) {
            Fail("fstatfs", errno);
        }
        opened->volume = volume.f_fsid;
    }
    opened->queue = ::kqueue();
    if (opened->queue < 0 || ::fcntl(opened->queue, F_SETFD, FD_CLOEXEC) != 0) {
        Fail("kqueue", errno);
    }
    struct kevent change {};
    EV_SET(&change, static_cast<uintptr_t>(opened->target), EVFILT_VNODE, EV_ADD | EV_CLEAR, HOST_NOTES, 0, nullptr);
    if (::kevent(opened->queue, &change, 1, nullptr, 0, nullptr) != 0) {
        Fail("registering the host vnode watch", errno);
    }
    if (opened->directory) {
        // The first listing is taken after the host watch is armed, so no change can fall between them.
        Scan(*opened, opened->listing, opened->scannedAt);
    }
    watch = std::move(opened);
    return 0;
}

void Run(Watch& watch, int stopDescriptor, const std::function<void(std::uint32_t)>& deliver) {
    struct kevent stop {};
    EV_SET(&stop, static_cast<uintptr_t>(stopDescriptor), EVFILT_READ, EV_ADD, 0, 0, nullptr);
    if (::kevent(watch.queue, &stop, 1, nullptr, 0, nullptr) != 0) {
        Fail("registering the watch stop signal", errno);
    }
    struct kevent events[HOST_EVENT_BATCH];
    for (;;) {
        const int count = ::kevent(watch.queue, nullptr, 0, events, static_cast<int>(HOST_EVENT_BATCH), nullptr);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            Fail("waiting on the host kqueue", errno);
        }
        std::uint32_t host = 0;
        for (int index = 0; index < count; ++index) {
            if (events[index].filter == EVFILT_READ) {
                return;
            }
            if ((events[index].flags & EV_ERROR) != 0) {
                Fail("host vnode watch", static_cast<int>(events[index].data));
            }
            host |= events[index].fflags;
        }
        // A revoked vnode (forced unmount) ends the watch silently, as IN_IGNORED does on Linux.
        if ((host & NOTE_REVOKE) != 0) {
            return;
        }
        for (const auto hint : watch.directory ? DirectoryHints(watch, host) : FileHints(watch, host)) {
            deliver(hint);
        }
    }
}

}
