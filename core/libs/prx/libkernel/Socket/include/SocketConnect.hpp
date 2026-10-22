#pragma once
#ifdef _WIN32
#include <winsock2.h>

extern "C" void RefuseClosedLoopbackAtOnce_nid_no_patch(SOCKET socket, const sockaddr* address);
#endif

#ifdef __APPLE__
#include <sys/socket.h>

// FreeBSD 11 (solisten, sonewconn) clamps a backlog to somaxconn and then queues up to 3 * backlog / 2 + 1
// connections, dropping later SYNs so that the client retransmits. When its queue is full, Darwin answers
// a new connection by resetting a random pending one, which can be the new one (connect fails with
// ECONNRESET) even below FreeBSD's limit. Darwin keeps at least the backlog it is given, so it gets
// FreeBSD's queue length.
inline int HostListenBacklog(int backlog) {
    if (backlog < 0 || backlog > SOMAXCONN) backlog = SOMAXCONN;
    return 3 * backlog / 2 + 1;
}
#endif
