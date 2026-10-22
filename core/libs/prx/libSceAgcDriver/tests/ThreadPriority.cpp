#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace {

using namespace AgcDriver;

constexpr std::uint64_t Unlimited = std::numeric_limits<std::uint64_t>::max();
constexpr RtkitLimits Rtkit{20, 200000};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void ParseTests() {
    for (const char* text : {static_cast<const char*>(nullptr), "", "off"}) Require(ParseThreadPriorityMode(text) == ThreadPriorityMode::Off, std::string("APS5_THREAD_PRIORITY=") + (text == nullptr ? "(unset)" : text) + " must leave priorities alone");
    Require(ParseThreadPriorityMode("rt") == ThreadPriorityMode::Realtime, "APS5_THREAD_PRIORITY=rt must ask for real-time priority");
    Require(ParseThreadPriorityMode("high") == ThreadPriorityMode::High, "APS5_THREAD_PRIORITY=high must ask for a nice level");
    for (const char* text : {"hgih", "HIGH", "realtime", "on", "1", "0", " high"}) {
        bool threw = false;
        try {
            static_cast<void>(ParseThreadPriorityMode(text));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Require(threw, std::string("APS5_THREAD_PRIORITY=") + text + " must be rejected");
    }
}

void QosParseTests() {
    for (const char* text : {static_cast<const char*>(nullptr), "", "user-interactive"})
        Require(ParseHostThreadQos(text) == HostThreadQos::UserInteractive, std::string("APS5_WORKER_QOS=") + (text == nullptr ? "(unset)" : text) + " must put frame-path threads at user-interactive");
    Require(ParseHostThreadQos("off") == HostThreadQos::Unchanged, "APS5_WORKER_QOS=off must leave the class alone");
    Require(ParseHostThreadQos("user-initiated") == HostThreadQos::UserInitiated, "user-initiated");
    Require(ParseHostThreadQos("default") == HostThreadQos::Default, "default");
    Require(ParseHostThreadQos("utility") == HostThreadQos::Utility, "utility");
    for (const char* text : {"background", "UI", "interactive", "1"}) {
        bool threw = false;
        try {
            static_cast<void>(ParseHostThreadQos(text));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Require(threw, std::string("APS5_WORKER_QOS=") + text + " must be rejected");
    }
}

void ExpectRefused(const RealtimePlan& plan, const std::string& what) {
    Require(!plan.refusal.empty() && !plan.limit.has_value(), what + " must refuse real-time priority without touching RLIMIT_RTTIME");
}

void ExpectLimit(const RealtimePlan& plan, RttimeLimit expected, const std::string& what) {
    Require(plan.refusal.empty() && plan.limit.has_value(), what + " must set RLIMIT_RTTIME");
    Require(*plan.limit == expected, what + " set RLIMIT_RTTIME to soft " + std::to_string(plan.limit->soft) + " hard " + std::to_string(plan.limit->hard) + ", expected soft " + std::to_string(expected.soft) + " hard " + std::to_string(expected.hard));
}

void PlanTests() {
    ExpectRefused(PlanRealtime(10, std::nullopt, {Unlimited, Unlimited}, 200000), "an unreachable rtkit");
    ExpectRefused(PlanRealtime(10, std::nullopt, {30000, 150000}, 200000), "an unreachable rtkit");
    ExpectRefused(PlanRealtime(21, Rtkit, {Unlimited, Unlimited}, 200000), "a priority above rtkit's MaxRealtimePriority");
    ExpectRefused(PlanRealtime(10, RtkitLimits{20, 0}, {Unlimited, Unlimited}, 200000), "rtkit without an RTTimeUSecMax");
    ExpectLimit(PlanRealtime(20, Rtkit, {Unlimited, Unlimited}, 200000), {100000, 200000}, "an unlimited hard limit");
    ExpectLimit(PlanRealtime(10, Rtkit, {100000, Unlimited}, 200000), {100000, 200000}, "a lowered soft limit under an unlimited hard limit");
    ExpectLimit(PlanRealtime(10, Rtkit, {Unlimited, Unlimited}, 500000), {100000, 200000}, "APS5_THREAD_RTTIME_US above RTTimeUSecMax");
    ExpectLimit(PlanRealtime(10, Rtkit, {Unlimited, Unlimited}, 100000), {50000, 100000}, "APS5_THREAD_RTTIME_US below RTTimeUSecMax");
    ExpectLimit(PlanRealtime(10, Rtkit, {150000, 150000}, 200000), {75000, 150000}, "a hard limit rtkit accepts with a soft limit above half of it");
    const auto kept = PlanRealtime(10, Rtkit, {30000, 150000}, 200000);
    Require(kept.refusal.empty() && !kept.limit.has_value(), "limits rtkit already accepts must be left alone");
}

#ifdef __APPLE__
// A frame-path thread is named after its role and runs at user-interactive QoS (D70).
void RoleTest() {
    std::string failure;
    std::thread([&failure] {
        try {
            RaiseWorkerThreadPriority("queue 0x0 worker");
            char name[64] = {};
            pthread_getname_np(pthread_self(), name, sizeof(name));
            if (std::string(name) != "queue 0x0 worker") failure = std::string("thread name is \"") + name + "\"";
            else if (qos_class_self() != QOS_CLASS_USER_INTERACTIVE) failure = "QoS class is " + std::to_string(qos_class_self());
        } catch (const std::exception& error) {
            failure = error.what();
        }
    }).join();
    Require(failure.empty(), "RaiseWorkerThreadPriority: " + failure);
}
#endif

void RefusalTest() {
    bool threw = false;
    try {
        RaiseWorkerThreadPriority("test");
    } catch (const std::runtime_error& error) {
        threw = true;
        std::cout << "refused: " << error.what() << '\n';
    }
    Require(threw, "a refused priority request must throw");
}

}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "refused") {
            RefusalTest();
            return 0;
        }
        ParseTests();
        QosParseTests();
#ifdef __APPLE__
        RoleTest();
#endif
        PlanTests();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
