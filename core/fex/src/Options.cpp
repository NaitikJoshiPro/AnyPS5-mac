#include "Options.hpp"

#include <FEXCore/Config/Config.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace Aps5Fex {

namespace {

constexpr std::pair<std::string_view, FEXCore::Config::ConfigOption> ConfigOptions[] = {
#define OPT_BASE(type, group, enum, json, default) {#json, FEXCore::Config::CONFIG_##enum},
#include <FEXCore/Config/ConfigValues.inl>
};

constexpr std::pair<std::string_view, qos_class_t> QosClasses[] = {
    {"user-interactive", QOS_CLASS_USER_INTERACTIVE},
    {"user-initiated", QOS_CLASS_USER_INITIATED},
    {"default", QOS_CLASS_DEFAULT},
    {"utility", QOS_CLASS_UTILITY},
    {"background", QOS_CLASS_BACKGROUND},
};

qos_class_t threadQos = QOS_CLASS_UNSPECIFIED;
bool clearCode = false;

void SetConfig(std::string_view entry) {
    const auto equals = entry.find('=');
    if (equals == std::string_view::npos) throw std::runtime_error("APS5_FEX_CONFIG: expected Key=Value, got \"" + std::string(entry) + "\"");
    const auto key = entry.substr(0, equals);
    const auto* option = std::find_if(std::begin(ConfigOptions), std::end(ConfigOptions), [key](const auto& known) { return known.first == key; });
    if (option == std::end(ConfigOptions)) throw std::runtime_error("APS5_FEX_CONFIG: FEXCore has no option " + std::string(key));
    FEXCore::Config::Set(option->second, entry.substr(equals + 1));
}

}

void LoadOptions() {
    if (const char* config = std::getenv("APS5_FEX_CONFIG")) {
        std::string_view rest(config);
        while (!rest.empty()) {
            const auto comma = rest.find(',');
            if (const auto entry = rest.substr(0, comma); !entry.empty()) SetConfig(entry);
            rest = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
        }
    }
    if (const char* clear = std::getenv("APS5_FEX_CLEAR_CODE")) clearCode = std::string_view(clear) == "1";
    if (const char* qos = std::getenv("APS5_FEX_QOS"); qos != nullptr && *qos != 0) {
        const std::string_view name(qos);
        const auto* found = std::find_if(std::begin(QosClasses), std::end(QosClasses), [name](const auto& known) { return known.first == name; });
        if (found == std::end(QosClasses))
            throw std::runtime_error("APS5_FEX_QOS: " + std::string(name) + " is not user-interactive, user-initiated, default, utility or background");
        threadQos = found->second;
    }
}

bool ClearCodeOnGuestEntry() {
    return clearCode;
}

void ApplyThreadQos() {
    if (threadQos != QOS_CLASS_UNSPECIFIED) pthread_set_qos_class_self_np(threadQos, 0);
}

}
