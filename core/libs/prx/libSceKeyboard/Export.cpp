#include "prx/libSceKeyboard/include/KeyboardState.hpp"

#include <stdexcept>
#include <string>

namespace {

// The process-wide keyboard settings the oops-sdk makes before it opens the keyboard
// (src/input/keyboard.c: SetProcessPrivilege(1), SetProcessFocus(1)). A host process always
// owns its keyboard and has its focus, so 1 is what this implementation already does; the
// meaning of other values is unknown.
int requireOne(const char* function, int value) {
    if (value != 1)
        throw std::runtime_error(std::string(function) + ": only 1 is implemented, got " + std::to_string(value));
    return KEYBOARD_OK;
}

}

extern "C" {

int APS5_VABI sceKeyboardInit(void) {
    return Keyboard::Initialize();
}

int APS5_VABI sceKeyboardOpen(int user_id, int32_t type, int32_t index, const void* param) {
    (void)param;
    return Keyboard::Open(user_id, type, index);
}

int APS5_VABI sceKeyboardSetProcessPrivilege(int privilege) {
    return requireOne(__func__, privilege);
}

int APS5_VABI sceKeyboardSetProcessFocus(int focus) {
    return requireOne(__func__, focus);
}

int APS5_VABI sceKeyboardClose(int32_t handle) {
    return Keyboard::Close(handle);
}

int APS5_VABI sceKeyboardRead(int32_t handle, KeyboardData* data, int32_t num) {
    return Keyboard::Read(handle, data, num);
}

int APS5_VABI sceKeyboardReadState(int32_t handle, KeyboardData* data) {
    return Keyboard::ReadState(handle, data);
}

int APS5_VABI sceKeyboardGetKey2Char(int32_t handle, int32_t arrange, uint32_t led, uint32_t modifier_key, uint16_t key_code, KeyboardCharData* char_data) {
    return Keyboard::GetKey2Char(handle, arrange, led, modifier_key, key_code, char_data);
}

void KeyboardPublishInput_nid_postfix(const KeyboardInputEvent& event) {
    Keyboard::Publish(event);
}

}
