#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

class Window{
public:

    Window(const char* name, uint16_t width, uint16_t height)
        : m_windowName(name)
        , m_wndWidth(width)
        , m_wndHeight(height)
    {}

    virtual bool NeedClose() = 0;

protected:
    const char* m_windowName;
    uint16_t    m_wndWidth;
    uint16_t    m_wndHeight;
};