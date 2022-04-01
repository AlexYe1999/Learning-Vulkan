#pragma once
#include <cstdint>
#include <cstdlib>

class Application{
public:
    Application(const char* name)
        : m_appName(name)
    {};

    virtual ~Application() {};

    virtual void OnLoop() = 0;   
    virtual void OnInit() = 0;
    virtual void OnTick() = 0;
    virtual void OnUpdate() = 0;
    virtual void OnRender() = 0;
    virtual void OnDestroy() = 0;

    const char* GetName() const { return m_appName; }

protected:
    const char* m_appName;
};