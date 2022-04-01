#pragma once
#include "Application.hpp"
#include <cstdint>
#include <cstdlib>

class AppFramework{
public:
    static int64_t Run(Application* pApp);

protected:
    inline static Application* app = nullptr;
};

inline int64_t AppFramework::Run(Application* pApp){
    pApp->OnInit();
    pApp->OnLoop();
    pApp->OnDestroy();
    return 0;
};