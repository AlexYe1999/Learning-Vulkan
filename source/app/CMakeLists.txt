
set(ALL_FILES
    AppFramework.hpp
    Application.hpp
    VkWindow.hpp
    VkWindow.cpp
    Window.hpp
)

add_library(AppFramework ${ALL_FILES})
target_link_libraries(AppFramework glfw)