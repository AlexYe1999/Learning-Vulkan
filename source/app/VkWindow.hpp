#pragma once
#include "Window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace LearningVulkan{

    class VkWindow : public Window{
    public:
        VkWindow(const char* name, uint16_t width, uint16_t height) 
            : Window(name, width, height)
        {
            InitWindow();
        }

        ~VkWindow(){
            DestroyWindow();
        }

        virtual bool NeedClose() override {
            return glfwWindowShouldClose(m_window);
        }

        VkWindow(VkWindow&& window) = delete;
        VkWindow(const VkWindow& window) = delete;
        void operator=(VkWindow&& window) = delete;
        void operator=(const VkWindow& window) = delete;

    private:
        void InitWindow();  
        void DestroyWindow();

    protected:
        GLFWwindow* m_window;
    };    
    
}

