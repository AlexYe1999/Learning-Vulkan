#include "VkWindow.hpp"

namespace LearningVulkan{

    void VkWindow::InitWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(m_wndWidth, m_wndHeight, m_windowName, nullptr, nullptr);
    }

    void VkWindow::DestroyWindow(){
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }


}