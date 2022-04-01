#include "../app/AppFramework.hpp"
#include "../app/Application.hpp"
#include "../app/VkWindow.hpp"

#include <memory>

namespace LearningVulkan{

    class FirstApp final : public Application{
    public:
        FirstApp(const char* name)
            : Application(name)
        {};

        virtual void OnLoop(){
            while(!m_mainWindow->NeedClose()){
                glfwPollEvents();
            }
        }

        virtual void OnInit(){
            m_mainWindow = std::make_unique<VkWindow>(m_appName, 720, 480);
        }

        virtual void OnTick(){}
        virtual void OnUpdate(){}
        virtual void OnRender(){}
        virtual void OnDestroy(){}

    private:
        std::unique_ptr<VkWindow> m_mainWindow;
    };
}


int main(int argc, char** argv){

    return AppFramework::Run(&LearningVulkan::FirstApp("Hello Vulkan!"));

}