#include "../app/AppFramework.hpp"
#include "../app/Application.hpp"
#include "../app/VkWindow.hpp"
#include "Pipeline.hpp"

#include <iostream>
#include <memory>

namespace LearningVulkan{

    class PipelineOverview final : public Application{
    public:
        PipelineOverview(const char* name)
            : Application(name)
        {};

        virtual void OnLoop(){
            while(!m_mainWindow->NeedClose()){
                glfwPollEvents();
            }
        }

        virtual void OnInit(){

            try{
                m_mainWindow = std::make_unique<VkWindow>(m_appName, 720, 480);
                m_pipeline = std::make_unique<Pipeline>(
                    "shader/02/SimpleShader.vert.spv",
                    "shader/02/SimpleShader.frag.spv"
                );                
            }
            catch(const std::exception& e){
                std::cerr << e.what() << std::endl;
            }

        }

        virtual void OnTick(){}
        virtual void OnUpdate(){}
        virtual void OnRender(){}
        virtual void OnDestroy(){}

    private:
        std::unique_ptr<VkWindow> m_mainWindow;
        std::unique_ptr<Pipeline> m_pipeline;
    };
}


int main(int argc, char** argv){

    return AppFramework::Run(&LearningVulkan::PipelineOverview("Overview Of Pipeline"));

}