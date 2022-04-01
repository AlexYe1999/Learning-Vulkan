#pragma once

#include <cstdint>
#include <vector>



namespace LearningVulkan{

    class Pipeline{
    public:
        Pipeline(const char* vertFilePath, const char* fragFilePath){
            CreatePipeline(vertFilePath, fragFilePath);
        }

    private:
        static std::vector<uint8_t> readFile(const char* filePath);

        void CreatePipeline(const char* vertFilePath, const char* fragFilePath);
    };
}