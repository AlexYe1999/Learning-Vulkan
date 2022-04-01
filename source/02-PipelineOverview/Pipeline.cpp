#include "Pipeline.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace LearningVulkan{

    std::vector<uint8_t> Pipeline::readFile(const char* filePath){

        std::ifstream file(filePath, std::ios::ate | std::ios::binary);

        if(!file.is_open()){
            throw std::runtime_error("Failed to open file" + std::string(filePath));
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<uint8_t> buffer(fileSize);

        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

        return buffer;
    }


    void Pipeline::CreatePipeline(const char* vertFilePath, const char* fragFilePath){
        auto vertexCode = readFile(vertFilePath);
        auto fragCode   = readFile(fragFilePath);

    #if defined(_DEBUG)
        std::cout << "Vertex   Shader Code Size: " << vertexCode.size() << '\n';
        std::cout << "Fragment Shader Code Size: " << fragCode.size()   << '\n';

    #endif // _DEBUG
    }


}