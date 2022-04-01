
set(ALL_FILES
    main.cpp
    Pipeline.hpp
    Pipeline.cpp
)

add_executable(PipelineOverview ${ALL_FILES})
target_link_libraries(PipelineOverview AppFramework)

set_property(TARGET PipelineOverview PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")