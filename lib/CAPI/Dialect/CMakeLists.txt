add_mlir_public_c_api_library(PTOCAPI
    PTO.cpp

    LINK_LIBS PUBLIC
    PTOIR
    MLIRCAPIIR
)

install(TARGETS PTOCAPI
    EXPORT PTOASTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 设置头文件包含路径
target_include_directories(PTOCAPI INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)