# =========================================================
# Python Extension Configuration (Pybind11 Version)
# =========================================================

# 1. 查找依赖
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 2. 创建模块 (使用 pybind11)
pybind11_add_module(_pto
  PTOModule.cpp
)

# 3. 链接依赖
target_link_libraries(_pto PRIVATE
  PTOIR                       # 您的 Dialect 核心库
  PTOCAPI                     # 您的 C API 库
  MLIRIR                      # MLIR 核心
  MLIRBindingsPythonHeaders   # 包含 PybindAdaptors.h 的库
)

# [关键修复] 显式添加 PybindAdaptors.h 所在的源码路径
# 假设 LLVM_SOURCE_DIR 是 LLVM 的根目录，我们需要指向 mlir/include
target_include_directories(_pto PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# 4. 设置编译定义
target_compile_definitions(_pto PRIVATE
  MLIR_CAPI_BUILDING_LIBRARY=1
)

# 5. 设置输出名称
set_target_properties(_pto PROPERTIES OUTPUT_NAME "_pto")
