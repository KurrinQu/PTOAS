#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H

#include "PTO/Transforms/VPTOTextEmitter.h"

namespace mlir {
class ModuleOp;
}

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {

LogicalResult
translateVPTOModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const VPTOEmissionOptions &options,
                              llvm::raw_ostream &diagOS);

LogicalResult
translateVPTOModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const VPTOEmissionOptions &options,
                                 llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H
