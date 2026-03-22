#ifndef MLIR_DIALECT_PTO_TRANSFORMS_A5VMLLVMEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_A5VMLLVMEMITTER_H

#include "PTO/Transforms/A5VMTextEmitter.h"

namespace mlir {
class ModuleOp;
}

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {

LogicalResult
translateA5VMModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const A5VMEmissionOptions &options,
                              llvm::raw_ostream &diagOS);

LogicalResult
translateA5VMModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const A5VMEmissionOptions &options,
                                 llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_A5VMLLVMEMITTER_H
