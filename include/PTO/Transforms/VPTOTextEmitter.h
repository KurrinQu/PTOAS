#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOTEXTEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOTEXTEMITTER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

struct VPTOEmissionOptions {
  bool dumpVPTOIR = false;
  bool printIntrinsicSelections = false;
  bool allowUnresolved = true;
  std::string unresolvedReportPath;
  std::string targetTriple;
  std::string march;
  std::string aicoreArch;
  std::string defaultTargetCPU;
  std::string defaultTargetFeatures;
};

LogicalResult translateVPTOModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const VPTOEmissionOptions &options,
                                        llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOTEXTEMITTER_H
