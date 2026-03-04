#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

/// A cross-domain data dependency that needs bridging.
struct BridgePoint {
  Value producerValue;         // SSA value produced in one section
  Operation *producerSection;  // SectionCubeOp or SectionVectorOp
  SmallVector<OpOperand *> consumerUses; // uses in the other section
  Operation *consumerSection;
};

class CVInsertBridgePass
    : public PassWrapper<CVInsertBridgePass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVInsertBridgePass)

  StringRef getArgument() const override {
    return "pto-cv-insert-bridge";
  }
  StringRef getDescription() const override {
    return "Insert GM workspace bridges for cross-domain data dependencies";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto bridges = findBridgePoints(func);

    if (bridges.empty())
      return; // no cross-domain deps

    // TODO: find workspace param and insert bridges (Task 7)
    llvm::errs() << "[CVInsertBridge] Found " << bridges.size()
                 << " bridge points\n";
  }

private:
  /// Find the enclosing section op for an operation, or nullptr.
  Operation *getEnclosingSection(Operation *op) {
    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<SectionCubeOp, SectionVectorOp>(parent))
        return parent;
      parent = parent->getParentOp();
    }
    return nullptr;
  }

  /// Collect all cross-domain bridge points.
  SmallVector<BridgePoint> findBridgePoints(func::FuncOp func) {
    // Map: (producerValue, consumerSection) -> index in bridges
    llvm::DenseMap<std::pair<Value, Operation *>, unsigned> bridgeMap;
    SmallVector<BridgePoint> bridges;

    func.walk([&](Operation *consumerOp) {
      Operation *consumerSec = getEnclosingSection(consumerOp);
      if (!consumerSec)
        return; // not inside a section

      for (OpOperand &operand : consumerOp->getOpOperands()) {
        Value val = operand.get();
        Operation *defOp = val.getDefiningOp();
        if (!defOp)
          continue;

        Operation *producerSec = getEnclosingSection(defOp);
        if (!producerSec || producerSec == consumerSec)
          continue; // same section or not in a section

        // Cross-domain dependency found
        auto key = std::make_pair(val, consumerSec);
        auto it = bridgeMap.find(key);
        if (it != bridgeMap.end()) {
          bridges[it->second].consumerUses.push_back(&operand);
        } else {
          bridgeMap[key] = bridges.size();
          bridges.push_back({val, producerSec, {&operand}, consumerSec});
        }
      }
    });

    return bridges;
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createCVInsertBridgePass() {
  return std::make_unique<CVInsertBridgePass>();
}
} // namespace pto
} // namespace mlir
