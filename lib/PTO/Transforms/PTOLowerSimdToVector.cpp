#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWERSIMDTOVECTOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kSimdCoreSlotAttr = "pto.simd.core_slot";

static Value buildZeroPassthru(OpBuilder &builder, Location loc,
                               VectorType vecTy) {
  Attribute zeroElem = builder.getZeroAttr(vecTy.getElementType());
  auto zeroVec = DenseElementsAttr::get(vecTy, zeroElem);
  return builder.create<arith::ConstantOp>(loc, vecTy, zeroVec);
}

struct PTOLowerSimdToVectorPass
    : public pto::impl::PTOLowerSimdToVectorBase<PTOLowerSimdToVectorPass> {
  using pto::impl::PTOLowerSimdToVectorBase<
      PTOLowerSimdToVectorPass>::PTOLowerSimdToVectorBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() || func.empty())
        continue;

      SmallVector<Operation *, 64> worklist;
      func.walk([&](Operation *op) { worklist.push_back(op); });

      for (Operation *op : llvm::reverse(worklist)) {
        if (!op || !op->getBlock())
          continue;

        if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
          OpBuilder b(pred);
          auto maskTy = cast<VectorType>(pred.getMask().getType());
          auto newPred = b.create<vector::CreateMaskOp>(
              pred.getLoc(), maskTy, ValueRange{pred.getActiveCount()});
          pred.replaceAllUsesWith(newPred.getResult());
          pred.erase();
          continue;
        }

        if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
          OpBuilder b(load);
          auto vecTy = cast<VectorType>(load.getValue().getType());
          Value passthru = buildZeroPassthru(b, load.getLoc(), vecTy);
          auto masked = b.create<vector::MaskedLoadOp>(
              load.getLoc(), vecTy, load.getSrc(), ValueRange{load.getOffset()},
              load.getMask(), passthru);
          load.replaceAllUsesWith(masked.getResult());
          load.erase();
          continue;
        }

        if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
          OpBuilder b(store);
          b.create<vector::MaskedStoreOp>(store.getLoc(), store.getDst(),
                                          ValueRange{store.getOffset()},
                                          store.getMask(), store.getValue());
          store.erase();
          continue;
        }

        if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
          OpBuilder b(loadPU);
          auto vecTy = cast<VectorType>(loadPU.getValue().getType());
          Value passthru = buildZeroPassthru(b, loadPU.getLoc(), vecTy);
          Value value =
              b.create<vector::MaskedLoadOp>(
                   loadPU.getLoc(), vecTy, loadPU.getSrc(),
                   ValueRange{loadPU.getOffset()}, loadPU.getMask(), passthru)
                  .getResult();
          Value step = b.create<arith::ConstantIndexOp>(loadPU.getLoc(),
                                                        loadPU.getStep());
          Value next = b.create<arith::AddIOp>(loadPU.getLoc(),
                                               loadPU.getOffset(), step);
          loadPU.replaceAllUsesWith(ValueRange{value, next});
          loadPU.erase();
          continue;
        }

        if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
          OpBuilder b(storePU);
          b.create<vector::MaskedStoreOp>(storePU.getLoc(), storePU.getDst(),
                                          ValueRange{storePU.getOffset()},
                                          storePU.getMask(),
                                          storePU.getValue());
          Value step = b.create<arith::ConstantIndexOp>(storePU.getLoc(),
                                                        storePU.getStep());
          Value next = b.create<arith::AddIOp>(storePU.getLoc(),
                                               storePU.getOffset(), step);
          storePU.replaceAllUsesWith(next);
          storePU.erase();
          continue;
        }
      }

      // Bridge-only marker attribute; drop once lowering reached vector IR.
      func.walk([&](Operation *op) { op->removeAttr(kSimdCoreSlotAttr); });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerSimdToVectorPass() {
  return std::make_unique<PTOLowerSimdToVectorPass>();
}
