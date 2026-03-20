#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVERIFYTFREE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  Operation *current = op;
  while (current && current->getBlock() != block) {
    Region *parentRegion = current->getParentRegion();
    if (!parentRegion)
      return nullptr;
    current = parentRegion->getParentOp();
  }
  return current;
}

static LogicalResult verifyNoTileUsesAfterTFree(TPopInternalOp tpopOp,
                                                TFreeInternalOp tfreeOp) {
  Value tile = tpopOp.getTile();
  Block *block = tpopOp->getBlock();

  for (OpOperand &use : tile.getUses()) {
    Operation *topLevelOwner = getTopLevelAncestorInBlock(use.getOwner(), block);
    if (!topLevelOwner) {
      return tpopOp.emitOpError(
          "borrowed tile uses must stay in the same parent block as the producing tpop");
    }
    if (tfreeOp->isBeforeInBlock(topLevelOwner)) {
      return tpopOp.emitOpError(
          "borrowed tile must not be used after its matched tfree");
    }
  }

  return success();
}

static bool isInsideExplicitSection(Operation *op) {
  return op->getParentOfType<SectionCubeOp>() ||
         op->getParentOfType<SectionVectorOp>();
}

static LogicalResult verifyBlockFIFO(Block &block) {
  // TPOP/TFREE matching stays block-local: each pipe keeps a FIFO queue of
  // outstanding pops, and each TFREE releases the oldest pop in that block.
  DenseMap<Value, SmallVector<TPopInternalOp>> outstandingByPipe;

  for (Operation &op : block) {
    if (auto tpopOp = dyn_cast<TPopInternalOp>(&op)) {
      if (isInsideExplicitSection(tpopOp))
        outstandingByPipe[tpopOp.getPipeHandle()].push_back(tpopOp);
      continue;
    }

    if (auto tfreeOp = dyn_cast<TFreeInternalOp>(&op)) {
      if (!isInsideExplicitSection(tfreeOp))
        continue;

      auto queueIt = outstandingByPipe.find(tfreeOp.getPipeHandle());
      if (queueIt == outstandingByPipe.end() || queueIt->second.empty()) {
        return tfreeOp.emitOpError(
            "requires a prior outstanding tpop on the same pipe");
      }

      // TFREE releases the oldest outstanding pop on the same logical pipe.
      TPopInternalOp matchedTPop = queueIt->second.front();
      queueIt->second.erase(queueIt->second.begin());
      if (failed(verifyNoTileUsesAfterTFree(matchedTPop, tfreeOp)))
        return failure();
      continue;
    }

    for (Region &region : op.getRegions()) {
      for (Block &nestedBlock : region) {
        if (failed(verifyBlockFIFO(nestedBlock)))
          return failure();
      }
    }
  }

  for (auto &it : outstandingByPipe) {
    if (it.second.empty())
      continue;
    return it.second.front().emitOpError("requires an explicit matching tfree");
  }

  return success();
}

struct PTOVerifyTFreePass
    : public mlir::pto::impl::PTOVerifyTFreeBase<PTOVerifyTFreePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    for (Block &block : funcOp.getBody()) {
      if (failed(verifyBlockFIFO(block))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVerifyTFreePass() {
  return std::make_unique<PTOVerifyTFreePass>();
}
