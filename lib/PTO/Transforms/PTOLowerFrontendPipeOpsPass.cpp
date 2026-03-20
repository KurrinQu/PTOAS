#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERFRONTENDPIPEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct FrontendPipeHandles {
  Value c2vPipe;
  Value v2cPipe;
  Operation *anchorOp = nullptr;
};

static PTOArch getTargetArch(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return PTOArch::A3;

  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue().equals_insensitive("a5"))
    return PTOArch::A5;
  return PTOArch::A3;
}

static std::optional<FunctionKernelKind>
getFunctionKernelKind(func::FuncOp funcOp) {
  auto kernelKindAttr = funcOp->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;
  return kernelKindAttr.getKernelKind();
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerFrontendInitOp(InitOpT initOp, IRRewriter &rewriter) {
  FrontendPipeHandles handles;
  Location loc = initOp.getLoc();
  MLIRContext *ctx = initOp.getContext();
  auto pipeTy = PipeType::get(ctx);
  PTOArch arch = getTargetArch(initOp.getOperation());

  auto createPipe = [&](int8_t dirMask, int32_t slotNum,
                        Value localAddr) -> FailureOr<Value> {
    auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
    auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
    auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);

    if (arch == PTOArch::A5) {
      auto pipe = rewriter.create<InitializeL2LPipeOp>(
          loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
          localAddr);
      return pipe.getPipe();
    }

    if (!initOp.getGmSlotBuffer()) {
      initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
      return failure();
    }

    auto localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
    auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
        loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
        IntegerAttr{}, initOp.getGmSlotBuffer(), localAddr);
    return pipe.getPipe();
  };

  switch (initOp.getDirMask()) {
  case 1: {
    auto pipeOr =
        createPipe(/*dirMask=*/1, /*slotNum=*/8, initOp.getC2vConsumerBuf());
    if (failed(pipeOr))
      return failure();
    handles.c2vPipe = *pipeOr;
    handles.anchorOp = handles.c2vPipe.getDefiningOp();
    break;
  }
  case 2: {
    auto pipeOr =
        createPipe(/*dirMask=*/2, /*slotNum=*/8, initOp.getV2cConsumerBuf());
    if (failed(pipeOr))
      return failure();
    handles.v2cPipe = *pipeOr;
    handles.anchorOp = handles.v2cPipe.getDefiningOp();
    break;
  }
  case 3: {
    auto c2vPipeOr =
        createPipe(/*dirMask=*/1, /*slotNum=*/4, initOp.getC2vConsumerBuf());
    if (failed(c2vPipeOr))
      return failure();
    handles.c2vPipe = *c2vPipeOr;
    handles.anchorOp = handles.c2vPipe.getDefiningOp();

    auto v2cPipeOr =
        createPipe(/*dirMask=*/2, /*slotNum=*/4, initOp.getV2cConsumerBuf());
    if (failed(v2cPipeOr))
      return failure();
    handles.v2cPipe = *v2cPipeOr;
    break;
  }
  default:
    break;
  }

  return handles;
}

static FailureOr<FrontendPipeHandles> lowerInitIfPresent(func::FuncOp funcOp,
                                                         IRRewriter &rewriter) {
  FrontendPipeHandles handles;
  InitializePipeOp initOp;
  unsigned initCount = 0;

  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<InitializePipeOp>(op)) {
      ++initCount;
      if (!initOp)
        initOp = init;
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (initCount > 1) {
    funcOp.emitOpError("requires at most one pto.initialize_pipe");
    return failure();
  }

  if (!initOp)
    return handles;

  rewriter.setInsertionPoint(initOp);
  auto loweredOr = lowerFrontendInitOp(initOp, rewriter);
  if (failed(loweredOr))
    return failure();
  handles = *loweredOr;
  rewriter.eraseOp(initOp);

  return handles;
}

static bool hasFrontendPipeOps(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<InitializePipeOp, TPushOp, TPopOp, TFreeOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandles &handles,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TPushOp, TPopOp, TFreeOp>(op))
      frontendOps.push_back(op);
  });

  for (Operation *op : frontendOps) {
    if (!handles.anchorOp) {
      op->emitOpError(
          "requires a frontend initialize_pipe op in the same function");
      return failure();
    }
    if (!dom.dominates(handles.anchorOp, op)) {
      op->emitOpError("requires a dominating frontend initialize_pipe op");
      return failure();
    }

    rewriter.setInsertionPoint(op);
    auto kernelKind = getFunctionKernelKind(funcOp);
    if (!kernelKind) {
      op->emitOpError(
          "requires the containing function to carry pto.kernel_kind");
      return failure();
    }
    bool isCube = *kernelKind == FunctionKernelKind::Cube;
    Value pushPipe = isCube ? handles.c2vPipe : handles.v2cPipe;
    Value popPipe = isCube ? handles.v2cPipe : handles.c2vPipe;

    if (auto push = dyn_cast<TPushOp>(op)) {
      if (!pushPipe) {
        op->emitOpError("requires the dominating initialize_pipe op to enable "
                        "ring-buffer push");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushInternalOp>(
          push, push.getTile(), pushPipe, push.getSplitAttr());
      continue;
    }

    if (auto pop = dyn_cast<TPopOp>(op)) {
      if (!popPipe) {
        op->emitOpError("requires the dominating initialize_pipe op to enable "
                        "ring-buffer pop/free");
        return failure();
      }
      auto decl =
          rewriter.create<DeclareTileOp>(pop.getLoc(), pop.getTile().getType());
      rewriter.create<TPopInternalOp>(pop.getLoc(), decl.getTile(), popPipe,
                                      pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    auto free = cast<TFreeOp>(op);
    if (!popPipe) {
      op->emitOpError("requires the dominating initialize_pipe op to enable "
                      "ring-buffer pop/free");
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFreeInternalOp>(free, popPipe,
                                                 free.getSplitAttr());
  }

  return success();
}

struct PTOLowerFrontendPipeOpsPass
    : public mlir::pto::impl::PTOLowerFrontendPipeOpsBase<
          PTOLowerFrontendPipeOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasFrontendPipeOps(funcOp))
      return;

    IRRewriter rewriter(funcOp.getContext());
    auto loweredOr = lowerInitIfPresent(funcOp, rewriter);
    if (failed(loweredOr)) {
      signalPassFailure();
      return;
    }

    if (failed(lowerFrontendDataOps(funcOp, *loweredOr, rewriter)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerFrontendPipeOpsPass() {
  return std::make_unique<PTOLowerFrontendPipeOpsPass>();
}
