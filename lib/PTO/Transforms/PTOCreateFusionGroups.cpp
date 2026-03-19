#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <functional>

#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCREATEFUSIONGROUPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct FusibleOpInfo {
  Operation *op = nullptr;
  Value dst;
  SmallVector<Value, 4> inputs;
  SmallVector<Value, 4> tileInputs;
};

static bool isSupportedFusionOpName(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tmul", "tdiv", "tadd", "tsub", "tmax", "tmin", true)
      .Cases("tmuls", "tdivs", "tadds", "tsubs", "tmaxs", "tmins", true)
      .Default(false);
}

static FailureOr<FusibleOpInfo> getFusibleOpInfo(Operation *op) {
  auto oplibIface = dyn_cast<pto::OpLibOpInterface>(op);
  auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
  if (!oplibIface || !dpsIface)
    return failure();

  FailureOr<pto::OpLibMatchDescriptor> descOr =
      oplibIface.getOpLibMatchDescriptor();
  if (failed(descOr))
    return failure();

  const pto::OpLibMatchDescriptor &desc = *descOr;
  if (!isSupportedFusionOpName(desc.opName) || desc.operands.size() < 2 ||
      desc.operands.size() != desc.operandRoles.size())
    return failure();

  OperandRange dpsInits = dpsIface.getDpsInits();
  if (dpsInits.size() != 1)
    return failure();

  Value dst = dpsInits.front();
  if (desc.operands.back() != dst)
    return failure();
  if (desc.operandRoles.back() != static_cast<int64_t>(pto::OpLibArgRole::Tile))
    return failure();

  FusibleOpInfo info;
  info.op = op;
  info.dst = dst;
  info.inputs.append(ArrayRef<Value>(desc.operands).drop_back().begin(),
                     ArrayRef<Value>(desc.operands).drop_back().end());

  for (auto [input, role] : llvm::zip(
           info.inputs, ArrayRef<int64_t>(desc.operandRoles).drop_back())) {
    if (role == static_cast<int64_t>(pto::OpLibArgRole::Tile))
      info.tileInputs.push_back(input);
  }

  return info;
}

struct PTOCreateFusionGroupsPass
    : public pto::impl::PTOCreateFusionGroupsBase<PTOCreateFusionGroupsPass> {
  void runOnOperation() override {
    // This pass runs after PlanMemory/InsertSync in the pipeline. Grouping is
    // strict-contiguous; intervening ops (including sync) will split chains.
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    int64_t nextGroupId = 0;

    std::function<void(Region &)> processRegion = [&](Region &region) {
      for (Block &block : region.getBlocks()) {
        SmallVector<FusibleOpInfo, 8> chain;

        auto flushChain = [&]() {
          if (chain.size() < 2) {
            chain.clear();
            return;
          }
          int64_t gid = nextGroupId++;
          for (auto it : llvm::enumerate(chain)) {
            int64_t idx = static_cast<int64_t>(it.index());
            Operation *op = it.value().op;
            op->setAttr("pto.fusion.group_id",
                        IntegerAttr::get(IntegerType::get(ctx, 64), gid));
            op->setAttr("pto.fusion.order",
                        IntegerAttr::get(IntegerType::get(ctx, 64), idx));
          }
          chain.clear();
        };

        for (Operation &op : block.getOperations()) {
          Operation *cur = &op;
          FailureOr<FusibleOpInfo> curInfoOr = getFusibleOpInfo(cur);

          if (failed(curInfoOr)) {
            flushChain();
            for (Region &nested : cur->getRegions())
              processRegion(nested);
            continue;
          }

          FusibleOpInfo curInfo = *curInfoOr;

          if (chain.empty()) {
            chain.push_back(std::move(curInfo));
            continue;
          }

          Value prevDst = chain.back().dst;
          bool dependsOnPrev = llvm::is_contained(curInfo.tileInputs, prevDst);
          if (!dependsOnPrev) {
            flushChain();
            chain.push_back(std::move(curInfo));
            continue;
          }
          chain.push_back(std::move(curInfo));
        }
        flushChain();
      }
    };

    processRegion(func.getRegion());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCreateFusionGroupsPass() {
  return std::make_unique<PTOCreateFusionGroupsPass>();
}
