//===- PTOA5VMVersionSelection.cpp - A5VM version selection pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOA5VMVERSIONSELECTION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
namespace {

static FailureOr<A5VMLoweringChoiceAttr>
selectA5VMLoweringChoiceForOp(Operation *op) {
  if (!op->getBlock()) {
    op->emitOpError()
        << "A5VM version selection requires candidate op to remain attached to "
           "a block";
    return failure();
  }

  const bool inFusionRegion = op->getParentOfType<FusionRegionOp>() != nullptr;
  A5VMLoweringStrategy strategy = inFusionRegion
                                      ? A5VMLoweringStrategy::NoPostUpdate
                                      : A5VMLoweringStrategy::PostUpdate;
  A5VMLoweringChoiceAttr choice = A5VMLoweringChoiceAttr::get(
      op->getContext(), convertA5VMLoweringStrategyToUpdateMode(strategy),
      A5VMLoopShape::TwoD);
  if (failed(validateA5VMLoweringChoiceAttr(op, choice)))
    return failure();
  return choice;
}

static bool isA5VMCandidatePTOOp(Operation *op) {
  return isa<TLoadOp, TAbsOp, TAddOp, TSubOp, TMulOp, TDivOp, TMaxOp, TMinOp,
             TAndOp, TAndSOp, TOrOp, TOrSOp, TXorOp, TXorSOp, TExpOp, TLogOp,
             TSqrtOp, TRsqrtOp, TRecipOp, TNegOp, TLReluOp, TCIOp, TCvtOp,
             TCmpOp, TCmpSOp, TSelOp, TAddCOp, TAddSOp, TAddSCOp, TMinSOp,
             TSubCOp, TSubSOp, TSubSCOp, TMaxSOp, TDivSOp, TMulSOp, TSelSOp,
             TReluOp, TNotOp, TTransOp, TFillPadOp, TFillPadExpandOp,
             TRowMaxOp, TRowMinOp, TRowSumOp, TColMaxOp, TColMinOp, TColSumOp,
             TRowExpandOp, TColExpandOp, TRowExpandMulOp, TRowExpandDivOp,
             TRowExpandSubOp, TPartAddOp, TPartMaxOp, TPartMinOp, TExpandsOp,
             TGatherOp, TGatherBOp, TScatterOp, TSort32Op, TMrgSortOp,
             TStoreOp>(op);
}

template <typename CallbackT>
static void walkA5VMCandidatePTOOps(func::FuncOp func, CallbackT &&callback) {
  SmallVector<Operation *> fusionRegionCandidates;
  SmallVector<Operation *> residualCandidates;

  func.walk([&](FusionRegionOp fusionRegion) {
    fusionRegion.walk([&](Operation *op) {
      if (op == fusionRegion.getOperation())
        return;
      if (isA5VMCandidatePTOOp(op))
        fusionRegionCandidates.push_back(op);
    });
  });

  func.walk([&](Operation *op) {
    if (isa<FusionRegionOp, YieldOp>(op))
      return;
    if (op->getParentOfType<FusionRegionOp>())
      return;
    if (isA5VMCandidatePTOOp(op))
      residualCandidates.push_back(op);
  });

  for (Operation *op : fusionRegionCandidates)
    callback(op);
  for (Operation *op : residualCandidates)
    callback(op);
}

struct PTOA5VMVersionSelectionPass
    : public impl::PTOA5VMVersionSelectionBase<PTOA5VMVersionSelectionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOA5VMVersionSelectionPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool sawFailure = false;
    walkA5VMCandidatePTOOps(func, [&](Operation *op) {
      FailureOr<A5VMLoweringChoiceAttr> choice = selectA5VMLoweringChoiceForOp(op);
      if (failed(choice)) {
        sawFailure = true;
        return;
      }
      op->setAttr(kA5VMLoweringChoiceAttrName, *choice);
    });
    if (sawFailure) {
      signalPassFailure();
      return;
    }
    func->setAttr(kA5VMVersionSelectionAppliedAttrName, UnitAttr::get(&getContext()));
  }
};

} // namespace

std::unique_ptr<Pass> createPTOA5VMVersionSelectionPass() {
  return std::make_unique<PTOA5VMVersionSelectionPass>();
}

} // namespace pto
} // namespace mlir
