//===- PTOVPTOVersionSelection.cpp - VPTO version selection pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOLowering.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOVERSIONSELECTION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
namespace {

static FailureOr<PTOLoweringChoiceAttr>
selectPTOLoweringChoiceForOp(Operation *op) {
  if (!op->getBlock()) {
    op->emitOpError()
        << "VPTO version selection requires candidate op to remain attached to "
           "a block";
    return failure();
  }

  const bool inFusionRegion = op->getParentOfType<FusionRegionOp>() != nullptr;
  VPTOLoweringStrategy strategy = inFusionRegion
                                      ? VPTOLoweringStrategy::NoPostUpdate
                                      : VPTOLoweringStrategy::PostUpdate;
  PTOLoweringChoiceAttr choice = PTOLoweringChoiceAttr::get(
      op->getContext(), convertVPTOLoweringStrategyToPTOUpdateMode(strategy),
      PTOLoopShape::TwoD);
  if (failed(validateVPTOLoweringChoiceAttr(op, choice)))
    return failure();
  return choice;
}

static bool isVPTOCandidatePTOOp(Operation *op) {
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
static void walkVPTOCandidatePTOOps(func::FuncOp func, CallbackT &&callback) {
  SmallVector<Operation *> fusionRegionCandidates;
  SmallVector<Operation *> residualCandidates;

  func.walk([&](FusionRegionOp fusionRegion) {
    fusionRegion.walk([&](Operation *op) {
      if (op == fusionRegion.getOperation())
        return;
      if (isVPTOCandidatePTOOp(op))
        fusionRegionCandidates.push_back(op);
    });
  });

  func.walk([&](Operation *op) {
    if (isa<FusionRegionOp, YieldOp>(op))
      return;
    if (op->getParentOfType<FusionRegionOp>())
      return;
    if (isVPTOCandidatePTOOp(op))
      residualCandidates.push_back(op);
  });

  for (Operation *op : fusionRegionCandidates)
    callback(op);
  for (Operation *op : residualCandidates)
    callback(op);
}

struct PTOVPTOVersionSelectionPass
    : public impl::PTOVPTOVersionSelectionBase<PTOVPTOVersionSelectionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOVPTOVersionSelectionPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool sawFailure = false;
    walkVPTOCandidatePTOOps(func, [&](Operation *op) {
      FailureOr<PTOLoweringChoiceAttr> choice = selectPTOLoweringChoiceForOp(op);
      if (failed(choice)) {
        sawFailure = true;
        return;
      }
      op->setAttr(kPTOLoweringChoiceAttrName, *choice);
    });
    if (sawFailure) {
      signalPassFailure();
      return;
    }
    func->setAttr(kPTOVersionSelectionAppliedAttrName, UnitAttr::get(&getContext()));
  }
};

} // namespace

std::unique_ptr<Pass> createPTOVPTOVersionSelectionPass() {
  return std::make_unique<PTOVPTOVersionSelectionPass>();
}

} // namespace pto
} // namespace mlir
