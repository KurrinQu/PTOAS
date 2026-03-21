#include "FusionAnalysis.h"

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PRINTPREFUSIONANALYSIS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static StringRef stringifyComputeFamily(pto::FusionComputeFamily family) {
  switch (family) {
  case pto::FusionComputeFamily::Elementwise:
    return "elementwise";
  case pto::FusionComputeFamily::ScalarExpand:
    return "scalar_expand";
  case pto::FusionComputeFamily::RowBroadcastBinary:
    return "row_broadcast_binary";
  case pto::FusionComputeFamily::ReduceRow:
    return "reduce_row";
  case pto::FusionComputeFamily::ReduceCol:
    return "reduce_col";
  case pto::FusionComputeFamily::Unknown:
    return "unknown";
  }
  return "unknown";
}

static StringRef stringifyIterationProof(pto::IterationDomainProof proof) {
  switch (proof) {
  case pto::IterationDomainProof::Proven:
    return "proven";
  case pto::IterationDomainProof::Unproven:
    return "unproven";
  }
  return "unproven";
}

static StringRef stringifyUnprovenReason(
    pto::IterationDomainUnprovenReason reason) {
  switch (reason) {
  case pto::IterationDomainUnprovenReason::None:
    return "none";
  case pto::IterationDomainUnprovenReason::MissingTileDomain:
    return "missing_tile_domain";
  case pto::IterationDomainUnprovenReason::DynamicShape:
    return "dynamic_shape";
  case pto::IterationDomainUnprovenReason::InconsistentShape:
    return "inconsistent_shape";
  }
  return "missing_tile_domain";
}

static void appendIndexList(llvm::raw_ostream &os, ArrayRef<unsigned> values) {
  os << "[";
  for (auto [idx, value] : llvm::enumerate(values)) {
    if (idx)
      os << ", ";
    os << value;
  }
  os << "]";
}

static void appendOptionalIndex(llvm::raw_ostream &os,
                                std::optional<unsigned> value) {
  if (!value) {
    os << "<none>";
    return;
  }
  os << *value;
}

static void appendDomain(llvm::raw_ostream &os,
                         const pto::IterationDomainInfo &info) {
  auto printDim = [&](int64_t dim) {
    if (dim == ShapedType::kDynamic)
      os << "?";
    else
      os << dim;
  };

  os << "(";
  printDim(info.vRow);
  os << "x";
  printDim(info.vCol);
  os << ")";
}

static std::string makeExternalValueLabel(unsigned ordinal) {
  return (llvm::Twine("external#") + llvm::Twine(ordinal)).str();
}

static std::string makeBoundaryValueLabel(unsigned ordinal) {
  return (llvm::Twine("boundary#") + llvm::Twine(ordinal)).str();
}

static DenseMap<Value, std::string>
buildValueLabels(Block &block, const pto::FusionBlockAnalysis &analysis) {
  DenseMap<Value, std::string> labels;
  unsigned externalOrdinal = 0;
  unsigned boundaryOrdinal = 0;

  for (const pto::FusionComputeNode &node : analysis.computeNodes) {
    for (auto [idx, output] : llvm::enumerate(node.semantics.tileOutputs))
      labels.try_emplace(
          output,
          (llvm::Twine("node") + llvm::Twine(node.id) + ".out" +
           llvm::Twine(idx))
              .str());
  }

  for (Operation &op : block) {
    FailureOr<pto::FusionOpSemantics> semanticsOr =
        pto::getFusionOpSemantics(&op);
    if (failed(semanticsOr))
      continue;

    if (semanticsOr->kind == pto::FusionOpKind::LocalBoundary) {
      for (Value input : semanticsOr->tileInputs)
        if (!labels.count(input))
          labels.try_emplace(input, makeExternalValueLabel(externalOrdinal++));
      for (Value output : semanticsOr->tileOutputs)
        if (!labels.count(output))
          labels.try_emplace(output, makeBoundaryValueLabel(boundaryOrdinal++));
      continue;
    }

    for (Value input : semanticsOr->tileInputs)
      if (!labels.count(input))
        labels.try_emplace(input, makeExternalValueLabel(externalOrdinal++));
  }

  return labels;
}

static void printLocalBoundaries(llvm::raw_ostream &os, Block &block,
                                 DenseMap<Value, std::string> &valueLabels) {
  unsigned boundaryId = 0;
  for (Operation &op : block) {
    FailureOr<pto::FusionOpSemantics> semanticsOr =
        pto::getFusionOpSemantics(&op);
    if (failed(semanticsOr) ||
        semanticsOr->kind != pto::FusionOpKind::LocalBoundary)
      continue;

    os << "    local_boundary[" << boundaryId++ << "] op=" << semanticsOr->opName
       << " inputs=[";
    for (auto [idx, input] : llvm::enumerate(semanticsOr->tileInputs)) {
      if (idx)
        os << ", ";
      os << valueLabels.lookup(input);
    }
    os << "] outputs=[";
    for (auto [idx, output] : llvm::enumerate(semanticsOr->tileOutputs)) {
      if (idx)
        os << ", ";
      os << valueLabels.lookup(output);
    }
    os << "]\n";
  }
}

struct PrintPreFusionAnalysisPass
    : public pto::impl::PrintPreFusionAnalysisBase<PrintPreFusionAnalysisPass> {
  using pto::impl::PrintPreFusionAnalysisBase<
      PrintPreFusionAnalysisPass>::PrintPreFusionAnalysisBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    FailureOr<pto::PreFusionAnalysisResult> resultOr =
        pto::buildPreFusionAnalysis(func);
    if (failed(resultOr)) {
      signalPassFailure();
      return;
    }

    llvm::raw_ostream &os = llvm::outs();
    os << "PreFusionAnalysis @" << func.getSymName() << "\n";

    for (auto [blockIndex, blockAnalysis] : llvm::enumerate(resultOr->blocks)) {
      os << "  block[" << blockIndex << "]\n";
      DenseMap<Value, std::string> valueLabels =
          buildValueLabels(*blockAnalysis.block, blockAnalysis);
      printLocalBoundaries(os, *blockAnalysis.block, valueLabels);

      for (const pto::IterationDomainClass &klass :
           blockAnalysis.iterationDomainClasses) {
        os << "    domain_class[" << klass.id << "] domain=";
        appendDomain(os, klass.info);
        os << " proof=" << stringifyIterationProof(klass.info.proof)
           << " reason="
           << stringifyUnprovenReason(klass.info.unprovenReason)
           << " members=";
        appendIndexList(os, klass.members);
        os << "\n";
      }

      for (const pto::FusionComputeNode &node : blockAnalysis.computeNodes) {
        os << "    compute[" << node.id << "] op=" << node.semantics.opName
           << " family="
           << stringifyComputeFamily(node.semantics.computeFamily)
           << " domain_class=" << node.iterationDomainClass << " inputs=[";
        for (auto [idx, input] : llvm::enumerate(node.semantics.tileInputs)) {
          if (idx)
            os << ", ";
          os << valueLabels.lookup(input);
        }
        os << "] outputs=[";
        for (auto [idx, output] : llvm::enumerate(node.semantics.tileOutputs)) {
          if (idx)
            os << ", ";
          os << valueLabels.lookup(output);
        }
        os << "] incoming=";
        appendIndexList(os, node.incomingEdges);
        os << " outgoing=";
        appendIndexList(os, node.outgoingEdges);
        os << "\n";
      }

      for (auto [edgeIndex, edge] : llvm::enumerate(blockAnalysis.edges)) {
        os << "    edge[" << edgeIndex << "] producer=" << edge.producerNode
           << " consumer=" << edge.consumerNode
           << " value=" << valueLabels.lookup(edge.value) << "\n";
      }

      for (const pto::FusionValueLiveness &live : blockAnalysis.liveness) {
        os << "    liveness value=" << valueLabels.lookup(live.value)
           << " producer=";
        appendOptionalIndex(os, live.producerNode);
        os << " consumers=";
        appendIndexList(os, live.consumerNodes);
        os << " last_local_consumer=";
        appendOptionalIndex(os, live.lastLocalConsumer);
        os << " external_users=" << (live.hasExternalUsers ? "true" : "false")
           << " escapes_block=" << (live.escapesBlock ? "true" : "false")
           << " boundary_users="
           << (live.hasLocalBoundaryUsers ? "true" : "false")
           << " hard_boundary_users="
           << (live.hasLocalHardBoundaryUsers ? "true" : "false") << "\n";
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPrintPreFusionAnalysisPass() {
  return std::make_unique<PrintPreFusionAnalysisPass>();
}
