#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

/// Compute domain classification for an operation.
enum class ComputeDomain { CUBE, VECTOR, SHARED };

class CVClassifyAndSplitPass
    : public PassWrapper<CVClassifyAndSplitPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVClassifyAndSplitPass)

  StringRef getArgument() const override {
    return "pto-cv-classify-and-split";
  }
  StringRef getDescription() const override {
    return "Classify ops into Cube/Vector domains and split into sections";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Block &body = func.front();

    // Step 1: Classify all top-level ops
    struct OpEntry {
      Operation *op;
      ComputeDomain domain;
    };
    SmallVector<OpEntry> entries;
    for (Operation &op : body.without_terminator()) {
      // Skip existing sections
      if (isa<SectionCubeOp, SectionVectorOp>(&op)) {
        entries.push_back({&op, ComputeDomain::SHARED});
        continue;
      }
      // For scf.for, recursively determine domain
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        entries.push_back({&op, classifyRegion(forOp.getBody())});
        continue;
      }
      entries.push_back({&op, classifyOp(&op)});
    }

    // Step 2: Group consecutive same-domain ops and wrap in sections
    OpBuilder builder(func.getContext());
    unsigned i = 0;
    while (i < entries.size()) {
      ComputeDomain domain = entries[i].domain;

      // SHARED ops: if it's a mixed-domain scf.for, split it
      if (domain == ComputeDomain::SHARED) {
        if (auto forOp = dyn_cast<scf::ForOp>(entries[i].op))
          splitMixedLoop(forOp, builder);
        ++i;
        continue;
      }

      // Find the end of consecutive same-domain ops
      unsigned j = i + 1;
      while (j < entries.size() && entries[j].domain == domain)
        ++j;

      // Create section op before the first op in this run
      builder.setInsertionPoint(entries[i].op);
      Operation *sectionOp = nullptr;
      if (domain == ComputeDomain::CUBE)
        sectionOp = builder.create<SectionCubeOp>(entries[i].op->getLoc());
      else
        sectionOp = builder.create<SectionVectorOp>(entries[i].op->getLoc());

      // Ensure the section has a block (builder doesn't create one)
      Region &region = sectionOp->getRegion(0);
      if (region.empty())
        region.emplaceBlock();
      Block &sectionBody = region.front();
      for (unsigned k = i; k < j; ++k) {
        entries[k].op->moveBefore(&sectionBody, sectionBody.end());
      }

      i = j;
    }
  }

private:
  /// Get the address space from a memref type, if present.
  static std::optional<pto::AddressSpace> getAddrSpace(Type t) {
    auto mr = dyn_cast<MemRefType>(t);
    if (!mr)
      return std::nullopt;
    auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace());
    if (!attr)
      return std::nullopt;
    return attr.getAddressSpace();
  }

  /// Check if any operand or result has an address space in the given set.
  static bool hasAddrSpaceIn(Operation *op,
                             ArrayRef<pto::AddressSpace> spaces) {
    auto check = [&](Type t) -> bool {
      auto as = getAddrSpace(t);
      return as && llvm::is_contained(spaces, *as);
    };
    for (Type t : op->getOperandTypes())
      if (check(t))
        return true;
    for (Type t : op->getResultTypes())
      if (check(t))
        return true;
    return false;
  }

  /// Classify a single operation into its compute domain.
  ///
  /// Classification priority:
  /// 1. Section ops → SHARED (already classified)
  /// 2. OpPipeInterface → use getPipe() to determine domain:
  ///    - PIPE_M → CUBE (matmul/gemv)
  ///    - PIPE_V → VECTOR (element-wise, transpose, etc.)
  ///    - PIPE_MTE1 → CUBE (L1→L0 data movement)
  ///    - PIPE_MTE2/MTE3 → address-space fallback (shared DMA)
  ///    - PIPE_FIX → CUBE (ACC→GM store)
  /// 3. Address space match as fallback for DMA ops and non-pipe ops
  /// 4. SHARED for infrastructure ops (arith, scf, etc.)
  ComputeDomain classifyOp(Operation *op) {
    // Already in a section — skip
    if (isa<SectionCubeOp, SectionVectorOp>(op))
      return ComputeDomain::SHARED;

    // 1. Use OpPipeInterface for ops that declare their pipe
    if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
      auto pipe = pipeOp.getPipe();
      switch (pipe) {
      case pto::PIPE::PIPE_M:
        return ComputeDomain::CUBE;
      case pto::PIPE::PIPE_V:
      case pto::PIPE::PIPE_V2:
        return ComputeDomain::VECTOR;
      case pto::PIPE::PIPE_MTE1:
      case pto::PIPE::PIPE_FIX:
        return ComputeDomain::CUBE;
      default:
        // PIPE_MTE2, PIPE_MTE3, PIPE_S, etc. — fall through to address space
        break;
      }
    }

    // 2. Address space match — Cube
    static const pto::AddressSpace cubeSpaces[] = {
        pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
        pto::AddressSpace::ACC, pto::AddressSpace::MAT,
        pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
    if (hasAddrSpaceIn(op, cubeSpaces))
      return ComputeDomain::CUBE;

    // 2. Address space match — Vector
    static const pto::AddressSpace vecSpaces[] = {pto::AddressSpace::VEC};
    if (hasAddrSpaceIn(op, vecSpaces))
      return ComputeDomain::VECTOR;

    // 3. Fallback
    return ComputeDomain::SHARED;
  }

  /// Split a mixed-domain scf.for into two loops (cube + vector),
  /// each wrapped in the appropriate section.
  void splitMixedLoop(scf::ForOp forOp, OpBuilder &builder) {
    builder.setInsertionPoint(forOp);

    // Clone the loop twice
    auto cubeLoop = cast<scf::ForOp>(builder.clone(*forOp));
    auto vecLoop = cast<scf::ForOp>(builder.clone(*forOp));

    // Remove non-cube ops from cubeLoop body
    filterLoopBody(cubeLoop, ComputeDomain::CUBE);
    // Remove non-vector ops from vecLoop body
    filterLoopBody(vecLoop, ComputeDomain::VECTOR);

    // Wrap in sections (ensure block exists)
    auto cubeSec = builder.create<SectionCubeOp>(forOp->getLoc());
    if (cubeSec.getBody().empty())
      cubeSec.getBody().emplaceBlock();
    cubeLoop->moveBefore(&cubeSec.getBody().front(),
                         cubeSec.getBody().front().end());

    auto vecSec = builder.create<SectionVectorOp>(forOp->getLoc());
    if (vecSec.getBody().empty())
      vecSec.getBody().emplaceBlock();
    vecLoop->moveBefore(&vecSec.getBody().front(),
                        vecSec.getBody().front().end());

    // Erase original loop
    forOp->erase();
  }

  /// Remove ops from loop body that don't belong to the target domain.
  void filterLoopBody(scf::ForOp loop, ComputeDomain keep) {
    SmallVector<Operation *> toErase;
    for (Operation &op : loop.getBody()->without_terminator()) {
      ComputeDomain d = classifyOp(&op);
      if (d != keep && d != ComputeDomain::SHARED)
        toErase.push_back(&op);
    }
    for (Operation *op : llvm::reverse(toErase))
      op->erase();
  }

  /// Classify a region: if all ops are same domain, return that; else SHARED.
  ComputeDomain classifyRegion(Block *block) {
    std::optional<ComputeDomain> result;
    for (Operation &op : block->without_terminator()) {
      ComputeDomain d = classifyOp(&op);
      if (d == ComputeDomain::SHARED)
        continue;
      if (!result)
        result = d;
      else if (*result != d)
        return ComputeDomain::SHARED; // mixed
    }
    return result.value_or(ComputeDomain::SHARED);
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createCVClassifyAndSplitPass() {
  return std::make_unique<CVClassifyAndSplitPass>();
}
} // namespace pto
} // namespace mlir
