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
    // TODO: implement in subsequent tasks
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
  ComputeDomain classifyOp(Operation *op) {
    // Already in a section — skip
    if (isa<SectionCubeOp, SectionVectorOp>(op))
      return ComputeDomain::SHARED;

    // 1. Op type match — Cube ops
    if (isa<MatmulOp, MatmulAccOp, TMatmulOp, TMatmulAccOp,
            TMatmulBiasOp, TMatmulMxOp, TMatmulMxAccOp,
            TMatmulMxBiasOp, TGemvOp, TGemvAccOp, TGemvBiasOp>(op))
      return ComputeDomain::CUBE;

    // 1. Op type match — Vector ops
    if (isa<AddFOp, AddFDpsOp, TransOp, TTransOp, TMovOp, MovOp>(op))
      return ComputeDomain::VECTOR;

    // 2. Address space match — Cube
    static const pto::AddressSpace cubeSpaces[] = {
        pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
        pto::AddressSpace::ACC};
    if (hasAddrSpaceIn(op, cubeSpaces))
      return ComputeDomain::CUBE;

    // 2. Address space match — Vector
    static const pto::AddressSpace vecSpaces[] = {pto::AddressSpace::VEC};
    if (hasAddrSpaceIn(op, vecSpaces))
      return ComputeDomain::VECTOR;

    // 3. Fallback
    return ComputeDomain::SHARED;
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
