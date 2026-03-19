#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <string>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOOUTLINEFUSIONGROUPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

struct GroupInfo {
  int64_t groupId = -1;
  SmallVector<Operation *, 8> ops;
};

struct FusibleOpInfo {
  Operation *op = nullptr;
  Value dst;
  SmallVector<Value, 4> inputs;
};

struct GroupInterface {
  SmallVector<Value, 8> producedInOrder;
  DenseSet<Value> producedSet;
  SmallVector<Value, 8> externalInputs;
  SmallVector<Value, 8> callArgs;
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
  return info;
}

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool valueInList(ArrayRef<Value> vals, Value v) {
  return llvm::is_contained(vals, v);
}

static void appendUniqueValue(SmallVectorImpl<Value> &vals, Value v) {
  if (!valueInList(vals, v))
    vals.push_back(v);
}

static FailureOr<GroupInterface> buildGroupInterface(GroupInfo &group) {
  GroupInterface iface;

  for (Operation *op : group.ops) {
    FailureOr<FusibleOpInfo> opInfoOr = getFusibleOpInfo(op);
    if (failed(opInfoOr))
      return failure();
    appendUniqueValue(iface.producedInOrder, opInfoOr->dst);
    iface.producedSet.insert(opInfoOr->dst);
  }

  for (Operation *op : group.ops) {
    FailureOr<FusibleOpInfo> opInfoOr = getFusibleOpInfo(op);
    if (failed(opInfoOr))
      return failure();
    for (Value input : opInfoOr->inputs)
      if (!iface.producedSet.contains(input))
        appendUniqueValue(iface.externalInputs, input);
  }

  for (Value v : iface.externalInputs)
    appendUniqueValue(iface.callArgs, v);
  for (Value v : iface.producedInOrder)
    appendUniqueValue(iface.callArgs, v);

  return iface;
}

static std::string makeUniqueFusedName(SymbolTable &symbolTable,
                                       int64_t groupId, int &fusedCounter) {
  std::string fusedName = "__pto_fused_group_" + std::to_string(groupId) + "_" +
                          std::to_string(fusedCounter++);
  while (symbolTable.lookup(fusedName))
    fusedName += "_x";
  return fusedName;
}

static func::FuncOp createFusedFunc(ModuleOp module, Location loc,
                                    StringRef name, ArrayRef<Value> callArgs,
                                    int64_t groupId) {
  SmallVector<Type, 8> argTypes;
  for (Value v : callArgs)
    argTypes.push_back(v.getType());

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  auto fusedFunc = moduleBuilder.create<func::FuncOp>(
      loc, name, FunctionType::get(module.getContext(), argTypes, {}));
  fusedFunc.setPrivate();
  fusedFunc->setAttr(
      kFusionGroupIdAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), groupId));
  return fusedFunc;
}

static DenseMap<Value, Value> mapCallArgsToEntry(Block *entry,
                                                 ArrayRef<Value> callArgs) {
  DenseMap<Value, Value> valueMap;
  for (auto [orig, arg] : llvm::zip(callArgs, entry->getArguments()))
    valueMap[orig] = arg;
  return valueMap;
}

struct PTOOutlineFusionGroupsPass
    : public pto::impl::PTOOutlineFusionGroupsBase<PTOOutlineFusionGroupsPass> {
  using pto::impl::PTOOutlineFusionGroupsBase<
      PTOOutlineFusionGroupsPass>::PTOOutlineFusionGroupsBase;

  void collectGroupsInRegion(Region &region,
                             SmallVectorImpl<GroupInfo> &groups) {
    for (Block &block : region.getBlocks()) {
      GroupInfo current;
      int64_t expectedOrder = 0;

      auto flush = [&]() {
        if (current.ops.size() >= 2)
          groups.push_back(current);
        current = GroupInfo();
        expectedOrder = 0;
      };

      for (Operation &op : block.getOperations()) {
        Operation *curOp = &op;

        if (succeeded(getFusibleOpInfo(curOp)) && hasFusionGroupAttrs(curOp)) {
          auto gidAttr = curOp->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr);
          auto orderAttr = curOp->getAttrOfType<IntegerAttr>(kFusionOrderAttr);

          if (!gidAttr || !orderAttr) {
            flush();
          } else {
            int64_t gid = gidAttr.getInt();
            int64_t order = orderAttr.getInt();
            if (current.ops.empty()) {
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            } else if (current.groupId == gid && order == expectedOrder) {
              current.ops.push_back(curOp);
              ++expectedOrder;
            } else {
              flush();
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            }
          }
        } else {
          flush();
        }

        for (Region &nested : curOp->getRegions())
          collectGroupsInRegion(nested, groups);
      }

      flush();
    }
  }

  LogicalResult outlineOneGroup(ModuleOp module, SymbolTable &symbolTable,
                                GroupInfo &group, int &fusedCounter) {
    if (group.ops.empty())
      return success();

    Operation *firstOp = group.ops.front();
    Location loc = firstOp->getLoc();

    for (Operation *op : group.ops) {
      if (failed(getFusibleOpInfo(op))) {
        firstOp->emitWarning() << "fusion-group fallback: unsupported grouped "
                                  "PTO op for outlining";
        return success();
      }
    }

    FailureOr<GroupInterface> ifaceOr = buildGroupInterface(group);
    if (failed(ifaceOr)) {
      firstOp->emitWarning()
          << "fusion-group fallback: failed to build group interface, keep "
             "original ops";
      return success();
    }
    GroupInterface iface = *ifaceOr;
    if (iface.callArgs.empty()) {
      firstOp->emitWarning()
          << "fusion-group fallback: group has no external interface values, "
             "keep original ops";
      return success();
    }

    std::string fusedName =
        makeUniqueFusedName(symbolTable, group.groupId, fusedCounter);
    auto fusedFunc =
        createFusedFunc(module, loc, fusedName, iface.callArgs, group.groupId);

    Block *entry = fusedFunc.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
    IRMapping mapping;
    DenseMap<Value, Value> valueMap = mapCallArgsToEntry(entry, iface.callArgs);
    for (auto &[orig, mapped] : valueMap)
      mapping.map(orig, mapped);

    for (Operation *op : group.ops) {
      if (failed(getFusibleOpInfo(op))) {
        fusedFunc.erase();
        return success();
      }
      bodyBuilder.clone(*op, mapping);
    }

    bodyBuilder.create<func::ReturnOp>(loc);

    OpBuilder callerBuilder(firstOp);
    callerBuilder.create<func::CallOp>(loc, fusedFunc, iface.callArgs);

    for (Operation *op : llvm::reverse(group.ops))
      op->erase();

    if (debug) {
      llvm::errs() << "[op-fusion] outlined group_id=" << group.groupId
                   << " into @" << fusedFunc.getSymName() << "\n";
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);

    int fusedCounter = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;
      if (func.getSymName().starts_with("__pto_fused_group_"))
        continue;

      SmallVector<GroupInfo, 8> groups;
      collectGroupsInRegion(func.getRegion(), groups);
      if (debug && !groups.empty()) {
        llvm::errs() << "[op-fusion] found " << groups.size()
                     << " group(s) in @" << func.getSymName() << "\n";
      }

      for (GroupInfo &group : groups) {
        if (failed(outlineOneGroup(module, symbolTable, group, fusedCounter))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOOutlineFusionGroupsPass(
    const PTOOutlineFusionGroupsOptions &options) {
  return std::make_unique<PTOOutlineFusionGroupsPass>(options);
}
