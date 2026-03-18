//===- A5VMOps.td - A5VM operations -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VMOPS
#define MLIR_DIALECT_PTO_IR_A5VMOPS

include "PTO/IR/A5VMDialect.td"
include "PTO/IR/A5VMTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def A5VM_VectorType : Type<CPred<"::llvm::isa<::mlir::a5vm::VecType>($_self)">,
                           "A5VM vector type">;

class A5VM_Op<string mnemonic, list<Trait> traits = []>
    : Op<A5VM_Dialect, mnemonic, traits>;

class A5VM_TransferMetadata : Arguments<(ins
    StrAttr:$layout,
    I64Attr:$valid_rows,
    I64Attr:$valid_cols,
    I64Attr:$burst_count,
    I64Attr:$burst_len,
    I64Attr:$gm_stride,
    I64Attr:$ub_stride,
    BoolAttr:$ub_pad
  )>;

def A5VM_CopyGmToUbufOp : A5VM_Op<"copy_gm_to_ubuf", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = !con((
    ins
      AnyMemRef:$source,
      AnyMemRef:$destination
  ), A5VM_TransferMetadata);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination attr-dict `:` type($source) `,` type($destination)
  }];
}

def A5VM_VldsOp : A5VM_Op<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    AnyMemRef:$source,
    Index:$offset
  );

  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def A5VM_VabsOp : A5VM_Op<"vabs", [Pure]> {
  let arguments = (ins A5VM_VectorType:$input);
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_VstsOp : A5VM_Op<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    AnyMemRef:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination)
  }];
}

def A5VM_CopyUbufToGmOp : A5VM_Op<"copy_ubuf_to_gm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = !con((
    ins
      AnyMemRef:$source,
      AnyMemRef:$destination
  ), A5VM_TransferMetadata);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination attr-dict `:` type($source) `,` type($destination)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_A5VMOPS
