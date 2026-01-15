//===- ListDialect.td - List dialect definition ------------*- tablegen -*-===//
//
// Defines a small list dialect for the MLIR tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_LIST_DIALECT
#define TUTORIAL_LIST_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def List_Dialect : Dialect {
  let name = "list";
  let summary = "A small list dialect for tutorial examples.";
  let description = [{
    The list dialect models immutable lists with a few operations that are
    useful for parser, printer, and round-trip examples.
  }];
  let cppNamespace = "::mlir::tutorial::list";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

def List_ListType : TypeDef<List_Dialect, "List"> {
  let mnemonic = "list";
  let summary = "List type.";
  let description = [{
    A list whose elements all have the same MLIR type.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class List_Op<string mnemonic, list<Trait> traits = []>
    : Op<List_Dialect, mnemonic, traits>;

def List_EmptyOp : List_Op<"empty", [Pure]> {
  let summary = "Create an empty list.";
  let arguments = (ins);
  let results = (outs List_ListType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def List_AppendOp : List_Op<"append", [Pure]> {
  let summary = "Append one value to a list.";
  let arguments = (ins List_ListType:$input, AnyType:$value);
  let results = (outs List_ListType:$result);
  let assemblyFormat =
      "$input `,` $value attr-dict `:` type($input) `,` type($value) `->` "
      "type($result)";
}

def List_ConcatOp : List_Op<"concat", [Pure]> {
  let summary = "Concatenate two lists.";
  let arguments = (ins List_ListType:$lhs, List_ListType:$rhs);
  let results = (outs List_ListType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` "
      "type($result)";
}

def List_LengthOp : List_Op<"length", [Pure]> {
  let summary = "Return the length of a list.";
  let arguments = (ins List_ListType:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // TUTORIAL_LIST_DIALECT
