//===- LinalgExtInterface.td -[Desc]------------------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_TD
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_TD

include "mlir/IR/OpBase.td"

// The interface is a subset of LinalgStructuredInterface.
def LinalgExtInterface : OpInterface<"LinalgExtOp"> {
  let cppNamespace = "::mlir::triton";

  let extraClassDeclaration = [{
    //========================================================================//
    // Forwarding functions to access interface methods from the
    // DestinationStyleOpInterface.
    // MLIR currently does not support dependent interfaces or interface
    // inheritance. By construction all ops with StructuredOpInterface must
    // implement DestinationStyleOpInterface.
    // TODO: reevalute the need for a cast when a better mechanism exists.
    //========================================================================//

    int64_t getNumDpsInputs() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getNumDpsInputs();
    }

    int64_t getNumDpsInits() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getNumDpsInits();
    }

    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getDpsInputOperands();
    }

    OpOperand *getDpsInputOperand(int64_t i) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getDpsInputOperand(i);
    }

    void setDpsInitOperand(int64_t i, Value value) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .setDpsInitOperand(i, value);
    }

    MutableOperandRange getDpsInitsMutable() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getDpsInitsMutable();
    }

    OpOperand *getDpsInitOperand(int64_t i) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getDpsInitOperand(i);
    }

    bool isDpsInput(OpOperand *opOperand) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .isDpsInput(opOperand);
    }

    bool isDpsInit(OpOperand *opOperand) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .isDpsInit(opOperand);
    }

    bool isScalar(OpOperand *opOperand) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .isScalar(opOperand);
    }

    OpResult getTiedOpResult(OpOperand *opOperand) {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .getTiedOpResult(opOperand);
    }

    bool hasPureBufferSemantics() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .hasPureBufferSemantics();
    }

    bool hasPureTensorSemantics() {
      return mlir::cast<mlir::DestinationStyleOpInterface>(*this->getOperation())
          .hasPureTensorSemantics();
    }

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible.
    LogicalResult reifyResultShapes(OpBuilder &b,
        mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = mlir::cast<DenseIntElementsAttr>(
          (*this)->getAttr("operand_segment_sizes"));
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyLinalgExtOpInterface($_op); }];
}

#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_TD
