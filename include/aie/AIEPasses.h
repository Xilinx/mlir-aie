#ifndef AIE_PASSES_H
#define AIE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace aie {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/AIEPasses.h.inc"

} // namespace aie

#endif // AIE_PASSES_H
