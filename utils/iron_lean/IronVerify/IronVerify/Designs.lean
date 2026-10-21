-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Designs.WholeArrayC1
import IronVerify.Designs.WholeArrayC2
import IronVerify.Designs.WholeArrayC3
import IronVerify.Designs.WholeArrayC4
import IronVerify.Designs.WholeArrayC5
import IronVerify.Designs.WholeArrayC6
import IronVerify.Designs.WholeArrayC7
import IronVerify.Designs.WholeArrayC8

namespace IronVerify.Designs

def registry : List (String × Net × (List Int → Option Seq)) := [
    ("WholeArrayC1", Designs.WholeArrayC1.net, Designs.WholeArrayC1.seqOfList),
    ("WholeArrayC2", Designs.WholeArrayC2.net, Designs.WholeArrayC2.seqOfList),
    ("WholeArrayC3", Designs.WholeArrayC3.net, Designs.WholeArrayC3.seqOfList),
    ("WholeArrayC4", Designs.WholeArrayC4.net, Designs.WholeArrayC4.seqOfList),
    ("WholeArrayC5", Designs.WholeArrayC5.net, Designs.WholeArrayC5.seqOfList),
    ("WholeArrayC6", Designs.WholeArrayC6.net, Designs.WholeArrayC6.seqOfList),
    ("WholeArrayC7", Designs.WholeArrayC7.net, Designs.WholeArrayC7.seqOfList),
    ("WholeArrayC8", Designs.WholeArrayC8.net, Designs.WholeArrayC8.seqOfList)]

end IronVerify.Designs
