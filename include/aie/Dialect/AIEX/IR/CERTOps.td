//===- CERTOps.td ------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CERT_OPS
#define CERT_OPS

def AIE_CertUcDmaChainOp: AIEX_Op<"cert.uc_dma_chain", [Symbol,
                                                    SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "uC DMA Chain operation";
  let description = [{
    This operation defines a chain of uC DMA buffer descriptors. It contains a
    single region where the buffer descriptors can be specified.

    Example:
    ```
      cert.uc_dma_chain @dma_chain {
        aiex.uc_dma_bd 0x600400, 0x20400, 128, true, false, false
        ...
      }
    ```
  }];
  let arguments = (
    ins OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    ($sym_name^)? $body attr-dict
  }];
}

def AIE_CertUcDmaBdOp : AIEX_Op<"cert.uc_dma_bd", [HasParent<"CertUcDmaChainOp">]> {
  let summary = "A DMA Buffer Descriptor for uC-DMA";
  let description = [{
    Defines a uC-DMA buffer descriptor within a chain. The descriptor specifies
    how the uC-DMA should transfer data from a remote address to a local address.

    - `remote_address`: Symbol reference to the data to be written
    - `local_address`: Address in the AIE array register that uC-DMA will access
    - `length`: Number of 32-bit words to transfer
    - `next_bd`: Flag indicating if this is the last uC-DMA BD in the chain
      (true if there are more BDs following, false if this is the last)

    Example:
    ```mlir
      cert.uc_dma_chain @dma_chain {
        aiex.cert.uc_dma_bd @mem01_bd0, 0x20400, 8, true
        aiex.cert.uc_dma_bd @mem01_bd1, 0x20420, 8, true
        aiex.cert.uc_dma_bd @mem11_bd0, 0x40400, 8, false
      }
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$remote_address,
        I32Attr:$local_address,
        I32Attr:$length,
        BoolAttr:$next_bd
  );
  let results = (outs);
  let assemblyFormat = [{
    $remote_address `,` $local_address `,` $length `,` $next_bd  attr-dict
  }];
}

def AIE_CertPageOp: AIEX_Op<"cert.page",
                            [ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp", "CertSectionOp"]>,
                             SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "A control code page";
  let description = [{
    This operation represents a page of control code. Pages are the basic unit
    of control code organization, with a maximum size of 8192 bytes. Each page
    typically contains one job along with its associated data.

    Corresponds to the `.eop` (end of page) directive in assembly.

    Example:
    ```
      cert.page {
        cert.job(0) {
          cert.write32(0x1000, 42)
        }
      }
    ```
  }];
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{ $body attr-dict }];
}

def AIE_CertJobOp: AIEX_Op<"cert.job", [ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp",
                                                      "CertPageOp", "CertSectionOp"]>,
                                        SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Start a job with a given job ID";
  let description = [{
    This operation starts a job identified by the given job ID. The job ID is an
    integer that uniquely identifies the job to be started.

    Example:
    ```
      cert.job(42) {
        ...
      }
    ```
  }];
  let arguments = (
    ins I32Attr:$job_id
  );
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{ `(` $job_id `)` $body attr-dict }];
  let extraClassDeclaration = [{
    int getJobID() { return getJobId(); }
  }];
}

def AIE_CertWrite32Op: AIEX_Op<"cert.write32", [HasParent<"CertJobOp">]> {
  let summary = "Write a 32-bit value to a specified address";
  let description = [{
    This operation writes a 32-bit value to a specified address.

    Example:
    ```
      cert.write32(0x1000, 42)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict
  }];
}

def AIE_CertMaskWrite32Op: AIEX_Op<"cert.maskwrite32", [HasParent<"CertJobOp">]> {
  let summary = "Write a masked 32-bit value to a specified address";
  let description = [{
    This operation writes a masked 32-bit value to a specified address.

    Example:
    ```
      cert.maskwrite32(0x1000, 0xFF, 42)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$mask,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $mask `,` $value `)` attr-dict
  }];
}

def AIE_CertApplyOffset57Op: AIEX_Op<"cert.apply_offset_57", [HasParent<"CertJobOp">]> {
  let summary = "Apply offset 57 operation";
  let description = [{
    This operation adds a 57-bit offset to Shim DMA BD base address fields.
    The offset is loaded from the 32-bit registers arg[offset] and arg[offset+1]
    from the current argument list.
    'symbol' - A symbol pointing to the data to be patched
    'num_entries' - the number of Shim DMA BD entries to patch
    'offset' - index into the argument list to the start of offset data

    Example:
    ```
      memref.global "private" constant @bd_table : memref<9xi32> = dense<[0, -2147483648, 256, 117440512, 4351, 1, 1, 1, 1024]>
      ...
      cert.apply_offset_57(@bd_table, 1, 2)
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol,
        I16Attr:$num_entries,
        I16Attr:$offset
  );
  let assemblyFormat = [{
    `(` $symbol `,` $num_entries `,` $offset `)` attr-dict
  }];
}

def AIE_CertWaitTCTSOp: AIEX_Op<"cert.wait_tcts", [HasParent<"CertJobOp">]> {
  let summary = "Wait for a specified number of TCTs (Task Completion Tokens)";
  let description = [{
    This operation waits for a specified number of Task Completion Tokens (TCTs)
    for a given shim tile and channel.

    Example:
    ```
      cert.wait_tcts(1, 2, 3) // Wait for 3 TCTs on shim column 1, channel 2
    ```
  }];
  let arguments = (
    ins I32Attr:$tile_id,
        I32Attr:$channel_id,
        I32Attr:$target_tcts
  );
  let assemblyFormat = [{
    `(` $tile_id `,` $channel_id `,` $target_tcts `)` attr-dict
  }];
}

def AIE_CertUcDmaWriteDesSyncOp: AIEX_Op<"cert.uc_dma_write_des_sync",
                                        [HasParent<"CertJobOp">]> {
  let summary = "uC DMA Write Descriptor Sync operation";
  let description = [{
    This operation enqueues a uC DMA transfer. The symbol operand refers to a
    cert.uc_dma_bd_chain operation.

    Example:
    ```
      cert.uc_dma_write_des_sync(@dma_descriptor)
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    `(` $symbol `)` attr-dict
  }];
}

def AIE_CertLocalBarrierOp: AIEX_Op<"cert.local_barrier", [HasParent<"CertJobOp">]> {
  let summary = "Local barrier operation";
  let description = [{
    This operation represents a local barrier that synchronizes execution based on a barrier ID and the number of participants.

    Example:
    ```
      cert.local_barrier(1, 2)
    ```
  }];
  let arguments = (
    ins I32Attr:$local_barrier_id,
        I32Attr:$num_participants
  );
  let assemblyFormat = [{
    `(` $local_barrier_id `,` $num_participants `)` attr-dict
  }];
}

def AIE_CertRemoteBarrierOp: AIEX_Op<"cert.remote_barrier", [HasParent<"CertJobOp">]> {
  let summary = "Remote barrier operation";
  let description = [{
    This operation represents a remote barrier that synchronizes execution
    based on a barrier ID `remote_barrier_id` and `party_mask`, a bit map of
    participant uC IDs.

    Example:
    ```
      cert.remote_barrier(1, 0x3)
    ```
  }];
  let arguments = (
    ins I32Attr:$remote_barrier_id,
        I32Attr:$party_mask
  );
  let assemblyFormat = [{
    `(` $remote_barrier_id `,` $party_mask `)` attr-dict
  }];
}

def AIE_CertPreemptOp: AIEX_Op<"cert.preempt", [HasParent<"CertJobOp">]> {
  let summary = "Preemption point with save/restore control code references";
  let description = [{
    This operation defines a preemption point in control code. If preemption is
    required, the runtime will execute either the save or restore control code
    section depending on the context switch direction.

    - `id`: Preemption point index (must be consistent across all uCs in multi-uC case)
    - `save_section`: Symbol reference to control code section for saving state
    - `restore_section`: Symbol reference to control code section for restoring state

    This opcode should take one whole job which in turn should take one whole page.

    Example:
    ```
      cert.preempt(0, @save0, @restore0)
    ```
  }];
  let arguments = (
    ins I32Attr:$id,
        FlatSymbolRefAttr:$save_section,
        FlatSymbolRefAttr:$restore_section
  );
  let assemblyFormat = [{
    `(` $id `,` $save_section `,` $restore_section `)` attr-dict
  }];
}

def AIE_CertNopOp: AIEX_Op<"cert.nop", []> {
  let summary = "No operation (NOP)";
  let description = [{
    This operation represents a no-op instruction. It does nothing.

    Example:
    ```
      cert.nop
    ```
  }];
  let assemblyFormat = [{ attr-dict }];
}

def AIE_CertSectionOp: AIEX_Op<"cert.section",
                               [Symbol,
                                ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp"]>,
                                SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Named control code section";
  let description = [{
    This operation defines a named section of control code that can be referenced
    by other operations (like cert.load_pdi or cert.preempt). It corresponds to the
    assembly pattern of `label: ... .endl label`.

    The section contains one or more jobs and can be loaded/executed by referencing
    its symbol.

    Example:
    ```
      cert.section @pdi0 {
        cert.job(10) {
          cert.write32(0x6A3000, 0x2)
        }
      }

      cert.job(0) {
        cert.load_pdi(0, @pdi0)
      }
    ```
  }];
  let arguments = (
    ins SymbolNameAttr:$sym_name
  );
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
}

def AIE_CertLoadPdiOp: AIEX_Op<"cert.load_pdi", [HasParent<"CertJobOp">]> {
  let summary = "Load PDI (Platform Device Image)";
  let description = [{
    This operation loads a PDI (Platform Device Image), which is a piece of control code.
    It can be loaded by other control code at any time. The `pdi_id` is an ELF-wide unique
    identifier that specifies a unique PDI. Consecutive loading of the same PDI results in
    subsequent loads being skipped by the microcontroller. The `symbol` operand refers to
    a cert.section containing the PDI jobs.

    This opcode should take one whole job which in turn should take one whole page.

    Example:
    ```
      cert.load_pdi(0, @pdi_section)
    ```
  }];
  let arguments = (
    ins UI32Attr:$pdi_id,
        FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    `(` $pdi_id `,` $symbol `)` attr-dict
  }];
}

def AIE_CertAttachToGroupOp: AIEX_Op<"cert.attach_to_group", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Attach to a group operation";
  let description = [{
    This operation attaches enclosed jobs to a specified group identified by an
    integer ID.

    Example:
    ```
      cert.attach_to_group(1) {
        ...
        cert.job(2) {
          ...
        }
        cert.job(3) {
          ...
        }
        ...
      }
    ```
  }];
  let arguments = (
    ins I32Attr:$group_id
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $group_id `)` $body attr-dict }];
  let extraClassDeclaration = [{
    int getGroupID() { return getGroupId(); }
  }];
}

#endif // CERT_OPS
