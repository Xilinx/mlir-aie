
all: trace.json

trace_output := ../../../notebook/traces/bottleneck_cifar_split_vector.txt

# This is based on trace_output but with some edits
# 1. Prepend hex data with 0x
# 2. Remove extra 0xdbffdbff in the beginning
# 3. Replace start timer with smaller value usch as 0x0005d0f7
trace_output_IR := bottleneck_cifar_split_vector_IR.txt

mlir_src := ../aieWithTrace.mlir
trace_config := bottleneck_cifar_split_vector.json

trace.json: $(trace_output) $(mlir_src)
	../../../../../utils/parse_trace.py --filename $< --mlir $(word 2,$^) --colshift 2 > $@

$(trace_output_IR): $(trace_output)
	cp $(trace_output) ./$(trace_output_IR)
	# Do edits here

eventIR.txt: $(trace_output_IR) $(trace_config)
	hwfrontend --trace $< --trace_config $(word 2,$^) --pkg-dir . --outfile $@

trace_eventIR.json: eventIR.txt $(mlir_src)
	../../../../../utils/parse_eventIR.py --filename $< --mlir $(word 2,$^) --colshift 2 > $@

