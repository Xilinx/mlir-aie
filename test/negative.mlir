module {
%c2 = constant 2 : index
  %c3 = constant 3 : index
%1 = "aie.switchbox"(%c2, %c3) ({
   "aie.connect"() {sourcePort=0:i32,destPort=0:i32}: () -> ()
	"aie.connect"() {sourcePort=1:i32,destPort=0:i32}: () -> ()
	"aie.endswitch"() : () -> ()
	}) : (index, index) -> index
}
