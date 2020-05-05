module {
%c2 = constant 2 : index
  %c3 = constant 3 : index
%1 = "AIE.switchbox"(%c2, %c3) ({
   "AIE.connect"() {sourcePort=0:i32,destPort=0:i32}: () -> ()
	"AIE.connect"() {sourcePort=1:i32,destPort=0:i32}: () -> ()
	"AIE.endswitch"() : () -> ()
	}) : (index, index) -> index
}
