module {
%c2 = constant 2 : index
  %c3 = constant 3 : index
%1 = "aie.switchbox"(%c2, %c3) ({
   "aie.connect"() {sourcePort=0:i32,destPort=0:i32}: () -> ()
	"aie.connect"() {sourcePort=1:i32,destPort=1:i32}: () -> ()
	"aie.connect"() {sourcePort=2:i32,destPort=2:i32}: () -> ()
	"aie.connect"() {sourcePort=3:i32,destPort=3:i32}: () -> ()
   "aie.connect"() {sourcePort=4:i32,destPort=4:i32}: () -> ()
	"aie.connect"() {sourcePort=5:i32,destPort=5:i32}: () -> ()
	"aie.connect"() {sourcePort=6:i32,destPort=6:i32}: () -> ()
	"aie.connect"() {sourcePort=7:i32,destPort=7:i32}: () -> ()
   "aie.connect"() {sourcePort=8:i32,destPort=8:i32}: () -> ()
	"aie.connect"() {sourcePort=9:i32,destPort=9:i32}: () -> ()
   "aie.connect"() {sourcePort=10:i32,destPort=10:i32}: () -> ()
	"aie.connect"() {sourcePort=11:i32,destPort=11:i32}: () -> ()
	"aie.connect"() {sourcePort=12:i32,destPort=12:i32}: () -> ()
	"aie.connect"() {sourcePort=13:i32,destPort=13:i32}: () -> ()
   "aie.connect"() {sourcePort=14:i32,destPort=14:i32}: () -> ()
	"aie.connect"() {sourcePort=15:i32,destPort=15:i32}: () -> ()
	"aie.connect"() {sourcePort=16:i32,destPort=16:i32}: () -> ()
	"aie.connect"() {sourcePort=17:i32,destPort=17:i32}: () -> ()
   "aie.connect"() {sourcePort=18:i32,destPort=18:i32}: () -> ()
	"aie.connect"() {sourcePort=19:i32,destPort=19:i32}: () -> ()
   "aie.connect"() {sourcePort=20:i32,destPort=20:i32}: () -> ()
	"aie.connect"() {sourcePort=21:i32,destPort=21:i32}: () -> ()
	"aie.connect"() {sourcePort=22:i32,destPort=22:i32}: () -> ()
	"aie.connect"() {sourcePort=23:i32,destPort=23:i32}: () -> ()
   "aie.connect"() {sourcePort=24:i32,destPort=24:i32}: () -> ()

	"aie.endswitch"() : () -> ()
	}) : (index, index) -> index
	%3 = "aie.core"(%c2, %c3) : (index, index) -> index
		%4 = "aie.core"(%c3, %c2) : (index, index) -> index
%2 = "aie.switchbox"(%c3, %c2) ({
   "aie.connect"() {sourcePort=1:i32,destPort=0:i32}: () -> ()
	"aie.connect"() {sourcePort=25:i32,destPort=15:i32}: () -> ()
	"aie.connect"() {sourcePort=26:i32,destPort=16:i32}: () -> ()
	"aie.endswitch"() : () -> ()
	}) : (index, index) -> index
"aie.wire"(%1, %2) {sourcePort=1:i32,destPort=0:i32} : (index, index) -> ()
"aie.flow"(%3, %4) {sourcePort=1:i32,destPort=0:i32} : (index, index) -> ()
}
