import os
import random

class PacketFlow:
    def __init__(self, id, src, dests):
        self.id = id
        self.src = src
        self.dests = dests

def getNextFilename():
    if not os.path.exists("./generated"):
        os.makedirs("./generated")
    num = 0
    filename = "generated/test_packet_flows"+str(num)+".mlir"
    while os.path.exists(filename):
        num += 1
        filename = "generated/test_packet_flows"+str(num)+".mlir"
    return filename

def buildTileOp(file, col, row):
    col = str(col)
    row = str(row)
    file.write("%tile"+col+"_"+row+" = AIE.tile("+col+", "+row+")\n")

def buildPacketFlowOp(file, pf):
    file.write("\nAIE.packet_flow("+str(pf.id)+") {\n")

    src_string = "%tile"+str(pf.src[0])+"_"+str(pf.src[1])
    file.write("\tAIE.packet_source<"+src_string+", DMA : "+str(random.randint(0,1))+">\n")

    for dest in pf.dests:
        dest_string = "%tile"+str(dest[0])+"_"+str(dest[1])
        file.write("\tAIE.packet_dest<"+dest_string+", DMA : "+str(random.randint(0,1))+">\n")
    file.write("}\n")


def generatePacketFlows(cols, rows, flow_count):
    filename = getNextFilename()
    with open(filename, "w") as file:
        # generate all tiles
        for col in range(cols):
            for row in range(rows):
                buildTileOp(file, col, row)

        # generate random packet flows
        for id in range(flow_count):
            src_coords = (random.randint(0, cols-1), random.randint(0, rows-1))
            dest_coords = [(random.randint(0, cols-1), random.randint(0, rows-1))]
            pf = PacketFlow( id, src_coords, dest_coords)
            buildPacketFlowOp(file, pf)


if __name__ == "__main__":
    generatePacketFlows(5, 5, 4)