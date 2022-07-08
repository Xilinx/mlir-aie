#include "graph.h"

#if !defined(__AIESIM__) && !defined(__ADF_FRONTEND__)
//#include "cardano/cardano_api/XRTConfig.h"
#include "adf/adf_api/XRTConfig.h"
//#include "experimental/xrt_kernel.h"
#include <fstream>
#endif

using namespace adf;

GMIO gmioIn0("gmioIn0", 64, 1);
GMIO gmioIn1("gmioIn1", 64, 1);
GMIO gmioIn2("gmioIn2", 64, 1);
GMIO gmioIn3("gmioIn3", 64, 1);
GMIO gmioIn4("gmioIn4", 64, 1);
GMIO gmioIn5("gmioIn5", 64, 1);
GMIO gmioIn6("gmioIn6", 64, 1);
GMIO gmioIn7("gmioIn7", 64, 1);
GMIO gmioIn8("gmioIn8", 64, 1);
GMIO gmioIn9("gmioIn9", 64, 1);
GMIO gmioIn10("gmioIn10", 64, 1);
GMIO gmioIn11("gmioIn11", 64, 1);
GMIO gmioIn12("gmioIn12", 64, 1);
GMIO gmioIn13("gmioIn13", 64, 1);
GMIO gmioIn14("gmioIn14", 64, 1);
GMIO gmioIn15("gmioIn15", 64, 1);
GMIO gmioIn16("gmioIn16", 64, 1);
GMIO gmioIn17("gmioIn17", 64, 1);
GMIO gmioIn18("gmioIn18", 64, 1);
GMIO gmioIn19("gmioIn19", 64, 1);
GMIO gmioIn20("gmioIn20", 64, 1);
GMIO gmioIn21("gmioIn21", 64, 1);
GMIO gmioIn22("gmioIn22", 64, 1);
GMIO gmioIn23("gmioIn23", 64, 1);
GMIO gmioIn24("gmioIn24", 64, 1);
GMIO gmioIn25("gmioIn25", 64, 1);
GMIO gmioIn26("gmioIn26", 64, 1);
GMIO gmioIn27("gmioIn27", 64, 1);
GMIO gmioIn28("gmioIn28", 64, 1);
GMIO gmioIn29("gmioIn29", 64, 1);
GMIO gmioIn30("gmioIn30", 64, 1);
GMIO gmioIn31("gmioIn31", 64, 1);

GMIO gmioOut0("gmioOut0", 64, 1);
GMIO gmioOut1("gmioOut1", 64, 1);
GMIO gmioOut2("gmioOut2", 64, 1);
GMIO gmioOut3("gmioOut3", 64, 1);
GMIO gmioOut4("gmioOut4", 64, 1);
GMIO gmioOut5("gmioOut5", 64, 1);
GMIO gmioOut6("gmioOut6", 64, 1);
GMIO gmioOut7("gmioOut7", 64, 1);
GMIO gmioOut8("gmioOut8", 64, 1);
GMIO gmioOut9("gmioOut9", 64, 1);
GMIO gmioOut10("gmioOut10", 64, 1);
GMIO gmioOut11("gmioOut11", 64, 1);
GMIO gmioOut12("gmioOut12", 64, 1);
GMIO gmioOut13("gmioOut13", 64, 1);
GMIO gmioOut14("gmioOut14", 64, 1);
GMIO gmioOut15("gmioOut15", 64, 1);
GMIO gmioOut16("gmioOut16", 64, 1);
GMIO gmioOut17("gmioOut17", 64, 1);
GMIO gmioOut18("gmioOut18", 64, 1);
GMIO gmioOut19("gmioOut19", 64, 1);
GMIO gmioOut20("gmioOut20", 64, 1);
GMIO gmioOut21("gmioOut21", 64, 1);
GMIO gmioOut22("gmioOut22", 64, 1);
GMIO gmioOut23("gmioOut23", 64, 1);
GMIO gmioOut24("gmioOut24", 64, 1);
GMIO gmioOut25("gmioOut25", 64, 1);
GMIO gmioOut26("gmioOut26", 64, 1);
GMIO gmioOut27("gmioOut27", 64, 1);
GMIO gmioOut28("gmioOut28", 64, 1);
GMIO gmioOut29("gmioOut29", 64, 1);
GMIO gmioOut30("gmioOut30", 64, 1);
GMIO gmioOut31("gmioOut31", 64, 1);

simulation::platform<NUM,NUM> plat(
&gmioIn0,
&gmioIn1,
&gmioIn2,
&gmioIn3,
&gmioIn4,
&gmioIn5,
&gmioIn6,
&gmioIn7,
&gmioIn8,
&gmioIn9,
&gmioIn10,
&gmioIn11,
&gmioIn12,
&gmioIn13,
&gmioIn14,
&gmioIn15,
&gmioIn16,
&gmioIn17,
&gmioIn18,
&gmioIn19,
&gmioIn20,
&gmioIn21,
&gmioIn22,
&gmioIn23,
&gmioIn24,
&gmioIn25,
&gmioIn26,
&gmioIn27,
&gmioIn28,
&gmioIn29,
&gmioIn30,
&gmioIn31,

&gmioOut0,
&gmioOut1,
&gmioOut2,
&gmioOut3,
&gmioOut4,
&gmioOut5,
&gmioOut6,
&gmioOut7,
&gmioOut8,
&gmioOut9,
&gmioOut10,
&gmioOut11,
&gmioOut12,
&gmioOut13,
&gmioOut14,
&gmioOut15,
&gmioOut16,
&gmioOut17,
&gmioOut18,
&gmioOut19,
&gmioOut20,
&gmioOut21,
&gmioOut22,
&gmioOut23,
&gmioOut24,
&gmioOut25,
&gmioOut26,
&gmioOut27,
&gmioOut28,
&gmioOut29,
&gmioOut30,
&gmioOut31
);

//for indexed access
GMIO* gmioIn[NUM] = {
    &gmioIn0,&gmioIn1,&gmioIn2,&gmioIn3,&gmioIn4,&gmioIn5,&gmioIn6,&gmioIn7,&gmioIn8,&gmioIn9,
    &gmioIn10,&gmioIn11,&gmioIn12,&gmioIn13,&gmioIn14,&gmioIn15,&gmioIn16,&gmioIn17,&gmioIn18,&gmioIn19,
    &gmioIn20,&gmioIn21,&gmioIn22,&gmioIn23,&gmioIn24,&gmioIn25,&gmioIn26,&gmioIn27,&gmioIn28,&gmioIn29,
    &gmioIn30,&gmioIn31};
    
GMIO* gmioOut[NUM] = {
    &gmioOut0,&gmioOut1,&gmioOut2,&gmioOut3,&gmioOut4,&gmioOut5,&gmioOut6,&gmioOut7,&gmioOut8,&gmioOut9,
    &gmioOut10,&gmioOut11,&gmioOut12,&gmioOut13,&gmioOut14,&gmioOut15,&gmioOut16,&gmioOut17,&gmioOut18,&gmioOut19,
    &gmioOut20,&gmioOut21,&gmioOut22,&gmioOut23,&gmioOut24,&gmioOut25,&gmioOut26,&gmioOut27,&gmioOut28,&gmioOut29,
    &gmioOut30,&gmioOut31};

//graph declaration
mygraph g;

//connection
class GlobalConnection
{
public:
    GlobalConnection()
    {
        for (int i=0; i<NUM; i++)
        {
            connect<>(plat.src[i], g.input[i]);
            connect<>(g.output[i], plat.sink[i]);
        }
    }
} connection;

//M is starting index
#define M 0
//P is number of GMIO channels to test
#define P 32
//ITER is number of graph iteration
#define ITER 8

int main(int argc, char **argv)
{
#if !defined(__AIESIM__) && !defined(__ADF_FRONTEND__)
//    auto dhdl = xclOpen(0, nullptr, XCL_QUIET);
//    auto xclbin = load_xclbin(dhdl, "a.xclbin");
//    auto top = reinterpret_cast<const axlf*>(xclbin.data());
//    adf::registerXRT(dhdl, top->m_header.uuid);
    char* xclbinFilename = argv[1];
    auto dhdl = xrtDeviceOpen(0); //device index=0
    xrtDeviceLoadXclbinFile(dhdl, xclbinFilename);
    xuid_t uuid;
    xrtDeviceGetXclbinUUID(dhdl, uuid);
    adf::registerXRT(dhdl, uuid);

#endif
    
    g.init();
    
    for (int i=M; i<M+P; i++)
    {
        g.update(g.k[i].in[1], i+1);
    }

    int32* inputArray[NUM];
    for (int i=M; i<M+P; i++)
    {
        inputArray[i] = (int32*)GMIO::malloc(256*sizeof(int32));
        for (int j=0; j<256; j++)
            inputArray[i][j] = i+1;
    }
    
    int32* outputArray[NUM];
    for (int i=M; i<M+P; i++)
    {
        outputArray[i] = (int32*)GMIO::malloc(256*sizeof(int32));
        for (int j=0; j<256; j++)
            outputArray[i][j] = 0xABCDEF00;
    }
    
    std::cout<<"GMIO::malloc completed"<<std::endl;
    
    g.run(ITER);
    for (int j=0; j<ITER; j++)
    {
        for (int i=M; i<M+P; i++)
        {
            gmioIn[i]->gm2aie_nb(&inputArray[i][j*32], 32*sizeof(int32));
            gmioOut[i]->aie2gm_nb(&outputArray[i][j*32], 32*sizeof(int32));
        }
    }
    
    std::cout<<"GMIO::gm2aie_nb and GMIO::aie2gm_nb enqueing completed"<<std::endl;
    
    for (int i=M; i<M+P; i++)
    {
        gmioOut[i]->wait(); //assuming data from gmioIn[i] are processed by the graph and output to gmioOut[i]
    }
    
    std::cout<<"GMIO::wait finished"<<std::endl;
    
    if (M==0 && P==32)
        g.end();
    
    //check results
    int errorCount = 0;
    for (int i=M; i<M+P; i++)
    {
        for (int j=0; j<32*ITER; j++)
        {
            //std::cout<<outputArray[i][j]<<" ";
            if (outputArray[i][j] != (1+i)*2)
                errorCount++;
        }
        //std::cout<<std::endl;
    }
    if (errorCount)
        printf("Test failed with %d errors\n", errorCount);
    else
        printf("Test passed\n");
        
    for (int i=M; i<M+P; i++)
    {        
        GMIO::free(inputArray[i]);
        GMIO::free(outputArray[i]);
    }
    std::cout<<"GMIO::free completed"<<std::endl;
 
#if !defined(__AIESIM__) && !defined(__ADF_FRONTEND__)
//    xclClose(dhdl);
    xrtDeviceClose(dhdl);
#endif
   
    return errorCount;
}
