#include "adf.h"
#include "aie/add.h"

using namespace adf;

#define NUM 32

class mygraph : public graph
{
public:
    kernel k[NUM];
    input_port input[NUM];
    input_port output[NUM];
    
    mygraph()
    {
        for (int i=0; i<NUM; i++)
        {
            k[i] = kernel::create(add);
            source(k[i]) = "aie/add.cpp";
            runtime<ratio>(k[i]) = 0.9;
            
            connect<window<128>>(input[i], k[i].in[0]);
            connect<window<128>>(k[i].out[0], output[i]);
            async(k[i].in[1]);
        }
        
        /*location<kernel>(k[0]) = tile(2,0);
        location<kernel>(k[1]) = tile(2,1);
        location<kernel>(k[2]) = tile(3,0);
        location<kernel>(k[3]) = tile(3,1);
        location<kernel>(k[4]) = tile(6,0);
        location<kernel>(k[5]) = tile(6,1);
        location<kernel>(k[6]) = tile(7,0);
        location<kernel>(k[7]) = tile(7,1);
        location<kernel>(k[8]) = tile(10,0);
        location<kernel>(k[9]) = tile(10,1);
        location<kernel>(k[10]) = tile(11,0);
        location<kernel>(k[11]) = tile(11,1);
        location<kernel>(k[12]) = tile(18,0);
        location<kernel>(k[13]) = tile(18,1);
        location<kernel>(k[14]) = tile(19,0);
        location<kernel>(k[15]) = tile(19,1);
        location<kernel>(k[16]) = tile(26,0);
        location<kernel>(k[17]) = tile(26,1);
        location<kernel>(k[18]) = tile(27,0);
        location<kernel>(k[19]) = tile(27,1);
        location<kernel>(k[20]) = tile(34,0);
        location<kernel>(k[21]) = tile(34,1);
        location<kernel>(k[22]) = tile(35,0);
        location<kernel>(k[23]) = tile(35,1);
        location<kernel>(k[24]) = tile(42,0);
        location<kernel>(k[25]) = tile(42,1);
        location<kernel>(k[26]) = tile(43,0);
        location<kernel>(k[27]) = tile(43,1);
        location<kernel>(k[28]) = tile(46,0);
        location<kernel>(k[29]) = tile(46,1);
        location<kernel>(k[30]) = tile(47,0);
        location<kernel>(k[31]) = tile(47,1);*/
    }
};
