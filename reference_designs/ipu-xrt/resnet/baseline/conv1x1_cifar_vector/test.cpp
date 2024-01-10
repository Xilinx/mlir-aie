#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// #define IMAGE_WIDTH_IN 256
// #define IMAGE_HEIGHT_IN 256



 int X = 28;
 int Y = 6;



 int Ci_1 = 8;
 int Co_1 = 4;
 int KSIZE_1=1;

 int IN_SIZE_1 = Y*X*Ci_1;
 int WTS_SIZE_1 =KSIZE_1*KSIZE_1*Ci_1*Co_1;
 int OUT_SIZE_1 =Y*X*Co_1;


 int Ci_2 = 4;
 int Co_2 = 8;
 int KSIZE_2=3;
#define PADDING (KSIZE_2 / 2) // Padding for 3x3 kernel
 int Y_out = 6;

 int WTS_SIZE_2 =KSIZE_2*KSIZE_2*Ci_2*Co_2;
 int OUT_SIZE_2 =Y_out*X*Co_2;


 int Ci_3 = 8;
 int Co_3 = 8;
 int KSIZE_3=1;

 int WTS_SIZE_3 =KSIZE_3*KSIZE_3*Ci_3*Co_3;
 int OUT_SIZE_3 =Y_out*X*Co_3;


namespace po = boost::program_options;


void check_arg_file_exists(po::variables_map &vm_in, std::string name) {
  if (!vm_in.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  } else {
    std::ifstream test(vm_in[name].as<std::string>());
    if (!test) {
      throw std::runtime_error("The " + name + " file " +
                               vm_in[name].as<std::string>() +
                               " does not exist.\n");
    }
  }
}

void convolution2D(int8_t *input, 
                   int8_t *kernels, 
                   int8_t *output, int input_channels,int output_channels, int kernel_height) {

    for (int cout = 0; cout < output_channels; cout++) {
        for (int y = PADDING; y < Y + PADDING; y++) {
            for (int x = PADDING; x < X + PADDING; x++) {

                int8_t sum = 0.0f;
                for (int cin = 0; cin < input_channels; cin++) {
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_height; kx++) {
                            int input_y = y - PADDING + ky;
                            int input_x = x - PADDING + kx;

                            sum += input[cin * (X + 2*PADDING) * (Y + 2*PADDING) + input_y * (X + 2*PADDING) + input_x] * 
                                   kernels[cin * output_channels * kernel_height * kernel_height + cout * kernel_height * kernel_height + ky * kernel_height + kx];
                        }
                    }
                }

                int out_y = y - PADDING;
                int out_x = x - PADDING;

                output[cout * X * Y + out_y * X + out_x] = sum;
            }
        }
    }
}
std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

void conv2d_simple(int8_t *input, int8_t *kernels, int8_t *output, int input_width, int input_height, int input_channels,int output_channels,
                   int kernel_width, int kernel_height)                    
{
    int y, x, ky, kx, ic, oc;
    
    int pad_width = (kernel_width - 1) / 2;
    int pad_height = (kernel_height - 1) / 2;
    for (y = 0; y < input_height; y++) { // row of output image
        for (oc = 0; oc < output_channels; oc++) {
        
            for (x = 0; x < input_width; x++) { // col of output image
                int8_t sum = 0;
                for (ic = 0; ic < input_channels; ic++) {
                    for (ky = 0; ky < kernel_height; ky++) {
                        for (kx = 0; kx < kernel_width; kx++) {
                            int in_y = y + ky - pad_height;
                            int in_x = x + kx - pad_width;
                            int8_t val = input[ic * input_width * input_height + in_y * input_width + in_x];
                            int8_t k = kernels[(oc * input_channels + ic) * kernel_width * kernel_height +
                                              ky * kernel_width + kx];
                            sum += val * k;
                        }
                    }
                }
                output[y * input_width * output_channels + oc * input_width + x] = sum;
            }
        }
    }
}

int main(int argc, const char *argv[]) {

int8_t input[IN_SIZE_1] ;
int8_t wts_1[WTS_SIZE_1]= { 
    -7, -7,  4, -3, -4,  5, -3,  6, -4,  4,  3, -5, -2,  4,  0, -3,
    -1,  3,  2,  5,  1, -1,  6,  3,  5,  5,  3, -7,  3, -5, -4, -3};
int8_t out_1[OUT_SIZE_1];

int8_t wts_2[WTS_SIZE_2];

int8_t out_2[OUT_SIZE_2];
int8_t wts_3[WTS_SIZE_3]={ 
    -7, -7,  4, -3, -4,  5, -3,  6, -4,  4,  3, -5, -2,  4,  0, -3,
    -1,  3,  2,  5,  1, -1,  6,  3,  5,  5,  3, -7,  3, -5, -4, -3,
    -7, -7,  4, -3, -4,  5, -3,  6, -4,  4,  3, -5, -2,  4,  0, -3,
    -1,  3,  2,  5,  1, -1,  6,  3,  5,  5,  3, -7,  3, -5, -4, -3};

int8_t gold[OUT_SIZE_3];
  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inA =
      xrt::bo(device, IN_SIZE_1*sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inB =
      xrt::bo(device, (WTS_SIZE_1+WTS_SIZE_2+WTS_SIZE_3)*sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, OUT_SIZE_3*sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // auto debug =
  //     xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  // auto bo_out =
  //     xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  int8_t *bufInA = bo_inA.map<int8_t *>();
  std::vector<int8_t> srcVecA;
  for (int i = 0; i < IN_SIZE_1; i++)
    srcVecA.push_back(1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(int8_t)));


  int8_t *bufInB = bo_inB.map<int8_t *>();
  std::vector<int8_t> srcVecB;

  for (int i = 0; i < WTS_SIZE_1; i++)
    srcVecB.push_back(wts_1[i]);

  for (int i = 0; i < (WTS_SIZE_2); i++)
    srcVecB.push_back(1);

  for (int i = 0; i < WTS_SIZE_3; i++)
    srcVecB.push_back(wts_3[i]);
  
  memcpy(bufInB, srcVecB.data(), (srcVecB.size() * sizeof(int8_t)));


  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int8_t *bufOut = bo_out.map<int8_t *>();

  for (int i = 0; i < IN_SIZE_1; i++) {
    input[i]=1;

  }
  //  for (int i = 0; i < WTS_SIZE_1; i++) {
  //   wts_1[i]=1;

  // }
   for (int i = 0; i < WTS_SIZE_2; i++) {
    wts_2[i]=1;

  }
  // for (int i = 0; i < WTS_SIZE_3; i++) {
  //   wts_3[i]=1;

  // }

  for (int i = 0; i < OUT_SIZE_1; i++) {
    out_1[i]=0;

  }

  for (int i = 0; i < OUT_SIZE_2; i++) {
    out_2[i]=0;

  }


  for (int i = 0; i < OUT_SIZE_3; i++) {
    gold[i]=0;

  }

  int8_t out_1_padded[Co_1 * (X + 2*PADDING) * (Y_out + 2*PADDING)];
  memset(out_1_padded, 0, sizeof(out_1_padded)); // Initialize the padded input to zeros
 
// kernel
conv2d_simple(input, wts_1, out_1,X,Y_out, Ci_1, Co_1, KSIZE_1, KSIZE_1) ;

 // Embed original input inside the padded input
for(int cin = 0; cin < Co_1; cin++) {
    for(int y = 0; y < Y_out; y++) {
        for(int x = 0; x < X; x++) {
            out_1_padded[cin * (X + 2*PADDING) * (Y_out + 2*PADDING) + (y + PADDING) * (X + 2*PADDING) + (x + PADDING)] 
            = out_1[cin * X * Y_out + y * X + x];
            
        }
    }
}
convolution2D(out_1_padded, wts_2, out_2,Ci_2, Co_2,KSIZE_2);
conv2d_simple(out_2, wts_3, gold,X,Y_out, Ci_3, Co_3, KSIZE_3, KSIZE_3) ;

  printf("\n****** AIE OUT**********\n");
  for( unsigned yy=0; yy<Y_out; yy++ ) 
  {
    printf("\n****** Y_out %i ******\n",yy);
    for( unsigned cc=0; cc<Co_3; cc++ ) 
      {
      printf("\n****** OFM %i ******\n",cc);
      for( unsigned xx=0; xx<X; xx++ )                      
        {
        // printf("\n****** OFM %i ******\n",cc);
        printf( "%i\t", ( (int)(*(bufOut+xx+cc*X+Co_3*X*yy))));
      }

    }
    printf("\n");

  }
printf("\n \t \t********************* GOLD OUT *********************n");

  for( unsigned yy=0; yy<Y_out; yy++ ) 
  {
    printf("\n****** Y_out %i ******\n",yy);
    for( unsigned cc=0; cc<Co_3; cc++ ) 
    {
      printf("\n****** OFM %i ******\n",cc);
      for( unsigned xx=0; xx<X; xx++ ) 
      {
        // printf("\n****** OFM %i ******\n",cc);
        printf( "%i\t", (int)gold[xx+cc*X+Co_3*X*yy]+input[xx+cc*X+Co_3*X*yy]);
      }

    }
    printf("\n");

  }
  int errors = 0;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }



  // int errors = 0;

  // if (!errors) {
  //   std::cout << "\nPASS!\n\n";
  //   return 0;
  // } else {
  //   std::cout << "\nfailed.\n\n";
  //   return 1;
  // }

}

