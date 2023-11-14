## Simple Log Hello World

This reference design demonstrates a simple, low overhead, printf-style log message from AIE tiles.

Features:
* Low instruction memory overhead (based on variadic templates)
* Efficient transfers.
    + Format string addresses are parsed from the compiled elfs host side.
    + Data transfers from the AIE tile are just string addresses and parameters; no strings are sent.
    + Host-side string addresses are used to look up format strings and populated with parameters.

### Building and executing (on a phx laptop)
Type the following to build and run the design in a wsl terminal.
```
make run
```

### Logging from the kernel code
Below is a simple example of how to use `ipulog.h` in a kernel.
```c++
#include "ipulog.h"

void kernel(uint32_t *logbuffer) {
	IPULogger log(logbuffer, 2048); // buffer to use, and length of buffer
	log.write("Hello!");
}
```

### Extracting format string addresses at compile time
After building the `.xclbin` in the directory where the AIE Tile elfs are, call the following to create the mappings from the format strings to addresses.
```bash
python3 elfStringParser.py --input <directory where generated elfs are> --output formatStrings.csv
```

### Decoding the log at runtime 
At runtime we can run the IPU and then run a decoder on the output buffer to render all the strings.

```c++
  #include "decodelog.hpp"
  // ...
  IPULogDecoder log_decoder("formatString.csv");
  for (const std::string &str : log_decoder.decode(logbuffer)) {
    std::cout << str << std::endl;
  }
```

