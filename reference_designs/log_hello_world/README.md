## Simple Log Hello World

This reference design demonstrates a simple, low overhead, printf-style log message from AIE tiles.

Features:
* Low instruction memory overhead (based on variadic templates)
* Efficient transfers.
        + Format string addresses are parsed from the compiled elfs host side.
        + Data transfers from the AIE tile are just string addresses and parameters; no strings are sent.
        + Host-side string addresses are used to look up format strings and populated with parameters.

### Building and execution (on a phx laptop)
Type the following to build and run the design in a wsl terminal.
```
make run
```
