Using precompiled functions:

## Precompiled core function

Users should interact with the flow at the highest level. The way the flow works, the *kernel.cc* is compiled to an objective file, and the flow manages low-level details (i.e., locks).

Tests 01, 03, 05, and 07 show different examples of this method.

## Precompiled kernel (tests 02 and 04)

In this method, users have to administer low-level details in the *kernel.cc* (i.e., controlling the locks). This can lead to information replication in multiple places. For instance, the code for both processors that access a particular buffer needs to be kept synchronized.

As shown in tests 02 and 04, *kernel.cc* can be modified and recompiled without recompiling the host.
