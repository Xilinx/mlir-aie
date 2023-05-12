This document details the current work done on improving the packet flow passes as of May 12, 2023. It serves as a summary of my internship work done, and will hopefully be of use to anyone who works on packet flows in the future.


# *aie-find-flows*

This pass was already capable of finding packet flows (and circuit flows), but it had some flaws which caused it to sometimes report packet flows that were not correctly implemented. This occured when one packet rule directed more packet IDs than intended, which can lead to other rules being precluded and some IDs being misrouted. Here is an example:

    AIE.packetrules(DMA : 0) {
        AIE.rule(24, 0, %41)
        AIE.rule(24, 0, %40)
    }

This is obviously a poor set of packet rules. Any packets the 2nd rule would route have already been routed by the 1st rule!
In this benchmark, these packet rules are “trying” to send packet IDs 1, 2, 3, and 7 to the West, and send packet IDs 0, 4, 5, 6 to the South. And indeed, these two rules working in a void would accomplish that goal. But when put next to each other, the priority of the 1st rule effectively hides the 2nd rule completely. This is a consequence of how the packet rules are generated in aie-create-packet-flows.


Originally, *aie-find-flows* detected packet flows by finding connections through switchboxes until an endpoint (either a Core or DMA op) was found. This works well for circuit switched flows, but finding a connection does not guarantee that a packet would route along that connection (due to packet rule priorities). In the example, all 8 flow IDs (0-7) would be seen as "connected", but of course four of them would never reach their destination!

The pass was updated to track what individual IDs would be routed according to the packet rules; a "connection" is not enough! This modification correctly models that an ID entering a switchbox will follow the rule with highest priority, and therefore will only find a flow when a packet could actually traverse the path.

*aie-find-flows* now only detects packet flows when the packetrules truly direct it to its destination. This makes it a useful verification tool for packet flows.


# *aie-create-packet-flows*

The main goal here is to enable Pathfinder routing with packet flows. This was accomplished, but some complications arose in doing so.

Previously, a function called *buildPSroute()* was used to naively implement packet routes by simply moving left (or right) until the target column was reached, and then moving up (or down) until the target row was reached. This was fairly easy to implement, but was not congestion aware. Instead, the same Pathfinder algorithm used for routing circuit flows was adapted to be useable on packet flows. This was accomplished by:

* Creating a Pathfinder object and adding all packet flows to the object.
* Run the congestion aware routing algorithm
* The routing solution was given back to the packet flows pass as a set of SwitchSettings. 

        DenseMap<Flow*, SwitchSettings*> flow_solutions;

* The remainder of *aie-create-packet-flows* generated amsel, masterset, and packetrule ops which implement the flow_solutions. This is where the complications began.

# Complications
* As seen in *aie-find-flows* some packetrules can preclude other rules! This can lead to misrouting of packets. 
* *aie-find-flows* can correctly detect when this happens, but now we need to make *aie-create-packet-flows* avoid it altogether!


## Nomenclature on packet rules:
A “specific” packet rule is one that targets only a few flow IDs, and no unintended or "extra" IDs. A “broad” packet rule casts a wide net and will direct many flow IDs, possibly more than intended. 
Broad rules are great for directing many packet flows, but have a risk of misdirecting some if we are not extremely careful!

### For example...

### Specific packet rules:
	AIE.rule(31, 13, %) // this rule will direct only flowID 13	(Hamming weight = 5)
    AIE.rule(30, 8, %)  // directs only flow ID 8 or 9		    (Hamming weight = 4)
    AIE.rule(28, 0, %)  // directs only flow ID 0, 1, 2, 3		(Hamming weight = 3)

### Broad packet rules:
	AIE.rule(16, 0, %)  // directs any ID 0-15			(Hamming weight = 1)
	AIE.rule(24, 8, %)  // directs any ID 8-15			(Hamming weight = 2)

Notice that a more specific packet rule will have a larger hamming weight. That is, the more bits that are 1 in the mask, the more specific the rule is, and the fewer IDs it will direct. We can therefore measure the "specificity" of a rule by looking at the Hamming weight.

If packet rules are too broad, then we risk having some IDs precluded (i.e. misdirected). For example:

    AIE.packetrules(DMA : 0) {
      AIE.rule(24, 8, %40)
      AIE.rule(26, 10, %41)
    }

Here, the routing intended by Pathfinder is to send Flow IDs (10, 11, 15) West, and send Flow IDs (8, 9, 12, 13, 14) South. However, it can be seen that the 1st rule dominates all eight of these flow IDs and it will send all IDs (8-15) to the South! 3 flows will be misrouted! The 2nd rule might as well not exist! In this case, we can say that the first rule is too broad, because it “accidentally” picks up flow IDs it did not intend and causes misroutes.

On the other hand, if packet rules are too specific, we quickly run out of space (each switchbox input port can only have 4 rules associated with it! This is a hardware constraint of the AIE array.)

    AIE.packetrules(DMA : 0) {
      AIE.rule(31, 8, %40)
      AIE.rule(31, 9, %40)
      AIE.rule(31, 12, %40)
      AIE.rule(31, 13, %40)
      AIE.rule(31, 14, %40)
      AIE.rule(26, 10, %41)
    }

Using very specific rules can clearly lead to more than 4 rules per input port, and therefore cannot be realized in hardware!

We are therefore left with a tricky tradeoff between making rules too specific (we run out of rules!) or too broad (some packets might be misdirected). Both might lead to an incorrect implementation. We are looking for a happy medium.

My approach to finding this happy medium is as follows:

1) Group all of the required flow IDs for a given input port (this part was already done by the object *slaveGroups* in AIECreatePacketFlows.cpp).
2) For each group, generate a very specific ruleset for the input port. That is, one rule for each ID with mask=31, match=ID. (This is done in the function findBestMaskRules())
3) Try to combine these specific rules into less specific rules without picking up too many extra IDs. (This is done in the function combineMaskRules())

This approach combines as many packet rules as possible without adding extra flow IDs, but extra IDs can be allowed by setting the "margin" parameter of findBestMaskRules() (this can reduce the number of rules, but might cause a misroute!). However, this method makes no guarantee as to the number of rules it will use, and therefore it is not hard to find a benchmark which results in more than 4 rules on a single input port (not allowed!). 


# Known limitations and issues
## (and suggestions on how to solve them):
1) There is no guarantee in the pass that there won’t be more than 4 rules on a single input port.

    *Possible mitigation:* Allow use of more than just channel 0 (see below).

2) There is also no guarantee that there won't be any misroutes! Unless the "margin" parameter is set to 0 (which means that NO additional flow IDs will be accepted when combining rules), then a misroute is possible. 

    *Possible mitigation:* Allow packet IDs to be modified (see below).

3)	Thorough testing is needed, including tests with fanouts. When tests are designed, they should run *aie-create-packet-flows*, check that no *AIE.packet_flow* ops exist, then run *aie-find-flows*, and then check for expected *AIE.packet_flow* ops.


# Additional possibilities to explore:

1)	Enable *aie-create-packet-flows* pass to modify packet IDs as necessary. When Pathfinder routes flows, it has no notion of generating packet rules, and so in congested areas it will route flow IDs that are not ideal for combining into correct packet rules. For example:

    * Let’s say that Pathfinder wants to route flow IDs 1, 2, 3, 7 through a switchbox, but there is no concise rule to combine these flow IDs (without possibly picking up extra IDs and messing up other flows). AIE.rule(24, 0, %) directs these IDs, but it also picks up ID 0, 4, 5, 6 which might misroute another packet flow. 

    * But, if we can change flow ID 7 to ID 0, then we have a "perfect" rule: AIE.rule(28, 0, %) which directs only IDs 0, 1, 2, 3 and no others.

    * This change might solve the other problems previously discussed too, since making "perfect" packet rules will guarantee no misroutes, and reduce the total number of rules at the same time!


2) Currently, all packets are routed on channel 0 (in an attempt to leave other channels open for circuit-switched flows). When more than 4 packet rules are required, another channel could be used. 
    * Add a mechanism in *aie-create-packet-flows* to use more channels when needed.


3)	Give Pathfinder the ability to “promote” circuit-switched flows to packet-switched flows. This would only be done when no solution can be found, and packet routes would allow using the same resources for more than one flow. This comes with a few complications:

    a.	Pathfinder might have trouble routing both packet and circuit flows at the same time, since it does not distinguish between them. To Pathfinder, a flow is a flow is a flow.

    b.	Since there is no notion of required bandwidth in each flow, Pathfinder has no good criteria for choosing which flows to “promote”. Ideally, the lower bandwidth flows would be combined into packet flows. 

    c.  **Suggestion: add estimated bandwidth information to packet flows.**

4) An alternative approach I attempted was to identify when some packet IDs were precluded by a higher priority rule, and simply add an extra specific rule to "protect" that packet ID. Code for this can be found in previous commits, but has been removed since it did not work as well as I'd hoped. This approach was good because it guaranteed that there would be no misrouted packets, but it resulted in a lot of extra rules and very often exceeded 4 per input port.


