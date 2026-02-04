/*  (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
    (c) Copyright 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
   
    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual property
    laws.
   
    DISCLAIMER
    This disclaimer is not a license and does not grant any
    rights to the materials distributed herewith. Except as
    otherwise provided in a valid license issued to you by
    Xilinx, and to the maximum extent permitted by applicable
    law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
    WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
    AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
    BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
    INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
    (2) Xilinx shall not be liable (whether in contract or tort,
    including negligence, or under any other theory of
    liability) for any loss or damage of any kind or nature
    related to, arising under or in connection with these
    materials, including for any direct, or any indirect,
    special, incidental, or consequential loss or damage
    (including loss of data, profits, goodwill, or any type of
    loss or damage suffered as a result of any action brought
    by a third party) even if such damage or loss was
    reasonably foreseeable or Xilinx had been advised of the
    possibility of the same.
   
    CRITICAL APPLICATIONS
    Xilinx products are not designed or intended to be fail-
    safe, or for use in any application requiring fail-safe
    performance, such as life-support or safety devices or
    systems, Class III medical devices, nuclear facilities,
    applications related to the deployment of airbags, or any
    other applications that could lead to death, personal
    injury, or severe property or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.
   
    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */

#pragma once
// #include "adf.h"
#include "aie_api/aie.hpp"

#define GELU_OFFSET 14512
#define GELU_ADDRSHFT 3

//------------------------------------------------------
// find look-up table address
//------------------------------------------------------
void gelu_getaddr_relu(int16 * pd_in, int16 * pa_out){

	const int sz = 508*12;
    v16int16    * __restrict pd = (v16int16 * __restrict) pd_in;
    v32uint8    * __restrict pa = (v32uint8 * __restrict) pa_out;
	v16bfloat16 * __restrict py = (v16bfloat16 * __restrict) pd_in;

    v32int16 a,b;
	
	const v32acc32    offset = ups_to_v32acc32(broadcast_to_v32int16(GELU_OFFSET), 0);
	const v32int16    absmsk = broadcast_to_v32int16(0x7fff);
	const v32bfloat16 zero   = broadcast_zero_to_v32bfloat16();
	

	for(int j=0; j< (sz>>5); j++)
		chess_prepare_for_pipelining
        chess_loop_range(4,)
	{
		// load 32 values from input
		a = set_v32int16(0, *pd++);
		a = insert(a,    1, *pd++);
		
		// relu operation
		v32bfloat16 y = max(v32bfloat16(a), zero);
		*py++ = extract_v16bfloat16(y, 0);
		*py++ = extract_v16bfloat16(y, 1);

		// remove the sign bit
		b = band(a, absmsk);

		// subtract the offset
		v32acc32 c = sub(ups_to_v32acc32(b, 0), offset);
		
		// shift and save to memory
		*pa++ = srs_to_v32uint8(c, GELU_ADDRSHFT);
	}
	
		// remaining 16 words
		a = insert(broadcast_zero_to_v32int16(), 0, *pd++);

		// relu operation
		v32bfloat16 y = max(v32bfloat16(a), zero);
		*py = extract_v16bfloat16(y, 0);

		// remove the sign bit
		b = band(a, absmsk);

		// subtract the offset
		v32acc32 c = sub(ups_to_v32acc32(b, 0), offset);
		
		// shift and save to memory
		*pa++ = srs_to_v32uint8(c, GELU_ADDRSHFT);
}


//---------------------------------------------------------
// use the look up table to find the compensation value
// output goes to streaming interface
//---------------------------------------------------------
void gelu_lkup_add_max(int16 * pd_in, int16 * addr_in, int16 * lut0_in, int16 * lut1_in, int16 * yout){
	
	const int sz_in = 508*4;

    v16uint8    __aie_dm_resource_a * __restrict pa =  (v16uint8     __aie_dm_resource_a * __restrict ) addr_in;
    v16bfloat16 __aie_dm_resource_b * __restrict pd =  (v16bfloat16  __aie_dm_resource_b * __restrict ) pd_in;
	
	const int8  __aie_dm_resource_c * lut0 = (int8  __aie_dm_resource_c *)lut0_in;
	const int8  __aie_dm_resource_d * lut1 = (int8  __aie_dm_resource_d *)lut1_in;
	
	v4int16 * __restrict py = (v4int16 * __restrict) yout;
	
	v32bfloat16 ones = broadcast_one_to_v32bfloat16();
	
	// gelu value will never be smaller than -1
	v32bfloat16 ymax = broadcast_to_v32bfloat16(-10);

    // Process 16 entries in parallel
    const int loopcnt = (sz_in/16);

	for (int idx = 0; idx < loopcnt; idx++)
		chess_prepare_for_pipelining
		chess_loop_range(8,)
	{
		v64int8 y1, y2;
		
		// load one data
		v32acc32 a = sups(set_v32uint8(0, *pa++), 2);
		v16acc32 b = extract_v16acc32(a, 0);
		
		load_lut_2x_int8(lut0, lut1, v16int32(b), y1, y2);
		
		v32int16 y = shuffle(v32int16(y1), v32int16(y2), shuffle_T16_16x4_lo);

		v32bfloat16 ya = insert(v32bfloat16(y), 1, *pd++);
		v16accfloat yy = mul_elem_16_2(ya, ones);
		
		ymax = max(ymax, set_v32bfloat16(0, to_v16bfloat16(yy)));
		
	}
	
	// ymax has the format of 
	// ch0_0, ch1_0, ch2_0, ch3_0
	// ch0_1, ch1_1, ch2_1, ch3_1
	// ch0_2, ch1_2, ch2_2, ch3_2
	// ch0_3, ch1_3, ch2_3, ch3_3
	
	ymax = max(ymax, shift(ymax, undef_v32bfloat16(), 8));
	ymax = max(ymax, shift(ymax, undef_v32bfloat16(), 4));
	
	// max values are in first 4 entries
	//put_ms(ext_elem(v16cint16(ymax), 0), 0);
	//put_ms(ext_elem(v16cint16(ymax), 1), tlast);
	
	*py = extract_v4int16(v32int16(ymax),0);

}

inline void gelu_max_12x508(int16 * pd_in, int16 * tempbuf, int16 * lut0_in, int16 * lut1_in, int16 * y_out)
{
	
	// clear the buffer to all zero
	*((v16bfloat16 *) y_out) = extract_v16bfloat16(broadcast_zero_to_v32bfloat16(), 0);
	
	// get address for all
	gelu_getaddr_relu(pd_in, tempbuf);

	// find max for each group of 4 channels
	gelu_lkup_add_max(pd_in,       tempbuf,       lut0_in, lut1_in, y_out);
	gelu_lkup_add_max(pd_in+508*4, tempbuf+508*2, lut0_in, lut1_in, y_out+4);
	gelu_lkup_add_max(pd_in+508*8, tempbuf+508*4, lut0_in, lut1_in, y_out+8);
	
}