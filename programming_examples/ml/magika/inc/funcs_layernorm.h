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

// #include "adf.h"
#include "aie_api/aie.hpp"

//------------------------------------------------
// ptmp is a temp buffer, size = num_row x 4 x 2
//------------------------------------------------
__attribute__((noinline)) void layernorm_sum_row_512x8(int16 * px_in, v8bfloat16 &mean, v8bfloat16 &var, int16 * py_out)
{

	v16bfloat16 * restrict px  = (v16bfloat16 * restrict) px_in;
	v16bfloat16 * restrict py  = (v16bfloat16 * restrict) py_out;
	
	v32bfloat16   xx[2];
	v16accfloat  acc[2];
	v16accfloat xacc[2];

	
	v32bfloat16 coef = broadcast_to_v32bfloat16(-0.001953125);
	
	xx[0] = set_v32bfloat16(0, *px++);
	xx[0] = insert(xx[0],   1, *px++);

	acc[0] = mul_elem_16_2(xx[0], coef);  xx[1] = set_v32bfloat16(0, *px++);  *py++ = extract_v16bfloat16(xx[0], 0);
	acc[1] = mul_elem_16_2(xx[0], xx[0]); xx[1] = insert(xx[1],   1, *px++);  *py++ = extract_v16bfloat16(xx[0], 1);

		
	for(int rowid = 0; rowid<(512/8-1); rowid++)
		chess_prepare_for_pipelining
		chess_loop_range(8,)
	{

			// (1)
			acc[0] = mac_elem_16_2(xx[1], coef,  acc[0]);  xx[0] = set_v32bfloat16(0, *px++);   *py++ = extract_v16bfloat16(xx[1], 0);
			acc[1] = mac_elem_16_2(xx[1], xx[1], acc[1]);  xx[0] = insert(xx[0],   1, *px++);   *py++ = extract_v16bfloat16(xx[1], 1);

			// (2)
			acc[0] = mac_elem_16_2(xx[0], coef,  acc[0]);  xx[1] = set_v32bfloat16(0, *px++);  *py++ = extract_v16bfloat16(xx[0], 0);
			acc[1] = mac_elem_16_2(xx[0], xx[0], acc[1]);  xx[1] = insert(xx[1],   1, *px++);  *py++ = extract_v16bfloat16(xx[0], 1);

	}
	
	
	// (last addition)
	acc[0] = mac_elem_16_2(xx[1], coef,  acc[0]);  *py++ = extract_v16bfloat16(xx[1], 0);
	acc[1] = mac_elem_16_2(xx[1], xx[1], acc[1]);  *py++ = extract_v16bfloat16(xx[1], 1);

	// Final summation 
	xacc[0] = add(acc[0], shift(acc[0], undef_v16accfloat(), (unsigned int)8));
	xacc[1] = add(acc[1], shift(acc[1], undef_v16accfloat(), (unsigned int)8));
	
	v16bfloat16 xmean = to_v16bfloat16(xacc[0]);
	v16bfloat16	xsqr  = to_v16bfloat16(xacc[1]);
	v32bfloat16 aa    = set_v32bfloat16(0, xmean);
	            aa    = insert(aa,      1, xsqr);
				coef = insert(coef, 0, xmean);

	 acc[1] = mac_elem_16_2(aa, coef, broadcast_to_v16accfloat(-9.999999974752427e-7));
	
	v32bfloat16 bb = set_v32bfloat16(0, to_v16bfloat16(acc[1]));
	
	mean = extract_v8bfloat16(aa, 0);
	var  = extract_v8bfloat16(bb, 0);

}



//------------------------------------------------
// compute 1/sqrt(sum(x^2) - mean(x)^2 + eps)
//------------------------------------------------
__attribute__((noinline)) v8bfloat16 layernorm_denom(v8bfloat16 var, int16 * lut_layernorm0, int16 * lut_layernorm1)
{

	int8  __aie_dm_resource_c * lut0 = (int8  __aie_dm_resource_c *)lut_layernorm0;
	int8  __aie_dm_resource_d * lut1 = (int8  __aie_dm_resource_d *)lut_layernorm1;

	v64int8 y1;
	
	
	// compute 1/sqrt(x)
	// (1) force it to int16
	v32int16 a = insert(broadcast_zero_to_v32int16(), 0, v8int16(var));
		
	// (2.a) extract low 8 bits to address look-up table
	v32int16 b = band(a, broadcast_to_v32int16(0x00ff));
		
	// (2.b) x4 to translate to addresses
	v16acc32 d = sups(extract_v16int16(b,0), 2);
		
	// (2.c) look up table
	load_lut_int8(lut0, lut1, v16int32(d), y1);
	

	// (2.d) remove zeros in the look up results. lower part of u0 is the inverse of fractional part
	v32int16 u0 = shuffle(v32int16(y1), undef_v32int16(), T16_16x4_lo);
	
	
	// (3.a) get exponent and do the inverse
	v32int16  aa = band(a, broadcast_to_v32int16(0x7f00));
	v32uint16 bb = sub(broadcast_u16(190<<8), v32uint16(aa));
	
	v16acc32 c1 = sups( extract_v16uint16(bb, 0), 0);

	// (3.b) convert the exponents into bfloat16 values and make sure the upper 16 values are 0
	v32int16 v0 = insert(broadcast_zero_to_v32int16(), 0, lsrs(c1,1));


	// (4) multiply exponent with the inserse of fraction part
	v16accfloat yy = mul_elem_16_2( v32bfloat16(u0), v32bfloat16(v0) );

	// (5) write out the result
	return( extract_v8bfloat16( set_v32bfloat16(0, to_v16bfloat16(yy)), 0) );

}


//--------------------------------------------------------------------------
// compute (x - mean) * std = x*std - mean*std = {x, -mean}, {std, std}
//--------------------------------------------------------------------------
__attribute__((noinline)) void layernorm_norm_512x8(int16 * px_in, v8bfloat16 mean, v8bfloat16 std, int16 * py_out)
{

	v16bfloat16 * restrict px = (v16bfloat16 * restrict) px_in;
	v16bfloat16 * restrict py = (v16bfloat16 * restrict) py_out;

	v16bfloat16 aa = concat(mean, mean);
	v32bfloat16 bb = concat(std, std, std, std);
	v32bfloat16 ab;

	for(int rowid = 0; rowid<(512/2); rowid++)
		chess_prepare_for_pipelining
		chess_loop_range(16,)
		chess_unroll_loop(2)
	{
		   ab = concat(*px++, aa);
		*py++ = to_v16bfloat16(mul_elem_16_2(ab, bb));
	}

}



//--------------------------------------------------------------------------
// compute x * gamma + beta = {x, 1} x {gamma, beta}
//--------------------------------------------------------------------------
__attribute__((noinline)) void layernorm_rescale_512x8(int16 * px_in, int16 *pg_in, int16 *pb_in, int16 * py_out)
{

	v16bfloat16 * restrict px = (v16bfloat16 * restrict) px_in;
	v16bfloat16 * restrict pg = (v16bfloat16 * restrict) pg_in; // gamma
	v16bfloat16 * restrict pb = (v16bfloat16 * restrict) pb_in; // beta

	v16bfloat16 * restrict py  = (v16bfloat16 * restrict)  py_out;

	v32bfloat16 vv = broadcast_one_to_v32bfloat16();

	for(int colid = 0; colid<(512/2); colid++)
		chess_prepare_for_pipelining
		chess_loop_range(4,)
		chess_unroll_loop(2)
	{
		vv = insert(vv, 0, *px++);
		v32bfloat16 aa = concat(*pg++, *pb++);

		*py++ = to_v16bfloat16(mul_elem_16_2(vv, aa));
	}

}

