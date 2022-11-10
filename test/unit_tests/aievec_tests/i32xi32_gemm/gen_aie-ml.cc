void matmul(int32_t *restrict v1, int32_t *restrict v2, int32_t *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 64;
  size_t v6 = 1;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      size_t v8 = 0;
      size_t v9 = 64;
      size_t v10 = 16;
      for (size_t v11 = v8; v11 < v9; v11 += v10)
        chess_prepare_for_pipelining chess_loop_range(4, 4) {
          v16int32 v12 = *(v16int32 *)(v3 + 64 * v7 + v11);
          v16acc64 v13 = ups_to_v16acc64(v12, 0);
          size_t v14 = 0;
          size_t v15 = 64;
          size_t v16 = 16;
          for (size_t v17 = v14; v17 < v15; v17 += v16)
            chess_prepare_for_pipelining chess_loop_range(4, 4) {
              v16int32 v18 = *(v16int32 *)(v1 + 64 * v7 + v17);
              v16int32 v19 = *(v16int32 *)(v2 + 64 * v17 + v11);
              v16int32 v20 = broadcast_elem(v18, 0);
              v13 = mac_elem_16_2(v20, broadcast_zero_s32(), v19,
                                  undef_v16int32(), v13);
              size_t v21 = 1;
              size_t v22 = v17 + v21;
              v16int32 v23 = *(v16int32 *)(v2 + 64 * v22 + v11);
              v16int32 v24 = broadcast_elem(v18, 1);
              v13 = mac_elem_16_2(v24, broadcast_zero_s32(), v23,
                                  undef_v16int32(), v13);
              size_t v25 = 2;
              size_t v26 = v17 + v25;
              v16int32 v27 = *(v16int32 *)(v2 + 64 * v26 + v11);
              v16int32 v28 = broadcast_elem(v18, 2);
              v13 = mac_elem_16_2(v28, broadcast_zero_s32(), v27,
                                  undef_v16int32(), v13);
              size_t v29 = 3;
              size_t v30 = v17 + v29;
              v16int32 v31 = *(v16int32 *)(v2 + 64 * v30 + v11);
              v16int32 v32 = broadcast_elem(v18, 3);
              v13 = mac_elem_16_2(v32, broadcast_zero_s32(), v31,
                                  undef_v16int32(), v13);
              size_t v33 = 4;
              size_t v34 = v17 + v33;
              v16int32 v35 = *(v16int32 *)(v2 + 64 * v34 + v11);
              v16int32 v36 = broadcast_elem(v18, 4);
              v13 = mac_elem_16_2(v36, broadcast_zero_s32(), v35,
                                  undef_v16int32(), v13);
              size_t v37 = 5;
              size_t v38 = v17 + v37;
              v16int32 v39 = *(v16int32 *)(v2 + 64 * v38 + v11);
              v16int32 v40 = broadcast_elem(v18, 5);
              v13 = mac_elem_16_2(v40, broadcast_zero_s32(), v39,
                                  undef_v16int32(), v13);
              size_t v41 = 6;
              size_t v42 = v17 + v41;
              v16int32 v43 = *(v16int32 *)(v2 + 64 * v42 + v11);
              v16int32 v44 = broadcast_elem(v18, 6);
              v13 = mac_elem_16_2(v44, broadcast_zero_s32(), v43,
                                  undef_v16int32(), v13);
              size_t v45 = 7;
              size_t v46 = v17 + v45;
              v16int32 v47 = *(v16int32 *)(v2 + 64 * v46 + v11);
              v16int32 v48 = broadcast_elem(v18, 7);
              v13 = mac_elem_16_2(v48, broadcast_zero_s32(), v47,
                                  undef_v16int32(), v13);
              size_t v49 = 8;
              size_t v50 = v17 + v49;
              v16int32 v51 = *(v16int32 *)(v2 + 64 * v50 + v11);
              v16int32 v52 = broadcast_elem(v18, 8);
              v13 = mac_elem_16_2(v52, broadcast_zero_s32(), v51,
                                  undef_v16int32(), v13);
              size_t v53 = 9;
              size_t v54 = v17 + v53;
              v16int32 v55 = *(v16int32 *)(v2 + 64 * v54 + v11);
              v16int32 v56 = broadcast_elem(v18, 9);
              v13 = mac_elem_16_2(v56, broadcast_zero_s32(), v55,
                                  undef_v16int32(), v13);
              size_t v57 = 10;
              size_t v58 = v17 + v57;
              v16int32 v59 = *(v16int32 *)(v2 + 64 * v58 + v11);
              v16int32 v60 = broadcast_elem(v18, 10);
              v13 = mac_elem_16_2(v60, broadcast_zero_s32(), v59,
                                  undef_v16int32(), v13);
              size_t v61 = 11;
              size_t v62 = v17 + v61;
              v16int32 v63 = *(v16int32 *)(v2 + 64 * v62 + v11);
              v16int32 v64 = broadcast_elem(v18, 11);
              v13 = mac_elem_16_2(v64, broadcast_zero_s32(), v63,
                                  undef_v16int32(), v13);
              size_t v65 = 12;
              size_t v66 = v17 + v65;
              v16int32 v67 = *(v16int32 *)(v2 + 64 * v66 + v11);
              v16int32 v68 = broadcast_elem(v18, 12);
              v13 = mac_elem_16_2(v68, broadcast_zero_s32(), v67,
                                  undef_v16int32(), v13);
              size_t v69 = 13;
              size_t v70 = v17 + v69;
              v16int32 v71 = *(v16int32 *)(v2 + 64 * v70 + v11);
              v16int32 v72 = broadcast_elem(v18, 13);
              v13 = mac_elem_16_2(v72, broadcast_zero_s32(), v71,
                                  undef_v16int32(), v13);
              size_t v73 = 14;
              size_t v74 = v17 + v73;
              v16int32 v75 = *(v16int32 *)(v2 + 64 * v74 + v11);
              v16int32 v76 = broadcast_elem(v18, 14);
              v13 = mac_elem_16_2(v76, broadcast_zero_s32(), v75,
                                  undef_v16int32(), v13);
              size_t v77 = 15;
              size_t v78 = v17 + v77;
              v16int32 v79 = *(v16int32 *)(v2 + 64 * v78 + v11);
              v16int32 v80 = broadcast_elem(v18, 15);
              v13 = mac_elem_16_2(v80, broadcast_zero_s32(), v79,
                                  undef_v16int32(), v13);
              v16int32 v81 = srs_to_v16int32(v13, 0);
              *(v16int32 *)(v3 + 64 * v7 + v11) = v81;
            }
        }
    }
  return;
}
