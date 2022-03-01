

#pragma once

#define AP_INT_MAX_W 2048

#include <ap_int.h>
#include <hls_stream.h>

#include "shift_reg.h"

using namespace hls;
using namespace std;

/**
 * REORG
 * S = 2, K = 2
 */
template <	unsigned IN_ROW,
            unsigned IN_COL,
			unsigned IN_CH,
            unsigned IN_CH_PARA,
			unsigned IN_BIT>
void ReOrg(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in,
	stream<ap_uint<4*IN_CH*IN_BIT> >&    out,
	const unsigned reps = 0)
{
    const unsigned K = 2;
	ap_uint<K*K*IN_CH*IN_BIT> result = 0;
	unsigned k_cnt  = 0;
	unsigned ch_cnt = 0;
    unsigned ITERATION = IN_ROW * IN_COL * IN_CH / IN_CH_PARA;

	for (unsigned rep = 0; rep < (ITERATION << reps); rep++) {
#pragma HLS PIPELINE II=1

		ap_uint<IN_CH_PARA*IN_BIT> temp_in = in.read();	
		result( ((k_cnt*IN_CH + (ch_cnt+1)*IN_CH_PARA)*IN_BIT -1), ((k_cnt*IN_CH + ch_cnt*IN_CH_PARA)*IN_BIT) ) = temp_in;
		//same as the shift register replacement order

		k_cnt++;
        if(k_cnt == K*K) {
			k_cnt = 0;
			ch_cnt++;
		}

		if(ch_cnt == (IN_CH / IN_CH_PARA) ){
			out.write(result);
			ch_cnt = 0;
		}
	}
}

/**
 * ReOrg_2D
 * K = 2, S = 2
 * In width: IN_CH_PARA * IN_BIT
 * Out width: IN_CH * IN_BIT * K * K
 */
template <	unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
            unsigned IN_CH_PARA,
			unsigned IN_BIT>
void ReOrg_2D(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in,
	stream<ap_uint<4*IN_CH*IN_BIT> >&    out, 
	const unsigned reps = 0)
{
#pragma HLS DATAFLOW

    const unsigned K = 2;
    const unsigned S = 2;

    //Shift Register
    //1IN1OUT
    //Width: IN_CH_PARA * IN_BIT 
    stream<ap_uint<IN_CH_PARA*IN_BIT> > sr_out("sr_out");
    Shift_Register_1O<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_CH_PARA>(in, sr_out, reps);

    ReOrg<IN_ROW, IN_COL, IN_CH, IN_CH_PARA, IN_BIT>(sr_out, out, reps);
}
