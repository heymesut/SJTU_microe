

#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "shift_reg.h"

using namespace hls;
using namespace std;

/**
 * MaxPool
 * S = 2, K = 2
 */
template <	unsigned IN_ROW,
            unsigned IN_COL,
			unsigned IN_CH,
            unsigned IN_CH_PARA,
			unsigned IN_BIT>
void MaxPool(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in,
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& out,
	const unsigned reps = 0)
{
    const unsigned K = 2;
	ap_uint<IN_CH_PARA*IN_BIT> result = 0;
	unsigned k_cnt = 0;
    unsigned ITERATION = IN_ROW * IN_COL * IN_CH / IN_CH_PARA;

	for (unsigned rep = 0; rep < (ITERATION << reps); rep++) {
#pragma HLS PIPELINE II=1

		ap_uint<IN_CH_PARA*IN_BIT> temp_in = in.read();

		for (unsigned c = 0; c < IN_CH_PARA; c++) {
#pragma HLS UNROLL

			ap_uint<IN_BIT> temp = temp_in( (c+1)*IN_BIT-1 , c*IN_BIT );
				
			result( (c+1)*IN_BIT-1, c*IN_BIT ) = (temp > result( (c+1)*IN_BIT-1, c*IN_BIT )) ? temp : result( (c+1)*IN_BIT-1, c*IN_BIT );
		}

        if(++ k_cnt == K*K) {
            out.write(result);
            result = 0;
            k_cnt = 0;
        }
	}
}

/**
 * MaxPool_2D
 * K = 2, S = 2
 * In, Out width: IN_CH_PARA * IN_BIT
 */
template <	unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
            unsigned IN_CH_PARA,
			unsigned IN_BIT>
void MaxPool_2D(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in,
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& out, 
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

    MaxPool<IN_ROW, IN_COL, IN_CH, IN_CH_PARA, IN_BIT>(sr_out, out, reps);
}
