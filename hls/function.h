

#pragma once

#define AP_INT_MAX_W 2048

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>

#include "stream_tools.h"

using namespace hls;
using namespace std;

/*
* BN + QUReLU
*/
template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT, // for BN

			unsigned BN_BIT,
			unsigned DATA_BIT, //ACT_IN BIT
			unsigned W_BIT,
			unsigned L_SHIFT
			>
ap_uint<OUT_BIT> BN_QUReLU( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {   

	const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT); 

//	ap_int<BN_BIT> product = in * inc;
//#pragma HLS RESOURCE variable=product core=Mul_LUT
// BatchNorm
    ap_int<BN_BIT> bn_res = in * inc + bias;
	ap_uint<OUT_BIT> res;

// quantize to 2^N levels ∈ [0,1] with DoreFa Net && ReLU   
	if (bn_res > 0) {
		bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);  
		if (bn_res > 15){
			res = 15;
		} else {
			res = bn_res;
		}
	} else {
		res = 0;
	}
	return res;
    
}

/*
* resize
*/
template<unsigned IN_IMAGE_HEIGHT,
		 unsigned IN_IMAGE_WIDTH,
		 unsigned RESIZE_IMAGE_HEIGHT,
		 unsigned RESIZE_IMAGE_WIDTH
		>
void resize(stream<ap_uint<24> > &in, stream<ap_uint<24> > & out) {
#pragma HLS dataflow
    Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;
#pragma HLS STREAM variable=raw_img depth=128 dim=1
    Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;
#pragma HLS STREAM variable=resize_img depth=128 dim=1
    stream_to_mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH>(in, raw_img);
    Resize_opr_linear(raw_img, resize_img);
    mat_to_stream<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH>(resize_img, out);
}

template<unsigned IN_IMAGE_HEIGHT,
		 unsigned IN_IMAGE_WIDTH,
		 unsigned RESIZE_IMAGE_HEIGHT,
		 unsigned RESIZE_IMAGE_WIDTH
		>
void resize_batch(stream<ap_uint<24> > &in, stream<ap_uint<24> > & out, unsigned int reps) {
	unsigned batch_size = 1 << reps;
    for (unsigned int rep=0; rep < batch_size; rep ++) {
        resize<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH>(in, out);
    }
}

/**
 *  padding
 */ 
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P>
void padding(

	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 0)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < (1 << reps); rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {
			for ( unsigned s = 0; s < OUT_COL; s++ ) {
#pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				out.write(0);
			}
		}

	}
}





