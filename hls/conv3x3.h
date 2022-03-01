

#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "function.h"
#include "stream_tools.h"
#include "shift_reg.h"
#include "PE_array.h"

using namespace hls;
using namespace std;

/**
 * CONV3x3, P=1, S=1
 * NOT the first layer
 * In: IN_BIT * IN_CH
 * Out0, Out1: OUT_BIT * OUT_CH_PARA
 */
template <
			unsigned IN_ROW,   
			unsigned IN_COL,   
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT, 

			unsigned W_BIT,
			unsigned M_BIT,   //psum width
            unsigned BN_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			
			unsigned IN_CH_PARA, 
			unsigned OUT_CH_PARA,    
			unsigned L_SHIFT>
void conv3x3_bn_act(
	stream<ap_uint<IN_BIT * IN_CH> >& in, 
	const ap_uint<IN_CH_PARA*W_BIT> weights[OUT_CH_PARA][((IN_CH*9)/IN_CH_PARA)*(OUT_CH/OUT_CH_PARA)],
	const ap_int<INC_BIT> inc[OUT_CH_PARA][OUT_CH/OUT_CH_PARA],
	const ap_int<BIAS_BIT> bias[OUT_CH_PARA][OUT_CH/OUT_CH_PARA],
	stream<ap_uint<OUT_BIT*OUT_CH_PARA> >& out0,
    stream<ap_uint<OUT_BIT*OUT_CH_PARA> >& out1,
	const unsigned reps = 0)
{
#pragma HLS DATAFLOW

    // size after padding
	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// P=1, S=1, K=3
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("padding_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

	// width adjust, IN_CH --> IN_CH_RAR
	stream<ap_uint<IN_CH_PARA*IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_CH_PARA*IN_BIT, INTER_ROW*INTER_COL>(padding_out, adj_out, reps);

	// shift reg
	stream<ap_uint<IN_CH_PARA*IN_BIT> > sr_out0("sr_out0");
    stream<ap_uint<IN_CH_PARA*IN_BIT> > sr_out1("sr_out1");
    Shift_Register_2O<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT,IN_CH_PARA >(adj_out, sr_out0, sr_out1, reps);
	
    // PE array
	_2D_PE_array_act<9*IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_CH_PARA, OUT_CH_PARA, L_SHIFT, OUT_ROW*OUT_COL>
    (sr_out0, sr_out1, weights, inc, bias, out0, out1, reps);
}


/**
 * CONV3x3, P=1, S=1
 * for the first layer
 * In: IN_BIT * IN_CH
 * Out0, Out1: OUT_BIT * OUT_CH_PARA
 */
template <
			unsigned IN_ROW,   
			unsigned IN_COL,   
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT, 

			unsigned W_BIT,
			unsigned M_BIT,   //psum width
            unsigned BN_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			
			unsigned IN_CH_PARA, 
			unsigned OUT_CH_PARA,    
			unsigned L_SHIFT>
void conv3x3_bn_act_L1(
	stream<ap_uint<IN_BIT * IN_CH> >& in, 
	const ap_uint<IN_CH_PARA*W_BIT> weights[OUT_CH_PARA][((IN_CH*9)/IN_CH_PARA)*(OUT_CH/OUT_CH_PARA)],
	const ap_int<INC_BIT> inc[OUT_CH_PARA][OUT_CH/OUT_CH_PARA],
	const ap_int<BIAS_BIT> bias[OUT_CH_PARA][OUT_CH/OUT_CH_PARA],
	stream<ap_uint<OUT_BIT*OUT_CH_PARA> >& out0,
    stream<ap_uint<OUT_BIT*OUT_CH_PARA> >& out1,
	const unsigned reps = 0)
{
#pragma HLS DATAFLOW

    // size after padding
	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// P=1, S=1, K=3
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("padding_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

	// width adjust, IN_CH --> IN_CH_RAR
	stream<ap_uint<IN_CH_PARA*IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_CH_PARA*IN_BIT, INTER_ROW*INTER_COL>(padding_out, adj_out, reps);

	// shift reg
	stream<ap_uint<IN_CH_PARA*IN_BIT> > sr_out0("sr_out0");
    stream<ap_uint<IN_CH_PARA*IN_BIT> > sr_out1("sr_out1");
    Shift_Register_2O<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT,IN_CH_PARA >(adj_out, sr_out0, sr_out1, reps);

    // PE array
	_2D_PE_array_act_L1<9*IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_CH_PARA, OUT_CH_PARA, L_SHIFT, OUT_ROW*OUT_COL>
    (sr_out0, sr_out1, weights, inc, bias, out0, out1, reps);
}

