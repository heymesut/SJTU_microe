

#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include "function.h"

using namespace hls;
using namespace std;

/*
* IN_CH_PARA DSPs to calculate IN_CH_PARA IN_CHs in parallel;
* 4 MUL in 1 DSP; (2 CONV windows and 2 OUT_CHs in parallel);
*/
template <	unsigned W_BIT,
			unsigned IN_ACT_BIT,
			unsigned IN_CH_PARA>
ap_int<45>  _1D_PE_array(
	ap_uint<IN_CH_PARA*W_BIT> weight0,
    ap_uint<IN_CH_PARA*W_BIT> weight1,
	ap_uint<IN_CH_PARA*IN_ACT_BIT> in_act0,
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_act1)
{
	ap_int<45> accumulation_shrink = 0;

	for (unsigned p = 0; p < IN_CH_PARA; p++) {
#pragma HLS UNROLL
        ap_int<W_BIT>       temp_w0      = weight0((p+1)*W_BIT-1, p*W_BIT);
        ap_int<W_BIT>       temp_w1      = weight1((p+1)*W_BIT-1, p*W_BIT);
        ap_uint<IN_ACT_BIT> temp_in_act0 = in_act0((p+1)*IN_ACT_BIT-1, p*IN_ACT_BIT);
        ap_uint<IN_ACT_BIT> temp_in_act1 = in_act1((p+1)*IN_ACT_BIT-1, p*IN_ACT_BIT);

		ap_int<18>  weight_shrink = (ap_int<18>(temp_w1) << 11) + temp_w0; //two weights in one operand
        ap_uint<27> in_act_shrink = (ap_uint<27>(temp_in_act1) << 22) + temp_in_act0; //two in_acts in one operand

		ap_int<45> result_shrink = weight_shrink * in_act_shrink;
		accumulation_shrink += result_shrink;
	}

    return accumulation_shrink;
}

/*
* IN_CH_PARA x (OUT_CH_PARA/2) DSPs to calculate IN_CH_PARA IN_CHs x (OUT_CH_PARA/2) OUT_CHs in parallel;
* (actual parallelism: IN_CH_PARA x OUT_CH_PARA x 2 due to 4 MUL in 1 DSP)
* Replacement Order: [last] conv window, OUT_CH, IN_CH, inner_core [first];
* IN BUF width: IN_CH_PARA x IN_ACT_BIT, depth: K x K x IN_CH / IN_CH_PARA; 
*/
template <	unsigned MAT_ROW,		// K x K x IN_CH
			unsigned MAT_COL,		// OUT_CH

			unsigned IN_ACT_BIT,
            unsigned OUT_ACT_BIT,   // width after BN and quant_act

			unsigned W_BIT,
			unsigned M_BIT,			// psum width

			unsigned INC_BIT,	 
			unsigned BIAS_BIT,
            unsigned BN_BIT,	   

			unsigned IN_CH_PARA,     // IN_CH  Parallelism      
			unsigned OUT_CH_PARA,    // OUT_CH Parallelism     
			unsigned L_SHIFT,
			unsigned OUT_ACT_NUMS>  // OUT_ROW*OUT_COL
void _2D_PE_array_act(
	stream<ap_uint<IN_CH_PARA*IN_ACT_BIT> >& in_act0,
    stream<ap_uint<IN_CH_PARA*IN_ACT_BIT> >& in_act1,
	const ap_uint<IN_CH_PARA*W_BIT> weights[OUT_CH_PARA][(MAT_ROW/IN_CH_PARA)*(MAT_COL/OUT_CH_PARA)], 
	const ap_int<INC_BIT>  inc[OUT_CH_PARA][MAT_COL/OUT_CH_PARA],   // for BN
	const ap_int<BIAS_BIT> bias[OUT_CH_PARA][MAT_COL/OUT_CH_PARA], // for BN
	stream<ap_uint<OUT_CH_PARA*OUT_ACT_BIT> >& out_act0, 
    stream<ap_uint<OUT_CH_PARA*OUT_ACT_BIT> >& out_act1,
	const unsigned reps = 0   // log2(batch size)
    ) 
{

    const unsigned IN_CH_ITER  = MAT_ROW/IN_CH_PARA;    // #IN_CH iteration x K x K
	const unsigned OUT_CH_ITER = MAT_COL/OUT_CH_PARA;  //  #OUT_CH iteration
    const unsigned total_iter  = ((IN_CH_ITER * OUT_CH_ITER * OUT_ACT_NUMS)/2) << reps;  // #iteration of calculating 2^reps fmap_out

    // in_buffer, store (2 x K x K x IN_CH) in_act for "data reuse"
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer0[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer0 core=RAM_S2P_BRAM
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer1[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer1 core=RAM_S2P_BRAM

    unsigned in_ch_iter_cnt  = 0; // inter_core iteration included 
    unsigned out_ch_iter_cnt = 0;  
    unsigned index = 0;  //weights index

    ap_uint<IN_CH_PARA*IN_ACT_BIT> temp_in_act0;
    ap_uint<IN_CH_PARA*IN_ACT_BIT> temp_in_act1;

    ap_int<M_BIT> acc0[OUT_CH_PARA]; 
    ap_int<M_BIT> acc1[OUT_CH_PARA];  //psum buffer 

	for (unsigned iter = 0; iter < total_iter; iter++){
#pragma HLS PIPELINE II=1

        // read data in the first iteration of OUT_CH_ITER, then reuse
		if (out_ch_iter_cnt == 0) {
			temp_in_act0 = in_act0.read();
            temp_in_act1 = in_act1.read();
			in_buffer0[in_ch_iter_cnt] = temp_in_act0;
            in_buffer1[in_ch_iter_cnt] = temp_in_act1;
		}
		else {
			temp_in_act0 = in_buffer0[in_ch_iter_cnt];
            temp_in_act1 = in_buffer1[in_ch_iter_cnt];
		}

        // one OUT CH iteration done, clear acc
        if(in_ch_iter_cnt == 0){
            for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL
                acc0[i] = 0;
                acc1[i] = 0;                
            }
        }

        // 2D PE array
        // OUT CH Parallel
        for(unsigned i = 0 ; i < (OUT_CH_PARA/2) ; i++){
#pragma HLS UNROLL

        // read weights
        ap_uint<IN_CH_PARA*W_BIT> temp_weights0 = weights[2*i][index];
        ap_uint<IN_CH_PARA*W_BIT> temp_weights1 = weights[2*i+1][index];

        // 1D PE array and 4 MUL in 1 DSP
        ap_int<45> acc_shrink = _1D_PE_array<W_BIT, IN_ACT_BIT, IN_CH_PARA>( temp_weights0, temp_weights1, temp_in_act0, temp_in_act1 );
        
        //shrink word extraction
        ap_int<11> acc_temp_11 = acc_shrink(43, 33) + acc_shrink[32];  
        acc1[2*i+1] += acc_temp_11;        
        ap_int<11> acc_temp_10 = acc_shrink(32, 22) + acc_shrink[21];  
        acc1[2*i] += acc_temp_10;  
        ap_int<11> acc_temp_01 = acc_shrink(21, 11) + acc_shrink[10];  
        acc0[2*i+1] += acc_temp_01;
        ap_int<11> acc_temp_00 = acc_shrink(10, 0);  
        acc0[2*i] += acc_temp_00;        
        }

        // replacement order and output control
        index ++;
        in_ch_iter_cnt ++;

        if(in_ch_iter_cnt == IN_CH_ITER){
        //one OUT CH iteration done(i.e. all IN CH iteration done), output out_act and start next OUT CH iteration
           in_ch_iter_cnt = 0;

        //BN, ReLU and output out_act
           ap_uint<OUT_CH_PARA*OUT_ACT_BIT> out_buffer0;
           ap_uint<OUT_CH_PARA*OUT_ACT_BIT> out_buffer1;

           for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL
            // BN + QUANT ReLU
            out_buffer0((i+1)*OUT_ACT_BIT-1, i*OUT_ACT_BIT) = BN_QUReLU<M_BIT, OUT_ACT_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_ACT_BIT, W_BIT, L_SHIFT>(acc0[i], inc[i][out_ch_iter_cnt], bias[i][out_ch_iter_cnt]);
            out_buffer1((i+1)*OUT_ACT_BIT-1, i*OUT_ACT_BIT) = BN_QUReLU<M_BIT, OUT_ACT_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_ACT_BIT, W_BIT, L_SHIFT>(acc1[i], inc[i][out_ch_iter_cnt], bias[i][out_ch_iter_cnt]);    
           }
           out_act0.write(out_buffer0);
           out_act1.write(out_buffer1);

           out_ch_iter_cnt ++;

           if(out_ch_iter_cnt == OUT_CH_ITER){
            //one out_act done, start next point
               out_ch_iter_cnt = 0;
               index = 0;
           }

        }

    }   

}

// ====================================================================================================
//                                     FOR Layer1 
// ====================================================================================================

/*
* IN_CH_PARA DSPs to calculate IN_CH_PARA IN_CHs in parallel;
* for Layer1,ACT_IN_BIT = 8
* 2 MUL in 1 DSP; (2 CONV windows);
*/
template <	unsigned W_BIT,
			unsigned IN_ACT_BIT,
			unsigned IN_CH_PARA>
ap_int<45>  _1D_PE_array_L1(
	ap_uint<IN_CH_PARA*W_BIT> weight,
	ap_uint<IN_CH_PARA*IN_ACT_BIT> in_act0,
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_act1)
{
	ap_int<45> accumulation_shrink = 0;

	for (unsigned p = 0; p < IN_CH_PARA; p++) {
#pragma HLS UNROLL
        ap_int<W_BIT>       temp_w       = weight((p+1)*W_BIT-1, p*W_BIT);
        ap_uint<IN_ACT_BIT> temp_in_act0 = in_act0((p+1)*IN_ACT_BIT-1, p*IN_ACT_BIT);
        ap_uint<IN_ACT_BIT> temp_in_act1 = in_act1((p+1)*IN_ACT_BIT-1, p*IN_ACT_BIT);

        ap_uint<27> in_act_shrink = (ap_uint<27>(temp_in_act1) << 16) + temp_in_act0; //two in_acts in one operand

		ap_int<45> result_shrink = temp_w * in_act_shrink;
		accumulation_shrink += result_shrink;
	}

    return accumulation_shrink;
}

/*
* IN_CH_PARA x OUT_CH_PARA DSPs to calculate IN_CH_PARA IN_CHs x OUT_CH_PARA OUT_CHs in parallel;
* (actual parallelism: IN_CH_PARA x OUT_CH_PARA x 2 due to 2 MUL in 1 DSP)
* Replacement Order: [last] conv window, OUT_CH, IN_CH, inner_core [first];
* IN BUF width: IN_CH_PARA x IN_ACT_BIT, depth: K x K x IN_CH / IN_CH_PARA; 
*/
template <	unsigned MAT_ROW,		// K x K x IN_CH
			unsigned MAT_COL,		// OUT_CH

			unsigned IN_ACT_BIT,
            unsigned OUT_ACT_BIT,   // width after BN and quant_act

			unsigned W_BIT,
			unsigned M_BIT,			// psum width

			unsigned INC_BIT,	 
			unsigned BIAS_BIT,
            unsigned BN_BIT,	   

			unsigned IN_CH_PARA,     // IN_CH  Parallelism      
			unsigned OUT_CH_PARA,    // OUT_CH Parallelism     
			unsigned L_SHIFT,
			unsigned OUT_ACT_NUMS>  // OUT_ROW*OUT_COL
void _2D_PE_array_act_L1(
	stream<ap_uint<IN_CH_PARA*IN_ACT_BIT> >& in_act0,
    stream<ap_uint<IN_CH_PARA*IN_ACT_BIT> >& in_act1,
	const ap_uint<IN_CH_PARA*W_BIT> weights[OUT_CH_PARA][(MAT_ROW/IN_CH_PARA)*(MAT_COL/OUT_CH_PARA)], 
	const ap_int<INC_BIT>  inc[OUT_CH_PARA][MAT_COL/OUT_CH_PARA],   // for BN
	const ap_int<BIAS_BIT> bias[OUT_CH_PARA][MAT_COL/OUT_CH_PARA], // for BN
	stream<ap_uint<OUT_CH_PARA*OUT_ACT_BIT> >& out_act0, 
    stream<ap_uint<OUT_CH_PARA*OUT_ACT_BIT> >& out_act1,
	const unsigned reps = 0   // log2(batch size)
    ) 
{

    const unsigned IN_CH_ITER  = MAT_ROW/IN_CH_PARA;    // #IN_CH iteration x K x K
	const unsigned OUT_CH_ITER = MAT_COL/OUT_CH_PARA;  //  #OUT_CH iteration
    const unsigned total_iter  = (IN_CH_ITER * OUT_CH_ITER * OUT_ACT_NUMS / 2) << reps;  // #iteration of calculating 2^reps fmap_out

    // in_buffer, store (2 x K x K x IN_CH) in_act for "data reuse"
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer0[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer0 core=RAM_S2P_BRAM
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer1[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer1 core=RAM_S2P_BRAM

    unsigned in_ch_iter_cnt  = 0; // inter_core iteration included 
    unsigned out_ch_iter_cnt = 0;  
    unsigned index = 0;  //weights index

    ap_uint<IN_CH_PARA*IN_ACT_BIT> temp_in_act0;
    ap_uint<IN_CH_PARA*IN_ACT_BIT> temp_in_act1;

    ap_int<M_BIT> acc0[OUT_CH_PARA]; 
    ap_int<M_BIT> acc1[OUT_CH_PARA];  //psum buffer 

	for (unsigned iter = 0; iter < total_iter; iter++){
#pragma HLS PIPELINE II=1

        // read data in the first iteration of OUT_CH_ITER, then reuse
		if (out_ch_iter_cnt == 0) {
			temp_in_act0 = in_act0.read();
            temp_in_act1 = in_act1.read();
			in_buffer0[in_ch_iter_cnt] = temp_in_act0;
            in_buffer1[in_ch_iter_cnt] = temp_in_act1;
		}
		else {
			temp_in_act0 = in_buffer0[in_ch_iter_cnt];
            temp_in_act1 = in_buffer1[in_ch_iter_cnt];
		}

        // one OUT CH iteration done, clear acc
        if(in_ch_iter_cnt == 0){
            for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL
                acc0[i] = 0;
                acc1[i] = 0;                
            }
        }

        // 2D PE array
        // OUT CH Parallel
        for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL

        // read weights
        ap_uint<IN_CH_PARA*W_BIT> temp_weights = weights[i][index];

        // 1D PE array and 2 MUL in 1 DSP
        ap_int<45> acc_shrink = _1D_PE_array_L1<W_BIT, IN_ACT_BIT, IN_CH_PARA>( temp_weights, temp_in_act0, temp_in_act1 );
        
        //shrink word extraction     
        ap_int<16> acc_temp_1 = acc_shrink(31, 16) + acc_shrink[15];  
        acc1[i] += acc_temp_1;  
        ap_int<16> acc_temp_0 = acc_shrink(15, 0);  
        acc0[i] += acc_temp_0;        
        }

        // replacement order and output control
        
        index ++;
        in_ch_iter_cnt ++;

        if(in_ch_iter_cnt == IN_CH_ITER){
        //one OUT CH iteration done(i.e. all IN CH iteration done), output out_act and start next OUT CH iteration
           in_ch_iter_cnt = 0;

        //BN, ReLU and output out_act
           ap_uint<OUT_CH_PARA*OUT_ACT_BIT> out_buffer0;
           ap_uint<OUT_CH_PARA*OUT_ACT_BIT> out_buffer1;

           for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL
            // BN + QUANT ReLU
            out_buffer0((i+1)*OUT_ACT_BIT-1, i*OUT_ACT_BIT) = BN_QUReLU<M_BIT, OUT_ACT_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_ACT_BIT, W_BIT, L_SHIFT>(acc0[i], inc[i][out_ch_iter_cnt], bias[i][out_ch_iter_cnt]);
            out_buffer1((i+1)*OUT_ACT_BIT-1, i*OUT_ACT_BIT) = BN_QUReLU<M_BIT, OUT_ACT_BIT, INC_BIT, BIAS_BIT, BN_BIT, IN_ACT_BIT, W_BIT, L_SHIFT>(acc1[i], inc[i][out_ch_iter_cnt], bias[i][out_ch_iter_cnt]);    
           }
           out_act0.write(out_buffer0);
           out_act1.write(out_buffer1);

           out_ch_iter_cnt ++;

           if(out_ch_iter_cnt == OUT_CH_ITER){
            //one out_act done, start next point
               out_ch_iter_cnt = 0;
               index = 0;
           }

        }

    }   

}


