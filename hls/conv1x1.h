

#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "PE_array.h"

using namespace hls;
using namespace std;

/**
 * CONV1x1
 * just a PE array
 * In0, In1: IN_CH_PARA * IN_BIT
 * Out0, Out1: OUT_CH_PARA * M_BIT
 */
template <
			unsigned IN_ROW,   
			unsigned IN_COL,   
			unsigned IN_CH,
			unsigned IN_ACT_BIT,
			unsigned OUT_CH,

			unsigned W_BIT,
			unsigned M_BIT,   //psum width
			
			unsigned IN_CH_PARA,   // IN_CH  Parallelism
			unsigned OUT_CH_PARA>  // OUT_CH Parallelism
void conv1x1(
	stream<ap_uint<IN_CH_PARA * IN_ACT_BIT> >& in_act0,
    stream<ap_uint<IN_CH_PARA * IN_ACT_BIT> >& in_act1,
	const ap_uint<IN_CH_PARA*W_BIT> weights[OUT_CH_PARA][(IN_CH/IN_CH_PARA)*(OUT_CH/OUT_CH_PARA)],
	stream<ap_uint<M_BIT*OUT_CH_PARA> >& out0,
    stream<ap_uint<M_BIT*OUT_CH_PARA> >& out1,
	const unsigned reps = 0   // log2(batch size)
    )
{
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

    const unsigned IN_CH_ITER  = IN_CH/IN_CH_PARA;    // #IN_CH iteration
	const unsigned OUT_CH_ITER = OUT_CH/OUT_CH_PARA;  //  #OUT_CH iteration
    const unsigned total_iter  = ((IN_CH_ITER * OUT_CH_ITER * OUT_ROW * OUT_COL)/2) << reps;  // #iteration of calculating 2^reps fmap_out

    // in_buffer, store (2 x IN_CH) in_act for "data reuse"
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer0[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer0 core=RAM_S2P_BRAM
    ap_uint<IN_CH_PARA*IN_ACT_BIT> in_buffer1[IN_CH_ITER];
#pragma HLS RESOURCE variable=in_buffer1 core=RAM_S2P_BRAM

    unsigned in_ch_iter_cnt  = 0; 
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
        //one OUT CH iteration done(i.e. all IN CH iteration done), output and start next OUT CH iteration
           in_ch_iter_cnt = 0;

        //output
           ap_uint<OUT_CH_PARA*M_BIT> out_buffer0;
           ap_uint<OUT_CH_PARA*M_BIT> out_buffer1;

           for(unsigned i = 0 ; i < OUT_CH_PARA ; i++){
#pragma HLS UNROLL
            out_buffer0((i+1)*M_BIT-1, i*M_BIT) = acc0[i];
            out_buffer1((i+1)*M_BIT-1, i*M_BIT) = acc1[i];    
           }
           out0.write(out_buffer0);
           out1.write(out_buffer1);

           out_ch_iter_cnt ++;

           if(out_ch_iter_cnt == OUT_CH_ITER){
            //one point done, start next point
               out_ch_iter_cnt = 0;
               index = 0;
           }

        }

    }   
}
