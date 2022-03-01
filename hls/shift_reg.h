

#pragma once

#include <ap_int.h>
#include <hls_stream.h>

using namespace std;
using namespace hls;

/*
* shift register with 1 in and 2 out
* for CONV
* replace order : (first) CH, inner_window, window (last)
*/
template <	unsigned K, //kernel size
			unsigned S, //stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned IN_CH_PARA >
void Shift_Register_2O(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in, 
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& out0,
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& out1,    
	const unsigned reps = 0)
{

    // #IN_CH iteration
    const unsigned IN_CH_ITER = IN_CH / IN_CH_PARA;
	// #shift down
	const unsigned ROW_STEPS = (IN_ROW-K) / S + 1;
	// #shift right
	const unsigned COL_STEPS = ((IN_COL-K) / S + 1) / 2;

	// use BRAM to imitate the behavior of the shift register 
    //shift register size ((K - 1) * IN_COL + K + S) * IN_CH_ITER
    //shift right first
    //TO-DO: shift down first better 
	const unsigned BUF_SIZE = ((K - 1) * IN_COL + K + S) * IN_CH_ITER;
	ap_uint<IN_CH_PARA*IN_BIT> shift_reg[BUF_SIZE];
#pragma HLS RESOURCE variable shift_reg core=RAM_2P

	unsigned buf_len = 0;
	unsigned buf_pointer = 0;
	ap_uint<IN_CH_PARA*IN_BIT> temp_in;

	// counter
	unsigned right_slid = 0;
	unsigned down_slid = 0;
    
	// total iteration , i.e. #input data
	for(unsigned rep=0; rep < ((IN_ROW*IN_COL*IN_CH_ITER) << reps); rep ++) {
		// write data to shift reg and shift
		if(buf_len < BUF_SIZE) {
			temp_in = in.read();
			shift_reg[buf_pointer++] = temp_in;
		// when shift reg is full for the first time, shift data out of the reg and write new data in since next cycle;
		// (i.e. write new data from the beginning of the BRAM again) 
		// when all the old data are shifted out, the pointer move to the beginning of the BRAM and repeat
			if(buf_pointer == BUF_SIZE) {
				buf_pointer = 0;
			}
			buf_len ++;
		}

		// when the shift reg is full, then read two conv window in K x K x IN_CH_ITER cycles simultaneously
		if(buf_len == BUF_SIZE) {

			for(unsigned i=0; i < K; i ++) {
				for(unsigned j=0; j < K; j ++) {
					for(unsigned c=0; c < IN_CH_ITER; c++){
#pragma HLS PIPELINE II = 1
					// find the correct data (buf_pointer point to the oldest data)
					unsigned temp_pointer0 = (buf_pointer + (i * IN_COL * IN_CH_ITER) + (j * IN_CH_ITER) + c);
					unsigned temp_pointer1 = (buf_pointer + (i * IN_COL * IN_CH_ITER) + ((j + S) * IN_CH_ITER) + c);

					if(temp_pointer0 >= BUF_SIZE) {
						temp_pointer0 -= BUF_SIZE;
					}
					if(temp_pointer1 >= BUF_SIZE) {
						temp_pointer1 -= BUF_SIZE;
					}
					
					ap_uint<IN_CH_PARA*IN_BIT> temp_out0 = shift_reg[temp_pointer0];
					out0.write(temp_out0);
					ap_uint<IN_CH_PARA*IN_BIT> temp_out1 = shift_reg[temp_pointer1];
					out1.write(temp_out1);
					}
				}
			}

			// shift right first
			// shift right to the end 
			if(++ right_slid == COL_STEPS) {
				right_slid = 0;
				//shift down to the end and finish one map
				if(++ down_slid == ROW_STEPS) {
					down_slid = 0;
					buf_len = 0;
					} 
				else {
					// just shift right to the end
					buf_len = buf_len - ((S-1) * IN_COL + (K + S)) * IN_CH_ITER;
					}
				} 
			else {
				buf_len -= (S * 2) * IN_CH_ITER;
				}
		}
	}
}


/*
* shift register with 1 in and 1 out
* for MAXPOOL and REORG
* replace order : (first) inner_window, CH, window (last)
*/
template <	unsigned K, //kernel size
			unsigned S, //stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned IN_CH_PARA>
void Shift_Register_1O(
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& in, 
	stream<ap_uint<IN_CH_PARA*IN_BIT> >& out,    
	const unsigned reps = 0)
{

    // #IN_CH iteration
    const unsigned IN_CH_ITER = IN_CH / IN_CH_PARA;
	// #shift down
	const unsigned ROW_STEPS = (IN_ROW-K) / S + 1;
	// #shift right
	const unsigned COL_STEPS = (IN_COL-K) / S + 1;

	// use BRAM to imitate the behavior of the shift register 
    //shift register size ((K - 1) * IN_COL + K) * IN_CH_ITER
    //shift right first
    //TO-DO: shift down first better 
	const unsigned BUF_SIZE = ((K - 1) * IN_COL + K ) * IN_CH_ITER;
	ap_uint<IN_CH_PARA*IN_BIT> shift_reg[BUF_SIZE];
#pragma HLS RESOURCE variable shift_reg core=RAM_S2P_BRAM

	unsigned buf_len = 0;
	unsigned buf_pointer = 0;
	ap_uint<IN_CH_PARA*IN_BIT> temp_in;

	// counter
	unsigned right_slid = 0;
	unsigned down_slid = 0;
    
	// total iteration , i.e. #input data
	for(unsigned rep=0; rep < ((IN_ROW*IN_COL*IN_CH_ITER) << reps); rep ++) {
		// write data to shift reg and shift
		if(buf_len < BUF_SIZE) {
			temp_in = in.read();
			shift_reg[buf_pointer++] = temp_in;
		// when shift reg is full for the first time, shift data out of the reg and write new data in since next cycle;
		// (i.e. write new data from the beginning of the BRAM again) 
		// when all the old data are shifted out, the pointer move to the beginning of the BRAM and repeat
			if(buf_pointer == BUF_SIZE) {
				buf_pointer = 0;
			}
			buf_len ++;
		}

		// when the shift reg is full, then read 1 window in K x K x IN_CH_ITER cycles
		if(buf_len == BUF_SIZE) {

			for(unsigned c=0; c < IN_CH_ITER; c++){
				for(unsigned i=0; i < K; i ++) {
					for(unsigned j=0; j < K; j ++) {
#pragma HLS PIPELINE II = 1
					// find the correct data (buf_pointer point to the oldest data)
					unsigned temp_pointer = (buf_pointer + (i * IN_COL * IN_CH_ITER) + (j * IN_CH_ITER) + c);

					if(temp_pointer >= BUF_SIZE) {
						temp_pointer -= BUF_SIZE;
					}
					
					ap_uint<IN_CH_PARA*IN_BIT> temp_out = shift_reg[temp_pointer];
					out.write(temp_out);
					}
				}
			}

			// shift right first
			// shift right to the end 
			if(++ right_slid == COL_STEPS) {
				right_slid = 0;
				//shift down to the end and finish one map
				if(++ down_slid == ROW_STEPS) {
					down_slid = 0;
					buf_len = 0;
					} 
				else {
					// just shift right to the end
					buf_len = buf_len - ((S-1) * IN_COL + K ) * IN_CH_ITER;
					}
				} 
			else {
				buf_len -= (S * IN_CH_ITER);
				}
		}
	}
}
