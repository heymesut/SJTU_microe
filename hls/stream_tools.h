

#pragma once

#define AP_INT_MAX_W 2048

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>

using namespace hls;
using namespace std;

// axi data
struct my_ap_axis {
    ap_uint<64> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

/**
 * extract pixel from my_ap_axis stream type
 */
template <unsigned OutStreamW, unsigned NumLines>
void ExtractPixels(stream<my_ap_axis> &in, stream<ap_uint<OutStreamW> > &out,
                   const unsigned reps) {
    my_ap_axis temp;

    for (unsigned rep = 0; rep <  (NumLines << reps); rep++) {
#pragma HLS PIPELINE II = 1
        temp = in.read();
        out.write(temp.data(OutStreamW - 1, 0));
    }
}

/**
 * out = [0...0, in]
 */
template <unsigned InStreamW, unsigned OutStreamW, unsigned NumLines>
void AppendZeros(stream<ap_uint<InStreamW> > &in,
                 stream<ap_uint<OutStreamW> > &out, const unsigned reps) {

    ap_uint<OutStreamW> buffer;

    for (unsigned rep = 0; rep < (NumLines << reps); rep++) {
        buffer(OutStreamW - 1, InStreamW) = 0;
        buffer(InStreamW - 1, 0) = in.read();
        out.write(buffer);
    }
}

/**
 * reduce stream width ,i.e. in = [out_n ... out_1, out_0]
 */
template <unsigned int InWidth,   // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords // number of input words to process
          >
void ReduceDataWidth(stream<ap_uint<InWidth> > &in,
                     stream<ap_uint<OutWidth> > &out,
                     const unsigned int reps) {

    // emit multiple output words per input word read
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = (NumInWords * outPerIn) << reps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
        // read new input word if current out count is zero
        if (o == 0) {
            ei = in.read();
        }
        // pick output word from the rightmost position
        ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
        out.write(eo);
        // shift input to get new output word for next iteration
        ei = ei >> OutWidth;
        // increment written output count
        o++;
        // wrap around indices to recreate the nested loop structure
        if (o == outPerIn) {
            o = 0;
        }
    }
}

/**
 * expand stream width ,i.e. out = [in_n ... in_1, in_0]
 */
template <unsigned int InWidth,   // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords // number of input words to process
          >
void ExpandDataWidth(stream<ap_uint<InWidth> > &in,
                     stream<ap_uint<OutWidth> > &out,
                     const unsigned int reps) {

    // read multiple input words per output word emitted
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords << reps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
        // read input and shift into output buffer
        ap_uint<InWidth> ei = in.read();
        eo = eo >> InWidth;
        eo(OutWidth - 1, OutWidth - InWidth) = ei;
        // increment read input count
        i++;
        // wrap around logic to recreate nested loop functionality
        if (i == inPerOut) {
            i = 0;
            out.write(eo);
        }
    }
}

/**
 * expand stream width ,i.e. out = [in_n ... in_1, in_0]
 * parallel IN, serial OUT
 */
template <unsigned int InWidth,   // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords // number of input words to process (#words of one input stream)
          >
void ExpandDataWidth_PISO(stream<ap_uint<InWidth> > &in0,
                        stream<ap_uint<InWidth> > &in1,
                     stream<ap_uint<OutWidth> > &out,
                     const unsigned int reps) {

    // read multiple input words per output word emitted
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int NumOutWords = NumInWords / inPerOut;
    unsigned int i = 0;
    ap_uint<OutWidth> eo_0 = 0;
    ap_uint<OutWidth> eo_1 = 0;

    //inPerOut cycles read data and output [in0_n ... in0_1, in0_0], then one cycle output [in1_n ... in1_1, in1_0] and then repeat
    for (unsigned int t = 0; t < ((NumInWords + NumOutWords) << reps) ; t++) {
#pragma HLS PIPELINE II = 1

    	if(i < inPerOut){
            // read input and shift into output buffer
            ap_uint<InWidth> ei_0 = in0.read();
            ap_uint<InWidth> ei_1 = in1.read();
            eo_0 = eo_0 >> InWidth;
            eo_1 = eo_1 >> InWidth;
            eo_0(OutWidth - 1, OutWidth - InWidth) = ei_0;
            eo_1(OutWidth - 1, OutWidth - InWidth) = ei_1;
    	}
        // increment control count
        i++;
        // output logic
        // wrap around logic to recreate nested loop functionality
        if (i == inPerOut) {
            out.write(eo_0);
        }
        else{
            if(i == (inPerOut + 1)){
                out.write(eo_1);
                i = 0;
            }
        }
    }
}

/**
 *  data width adjust(better)
 */
template <unsigned int InWidth,   // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords // number of input words to process
          >
void StreamingDataWidthConverter_Batch(stream<ap_uint<InWidth> > &in,
                                       stream<ap_uint<OutWidth> > &out,
                                       const unsigned int reps) {
    // reduce width
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = (NumInWords * outPerIn) << reps;
        unsigned int o = 0;
        ap_uint<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word from the rightmost position
            ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wrap around indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    } else if (InWidth == OutWidth) {
        // same width, just straight-through copy
        for (unsigned int i = 0; i < (NumInWords << reps); i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<InWidth> e = in.read();
            out.write(e);
        }
    } else { // InWidth < OutWidth, expand width
        // read multiple input words per output word emitted
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords << reps;
        unsigned int i = 0;
        ap_uint<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read input and shift into output buffer
            ap_uint<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wrap around logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);
            }
        }
    }
}

/**
 * write n zero to stream
 */
template <unsigned IN_BIT>
void append_zero_to_stream(stream<ap_uint<IN_BIT> > &in, const unsigned n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
        in.write(0);
    }
}

/**
 * copy data from stream A to stream B
 */
template <unsigned StreamW, unsigned NumLines>
void StreamCopy(stream<ap_uint<StreamW> > &in, stream<ap_uint<StreamW> > &out,
                const unsigned reps) {
    ap_uint<StreamW> temp;

    for (unsigned rep = 0; rep < (NumLines << reps); rep++) {
#pragma HLS PIPELINE II = 1
        temp = in.read();
        out.write(temp);
    }
}

/**
 * 2 parallel in, serial out
 * in0 = [in0_mn,...,in0_11,in0_10,in0_0n,...,in0_01,in0_00], in1 = [in1_mn,...,in1_11,in1_10,in1_0n,...,in1_01,in1_00]
 * out = [in1_mn,...,in0_11,in0_10,in1_0n,...,in1_01,in1_00,in0_0n,...,in0_01,in0_00]
 * n: IN_CH / IN_CH_PARA - 1, i.e. IN_CH_ITER - 1
 * m: #data/IN_CH_ITER - 1
 */
template <unsigned StreamW, unsigned NumData, unsigned IN_CH_ITER> //NumData: number of input words to process (#words of one input stream)
void Stream_PISO(stream<ap_uint<StreamW> > &in0,
                stream<ap_uint<StreamW> > &in1,
                stream<ap_uint<StreamW> > &out,
                const unsigned reps) {

    ap_uint<StreamW> temp0;
    ap_uint<StreamW> temp1 [IN_CH_ITER];
    unsigned int i = 0; // control count

    for (unsigned rep = 0; rep < ((NumData * 2) << reps); rep++) {
#pragma HLS PIPELINE II = 1
        // read in0, in1 and output in0
        if(i < IN_CH_ITER){
            temp0 = in0.read();
            out.write(temp0);
            temp1[i] = in1.read();
            i++;
        }
        else
            // output in1
            if(i >= IN_CH_ITER){
                out.write(temp1[i-IN_CH_ITER]);
                i++;
            }
        // wrap around indices to recreate the nested loop structure
        if(i == (2*IN_CH_ITER))
            i = 0;
    }
}

/**
 * DEMUX3
 */
template <unsigned BIT, unsigned NumLines>
void demux_stream3(stream<ap_uint<BIT> > &in, stream<ap_uint<BIT> > &out1,
                   stream<ap_uint<BIT> > &out2, stream<ap_uint<BIT> > &out3,
                   const unsigned short which, const unsigned reps) {
    for (unsigned i = 0; i < (NumLines << reps); i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<BIT> temp = in.read();
        if (which == 0)
            out1.write(temp);
        else if (which == 1)
            out2.write(temp);
        else
            out3.write(temp);
    }
}

/**
 * DEMUX2
 */
template <unsigned BIT, unsigned NumLines>
void demux_stream2(stream<ap_uint<BIT> > &in,
                          stream<ap_uint<BIT> > &out0,
                          stream<ap_uint<BIT> > &out1,
                          const unsigned short which,
                          const unsigned reps) {
    for (unsigned i = 0; i < (NumLines << reps); i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<BIT> temp = in.read();
        if (which == 0)
            out0.write(temp);
        else
            out1.write(temp);
    }
}

/**
 * MUX2
 */
template <unsigned BIT, unsigned NumLines>
void mux_stream2(stream<ap_uint<BIT> > &out,
                          stream<ap_uint<BIT> > &in0,
                          stream<ap_uint<BIT> > &in1,
                          const unsigned short which,
                          const unsigned reps) {
    for (unsigned i = 0; i < (NumLines << reps); i++) {
#pragma HLS PIPELINE II = 1
        if (which == 0){
            ap_uint<BIT> temp = in0.read();
            out.write(temp);
        }   
        else{
            ap_uint<BIT> temp = in1.read();
            out.write(temp);            
        }
    }
}

/**
 * out = [in1, in0]
 */
template <unsigned InStream0W, unsigned InStream1W, unsigned NumLines>
void StreamConcat(stream<ap_uint<InStream0W> > &in0,
                 stream<ap_uint<InStream1W> > &in1,
                 stream<ap_uint<InStream0W + InStream1W> > &out, const unsigned reps) {

    ap_uint<InStream0W + InStream1W> buffer;

    for (unsigned rep = 0; rep < (NumLines << reps); rep++) {
        buffer((InStream0W - 1), 0) = in0.read();
        buffer((InStream0W + InStream1W - 1), InStream0W) = in1.read();
        out.write(buffer);
    }
}

/**
 * out1 = in, out0 = in 
 */
template <unsigned InStreamW,  unsigned NumLines>
void Stream_Broadcast(stream<ap_uint<InStreamW> > &in,
                 stream<ap_uint<InStreamW> > &out0,
                 stream<ap_uint<InStreamW> > &out1, const unsigned reps) {

    ap_uint<InStreamW> buffer;

    for (unsigned rep = 0; rep < (NumLines << reps); rep++) {
#pragma HLS PIPELINE II = 1
        buffer = in.read();
        out0.write(buffer);
        out1.write(buffer);
    }
}

/**
 * mem to stream
 */
template <unsigned LineWidth, unsigned NumLines>
void Mem2Stream(ap_uint<LineWidth> *in, stream<ap_uint<LineWidth> > &out,
                const unsigned reps) {
    for (unsigned i = 0; i < (NumLines << reps); i++) {
#pragma HLS PIPELINE II = 1
        out.write(in[i]);
    }
}

/**
 * stream to mem 
 */
template <unsigned LineWidth, unsigned NumLines>
void Stream2Mem(stream<ap_uint<LineWidth> > &in, ap_uint<LineWidth> *out,
                const unsigned reps) {
    for (unsigned i = 0; i < (NumLines << reps); i++) {
#pragma HLS PIPELINE II = 1
        out[i] = in.read();
    }
}

/**
 * stream to mat 
 */
template <unsigned IMAGE_HEIGHT, unsigned IMAGE_WIDTH>
void stream_to_mat (stream<ap_uint<24> >&in, 
		 Mat<IMAGE_HEIGHT, IMAGE_WIDTH, HLS_8UC3> & raw_img) {
    
	for (int i=0; i<IMAGE_HEIGHT; i++) {
		for (int j=0; j<IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            Scalar<3, ap_uint<8> > pix;
            ap_uint<24> in_data = in.read();
            for (unsigned int p=0; p < 3; p ++) {
                pix.val[p] = in_data(8*p+7, 8*p);
            }
			raw_img << pix;
		}	
	}
}

/**
 * mat to stream 
 */
template <unsigned IMAGE_HEIGHT, unsigned IMAGE_WIDTH>
void mat_to_stream (Mat<IMAGE_HEIGHT, IMAGE_WIDTH, HLS_8UC3> & resize_img,
                    stream<ap_uint<24> > & out ) {
    
	for (int i=0; i<IMAGE_HEIGHT; i++) {
		for (int j=0; j<IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            Scalar<3, ap_uint<8> > pix;
            resize_img >> pix;
            ap_uint<24> out_data;
            for (unsigned int p=0; p < 3; p ++) {
                out_data(8*p+7, 8*p) = pix.val[p];
            }
            out.write(out_data);
		}	
	}
}

/**
 * read nums data from axis port to stream
 */
template <unsigned BIT>
void in_to_stream(stream<my_ap_axis> &in, stream<ap_uint<BIT> > &out,
                  const unsigned nums = 1) {
    my_ap_axis temp;
    for (unsigned num = 0; num < nums; num++) {
        temp = in.read();
        out.write(temp.data(BIT - 1, 0));
    }
}

/**
 * write nums data from stream to axis port
 */
template <unsigned BIT>
void stream_to_out(stream<ap_uint<BIT> > &in, stream<my_ap_axis> &out,
                   const unsigned nums = 1) {
    my_ap_axis temp;
    temp.keep = "0xffffffffffffffff";

    for (unsigned i = 0; i < nums - 1; i++) {
#pragma HLS PIPELINE II = 1
        temp.data = in.read();
        temp.last = 0;
        out.write(temp);
    }

    temp.data = in.read();
    temp.last = 1;
    out.write(temp);
}

/**
 * stream ap_int type to my_ap_axis stream type (another stream_to_out)
 */
template <unsigned NumLines>
void AddLast(stream<ap_uint<64> > &in, stream<my_ap_axis> &out,
             const unsigned reps) {
    my_ap_axis temp;
    temp.keep = 0xff;

    for (unsigned i = 0; i < ((NumLines << reps) - 1); i++) {
#pragma HLS PIPELINE II = 1
        temp.data = in.read();
        temp.last = 0;
        out.write(temp);
    }

    temp.data = in.read();
    temp.last = 1;
    out.write(temp);
}
