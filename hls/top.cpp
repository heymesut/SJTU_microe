
#define AP_INT_MAX_W 2048

//#define DEBUG

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>

#ifdef DEBUG
#include <iostream>
#include <fstream>
#endif

#include "config.h"
#include "param.h"
#include "conv3x3.h"
#include "conv1x1.h"
#include "function.h"
#include "maxpool.h"
#include "reorg.h"
#include "stream_tools.h"

#define IN_IMAGE_WIDTH  640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160

#define M_BIT         24
#define BN_BIT        40
#define M_BIT_CONV1x1 32

using namespace hls;
using namespace std;

#ifdef DEBUG
void write_data(const char *path, char *ptr, unsigned int size);
#endif

void UltraNet_Bypass(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps) { //reps = log2(batch size)
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1

//================================================
// compute
//================================================
#pragma HLS DATAFLOW

// pre process
    const unsigned int num_per_rep = 360 * 640 * 3 * 8 / 64;

    stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable=in_stream_extract depth=16 dim=1
	ExtractPixels<64, num_per_rep> (in, in_stream_extract, reps);

    stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable=in_stream0 depth=16 dim=1
    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);

	stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream1("in_stream1");
#pragma HLS STREAM variable=in_stream1 depth=16 dim=1
	StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT * CONV_0_IFM_CH, num_per_rep / 3> (in_stream0, in_stream1, reps);

    stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream2("in_stream2");
#pragma HLS STREAM variable=in_stream2 depth=16 dim=1
    resize_batch<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH>(in_stream1, in_stream2, reps);

#ifdef DEBUG
    cout << "img after resize size " << in_stream2.size() << endl;

    ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> img_resize[CONV_0_IFM_ROW][CONV_0_IFM_COL];
    for(unsigned i = 0; i<CONV_0_IFM_ROW; i++)
        for(unsigned j = 0; j<CONV_0_IFM_COL; j++){
            ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> temp = in_stream2.read();
            img_resize[i][j] = temp;
            in_stream2.write(temp);
        }
    write_data("./test_img/0_res/0_resize.bin", (char *) img_resize, sizeof(img_resize));
#endif

// UltraNet-Bypass
// CONV0
    stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> >  conv_0_out0("conv_0_out0");
#pragma HLS STREAM variable=conv_0_out0 depth=256 dim=1
    stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> >  conv_0_out1("conv_0_out1");
#pragma HLS STREAM variable=conv_0_out1 depth=256 dim=1
    conv3x3_bn_act_L1< 
                    CONV_0_IFM_ROW,
                    CONV_0_IFM_COL,
                    CONV_0_IFM_CH,
                    CONV_0_IN_BIT,

                    CONV_0_OFM_CH,
                    CONV_0_OUT_BIT,

                    CONV_0_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_0_INC_BIT,
                    CONV_0_BIAS_BIT,

                    CONV_0_IN_CH_PARA,
                    CONV_0_OUT_CH_PARA,
                    CONV_0_L_SHIFT>(
                in_stream2,
                conv_0_w,
                conv_0_inc,
                conv_0_bias,
                conv_0_out0,
                conv_0_out1,
                reps );

#ifdef DEBUG

    ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> conv0_out0[CONV_0_IFM_ROW][CONV_0_IFM_COL/2][CONV_0_OFM_CH/CONV_0_OUT_CH_PARA];
    ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> conv0_out1[CONV_0_IFM_ROW][CONV_0_IFM_COL/2][CONV_0_OFM_CH/CONV_0_OUT_CH_PARA];
    for(unsigned i = 0; i<CONV_0_IFM_ROW; i++)
        for(unsigned j = 0; j<(CONV_0_IFM_COL/2); j++)
            for(unsigned k = 0; k<(CONV_0_OFM_CH/CONV_0_OUT_CH_PARA); k++){
            ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> temp0 = conv_0_out0.read();
            ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> temp1 = conv_0_out1.read();
            conv0_out0[i][j][k] = temp0;
            conv0_out1[i][j][k] = temp1;
            conv_0_out0.write(temp0);
            conv_0_out1.write(temp1);
        }
    write_data("./test_img/0_res/0_conv0_out0.bin", (char *) conv0_out0, sizeof(conv0_out0));
    write_data("./test_img/0_res/0_conv0_out1.bin", (char *) conv0_out1, sizeof(conv0_out1));
#endif

// PARA --> SERIAL
    stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> >  conv_0_out_serial("conv_0_out_serial");
    const unsigned int numData0 = CONV_0_OFM_ROW * CONV_0_OFM_COL * CONV_0_OFM_CH / CONV_0_OUT_CH_PARA;
    Stream_PISO< CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA, numData0/2, CONV_0_OFM_CH / CONV_0_OUT_CH_PARA > 
    (conv_0_out0, conv_0_out1, conv_0_out_serial, reps);

#ifdef DEBUG
    cout << "conv0 out size " << conv_0_out_serial.size() << endl;

    ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> conv0_out_s[CONV_0_IFM_ROW][CONV_0_IFM_COL][CONV_0_OFM_CH/CONV_0_OUT_CH_PARA];
    for(unsigned i = 0; i<CONV_0_IFM_ROW; i++)
        for(unsigned j = 0; j<CONV_0_IFM_COL; j++)
            for(unsigned k = 0; k<(CONV_0_OFM_CH/CONV_0_OUT_CH_PARA); k++){
            ap_uint<CONV_0_OUT_BIT * CONV_0_OUT_CH_PARA> temp = conv_0_out_serial.read();
            conv0_out_s[i][j][k] = temp;
            conv_0_out_serial.write(temp);
        }
    write_data("./test_img/0_res/0_conv0_out_serial.bin", (char *) conv0_out_s, sizeof(conv0_out_s));
#endif

// MAXPOOL0
    stream<ap_uint<CONV_0_OUT_BIT*CONV_0_OUT_CH_PARA> > pool_0_out("pool_0_out");
#pragma HLS STREAM variable=pool_0_out depth=256 dim=1
    MaxPool_2D< CONV_0_OFM_ROW,
                CONV_0_OFM_COL,
                CONV_0_OFM_CH,
                CONV_0_OUT_CH_PARA,
                CONV_0_OUT_BIT>(
                    conv_0_out_serial,
                    pool_0_out,
                    reps);

#ifdef DEBUG
    cout << "pool0 out size " << pool_0_out.size() << endl;
#endif

// expand width
    stream<ap_uint<CONV_0_OUT_BIT*CONV_0_OFM_CH> > pool_0_out_expand("pool_0_out_expand");
    const unsigned int numData0_after_pool = numData0 / 4;
    StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT*CONV_0_OUT_CH_PARA, CONV_0_OUT_BIT*CONV_0_OFM_CH, numData0_after_pool> 
    (pool_0_out, pool_0_out_expand, reps);

#ifdef DEBUG
    cout << "pool0 out(expand width) size " << pool_0_out_expand.size() << endl;

    ap_uint<CONV_0_OUT_BIT*CONV_0_OFM_CH> pool0_out[CONV_1_IFM_ROW][CONV_1_IFM_COL];
    for(unsigned i = 0; i<CONV_1_IFM_ROW; i++)
        for(unsigned j = 0; j<CONV_1_IFM_COL; j++){
            ap_uint<CONV_0_OUT_BIT*CONV_0_OFM_CH> temp = pool_0_out_expand.read();
            pool0_out[i][j] = temp;
            pool_0_out_expand.write(temp);
        }
    write_data("./test_img/0_res/0_pool_0_out.bin", (char *) pool0_out, sizeof(pool0_out));
#endif

// CONV1
    stream<ap_uint<CONV_1_OUT_BIT * CONV_1_OUT_CH_PARA> >  conv_1_out0("conv_1_out0");
#pragma HLS STREAM variable=conv_1_out0 depth=256 dim=1
    stream<ap_uint<CONV_1_OUT_BIT * CONV_1_OUT_CH_PARA> >  conv_1_out1("conv_1_out1");
#pragma HLS STREAM variable=conv_1_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_1_IFM_ROW,
                    CONV_1_IFM_COL,
                    CONV_1_IFM_CH,
                    CONV_1_IN_BIT,

                    CONV_1_OFM_CH,
                    CONV_1_OUT_BIT,

                    CONV_1_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_1_INC_BIT,
                    CONV_1_BIAS_BIT,

                    CONV_1_IN_CH_PARA,
                    CONV_1_OUT_CH_PARA,
                    CONV_1_L_SHIFT>(
                pool_0_out_expand,
                conv_1_w,
                conv_1_inc,
                conv_1_bias,
                conv_1_out0,
                conv_1_out1,
                reps );

// PARA --> SERIAL
    stream<ap_uint<CONV_1_OUT_BIT * CONV_1_OUT_CH_PARA> >  conv_1_out_serial("conv_1_out_serial");
    const unsigned int numData1 = CONV_1_OFM_ROW * CONV_1_OFM_COL * CONV_1_OFM_CH / CONV_1_OUT_CH_PARA;
    Stream_PISO< CONV_1_OUT_BIT * CONV_1_OUT_CH_PARA, numData1/2, CONV_1_OFM_CH / CONV_1_OUT_CH_PARA > 
    (conv_1_out0, conv_1_out1, conv_1_out_serial, reps);

#ifdef DEBUG
    cout << "conv1 out size " << conv_1_out_serial.size() << endl;
#endif

// MAXPOOL1
    stream<ap_uint<CONV_1_OUT_BIT*CONV_1_OUT_CH_PARA> > pool_1_out("pool_1_out");
#pragma HLS STREAM variable=pool_1_out depth=256 dim=1
    MaxPool_2D< CONV_1_OFM_ROW,
                CONV_1_OFM_COL,
                CONV_1_OFM_CH,
                CONV_1_OUT_CH_PARA,
                CONV_1_OUT_BIT>(
                    conv_1_out_serial,
                    pool_1_out,
                    reps);

#ifdef DEBUG
    cout << "pool1 out size " << pool_1_out.size() << endl;
#endif

// expand width
    stream<ap_uint<CONV_1_OUT_BIT*CONV_1_OFM_CH> > pool_1_out_expand("pool_1_out_expand");
    const unsigned int numData1_after_pool = numData1 / 4;
    StreamingDataWidthConverter_Batch<CONV_1_OUT_BIT*CONV_1_OUT_CH_PARA, CONV_1_OUT_BIT*CONV_1_OFM_CH, numData1_after_pool> 
    (pool_1_out, pool_1_out_expand, reps);

#ifdef DEBUG
    cout << "pool1 out(expand width) size " << pool_1_out_expand.size() << endl;

    ap_uint<CONV_1_OUT_BIT*CONV_1_OFM_CH> pool1_out[CONV_2_IFM_ROW][CONV_2_IFM_COL];
    for(unsigned i = 0; i<CONV_2_IFM_ROW; i++)
        for(unsigned j = 0; j<CONV_2_IFM_COL; j++){
            ap_uint<CONV_1_OUT_BIT*CONV_1_OFM_CH> temp = pool_1_out_expand.read();
            pool1_out[i][j] = temp;
            pool_1_out_expand.write(temp);
        }
    write_data("./test_img/0_res/0_pool_1_out.bin", (char *) pool1_out, sizeof(pool1_out));
#endif

// CONV2
    stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OUT_CH_PARA> >  conv_2_out0("conv_2_out0");
#pragma HLS STREAM variable=conv_2_out0 depth=256 dim=1
    stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OUT_CH_PARA> >  conv_2_out1("conv_2_out1");
#pragma HLS STREAM variable=conv_2_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_2_IFM_ROW,
                    CONV_2_IFM_COL,
                    CONV_2_IFM_CH,
                    CONV_2_IN_BIT,

                    CONV_2_OFM_CH,
                    CONV_2_OUT_BIT,

                    CONV_2_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_2_INC_BIT,
                    CONV_2_BIAS_BIT,

                    CONV_2_IN_CH_PARA,
                    CONV_2_OUT_CH_PARA,
                    CONV_2_L_SHIFT>(
                pool_1_out_expand,
                conv_2_w,
                conv_2_inc,
                conv_2_bias,
                conv_2_out0,
                conv_2_out1,
                reps );

// PARA --> SERIAL
    stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OUT_CH_PARA> >  conv_2_out_serial("conv_2_out_serial");
    const unsigned int numData2 = CONV_2_OFM_ROW * CONV_2_OFM_COL * CONV_2_OFM_CH / CONV_2_OUT_CH_PARA;
    Stream_PISO< CONV_2_OUT_BIT * CONV_2_OUT_CH_PARA, numData2/2, CONV_2_OFM_CH / CONV_2_OUT_CH_PARA > 
    (conv_2_out0, conv_2_out1, conv_2_out_serial, reps);

#ifdef DEBUG
    cout << "conv2 out size " << conv_2_out_serial.size() << endl;
#endif

// MAXPOOL2
    stream<ap_uint<CONV_2_OUT_BIT*CONV_2_OUT_CH_PARA> > pool_2_out("pool_2_out");
#pragma HLS STREAM variable=pool_2_out depth=256 dim=1
    MaxPool_2D< CONV_2_OFM_ROW,
                CONV_2_OFM_COL,
                CONV_2_OFM_CH,
                CONV_2_OUT_CH_PARA,
                CONV_2_OUT_BIT>(
                    conv_2_out_serial,
                    pool_2_out,
                    reps);

#ifdef DEBUG
    cout << "pool2 out size " << pool_2_out.size() << endl;
#endif

// expand width
    stream<ap_uint<CONV_2_OUT_BIT*CONV_2_OFM_CH> > pool_2_out_expand("pool_2_out_expand");
    const unsigned int numData2_after_pool = numData2 / 4;
    StreamingDataWidthConverter_Batch<CONV_2_OUT_BIT*CONV_2_OUT_CH_PARA, CONV_2_OUT_BIT*CONV_2_OFM_CH, numData2_after_pool> 
    (pool_2_out, pool_2_out_expand, reps);

#ifdef DEBUG
    cout << "pool2 out(expand width) size " << pool_2_out_expand.size() << endl;
#endif

// CONV3
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA> >  conv_3_out0("conv_3_out0");
#pragma HLS STREAM variable=conv_3_out0 depth=256 dim=1
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA> >  conv_3_out1("conv_3_out1");
#pragma HLS STREAM variable=conv_3_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_3_IFM_ROW,
                    CONV_3_IFM_COL,
                    CONV_3_IFM_CH,
                    CONV_3_IN_BIT,

                    CONV_3_OFM_CH,
                    CONV_3_OUT_BIT,

                    CONV_3_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_3_INC_BIT,
                    CONV_3_BIAS_BIT,

                    CONV_3_IN_CH_PARA,
                    CONV_3_OUT_CH_PARA,
                    CONV_3_L_SHIFT>(
                pool_2_out_expand,
                conv_3_w,
                conv_3_inc,
                conv_3_bias,
                conv_3_out0,
                conv_3_out1,
                reps );

// PARA --> SERIAL
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA> >  conv_3_out_serial("conv_3_out_serial");
    const unsigned int numData3 = CONV_3_OFM_ROW * CONV_3_OFM_COL * CONV_3_OFM_CH / CONV_3_OUT_CH_PARA;
    Stream_PISO< CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA, numData3/2, CONV_3_OFM_CH / CONV_3_OUT_CH_PARA > 
    (conv_3_out0, conv_3_out1, conv_3_out_serial, reps);

#ifdef DEBUG
    cout << "conv3 out size " << conv_3_out_serial.size() << endl;
#endif

// conv3 --> maxpool3 --> ... --> conv6
//   |                              |
//   | bypass                       |
//   |                              | 
//  reorg-----------------------> concat --> conv7 

// Broadcast 1IN_2OUT
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA> >  pool_3_in("pool_3_in");
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA> >  reorg_in("reorg_in"); 
    Stream_Broadcast<CONV_3_OUT_BIT * CONV_3_OUT_CH_PARA, numData3> (conv_3_out_serial, pool_3_in, reorg_in, reps);

#ifdef DEBUG
    cout << "pool_3 in size " << pool_3_in.size() << endl;
    cout << "reorg  in size " << reorg_in.size()  << endl;
#endif

// reorg, K=2, S=2
    stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OFM_CH * 4> >  reorg_out("reorg_out");
#pragma HLS STREAM variable=reorg_out depth=512 dim=1

    ReOrg_2D<
            CONV_3_OFM_ROW,
            CONV_3_OFM_COL,
            CONV_3_OFM_CH,
            CONV_3_OUT_CH_PARA,
            CONV_3_OUT_BIT>(
            reorg_in,
            reorg_out,
            reps ); 

#ifdef DEBUG
    cout << "reorg out size " << reorg_out.size()  << endl;
#endif

// MAXPOOL3
    stream<ap_uint<CONV_3_OUT_BIT*CONV_3_OUT_CH_PARA> > pool_3_out("pool_3_out");
#pragma HLS STREAM variable=pool_3_out depth=256 dim=1
    MaxPool_2D< CONV_3_OFM_ROW,
                CONV_3_OFM_COL,
                CONV_3_OFM_CH,
                CONV_3_OUT_CH_PARA,
                CONV_3_OUT_BIT>(
                    pool_3_in,
                    pool_3_out,
                    reps);

#ifdef DEBUG
    cout << "pool3 out size " << pool_3_out.size() << endl;
#endif

// expand width
    stream<ap_uint<CONV_3_OUT_BIT*CONV_3_OFM_CH> > pool_3_out_expand("pool_3_out_expand");
    const unsigned int numData3_after_pool = numData3 / 4;
    StreamingDataWidthConverter_Batch<CONV_3_OUT_BIT*CONV_3_OUT_CH_PARA, CONV_3_OUT_BIT*CONV_3_OFM_CH, numData3_after_pool> 
    (pool_3_out, pool_3_out_expand, reps);

#ifdef DEBUG
    cout << "pool3 out(expand width) size " << pool_3_out_expand.size() << endl;
#endif


// CONV4
    stream<ap_uint<CONV_4_OUT_BIT * CONV_4_OUT_CH_PARA> >  conv_4_out0("conv_4_out0");
#pragma HLS STREAM variable=conv_4_out0 depth=256 dim=1
    stream<ap_uint<CONV_4_OUT_BIT * CONV_4_OUT_CH_PARA> >  conv_4_out1("conv_4_out1");
#pragma HLS STREAM variable=conv_4_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_4_IFM_ROW,
                    CONV_4_IFM_COL,
                    CONV_4_IFM_CH,
                    CONV_4_IN_BIT,

                    CONV_4_OFM_CH,
                    CONV_4_OUT_BIT,

                    CONV_4_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_4_INC_BIT,
                    CONV_4_BIAS_BIT,

                    CONV_4_IN_CH_PARA,
                    CONV_4_OUT_CH_PARA,
                    CONV_4_L_SHIFT>(
                pool_3_out_expand,
                conv_4_w,
                conv_4_inc,
                conv_4_bias,
                conv_4_out0,
                conv_4_out1,
                reps );

// PARA --> SERIAL & expand width
    stream<ap_uint<CONV_4_OUT_BIT * CONV_4_OFM_CH> >  conv_4_out_serial_expand("conv_4_out_serial_expand");
    const unsigned int numData4 = CONV_4_OFM_ROW * CONV_4_OFM_COL * CONV_4_OFM_CH / CONV_4_OUT_CH_PARA;
    ExpandDataWidth_PISO<CONV_4_OUT_BIT * CONV_4_OUT_CH_PARA, CONV_4_OUT_BIT * CONV_4_OFM_CH, numData4 / 2> 
    (conv_4_out0, conv_4_out1, conv_4_out_serial_expand, reps);

#ifdef DEBUG
    cout << "conv4 out(serial & expand width) size " << conv_4_out_serial_expand.size() << endl;
#endif

// CONV5
    stream<ap_uint<CONV_5_OUT_BIT * CONV_5_OUT_CH_PARA> >  conv_5_out0("conv_5_out0");
#pragma HLS STREAM variable=conv_5_out0 depth=256 dim=1
    stream<ap_uint<CONV_5_OUT_BIT * CONV_5_OUT_CH_PARA> >  conv_5_out1("conv_5_out1");
#pragma HLS STREAM variable=conv_5_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_5_IFM_ROW,
                    CONV_5_IFM_COL,
                    CONV_5_IFM_CH,
                    CONV_5_IN_BIT,

                    CONV_5_OFM_CH,
                    CONV_5_OUT_BIT,

                    CONV_5_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_5_INC_BIT,
                    CONV_5_BIAS_BIT,

                    CONV_5_IN_CH_PARA,
                    CONV_5_OUT_CH_PARA,
                    CONV_5_L_SHIFT>(
                conv_4_out_serial_expand,
                conv_5_w,
                conv_5_inc,
                conv_5_bias,
                conv_5_out0,
                conv_5_out1,
                reps );

// PARA --> SERIAL & expand width
    stream<ap_uint<CONV_5_OUT_BIT * CONV_5_OFM_CH> >  conv_5_out_serial_expand("conv_5_out_serial_expand");
    const unsigned int numData5 = CONV_5_OFM_ROW * CONV_5_OFM_COL * CONV_5_OFM_CH / CONV_5_OUT_CH_PARA;
    ExpandDataWidth_PISO<CONV_5_OUT_BIT * CONV_5_OUT_CH_PARA, CONV_5_OUT_BIT * CONV_5_OFM_CH, numData5 / 2> 
    (conv_5_out0, conv_5_out1, conv_5_out_serial_expand, reps);

#ifdef DEBUG
    cout << "conv5 out(serial & expand width) size " << conv_5_out_serial_expand.size() << endl;
#endif

// CONV6
    stream<ap_uint<CONV_6_OUT_BIT * CONV_6_OUT_CH_PARA> >  conv_6_out0("conv_6_out0");
#pragma HLS STREAM variable=conv_6_out0 depth=256 dim=1
    stream<ap_uint<CONV_6_OUT_BIT * CONV_6_OUT_CH_PARA> >  conv_6_out1("conv_6_out1");
#pragma HLS STREAM variable=conv_6_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_6_IFM_ROW,
                    CONV_6_IFM_COL,
                    CONV_6_IFM_CH,
                    CONV_6_IN_BIT,

                    CONV_6_OFM_CH,
                    CONV_6_OUT_BIT,

                    CONV_6_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_6_INC_BIT,
                    CONV_6_BIAS_BIT,

                    CONV_6_IN_CH_PARA,
                    CONV_6_OUT_CH_PARA,
                    CONV_6_L_SHIFT>(
                conv_5_out_serial_expand,
                conv_6_w,
                conv_6_inc,
                conv_6_bias,
                conv_6_out0,
                conv_6_out1,
                reps );

// PARA --> SERIAL & expand width
    stream<ap_uint<CONV_6_OUT_BIT * CONV_6_OFM_CH> >  conv_6_out_serial_expand("conv_6_out_serial_expand");
    const unsigned int numData6 = CONV_6_OFM_ROW * CONV_6_OFM_COL * CONV_6_OFM_CH / CONV_6_OUT_CH_PARA;
    ExpandDataWidth_PISO<CONV_6_OUT_BIT * CONV_6_OUT_CH_PARA, CONV_6_OUT_BIT * CONV_6_OFM_CH, numData6 / 2> 
    (conv_6_out0, conv_6_out1, conv_6_out_serial_expand, reps);

#ifdef DEBUG
    cout << "conv6 out(serial & expand width) size " << conv_6_out_serial_expand.size() << endl;
#endif

// concat [conv6_out, reorg_out]
    stream<ap_uint<(CONV_6_OUT_BIT * CONV_6_OFM_CH + CONV_3_OUT_BIT * CONV_3_OFM_CH * 4)> >  concat_out("concat_out");
#pragma HLS RESOURCE variable=concat_out core=FIFO_LUTRAM
    StreamConcat<CONV_3_OUT_BIT * CONV_3_OFM_CH * 4, CONV_6_OUT_BIT * CONV_6_OFM_CH, CONV_6_OFM_ROW * CONV_6_OFM_COL>
    (reorg_out, conv_6_out_serial_expand, concat_out, reps);


#ifdef DEBUG
    cout << "concat out size " << concat_out.size() << endl;
#endif

// CONV7
    stream<ap_uint<CONV_7_OUT_BIT * CONV_7_OUT_CH_PARA> >  conv_7_out0("conv_7_out0");
#pragma HLS STREAM variable=conv_7_out0 depth=256 dim=1
    stream<ap_uint<CONV_7_OUT_BIT * CONV_7_OUT_CH_PARA> >  conv_7_out1("conv_7_out1");
#pragma HLS STREAM variable=conv_7_out1 depth=256 dim=1
    conv3x3_bn_act< 
                    CONV_7_IFM_ROW,
                    CONV_7_IFM_COL,
                    CONV_7_IFM_CH,
                    CONV_7_IN_BIT,

                    CONV_7_OFM_CH,
                    CONV_7_OUT_BIT,

                    CONV_7_W_BIT,
                    M_BIT,  
                    BN_BIT,                   
                    CONV_7_INC_BIT,
                    CONV_7_BIAS_BIT,

                    CONV_7_IN_CH_PARA,
                    CONV_7_OUT_CH_PARA,
                    CONV_7_L_SHIFT>(
                concat_out,
                conv_7_w,
                conv_7_inc,
                conv_7_bias,
                conv_7_out0,
                conv_7_out1,
                reps );

#ifdef DEBUG
    cout << "conv7 out0 size " << conv_7_out0.size() << endl;
    cout << "conv7 out1 size " << conv_7_out1.size() << endl;
#endif


// adjust width
    stream<ap_uint<CONV_8_IN_BIT*CONV_8_IN_CH_PARA> > conv_8_in0("conv_8_in0");
    stream<ap_uint<CONV_8_IN_BIT*CONV_8_IN_CH_PARA> > conv_8_in1("conv_8_in1");
    const unsigned int numData7 = CONV_7_OFM_ROW * CONV_7_OFM_COL * CONV_7_OFM_CH / CONV_7_OUT_CH_PARA;
    StreamingDataWidthConverter_Batch<CONV_7_OUT_BIT*CONV_7_OUT_CH_PARA, CONV_8_IN_BIT*CONV_8_IN_CH_PARA, numData7 / 2> 
    (conv_7_out0, conv_8_in0, reps);
    StreamingDataWidthConverter_Batch<CONV_7_OUT_BIT*CONV_7_OUT_CH_PARA, CONV_8_IN_BIT*CONV_8_IN_CH_PARA, numData7 / 2> 
    (conv_7_out1, conv_8_in1, reps);

#ifdef DEBUG
    cout << "conv8 in0 size " << conv_8_in0.size() << endl;
    cout << "conv8 in1 size " << conv_8_in1.size() << endl;
#endif

// CONV8 (CONV1x1)
    stream<ap_uint<M_BIT_CONV1x1 * CONV_8_OUT_CH_PARA> >  conv_8_out0("conv_8_out0");
#pragma HLS STREAM variable=conv_8_out0 depth=256 dim=1
    stream<ap_uint<M_BIT_CONV1x1 * CONV_8_OUT_CH_PARA> >  conv_8_out1("conv_8_out1");
#pragma HLS STREAM variable=conv_8_out1 depth=256 dim=1
    conv1x1< 
            CONV_8_IFM_ROW,
            CONV_8_IFM_COL,
            CONV_8_IFM_CH,
            CONV_8_IN_BIT,
            CONV_8_OFM_CH,

            CONV_8_W_BIT,
            M_BIT_CONV1x1,                     

            CONV_8_IN_CH_PARA,
            CONV_8_OUT_CH_PARA>(
            conv_8_in0,
            conv_8_in1,
            conv_8_w,
            conv_8_out0,
            conv_8_out1,
            reps );

// PARA --> SERIAL
    stream<ap_uint<M_BIT_CONV1x1 * CONV_8_OUT_CH_PARA> >  conv_8_out_serial("conv_8_out_serial");
    const unsigned int numData8 = CONV_8_OFM_ROW * CONV_8_OFM_COL * CONV_8_OFM_CH / CONV_8_OUT_CH_PARA;
    Stream_PISO< M_BIT_CONV1x1 * CONV_8_OUT_CH_PARA, numData8/2, CONV_8_OFM_CH / CONV_8_OUT_CH_PARA > 
    (conv_8_out0, conv_8_out1, conv_8_out_serial, reps);

#ifdef DEBUG
    cout << "conv8 out size " << conv_8_out_serial.size() << endl;

    ap_int<M_BIT_CONV1x1> conv8_out[CONV_8_IFM_ROW][CONV_8_IFM_COL][CONV_8_OFM_CH];
    for(unsigned i = 0; i<CONV_8_IFM_ROW; i++)
        for(unsigned j = 0; j<CONV_8_IFM_COL; j++)
            for(unsigned k = 0; k<(CONV_8_OFM_CH/CONV_8_OUT_CH_PARA); k++){
            ap_uint<M_BIT_CONV1x1 * CONV_8_OUT_CH_PARA> temp = conv_8_out_serial.read();
            conv8_out[i][j][2*k] =  temp(31 , 0);
            conv8_out[i][j][2*k+1] = temp(63 , 32);
            conv_8_out_serial.write(temp);
        }
    write_data("./test_img/0_res/0_conv8_out.bin", (char *) conv8_out, sizeof(conv8_out));
#endif

// output
    AddLast<CONV_8_OFM_ROW*CONV_8_OFM_COL*CONV_8_OFM_CH/CONV_8_OUT_CH_PARA>(conv_8_out_serial, out, reps);

}
