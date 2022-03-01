

#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include "stream_tools.h"

#define grid_row 10
#define grid_col 20
#define div 15*7
#define BS 2

using namespace hls;
using namespace std;

void UltraNet_Bypass(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps);

void load_data(const char *path, char *ptr, unsigned int size)
{
    ifstream f(path, ios::in | ios::binary);
    if (!f)
    {
        cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}

void write_data(const char *path, char *ptr, unsigned int size)
{
    ofstream f(path, ios::out | ios::binary);
    if (!f)
    {
        cout << "write no such file,please check the file name!/n";
        exit(0);
    }
    f.write(ptr, size);
    f.close();
}

int main(int argc, char const *argv[])
{

	const char* img_path[] = { "./test_img/img0.bin"};

	uint8_t img[BS][360][640][3];
	for (unsigned int n = 0; n < BS; n++)
    	load_data(img_path[n], (char *) img[n], (sizeof(img)/BS));

    uint8_t * data = (uint8_t *) img;
    const int data_points_per_line = 8;       
    const int nums_line_per_img = 360 * 640 * 3 / 8;

    stream<my_ap_axis> input_stream("input stream");
    for (unsigned int n = 0; n < BS; n++)
		for (unsigned int i = 0; i < nums_line_per_img; i++) {
			my_ap_axis temp;
			for (unsigned int j = 0; j < data_points_per_line; j++) {
				temp.data( 8*(j+1)-1, 8*j ) = data[n * 360 * 640 * 3 + i * data_points_per_line + j];
			}
			input_stream.write(temp);
		}

    cout << "start ..... " << endl;
    stream<my_ap_axis> output_stream("output stream");
    UltraNet_Bypass(input_stream, output_stream, unsigned(log2(BS)));

    cout << "output size :" << output_stream.size() << endl;

    ap_int<32> conv8_out [BS][grid_row*grid_col][6][6];
    for(unsigned int n = 0; n<BS; n++)
		for(unsigned int i = 0; i< (grid_row*grid_col); i++)
			for(unsigned int j = 0; j<6; j++)
				for(unsigned int k = 0; k<3; k++){
					my_ap_axis output = output_stream.read();
					ap_uint<64> out_data = output.data;
					conv8_out[n][i][j][2*k]   = out_data(31,0);
					conv8_out[n][i][j][2*k+1] = out_data(63,32);
				}

    float bias[6][6];
    load_data("./bias/last_bias.bin", (char *) bias, sizeof(bias));


    int conf [BS][grid_row*grid_col] = {0};
    for(unsigned int n = 0; n<BS; n++)
		for(unsigned int i = 0; i< (grid_row*grid_col); i++)
			for(unsigned int j = 0; j<6; j++){
				conf[n][i] += conv8_out[n][i][j][4];
			}


    
    unsigned int max_index[BS];

    int max[BS];
    for(unsigned int n = 0; n<BS; n++){
    	max[n] = -9999999;
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
            if(conf[n][i] > max[n]){
                max[n] = conf[n][i];
                max_index[n] = i;
            }
        cout << "max index" << n  << ": " << max_index[n] << endl;
    }
    

    unsigned int grid_x[BS];
    unsigned int grid_y[BS];
    for(unsigned int n = 0; n<BS; n++){
        grid_x[n] = max_index[n] % grid_col;
        grid_y[n] = max_index[n] / grid_col;
    }

    float boxs[BS][6][4];
    for(unsigned int n = 0; n<BS; n++)
        for(unsigned int i = 0; i<6; i++)
            for(unsigned int j = 0; j<4; j++){
                boxs[n][i][j] = conv8_out[n][max_index[n]][i][j] / float((div)) + bias[i][j];
            }

    float x[BS] = {0}, y[BS] = {0}, w[BS] = {0}, h[BS] = {0};
    for(unsigned int n = 0; n<BS; n++){
        for(unsigned int i = 0; i<6; i++){
            x[n] += 1 / (1 + std::exp(-boxs[n][i][0]));
            y[n] += 1 / (1 + std::exp(-boxs[n][i][1]));
            w[n] += std::exp(boxs[n][i][2]);
            h[n] += std::exp(boxs[n][i][3]);
        }
        x[n] = x[n] / 6;
        y[n] = y[n] / 6;
        w[n] = w[n] / 6;
        h[n] = h[n] / 6;

        x[n] = (x[n] + grid_x[n]) * 16;
        y[n] = (y[n] + grid_y[n]) * 16;
        w[n] = w[n]*20;
        h[n] = h[n]*20;

        cout << "result" << n <<" :" << endl;
        cout << "x: " << x[n] << " y: " << y[n] << " w: " << w[n] << " h: " << h[n] << endl;
    }

    return 0;
}
