#include "num_extract.hpp"

int main (int argc , char *argv[]){
    Mat img = imread(argv[1],1);
	Num_Extract Num1;
	Num1.run(img);
    return 0;
}
