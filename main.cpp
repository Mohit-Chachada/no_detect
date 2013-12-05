#include "num_extract.hpp"

int main (int argc , char **argv){
    Mat img = imread(argv[1],1);
    Num_Extract::InParams params;
    params.classes = 10;
    params.classifier = 1;
    strcpy(params.pathToImages,"./images");
    params.print_nos[0] = 10;
    params.print_nos[1] = 16;
    params.print_nos[2] = 37;
    params.print_nos[3] = 98;
    params.temp_match = false;
    params.train_samples = 4;
    Num_Extract Num1(params);
    //Num1.setParams(params);
	Num1.run(img);
    return 0;
}
