#include "num_extract.hpp"

int main (int argc , char *argv[]){
    Mat img = imread(argv[1],1);
    InParams params;
    params._classes = 10;
    params._classifier = 1;
    strcpy(params._pathToImages,"./images");
    params._print_nos[0] = 10;
    params._print_nos[1] = 16;
    params._print_nos[2] = 37;
    params._print_nos[3] = 98;
    params._temp_match = false;
    params._train_samples = 4;
    Num_Extract Num1 = Num_Extract();
    Num1.setParams(params);
	Num1.run(img);
    return 0;
}
