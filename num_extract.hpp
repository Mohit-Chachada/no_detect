#ifndef NUM_EXTRACT_HPP
#define NUM_EXTRACT_HPP
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cv;
using namespace std;

class Num_Extract
{
protected:

    bool A_encloses_B(RotatedRect A, RotatedRect B);
    bool validate (Mat mask, Mat pre);
    void extract_Number(Mat pre , vector<Mat>src);

public:

    void extract(Mat mask, Mat pre);

    vector<Mat> dst;
    bool is_valid ;

};
#endif
