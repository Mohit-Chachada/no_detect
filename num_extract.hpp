#ifndef NUM_EXTRACT_HPP
#define NUM_EXTRACT_HPP
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace cv;
using namespace std;

class Num_Extract
{
private:
	int classifier;    // use 1 SVM
	int train_samples;
	int classes;
	int sizex ;
	int sizey ;
	int ImageSize ;
	int HOG3_size;
	char pathToImages[200];
	bool temp_match;
    int print_nos[4];

protected:

    bool A_encloses_B(RotatedRect A, RotatedRect B);
    bool validate (Mat mask, Mat pre);
    void extract_Number(Mat pre , vector<Mat>src);
	
	void PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey);
	void LearnFromImages(CvMat* trainData, CvMat* trainClasses);
	void RunSelfTest(KNearest& knn2, CvSVM& SVM2);
	vector<int> AnalyseImage(KNearest knearest, CvSVM SVM, Mat _image);
	
	float maximum(float x, float y, float z); 
	
	void HOG3(IplImage *Im,vector<float>& descriptors);
	
	vector<Mat> HOGMatching_Template();

	vector<int> HOGMatching_Compare(vector<Mat> hist, Mat test_img);

	void extract(Mat mask, Mat pre);
	

	
public:
	Num_Extract();
	~Num_Extract();
	double pi;
    
	void run (Mat img);

        vector<Mat> dst;
        bool is_valid ;

};
#endif
