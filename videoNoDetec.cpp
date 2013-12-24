#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    VideoCapture cap(1); 
    if(!cap.isOpened()) {
    	cout<<"Unable to start video capture\n";
    	return -1;
    }  
        
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
                
    while (1) {
    
    //Mat img = imread(argv[1],1);
    cap >> img;
    Num_Extract::InParams params;
    params.classes = 4;
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
    Num_Extract::SubTaskReturn marker;

    Scalar lower(0,92,114);
    Scalar higher(74,256,256);
    //Scalar lower(0,126,35);
    //Scalar higher(101,256,212);
    Mat img2 = Mat::zeros( img.size(), CV_8UC3 );
    cvtColor(img,img2,CV_BGR2HSV);
    Mat output;
    inRange(img2 , lower , higher , output);
    imshow ("O/P",output);
    waitKey(1);
    marker = Num1.run(output,img);
 
}    
    cap.release();
    return 0;
    
    }
