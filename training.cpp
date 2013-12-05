#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include "svm.h"
#include "svm-scale.h"
#include <string.h>

using namespace cv;
using namespace std;

const int train_samples = 2;
const int classes = 4;  // 10,16,37.98
const int sizex = 20;
const int sizey = 30;
const int ImageSize = sizex * sizey;
const int HOG3_size=81;
char pathToImages[] = "./images";
ofstream trainingData;
int print_nos[]	= {10,16,37,98};

/**
* @brief function calculates the maximum of 3 floating point integers
* @param x float
* @param y float
* @param z float
* @return max(x,y,z)
*/
static double pi = 3.1416;

static float maximum(float x, float y, float z) {
    int max = x; /* assume x is the largest */

    if (y > max) { /* if y is larger than max, assign y to max */
        max = y;
    } /* end if */

    if (z > max) { /* if z is larger than max, assign z to max */
        max = z;
    } /* end if */

    return max; /* max is the largest value */
}

/**
* @brief function computes the histogram of oriented gradient for input image
* @param Im - input image
* @param descriptors -output desciptors
*/
static void HOG3(IplImage *Im,vector<float>& descriptors)
{


    int nwin_x=3; //number of windows in x directions
    int nwin_y=3; //number of windows in y directions
    int B=9; //number of orientations


    int L=Im->height; //image height
    int C=Im->width; //image widht

    descriptors.resize(nwin_x*nwin_y*B); //allocating memory for descriptors

    CvMat angles2;
    CvMat magnit2;
    CvMat* H = cvCreateMat(nwin_x*nwin_y*B,1, CV_32FC3);
    IplImage *Im1=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,3);
    cvConvertScale(Im,Im1,1.0,0.0);



    int step_x=floor(C/(nwin_x+1));
    int step_y=floor(L/(nwin_y+1));
    int cont=0;
    CvMat *v_angles=0, *v_magnit=0,h1,h2,*v_angles1=0,*v_magnit1=0;

    CvMat *hx=cvCreateMat(1,3,CV_32F); hx->data.fl[0]=-1;
    hx->data.fl[1]=0; hx->data.fl[2]=1;
    CvMat *hy=cvCreateMat(3,1,CV_32F);
    hy->data.fl[0]=1;
    hy->data.fl[1]=0;
    hy->data.fl[2]=-1;


    IplImage *grad_xr = cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F, 3);
    IplImage *grad_yu = cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F, 3);



    //calculating gradient in x and y directions
    cvFilter2D(Im1, grad_xr,hx,cvPoint(1,0));
    cvFilter2D(Im1, grad_yu,hy,cvPoint(-1,-1));

    cvReleaseImage(&Im1);
    cvReleaseMat(&hx);
    cvReleaseMat(&hy);



    IplImage *magnitude=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,3);
    IplImage *orientation=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,3);
    IplImage *magnitude1=cvCreateImage(cvSize(C,L),IPL_DEPTH_32F,1);
    IplImage *orientation1=cvCreateImage(cvSize(C,L),IPL_DEPTH_32F,1);
    IplImage *I1=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);
    IplImage *I2=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);
    IplImage *I3=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);
    IplImage *I4=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);
    IplImage *I5=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);
    IplImage *I6=cvCreateImage(cvGetSize(Im),IPL_DEPTH_32F,1);


    //cartesian to polar transformations
    cvCartToPolar(grad_xr, grad_yu, magnitude, orientation,0);
    cvReleaseImage(&grad_xr);
    cvReleaseImage(&grad_yu);

    cvSubS(orientation,cvScalar(pi,pi,pi),orientation,0);




    cvSplit( magnitude, I4, I5, I6, 0 );


    cvSplit( orientation, I1, I2, I3, 0 );

    int step = I1->widthStep/sizeof(uchar);


    for(int i=0;i<I4->height;i++)
    {
        for(int j=0;j<I4->width;j++)
        {


            float *pt1= (float*) (I4->imageData + (i * I4->widthStep));
            float *pt2= (float*) (I5->imageData + (i * I5->widthStep));
            float *pt3= (float*) (I6->imageData + (i * I6->widthStep));
            float max = pt1[j]; /* assume x is the largest */

            if (pt2[j] > max) { /* if y is larger than max, assign y to max */
                ((float *)(I4->imageData + i*I4->widthStep))[j] = ((float *)(I5->imageData + i*I5->widthStep))[j];
                ((float *)(I1->imageData + i*I1->widthStep))[j] =((float *)(I2->imageData + i*I2->widthStep))[j];
            } /* end if */
            //consider only H and S channels.
            if (pt3[j] > max) { /* if z is larger than max, assign z to max */
                ((float *)(I4->imageData + i*I4->widthStep))[j] = ((float *)(I6->imageData + i*I6->widthStep))[j];
                ((float *)(I1->imageData + i*I1->widthStep))[j] =((float *)(I3->imageData + i*I3->widthStep))[j];

            }

            float * pt=((float *)(I1->imageData + i*I1->widthStep));


            if(pt[j]>0)
            {

                if(pt[j]>pi && (pt[j]-pi <0.001))
                    pt[j]=0;
                else if(pt[j]<pi && (pt[j]+pi<0.001))
                    pt[j]=0;
                else
                    pt[j]=pt[j];
                if(pt[j]>0)
                    pt[j]=-pt[j]+pi;

                pt[j]=-pt[j];
            }
            else if(pt[j]<0)
            {
                if(pt[j]>pi && (pt[j]-pi <0.001))
                    pt[j]=0;
                else if(pt[j]<pi && (pt[j]+pi<0.001))
                    pt[j]=0;
                else
                    pt[j]=pt[j];
                if(pt[j]<0)
                    pt[j]=pt[j]+pi;


            }


        }


    }
    //finding the dominant orientation
    cvCopy(I4,magnitude1,0);
    cvCopy(I1,orientation1,0);





    cvReleaseImage(&orientation);
    cvReleaseImage(&magnitude);
    cvReleaseImage(&I1);
    cvReleaseImage(&I2);
    cvReleaseImage(&I3);
    cvReleaseImage(&I4);
    cvReleaseImage(&I5);
    cvReleaseImage(&I6);




    int x, y;

    int m=0,n=0;


    //for each subwindow computing the histogram

    for(int n=0;n<nwin_x;n++)
    {
        for(int m=0;m<nwin_y;m++)
        {

            cont=cont+1;


            cvGetSubRect(magnitude1,&magnit2,cvRect((m*step_x),(n*step_y),2*step_x,2*step_y));

            v_magnit1=cvCreateMat(magnit2.cols,magnit2.rows,magnit2.type);
            cvT(&magnit2,v_magnit1);
            v_magnit=cvReshape(v_magnit1, &h2,1,magnit2.cols*magnit2.rows);



            cvGetSubRect(orientation1,&angles2,cvRect((m*step_x),(n*step_y),2*step_x,2*step_y));

            v_angles1=cvCreateMat(angles2.cols,angles2.rows,angles2.type);
            cvT(&angles2,v_angles1);
            v_angles=cvReshape(v_angles1, &h1,1,angles2.cols*angles2.rows);

            int K=0;
            if(v_angles->cols>v_angles->rows)
                K=v_angles->cols;
            else
                K=v_angles->rows;
            int bin=0;

            CvMat* H2 = cvCreateMat(B,1, CV_32FC1);
            cvZero(H2);


            float temp_gradient;

            //adding histogram count for each bin
            for(int k=0;k<K;k++)
            {
                float* pt = (float*) ( v_angles->data.ptr + (0 * v_angles->step));
                float* pt1 = (float*) ( v_magnit->data.ptr + (0 * v_magnit->step));
                float* pt2 = (float*) ( H2->data.ptr + (0 * H2->step));
                temp_gradient=pt[k];
                if (temp_gradient <= -pi+((2*pi)/B)) {
                    bin=0;
                    pt2[bin]=pt2[bin]+(pt1[k]);
                }
                else if ( temp_gradient <= -pi+4*pi/B) {
                    bin=1;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if (temp_gradient <= -pi+6*pi/B) {
                    bin=2;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if ( temp_gradient <= -pi+8*pi/B) {
                    bin=3;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if (temp_gradient <= -pi+10*pi/B) {
                    bin=4;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if (temp_gradient <= -pi+12*pi/B) {
                    bin=5;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if (temp_gradient <= -pi+14*pi/B) {
                    bin=6;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else if (temp_gradient <= -pi+16*pi/B) {
                    bin=7;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }
                else {
                    bin=8;
                    pt2[bin]=pt2[bin]+(pt1[k]);

                }

            }
            cvReleaseMat(&v_magnit1);
            cvReleaseMat(&v_angles1);
            cvNormalize(H2, H2, 1, 0, 4);



            for(int y1=0;y1<H2->rows;y1++)
            {
                float* pt2 = (float*) ( H2->data.ptr + (0 * H2->step));
                float* pt3 = (float*) ( H->data.ptr + (0 * H->step));
                pt3[(cont-1)*B+y1]=pt2[y1];

            }


            v_angles=0;
            v_magnit=0;
            cvReleaseMat(&H2);

        }
    }

    for(int i=0;i<descriptors.capacity();i++)
    {

        float* pt2 = (float*) ( H->data.ptr + (0 * H->step));
        descriptors[i]=pt2[i];

    }
    cvReleaseImage(&magnitude1);
    cvReleaseImage(&orientation1);
    cvReleaseMat(&H);


}

void storeTrainingData()
{
    Mat img,outfile;
    char file[255];

    for (int i = 0; i < classes; i++)
    {
        
        int nr = print_nos[i];
        for (int j=0; j < train_samples ; j++)
        {
            sprintf(file, "%s/%d/%d.png", pathToImages, nr, j);
            img = imread(file, 1);
            if (!img.data)
            {
                cout << "File " << file << " not found\n";
                exit(1);
            }

            resize(img,outfile,Size(sizex,sizey));
            IplImage copy = outfile;
            IplImage* img2 = &copy;
            vector<float> ders;
            HOG3(img2,ders);
            trainingData<<nr<<" ";
            for (int n = 0; n < ders.size(); n++)
            {
                trainingData << (n+1) << ":";
                trainingData << ders.at(n) << " ";
            }
            //trainingData << "-1:0";
            trainingData << "\n";
        }
    }

}





int main (int argc, char** argv) {
    //ofstream trainingData;

    /*******************************/
    // Model Generation and Prediction using System Commands part
    /*******************************/

    /*
    char command[1000];
    sprintf(command,"./svm-scale -s training_data.range training_data > training_data.scale");
    system(command);
    sprintf(command,"./svm-train -s 0 -t 2 -b 1 training_data training_data.model");
    system(command);

    sprintf(command,"./svm-scale -r training_data.range test_data > test_data.scale");
    system(command);
    sprintf(command,"./svm-predict -b 1 test_data.scale training_data.model svmOutput");
    system(command);
*/

    // Store Training Data
    trainingData.open("training_data");
    storeTrainingData();
    trainingData.close();

    // scale training_data
    int Mscale_argc = 4;
    char **Mscale_argv;   // -s training_data.range training_data > training_data.scale
    Mscale_argv = new char* [Mscale_argc];

    for (int i=0; i< Mscale_argc; i++) {
        Mscale_argv[i] = new char [100];
    }
    sprintf(Mscale_argv[1], "%s", "-s");
    sprintf(Mscale_argv[2], "%s", "training_data.range");
    sprintf(Mscale_argv[3], "%s", "training_data");
    char* MscaleOP;
    MscaleOP = new char [100];
    sprintf(MscaleOP, "%s", "training_data.scale");

    scale_main(Mscale_argc,Mscale_argv,MscaleOP);
    cout<<"scaled training data\n";

    /*
    // generate model and store it
    svm_problem *prob;
    svm_parameter *param;
    const char* modelName = "training_data.model";
    prob->l = classes*train_samples;
    prob->y;
    prob->x;
    svm_model* model = svm_train(prob,param);
    int a = svm_save_model(modelName,model);
*/

    char command[1000];
    sprintf(command,"./svm-train -s 0 -t 2 -b 1 training_data.scale training_data.model");
    system(command);

    // load model
    const char* modelName = "training_data.model";
    svm_model* model = svm_load_model(modelName);

    // test data [O/P]
    Mat img,outfile;
    char file[255];
    sprintf(file, "%s/%d.png", pathToImages, 98);
    img = imread(file, 1);
    if (!img.data)
    {
        cout << "File " << file << " not found\n";
        exit(1);
    }

    resize(img,outfile,Size(sizex,sizey));
    imshow ("outfile",outfile);
    waitKey(0);
    IplImage copy = outfile;
    IplImage* img2 = &copy;
    vector<float> ders;
    HOG3(img2,ders);

    ofstream testData;
    testData.open("test_data");
    testData<<10<<" ";
    for (int n = 0; n < ders.size(); n++)
    {
        testData << (n+1) << ":";
        testData << ders.at(n) << " ";
    }
    //trainingData << "-1:0";
    testData << "\n";
    testData.close();

    // scaling test_data
    int Tscale_argc = 4;
    char **Tscale_argv;   // -r training_data.range test_data > test_data.scale
    Tscale_argv = new char* [Tscale_argc];

    for (int i=0; i< Tscale_argc; i++) {
        Tscale_argv[i] = new char [100];
    }
    sprintf(Tscale_argv[1], "%s", "-r");
    sprintf(Tscale_argv[2], "%s", "training_data.range");
    sprintf(Tscale_argv[3], "%s", "test_data");
    char* TscaleOP;
    TscaleOP = new char [100];
    sprintf(TscaleOP, "%s", "test_data.scale");
    scale_main(Tscale_argc,Tscale_argv,TscaleOP);

    // reading scaled test data and predict
    ifstream testScaledData;
    testScaledData.open("test_data.scale");
    svm_node* x;
    x = new svm_node [ders.size()+1];

    int ffargc = ders.size()+1;
    char **ff;   // -s training_data.range training_data > training_data.scale
    ff = new char* [ffargc];
    string gg;
    for (int i=0; i< ffargc; i++) {
        ff[i] = new char [100];
    }

    testScaledData>>ff[0];     // read 1st no
    for (int n = 1; n < ffargc; n++)
    {
        testScaledData>>gg;
        cout<< " gg "<<gg<<"\n";
/*        testScaledData>>ff[n];     // read
        int k=0;
        while (ff[n][k]!=':') {
            k++;
        }
        while (ff[n][k]!=)
        cout<< "sizeof "<<ff[n]<<"\n";
        //for (int j=k+1; j<)
        svm_node tmp;
        tmp.index = n;
        tmp.value = ders.at(n-1);
        x[n-1] = tmp;
        cout<< " n " <<n<<"ff " <<ff[n] <<endl;
*/
    }
    x[81].index = -1;
    /*
    double predictedValue = svm_predict(model, x);
    cout<< "Guess Value " << predictedValue <<endl;
*/
    return 0;
}
