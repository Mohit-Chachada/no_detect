    #include "opencv2/ml/ml.hpp" 
    #include "opencv2/highgui/highgui.hpp"  
    #include "opencv2/objdetect/objdetect.hpp"    
    #include "opencv2/imgproc/imgproc.hpp"
    #include "opencv2/imgproc/imgproc_c.h"        
    #include <iostream>  
    #include <stdio.h>  
    #include <time.h>
    
    using namespace cv;  
    using namespace std;  

    const int classifier = 1;    // use 1 SVM    
    const int descriptor = 3;    // use 3 HOG3 
    const int train_samples = 4;  
    const int classes = 10;  
    const int sizex = 20;  
    const int sizey = 30;  
    const int ImageSize = sizex * sizey; 
    const int HOG1_size=3780;    
    const int HOG3_size=81; 
    const int HOG4_size=9;    
    char pathToImages[] = "./images";  
    
    void PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey);  
    void LearnFromImages(CvMat* trainData, CvMat* trainClasses);  
    void RunSelfTest(KNearest& knn2, CvSVM& SVM2);  
    void AnalyseImage(KNearest knearest, CvSVM SVM);  
    
    
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
    
   
    
    
    /** @function main */  
    int main(int argc, char** argv)  
    {  
      int descriptor_size;
     // start the timer
     clock_t time=clock(); 
      
//     CvMat* trainData = cvCreateMat(classes * train_samples,ImageSize, CV_32FC1);  
     CvMat* trainData;
     if (descriptor==1) descriptor_size=HOG1_size; 
     else if (descriptor==2) descriptor_size=ImageSize;
     else if (descriptor==3) descriptor_size=HOG3_size;
     else if (descriptor==4) descriptor_size=HOG4_size;
     
     trainData = cvCreateMat(classes * train_samples,descriptor_size, CV_32FC1);
     CvMat* trainClasses = cvCreateMat(classes * train_samples, 1, CV_32FC1);  
      
     namedWindow("single", CV_WINDOW_AUTOSIZE);  
     // namedWindow("all",CV_WINDOW_AUTOSIZE);  
      
     LearnFromImages(trainData, trainClasses);  
     
     KNearest knearest;
     CvSVM SVM;  
         
     switch (classifier) {
      case 1:
      {
       // Set up SVM's parameters
       CvSVMParams params;
      // params.svm_type    = CvSVM::C_SVC;
       params.kernel_type = CvSVM::LINEAR;
       params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

       // Train the SVM
       SVM.train(trainData, trainClasses, Mat(), Mat(), params);
       SVM.save("SVM_training_data");
       break;
      }
      case 2:
      {
       knearest.train(trainData, trainClasses);  
       break;
      }
     }
     
     time=clock()-time;
     float training_time=((float)time)/CLOCKS_PER_SEC;   //time for single run
     cout<<"Training Time "<<training_time<<"\n";     
     
     //RunSelfTest(knearest, SVM);  
     cout << "Testing\n";  
     
     time=clock();
     AnalyseImage(knearest, SVM); 
     time=clock()-time;
     float run_time=((float)time)/CLOCKS_PER_SEC;   //time for single run
     cout<<"Run Time "<<run_time<<"\n"; 
          
     waitKey(0);
     return 0;  
      
    }  
      
    void PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey)  
    {  
     Mat grayImage,blurredImage,thresholdImage,contourImage,regionOfInterest;  
      
     vector<vector<Point> > contours;  
      
     cvtColor(*inImage,grayImage , COLOR_BGR2GRAY);  
      
     GaussianBlur(grayImage, blurredImage, Size(5, 5), 2, 2);  
     adaptiveThreshold(blurredImage, thresholdImage, 255, 1, 1, 11, 2);  
      
     thresholdImage.copyTo(contourImage);  
      
     findContours(contourImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);  
      
     int idx = 0;  
     size_t area = 0;  
     for (size_t i = 0; i < contours.size(); i++)  
     {  
      if (area < contours[i].size() )  
      {  
       idx = i;  
       area = contours[i].size();  
      }  
     }  
      
     Rect rec = boundingRect(contours[idx]);  
      
     regionOfInterest = thresholdImage(rec);  
      
     resize(regionOfInterest,*outImage, Size(sizex, sizey));  
      
    }  
      
    void LearnFromImages(CvMat* trainData, CvMat* trainClasses)  
    {  
     Mat img;  
     char file[255];  
     for (int i = 0; i < classes; i++)  
     {  
      for (int j=0; j < train_samples;j++)
      {
       sprintf(file, "%s/%d/%d.png", pathToImages, i, j);  
       img = imread(file, 1);  
       if (!img.data)  
       {  
         cout << "File " << file << " not found\n";  
         exit(1);  
       }  
       
       Mat outfile; 
       switch (descriptor) {
       	case 1:
        {// OpenCV HOG descriptor
          HOGDescriptor hog;
          vector<float> ders;
       	  vector<Point> locs;
          resize(img,outfile,Size(64,128));
          hog.compute(outfile,ders,Size(0,0),Size(0,0),locs);
          //cout<<ders.size()<<"\n";
          //trainData = cvCreateMat(classes * train_samples,ders.size(), CV_32FC1);  
          for (int n = 0; n < ders.size(); n++)  
          {  
           trainData->data.fl[i*train_samples*ders.size()+ j * ders.size() + n] = ders.at(n);  
          }          
          break;
        }
        case 2:
       	{ // Image intesnity as descriptor
       	  PreProcessImage(&img, &outfile, sizex, sizey);
       	  //trainData = cvCreateMat(classes * train_samples,ImageSize, CV_32FC1);  
          for (int n = 0; n < ImageSize; n++)  
          {  
           trainData->data.fl[i*train_samples*ImageSize+ j * ImageSize + n] = outfile.data[n]; 
          } 
       	  break;
       	}
       	case 3:
       	{ // HOG3 descriptor
          resize(img,outfile,Size(sizex,sizey));       	
	  IplImage copy = outfile;
	  IplImage* img2 = &copy;	
          vector<float> ders;       	  
       	  HOG3(img2,ders);
          for (int n = 0; n < ders.size(); n++)  
          {  
           trainData->data.fl[i*train_samples*ders.size()+ j * ders.size() + n] = ders.at(n);  
          }
          break;       	  
       	}
       	case 4:
       	{ // Histogram as descriptor
       	  resize(img,outfile,Size(sizex,sizey));
       	  // Establish the number of bins
	  int histSize = 9;
	  // Set the ranges (for grayscale values)
	  float range[] = { 0, 256 } ;
	  const float* histRange = { range };
	  bool uniform = true; bool accumulate = false;
	  Mat ders; // histogram descriptor
	  // Compute the histograms:
	  calcHist( &outfile, 1, 0, Mat(), ders, 1, &histSize, &histRange, uniform, accumulate );
	  normalize( ders, ders, 0, 1, NORM_MINMAX, -1, Mat() );
	  for (int n = 0; n < ders.rows*ders.cols; n++)  
          {  
           trainData->data.fl[i*train_samples*ImageSize+ j * ImageSize + n] = ders.data[n]; 
          } 
       	  break;	
       	}
       }
        
       trainClasses->data.fl[i*train_samples+j] = i;  
     }  
    }  
   }   
    
   void RunSelfTest(KNearest& knn2, CvSVM& SVM2)  
    {  
     Mat img;  
     //CvMat* sample2 = cvCreateMat(1, ImageSize, CV_32FC1);  
     CvMat* sample2;
     // SelfTest  
     char file[255];  
     int z = 0;  
     while (z++ < 10)  
     {  
      int iSecret = rand() % classes;  
      //cout << iSecret;  
      sprintf(file, "%s/%d/%d.png", pathToImages, iSecret, rand()%train_samples);  
      img = imread(file, 1);  
      Mat stagedImage;  

      switch (descriptor) {
       case 1:
       {// HOG descriptor
         HOGDescriptor hog;
         vector<float> ders;
       	 vector<Point> locs;
         resize(img,stagedImage,Size(64,128));
         hog.compute(stagedImage,ders,Size(0,0),Size(0,0),locs);
         //cout<<ders.size()<<"\n";
         sample2 = cvCreateMat(1, ders.size(), CV_32FC1);  
         for (int n = 0; n < ders.size(); n++)  
         {  
          sample2->data.fl[n] = ders.at(n);  
         }          
         break;
       }
       case 2:
       { // Image data as descriptor
         sample2 = cvCreateMat(1, ImageSize, CV_32FC1);
         PreProcessImage(&img, &stagedImage, sizex, sizey);  
         for (int n = 0; n < ImageSize; n++)  
         {  
          sample2->data.fl[n] = stagedImage.data[n];  
         }  
         break;
       }
       case 3:
       { // HOG3 descriptor
         resize(img,stagedImage,Size(sizex,sizey));
	 IplImage copy = stagedImage;
	 IplImage* img2 = &copy;	
         vector<float> ders;       	  
       	 HOG3(img2,ders);
         sample2 = cvCreateMat(1, ders.size(), CV_32FC1);  
         for (int n = 0; n < ders.size(); n++)  
         {  
          sample2->data.fl[n] = ders.at(n);  
         } 
         break;       	  
       } 
       case 4:
       { // Histogram as descriptor
       	 resize(img,stagedImage,Size(sizex,sizey));
       	 // Establish the number of bins
	 int histSize = 9;
	 // Set the ranges (for grayscale values)
	 float range[] = { 0, 256 } ;
	 const float* histRange = { range };
	 bool uniform = true; bool accumulate = false;
	 Mat ders; // histogram descriptor
	 int ders_size=ders.rows*ders.cols;
         sample2 = cvCreateMat(1, ders_size, CV_32FC1);	 
	 // Compute the histograms:
	 calcHist( &stagedImage, 1, 0, Mat(), ders, 1, &histSize, &histRange, uniform, accumulate );
	 normalize( ders, ders, 0, 1, NORM_MINMAX, -1, Mat() );	 
	 for (int n = 0; n < ders_size; n++)  
         {  
          sample2->data.fl[n] = ders.data[n]; 
         } 
       	 break;	
       }      
      } 
 
      float detectedClass;
      switch (classifier) {
       case 1:
       {
       	 detectedClass = SVM2.predict(sample2);
         break;
       }
       case 2:
       {
         detectedClass = knn2.find_nearest(sample2, 1);       	 	
       	 break;
       }
      }
     
      if (iSecret != (int) ((detectedClass)))  
      {  
       cout << "False " << iSecret << " matched with "  
         << (int) ((detectedClass));  
       exit(1);  
      }  
      cout << "Right " << (int) ((detectedClass)) << "\n";  
      imshow("single", stagedImage);  
      waitKey(0);  
     }  
      
    }  
      
    void AnalyseImage(KNearest knearest, CvSVM SVM)  
    {  
      
     //CvMat* sample2 = cvCreateMat(1, ImageSize, CV_32FC1);  
     CvMat* sample2;      
     Mat _image,image, gray, blur, thresh;  
      
     vector < vector<Point> > contours;  
     _image = imread("./images/37.png", 1);  
     //image = imread("./images/all_4.png", 1);  
     
     resize(_image,image,Size(2*sizex,1.2*sizey));
     cvtColor(image, gray, COLOR_BGR2GRAY);  
     GaussianBlur(gray, blur, Size(5, 5), 2, 2);  
     adaptiveThreshold(blur, thresh, 255, 1, 1, 11, 2);  
     findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);  
     float digits[contours.size()];
     float number;
      
     for (size_t i = 0; i < contours.size(); i++)  
     {  
      vector < Point > cnt = contours[i];  
      if (contourArea(cnt) > 50)  
      {  
       Rect rec = boundingRect(cnt);  
       if (rec.height > 28)  
       {  
        Mat roi = image(rec);  
        Mat stagedImage;  

        // Descriptor    
        switch (descriptor) {
         case 1:
          {// HOG descriptor
           HOGDescriptor hog;
           vector<float> ders;
       	   vector<Point> locs;
           resize(roi,stagedImage,Size(64,128));
           hog.compute(stagedImage,ders,Size(0,0),Size(0,0),locs);
           //cout<<ders.size()<<"\n";
           sample2 = cvCreateMat(1, ders.size(), CV_32FC1);  
           for (int n = 0; n < ders.size(); n++)  
           {  
            sample2->data.fl[n] = ders.at(n);  
           }          
           break;
          }
         case 2:
          { // Image Data descriptor
           sample2 = cvCreateMat(1, ImageSize, CV_32FC1);
           PreProcessImage(&roi, &stagedImage, sizex, sizey);  
           for (int n = 0; n < ImageSize; n++)  
           {  
            sample2->data.fl[n] = stagedImage.data[n];  
           }  
           break;
          }
         case 3:
          {// HOG3 detector
           resize(roi,stagedImage,Size(sizex,sizey));
	   IplImage copy = stagedImage;
	   IplImage* img2 = &copy;	
           vector<float> ders;
       	   HOG3(img2,ders);
           sample2 = cvCreateMat(1, ders.size(), CV_32FC1);  
           for (int n = 0; n < ders.size(); n++)  
           {  
            sample2->data.fl[n] = ders.at(n);  
           }  
           break;       	  
          }
         case 4:
          { // Histogram as descriptor
       	   resize(roi,stagedImage,Size(sizex,sizey));
       	   // Establish the number of bins
	   int histSize = 9;
	   // Set the ranges (for grayscale values)
	   float range[] = { 0, 256 } ;
	   const float* histRange = { range };
	   bool uniform = true; bool accumulate = false;
	   Mat ders; // histogram descriptor
           int ders_size=ders.rows*ders.cols;
           sample2 = cvCreateMat(1, ders_size, CV_32FC1);	   
	   // Compute the histograms:
	   calcHist( &stagedImage, 1, 0, Mat(), ders, 1, &histSize, &histRange, uniform, accumulate );
           normalize( ders, ders, 0, 1, NORM_MINMAX, -1, Mat() );	   
	   for (int n = 0; n < ders_size; n++)  
           {  
            sample2->data.fl[n] = ders.data[n]; 
           } 
       	   break;	
          }          
        }
	
	// Classifier
        float result;
        switch (classifier) {
         case 1:
         {
       	   result = SVM.predict(sample2);
           break;
         }
         case 2:
         {
           result = knearest.find_nearest(sample2, 1);       	 	
       	   break;
         }
        }
        digits[contours.size()-i-1]=result;
        rectangle(image, Point(rec.x, rec.y),  
        Point(rec.x + rec.width, rec.y + rec.height),  
        Scalar(0, 0, 255), 2);  
      
        imshow("all", image);  
        cout << result << "\n";  
      
        imshow("single", stagedImage);  
        waitKey(0);  
       }  
      
      }  
      
     }
     number=digits[0]*10+digits[1];
     cout<< "number is "<<number<<"\n";
    }  
