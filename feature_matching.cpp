#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


void ransacTest(const std::vector<cv::DMatch> matches,const std::vector<cv::KeyPoint>&keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& goodMatches,double distance,double confidence)
{
    goodMatches.clear();
    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it)
    {
        // Get the position of left keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        // Get the position of right keypoints
        x= keypoints2[it->trainIdx].pt.x;
        y= keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
    }
    
    // Compute F matrix using RANSAC
    std::vector<uchar> inliers(points1.size(),0);
    cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2),inliers,FM_RANSAC,distance,confidence); // confidence probability
    // extract the surviving (inliers) matches
    std::vector<uchar>::const_iterator
    itIn= inliers.begin();
    std::vector<cv::DMatch>::const_iterator
    itM= matches.begin();
    // for all matches
    for ( ;itIn!= inliers.end(); ++itIn, ++itM)
    {
        if (*itIn)
        { // it is a valid match
            goodMatches.push_back(*itM);
        }
    }
}



int main(int argc, char** argv)
{
int feature,extract,match,outlier;
//Default option values
feature=1;
extract=1;
match=1;
outlier=1;

//Argument input for option selection
if (argc>=6){
 feature=atoi(argv[3]);
 extract=atoi(argv[4]);
 match=atoi(argv[5]);
 outlier=atoi(argv[6]);
}
// vector<Point2f> keypoints1_2f,keypoints2_2f;
 // vector<DMatch> matches;

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
   
adaptiveThreshold(img1, img1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 0	);
adaptiveThreshold(img2, img2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 0);

imshow("Adp thresh 1", img1);
waitKey(0);
imshow("Adp thresh 2", img2);
waitKey(0);

    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    vector<KeyPoint> keypoints1, keypoints2;

    switch(feature)
    {
     case 1: //FAST
     {int threshold=80;
     FastFeatureDetector detector(threshold);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 2: //SURF
     {SurfFeatureDetector detector(130);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 3: //GFTT
     {int maxCorners=150;
      GoodFeaturesToTrackDetector detector(maxCorners);
      detector.detect(img1, keypoints1);
      detector.detect(img2, keypoints2);
      break;
     }
     case 4: //ORB
     {int maxCorners=150;
      OrbFeatureDetector detector(maxCorners);
      detector.detect(img1, keypoints1);
      detector.detect(img2, keypoints2);     
      break;
     }
     case 5: //Harris  (change threshold, presently some default threshold)
     {
      Ptr<FeatureDetector> detector= FeatureDetector::create("HARRIS");
      detector->detect(img1, keypoints1);
      detector->detect(img2, keypoints2);      
     }     
    }
   
    // computing descriptors
    Mat descriptors1, descriptors2;
    switch(extract)
    {
     case 1: //SURF
     {
      SurfDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }
     case 2: //SIFT
     {
      SiftDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }
     case 3: //ORB
     {
      OrbDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }     
    }
    
    // matching descriptors
    vector<DMatch> matches;
    switch (match)
    {
     case 1: //BruteForce
     {
     BFMatcher matcher(NORM_L2);
     matcher.match(descriptors1, descriptors2, matches);
     break;
     }
     case 2: //Flann
     {
     FlannBasedMatcher matcher;
     matcher.match(descriptors1, descriptors2, matches);
     break;
     }
    }
    
     
    // finding good matches
    vector< DMatch > good_matches; 
    switch (outlier)
    { 
     case 1:
     {
     double distance=50.; //quite adjustable/variable
     double confidence=0.99; //doesnt affect much when changed
     ransacTest(matches,keypoints1,keypoints2,good_matches,distance,confidence); 
     break;
     }
     case 2:
     {
     //look whether the match is inside a defined area of the image
     //only 25% of maximum of possible distance
     double tresholdDist = 0.25*sqrt(double(img1.size().height*img1.size().height + img1.size().width*img1.size().width));
     good_matches.reserve(matches.size());  
     for (size_t i = 0; i < matches.size(); ++i)
       {
        Point2f from = keypoints1[matches[i].queryIdx].pt;
        Point2f to = keypoints2[matches[i].trainIdx].pt;
        //calculate local distance for each possible match
        double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
        //save as best match if local distance is in specified area and on same height
        if (dist < tresholdDist)
          {
          good_matches.push_back(matches[i]);
          }
      }
     break;	
     }	
     case 3: //dist<2*min_dist
     {
        double max_dist = 0; double min_dist = 100;

 	 //-- Quick calculation of max and min distances between keypoints
 	 for( int i = 0; i < descriptors1.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
  	  }

	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	  //-- PS.- radiusMatch can also be used here.
	
	  for( int i = 0; i < descriptors1.rows; i++ )
	  { if( matches[i].distance < 2*min_dist )
	    { good_matches.push_back( matches[i]); }
	  }		
     }
    }
 int N;
 matches.clear();
 matches=good_matches; // update matches by good_matches
 N=matches.size();  // no of matched feature points   
    
cout<<"No of keypoints1 "<<keypoints1.size()<<"\n";    
cout<<"No of keypoints2 "<<keypoints2.size()<<"\n";
cout<<"No of matched points are "<<N<<"\n";
return 0;
}
