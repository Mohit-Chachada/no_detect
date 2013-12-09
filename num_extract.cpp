#include "num_extract.hpp"

Num_Extract::Num_Extract(){

}

Num_Extract::TaskReturn::TaskReturn(){

}

//Num_Extract::TaskReturn::~TaskReturn(){

//}

Num_Extract::Num_Extract(Num_Extract::InParams params){
    _classifier = params.classifier;    // use 1 SVM
    _train_samples = params.train_samples;
    _classes = params.classes;
    sizex = 30;
    sizey = 80;
    ImageSize = sizex * sizey;
    HOG3_size=81;
    sprintf(_pathToImages,"%s",params.pathToImages);
    //_temp_match=true;
    _temp_match=params.temp_match;
    pi = 3.1416;

    _print_nos[0]= params.print_nos[0];
    _print_nos[1]= params.print_nos[1];
    _print_nos[2]= params.print_nos[2];
    _print_nos[3]= params.print_nos[3];

}


Num_Extract::~Num_Extract(){
}

bool Num_Extract::A_encloses_B(RotatedRect A, RotatedRect B){
    Point2f ptsA[4];
    Point2f ptsB[4];
    A.points(ptsA);
    B.points(ptsB);
    bool encloses = true;
    Point2f p1,p2,p3,p4;
    double m = 0;
    double indicator = 0;
    double test_val = 0;
    for(int i = 0 ; i < 4 ; i++){
        p1 = ptsA[i];
        p2 = ptsA[(i+1)%4];
        p3 = ptsA[(i+2)%4];
        m = (p2.y-p1.y)/(p2.x-p1.x);
        indicator = (p3.y-p1.y)-m*(p3.x-p1.x);
        for(int j = 0 ; j<4 ; j++){
            p4 = ptsB[j];
            test_val = (p4.y-p1.y)-m*(p4.x-p1.x);
            if(test_val*indicator<0){
                encloses = false;
                break;
            }
        }
        if(!encloses) break;
    }
    return encloses;
}

bool Num_Extract::validate (Mat mask, Mat pre){
    std::vector<std::vector<cv::Point> > contour;
    Mat img;
    bool validate = false;
    bool validate1 = false;
    bool big = false;
    //Canny(mask,img,0,256,5);
    vector<Vec4i> hierarchy;
    //find contours from post color detection
    cv::findContours(mask, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>0.5*mask.rows*mask.cols)big = true;// If too close to object
    }
    int count = 0;

    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>0.013*mask.rows*mask.cols) count++;
    }

    if(count == 0 ){
        cout<<"not valid\n";
        return false;//filter out random noise
    }
    Mat grey,grey0,grey1,grey2,grey3;
    vector<Mat> bgr_planes;
    split(pre,bgr_planes);

    std::vector<std::vector<cv::Point> > contour1;
    std::vector<cv::Point> inner;
    double area = 0;
    vector<int> valid_index ;
    vector<int> valid_test;

    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>0.013*mask.rows*mask.cols){
            area = area + contourArea( contour[i],false);
            valid_test.push_back(i);
            for(int j = 0;j < contour[i].size();j++){
                inner.push_back(contour[i][j]);
            }
        }
    }
    RotatedRect inrect = minAreaRect(Mat(inner));//bounding rectangle of bins (if detected)
    RotatedRect outrect ;

    double thresh = 0;
    double threshf;


    vector<int> count1;
    int count2 = 0;
    if(!big){
        while(thresh < 2000 && (!validate && !validate1)){
            Canny(bgr_planes[0],grey1,0,thresh,5);//multi level canny thresholding
            Canny(bgr_planes[1],grey2,0,thresh,5);
            Canny(bgr_planes[2],grey3,0,thresh,5);
            max(grey1,grey2,grey1);
            max(grey1,grey3,grey);//getting strongest edges
            dilate(grey , grey0 , Mat() , Point(-1,-1));
            grey = grey0;
            double areamax = 0;
            cv::findContours(grey, contour1,hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
            for(int i = 0;i < contour1.size();i++){
                if(hierarchy[i][3]==-1){
                    areamax = contourArea(contour1[i],false);
                }
            }
            for(int i = 0;i < contour1.size();i++){
                if(contourArea(contour1[i],false)>area && contourArea(contour1[i],false)<0.8*areamax){
                    outrect = minAreaRect(Mat(contour1[i]));//bounding rectangle of detected contour
                    if(A_encloses_B(outrect,inrect)){
                        valid_index.push_back(i);
                    }
                }
                count2 = 0;
                //approxPolyDP(Mat(contour1[i]),poly,3,true);
                if(contourArea(contour1[i],false)>1.5*0.013*mask.rows*mask.cols){
                    for(int j = 0 ; j < valid_test.size(); j++){
                        RotatedRect test = minAreaRect(Mat(contour[valid_test[j]]));
                        double area1 = contourArea(contour1[i],false);
                        double area2 = contourArea(contour[valid_test[j]],false);
                        if(pointPolygonTest(Mat(contour1[i]),test.center,false)>0 && area1>area2){
                            count2++;
                        }
                    }
                }

                count1.push_back(count2);
            }
            bool val = false;
            for(int i = 0 ; i < count1.size(); i++){
                if(count1[i]>=1 && val){
                    validate1 = true ;
                    break;
                }
                if(count1[i]>=1){
                    val = true;
                }
            }


            if(valid_index.size()>=1){
                validate = true;
                threshf = thresh;
            }
            valid_index.clear();
            thresh = thresh + 2000/11;
            //valid_index.clear();
        }
    }
    else{
        validate = true;
    }
    cout<<"validate "<<validate;
    cout<<" validate1 "<<validate1<<endl;
    if(validate || validate1){
        return true;
    }
    return validate;
}

vector<Mat> Num_Extract::extract_Number(vector<Mat>masked,Mat pre){

    //bool validity = validate(mask,pre);

    //is_valid = validity;
    //is_valid = true;

    Mat rot_pre;

    vector<Mat> dst;

    Scalar color = Scalar(255,255,255);

    pre.copyTo(rot_pre);

    /*for(int i = 0 ; i < masked.size() ; i++){
          imshow("masked",masked[i]);
<<<<<<< HEAD
          //waitKey(0);
=======
          ////waitKey(0);
>>>>>>> ca1cfc93a1e3ed9c8b9079fc728bacdbd46864c8
      }*/

    Mat grey,grey0,grey1,grey2,grey3;

    vector<Vec4i> hierarchy;

    std::vector<std::vector<cv::Point> > contour,ext_contour;

    RotatedRect outrect;

    Mat rot_mat( 2, 3, CV_32FC1 );

    int out_ind;

    vector<Rect> valid,valid1,boxes;//valid and valid1 are bounding rectangles after testing validity conditions
    //boxes contains all bounding boxes
    vector<int> valid_index,valid_index1;

    Mat ext_number;


    bool prevBoxwasGood = false;
    bool badBoxAfterGood = false;


    /*
        Canny(bgr_planes[0],grey1,0,256,5);
        Canny(bgr_planes[1],grey2,0,256,5);
        Canny(bgr_planes[2],grey3,0,256,5);
        max(grey1,grey2,grey1);
        max(grey1,grey3,grey);//getting strongest edges
        //max(grey,grey5,grey);
*/

    Mat ext_prev;


    Rect box_prev;

    for(int i = 0 ; i<masked.size() ; i++){

        double thresh = 10;

        while (thresh < 2000) {

            vector<Mat> bgr_planes;

            split(masked[i],bgr_planes);

            cvtColor(masked[i],grey1,CV_BGR2GRAY);

            Canny(grey1,grey,0,thresh,5);
            /*

            Canny(bgr_planes[0],grey1,0,thresh,5);
            Canny(bgr_planes[1],grey2,0,thresh,5);
            Canny(bgr_planes[2],grey3,0,thresh,5);
            max(grey1,grey2,grey1);
            max(grey1,grey3,grey);//getting strongest edges
            */

            dilate(grey , grey0 , Mat() , Point(-1,-1));
            grey  = grey0;

            cv::findContours(grey, ext_contour,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            double areamax = 0;

            int index;
            for(int j = 0 ; j< ext_contour.size() ; j++){
                if(contourArea(ext_contour[j],false)>areamax){
                    index = j;
                    areamax = contourArea(ext_contour[j],false);
                }
            }

            bool goodBoxFound = false;

            outrect = minAreaRect(Mat(ext_contour[index]));//outer rectangle of the bin

            float angle,width;

            Point2f pts[4];

            outrect.points(pts);

            float dist1 = (sqrt((pts[0].y-pts[1].y)*(pts[0].y-pts[1].y) + (pts[0].x-pts[1].x)*(pts[0].x-pts[1].x)));

            float dist2 = (sqrt((pts[0].y-pts[3].y)*(pts[0].y-pts[3].y) + (pts[0].x-pts[3].x)*(pts[0].x-pts[3].x)));

            if (dist1>dist2) width = dist1;//getting the longer edge length of outrect

            else width = dist2;

            for(int j = 0 ; j<4 ; j++){
                float dist = sqrt((pts[j].y-pts[(j+1)%4].y)*(pts[j].y-pts[(j+1)%4].y) + (pts[j].x-pts[(j+1)%4].x)*(pts[j].x-pts[(j+1)%4].x));
                if(dist==width){
                    angle = atan((pts[j].y-pts[(j+1)%4].y)/(pts[(j+1)%4].x-pts[j].x));
                }
            }

            Mat outrect_img = Mat::zeros(pre.size(),CV_8UC3);

            /*for (int j = 0; j < 4; j++)
                line(image, pts[j], pts[(j+1)%4], Scalar(0,255,0));
            imshow("outrect" , outrect_img);
            ////waitKey(0);*/

            angle = angle * 180/3.14;

            //cout << angle <<endl;

            if(angle<0){//building rotation matrices
                rot_mat = getRotationMatrix2D(outrect.center,(-90-angle),1.0);
                //cout<<"orient "<<(-90-angle)<<endl;
            }
            else{
                //cout<<"orient "<<(90-angle)<<endl;
                rot_mat = getRotationMatrix2D(outrect.center,(90-angle),1.0);
            }
            Mat img;

            warpAffine(masked[i],img,rot_mat,grey.size());//rotating to make the outer bin straight

            bgr_planes.clear();

            split(img,bgr_planes);

            warpAffine(pre,rot_pre,rot_mat,rot_pre.size());//rotating the original (color) image by the same angle

            Canny(bgr_planes[0],grey1,0,thresh,5);
            Canny(bgr_planes[1],grey2,0,thresh,5);
            Canny(bgr_planes[2],grey3,0,thresh,5);
            max(grey1,grey2,grey1);
            max(grey1,grey3,grey);//getting strongest edges

            dilate(grey , grey0 , Mat() , Point(-1,-1));
            //warpAffine(grey,grey0,rot_mat,rot_pre.size());
            grey = grey0;

            cv::findContours(grey, contour,hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

            for(int j = 0 ; j<contour.size() ; j++){
                boxes.push_back(boundingRect(Mat(contour[j])));
            }//making boxes out of all contours
            areamax = 0;

            for(int j = 0 ; j<boxes.size() ; j++){
                if(boxes[j].width*boxes[j].height > areamax){
                    areamax = boxes[j].width*boxes[j].height;
                }
            }//finding the box with the largest area

            /*Mat all_contours = Mat::zeros(pre.size(),CV_8UC3);

            for(int k = 0 ; k < contour.size() ; k++){
                drawContours( all_contours, contour , k ,color , 1 ,8 ,vector<Vec4i>() ,0 , Point() );
            }
            imshow("all contours",all_contours);

            ////waitKey(0);
            */
            Mat box_n_contours = Mat::zeros(pre.size(),CV_8UC3);
            for(int k = 0 ; k < contour.size() ; k++){
                drawContours(box_n_contours , contour , k ,color , 1 ,8 ,vector<Vec4i>() ,0 , Point() );
                if(boxes[k].width*boxes[k].height==areamax){
                    continue;
                }
                rectangle(box_n_contours , boxes[k] , color );
            }


            imshow("contours with boxes except outermost",box_n_contours);
            //waitKey(0);


            for (int j = 0 ; j < boxes.size() ; j++){
                if(boxes[j].width*boxes[j].height < 0.5*areamax && boxes[j].width*boxes[j].height > 0.05*areamax){
                    valid.push_back(boxes[j]);//Filtering boxes on the basis of their area (rejecting the small ones)
                    valid_index.push_back(j); //this is the first validating condition
                }
            }

            for(int j = 0 ; j<valid.size() ; j++){
                double aspect = (float)valid[j].width/(float)valid[j].height;
                if(aspect < 1 && aspect > 0.3){//removing others on the basis of aspect ratio , second validating condition
                    valid1.push_back(valid[j]);//forming the list of valid bounding boxes
                    valid_index1.push_back(valid_index[j]);
                }
            }
            Mat first_test_boxes = Mat::zeros(pre.size(),CV_8UC3);
            for(int k = 0 ; k < valid.size() ; k++){
                rectangle(first_test_boxes , valid[k] , color );
            }

            imshow("after first test ",first_test_boxes);
            //waitKey(0);

            Mat final_boxes = Mat::zeros(pre.size(),CV_8UC3);
            for(int k = 0 ; k < valid1.size() ; k++){
                rectangle(final_boxes , valid1[k] , color );
                drawContours(final_boxes , contour , valid_index1[k] ,color , 1 ,8 ,vector<Vec4i>() ,0 , Point() );
            }//valid_index1 is required to draw the corresponding contours

            imshow("final valid boxes and contours",final_boxes);
            //waitKey(0);
            Rect box;
            if(valid1.size()>0){
                box = valid1[0];
                for(int j = 1 ; j<valid1.size() ; j++){ // now joining all valid boxes to extract the number
                    box = box | valid1[j];
                }
            }

            Mat final_mask = Mat::zeros(pre.size(),CV_8UC3);

            rectangle(final_mask , box , color ,CV_FILLED );//building the final mask

            ext_number = rot_pre & final_mask;//applying final_mask onto rot_pre

            imshow("extracted no." , ext_number);
            waitKey(0);

            //Mat ext_prev = Mat::zeros(ext_number.size(),CV_8UC3);

            boxes.clear();
            valid.clear();
            valid1.clear();
            valid_index.clear();
            valid_index1.clear();
            //cout<<"threshold level "<<thresh<<endl;
            thresh += 200;
            double ratio = (box.width*box.height)/areamax;
            double aspect_ratio = (float)box.height/(float)box.width;
            /*cout <<"areamax is "<<areamax<<"  detected area is "<<box.width*box.height<<endl;
            cout <<"ratio is : "<<ratio<<endl;
            cout<<"aspect ratio is :"<<aspect_ratio<<endl;*/

            if(aspect_ratio>1.4 && aspect_ratio<1.65 && ratio>0.25 && ratio<0.55){
                goodBoxFound = true;
            }

            if(prevBoxwasGood && !goodBoxFound){
                badBoxAfterGood = true;
            }

            if(badBoxAfterGood){
                ext_number = ext_prev;
                break;
            }/*
            cout<<"good box : "<<goodBoxFound<<endl;
            cout<<"bad box after good : "<<badBoxAfterGood<<endl;*/
            ext_number.copyTo(ext_prev);
            prevBoxwasGood = goodBoxFound;
            box_prev = box;

        }
        Mat num_img = Mat::zeros(box_prev.height,box_prev.width,CV_8UC3);
        int start_j,start_k;
        bool found = false;
        for(int j = 0 ; j<ext_number.rows ; j++){
            for(int k = 0 ; k<ext_number.cols ; k++ ){
                if(ext_number.at<Vec3b>(j,k)[0] != 0 || ext_number.at<Vec3b>(j,k)[1] != 0 || ext_number.at<Vec3b>(j,k)[2] != 0 ){
                    start_j = j;
                    start_k = k;
                    found = true;
                    break;
                }
                if(found){
                    break;
                }
            }
        }
        for(int j = start_j ; j<start_j+box_prev.height ; j++ ){
            for(int k = start_k ; k<start_k+box_prev.width ; k++){
                for(int l = 0 ; l<3 ; l++){
                    num_img.at<Vec3b>(j-start_j,k-start_k)[l] = ext_number.at<Vec3b>(j,k)[l];
                }
            }
        }

        dst.push_back(num_img);//building output list


    }


    //cout<<dst.size()<<endl;
    for(int i = 0 ; i<dst.size() ; i++){
        if(dst[i].empty()){
            cout<<"empty extracted number image \n";
        }
        else{
            //imshow("dst i",dst[i]);
            //waitKey(0);
        }

    }
    return dst;
    //cout<<valid.size()<<endl;
    //cout<<valid1.size()<<endl;
}


vector<Mat> Num_Extract::extract(Mat mask, Mat pre){
    vector<int> bins_ind ;
    std::vector<std::vector<cv::Point> > contour;
    vector<Mat> bins;

    Mat img ;
    Mat test = Mat::zeros( pre.size(), CV_8UC3 );
    if(is_valid){
        cout <<"validated\n";
        //Canny(mask,img,0,256,5);
        cv::findContours(mask, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        Scalar color(255,255,255);

        for(int i = 0 ; i<contour.size() ; i++){
            drawContours(test,contour,i,color,CV_FILLED);
        }

        imshow("bins",test);

        for(int i = 0 ; i<contour.size() ; i++){
            Mat img2 = Mat::zeros( pre.size(), CV_8UC3 );
            if(contourArea(contour[i],false)>0.013*mask.rows*mask.cols){
                cv::drawContours(img2,contour,i,color,CV_FILLED);
                bins.push_back(img2);
                bins_ind.push_back(i);
            }
        }
        vector<Mat>masked;
        for(int i = 0 ; i<bins.size() ; i++){
            Mat img = pre & bins[i];
            masked.push_back(img);
        }
        RotatedRect boxes[bins_ind.size()];
        for(int i = 0 ; i<bins_ind.size() ; i++){
            boxes[i] = minAreaRect(Mat(contour[bins_ind[i]]));
        }

        Mat tmp;

        if(masked.size()==2){
            if(boxes[0].center.x>boxes[1].center.x){
                tmp = masked[0];
                masked[0] = masked[1];
                masked[1] = tmp;
            }
        }

        if(masked.size()>2){
            for(int i = 0 ; i<bins_ind.size() ; i++){
                for(int j = i+1 ; j<bins_ind.size() ; j++){
                    if(boxes[j].center.y<boxes[i].center.y){
                        tmp = masked[j];
                        masked[j] = masked[i];
                        masked[i] = tmp;
                    }
                }
            }
            if(boxes[0].center.x>boxes[1].center.x && (boxes[2].center.y-boxes[1].center.y)>(boxes[1].center.y-boxes[0].center.y)){
                tmp = masked[0];
                masked[0] = masked[1];
                masked[1] = tmp;

            }
            if((boxes[2].center.y-boxes[1].center.y)<(boxes[1].center.y-boxes[0].center.y) && boxes[1].center.x>boxes[2].center.x){
                tmp = masked[1];
                masked[1] = masked[2];
                masked[2] = tmp;
            }
            if(masked.size()==4){
                if(boxes[0].center.x>boxes[1].center.x){
                    tmp = masked[0];
                    masked[0] = masked[1];
                    masked[1] = tmp;
                }
                if(boxes[2].center.x>boxes[3].center.x){
                    tmp = masked[2];
                    masked[2] = masked[3];
                    masked[3] = tmp;
                }
            }
        }
        return(masked);

        //extract_Number(pre,masked);

    }


    else {
        cout<<"not validated\n";
    }



}


float Num_Extract:: maximum(float x, float y, float z) {
    int max = x; /* assume x is the largest */

    if (y > max) { /* if y is larger than max, assign y to max */
        max = y;
    } /* end if */

    if (z > max) { /* if z is larger than max, assign z to max */
        max = z;
    } /* end if */

    return max; /* max is the largest value */
}

void Num_Extract::HOG3(IplImage *Im,vector<float>& descriptors)
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


            v_angles=0;\
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

vector<Mat> Num_Extract::HOGMatching_Template() {
    int nr_templ = 4;
    vector<Mat> hist;
    hist.resize(nr_templ); // 4 templates

    for (int i=0;i<nr_templ;i++){
        stringstream ss;
        ss << _print_nos[i];
        string s_no = ss.str();
        Mat temp=imread((string)(_pathToImages)+"/"+s_no+".png");
        if(temp.empty()){
            cout<<"empty image\n";
        }

        Mat outfile;
        resize(temp,outfile,Size(2*sizex,sizey));
        IplImage copy = outfile;
        IplImage* img2 = &copy;
        vector<float> ders;
        HOG3(img2,ders);

        CvMat* temp_hist;
        temp_hist = cvCreateMat(1, ders.size(), CV_32FC1);
        for (int j = 0; j < ders.size(); j++)
        {
            temp_hist->data.fl[j]= ders.at(j);
        }
        hist[i]=temp_hist;
        // cout<<i<<"\t"<<hist[i]<<"\n";
    }
    return hist;
}

vector<int> Num_Extract::HOGMatching_Compare(vector<Mat> hist, Mat test_img) {

    int nr_templ = 4;
    int nr_methods = 4;
    Mat outfile;
    int matched_templ[nr_templ];
    vector<int> result;

    // test histogram
    resize(test_img,outfile,Size(2*sizex,sizey));
    imshow("test_img",outfile);
    ////waitKey(0);

    IplImage copy = outfile;
    IplImage* img2 = &copy;
    vector<float> ders;
    HOG3(img2,ders);
    CvMat* test_hist;
    test_hist = cvCreateMat(1, ders.size(), CV_32FC1);
    for (int n = 0; n < ders.size(); n++)
    {
        test_hist->data.fl[n] = ders.at(n);
    }

    Mat test_hist2 = test_hist;
    float comparison [nr_templ][nr_methods]; // 4 templates x 4 methods

    for (int i=0;i<nr_templ;i++) {
        Mat temp_hist=hist[i];
        for (int j=0;j<nr_methods;j++) {
            int compare_method = j;
            comparison[i][j] = compareHist( test_hist2, temp_hist, compare_method );
            cout<<comparison[i][j]<<"\t";
        }
        cout<<"\n";
    }

    // finding matched template
    for (int j=0;j<nr_methods;j++) {
        float _minm,_maxm;
        if(j==1||j==3) {        // j==3 Bhattacharya Method
            _minm = min( min(comparison[0][j],comparison[1][j]) , min(comparison[2][j],comparison[3][j]) );
            for (int k=0;k<nr_templ;k++) {
                if (_minm==comparison[k][j]) matched_templ[j]=k;
            }
        }
        else {
            _maxm = max( max(comparison[0][j],comparison[1][j]) , max(comparison[2][j],comparison[3][j]) );
            for (int k=0;k<nr_templ;k++) {
                if (_maxm==comparison[k][j]) matched_templ[j]=k;
            }
        }
        result.push_back(_print_nos[matched_templ[j]]);
    }
    cout<<matched_templ[0]<<"\t"<<matched_templ[1]<<"\t"<<matched_templ[2]<<"\t"<<matched_templ[3]<<"\n";

    // result: no detected by all 4 methods
    //result=_print_nos[matched_templ[3]];
    return result;
}


void Num_Extract::LearnFromImages(CvMat* trainData, CvMat* trainClasses)
{
    Mat img,outfile;
    char file[255];
    for (int i = 0; i < _classes; i++)
    {
        for (int j=0; j < _train_samples;j++)
        {
            sprintf(file, "%s/%d/%d.png", _pathToImages, i, j);
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
            for (int n = 0; n < ders.size(); n++)
            {
                trainData->data.fl[i*_train_samples*ders.size()+ j * ders.size() + n] = ders.at(n);
            }

            trainClasses->data.fl[i*_train_samples+j] = i;
        }
    }

}

svm_model* Num_Extract::loadModel (const char* modelName) {
    //const char* modelName = "training_data.model";
    svm_model* model = svm_load_model(modelName);
    return model;
}

void Num_Extract::RunSelfTest(KNearest& knn2, CvSVM& SVM2)
{
    Mat img;
    CvMat* sample2;
    // SelfTest
    char file[255];
    int z = 0;
    while (z++ < 10)
    {
        int iSecret = rand() % _classes;
        sprintf(file, "%s/%d/%d.png", _pathToImages, iSecret, rand()%_train_samples);
        img = imread(file, 1);
        Mat stagedImage;

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

        float detectedClass;
        switch (_classifier) {
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
        ////waitKey(0);
    }

}

vector<int> Num_Extract::Classification(KNearest knearest, CvSVM SVM, Mat _image)
{

    CvMat* sample2;
    Mat image, gray, blur, thresh;
    vector<int> digits;

    vector < vector<Point> > contours;
    //_image = imread("./images/16.png", 1);


    resize(_image,image,Size(2*sizex,sizey));

    //image = _image;
    //cvtColor(image, gray, COLOR_BGR2GRAY);
    //GaussianBlur(gray, blur, Size(5, 5), 2, 2);
    //blur = gray;
    //adaptiveThreshold(gray, thresh, 255, 1, 1, 11, 2);

    vector<Mat> bgr_planes ;

    split(image,bgr_planes);

    Mat greyb,greyg,greyr,grey,grey0;

    Canny(bgr_planes[0],greyb,0,256,3);
    Canny(bgr_planes[1],greyg,0,256,3);
    Canny(bgr_planes[2],greyr,0,256,3);
    max(greyb,greyg,greyb);
    max(greyb,greyr,grey);//getting strongest edges
    //max(grey,grey5,grey);

    dilate(grey , grey0 , Mat() , Point(-1,-1));

    grey = grey0;

    findContours(grey, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    float _maxBoxArea=0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        Rect rec = boundingRect(contours[i]);
        //cout << "rec height " <<rec.height<<  "  rec width "<<rec.width<<endl;
        if (_maxBoxArea < rec.height*rec.width)
        {
            _maxBoxArea = rec.height*rec.width;
        }

    }
    cout<< " max area "<<_maxBoxArea<< endl;


    for (size_t i = 0; i < contours.size(); i++)
    {
        vector < Point > cnt = contours[i];
        Rect rec = boundingRect(cnt);
        float rec_area = rec.height*rec.width;
        if (rec_area > 0.60*_maxBoxArea)
        {
            float aspectR = float(rec.height)/float(rec.width);
            if ( aspectR > 1.0)
            {

                Mat roi = image(rec);
                Mat stagedImage;// = image(rec);


                // Descriptor
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

                cout<< "sample2 "<< sample2->data.fl[80]<<"\n";
                // _classifier
                float result;
                switch (_classifier) {
                case 1:
                {
                    cout<<"here\n";
                    ////waitKey(0);

                    result = SVM.predict(sample2);
                    break;
                }
                case 2:
                {
                    result = knearest.find_nearest(sample2, 1);
                    break;
                }
                }


                digits.push_back(int(result));
                rectangle(image, Point(rec.x, rec.y),
                          Point(rec.x + rec.width, rec.y + rec.height),
                          Scalar(0, 0, 255), 2);

                imshow("all", image);
                cout << result << "\n";

                imshow("single", stagedImage);
                ////waitKey(0);
            }

        }

    }

    return digits;
}

int Num_Extract::PredictNumber(svm_model* model, Mat _image) {
    Mat outfile;
    resize(_image,outfile,Size(2*sizex,sizey));

    IplImage copy = outfile;
    IplImage* img2 = &copy;
    vector<float> ders;
    HOG3(img2,ders);

    //svm_scale

    // storing test data
    ofstream testData;
    testData.open("test_data");
    //testData.clear();
    testData<<10<<" ";
    for (int n = 0; n < ders.size(); n++)
    {
        testData << (n+1) << ":";
        testData << ders.at(n) << " ";
    }
    //testData << "-1:0";
    testData << "\n";
    testData.close();

    // scaling test_data        // -r training_data.range test_data > test_data.scale
    int Tscale_argc = 4;
    char **Tscale_argv;
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
    vector<float> test_scaled;
    test_scaled = scale_main(Tscale_argc,Tscale_argv,TscaleOP);

    svm_node* x;
    x = new svm_node [ders.size()+1];
    for (int n = 0; n < ders.size(); n++)
    {
        svm_node tmp;
        tmp.index = n+1;
        tmp.value = test_scaled.at(n);
        x[n] = tmp;
    cout<< " n " <<n<<" test_scaled[n] " <<test_scaled.at(n) <<endl;
    }
    x[ders.size()].index = -1;
    double predictedValue = svm_predict(model, x);
    cout<< "Predicted No is "<<predictedValue<<"\n";
    return predictedValue;
}

/*void Num_Extract::run_cvSVM (Mat img){
    //Scalar lower(29,92,114);
    //Scalar higher(37,256,256);
    Scalar lower(0,92,114);
    Scalar higher(74,256,256);
    Mat img2 = Mat::zeros( img.size(), CV_8UC3 );
    cvtColor(img,img2,CV_BGR2HSV);
    Mat output;
    inRange(img2 , lower , higher , output);

    extract(output,img);

    vector<Mat> dst_flipped;

    Mat tmp;

    /* *****************************************
<<<<<<< HEAD
        time=clock();
        vector<vector<int> > digits;
        vector<int> digits1;
        cout<< "dst size "<<dst.size()<<endl;
        for(int i = 0 ; i<dst.size() ; i++){
            digits1 = Classification(knearest, SVM, dst[i]);
            digits.push_back(digits1);
            digits1.clear();
        }
        int result_ml[digits.size()];
        time=clock()-time;
        float run_time=((float)time)/CLOCKS_PER_SEC;
        cout<<"Run Time "<<run_time<<"\n";
        cout<<"no. of bins "<<digits.size()<<endl;
        //cout<<digits[0].size()<<endl;
        for(int i = 0 ; i<digits.size() ; i++){
            cout<< "digits of " <<i<<"th box are " << digits[i][0] <<" & "<<digits[i][1]<<"\n";
            if (digits[i][0]==8 || digits[i][1]==8) result_ml[i] = 98;
            if (digits[i][0]==7 || digits[i][1]==7) result_ml[i] = 37;
            if (digits[i][0]==6 || digits[i][1]==6) result_ml[i] = 16;
            if (digits[i][0]==0 || digits[i][1]==0) result_ml[i] = 10;
=======
    Mat rot_flip( 2, 3, CV_32FC1 );
>>>>>>> origin/extract
        ***************************************

    Mat rot_flip( 2, 3, CV_32FC1 );
    for(int i = 0 ; i<dst.size() ; i++){

        rot_flip = getRotationMatrix2D(Point2f(dst[i].cols/2,dst[i].rows/2),180,1.0);

        warpAffine(dst[i],tmp,rot_flip,dst[i].size());

        dst_flipped.push_back(tmp);
    }

    KNearest knearest;
    CvSVM SVM;

    if (!_temp_match) {

        // timer
        clock_t time=clock();

        CvMat* trainData = cvCreateMat(_classes * _train_samples, HOG3_size, CV_32FC1);
        CvMat* trainClasses = cvCreateMat(_classes * _train_samples, 1, CV_32FC1);

        LearnFromImages(trainData, trainClasses);



        switch (_classifier) {
        case 1:
        {
            // Set up SVM's parameters
            CvSVMParams params;
            params.svm_type    = CvSVM::C_SVC;
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
        float training_time=((float)time)/CLOCKS_PER_SEC;
        cout<<"Training Time "<<training_time<<"\n";

        //RunSelfTest(knearest, SVM);
        cout << "Testing\n";

        time=clock();
        vector<vector<int> > digits;
        vector<vector<int> > digits_rev;
        vector<int> digits1;

        cout<< "dst size "<<dst.size()<<endl;

        for(int i = 0 ; i<dst.size() ; i++){
            digits1 = Classification(knearest, SVM, dst[i]);

            digits.push_back(digits1);
            digits1.clear();
        }


        int result_ml[digits.size()];
        int result_ml_rev[digits_rev.size()];
        time=clock()-time;
        float run_time=((float)time)/CLOCKS_PER_SEC;
        cout<<"Run Time "<<run_time<<"\n";
        cout<<"no. of bins "<<digits.size()<<endl;

        cout<< "dst_flipped size "<<dst_flipped.size()<<endl;
        for(int i = 0 ; i<dst_flipped.size() ; i++){
            digits1 = Classification(knearest, SVM, dst_flipped[i]);

            digits_rev.push_back(digits1);
            digits1.clear();
        }

        //cout<<digits[0].size()<<endl;
        for(int i = 0 ; i<digits.size() ; i++){
            cout<< "digits of " <<i<<"th box are " << digits[i][0] <<" & "<<digits[i][1]<<"\n";
            if (digits[i][0]==8 || digits[i][1]==8) result_ml[i] = 98;
            if (digits[i][0]==7 || digits[i][1]==7) result_ml[i] = 37;
            if (digits[i][0]==6 || digits[i][1]==6) result_ml[i] = 16;
            if (digits[i][0]==0 || digits[i][1]==0) result_ml[i] = 10;

        }
        for(int i = 0 ; i<digits_rev.size() ; i++){
            cout<< "digits of " <<i<<"th box after flipping are " << digits_rev[i][0] <<" & "<<digits_rev[i][1]<<"\n";
            if (digits[i][0]==8 || digits[i][1]==8) result_ml_rev[i] = 98;
            if (digits[i][0]==7 || digits[i][1]==7) result_ml_rev[i] = 37;
            if (digits[i][0]==6 || digits[i][1]==6) result_ml_rev[i] = 16;
            if (digits[i][0]==0 || digits[i][1]==0) result_ml_rev[i] = 10;

        }
        cout<<"result ";
        for(int i = 0 ; i<digits.size() ; i++){
            cout<<result_ml[i]<<endl;
        }

        cout <<"result after flipping \n";
        for(int i = 0 ; i<digits_rev.size() ; i++){
            cout<<result_ml_rev[i]<<endl;
        }



        // find no from detected digits
        /* for (int i=0; i< _print_nos.cols; i++) {

            if (digits[0]==print_dgt[i][0]) result_ml=_print_nos[i];
            if (digits[0]==print_dgt[i][1]) result_ml=_print_nos[i];
            if (digits[1]==print_dgt[i][0]) result_ml=_print_nos[i];
            if (digits[1]==print_dgt[i][1]) result_ml=_print_nos[i];
        }


    }
    else {
        vector<Mat> hist;
        vector<vector<int> > result_hogm , result_hogm_rev; // result of all 4 matching methods
        vector<int> result;
        //Mat test_img=imread((string)(pathToImages)+"/"+"16.png");
        // Template Histograms
        hist = HOGMatching_Template();
        // Compare Histogram
        for(int i = 0 ; i<dst.size() ; i++){
            result = HOGMatching_Compare(hist,dst[i]);
            result_hogm.push_back(result);
        }
        for(int i = 0 ; i<dst_flipped.size() ; i++){
            result = HOGMatching_Compare(hist,dst_flipped[i]);
            result_hogm_rev.push_back(result);
        }
        cout<<"unflipped \n";
        for(int i = 0 ; i<result_hogm.size() ; i++ ){
            for(int j = 0 ; j<result_hogm[i].size() ; j++){
                cout << result_hogm[i][j]<<'\t';
            }
            cout << endl;
        }
        cout<<"flipped \n";
        for(int i = 0 ; i<result_hogm_rev.size() ; i++ ){
            for(int j = 0 ; j<result_hogm_rev[i].size() ; j++){
                cout << result_hogm_rev[i][j]<<'\t';
            }
            cout << endl;
        }
    }

    ////waitKey(0);

}*/


int Num_Extract::mode(vector<int> list){
    int size = list.size();
    int count[4]={0,0,0,0};
    for(int i = 0 ; i<size ; i++){
        for(int j = 0 ; j<4 ; j++){
            if(list[i]==_print_nos[j]){
                count[j]++;
            }
        }
    }
    int maxcount = 0;
    int index;
    for(int i = 0 ;i<4;i++){
        if(count[i]>maxcount){
            maxcount = count[i];
            index = i;
        }
    }
    return _print_nos[index];
}

Num_Extract::TaskReturn Num_Extract::run(Mat mask, Mat pre){
    vector<vector<int> > detected_nos;
    detected_nos.resize(2);
    //Scalar lower(29,92,114);
    //Scalar higher(37,256,256);
    clock_t time = clock();

    Num_Extract::TaskReturn marker;
    is_valid = validate(mask,pre);
    if(is_valid){
        vector<Mat> bins = extract(mask,pre);
        cout<<bins.size()<<endl;
        marker._no_of_bins = bins.size();
        RotatedRect boundingRects[bins.size()];
        Mat grey,thresh;
        marker._area_of_bins = 0;
        marker._orientation = 0;

        for(int i = 0 ; i<bins.size() ; i++){
            vector<vector<Point> > contour;
            cvtColor(bins[i],grey,CV_BGR2GRAY);
            Canny(grey,thresh,0,256,5);
            findContours(thresh,contour,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            double areamax = 0;
            int index;
            for(int j = 0 ; j< contour.size() ; j++){
                if(contourArea(contour[j],false)>areamax){
                    index = j;
                    areamax = contourArea(contour[j],false);
                }
            }
            boundingRects[i] = minAreaRect(Mat(contour[index]));
            marker._bin_centers.push_back(boundingRects[i].center);
            marker._area_of_bins += areamax;
            float angle,width;
            Point2f pts[4];
            boundingRects[i].points(pts);
            float dist1 = (sqrt((pts[0].y-pts[1].y)*(pts[0].y-pts[1].y) + (pts[0].x-pts[1].x)*(pts[0].x-pts[1].x)));
            float dist2 = (sqrt((pts[0].y-pts[3].y)*(pts[0].y-pts[3].y) + (pts[0].x-pts[3].x)*(pts[0].x-pts[3].x)));
            if (dist1>dist2) width = dist1;//getting the longer edge length of boundingRect[i]
            else width = dist2;
            for(int j = 0 ; j<4 ; j++){
                float dist = sqrt((pts[j].y-pts[(j+1)%4].y)*(pts[j].y-pts[(j+1)%4].y) + (pts[j].x-pts[(j+1)%4].x)*(pts[j].x-pts[(j+1)%4].x));
                if(dist==width){
                    angle = atan((pts[j].y-pts[(j+1)%4].y)/(pts[(j+1)%4].x-pts[j].x));
                }
            }
            angle = (180/pi)*angle;
            if(angle>0){
                marker._orientation += (90-angle);
            }
            else{
                marker._orientation += (-90-angle);
            }

        }
        marker._orientation = marker._orientation/marker._no_of_bins;
        cout<<"no of bins "<<marker._no_of_bins<<endl;
        cout<<"marker orientation "<<marker._orientation<<endl;
        cout<<"bin center "<<marker._bin_centers<<endl;


        vector<Mat> dst = extract_Number(bins,pre);
        time = clock()-time;
        cout<<"ex_Number "<<((float)time)/CLOCKS_PER_SEC<<endl;
        vector<Mat> dst_flipped;

        Mat tmp;

        Mat rot_flip( 2, 3, CV_32FC1 );
        for(int i = 0 ; i<dst.size() ; i++){

            rot_flip = getRotationMatrix2D(Point2f(dst[i].cols/2,dst[i].rows/2),180,1.0);

            warpAffine(dst[i],tmp,rot_flip,dst[i].size());

            dst_flipped.push_back(tmp);
        }
        for(int i = 0 ; i<dst.size() ; i++){
            imshow("extracted",dst[i]);
            waitKey(0);
            imshow("extracted flipped",dst_flipped[i]);
            waitKey(0);
        }

        int all_predicted[dst.size()],all_predicted_rev[dst.size()];

        const char* modelName = "training_data.model";

        svm_model* model = loadModel(modelName);

        for(int i = 0 ; i<dst.size() ; i++){
            int predictedValue = PredictNumber(model, dst[i]);
            cout<< "Guess Value " << predictedValue <<endl;

            all_predicted[i] = predictedValue;

            int predictedValue_rev = PredictNumber(model, dst_flipped[i]);
            cout<< "Guess Value after flipping " << predictedValue_rev <<endl;

            all_predicted_rev[i] = predictedValue_rev;
        }




        //HOG Matching Part//
        ////////////////////////////////////////////////////////////////////////////////////////////////


        vector<Mat> hist;
        vector<vector<int> > result_hogm , result_hogm_rev; // result of all 4 matching methods
        vector<int> result;
        //Mat test_img=imread((string)(pathToImages)+"/"+"16.png");
        // Template Histograms
        hist = HOGMatching_Template();
        // Compare Histogram
        for(int i = 0 ; i<dst.size() ; i++){
            result = HOGMatching_Compare(hist,dst[i]);
            result_hogm.push_back(result);
        }
        for(int i = 0 ; i<dst_flipped.size() ; i++){
            result = HOGMatching_Compare(hist,dst_flipped[i]);
            result_hogm_rev.push_back(result);
        }
        cout<<"unflipped \n";
        for(int i = 0 ; i<result_hogm.size() ; i++ ){
            for(int j = 0 ; j<result_hogm[i].size() ; j++){
                cout << result_hogm[i][j]<<'\t';
            }
            cout << endl;
        }
        cout<<"flipped \n";
        for(int i = 0 ; i<result_hogm_rev.size() ; i++ ){
            for(int j = 0 ; j<result_hogm_rev[i].size() ; j++){
                cout << result_hogm_rev[i][j]<<'\t';
            }
            cout << endl;
        }
        for(int i = 0 ; i<result_hogm.size() ; i++ ){
            result_hogm[i].push_back(all_predicted[i]);
            detected_nos[0].push_back(mode(result_hogm[i]));
            result_hogm_rev[i].push_back(all_predicted_rev[i]);
            detected_nos[1].push_back(mode(result_hogm_rev[i]));
        }
        marker._detected_nos = detected_nos;

    }
    return marker;
}







