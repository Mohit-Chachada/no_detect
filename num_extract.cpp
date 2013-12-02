#include "num_extract.hpp"

Num_Extract::Num_Extract(){
    classifier = 1;    // use 1 SVM
	train_samples = 4;
	classes = 10;
	sizex = 20;
	sizey = 30;
	ImageSize = sizex * sizey;
	HOG3_size=81;
	sprintf(pathToImages,"%s","./images");
	temp_match=false;
	pi = 3.1416;
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
    Canny(mask,img,0,256,5);
    vector<Vec4i> hierarchy;
    //find contours from post color detection
    cv::findContours(img, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>0.5*320*240)big = true;// If too close to object
	}
    int count = 0;

    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>1000) count++;
	}

    if(count == 0 )return validate;//filter out random noise
    Mat grey,grey0,grey1,grey2,grey3;
    vector<Mat> bgr_planes;
    split(pre,bgr_planes);

	std::vector<std::vector<cv::Point> > contour1;
	std::vector<cv::Point> inner;
	double area = 0;
    vector<int> valid_index ;
    vector<int> valid_test,bins_indices;

    for(int i = 0 ; i<contour.size();i++){
        if(contourArea( contour[i],false)>1000){
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
    vector<Point> poly;
    if(!big){
        while(thresh < 1000 && (!validate && !validate1)){
            Canny(bgr_planes[0],grey1,0,thresh,5);//multi level canny thresholding
            Canny(bgr_planes[1],grey2,0,thresh,5);
            Canny(bgr_planes[2],grey3,0,thresh,5);
            max(grey1,grey2,grey1);
            max(grey1,grey3,grey);//getting strongest edges
            dilate(grey , grey0 , Mat() , Point(-1,-1));
            grey = grey0;
            cv::findContours(grey, contour1,hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
            for(int i = 0;i < contour1.size();i++){
                if(hierarchy[i][3]==-1){
                    continue;//excluding the outermost contour (contour due to the mask)
                }
                if(contourArea(contour1[i],false)>area){
                    outrect = minAreaRect(Mat(contour1[i]));//bounding rectangle of detected contour
                    if(A_encloses_B(outrect,inrect)){
                        valid_index.push_back(i);
                    }
                }
                count2 = 0;
                approxPolyDP(Mat(contour1[i]),poly,3,true);
                if(contourArea(contour1[i],false)>1500){
                    for(int j = 0 ; j < valid_test.size(); j++){
                        RotatedRect test = minAreaRect(Mat(contour[valid_test[j]]));
                        double area1 = contourArea(contour1[i],false);
                        double area2 = contourArea(contour[valid_test[j]],false);
                        if(pointPolygonTest(Mat(poly),test.center,false)>0 && area1>area2){
                            count2++;
                        }
                    }
                }

                count1.push_back(count2);
                poly.clear();
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
            thresh = thresh + 1000/11;
            valid_index.clear();
        }
    }
    else{
        validate = true;
    }
    if(validate || validate1){

        return true;
    }
    return validate;
}

void Num_Extract::extract_Number(Mat pre , vector<Mat>src ){
    Mat rot_pre;

    Scalar color = Scalar(255,255,255);

    pre.copyTo(rot_pre);

    vector<Mat>masked;

    for(int i = 0 ; i<src.size() ; i++){
        masked.push_back(src[i]);
    }

    /*for(int i = 0 ; i < masked.size() ; i++){
          imshow("masked",masked[i]);
          waitKey(0);
      }*/

    Mat grey,grey0,grey1;

    //vector<Mat> bgr_planes;

    vector<Vec4i> hierarchy;

    std::vector<std::vector<cv::Point> > contour,ext_contour;

    RotatedRect outrect;

    Mat rot_mat( 2, 3, CV_32FC1 );

    int out_ind;

    vector<Rect> valid,valid1,boxes;//valid and valid1 are bounding rectangles after testing validity conditions
                                    //boxes contains all bounding boxes
    vector<int> valid_index,valid_index1;

    for(int i = 0 ; i<masked.size() ; i++){
        //split(masked[i],bgr_planes);

        cvtColor(masked[i],grey1,CV_BGR2GRAY);

        Canny(grey1,grey,0,256,5);

        /*Canny(bgr_planes[0],grey1,0,256,5);
        Canny(bgr_planes[1],grey2,0,256,5);
        Canny(bgr_planes[2],grey3,0,256,5);
        max(grey1,grey2,grey1);
        max(grey1,grey3,grey);
        max(grey,grey5,grey);//getting strongest edges*/

        dilate(grey , grey0 , Mat() , Point(-1,-1));

        grey = grey0;

        cv::findContours(grey, ext_contour,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        double areamax = 0;

        int index;
        for(int j = 0 ; j< ext_contour.size() ; j++){
            if(contourArea(ext_contour[j],false)>areamax){
                index = j;
                areamax = contourArea(ext_contour[j],false);
            }
        }

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
        waitKey(0);*/

        angle = angle * 180/3.14;

        cout << angle <<endl;

        if(angle<0){//building rotation matrices
            rot_mat = getRotationMatrix2D(outrect.center,(-90-angle),1.0);
        }
        else{
            rot_mat = getRotationMatrix2D(outrect.center,(90-angle),1.0);
        }

        warpAffine(grey1,grey0,rot_mat,grey0.size());//rotating to make the outer bin straight
                                                     //grey1 is the grayscale image (unrotated)
                                                     //after rotation stored in grey0
        warpAffine(pre,rot_pre,rot_mat,rot_pre.size());//rotating the original (color) image by the same angle
        Canny(grey0,grey,0,256,5);//thresholding the rotated image (grey0)

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
        waitKey(0);
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
        waitKey(0);

        for (int j = 0 ; j < boxes.size() ; j++){
            if(boxes[j].width*boxes[j].height < 0.7*areamax && boxes[j].width*boxes[j].height > 0.05*areamax){
                valid.push_back(boxes[j]);//Filtering boxes on the basis of their area (rejecting the small ones)
                valid_index.push_back(j); //this is the first validating condition
            }
        }

        for(int j = 0 ; j<valid.size() ; j++){
            double aspect = valid[j].width/valid[j].height;
            if(aspect < 1){//removing others on the basis of aspect ratio , second validating condition
                valid1.push_back(valid[j]);//forming the list of valid bounding boxes
                valid_index1.push_back(valid_index[j]);
            }
        }
        Mat first_test_boxes = Mat::zeros(pre.size(),CV_8UC3);
        for(int k = 0 ; k < valid.size() ; k++){
            rectangle(first_test_boxes , valid[k] , color );
        }
        imshow("after first test ",first_test_boxes);
        waitKey(0);

        Mat final_boxes = Mat::zeros(pre.size(),CV_8UC3);
        for(int k = 0 ; k < valid1.size() ; k++){
            rectangle(final_boxes , valid1[k] , color );
            drawContours(final_boxes , contour , valid_index1[k] ,color , 1 ,8 ,vector<Vec4i>() ,0 , Point() );
        }//valid_index1 is required to draw the corresponding contours

        imshow("final valid boxes and contours",final_boxes);

        waitKey(0);

        Rect box = valid1[0];
        for(int j = 1 ; j<valid1.size() ; j++){ // now joining all valid boxes to extract the number
            box = box | valid1[j];
        }
        Mat final_mask = Mat::zeros(pre.size(),CV_8UC3);

        rectangle(final_mask , box , color ,CV_FILLED );//building the final mask

        Mat ext_number = rot_pre & final_mask;//applying final_mask onto rot_pre

        imshow("extracted no." , ext_number);
        waitKey(0);

        /*for(int j = 0 ; j<contour.size() ; j++){
            if(hierarchy[j][3]!=-1){
                valid.push_back(boundingRect(Mat(contour[j])));
            }
        }
        for(int j = 0 ; j<valid.size() ; j++){
            double aspect = valid[j].width/valid[j].height;
            if(aspect < 1.5){//removing others on the basis of aspect ratio
                valid1.push_back(valid[j]);//forming the list of valid bounding boxes
            }
        }
        Rect box = valid1[0];
        for(int j = 1 ; j<valid1.size() ; j++){
            box = box | valid1[j];
        }
        Mat box_mat = Mat::zeros(rot_pre.size(),CV_8UC3);
        Mat drawing = Mat::zeros(rot_pre.size(),CV_8UC3);

        rectangle( box_mat, box , color ,  CV_FILLED );//drawing the rectangle on box_mat
        rot_pre.copyTo(drawing,box_mat);//applying mask (box_mat) onto rot_pre and saving on drawing*/

        dst.push_back(ext_number);//building output list
        boxes.clear();
        valid.clear();
        valid1.clear();
        valid_index.clear();
        valid_index1.clear();
    }
    //cout<<dst.size()<<endl;
    //cout<<valid.size()<<endl;
    //cout<<valid1.size()<<endl;
}

void Num_Extract::extract(Mat mask, Mat pre){
    bool valid = validate(mask,pre);
	
    is_valid = valid;
    //bool valid = true;

    vector<Mat> bins ;
    std::vector<std::vector<cv::Point> > contour;
    Mat img ;
    Mat test = Mat::zeros( pre.size(), CV_8UC3 );
    if(valid){
        cout <<"validated\n";
        Canny(mask,img,0,256,5);
        cv::findContours(img, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        Scalar color(255,255,255);
        for(int i = 0 ; i<contour.size() ; i++){
            drawContours(test,contour,i,color,CV_FILLED);
        }

        for(int i = 0 ; i<contour.size() ; i++){
            Mat img2 = Mat::zeros( img.size(), CV_8UC3 );
            if(contourArea(contour[i],false)>1000){
                drawContours(img2,contour,i,color,CV_FILLED);
                bins.push_back(img2);
            }
        }
        vector<Mat>masked;
        for(int i = 0 ; i<bins.size() ; i++){
            Mat img = pre & bins[i];
            masked.push_back(img);
        }
        extract_Number(pre,masked);
    }

    else {
        cout<<"not validated\n";
    }
    /*
    imshow("contour ext",test);
    waitKey(0);
    for(int i = 0 ; i<bins.size() ; i++){
        imshow("contour sent1",bins[i]);
        waitKey(0);
    }
    for(int i = 0 ; i<dst.size() ; i++){
        imshow("numbers extracted",dst[i]);
        waitKey(0);
    }

    */

    //cout << dst.size()<<endl;
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

vector<Mat> Num_Extract::HOGMatching_Template() {

    int print_nos[]={10,16,37,98};
    vector<Mat> hist;
    hist.resize(4);

    for (int i=0;i<4;i++){
        stringstream ss;
        ss << print_nos[i];
        string s_no = ss.str();
        Mat temp=imread((string)(pathToImages)+"/"+s_no+".png");
        if(temp.empty()){
            cout<<"empty image\n";
        }

        Mat outfile;
        resize(temp,outfile,Size(100,80));
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
    int print_nos[]={10,16,37,98};
    Mat outfile;
    int matched_templ[4];
    vector<int> result;

    // test histogram
    resize(test_img,outfile,Size(100,80));
    imshow("test_img",outfile);
    waitKey(0);
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
    Mat test_hist2=test_hist;
    float comparison [4][4];

    for (int i=0;i<4;i++) {
        Mat temp_hist=hist[i];
        for (int j=0;j<4;j++) {
            int compare_method = j;
            comparison[i][j] = compareHist( test_hist2, temp_hist, compare_method );
            cout<<comparison[i][j]<<"\t";
        }
        cout<<"\n";
    }

    // finding matched template
    for (int j=0;j<4;j++) {
        float _minm,_maxm;
        if(j==1||j==3) {
            _minm = min( min(comparison[0][j],comparison[1][j]) , min(comparison[2][j],comparison[3][j]) );
            for (int k=0;k<4;k++) {
                if (_minm==comparison[k][j]) matched_templ[j]=k;
            }
        }
        else {
            _maxm = max( max(comparison[0][j],comparison[1][j]) , max(comparison[2][j],comparison[3][j]) );
            for (int k=0;k<4;k++) {
                if (_maxm==comparison[k][j]) matched_templ[j]=k;
            }
        }
        result.push_back(print_nos[matched_templ[j]]);
    }
    cout<<matched_templ[0]<<"\t"<<matched_templ[1]<<"\t"<<matched_templ[2]<<"\t"<<matched_templ[3]<<"\n";

    // result: no detected by all 4 methods
    //result=print_nos[matched_templ[3]];
    return result;
}


void Num_Extract::PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey)
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


void Num_Extract::LearnFromImages(CvMat* trainData, CvMat* trainClasses)
{
    Mat img,outfile;
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

            resize(img,outfile,Size(sizex,sizey));
            IplImage copy = outfile;
            IplImage* img2 = &copy;
            vector<float> ders;
            HOG3(img2,ders);
            for (int n = 0; n < ders.size(); n++)
            {
                trainData->data.fl[i*train_samples*ders.size()+ j * ders.size() + n] = ders.at(n);
            }

            trainClasses->data.fl[i*train_samples+j] = i;
        }
    }

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
        int iSecret = rand() % classes;
        sprintf(file, "%s/%d/%d.png", pathToImages, iSecret, rand()%train_samples);
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

vector<int> Num_Extract::AnalyseImage(KNearest knearest, CvSVM SVM, Mat _image)
{

    CvMat* sample2;
    Mat image, gray, blur, thresh;
    vector<int> digits;

    vector < vector<Point> > contours;
    //_image = imread("./images/16.png", 1);


    resize(_image,image,Size(2*sizex,1.2*sizey));
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 2, 2);
    adaptiveThreshold(blur, thresh, 255, 1, 1, 11, 2);
    findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    float _maxBoxArea=0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        Rect rec = boundingRect(contours[i]);
        if (_maxBoxArea < rec.height*rec.width)
        {
            _maxBoxArea = rec.height*rec.width;
        }
    }

    for (size_t i = 0; i < contours.size(); i++)
    {
        vector < Point > cnt = contours[i];
        Rect rec = boundingRect(cnt);
        float rec_area = rec.height*rec.width;
        if (rec_area > 0.55*_maxBoxArea)
        {
            float aspectR = float(rec.height)/float(rec.width);
            if ( aspectR > 1.4)
            {
                Mat roi = image(rec);
                Mat stagedImage;

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
                digits.push_back(int(result));
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

    return digits;
}


void Num_Extract::run (Mat img){
    //Scalar lower(29,92,114);
    //Scalar higher(37,256,256);
    Scalar lower(0,92,114);
    Scalar higher(74,256,256);
    Mat img2 = Mat::zeros( img.size(), CV_8UC3 );
    cvtColor(img,img2,CV_BGR2HSV);
    Mat output;
    inRange(img2 , lower , higher , output);
    
    extract(output,img);

    cout << dst.size()<<endl;

    int print_nos[] = {10,16,37,98};

    if (!temp_match) {

        // timer
        clock_t time=clock();

        CvMat* trainData = cvCreateMat(classes * train_samples, HOG3_size, CV_32FC1);
        CvMat* trainClasses = cvCreateMat(classes * train_samples, 1, CV_32FC1);

        LearnFromImages(trainData, trainClasses);

        KNearest knearest;
        CvSVM SVM;

        switch (classifier) {
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
		vector<int> digits1;
        
		for(int i = 0 ; i<dst.size() ; i++){
			digits1 = AnalyseImage(knearest, SVM, dst[i]);
			digits.push_back(digits1);
			digits1.clear();
		}
        int result_ml[digits.size()];
        time=clock()-time;
        float run_time=((float)time)/CLOCKS_PER_SEC;
        cout<<"Run Time "<<run_time<<"\n";
		cout<<digits[0].size()<<endl;
		for(int i = 0 ; i<digits.size() ; i++){
			cout<< "digits of " <<i<<"th box are " << digits[i][0] <<" & "<<digits[i][1]<<"\n";
			if (digits[i][0]==8 || digits[i][1]==8) result_ml[i] = 98;
        	if (digits[i][0]==7 || digits[i][1]==7) result_ml[i] = 37;
        	if (digits[i][0]==6 || digits[i][1]==6) result_ml[i] = 16;
        	if (digits[i][0]==0 || digits[i][1]==0) result_ml[i] = 10;
        
		}
		cout<<"result ";
		for(int i = 0 ; i<digits.size() ; i++){
			cout<<result_ml[i]<<endl;
		}
        

        // find no from detected digits
      /* for (int i=0; i< print_nos.cols; i++) {

            if (digits[0]==print_dgt[i][0]) result_ml=print_nos[i];
            if (digits[0]==print_dgt[i][1]) result_ml=print_nos[i];
            if (digits[1]==print_dgt[i][0]) result_ml=print_nos[i];
            if (digits[1]==print_dgt[i][1]) result_ml=print_nos[i];
        }
      */
        
    }
    else {
        vector<Mat> hist;
        vector<int> result_hogm; // result of all 4 matching methods
        Mat test_img=imread((string)(pathToImages)+"/"+"16.png");
        // Template Histograms
        hist = HOGMatching_Template();
        // Compare Histogram
        result_hogm = HOGMatching_Compare(hist,test_img);
    }

    waitKey(0);
}





