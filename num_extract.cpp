#include "num_extract.hpp"

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





