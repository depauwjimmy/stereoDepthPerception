#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>

using namespace cv;
using std::string;
using std::vector;

cv::Mat pts3D;
cv::Mat filtered_disp_16;

double focalSize, doffs;
double baseLine;
float squareSize = 3.5f; // cm

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;

    Point seed = Point(x,y);
    float d = filtered_disp_16.at<float>(y,x);
    std::cout << "Clicked on : " << seed << " > Disparity 1 : " << d << " > Disparity 2 : " << d+doffs << '\n';

    double ZZ = ((focalSize * baseLine) / (d+doffs)) ;
    std::cout << "Distance (cm) : " << ZZ * squareSize << '\n';

    std::cout << "Reproject3D (raw) : " << pts3D.at<Vec3f>(y,x)[2] << '\n';
}

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv, "{intrinsics||}{extrinsics||}{leftImageP||}{rightImageP||}");
    string intrinsicsFilePath = parser.get<string>("intrinsics");
    string extrinsicsFilePath = parser.get<string>("extrinsics");
    string leftImagePath = parser.get<string>("leftImageP");
    string rightImagePath = parser.get<string>("rightImageP");

    Mat leftImage = imread(leftImagePath, 0);
    Mat rightImage = imread(rightImagePath, 0);

    // Values from stereoCalibrate
    Mat leftCameraMatrix, rightCameraMatrix;
    Mat leftDistorsionMatrix, rightDistorsionMatrix;
    Mat leftRectificationMatrix, rightRectificationMatrix;
    Mat leftProjectionMatrix, rightProjectionMatrix;
    Mat Q;

    FileStorage fsINT(intrinsicsFilePath, FileStorage::READ);
    fsINT["M1"] >> leftCameraMatrix;
    fsINT["M2"] >> rightCameraMatrix;
    fsINT["D1"] >> leftDistorsionMatrix;
    fsINT["D2"] >> rightDistorsionMatrix;
    fsINT.release();

    FileStorage fsEXT(extrinsicsFilePath, FileStorage::READ);
    fsEXT["R1"] >> leftRectificationMatrix;
    fsEXT["R2"] >> rightRectificationMatrix;
    fsEXT["P1"] >> leftProjectionMatrix;
    fsEXT["P2"] >> rightProjectionMatrix;
    fsEXT["Q"] >> Q;
    fsEXT.release();

    /*
     Q Matrix detail
     https://stackoverflow.com/questions/27374970/q-matrix-for-the-reprojectimageto3d-function-in-opencv
    */

    // Get useful values from camera matrix
    focalSize = Q.at<double>(2,3);
    std::cout << "Focal Length : " << focalSize << '\n';

    double offsetLeft = leftCameraMatrix.at<double>(0,2);
    double offsetRight = rightCameraMatrix.at<double>(0,2);
    doffs = offsetLeft - offsetRight;
    std::cout << "Doffs : " << doffs << '\n';

    double baseLineQ = Q.at<double>(3,2);
    baseLine = 1/baseLineQ;
    std::cout << "BaseLine 1 : " << baseLine << '\n';

    // Un-distort source images
    Mat rectified_mapping_[2][2];
    initUndistortRectifyMap(leftCameraMatrix, leftDistorsionMatrix, leftRectificationMatrix, leftProjectionMatrix,
                            Size(640, 480), CV_32F, rectified_mapping_[0][0], rectified_mapping_[0][1]);
    initUndistortRectifyMap(rightCameraMatrix, rightDistorsionMatrix, rightRectificationMatrix, rightProjectionMatrix,
                            Size(640, 480), CV_32F, rectified_mapping_[1][0], rectified_mapping_[1][1]);

    Mat leftImageCorr, rightImageCorr;
    remap(leftImage, leftImageCorr, rectified_mapping_[0][0], rectified_mapping_[0][1], INTER_LINEAR);
    remap(rightImage, rightImageCorr, rectified_mapping_[1][0], rectified_mapping_[1][1], INTER_LINEAR);

    // Matching
    // Tried values found in https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
    // Whatever i tried here gives me the same distance though
    Mat left_disp, right_disp;

    int blockSize = 3;
    int cn = leftImageCorr.channels();
    Size img_size = leftImageCorr.size();
    int P1 = 8 * cn * blockSize * blockSize;
    int P2 = 32 * cn * blockSize * blockSize;
    int numberOfDisparities = ((img_size.width/8) + 15) & -16;

    std::cout << "Number of Disparities : " << numberOfDisparities << '\n';

    auto left_matcher = cv::StereoSGBM::create(0, numberOfDisparities, blockSize);
    left_matcher->setP1(P1);
    left_matcher->setP2(P2);
    left_matcher->setUniquenessRatio(10);
    left_matcher->setSpeckleWindowSize(100);
    left_matcher->setSpeckleRange(32);
    left_matcher->setDisp12MaxDiff(1);
    left_matcher->setMode(StereoSGBM::MODE_HH);

    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    left_matcher->compute(leftImageCorr, rightImageCorr, left_disp);
    right_matcher->compute(rightImageCorr, leftImageCorr, right_disp);

    //Filtering
    double lambda = 8000.0;
    double sigma = 1.5;
    double vis_mult = 1.0;

    cv::Mat filtered_disp;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disp, leftImageCorr, filtered_disp, right_disp);

    filtered_disp.convertTo(filtered_disp_16, CV_32F, 1./16);
    reprojectImageTo3D(filtered_disp_16, pts3D, Q, false, CV_32F);

    //! [visualization]

    Mat raw_disp_vis;
    cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    namedWindow("raw_disparity", WINDOW_AUTOSIZE);
    imshow("raw_disparity", raw_disp_vis);

    Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    namedWindow("filtered_disparity", WINDOW_AUTOSIZE);
    setMouseCallback("filtered_disparity", onMouse, nullptr);

    imshow("filtered_disparity", filtered_disp_vis);

    waitKey(0);
}