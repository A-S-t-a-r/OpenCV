#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>  // 光流法
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <vector>

using namespace std;
using namespace cv;

// 使用光流法进行跟踪
void opticalFlowTracking(Mat& prevFrame, Mat& currentFrame, vector<Point2f>& prevPoints, vector<Point2f>& currentPoints) {
    if (prevPoints.empty()) return;
    
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    
    calcOpticalFlowPyrLK(prevFrame, currentFrame, prevPoints, currentPoints, status, err, Size(21, 21), 3, criteria);
    
    // 移除丢失的点
    size_t i, k;
    for (i = k = 0; i < currentPoints.size(); i++) {
        if (status[i]) {
            currentPoints[k++] = currentPoints[i];
        }
    }
    currentPoints.resize(k);
}

int main(int argc, char** argv) {
    // 显示帮助
    if (argc < 2) {
        cout << "用法：tracker <video_name>" << endl;
        cout << "示例：" << endl;
        cout << "tracker Bolt/img/%04d.jpg" << endl;
        cout << "tracker faceocc2.webm" << endl;
        return 0;
    }
    
    // 设置输入视频
    string video = "D:\\Codes\\OpenCV_Learning\\Week 2\\Day 12-13\\Data\\test.MP4";
    VideoCapture cap(video);
    
    if (!cap.isOpened()) {
        cout << "无法打开视频文件：" << video << endl;
        return -1;
    }
    
    Mat frame, prevFrame, gray, prevGray;
    vector<Point2f> prevPoints, currentPoints;
    Rect roi;
    bool tracking = false;
    
    cout << "按以下键控制：" << endl;
    cout << "  SPACE - 选择/重新选择跟踪区域" << endl;
    cout << "  ESC   - 退出" << endl;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 绘制当前跟踪点
        for (size_t i = 0; i < currentPoints.size(); i++) {
            circle(frame, currentPoints[i], 3, Scalar(0, 255, 0), -1);
        }
        
        // 绘制跟踪区域（如果有）
        if (roi.width > 0 && roi.height > 0) {
            rectangle(frame, roi, Scalar(255, 0, 0), 2);
        }
        
        imshow("光流跟踪", frame);
        
        int key = waitKey(30);
        
        if (key == 27) {  // ESC
            break;
        } else if (key == 32) {  // SPACE
            // 选择ROI
            roi = selectROI("光流跟踪", frame, false, false);
            if (roi.width > 0 && roi.height > 0) {
                // 在ROI内检测角点
                Mat roiGray = gray(roi);
                vector<Point2f> corners;
                goodFeaturesToTrack(roiGray, corners, 100, 0.01, 10);
                
                // 将角点坐标转换回原图坐标
                prevPoints.clear();
                for (size_t i = 0; i < corners.size(); i++) {
                    prevPoints.push_back(Point2f(corners[i].x + roi.x, corners[i].y + roi.y));
                }
                
                tracking = true;
                prevGray = gray.clone();
            }
        }
        
        // 如果正在跟踪且有点可以跟踪
        if (tracking && !prevPoints.empty()) {
            currentPoints = prevPoints;
            opticalFlowTracking(prevGray, gray, prevPoints, currentPoints);
            
            // 计算边界框
            if (!currentPoints.empty()) {
                vector<Point2f> pointsInRoi;
                for (size_t i = 0; i < currentPoints.size(); i++) {
                    if (roi.contains(currentPoints[i])) {
                        pointsInRoi.push_back(currentPoints[i]);
                    }
                }
                
                if (!pointsInRoi.empty()) {
                    // 计算点集的最小外接矩形
                    roi = boundingRect(pointsInRoi);
                }
            }
            
            // 更新为下一帧
            prevGray = gray.clone();
            prevPoints = currentPoints;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    return 0;
}