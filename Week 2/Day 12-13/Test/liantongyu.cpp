#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

void labelColor(Mat& labelImg, Mat& dst)
{
    map<int, Scalar> colors;

    int width = labelImg.cols;
    int height = labelImg.rows;

    dst = Mat::zeros(labelImg.size(), CV_8UC3);

    uchar r = 255 * (rand()/(1.0 + RAND_MAX));
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));

    for (int i = 0; i < height; i++)
    {
        int* data_src = (int*)labelImg.ptr<int>(i);
        uchar* data_dst = dst.ptr<uchar>(i);

        for (int j = 0; j < width; j++)
        {
            int pixelValue = data_src[j];
            if (pixelValue >= 1)
            {
                if (colors.count(pixelValue) == 0)
                {
                    colors[pixelValue] = Scalar(b,g,r);
                    r = 255 * (rand()/(1.0 + RAND_MAX));
                    g = 255 * (rand()/(1.0 + RAND_MAX));
                    b = 255 * (rand()/(1.0 + RAND_MAX));
                }

                Scalar color = colors[pixelValue];
                *data_dst++ = color[0];
                *data_dst++ = color[1];
                *data_dst++ = color[2];
            }
            else
            {
                data_dst++;
                data_dst++;
                data_dst++;
            }
        }
    }
}

int main(int argc, char **argv) {
    Mat src = imread("D:\\Codes\\OpenCV_Learning\\Week 2\\Day 12-13\\Data\\coin.png");
    imshow("src", src);

    Mat gray,thresh;
    cvtColor(src, gray, cv::COLOR_BGR2GRAY); // 灰度化P
    imshow("gray", gray);

    Mat gauss;
    GaussianBlur(gray, gauss, Size(15, 15),0); //降噪

    threshold(gauss, thresh,0,255,THRESH_BINARY | THRESH_OTSU );
    imshow("thresh", thresh);

    Mat labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(thresh, labels, stats, centroids,8, CV_32S);
    cout << "num_labels = " << num_labels << endl;

    for (int i = 1; i < num_labels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int left = stats.at<int>(i, CC_STAT_LEFT);
        int top = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);
        int cx = centroids.at<double>(i, 0);
        int cy = centroids.at<double>(i, 1);

        cout << "Object " << i << ": " << "Area=" << area << ", Left=" << left << ", Top=" << top << ", Width=" << width << ", Height=" << height << ", Centroid=(" << cx << ", " << cy << ")" << endl;

        rectangle(src, Point(left, top), Point(left + width, top + height), Scalar(0,0,255), 2);
        circle(src, Point(cx, cy), 2, Scalar(0,0,255), 2);
    }
    imshow("Connected Components", src);

    waitKey(0);

    return 0;
}