#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<string>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

int main(){
    //创建 VideoCapture 对象并打开视频文件
    VideoCapture cap("D:\\Codes\\OpenCV_Learning\\Week 1\\Homework\\Data\\test.mp4");

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        cout << "错误：无法加载图像，请检查路径是否正确。" << endl;
        return -1;
    }

    // 获取视频的帧率
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) {
        fps = 30.0;  // 如果获取失败，使用默认30fps
    }
    int delay_ms = static_cast<int>(1000.0 / fps);  // 每帧的延迟时间(ms)

    Mat frame,grayFrame,small;
    vector<string> lines;
    string ascii_art;
    
    while(true){
        cap>>frame;
        if(frame.empty()){
            break;
        }

        // 将帧转换为灰度图像
        if(frame.channels()==3){
            cvtColor(frame,grayFrame,COLOR_BGR2GRAY) ;
        }
        else{
            grayFrame = frame.clone();
        }

        // 调整大小
        resize(grayFrame,small,Size(120,40)) ;

        // 清空上一帧的数据
        lines.clear();
        ascii_art.clear();

        // 转化为字符画
        for(int row=0;row<small.rows;row++){
            string line = "";
            line.reserve(small.cols);  // 预分配空间提高性能
            for(int col=0;col<small.cols;col++){
                uchar pixel = small.at<uchar>(row, col);
                line += (pixel>127) ? '$' : ' ';
            }
            lines.push_back(line);
        }

        // 构建完整的ASCII艺术字符串
        for (size_t i = 0; i < lines.size(); i++) {
            ascii_art += lines[i];
            if (i != lines.size() - 1) {
                ascii_art += "\n";
            }
        }

        // 清屏并打印 (使用跨平台方法)
        #ifdef _WIN32
            system("cls");  // Windows清屏
        #else
            system("clear");  // Linux/macOS清屏
        #endif

        // 打印字符画
        cout<<ascii_art<<endl;

        // 控制播放速度
        this_thread::sleep_for(chrono::milliseconds(delay_ms));

    }

    cap.release();
    return 0;
}