#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图像
    // 替换为实际的图像路径，这里是当前目录下的 "bird.jpg"
    string image_path = "D:/Codes/OpenCV_Learning/Week 1/Day 1/data/bird.jpg";
    Mat image = imread(image_path);

    // 检查图像是否成功读取
    if (image.empty()) {
        cout << "错误：无法加载图像，请检查路径是否正确。" << endl;
        return -1;
    }

    // 2. 显示图像
    // 创建一个名为 "Display Image" 的窗口，并在其中显示图像
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    // 3. 等待用户按键
    // 参数 0 表示无限等待，直到用户按下任意键
    int key = waitKey(0);

    // 4. 根据用户按键执行操作
    if (key == 's') {  // 如果按下 's' 键
        // 保存图像
        string output_path = "saved_image.jpg";
        imwrite(output_path, image);
        cout << "图像已保存为 " << output_path << endl;
    } else {  // 如果按下其他键
        cout << "图像未保存。" << endl;
    }

    // 5. 关闭所有窗口
    destroyAllWindows();

    return 0;
}