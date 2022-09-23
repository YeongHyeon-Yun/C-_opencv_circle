#if 0
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

void sobel_derivative();
void sobel_edge();
void canny_edge();

int main(void)
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//sobel_derivative();
	//sobel_edge();
	canny_edge();

	return 0;
}

void sobel_derivative()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat mx = Mat_<float>({ 1, 3 }, { -1 / 2.f, 0, 1 / 2.f });
	Mat my = Mat_<float>({ 3, 1 }, { -1 / 2.f, 0, 1 / 2.f });

	Mat dx, dy;
	filter2D(src, dx, -1, mx, Point(-1, -1), 128);
	filter2D(src, dy, -1, my, Point(-1, -1), 128);

	imshow("src", src);
	imshow("dx", dx);
	imshow("dy", dy);

	waitKey();
	destroyAllWindows();
}

void sobel_edge()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dx, dy;
	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);

	Mat fmag, mag;
	magnitude(dx, dy, fmag);
	fmag.convertTo(mag, CV_8UC1);
	dx.convertTo(dx, CV_8UC1);
	dy.convertTo(dy, CV_8UC1);

	Mat edge = mag > 150;

	imshow("src", src);
	imshow("dx", dx);
	imshow("dy", dy);
	imshow("mag", mag);
	imshow("edge", edge);

	waitKey();
	destroyAllWindows();
}

void canny_edge()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat dst1, dst2;
	Canny(src, dst1, 50, 100);
	Canny(src, dst2, 50, 150);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

#endif


#if 0
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;
Mat cartoon_filter(Mat img)
{
	int h, w;
	Mat img2;
	Mat blr;
	Mat edge;
	Mat dst;
	h = img.rows;
	w = img.cols;

	resize(img, img2, Size(w / 2, h / 2));
	bilateralFilter(img2, blr, -1, 20, 7);
	Canny(img2, edge, 80, 120);
	cvtColor(edge, edge, COLOR_GRAY2BGR);

	bitwise_and(blr, edge, dst);
	resize(dst, dst, Size(w, h), INTER_NEAREST);
	return dst;
}

Mat pencil_sketch(Mat img)
{
	Mat gray;
	Mat blr;
	Mat dst;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blr, Size(), 10);
	divide(gray, blr, dst, 255);
	return dst;
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	VideoCapture cap(0);

	int cam_mode = 0;
	int key;

	if(!cap.isOpened()) {
		cerr << " Camera open failed! " << endl;
		return -1;
	}

	Mat frame;

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		if (cam_mode == 1)
			frame = cartoon_filter(frame);

		else if (cam_mode == 2)
			frame = pencil_sketch(frame);

		else if (cam_mode == 3)
			frame = ~frame;

		imshow("frame", frame);

		key = waitKey(10);
		if (key == 27)
			break;
		else if (key == 'i') {
			cam_mode++;
			if (cam_mode == 4)
				cam_mode = 0;
		}
	}
	destroyAllWindows();

}

#endif


//   HoughCircles 검출하기
#if 0
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
using namespace std;
using namespace cv;

void hough_lines();
void hough_line_segments();
void hough_circles();
void hough_circle_dial();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//hough_lines();
	//hough_line_segments();
	//hough_circles();
	hough_circle_dial();
	cout << "open :" << CV_VERSION << endl;
	return 0;
}

void hough_lines()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat edge;
	Canny(src, edge, 50, 150);

	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 250);

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		float cos_t = cos(theta), sin_t = sin(theta);
		float x0 = rho * cos_t, y0 = rho * sin_t;
		float alpha = 1000;

		Point pt1(cvRound(x0 - alpha * sin_t), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 + alpha * sin_t), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void hough_line_segments() {
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 100, 160, 50, 5);

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (Vec4i i : lines) {
		line(dst, Point(i[0], i[1]), Point(i[2], i[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_circle_dial() {
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}
void hough_circles() {
	Mat src = imread("coins.png", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}

#endif


#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
using namespace std;
using namespace cv;
void hough_lines();
void hough_line_segments();
void hough_circles();
void hough_circle_dial();


int min_ = 0;
int max_ = 0;
int thres = 50;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	//hough_lines();
	//hough_line_segments();
	//hough_circles();
	hough_circle_dial();

	cout << "opencv : " << CV_VERSION << endl;
	return 0;
}

void onChange(int pos, void* userdata) {
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, thres, 150, 30, min_, max_);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("dst", dst);
}
void hough_circle_dial() {
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, thres, 150, 30, min_, max_);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	namedWindow("dst");
	createTrackbar("min", "dst", &min_, 150, onChange);
	createTrackbar("max", "dst", &max_, 150, onChange);
	createTrackbar("thres", "dst", &thres, 150, onChange);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_circles()
{
	Mat src = imread("coins.png", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_line_segments()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5);	// 떨어져있는거 5개까진 그려짐
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (Vec4i i : lines) {
		line(dst, Point(i[0], i[1]), Point(i[2], i[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}
void hough_lines()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 250);
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t, y0 = r * sin_t;
		double alpha = 1000;
		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows;
}
#endif


#if 0
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

void hough_lines();
void hough_lines_segments();
void hough_circles();
void hough_circles_dial();
void count_money();

int minDist = 50;
int minRadius = 0;
int maxRadius = 0;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//hough_lines();
	//hough_lines_segments();
	//hough_circles();
	//hough_circles_dial(); // 다이얼 트랙바
	count_money();
	return 0;
}

void count_money() {
	Mat src = imread("coins1.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}

	Mat blurred;
	blur(src, blurred, Size(3, 3));

	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30, 30, 100);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	int total = 0;

	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		if (radius > 40) total += 100;
		else total += 10;

		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	string myText = "Total Money: ";
	myText += to_string(total);
	putText(dst, myText, Point(10, 70), FONT_HERSHEY_DUPLEX, 2, Scalar(0, 255, 255));


	imshow("src", src);
	imshow("dst", dst);
	cout << total << endl;
	waitKey(0);
	destroyAllWindows();
}

void onChange(int pos, void* userdata) {
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}

	Mat blurred;
	blur(src, blurred, Size(3, 3));

	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, minDist, 150, 30, minRadius, maxRadius);
	//HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, minDist, 30, minRadius, maxRadius);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("dst", dst);
}

void hough_circles_dial() {
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}

	Mat blurred;
	blur(src, blurred, Size(3, 3));

	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, minDist, 150, 30, minRadius, maxRadius);
	//HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, minDist, 30, minRadius, maxRadius);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	namedWindow("dst");
	createTrackbar("minRadius", "dst", &minRadius, 150, onChange);
	createTrackbar("maxRadius", "dst", &maxRadius, 150, onChange);
	createTrackbar("minDist", "dst", &minDist, 150, onChange);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_circles() {
	Mat src = imread("coins.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f>circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_lines_segments() {
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5); // 마지막 4 부분이 떨어지는 픽셀값이 4까지는 같은 라인으로 보라는 의미
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (Vec4i i : lines) {
		line(dst, Point(i[0], i[1]), Point(i[2], i[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_lines() {
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "image load failed" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 250);
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t, y0 = r * sin_t;
		double alpha = 1000;
		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();

}
#endif




#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

void hough_lines();
void hough_line_segments();
void hough_circles();
void hough_circle_dial();
void total_money();
void video_play();

int min_dist = 50, min_Radius = 10, max_Radius = 80;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//hough_lines();
	//hough_line_segments();
	//hough_circles();
	//hough_circle_dial();
	total_money();
	//video_play();
	return 0;
}
void video_play()
{
	VideoCapture cap1("woman.mp4");
	VideoCapture cap2("raining.mp4");

	int frame_cnt1 = round(cap1.get(CAP_PROP_FRAME_COUNT));
	int frame_cnt2 = round(cap1.get(CAP_PROP_FRAME_COUNT));

	double fps = cap1.get(CAP_PROP_FPS);
	int delay = (int)(1000 / fps);
	bool do_composit = false;

	while (true)
	{
		Mat frame1;
		cap1.read(frame1);
		if (frame1.empty())
			break;

		if (do_composit) {
			Mat frame2;
			cap2.read(frame2);
			if (frame2.empty())
				break;
			Mat hsv, mask;
			cvtColor(frame1, hsv, COLOR_BGR2HSV);
			inRange(hsv, Scalar(50, 150, 0), Scalar(70, 255, 255), mask);
			copyTo(frame2, frame1, mask);
		}
		imshow("frame", frame1);
		int key = waitKey(delay);
		if (key == 32)  // 스페이스바를 눌렀을때 배경을 전환
			do_composit = !do_composit;
		else if (key == 27)
			break;
	}
	cap1.release();
	cap2.release();
	destroyAllWindows();
}

void total_money()
{
	Mat src = imread("coins1.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 40, 150, 30, 10, 80);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	int total_money = 0;
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		if (radius >= 40)
			total_money += 100;
		else
			total_money += 10;
		circle(dst, center, radius, Scalar(100, 100, 255), 2, LINE_AA);
	}
	char mystr[30];
	sprintf_s(mystr, "Total Money : %d won", total_money);
	putText(dst, mystr, Point(20, 60), 2, 2, Scalar(0, 255, 255), 2);
	cout << "total money : " << total_money << endl;
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void on_trackbar(int pos, void* userdata)
{
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, min_dist, 150, 30, min_Radius, max_Radius);
	Mat dst = imread("dial.jpg");
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
	}
	imshow("dst", dst);
}

void hough_circle_dial()
{
	Mat src = imread("dial.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));
	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, min_dist, 150, 30, min_Radius, max_Radius);
	Mat dst = imread("dial.jpg");
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	namedWindow("dst");
	createTrackbar("minRadius", "dst", &min_Radius, 150, on_trackbar);
	createTrackbar("maxRadius", "dst", &max_Radius, 150, on_trackbar);
	createTrackbar("threshold", "dst", &min_dist, 100, on_trackbar);
	//imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_circles()
{
	Mat src = imread("coins1.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat blurred;
	blur(src, blurred, Size(3, 3));

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 150, 30);
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (Vec3f c : circles) {
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_line_segments()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5);	// 떨어져있는거 5개까진 그려짐
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (Vec4i i : lines) {
		line(dst, Point(i[0], i[1]), Point(i[2], i[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

void hough_lines()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat edge;
	Canny(src, edge, 50, 150);
	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 250);
	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t, y0 = r * sin_t;
		double alpha = 1000;
		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("src", src);
	imshow("edge", edge);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows;
}
#endif


#if 1
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
using namespace std;
using namespace cv;
Mat src;
Mat blr;
void on_trackbar(int, void*);
void hough_circle_dial()
{
	src = imread("dial.jpg");
	if (src.empty()) {
		cout << "Image load failed!" << endl;
		return;
	}
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blr, Size(0, 0), 1.0);
	imshow("img", src);
	createTrackbar("minRadius", "img", 0, 100, on_trackbar);
	createTrackbar("maxRadius", "img", 0, 150, on_trackbar);
	createTrackbar("threshold", "img", 0, 100, on_trackbar);
	setTrackbarPos("minRadius", "img", 10);
	setTrackbarPos("maxRadius", "img", 80);
	setTrackbarPos("threshold", "img", 40);
	waitKey();
	destroyAllWindows();

}
void on_trackbar(int pos, void* userdata)
{
	int rmin, rmax, th;
	static int count = 0;
	rmin = getTrackbarPos("minRadius", "img");
	rmax = getTrackbarPos("maxRadius", "img");
	th = getTrackbarPos("threshold", "img");
	count++;
	if (count < 4)
		return;
	vector<Vec3f> circles;
	HoughCircles(blr, circles, HOUGH_GRADIENT, 1, 50, 120.0, th, rmin, rmax);
	Mat dst;
	src.copyTo(dst);
	if (!circles.empty()) {
		for (Vec3f c : circles) {
			Point center(cvRound(c[0]), cvRound(c[1]));
			int radius = cvRound(c[2]);
			circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
		}
	}
	imshow("img", dst);
}
int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	hough_circle_dial();
	return 0;
}
#endif