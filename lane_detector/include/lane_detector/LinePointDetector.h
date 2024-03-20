#ifndef LINEPOINTDETECTOR_H
#define LINEPOINTDETECTOR_H

#include "opencv2/opencv.hpp"
#include <iostream>

class LinePointDetector
{
public:
	LinePointDetector()
		: left_reset_flag_(true), right_reset_flag_(true)
	{}

	bool getLeftResetFlag() const;
	bool getRightResetFlag() const;

	void setLeftResetFlag(const bool left_reset_flag);
	void setRightResetFlag(const bool right_reset_flag);

	/**
	 * 아래 함수들은 out에서 in으로 차선 인식을 하는 함수들이다
	 *
	 */

	/**
	 * 왼쪽 초기점 찾는 함수
	 *
	 * @param binary_img 차선을 찾을 roi 이진화 이미지
	 * @param detect_y_offset 차선을 찾을 roi 이미지 상의 y좌표 (0~100)
	 * @param last_point 초기 점을 다시 잡을때 사용하는 전에 인식된 점
	 *
	 * @return 찾은 왼쪽차선 초기점의 x좌표
	 */
	int find_L0_x_out2in(const cv::Mat& binary_img, const int detect_y_offset, const int last_point);

	/**
	 * 오른쪽 초기점 찾는 함수
	 *
	 */
	int find_R0_x_out2in(const cv::Mat& binary_img, const int detect_y_offset, const int last_point);

	/**
	 * 이전에 찾은 왼쪽 점을 사용해 다음점 찾는 함수
	 *
	 * @param binery_img 차선을 찾을 roi 이진화 이미지
	 * @param pre_point 전에 인식된 점
	 * @param detect_y_offset 차선을 찾을 roi 이미지 상의 y좌표 (0~100)
	 * @param line_pixel_threshold 하나의 차선을 나타내는 픽셀 수
	 *
	 * @return 찾은 왼쪽차선 다음점의 x좌표
	 */
	int find_LN_x_out2in(const cv::Mat& binary_img, const int pre_point, const int detect_y_offset, const int line_pixel_threshold);

	/**
	 * 이전에 찾은 오른쪽 점을 사용해 다음점 찾는 함수
	 *
	 */
	int find_RN_x_out2in(const cv::Mat& binary_img, const int pre_point, const int detect_y_offset, const int line_pixel_threshold);


	/**
	 * 아래 함수들은 in에서 out으로 차선 인식을 하는 함수들이다
	 *
	 */

	/**
	 * @param offset 안에서 인식이 시작되는 위치로, 센터를 기준으로의 offset을 의미한다.
	 * 		  		 이 함수는 left line을 인식하는 함수이므로 센터로부터 오른쪽 방향으로의 offset을 의미한다.
	 *
	 */
	int find_L0_x_in2out(const cv::Mat& binary_img, const int detect_y_offset, const int last_point, const int offset);

	/**
	 * @param offset 안에서 인식이 시작되는 위치로, 센터를 기준으로의 offset을 의미한다.
	 * 		  		 이 함수는 right line을 인식하는 함수이므로 센터로부터 왼쪽 방향으로의 offset을 의미한다.
	 *
	 */
	int find_R0_x_in2out(const cv::Mat& binary_img, const int detect_y_offset, const int last_point, const int offset);

	int find_LN_x_in2out(const cv::Mat& binary_img, const int pre_point, const int detect_y_offset, const int line_pixel_threshold, const int offset);

	int find_RN_x_in2out(const cv::Mat& binary_img, const int pre_point, const int detect_y_offset, const int line_pixel_threshold, const int offset);

private:
	bool left_reset_flag_, right_reset_flag_;

};

#endif
