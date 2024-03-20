#include <cmath>
#include <string>
#include <stdexcept>
#include "lane_detector/LaneDetectorNode.h"

using namespace std;
using namespace cv;

LaneDetectorNode::LaneDetectorNode()
{
	nh_ = ros::NodeHandle("~");
	/* if NodeHangle("~"), then (write -> /lane_detector/write)	*/
#if RC_CAR
	control_pub_ = nh_.advertise<std_msgs::String>("write", 100);
#elif	SCALE_PLATFORM
	control_pub_ = nh_.advertise<ackermann_msgs::AckermannDriveStamped>("camera_ackermann", 100);
#endif

#if DEBUG
	true_color_pub_ = nh_.advertise<sensor_msgs::Image>("truecolor", 10);
	final_bin_pub_ = nh_.advertise<sensor_msgs::Image>("final_bin", 10);
	bin_from_gray_pub_ = nh_.advertise<sensor_msgs::Image>("gray_bin", 10);
	bin_from_hsv_s_pub_ = nh_.advertise<sensor_msgs::Image>("hsv_s_bin", 10);
	printlog_pub_ = nh_.advertise<std_msgs::String>("printlog", 10);

	
#endif

#if WEBCAM
	image_sub_ = nh_.subscribe("/Camera_1/image_raw/compressed", 100, &LaneDetectorNode::imageCallback, this);
	status_sub_ = nh_.subscribe("/gps_state", 1, &LaneDetectorNode::GPSCallback, this);
#elif	PROSILICA_GT_CAM
	image_sub_ = nh_.subscribe("/camera/image_raw", 100, &LaneDetectorNode::imageCallback, this);
	   
#endif

	int resize_width = 0;
	int resize_height = 0;
	int steer_max_angle = 0;
	int detect_line_count = 0;
	

	getRosParamForConstValue(resize_width, resize_height, steer_max_angle, detect_line_count);

	lanedetector_ptr_ = unique_ptr<InToOutLaneDetector>(new InToOutLaneDetector(resize_width, resize_height, steer_max_angle, detect_line_count));

	getRosParamForUpdate();
}

void LaneDetectorNode::GPSCallback(const waypoint_maker::State::ConstPtr &lane_num){
	   cur_state_ = lane_num->current_state;
	    
}
    
void LaneDetectorNode::imageCallback(const sensor_msgs::CompressedImage::ConstPtr& image){   
    if(cur_state_ == 0 || cur_state_ == 8){
    
	    Mat raw_img;
	    try{
		    raw_img = parseRawimg(image);
	    } catch(const cv_bridge::Exception& e) {
		    ROS_ERROR("cv_bridge exception: %s", e.what());
		    return ;
	    } catch(const std::runtime_error& e) {
		    cerr << e.what() << endl;
	    }
    
        getRosParamForUpdate();

        int steer_control_value = lanedetector_ptr_->laneDetecting(raw_img);
#if	RC_CAR
	    std_msgs::String control_msg = makeControlMsg(steer_control_value);
	    printData(control_msg);
#elif	SCALE_PLATFORM
	    ackermann_msgs::AckermannDriveStamped control_msg = makeControlMsg();
	    printData();
#endif

	    control_pub_.publish(control_msg);

#if DEBUG
	    true_color_pub_.publish(getDetectColorImg());
	    final_bin_pub_.publish(getDetectFinalBinImg());
	    bin_from_gray_pub_.publish(getDetectGrayBinImg());
	    bin_from_hsv_s_pub_.publish(getDetectHsvSBinImg());
	    printlog_pub_.publish(getPrintlog());
    }
#endif
}
}

#if DEBUG
sensor_msgs::ImagePtr LaneDetectorNode::getDetectColorImg()
{
	const Mat& image = lanedetector_ptr_->getRoiColorImg();
	return cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
}

sensor_msgs::ImagePtr LaneDetectorNode::getDetectFinalBinImg()
{
	const Mat& image = lanedetector_ptr_->getRoiBinaryImg();
	return cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
}

sensor_msgs::ImagePtr LaneDetectorNode::getDetectGrayBinImg()
{
	const Mat& image = lanedetector_ptr_->getRoiGrayBinImg();
	return cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
}

sensor_msgs::ImagePtr LaneDetectorNode::getDetectHsvSBinImg()
{
	const Mat& image = lanedetector_ptr_->getRoiHsvSBinImg();
	return cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
}

std_msgs::String LaneDetectorNode::getPrintlog()
{
	string log;
 	log += "#### Algorithm Time #### \n";
	log += (string)"it took : " + to_string(lanedetector_ptr_->getOnceDetectTime()) + "ms, " + "avg: " + to_string(lanedetector_ptr_->getAvgDetectTime()) + " fps : " + to_string(1000 / lanedetector_ptr_->getAvgDetectTime()) + '\n';
	log += "#### Control #### \n";
	log += "steering angle: " + to_string(lanedetector_ptr_->getRealSteerAngle()) + '\n';
	log += "throttle: " + to_string(throttle_) + '\n';
	log += "#### Ros Param #### \n";
	log += "gray_bin_thres: " + to_string(lanedetector_ptr_->getGrayBinThres()) + '\n';
	log += "hsv_s_bin_thres: " + to_string(lanedetector_ptr_->getHsvSBinThres()) + '\n';
	log += "detect_line_count: " + to_string(lanedetector_ptr_->getDetectLineCount()) + '\n';
	for(int i = 0; i < lanedetector_ptr_->getDetectLineCount(); i++)
		log += "detect_y_offset_" + to_string(i+1) + ": " + to_string(lanedetector_ptr_->getDetectYOffset(i)) + '\n';
	log += "left_detect_offset: " + to_string(lanedetector_ptr_->getLeftDetectOffset()) + '\n';
	log += "right_detect_offset: " + to_string(lanedetector_ptr_->getRightDetectOffset()) + '\n';
	log += "steer_max_angle: " + to_string(lanedetector_ptr_->getSteerMaxAngle()) + '\n';
	log += "yaw_factor: " + to_string(lanedetector_ptr_->getYawFactor() * 100) + "% -> " + to_string(lanedetector_ptr_->getYawFactor()) + '\n';
	log += "lateral_factor: " + to_string(lanedetector_ptr_->getLateralFactor() * 100) + "% -> " + to_string(lanedetector_ptr_->getLateralFactor()) + '\n';
	log += "---------------------------------\n";

	std_msgs::String log_msg;
	log_msg.data = log;
	return log_msg;

}

#endif

Mat LaneDetectorNode::parseRawimg(const sensor_msgs::CompressedImage::ConstPtr& image)
{   
    
    
    if(cur_state_ == 0 || cur_state_ == 8){
	    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

	    Mat raw_img = cv_ptr->image;

	    if (raw_img.empty()) {
		    throw std::runtime_error("frame is empty!");
	    }

	    return raw_img;
	}
}

void LaneDetectorNode::getRosParamForConstValue(int& width, int& height, int& steer_max_angle, int& detect_line_count)
{
	nh_.getParam("resize_width", width);
	nh_.getParam("resize_height", height);
	nh_.getParam("steer_max_angle", steer_max_angle);
	nh_.getParam("detect_line_count", detect_line_count);
}

void LaneDetectorNode::getRosParamForUpdate()
{
	int paramArr[6];
	nh_.getParam("gray_bin_thres", paramArr[0]);
	nh_.getParam("hsv_s_bin_thres", paramArr[1]);
	nh_.getParam("left_detect_offset", paramArr[2]);
	nh_.getParam("right_detect_offset", paramArr[3]);
	nh_.getParam("yaw_factor", paramArr[4]);
	nh_.getParam("lateral_factor", paramArr[5]);
	nh_.getParam("throttle", throttle_);

	lanedetector_ptr_->setGrayBinThres(paramArr[0]);
	lanedetector_ptr_->setHsvSBinThres(paramArr[1]);
	lanedetector_ptr_->setLeftDetectOffset(paramArr[2]);
	lanedetector_ptr_->setRightDetectOffset(paramArr[3]);
	lanedetector_ptr_->setYawFactor((double)paramArr[4] / 100);
	lanedetector_ptr_->setLateralFactor((double)paramArr[5] / 100);

	int detect_line_count = lanedetector_ptr_->getDetectLineCount();
	for(int i = 0; i < detect_line_count; i++) {
		int y_offset = 0;
		nh_.getParam("detect_y_offset_" + to_string(i+1), y_offset);
		lanedetector_ptr_->setDetectYOffset(y_offset, i);
	}
}

#if RC_CAR
std_msgs::String LaneDetectorNode::makeControlMsg(int steer)
{
	std_msgs::String control_msg;
	control_msg.data = string(to_string(steer)) + "," + string(to_string(throttle_)) + ","; // Make message
	return control_msg;
}

void LaneDetectorNode::printData(std_msgs::String control_msg)
{
 	cout << "#### Algorithm Time ####" << endl;
	cout << "it took : " << lanedetector_ptr_->getOnceDetectTime() << "ms, " << "avg: " << lanedetector_ptr_->getAvgDetectTime() << " fps : " << 1000 / lanedetector_ptr_->getAvgDetectTime() << endl;
	cout << "#### Control ####" << endl;
	cout << "steering angle: " << lanedetector_ptr_->getRealSteerAngle() << endl;
	cout << "control msg: " << control_msg.data << endl;
	cout << "#### Ros Param ####" << endl;
	cout << "gray_bin_thres: " << lanedetector_ptr_->getGrayBinThres() << endl;
	cout << "hsv_s_bin_thres: " << lanedetector_ptr_->getHsvSBinThres() << endl;
	cout << "detect_line_count: " << lanedetector_ptr_->getDetectLineCount() << endl;
	for(int i = 0; i < lanedetector_ptr_->getDetectLineCount(); i++)
		cout << "detect_y_offset_" << i+1 << ": " << lanedetector_ptr_->getDetectYOffset(i) << endl;
	cout << "left_detect_offset: " << lanedetector_ptr_->getLeftDetectOffset() << endl;
	cout << "right_detect_offset: " << lanedetector_ptr_->getRightDetectOffset() << endl;
	cout << "steer_max_angle: " << lanedetector_ptr_->getSteerMaxAngle() << endl;
	cout << "yaw_factor: " << lanedetector_ptr_->getYawFactor() * 100 << "% -> " << lanedetector_ptr_->getYawFactor() << endl;
	cout << "lateral_factor: " << lanedetector_ptr_->getLateralFactor() * 100 << "% -> " << lanedetector_ptr_->getLateralFactor() << endl;
	cout << "---------------------------------" << endl;
}

#elif SCALE_PLATFORM
ackermann_msgs::AckermannDriveStamped LaneDetectorNode::makeControlMsg()
{
	ackermann_msgs::AckermannDriveStamped control_msg;
	//control_msg.drive.steering_angle = steer_control_value;
	control_msg.drive.steering_angle = lanedetector_ptr_->getRealSteerAngle();
	control_msg.drive.speed = throttle_;
	return control_msg;
}

void LaneDetectorNode::printData()
{
 	cout << "#### Algorithm Time ####" << endl;
	cout << "it took : " << lanedetector_ptr_->getOnceDetectTime() << "ms, " << "avg: " << lanedetector_ptr_->getAvgDetectTime() << " fps : " << 1000 / lanedetector_ptr_->getAvgDetectTime() << endl;
	cout << "#### Control ####" << endl;
	cout << "steering angle: " << lanedetector_ptr_->getRealSteerAngle() << endl;
	cout << "throttle: " << throttle_ << endl;
	cout << "#### Ros Param ####" << endl;
	cout << "gray_bin_thres: " << lanedetector_ptr_->getGrayBinThres() << endl;
	cout << "hsv_s_bin_thres: " << lanedetector_ptr_->getHsvSBinThres() << endl;
	cout << "detect_line_count: " << lanedetector_ptr_->getDetectLineCount() << endl;
	for(int i = 0; i < lanedetector_ptr_->getDetectLineCount(); i++)
		cout << "detect_y_offset_" << i+1 << ": " << lanedetector_ptr_->getDetectYOffset(i) << endl;
	cout << "left_detect_offset: " << lanedetector_ptr_->getLeftDetectOffset() << endl;
	cout << "right_detect_offset: " << lanedetector_ptr_->getRightDetectOffset() << endl;
	cout << "steer_max_angle: " << lanedetector_ptr_->getSteerMaxAngle() << endl;
	cout << "yaw_factor: " << lanedetector_ptr_->getYawFactor() * 100 << "% -> " << lanedetector_ptr_->getYawFactor() << endl;
	cout << "lateral_factor: " << lanedetector_ptr_->getLateralFactor() * 100 << "% -> " << lanedetector_ptr_->getLateralFactor() << endl;
	cout << "---------------------------------" << endl;
}
#endif
