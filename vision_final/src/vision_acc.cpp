#include <iostream>
#include <string>
#include <algorithm>
#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "vision_lib/vision_lib.hpp"


using namespace std;

boost::shared_ptr<vision> CF;
vector<box_size> last_car;
ros::Time ex_time_;
deque<double> speed_avg;
int last_car_id_=-2;
int succese_id;
double succese_val;

bool cmp(const box_size& box1, const box_size& box2) {
    return box1.dist < box2.dist;
}

double LPF(double cur_data, double prev_lpf)
{
    double alpha = 0.5;
    double lpf_data;
    lpf_data = alpha * prev_lpf + (1 - alpha) * cur_data;
	//if(prev_lpf != 0 && lpf_data > prev_lpf*1.2) return prev_lpf;
    return lpf_data;
}

// 이동 평균 필터(Moving Average Filter)를 구현하여
double movAvgFilter(std::deque<double>& x_n, double x_meas) {
    int n = x_n.size();
    for (int i = 0; i < n - 1; i++) {
        x_n[i] = x_n[i + 1];
    }
    x_n[n - 1] = x_meas;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x_n[i];
    }
    double x_avg = sum / n;
    return x_avg;
}

void get_speed(vector<box_size> &car_now, double dt){
    if (last_car.size()==0) return; // 처음 루프는 돌지않음
    for (int i = 0; i < car_now.size(); i++){
        for (int j = 0; j < last_car.size(); j++){
            if(car_now[i].tracking_id == last_car[j].tracking_id){  // 이전 루프의 car와 현재 루프의 car의  tracking id가 같다면 
                if (last_car_id_ == -2) last_car_id_ = car_now[i].tracking_id; // 초기화가 되어 있는 car id라면 id 새로 부여 처음 처음 !!
                double dx = car_now[i].dist - last_car[j].dist; // 이동거리 차이
                double dy = 0;
                // 내 차의 speed를 고려한 
                double cur_vel = sqrt(pow(dx/dt + 40,2)+pow(dy/dt + 40, 2));
                //double cur_vel = sqrt(pow(dx/dt,2)+pow(dy/dt, 2));
                double last_vel = last_car[j].vel;
                //현재 계산한 차량의 속도와 이전에 계산한 차량의 속도에 low pass filter를 적용해 현재 차량의 속도에 반영함
                car_now[i].vel = LPF(cur_vel , last_vel);
                if(car_now.size()==1){
                    if (last_car_id_ != car_now[i].tracking_id) { cout <<"another car" << endl; speed_avg.clear(); }
                    speed_avg.emplace_back(car_now[i].vel);
					if (speed_avg.size()>=20){ 
						speed_avg.pop_front();
					}
					car_now[i].vel = movAvgFilter(speed_avg, car_now[i].vel);
					last_car_id_ = car_now[i].tracking_id;
                }
				cout << "car_vel: "<< car_now[i].vel << endl;
				cout << "car_id: "<< car_now[i].tracking_id << endl;
                succese_id = car_now[i].tracking_id;
                succese_val = car_now[i].vel;
            }

        }
    }
}
void callback_acc(const sensor_msgs::Image::ConstPtr& msg, const yolov8_msgs::BoundingBoxes::ConstPtr &box_msg){

    double start = ros::Time::now().toSec();


// image callback ===========================================================

    cv_bridge::CvImagePtr cv_ptr;

    try 
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } 
    catch(cv_bridge::Exception& e)
    {
	    throw std::invalid_argument("CV_BRIDGE EXCEPTION");
    }
    
    cv::Mat frame = cv_ptr->image;

// BOX callback ===========================================================
    vector<box_size> car_now;
  	int size = box_msg->bounding_boxes.size();
	box_size box;
    for(int i=0; i<size; i++){
    	yolov8_msgs::BoundingBox bounding_box = box_msg->bounding_boxes[i];
		if (bounding_box.id == 2){ //if 차
			box.class_id = bounding_box.id;
			box.x_coord = bounding_box.xmin;
			box.y_coord = bounding_box.ymin;
			box.w_coord = bounding_box.width;
			box.h_coord = bounding_box.height;
            box.dist = bounding_box.dist;

      	if(bounding_box.width*bounding_box.height<921600){
			car_now.emplace_back(box);
		}
   		}
    }
    sort(car_now.begin(),car_now.end(),cmp);
    CF->CameraSORT(car_now);



    ros::Time cur_time = msg->header.stamp;
    ros::Duration dt_ = cur_time - ex_time_;
    double dt = dt_.toSec();
    get_speed(car_now, dt);
    ex_time_ = cur_time;


    box_size tmp;
    last_car.clear();
    for (int i = 0 ; i < car_now.size(); i++){

        int x = car_now[i].x_coord;
        int y = car_now[i].y_coord;
        int trackong_id_ = car_now[i].tracking_id;

        double dist = car_now[i].dist;
        double vel = car_now[i].vel;
        
        tmp.x_coord = x;
        tmp.y_coord = y;
        tmp.tracking_id = trackong_id_;
        tmp.dist = dist;
        tmp.vel = vel;

        last_car.emplace_back(tmp);
        // 트래킹 ID와 속도 정보를 이미지에 표시하는 코드
        stringstream id_;
        stringstream val_;
        id_ << succese_id;
        val_ << succese_val;
        cv::putText(frame, "tracking_id : " + id_.str(), cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "val_ : " + val_.str() + "km", cv::Point(x, y+car_now[i].h_coord + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + car_now[i].w_coord, y + car_now[i].h_coord), cv::Scalar(0, 255, 0), 2);
    }
    // 이미지를 화면에 표시
    cv::imshow("Tracking Result", frame);
    cv::waitKey(1);



}

int main(int argc, char**argv) {
    ros::init(argc, argv, "vision_acc_node");
    ros::NodeHandle nh;
    CF.reset(new vision());
	//callback
	// ros::Subscriber state_sub_ = nh.subscribe("gps_state", 1, &StateCallback);
	// ros::Subscriber sub_speed = nh.subscribe("course", 1, &SpeedCallback);

	message_filters::Subscriber <sensor_msgs::Image> sub_image(nh, "/kitti/camera_color_left/image_raw", 100);
	message_filters::Subscriber <yolov8_msgs::BoundingBoxes> sub_boxdepth(nh, "/monodepth2/box_dist", 1);

	typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::Image, yolov8_msgs::BoundingBoxes> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_image, sub_boxdepth);
	sync.registerCallback(boost::bind(&callback_acc, _1, _2));


    ros::Rate loop_rate(30);
    while (ros::ok()){
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
