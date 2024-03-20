#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>

// ros yolov8_msgs segment
#include <yolov8_msgs/BoundingBox.h>
#include <yolov8_msgs/BoundingBoxes.h>
#include <yolov8_msgs/MaskBox.h>
#include <yolov8_msgs/MaskBoxes.h>
#include <yolov8_msgs/CheckForSegmentationAction.h>
#include <yolov8_msgs/SegmentCount.h>

#include <ros/ros.h>
#include <iostream>

#include "car_struct.h"
#include "vision_lib/KalmanTracker.h"
#include "vision_lib/Hungarian.h"


struct box_size{
	int class_id;
	int tracking_id;
	int x_coord;
	int y_coord;
	int w_coord;
	int h_coord;
	double dist;
	double vel;
};

typedef struct 
{
	int id;
	Rect_<float> box;
}TrackingBox;


class vision{

private:

    //SORT
    int box_x_offset_;
    int box_y_offset_;
    int max_age;
    int min_hits;
    double iouThreshold;
    vector<KalmanTracker> trackers;

public:
	double GetIOU(const Rect_<float> bb_test, const Rect_<float> bb_gt);
	void CameraSORT(vector<box_size> &point);
    vision(){
        //SORT initialize
        max_age = 1;
        min_hits = 3;		
	}
    ~vision(){
	}



};

double vision::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void vision::CameraSORT(vector<box_size> &point)
{
	vector<TrackingBox> detData;
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	for (int i = 0 ; i< point.size(); i++){
		TrackingBox tb;
		tb.id = -1;
		float tpx = point[i].x_coord;
		float tpy = point[i].y_coord;
		float tpw = point[i].w_coord*box_x_offset_;
		float tph = point[i].h_coord*box_y_offset_;
		tb.box = Rect_<float>(tpx,tpy,tpw,tph);
		detData.emplace_back(tb);
	}

	// ===========================main loop==============================
		if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detData.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(detData[i].box);
				trackers.emplace_back(trk);
			}
			
			return;
		}

		// 3.1. get predicted locations from existing trackers.
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x>0 && pBox.y>0){
				predictedBoxes.emplace_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
			}
		}

		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		trkNum = predictedBoxes.size();
		detNum = detData.size();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);
			}
		}
    
    if(iouMatrix.size()>0){
  		// solve the assignment problem using hungarian algorithm.
	  	// the resulting assignment is [track(prediction) : detection], with len=preNum
      HungarianAlgorithm HungAlgo;
      HungAlgo.Solve(iouMatrix, assignment);
    }

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;
		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.emplace_back(cv::Point(i, assignment[i]));
		}

		// 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detData[detIdx].box); // %%%%%%% maybe I can get velocity using this part
			point[detIdx].tracking_id = trackers[trkIdx].m_id;

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detData[umd].box);
			trackers.emplace_back(tracker);
		}

   		// // get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			
			if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				frameTrackingResult.emplace_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}
		}
}