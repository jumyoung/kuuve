#include "chrono"
#include "opencv2/opencv.hpp"
#include "segment/yolov8-seg.hpp"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>


// ros yolov8_msgs segment
#include <yolov8_msgs/BoundingBox.h>
#include <yolov8_msgs/BoundingBoxes.h>
#include <yolov8_msgs/MaskBox.h>
#include <yolov8_msgs/MaskBoxes.h>
#include <yolov8_msgs/CheckForSegmentationAction.h>
#include <yolov8_msgs/SegmentCount.h>


const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};


class RosNode
{
public:
    RosNode();
    ~RosNode(){};
    void callback(const sensor_msgs::ImageConstPtr &msg);
private:
    std::string pkg_path_, engine_file_path_;
    std::shared_ptr<YOLOv8_seg> yolov8_seg_;
    
    cv::Mat  img_res_, image_;
    cv::Size size_         = cv::Size{640, 640};
    int      topk_         = 100;  // top 100 class only use 
    int      seg_h_        = 160;   // output segmentation map size
    int      seg_w_        = 160;
    int      seg_channels_ = 32;  // channels is class id 
    float    score_thres_  = 0.25f; // class prob
    float    iou_thres_    = 0.65f;  // box prob matching
    std::vector<Object> objs_;

    ros::NodeHandle n_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_img_;
    ros::Publisher box_pub;
    ros::Publisher mask_pub;
    std::string topic_img_, topic_res_img_, weight_name_, box_result_;
    yolov8_msgs::BoundingBox boundingbox;
    yolov8_msgs::BoundingBoxes boundingbox_vector;
    yolov8_msgs::MaskBox maskbox;
    yolov8_msgs::MaskBoxes result_vector;
};

void RosNode::callback(const sensor_msgs::ImageConstPtr &msg)
{   
    try {
        cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

        // Check if the image is empty
        if (image.empty()) {
            ROS_ERROR("Received an empty image. Skipping processing.");
            return;
        }

        objs_.clear();
        yolov8_seg_->copy_from_Mat(image, size_); // input raw image -> translate image for model
        auto start = std::chrono::system_clock::now();
        yolov8_seg_->infer();   // inferance time check
        auto end = std::chrono::system_clock::now();
        yolov8_seg_->postprocess(objs_, score_thres_, iou_thres_, topk_, seg_channels_, seg_h_, seg_w_);
        yolov8_seg_->draw_objects(image, img_res_, objs_, CLASS_NAMES, COLORS, MASK_COLORS);
        
        // Assuming maskbox.boxMask is supposed to be a vector of vectors
        // cv::Mat maskMat(obj.boxMask.size(), CV_8UC1, obj.boxMask.data);

        for (auto& obj : objs_) {
            boundingbox.xmin = obj.rect.x;
            boundingbox.ymin = obj.rect.y;
            boundingbox.width = obj.rect.width;
            boundingbox.height = obj.rect.height; 
            boundingbox.probability = obj.prob;
            boundingbox.id = obj.label;
            maskbox.bounding_boxes.push_back(boundingbox);

            // Flatten the cv::Mat to std::vector<uint8_t>
            std::vector<uint8_t> flattenedMask;
            if (!obj.boxMask.empty()) {
                flattenedMask.assign(obj.boxMask.data, obj.boxMask.data + obj.boxMask.total());
            }
            // Append the flattened mask to boxMask
            maskbox.boxMask.insert(maskbox.boxMask.end(), flattenedMask.begin(), flattenedMask.end());
            result_vector.mask_boxes.push_back(maskbox);

            // cv::Mat mask = image.clone();
            // cv::Mat maskMat(obj.rect.height, obj.rect.width, CV_8UC1, maskbox.boxMask.data());
            // mask(obj.rect).setTo(0, maskMat);
            // cv::imshow("mask_img", mask);
            // cv::waitKey(1);
        }
        
        // Publish bounding box and mask
        result_vector.header = msg->header; 
        box_pub.publish(result_vector);
        // Clear vectors
        maskbox.bounding_boxes.clear();
        maskbox.boxMask.clear();
        result_vector.mask_boxes.clear();


        // Display the processed image using imshow
        cv::imshow("Processed Image", img_res_);
        cv::waitKey(1);  // Add a small delay to allow OpenCV to update the window

        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        cv::putText(img_res_, "fps: " + std::to_string(int(1000/tc)) , cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, 8);
        ROS_INFO("Segmentation cost: %2.4f ms", tc);

        // Check if the result image is empty before publishing
        if (img_res_.empty()) {
            ROS_ERROR("Processed image is empty. Skipping publishing.");
            return;
        }

        sensor_msgs::ImagePtr msg_img_new;
        msg_img_new = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_res_).toImageMsg();
        pub_img_.publish(msg_img_new);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("CV Bridge Exception: %s", e.what());
    } catch (const std::exception& e) {
        ROS_ERROR("Exception: %s", e.what());
    } catch (...) {
        ROS_ERROR("Unknown exception occurred during image processing.");
    }
}



RosNode::RosNode()
    {   
        cudaSetDevice(0);
        pkg_path_ = ros::package::getPath("yolov8_trt");
        n_.param<std::string>("topic_img", topic_img_, "/camera/color/image_raw");
        n_.param<std::string>("topic_res_img", topic_res_img_, "/detect/image_raw");
        n_.param<std::string>("box_result", box_result_, "/detect/box");
        n_.param<std::string>("weight_name",  weight_name_, "yolov8n-seg.engine");
        
        engine_file_path_ = pkg_path_ + "/weights/" + weight_name_;

        std::cout << "\n\033[1;32m--engine_file_path: " << engine_file_path_ << "\033[0m" << std::endl;
        std::cout << "\033[1;32m" << "--topic_img       : " << topic_img_  << "\033[0m" << std::endl;
        std::cout << "\033[1;32m--topic_res_img   : " << topic_res_img_    << "\n\033[0m" << std::endl;
        std::cout << "\033[1;32m--box_result_   : " << box_result_    << "\n\033[0m" << std::endl;

        yolov8_seg_.reset(new YOLOv8_seg(engine_file_path_));
        yolov8_seg_->make_pipe(true);

        pub_img_ = n_.advertise<sensor_msgs::Image>(topic_res_img_, 10);
        box_pub = n_.advertise<yolov8_msgs::MaskBoxes>(box_result_, 10);
        sub_img_ = n_.subscribe(topic_img_, 10, &RosNode::callback, this);
    };

int main(int argc, char** argv)
{
    ros::init(argc, argv, "seg_node");
    ros::NodeHandle n;
    auto seg_node = std::make_shared<RosNode>();
    ros::spin();
    return 0;
}
