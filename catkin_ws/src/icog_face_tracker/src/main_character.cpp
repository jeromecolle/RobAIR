/**
**  Simple ROS Node
**/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/String.h>
#include <icog_face_tracker/facebox.h>
#include <icog_face_tracker/faces.h>

using namespace std;
using namespace cv;

class main_character_node {
    private:
        ros::NodeHandle nh;

        CascadeClassifier face_cascade;
        ros::Publisher pub;
        bool showPreview = true;

    public:
        main_character_node(){
            face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml");
            nh.param("/tracker_node/show_preview", showPreview, true);
            pub = nh.advertise<icog_face_tracker::faces>("/faces", 5);

            ros::Subscriber sub = nh.subscribe("/usb_cam_node/image_raw", 1, &main_character_node::imageCB, this);
            
            ros::Rate r(10);// this node will run at 10hz
            while (ros::ok()) {
                ros::spinOnce();//each callback is called once to collect new data: laser + robot_moving
                // update();//processing of data
                r.sleep();//we wait if the processing (ie, callback+update) has taken less than 0.1s (ie, 10 hz)
            }
        }

        vector<Rect> detectFaces(Mat frame) {
            vector<Rect> faces;
            Mat bufferMat;
            cvtColor(frame, bufferMat, COLOR_BGR2GRAY);
            equalizeHist(bufferMat, bufferMat);
            face_cascade.detectMultiScale(bufferMat, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            return faces;
        }

        void imageCB(const sensor_msgs::ImageConstPtr& msg) {
            cv_bridge::CvImagePtr cvPtr;
            try {
                cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            vector<Rect> faces = detectFaces(cvPtr->image);
            icog_face_tracker::faces faces_msg;
            icog_face_tracker::facebox _facebox;
            faces_msg.image_width = cvPtr->image.cols;
            faces_msg.image_height = cvPtr->image.rows;
            int closest_face_index;
            int closest_face_area = 0;

            for (int i = 0; i < faces.size(); i++) {
                _facebox.top = faces[i].y;
                _facebox.left = faces[i].x;
                _facebox.width = faces[i].width;
                _facebox.height = faces[i].height;
                faces_msg.face_boxes.push_back(_facebox);
                int area  = faces[i].width * faces[i].height;
                if (area >= closest_face_area){
                    closest_face_area = area;
                    closest_face_index = i;
                }
            }

            if (showPreview){
                for (int i = 0; i < faces.size(); i++) {
                    if (i == closest_face_index){
                        rectangle(cvPtr->image, faces[i], CV_RGB(255, 0, 0), 4);
                        putText(cvPtr->image, "Closest face", Point(faces[i].x, faces[i].y - 5), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 0, 0));
                    }
                    else{
                        rectangle(cvPtr->image, faces[i], CV_RGB(100, 100, 255), 4);
                    }
                }
            }
            const char * a = showPreview ? "True" : "False";
            ROS_INFO(a);
            pub.publish(faces_msg);
            if (showPreview) {
                ROS_INFO("I will publish");
                imshow("Live Feed", cvPtr->image);
                waitKey(3);
            }
        }
};

int main(int argc, char* argv[])    
{
    ros::init(argc, argv, "main_character_node");

    main_character_node mc;

    ROS_INFO("Adjuster initialised");
    ros::spin();
}