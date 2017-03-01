#ifndef SERVER_WIFIBOT_H
#define SERVER_WIFIBOT_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include "std_msgs/Float32.h"
#include "geometry_msgs/Point.h"

typedef struct _sposition
{
  double x;
  double y;
  double th;
} position;

class Wifibot
{
 public:
  Wifibot(const ros::NodeHandle& nh);
  ~Wifibot();

 private:
  void velocityCallback(const geometry_msgs::TwistConstPtr& vel);
  void change_odometryCallback(const geometry_msgs::Point::ConstPtr& o);
  void computeOdometry(double dleft, double dright);
  double getSpeedLinear(double speedLeft, double speedRight);
  double getSpeedAngular(double speedLeft, double speedRight);
  void update();

  ros::NodeHandle _nh;
  
  geometry_msgs::TransformStamped _odomTf;
  tf::TransformBroadcaster _odomBroadcast;

  wifibot::Driver *_pDriver;
  
  ros::Publisher _pubOdometry;
  ros::Subscriber _subSpeeds;

  //added by O. Aycard
  ros::Subscriber _change_odometry;
    
  ros::Time _timeCurrent;
  ros::Time _timeLast;

  std::string _frameBase;
  position _position;
  double _odometryLeftLast;
  double _odometryRightLast;
  double _entrax;
  bool _updated;
  double _speedLeft;
  double _speedRight;
};

#endif
