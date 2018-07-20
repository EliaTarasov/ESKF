#include <eskf/Node.hpp>

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "eskf");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");
	eskf::Node node(nh, pnh);
	ros::spin();
	return 0;
}
