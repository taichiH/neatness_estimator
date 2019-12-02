import rospy
from neatness_estimator_msgs.srv import GetDifference, GetDifferenceResponse

rospy.init_node('difference_estimation_client')
client = rospy.ServiceProxy(
    '/estimation_module_interface_color_and_geometry/call', GetDifference)
res = client(task='two_scene')
print(res)
