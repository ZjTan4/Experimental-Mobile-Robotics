# Introduction
The goals of this lab are to:
* Gain experience using the laser of TurtleBot3
* Gain experience obtaining odometry of TurtleBot3
* Gain experience using publishers and subscribers in Python

This lab will use the ROS noetic environment and requires gazebo and turtlebot3 (see below for installation instructions)

### Installing lab package
```
cd ~/catkin_ws/src

git clone <repo-url>

cd ~/catkin_ws && catkin_make
source ~/catkin_ws/devel/setup.bash
```

### Running code
Launch world and turtlebot:
```
roslaunch dwa main.launch
```

Launch `main.py`:
```
rosrun dwa main.py
```

### Deliverables checklist
- [ ] `answers.md` with with screenshots from your screencast of your turtlebot for steps 1-3
- [ ] Screencast of steps 1-3 including a view of rviz, gazebo, and terminal
- [ ] Completed code in `src/main.py` with inline citations of resources used
- [ ] Code in `src/main.py` to reset turtlebot
- [ ] Code in `src/main.py` that prints the execution time of the node

# Potentially Helpful Resources (non-exhaustive)
* http://wiki.ros.org/rospy/Overview/Initialization%20and%20Shutdown
* https://github.com/pirobot/ros-by-example/blob/master/rbx_vol_1/rbx1_nav/src/transform_utils.py
* https://github.com/pirobot/ros-by-example/blob/master/rbx_vol_1/rbx1_nav/nodes/nav_square.py
* https://www.cs.columbia.edu/~allen/F19/NOTES/laser_scan_simulator.pdf
* https://answers.ros.org/question/198843/need-explanation-on-sensor_msgslaserscanmsg/
* https://github.com/celisun/ROS_Turtlebot_Maze_Solving_task/blob/master/scan_twist_center_control_.py#L70-L101

# Instructions
Follow the steps below and provide your answers in the included `answers.md` with screenshots from your screencast of your turtlebot for steps 1-3

Provide all your code for steps 1-3 in `src/main.py` as a node that initializes and shuts down, you will have to use the `/cmd_vel`, `/scan` (laser) and `/odom` (odometry) topics

Example usage of subscribing to `/odom` is provided below, the resources above provide examples of how to publish to `/cmd_vel` and how to subscribe to `/scan` topics

Please cite all parts of code you use from elsewhere inline with your code with a url including `http`

<hr>

*Please print to terminal your end positions on steps 1-3*

1. Go forward until you are 2m from the 2mx2m square and stop<br>
**What is your position from the square?**

2. Trace a path (Turn until you are aligned with the square horizontally, go forward 2m, turn until you are aligned vertically with the square, go forward 5m, turn until you are aligned with the cube horizontally, go forward 2m).<br>You should be 1m past the square.<br><br> **According to simulation coordinates, how close are you to where you wanted to be (1m past the square)?**
<br><br>
Hint - The path should look something like this:
<br>![image](https://user-images.githubusercontent.com/9467666/149637976-f5addfaf-0aef-4e5e-b692-89d683d85696.png)

3. Continue your path to continue to your original position using any combination of commands, turn until your turtlebot faces the wall again<br>
**How far does the laser range finder say you are from the wall? How far are you from the wall actually?**<br>

4. There are 4 cubes in the center of the world making a larger single 2m x2m square<br>
**What are the positions of each of the 4 cubes? How did you determine the positions?**<br><br>
Hint: https://gazebosim.org/tutorials/?tut=ros_comm#RetrievingSimulationWorldandObjectProperties

5. **How can you reset the turtlebot position and velocity without closing and restarting everything?**<br> Answer the question and implement a solution in `src/main.py`

6. **How can you keep track of the time your node has been running?**<br> Answer the question and implement a solution in `src/main.py` that prints node execution time on shut down

7. **What did you find interesting in the lab? Or what challenges did you face in the lab?**<br>
8. **How could the lab be more interesting or improved?**<br>
<hr>

### Code to obtain odometry readings
```
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import numpy as np

class OdometryReader():
    def __init__(self, topic):
        self.pose = {}
        self.trajectory = []
        self.topic = topic
        self.subscribe()

    def callback(self, msg):
        self.pose['x'] = msg.pose.pose.position.x
        self.pose['y'] = msg.pose.pose.position.y
        self.trajectory.append((self.pose['x'], self.pose['y']))
        (_, _, self.pose['theta']) = euler_from_quaternion([msg.pose.pose.orientation.x, 
                                                            msg.pose.pose.orientation.y, 
                                                            msg.pose.pose.orientation.z, 
                                                            msg.pose.pose.orientation.w])
    def subscribe(self):
        self.subscriber = rospy.Subscriber(self.topic, Odometry, self.callback)
        rospy.sleep(0.1)

    def unregister(self):
        np.save('trajectory',self.trajectory)
        self.subscriber.unregister()
        
odometry = OdometryReader('/odom') # register subscriber
odometry.unregister() # unregister subscriber when you are done
```

### Launching rviz to see laser
`roslaunch dwa rviz.launch`

### Turtlebot3 install
```
mkdir -p ~/catkin_ws/src/
cd ~/catkin_ws/src/
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git

cd ~/catkin_ws && catkin_make
source ~/catkin_ws/devel/setup.bash
```
