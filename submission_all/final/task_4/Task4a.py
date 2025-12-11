#Task4a Trial3: Mathematical Approach
import math
#l1, l2, l3 are measurement result by the LiDAR
l1 = 24.5
l2 = 24.3
l3 = 7.84
theta1=math.acos((l2**2+l3**2-l1**2)/(2*l2*l3))*180/math.pi #cosine fmla
theta2=math.acos((l1**2+l3**2-l2**2)/(2*l1*l3))*180/math.pi
theta3=math.acos((l1**2+l2**2-l3**2)/(2*l1*l2))*180/math.pi
theta4=90-theta1
theta5=90-theta2
theta6=180-theta4-theta5
r1=l3*math.sin(theta4*math.pi/180)/math.sin(theta6*math.pi/180) #sine fmla
r2=l3*math.sin(theta5*math.pi/180)/math.sin(theta6*math.pi/180)
d1=2*r1
d2=2*r2
print("Radius 1:",r1, "Diameter 1:", d1)
print("Radius 2:",r2, "Diameter 2:", d2)


