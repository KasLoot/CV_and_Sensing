import math
import numpy as np
import matplotlib.pyplot as plt

#Set 1 distance measured by LiDAR
l1 = 17.3
l2 = 17.5
l3 = 7.64
theta1=math.acos((l2**2+l3**2-l1**2)/(2*l2*l3))*180/math.pi
theta2=math.acos((l1**2+l3**2-l2**2)/(2*l1*l3))*180/math.pi
theta3=math.acos((l1**2+l2**2-l3**2)/(2*l1*l2))*180/math.pi
theta4=90-theta1
theta5=90-theta2
theta6=180-theta4-theta5
r1=l3*math.sin(theta4*math.pi/180)/math.sin(theta6*math.pi/180)
r2=l3*math.sin(theta5*math.pi/180)/math.sin(theta6*math.pi/180)
d1=2*r1
d2=2*r2
print("Radius 1:",r1, "Diameter 1:", d1)
print("Radius 2:",r2, "Diameter 2:", d2)

#Set 2 distance measured by LiDAR
l4 = 17.5
l5 = 17.5
l6 = 7.38
theta7=math.acos((l5**2+l6**2-l4**2)/(2*l5*l6))*180/math.pi
theta8=math.acos((l4**2+l6**2-l5**2)/(2*l4*l6))*180/math.pi
theta9=math.acos((l4**2+l5**2-l6**2)/(2*l4*l5))*180/math.pi
theta10=90-theta7
theta11=90-theta8
theta12=180-theta10-theta11
r3=l6*math.sin(theta10*math.pi/180)/math.sin(theta12*math.pi/180)
r4=l6*math.sin(theta11*math.pi/180)/math.sin(theta12*math.pi/180)
d3=2*r3
d4=2*r4
print("Radius 3:",r3, "Diameter 3:", d3)
print("Radius 4:",r4, "Diameter 4:", d4)

R = (r1+r2+r3+r4)/4
T = 477.7      # rotation cycle time from Task 3c
latitudes_deg = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


omega = 2 * np.pi / T   # rad/s
print("Angular velocity =", omega, "rad/s\n")

#Linear velo along latitude
print("Latitude (deg) | cos(theta) | r(theta) (m) | v(theta) (m/s)")
print("-------------------------------------------------------------")
for lat in latitudes_deg:
    theta = np.radians(lat)
    r_theta = R * np.cos(theta)      # distance to rotation axis
    v_theta = omega * r_theta        # linear velocity
    print(f"{lat:>6}°       |  {np.cos(theta):.4f}     |   {r_theta:.4f}     |   {v_theta:.6f}")



# Plot graph
latitudes_plot = np.linspace(-90, 90, 180) 

# Calculate velocities for all latitudes
theta_plot = np.radians(latitudes_plot)
r_plot = R * np.cos(theta_plot)
v_plot = omega * r_plot

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(latitudes_plot, v_plot, 'b-', linewidth=2)

# Add grid
plt.grid(True, alpha=0.3)

# Labels and title
plt.xlabel('Latitude (degrees)', fontsize=12)
plt.ylabel('Linear Velocity (m/s)', fontsize=12)
plt.title('Linear Velocity along Latitude', fontsize=14)


# Set x-axis limits and ticks
plt.xlim(-90, 90)
plt.xticks(np.arange(-90, 91, 15))

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


print(f"Maximum velocity (at equator, 0°): {omega * R:.6f} m/s")
print(f"Velocity at ±45°: {omega * R * np.cos(np.radians(45)):.6f} m/s")
print(f"Velocity at poles (±90°): {omega * R * np.cos(np.radians(90)):.6f} m/s")