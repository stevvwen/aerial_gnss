import math
import requests
import numpy as np
import numpy as np
from numba import jit

# Precompute sensor parameters globally if they don't change:
sensor_width = 36.0
# Image and camera parameters

# Earth's approximate conversions
feet_per_degree_lat = 364000  # Approximate feet per degree of latitude


# Compute positional estimate
image_width = 1920
image_height = 1080

focal_length=  20.3




@jit(nopython=True, fastmath=True)
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius (mean radius in kilometers)
    R = 6371.0

    # Calculate and return distance
    return R * c

def positional_estimate_multi(altitude,drone_angle,camera_angle,heading,lat,lon,pixel_x,pixel_y, true_lat,true_lon):

    camera_angle = abs(90 + camera_angle+ drone_angle)

    feet_per_degree_lon = feet_per_degree_lat * np.cos(np.radians(lat))  # Adjusted for latitude

    sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

    # Compute FoV (Field of View) for horizontal and vertical axes
    fov_x = 2 * np.arctan((sensor_width / 2) / focal_length)
    fov_y = 2 * np.arctan((sensor_height / 2) / focal_length)

    # Compute angular displacement from center
    delta_x = pixel_x - (image_width / 2)  # Left/right shift
    delta_y = pixel_y - (image_height / 2)  # Up/down shift


    theta_x = np.arctan((2*delta_x/image_width)*np.tan(fov_x/2))# Horizontal angle shift
    theta_y = np.arctan((2*delta_y/image_height)*np.tan(fov_y/2))# Vertical angle shift


    # Compute new ground distance
    D = altitude * np.tan(np.radians(camera_angle + theta_y))

    # Compute lateral displacement due to theta_x
    Dp= np.sqrt(D**2 + altitude**2)     
    Dx= Dp* np.tan(theta_x) # Lateral distance

    Dr= np.sqrt((np.sin(theta_x)*altitude)**2 + D**2 )/np.cos(theta_x) # New ground distance
    
    beta= np.arcsin(Dx/Dr) if Dr!= 0 else 0 # Lateral angle shift 

    # Compute new offsets including lateral shift
    delta_north_object = Dr * np.cos(np.radians(heading)+ beta)
    delta_east_object = Dr * np.sin(np.radians(heading)+ beta)


    # Convert ground displacement from feet to degrees.
    delta_lat_object = delta_north_object / feet_per_degree_lat
    delta_lon_object = delta_east_object / feet_per_degree_lon

    # Calculate final GPS coordinates.
    object_lat_final = lat + delta_lat_object
    object_lon_final = lon + delta_lon_object

    #print("Object GPS Coordinates:", object_lat_final, object_lon_final)
    errors= np.array([haversine(object_lat_final, object_lon_final, true_lat[i], true_lon[i]) for i in range(len(true_lat))])
    return np.min(errors), object_lon_final, object_lat_final



@jit(nopython=True, fastmath=True)
def positional_estimate(altitude,drone_angle,camera_angle,heading,lat,lon,pixel_x,pixel_y, true_lat,true_lon):

    camera_angle = abs(90 + camera_angle+ drone_angle)

    feet_per_degree_lon = feet_per_degree_lat * np.cos(np.radians(lat))  # Adjusted for latitude

    sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

    # Compute FoV (Field of View) for horizontal and vertical axes
    fov_x = 2 * np.arctan((sensor_width / 2) / focal_length)
    fov_y = 2 * np.arctan((sensor_height / 2) / focal_length)

    # Compute angular displacement from center
    delta_x = pixel_x - (image_width / 2)  # Left/right shift
    delta_y = pixel_y - (image_height / 2)  # Up/down shift


    theta_x = np.arctan((2*delta_x/image_width)*np.tan(fov_x/2))# Horizontal angle shift
    theta_y = np.arctan((2*delta_y/image_height)*np.tan(fov_y/2))# Vertical angle shift


    # Compute new ground distance
    D = altitude * np.tan(np.radians(camera_angle + theta_y))

    # Compute lateral displacement due to theta_x
    Dp= np.sqrt(D**2 + altitude**2)     
    Dx= Dp* np.tan(theta_x) # Lateral distance

    Dr= np.sqrt((np.sin(theta_x)*altitude)**2 + D**2 )/np.cos(theta_x) # New ground distance
    
    beta= np.arcsin(Dx/Dr) if Dr!= 0 else 0 # Lateral angle shift 

    # Compute new offsets including lateral shift
    delta_north_object = Dr * np.cos(np.radians(heading)+ beta)
    delta_east_object = Dr * np.sin(np.radians(heading)+ beta)


    # Convert ground displacement from feet to degrees.
    delta_lat_object = delta_north_object / feet_per_degree_lat
    delta_lon_object = delta_east_object / feet_per_degree_lon

    # Calculate final GPS coordinates.
    object_lat_final = lat + delta_lat_object
    object_lon_final = lon + delta_lon_object

    error = haversine(object_lat_final, object_lon_final, true_lat, true_lon)
    return error, object_lon_final, object_lat_final

