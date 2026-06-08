from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf


def extract_tfrecord(tfrecord_file, interested_keys):
    """Extract data from TFRecord file"""
    # Create a dataset from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    records_list = []
    # Take the first record from the dataset
    for raw_record in raw_dataset:
        # Parse the example using tf.train.Example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Create a dictionary to store the data
        data_dict = {}

        # Iterate through all features in the example
        for key, feature in example.features.feature.items():
            if key in interested_keys:
                # Determine the type of the feature and extract the value
                if feature.HasField('bytes_list'):
                    byte_values = feature.bytes_list.value
                    if len(byte_values) == 1:
                        # Single string case
                        value = byte_values[0].decode('utf-8')
                    else:
                        # List of strings case
                        value = [item.decode('utf-8') for item in byte_values]
                            
                elif feature.HasField('float_list'):
                    # Float feature
                    value = np.array(feature.float_list.value)
                elif feature.HasField('int64_list'):
                    # Integer feature
                    value = np.array(feature.int64_list.value)
                else:
                    # Unknown type
                    value = None

                # Add to the dictionary
                data_dict[key] = value
        
        records_list.append(data_dict)

    return records_list

def format_dict(data):
    if 'roadgraph_samples/xyz' in data:
        data['roadgraph_samples/xyz'] = data['roadgraph_samples/xyz'].reshape((30000, 3))
    
    if 'roadgraph_samples/dir' in data:
        data['roadgraph_samples/dir'] = data['roadgraph_samples/dir'].reshape((30000, 3))
    
    if 'traffic_light_state/past/state' in data:
        data['traffic_light_state/past/state'] = data['traffic_light_state/past/state'].reshape((10, 16))
        data['traffic_light_state/future/state'] = data['traffic_light_state/future/state'].reshape((80, 16))
        light_states = np.concatenate((data['traffic_light_state/past/state'], np.array([data['traffic_light_state/current/state']]), data['traffic_light_state/future/state']), axis=0)  # Shape: (91, 16)

    if 'traffic_light_state/past/x' in data:
        data['traffic_light_state/past/x'] = data['traffic_light_state/past/x'].reshape((10, 16))
        data['traffic_light_state/past/y'] = data['traffic_light_state/past/y'].reshape((10, 16))
        data['traffic_light_state/future/x'] = data['traffic_light_state/future/x'].reshape((80, 16))
        data['traffic_light_state/future/y'] = data['traffic_light_state/future/y'].reshape((80, 16))
        light_xy_pos = np.stack((data['traffic_light_state/current/x'], data['traffic_light_state/current/y']), axis=-1)  # Shape: (16, 2)

    
    if 'state/past/x' in data:
        data['state/past/x'] = data['state/past/x'].reshape((128, 10))
        data['state/past/y'] = data['state/past/y'].reshape((128, 10))
        data['state/future/x'] = data['state/future/x'].reshape((128, 80))
        data['state/future/y'] = data['state/future/y'].reshape((128, 80))
        
        # Combining arrays to form the trajectory (91 time steps)
        past_xy = np.stack((data['state/past/x'], data['state/past/y']), axis=-1)  # Shape: (128, 10, 2)
        current_xy = np.stack((data['state/current/x'], data['state/current/y']), axis=-1)[:, np.newaxis, :]  # Shape: (128, 1, 2)
        future_xy = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)  # Shape: (128, 80, 2)

        full_trajectory = np.concatenate((past_xy, current_xy, future_xy), axis=1)  # Shape: (128, 91, 2)
    
    if 'state/past/bbox_yaw' in data:
        data['state/past/bbox_yaw'] = data['state/past/bbox_yaw'].reshape((128, 10))
        data['state/current/bbox_yaw'] = data['state/current/bbox_yaw'].reshape((128, 1))
        data['state/future/bbox_yaw'] = data['state/future/bbox_yaw'].reshape((128, 80))

        obj_orientation = np.concatenate((data['state/past/bbox_yaw'], data['state/current/bbox_yaw'], data['state/future/bbox_yaw']), axis=1)  # Shape: (128, 91)        

    # Dictionary to store the type + index as keys and the trajectory as values
    type_dict = {
        0: 'Unset',
        1: 'Vehicle',
        2: 'Pedestrian',
        3: 'Cyclist',
        4: 'Other'
    }
    trajectory_dict = {}
    obj_orientation_dict = {}

    # Populate the dictionary
    for i in range(128):
        if data['state/id'][i] < 0:
            continue
        if data['state/type'][i] < 0:
            continue
        agent_id = int(data['state/id'][i])
        obj_type = type_dict[int(data['state/type'][i])]
        
        # Add to dictionary
        trajectory_dict[agent_id] = {"trajectory": full_trajectory[i],
                                     "heading": obj_orientation[i],
                                "type": obj_type}
        obj_orientation_dict[agent_id] = obj_orientation[i]
    
    result_trajectory_dict = {data["scenario/id"]: trajectory_dict}

    map_dict = {
        data["scenario/id"]:
        {
            "obj_orientation": obj_orientation_dict,
            "roadgraph_samples/xyz": data["roadgraph_samples/xyz"],
            "roadgraph_samples/dir": data["roadgraph_samples/dir"],
            "roadgraph_samples/type": data["roadgraph_samples/type"],
            # "traffic_light/xy": light_xy_pos,
            # "traffic_light/state": light_states
        }
    }

    if 'state/tracks_to_predict' in data:
        return result_trajectory_dict, map_dict, data['state/tracks_to_predict']
    else:
        return result_trajectory_dict, map_dict 



def run_filter_process(filename):
    keys_of_interest = [
        'roadgraph_samples/xyz', 
        'roadgraph_samples/dir', 
        'roadgraph_samples/type', 
        'state/past/x',
        'state/past/y',
        'state/past/bbox_yaw',
        'state/current/x',
        'state/current/y',
        'state/current/bbox_yaw',
        'state/future/x',
        'state/future/y',
        'state/future/bbox_yaw',
        'state/type',
        'state/id',
        'scenario/id',
    ]
    data = extract_tfrecord(filename, keys_of_interest)

    with ThreadPoolExecutor() as executor:
        outputs = list(executor.map(format_dict, data)) # list of tuples (trajectory_dict, obj_orientation)

    