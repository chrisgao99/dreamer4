"""Minimal wire-compatible Waymo Scenario reader; no TensorFlow dependency.

Field numbers follow the official scenario.proto/map.proto. Unused fields are
ignored by protobuf. This is a reader only, not a replacement export schema.
"""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import struct


def scenario_class():
    f = descriptor_pb2.FileDescriptorProto(name='adaptation_scenario.proto', package='adaptation', syntax='proto2')
    def message(name, fields):
        m = f.message_type.add(name=name)
        for name, number, kind, repeated in fields:
            field = m.field.add(name=name, number=number, label=3 if repeated else 1)
            if isinstance(kind, str):
                field.type = 11
                field.type_name = '.adaptation.' + kind
            else:
                field.type = kind
    # double=1, int64=3, int32=5, string=9, float=2, bool=8
    message('Point', [('x',1,1,False),('y',2,1,False),('z',3,1,False)])
    message('Lane', [('type',2,5,False),('polyline',8,'Point',True)])
    message('RoadLine', [('type',1,5,False),('polyline',2,'Point',True)])
    message('RoadEdge', [('type',1,5,False),('polyline',2,'Point',True)])
    message('StopSign', [('lane',1,3,True),('position',2,'Point',False)])
    message('Polygon', [('polygon',1,'Point',True)])
    message('Feature', [('id',1,3,False),('lane',3,'Lane',False),('road_line',4,'RoadLine',False),
                        ('road_edge',5,'RoadEdge',False),('stop_sign',7,'StopSign',False),
                        ('crosswalk',8,'Polygon',False),('speed_bump',9,'Polygon',False),('driveway',10,'Polygon',False)])
    message('Light', [('lane',1,3,False),('state',2,5,False),('stop_point',3,'Point',False)])
    message('Dynamic', [('lane_states',1,'Light',True)])
    message('State', [('center_x',2,1,False),('center_y',3,1,False),('valid',11,8,False)])
    message('Track', [('id',1,5,False),('states',3,'State',True)])
    message('Scenario', [('scenario_id',5,9,False),('tracks',2,'Track',True),
                         ('dynamic_map_states',7,'Dynamic',True),('map_features',8,'Feature',True),
                         ('timestamps_seconds',1,1,True),('current_time_index',10,5,False)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('adaptation.Scenario'))


def records(path):
    cls = scenario_class()
    with open(path, 'rb') as stream:
        index = 0
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise IOError(f'Truncated TFRecord header: {path}:{index}')
            length = struct.unpack('<Q', header[:8])[0]
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise IOError(f'Truncated TFRecord: {path}:{index}')
            scene = cls()
            scene.ParseFromString(payload)
            if not scene.scenario_id:
                raise ValueError(f'Expected Scenario protobuf, not tf.Example: {path}:{index}')
            yield index, scene
            index += 1
