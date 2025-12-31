from ctlib.dota import load_dota
import os
from ctlib.os import *
from pprint import pprint

from M_Tools.Base_Data_infos.data_infos import data_infos

all_names = []
for data_info in data_infos.values():
    all_names.extend(data_info['class_names'])
print(all_names)

"""
给你一个类别list，你需要输出一个python dict，将他们映射到正规的名称：
1. 所有的'-'都替换为'_'
2. 每个单词的头字母大写

"""
# 构建正规名称映射字典
normalized_dict = {}

# 处理每个类别
for category in all_names:
    # 替换 '-' 为 '_'
    normalized = category.replace('-', '_')
    # 将每个单词的首字母大写
    normalized = '_'.join([word.capitalize() for word in normalized.split('_')])
    normalized_dict[category] = normalized

# 输出结果
print(normalized_dict)
print(len(set(list(normalized_dict.keys()))), len(set(list(normalized_dict.values()))))
print(sorted(list(set(sorted(normalized_dict.values())))))
pprint(normalized_dict)

normalized_class_dict = {
    'Aircraft_Hangar': 'Aircraft_Hangar',
    'Arleigh_Burke': 'Arleigh_Burke',
    'Austen': 'Austen',
    'Barge': 'Barge',
    'Building': 'Building',
    'Bus': 'Bus',
    'Car_carrier': 'Car_Carrier',
    'Cargo_Truck': 'Cargo_Truck',
    'Cargo_or_Container_Car': 'Cargo_Or_Container_Car',
    'Cement_Mixer': 'Cement_Mixer',
    'CntShip': 'Cntship',
    'Construction_Site': 'Construction_Site',
    'Container': 'Container',
    'Container_Crane': 'Container_Crane',
    'Container_Ship': 'Container_Ship',
    'Crane_Truck': 'Crane_Truck',
    'Cruise': 'Cruise',
    'Damaged_Building': 'Damaged_Building',
    'Dump_Truck': 'Dump_Truck',
    'Engineering_Vehicle': 'Engineering_Vehicle',
    'Enterprise': 'Enterprise',
    'Excavator': 'Excavator',
    'Expressway-Service-area': 'Expressway_Service_Area',
    'Expressway-toll-station': 'Expressway_Toll_Station',
    'Facility': 'Facility',
    'Ferry': 'Ferry',
    'Fishing_Vessel': 'Fishing_Vessel',
    'Fixed-wing_Aircraft': 'Fixed_Wing_Aircraft',
    'Flat_Car': 'Flat_Car',
    'Front_loader_or_Bulldozer': 'Front_Loader_Or_Bulldozer',
    'Ground_Grader': 'Ground_Grader',
    'Haul_Truck': 'Haul_Truck',
    'Helicopter': 'Helicopter',
    'Helipad': 'Helipad',
    'Hovercraft': 'Hovercraft',
    'Hut_or_Tent': 'Hut_Or_Tent',
    'Kuznetsov': 'Kuznetsov',
    'Locomotive': 'Locomotive',
    'Maritime_Vessel': 'Maritime_Vessel',
    'Medical': 'Medical',
    'Midway_class': 'Midway_Class',
    'Mobile_Crane': 'Mobile_Crane',
    'Motorboat': 'Motorboat',
    'Nimitz': 'Nimitz',
    'OXo': 'Oxo',
    'Oil_Tanker': 'Oil_Tanker',
    'Passenger_Car': 'Passenger_Car',
    'Passenger_Vehicle': 'Passenger_Vehicle',
    'Passenger_or_Cargo_Plane': 'Passenger_Or_Cargo_Plane',
    'Perry': 'Perry',
    'Pickup_Truck': 'Pickup_Truck',
    'Pylon': 'Pylon',
    'Railway_Vehicle': 'Railway_Vehicle',
    'Reach_Stacker': 'Reach_Stacker',
    'Sailboat': 'Sailboat',
    'Sanantonio': 'Sanantonio',
    'Scraper_or_Tractor': 'Scraper_Or_Tractor',
    'Shed': 'Shed',
    'Shipping_Container': 'Shipping_Container',
    'Shipping_container_lot': 'Shipping_Container_Lot',
    'Small_Aircraft': 'Small_Aircraft',
    'Small_Car': 'Small_Vehicle',
    'Storage_Tank': 'Storage_Tank',
    'Straddle_Carrier': 'Straddle_Carrier',
    'Tank_car': 'Tank_Car',
    'Tarawa': 'Tarawa',
    'Ticonderoga': 'Ticonderoga',
    'Tower': 'Tower',
    'Tower_crane': 'Tower_Crane',
    'Trailer': 'Trailer',
    'Truck': 'Truck',
    'Truck_Tractor': 'Truck_Tractor',
    'Truck_Tractor_with_Box_Trailer': 'Truck_Tractor_With_Box_Trailer',
    'Truck_Tractor_with_Flatbed_Trailer': 'Truck_Tractor_With_Flatbed_Trailer',
    'Truck_Tractor_with_Liquid_Tank': 'Truck_Tractor_With_Liquid_Tank',
    'Tugboat': 'Tugboat',
    'Utility_Truck': 'Utility_Truck',
    'Vehicle_Lot': 'Vehicle_Lot',
    'WhidbeyIsland': 'Whidbeyisland',
    'Yacht': 'Yacht',
    'a220': 'A220',
    'a321': 'A321',
    'a330': 'A330',
    'a350': 'A350',
    'aircraft_carrier': 'Aircraft_Carrier',
    'airplane': 'Airplane',
    'airport': 'Airport',
    'airport_hangar': 'Airport_Hangar',
    'airport_terminal': 'Airport_Terminal',
    'amusement_park': 'Amusement_Park',
    'aoe': 'Aoe',
    'archaeological_site': 'Archaeological_Site',
    'arj21': 'Arj21',
    'arleigh_burke_dd': 'Arleigh_Burke_Dd',
    'asagiri_dd': 'Asagiri_Dd',
    'atago_dd': 'Atago_Dd',
    'austin_ll': 'Austin_Ll',
    'barge': 'Barge',
    'baseball-diamond': 'Baseball_Field',
    'baseball_diamond': 'Baseball_Field',
    'baseball_field': 'Baseball_Field',
    'baseballfield': 'Baseball_Field',
    'basketball-court': 'Basketball_Court',
    'basketball_court': 'Basketball_Court',
    'basketballcourt': 'Basketball_Court',
    'boeing737': 'Boeing737',
    'boeing747': 'Boeing747',
    'boeing777': 'Boeing777',
    'boeing787': 'Boeing787',
    'border_checkpoint': 'Border_Checkpoint',
    'bridge': 'Bridge',
    'building': 'Building',
    'burial_site': 'Burial_Site',
    'bus': 'Bus',
    'c919': 'C919',
    'car_dealership': 'Car_Dealership',
    'cargo': 'Cargo',
    'cargo_truck': 'Cargo_Truck',
    'chimney': 'Chimney',
    'commander': 'Commander',
    'construction_site': 'Construction_Site',
    'container-crane': 'Container_Crane',
    'container_ship': 'Container_Ship',
    'crossroad': 'Crossroad',
    'dam': 'Dam',
    'dock': 'Dock',
    'dry_cargo_ship': 'Dry_Cargo_Ship',
    'dump_truck': 'Dump_Truck',
    'educational_institution': 'Educational_Institution',
    'electric_substation': 'Electric_Substation',
    'engineering_ship': 'Engineering_Ship',
    'enterprise': 'Enterprise',
    'epf': 'Epf',
    'excavator': 'Excavator',
    'factory_or_powerplant': 'Factory_Or_Powerplant',
    'ferry': 'Ferry',
    'fishing_boat': 'Fishing_Boat',
    'fishing_vessel': 'Fishing_Vessel',
    'football_field': 'Football_Field',
    'fountain': 'Fountain',
    'gas_station': 'Gas_Station',
    'golf_course': 'Golf_Field',
    'golffield': 'Golf_Field',
    'ground-track-field': 'Ground_Track_Field',
    'ground_track_field': 'Ground_Track_Field',
    'ground_transportation_station': 'Ground_Transportation_Station',
    'groundtrackfield': 'Ground_Track_Field',
    'harbor': 'Harbor',
    'hatsuyuki_dd': 'Hatsuyuki_Dd',
    'helicopter': 'Helicopter',
    'helipad': 'Helipad',
    'hovercraft': 'Hovercraft',
    'hyuga_dd': 'Hyuga_Dd',
    'interchange': 'Interchange',
    'intersection': 'Intersection',
    'lake_or_pond': 'Lake_Or_Pond',
    'large-vehicle': 'Large_Vehicle',
    'lha_ll': 'Lha_Ll',
    'lighthouse': 'Lighthouse',
    'liquid_cargo_ship': 'Liquid_Cargo_Ship',
    'lsd_41_ll': 'Lsd_41_Ll',
    'lute': 'Lute',
    'masyuu_as': 'Masyuu_As',
    'medical_ship': 'Medical_Ship',
    'merchant_ship': 'Merchant_Ship',
    'midway': 'Midway',
    'military_facility': 'Military_Facility',
    'motorboat': 'Motorboat',
    'nimitz': 'Nimitz',
    'nuclear_powerplant': 'Nuclear_Powerplant',
    'oil_or_gas_facility': 'Oil_Or_Gas_Facility',
    'oil_tanker': 'Oil_Tanker',
    'osumi_ll': 'Osumi_Ll',
    'other-airplane': 'Other_Airplane',
    'other-ship': 'Other_Ship',
    'other-vehicle': 'Other_Vehicle',
    'other_aircraft_carrier': 'Other_Aircraft_Carrier',
    'other_auxiliary_ship': 'Other_Auxiliary_Ship',
    'other_destroyer': 'Other_Destroyer',
    'other_frigate': 'Other_Frigate',
    'other_landing': 'Other_Landing',
    'other_merchant': 'Other_Merchant',
    'other_ship': 'Other_Ship',
    'other_warship': 'Other_Warship',
    'overpass': 'Overpass',
    'park': 'Park',
    'parking_lot': 'Parking_Lot',
    'parking_lot_or_garage': 'Parking_Lot_Or_Garage',
    'passenger_ship': 'Passenger_Ship',
    'patrol': 'Patrol',
    'perry_ff': 'Perry_Ff',
    'plane': 'Plane',
    'port': 'Port',
    'race_track': 'Race_Track',
    'railway_bridge': 'Railway_Bridge',
    'recreational_facility': 'Recreational_Facility',
    'road_bridge': 'Road_Bridge',
    'roro': 'Roro',
    'roundabout': 'Roundabout',
    'runway': 'Runway',
    'sailboat': 'Sailboat',
    'sanantonio_as': 'Sanantonio_As',
    'ship': 'Ship',
    'shipyard': 'Shipyard',
    'shopping_mall': 'Shopping_Mall',
    'small-vehicle': 'Small_Vehicle',
    'small_car': 'Small_Vehicle',
    'smokestack': 'Smokestack',
    'soccer-ball-field': 'Soccer_Ball_Field',
    'solar_farm': 'Solar_Farm',
    'space_facility': 'Space_Facility',
    'stadium': 'Stadium',
    'storage-tank': 'Storage_Tank',
    'storage_tank': 'Storage_Tank',
    'storagetank': 'Storage_Tank',
    'submarine': 'Submarine',
    'swimming-pool': 'Swimming_Pool',
    't_junction': 'T_Junction',
    'tennis-court': 'Tennis_Court',
    'tennis_court': 'Tennis_Court',
    'tenniscourt': 'Tennis_Court',
    'test_ship': 'Test_Ship',
    'ticonderoga': 'Ticonderoga',
    'tractor': 'Tractor',
    'trailer': 'Trailer',
    'training_ship': 'Training_Ship',
    'trainstation': 'Trainstation',
    'truck_tractor': 'Truck_Tractor',
    'tugboat': 'Tugboat',
    'van': 'Van',
    'vehicle': 'Vehicle',
    'warcraft': 'Warcraft',
    'warship': 'Warship',
    'wasp_ll': 'Wasp_Ll',
    'windmill': 'Windmill',
    'yacht': 'Yacht',
    'yudao_ll': 'Yudao_Ll',
    'yudeng_ll': 'Yudeng_Ll',
    'yuting_ll': 'Yuting_Ll',
    'yuzhao_ll': 'Yuzhao_Ll'}
from copy import deepcopy
extend_normalized_class_dict = deepcopy(normalized_class_dict)
for k, v in normalized_class_dict.items():
    lower_k = k.lower()  # 全部小写
    upper_k = k.upper()  # 全部大写
    space_k = k.replace('_', ' ')  # 空格
    min_k = k.replace('_', '-')  # -
    cap_k = '_'.join([word.capitalize() for word in k.split('_')])  # 首字母大写
    low_k = k[:1].lower() + k[1:]  # 首字母小写

    extend_normalized_class_dict[lower_k] = v
    extend_normalized_class_dict[upper_k] = v
    extend_normalized_class_dict[space_k] = v
    extend_normalized_class_dict[min_k] = v
    extend_normalized_class_dict[cap_k] = v
    extend_normalized_class_dict[low_k] = v
    extend_normalized_class_dict[v] = v


out_pth = '/data/space2/huangziyue/normalized_class_dict.pkl'
pklsave(extend_normalized_class_dict, out_pth)


