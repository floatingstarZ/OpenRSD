eval_cfgs = dict(
    Data1_DOTA2=dict(
        class_names=['airport', 'baseball-diamond', 'basketball-court', 'bridge',
                     'container-crane', 'ground-track-field', 'harbor', 'helicopter', 'helipad',
                     'large-vehicle', 'plane', 'roundabout',
                     'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court'],
    ),
    Data1_DOTA1=dict(
        class_names=['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                     'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court'],
    ),
    Data2_DIOR_R=dict(
        class_names=['airplane', 'airport', 'baseballfield', 'basketballcourt',
                     'bridge', 'chimney', 'dam',
                     'Expressway-Service-area', 'Expressway-toll-station',
                     'golffield', 'groundtrackfield',
                     'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                     'tenniscourt', 'trainstation', 'vehicle', 'windmill'],
    ),
    Data3_FAIR1M=dict(
        class_names=['a220', 'a321', 'a330', 'a350', 'arj21',
                     'baseball_field', 'basketball_court', 'boeing737', 'boeing747', 'boeing777', 'boeing787',
                     'bridge', 'bus', 'c919', 'cargo_truck', 'dry_cargo_ship', 'dump_truck',
                     'engineering_ship', 'excavator', 'fishing_boat', 'football_field', 'intersection',
                     'liquid_cargo_ship', 'motorboat', 'other-airplane', 'other-ship', 'other-vehicle',
                     'passenger_ship', 'roundabout',
                     'small_car', 'tennis_court', 'tractor', 'trailer', 'truck_tractor', 'tugboat', 'van', 'warship'],
    ),
    Data4_HRRSD=dict(
        class_names=['airplane', 'baseball_diamond', 'basketball_court',
                       'bridge', 'crossroad', 'ground_track_field',
                       'harbor', 'parking_lot', 'ship', 'storage_tank',
                       't_junction', 'tennis_court', 'vehicle'],
    ),
    Data5_SpaceNet=dict(
        class_names=['building',]
    ),
    Data6_Xview=dict(
        class_names=['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Truck',
                     'Cargo_or_Container_Car', 'Cement_Mixer', 'Construction_Site',
                     'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
                     'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry', 'Fishing_Vessel',
                     'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_or_Bulldozer', 'Ground_Grader',
                     'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_or_Tent', 'Locomotive', 'Maritime_Vessel',
                     'Mobile_Crane', 'Motorboat', 'Oil_Tanker', 'Passenger_Car', 'Passenger_Vehicle',
                     'Passenger_or_Cargo_Plane', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle', 'Reach_Stacker',
                     'Sailboat', 'Scraper_or_Tractor', 'Shed', 'Shipping_Container', 'Shipping_container_lot',
                     'Small_Aircraft', 'Small_Car', 'Storage_Tank', 'Straddle_Carrier', 'Tank_car', 'Tower',
                     'Tower_crane', 'Trailer', 'Truck', 'Truck_Tractor', 'Truck_Tractor_with_Box_Trailer',
                     'Truck_Tractor_with_Flatbed_Trailer', 'Truck_Tractor_with_Liquid_Tank', 'Tugboat',
                     'Utility_Truck', 'Vehicle_Lot', 'Yacht'],
    ),
    Data7_HRSC2016=dict(
        class_names=['Arleigh_Burke', 'Austen', 'Car_carrier', 'CntShip', 'Container', 'Cruise',
                     'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical', 'Midway_class', 'Nimitz',
                     'OXo', 'Perry', 'Sanantonio', 'Tarawa', 'Ticonderoga',
                     'WhidbeyIsland', 'aircraft_carrier','lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht'],
    ),
    Data8_GLH_Bridge=dict(
        class_names=['bridge',],
    ),
    Data9_FMoW=dict(
        class_names=['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'archaeological_site',
                     'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site',
                     'dam', 'educational_institution', 'electric_substation', 'factory_or_powerplant',
                     'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad',
                     'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'nuclear_powerplant',
                     'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'port', 'race_track', 'railway_bridge',
                     'recreational_facility', 'road_bridge', 'runway', 'shipyard', 'shopping_mall',
                     'smokestack', 'solar_farm', 'space_facility', 'stadium'],
    ),
)

"""
all_classes = []
for info in eval_cfgs.values():
    all_classes.extend(info['class_names'])
all_classes = sorted(list(set([cls.lower() for cls in all_classes])))
找出里头完全相同的类别名称，然后进行合并，例如：baseball_field和baseballfield，输出一个字典，格式为：{合并前类别1：合并后类别}
类别列表：['a220', 'a321', 'a330', 'a350', 'aircraft_carrier', 'aircraft_hangar', 'airplane', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'archaeological_site', 'arj21', 'arleigh_burke', 'austen', 'barge', 'baseball-diamond', 'baseball_diamond', 'baseball_field', 'baseballfield', 'basketball-court', 'basketball_court', 'basketballcourt', 'boeing737', 'boeing747', 'boeing777', 'boeing787', 'border_checkpoint', 'bridge', 'building', 'burial_site', 'bus', 'c919', 'car_carrier', 'car_dealership', 'cargo_or_container_car', 'cargo_truck', 'cement_mixer', 'chimney', 'cntship', 'construction_site', 'container', 'container-crane', 'container_crane', 'container_ship', 'crane_truck', 'crossroad', 'cruise', 'dam', 'damaged_building', 'dry_cargo_ship', 'dump_truck', 'educational_institution', 'electric_substation', 'engineering_ship', 'engineering_vehicle', 'enterprise', 'excavator', 'expressway-service-area', 'expressway-toll-station', 'facility', 'factory_or_powerplant', 'ferry', 'fishing_boat', 'fishing_vessel', 'fixed-wing_aircraft', 'flat_car', 'football_field', 'fountain', 'front_loader_or_bulldozer', 'gas_station', 'golf_course', 'golffield', 'ground-track-field', 'ground_grader', 'ground_track_field', 'ground_transportation_station', 'groundtrackfield', 'harbor', 'haul_truck', 'helicopter', 'helipad', 'hovercraft', 'hut_or_tent', 'interchange', 'intersection', 'kuznetsov', 'lake_or_pond', 'large-vehicle', 'lighthouse', 'liquid_cargo_ship', 'locomotive', 'lute', 'maritime_vessel', 'medical', 'merchant_ship', 'midway_class', 'military_facility', 'mobile_crane', 'motorboat', 'nimitz', 'nuclear_powerplant', 'oil_or_gas_facility', 'oil_tanker', 'other-airplane', 'other-ship', 'other-vehicle', 'overpass', 'oxo', 'park', 'parking_lot', 'parking_lot_or_garage', 'passenger_car', 'passenger_or_cargo_plane', 'passenger_ship', 'passenger_vehicle', 'perry', 'pickup_truck', 'plane', 'port', 'pylon', 'race_track', 'railway_bridge', 'railway_vehicle', 'reach_stacker', 'recreational_facility', 'road_bridge', 'roundabout', 'runway', 'sailboat', 'sanantonio', 'scraper_or_tractor', 'shed', 'ship', 'shipping_container', 'shipping_container_lot', 'shipyard', 'shopping_mall', 'small-vehicle', 'small_aircraft', 'small_car', 'smokestack', 'soccer-ball-field', 'solar_farm', 'space_facility', 'stadium', 'storage-tank', 'storage_tank', 'storagetank', 'straddle_carrier', 'submarine', 'swimming-pool', 't_junction', 'tank_car', 'tarawa', 'tennis-court', 'tennis_court', 'tenniscourt', 'ticonderoga', 'tower', 'tower_crane', 'tractor', 'trailer', 'trainstation', 'truck', 'truck_tractor', 'truck_tractor_with_box_trailer', 'truck_tractor_with_flatbed_trailer', 'truck_tractor_with_liquid_tank', 'tugboat', 'utility_truck', 'van', 'vehicle', 'vehicle_lot', 'warcraft', 'warship', 'whidbeyisland', 'windmill', 'yacht']
{
    'baseball-diamond': 'baseball_diamond',
    'baseball_field': 'baseball_diamond',
    'baseballfield': 'baseball_diamond',
    'basketball-court': 'basketball_court',
    'basketballcourt': 'basketball_court',
    'golffield': 'golf_course',
    'ground-track-field': 'ground_track_field',
    'groundtrackfield': 'ground_track_field',
    'soccer-ball-field': 'football_field',
    'storage-tank': 'storage_tank',
    'storagetank': 'storage_tank',
    'tennis-court': 'tennis_court',
    'tenniscourt': 'tennis_court'
}

"""

same_mapping = {
    'baseball-diamond': 'baseball_diamond',
    'baseball_field': 'baseball_diamond',
    'baseballfield': 'baseball_diamond',
    'basketball-court': 'basketball_court',
    'basketballcourt': 'basketball_court',
    'golffield': 'golf_course',
    'ground-track-field': 'ground_track_field',
    'groundtrackfield': 'ground_track_field',
    'soccer-ball-field': 'football_field',
    'storage-tank': 'storage_tank',
    'storagetank': 'storage_tank',
    'tennis-court': 'tennis_court',
    'tenniscourt': 'tennis_court'
}
all_classes = []
for info in eval_cfgs.values():
    all_classes.extend(info['class_names'])
all_classes = sorted(list(set([cls.lower() for cls in all_classes])))




