classes = {
    'Data1_DOTA2': ['airport', 'baseball-diamond', 'basketball-court', 'bridge',
                    'container-crane', 'ground-track-field', 'harbor',
                    'helicopter', 'helipad', 'large-vehicle', 'plane', 'roundabout',
                    'ship', 'small-vehicle', 'soccer-ball-field', 'storage-tank',
                    'swimming-pool', 'tennis-court'],
    'Data2_DIOR': ['golffield', 'vehicle', 'Expressway-toll-station',
                   'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
                   'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
                   'basketballcourt', 'Expressway-Service-area', 'stadium',
                   'airport', 'baseballfield', 'bridge', 'windmill', 'overpass'],
    'Data3_FAIR_1M':['A220', 'A321', 'A330', 'A350', 'ARJ21',
                      'Baseball_Field', 'Basketball_Court', 'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
                      'Bridge', 'Bus', 'C919', 'Cargo_Truck', 'Dry_Cargo_Ship', 'Dump_Truck', 'Engineering_Ship', 'Excavator',
                      'Fishing_Boat', 'Football_Field', 'Intersection', 'Liquid_Cargo_Ship', 'Motorboat', 'Passenger_Ship',
                      'Roundabout', 'Small_Car', 'Tennis_Court', 'Tractor', 'Trailer', 'Truck_Tractor', 'Tugboat',
                      'Van', 'Warship', 'other-airplane', 'other-ship', 'other-vehicle'],
    'Data4_HRRSD': ['Airplane', 'Storage_Tank', 'Bridge', 'Ground_Track_Field', 'Basketball_Court',
                    'Tennis_Court', 'Ship', 'Baseball_Diamond',
                    'T_Junction', 'Crossroad', 'Parking_Lot', 'Harbor', 'Vehicle'],
    'Data5_SpaceNet': ['building'],
    'Data6_Xview': ['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Container_Car',
                    'Cargo_Truck', 'Cement_Mixer', 'Construction_Site',
                    'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
                    'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
                    'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_Bulldozer',
                    'Ground_Grader', 'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_Tent', 'Locomotive',
                    'Maritime_Vessel', 'Mobile_Crane', 'Motorboat', 'Oil_Tanker', 'Passenger_Cargo_Plane',
                    'Passenger_Car', 'Passenger_Vehicle', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle',
                    'Reach_Stacker', 'Sailboat', 'Scraper_Tractor', 'Shed', 'Shipping_Container',
                    'Shipping_container_lot', 'Small_Aircraft', 'Small_Car', 'Storage_Tank',
                    'Straddle_Carrier', 'Tank_car', 'Tower', 'Tower_crane', 'Trailer', 'Truck',
                    'Truck_Tractor', 'Truck_Tractor_w__Box_Trailer', 'Truck_Tractor_w__Flatbed_Trailer',
                    'Truck_Tractor_w__Liquid_Tank', 'Tugboat', 'Utility_Truck', 'Vehicle_Lot', 'Yacht'],
    'Data7_HRSC2016': ['Arleigh_Burke', 'Austen',
                       'Car_carrier', 'CntShip', 'Container', 'Cruise',
                       'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
                       'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
                       'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
                       'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht'],
    'Data8_GLH_Bridge': ['bridge'],
    #
    # import os
    # from ctlib.dota import *
    # all_names = []
    # root = '/data/space2/huangziyue/FMoW'
    # for i in range(6):
    #     ann_dir = f'{root}/FMoW_Part{i}/annotations'
    #     a, names = load_dota(ann_dir)
    #     all_names.append(names)
    # all_names = sorted(list(set(all_names)))
    'Data9_FMoW': ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture',
                   'archaeological_site', 'barn', 'border_checkpoint', 'burial_site',
                   'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble',
                   'educational_institution', 'electric_substation', 'factory_or_powerplant',
                   'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course',
                   'ground_transportation_station', 'helipad', 'hospital', 'impoverished_settlement',
                   'interchange', 'lake_or_pond', 'lighthouse', 'military_facility',
                   'multi-unit_residential', 'nuclear_powerplant', 'office_building',
                   'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship',
                   'police_station', 'port', 'prison', 'race_track', 'railway_bridge',
                   'recreational_facility', 'road_bridge', 'runway', 'shipyard', 'shopping_mall',
                   'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium'],
    #####################################################################
    'Data10_OIRDS': ["car", "pick-up", "truck", "van", "unknown"],
    'Data11_VEDAI': ["Boat", "Camping_Car", "Car", "Others", "Pickup", "Plane", "Tractor", "Truck", "Vans",
                     "Small_land_Vehicles", "Large_Land_Vehicles"],
    'Data12_RESISC45': ["airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral",
                        "church", "circular farmland", "cloud", "commercial area", "dense residential", "desert",
                        "forest",
                        "freeway", "golf course", "ground track field", "harbor", "industrial area", "intersection",
                        "island",
                        "lake", "meadow", "medium residential", "mobilehome park", "mountain", "overpass", "palace",
                        "parking lot", "railway", "railway station", "rectangular farmland", "river", "roundabout",
                        "runway",
                        "sea ice", "ship", "snowberg", "sparse residential", "stadium", "storage tank", "tennis court",
                        "terrace", "thermal power station", "wetland"],
    'Data13_WHU-RS19': [
        "airport", "bridge", "river", "forest", "meadow", "pond", "parking", "port", "viaduct",
        "residential_area", "industrial_area", "commercial_area", "beach", "desert", "farmland",
        "football_field", "mountain", "park", "railway_station"
    ],
    'Data14_UC-Merced"': [
        "agricultural", "airplane", "baseball_diamond", "beach", "buildings", "chaparral", "denseresidential",
        "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
        "overpass", "parking_lot", "river", "runway", "sparseresidential", "storage_tank", "tenniscourt"
    ]
}

"""
Give you a list of scenes, objects, etc. contained in remote sensing images that can be classified. 
First, you need to filter out instances that have well-defined boundaries 
(deserts, for example, have no clear boundaries). 
Then, you need to remove the possible fine-grained categories and 
convert them to the parent classes they belong to. 
Finally, output these parent classes as a python list. 
The list is: 
classes = {
    'Data1_DOTA2': ['airplane', 'airport', 'baseballfield', 'basketballcourt',
                    'bridge', 'chimney', 'dam',
                    'Expressway-Service-area', 'Expressway-toll-station',
                    'golffield', 'groundtrackfield',
                    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                    'tenniscourt', 'trainstation', 'vehicle', 'windmill'],
    'Data2_DIOR': ['golffield', 'vehicle', 'Expressway-toll-station',
                   'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
                   'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
                   'basketballcourt', 'Expressway-Service-area', 'stadium',
                   'airport', 'baseballfield', 'bridge', 'windmill', 'overpass'],
    'Data3_FAIR-1M': ['A220', 'A321', 'A330', 'A350', 'ARJ21',
                      'Baseball', 'Basketball', 'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
                      'Bridge', 'Bus', 'C919', 'Cargo', 'Dry', 'Dump', 'Engineering', 'Excavator',
                      'Fishing', 'Football', 'Intersection', 'Liquid', 'Motorboat', 'Passenger',
                      'Roundabout', 'Small', 'Tennis', 'Tractor', 'Trailer', 'Truck', 'Tugboat',
                      'Van', 'Warship', 'other-airplane', 'other-ship', 'other-vehicle'],
    'Data4_HRRSD': ['airplane', 'storage', 'bridge', 'ground', 'basketball',
                    'tennis', 'ship', 'baseball',
                    'T', 'crossroad', 'parking', 'harbor', 'vehicle'],
    'Data5_SpaceNet': ['building'],
    'Data6_Xview': ['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Container_Car',
                    'Cargo_Truck', 'Cement_Mixer', 'Construction_Site',
                    'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
                    'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
                    'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_Bulldozer',
                    'Ground_Grader', 'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_Tent', 'Locomotive',
                    'Maritime_Vessel', 'Mobile_Crane', 'Motorboat', 'Oil_Tanker', 'Passenger_Cargo_Plane',
                    'Passenger_Car', 'Passenger_Vehicle', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle',
                    'Reach_Stacker', 'Sailboat', 'Scraper_Tractor', 'Shed', 'Shipping_Container',
                    'Shipping_container_lot', 'Small_Aircraft', 'Small_Car', 'Storage_Tank',
                    'Straddle_Carrier', 'Tank_car', 'Tower', 'Tower_crane', 'Trailer', 'Truck',
                    'Truck_Tractor', 'Truck_Tractor_w__Box_Trailer', 'Truck_Tractor_w__Flatbed_Trailer',
                    'Truck_Tractor_w__Liquid_Tank', 'Tugboat', 'Utility_Truck', 'Vehicle_Lot', 'Yacht'],
    'Data7_HRSC2016': ['Arleigh_Burke', 'Austen',
                       'Car_carrier', 'CntShip', 'Container', 'Cruise',
                       'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
                       'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
                       'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
                       'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht'],
    'Data8_GLH_Bridge': ['bridge'],
    'Data9_FMoW': ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
                   "archaeological_site",
                   "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field",
                   "dam",
                   "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant",
                   "fire_station", "flooded_road", "fountain", "gas_station", "golf_course",
                   "ground_transportation_station",
                   "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse",
                   "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building",
                   "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port",
                   "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway",
                   "shipyard",
                   "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium",
                   "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening",
                   "waste_disposal",
                   "water_treatment_facility", "wind_farm", "zoo"],
    #####################################################################
    'Data10_OIRDS': ["car", "pick-up", "truck", "van", "unknown"],
    'Data11_VEDAI': ["Boat", "Camping_Car", "Car", "Others", "Pickup", "Plane", "Tractor", "Truck", "Vans",
                     "Small_land_Vehicles", "Large_Land_Vehicles"],
    'Data12_RESISC45': ["airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral",
                        "church", "circular farmland", "cloud", "commercial area", "dense residential", "desert",
                        "forest",
                        "freeway", "golf course", "ground track field", "harbor", "industrial area", "intersection",
                        "island",
                        "lake", "meadow", "medium residential", "mobilehome park", "mountain", "overpass", "palace",
                        "parking lot", "railway", "railway station", "rectangular farmland", "river", "roundabout",
                        "runway",
                        "sea ice", "ship", "snowberg", "sparse residential", "stadium", "storage tank", "tennis court",
                        "terrace", "thermal power station", "wetland"],
    'Data13_WHU-RS19': [
        "airport", "bridge", "river", "forest", "meadow", "pond", "parking", "port", "viaduct",
        "residential_area", "industrial_area", "commercial_area", "beach", "desert", "farmland",
        "football_field", "mountain", "park", "railway_station"
    ],
    'Data14_UC-Merced"': [
        "agricultural", "airplane", "baseball_diamond", "beach", "buildings", "chaparral", "denseresidential",
        "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
        "overpass", "parking_lot", "river", "runway", "sparseresidential", "storage_tank", "tenniscourt"
    ]
}

Output: 
parent_classes = [
    'airplane', 'airport', 'athletic_field', 'vehicle', 'bridge', 'industrial_facility',
    'dam', 'expressway_service_area', 'expressway_toll_station', 'harbor', 'stadium',
    'storage_tank', 'tennis_court', 'train_station', 'windmill', 'ship', 'building',
    'ground_transportation_station', 'runway', 'parking_lot', 'military_facility', 'amusement_park',
    'hospital', 'power_plant', 'flooded_area', 'port', 'nuclear_facility', 'aquaculture', 'recreational_facility',
    'residential_area', 'commercial_area', 'forest', 'farmland', 'river', 'lake', 'mountain', 'wetland'
]
之后进行了一些筛选：
The following is a list of objects and scenes in some remote sensing images. 
Summarize them into a parent class list and output them in python format. 
classes = [
    'airplane', 'airport', 'athletic_field', 'vehicle', 'bridge', 'industrial_facility',
    'dam', 'expressway_service_area', 'expressway_toll_station', 'harbor', 'stadium',
    'storage_tank', 'tennis_court', 'train_station', 'windmill', 'ship', 'building',
    'ground_transportation_station', 'runway', 'parking_lot', 'military_facility', 'amusement_park',
    'hospital', 'power_plant', 'flooded_area', 'port', 'nuclear_facility', 'aquaculture', 'recreational_facility',
    'residential_area', 'commercial_area', 'forest', 'farmland', 'river', 'lake', 'mountain', 'wetland'
]
Outputs: 
parent_classes = [
    'aircraft',                # 'airplane', 'runway'
    'transportation_hub',      # 'airport', 'train_station', 'ground_transportation_station'
    'athletic_facility',       # 'athletic_field', 'stadium', 'tennis_court'
    'vehicle',                 # 'vehicle'
    'infrastructure',          # 'bridge', 'expressway_service_area', 'expressway_toll_station', 'parking_lot'
    'industrial_facility',     # 'industrial_facility', 'storage_tank', 'power_plant', 'nuclear_facility', 'windmill'
    'water_structure',         # 'dam', 'harbor', 'port'
    'maritime',                # 'ship'
    'building',                # 'building', 'residential_area', 'commercial_area', 'hospital'
    'military_facility',       # 'military_facility'
    'recreational_area',       # 'amusement_park', 'recreational_facility'
    'agriculture',             # 'aquaculture', 'farmland'
    'natural_environment',     # 'forest', 'river', 'lake', 'mountain', 'wetland'
    'disaster_area',           # 'flooded_area'
]

"""

parent_classes = [
    'aircraft',
    'transportation_hub',
    'athletic_facility',
    'vehicle',
    'infrastructure',
    'industrial_facility',
    'water_structure',
    'maritime',
    'building',
    'military_facility',
    'recreational_area',
    'agriculture',
    'natural_environment',
    'disaster_area',
]

"""
对parent list进行扩充（重复多次）

Gives you a list of parent classes (Coarse classes) and a list of categories to judge, 
including remote sensing objects or scenes. 
Firstly, complete the category names, as they are objects in the remote sensing image.
Then, outputs elements that do not belong to the parent list, 
i.e. they cannot be classified into any parent element at all. Output as python lists. 
Then, summarize the non-belong elements into parent classes, and merge into the given parent classes list, output as python list. 
Finally, given the classification and interpretation of each element mapping to the parent class (This part in chinese).
parent_classes = [
    'aircraft',
    'transportation_hub',
    'vehicle',
    'infrastructure',
    'industrial_facility',
    'water_structure',
    'maritime',
    'building',
    'military_facility',
    'recreational_area',
    'agriculture',
    'natural_environment',
    'disaster_area',
    'athletic_field',
    'industrial_structure',
    'sports_facility',
    'energy_facility',
    'storage_structure'
]

List of categories to be judged: 
[
        "agricultural", "airplane", "baseball_diamond", "beach", "buildings", "chaparral", "denseresidential",
        "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
        "overpass", "parking_lot", "river", "runway", "sparseresidential", "storage_tank", "tenniscourt"
    ]
    

#### --- 获得这个列表，：
parent_classes_v2 = [
    'aircraft',
    'transportation_hub',
    'vehicle',
    'infrastructure',
    'industrial_facility',
    'water_structure',
    'maritime',
    'building',
    'military_facility',
    'recreational_area',
    'agriculture',
    'natural_environment',
    'disaster_area',
    'athletic_field',
    'industrial_structure',
    'sports_facility',
    'energy_facility',
    'storage_structure'
]

"""


"""
首先，解释这个父类列表的每一个元素，然后列举其中可能包含的子类。其次，判断这个父类列表有没有可能进一步精炼，给出理由。
# 之后手工去掉了一些类别，以及合并
parent_classes_v2 = [
    'aircraft', # 航空器
    'vehicle', # 车辆
    'ship', # 船只
    'transportation_hub', # 交通枢纽
    'infrastructure', # 基础设施
    'industrial_facility_and_structure', # 工业设施和结构，包括了存储罐
    'water_structure', # 水利结构
    'maritime', # 海事（专门将ship去除了）
    'building', 
    'military_facility', # 军事设施
    'recreational_area',# 娱乐区
    'sports_facility',  # 体育设施
    'energy_facility', # 能源设施
]
"""
parent_classes_v2 = [
    'aircraft', # 航空器
    'vehicle', # 车辆
    'ship', # 船只
    'transportation_hub', # 交通枢纽
    'infrastructure', # 基础设施
    'industrial_facility_and_structure', # 工业设施和结构，包括了存储罐
    'water_structure', # 水利结构
    'maritime', # 海事（专门将ship去除了）
    'building',
    'military_facility', # 军事设施
    'recreational_area',# 娱乐区
    'sports_facility',  # 体育设施
    'energy_facility', # 能源设施
]

"""
Gives you a list of parent classes (Coarse classes) and a list of categories to judge, 
including remote sensing objects or scenes. 
Map a given class to its parent class and output the mapping as a python dict. 
If it is impossible to classify, give reasons. 

parent classes: 
[
    'aircraft', 
    'vehicle', 
    'ship', 
    'transportation_hub', 
    'infrastructure', 
    'industrial_facility_and_structure', 
    'water_structure', 
    'maritime', 
    'building', 
    'military_facility', 
    'recreational_area',
    'sports_facility',  
    'energy_facility',
]

List of categories to be judged: 
['airport', 'baseball-diamond', 'basketball-court', 'bridge', 
                    'container-crane', 'ground-track-field', 'harbor', 
                    'helicopter', 'helipad', 'large-vehicle', 'plane', 'roundabout', 
                    'ship', 'small-vehicle', 'soccer-ball-field', 'storage-tank', 
                    'swimming-pool', 'tennis-court']

"""

# Mapping each category to its corresponding parent class
Data1_DOTA2 = {
    'airport': 'transportation_area',
    'baseball-diamond': 'sports_facility',
    'basketball-court': 'sports_facility',
    'bridge': 'transportation_area',
    'container-crane': 'industrial_facility_and_structure',
    'ground-track-field': 'sports_facility',
    'harbor': 'water_structure',
    'helicopter': 'aircraft',
    'helipad': 'transportation_area',
    'large-vehicle': 'vehicle',
    'plane': 'aircraft',
    'roundabout': 'transportation_area',
    'ship': 'ship',
    'small-vehicle': 'vehicle',
    'soccer-ball-field': 'sports_facility',
    'storage-tank': 'industrial_facility_and_structure',
    'swimming-pool': 'recreational_area',
    'tennis-court': 'sports_facility',
}

Data2_DIOR = {
    'golffield': 'sports_facility',          # Golf field is a recreational area
    'vehicle': 'vehicle',                      # Vehicle is its own parent class
    'Expressway-toll-station': 'transportation_area', # Toll stations are part of infrastructure
    'trainstation': 'transportation_area',      # Train station is a transportation hub
    'chimney': 'industrial_facility_and_structure', # Chimney is an industrial structure
    'storagetank': 'industrial_facility_and_structure', # Storage tank is an industrial structure
    'ship': 'ship',                            # Ship is its own parent class
    'harbor': 'water_structure',                      # Harbor falls under maritime
    'airplane': 'aircraft',                    # Airplane is a type of aircraft
    'tenniscourt': 'sports_facility',          # Tennis court is a sports facility
    'groundtrackfield': 'sports_facility',     # Ground track field is a sports facility
    'dam': 'water_structure',                  # Dam is a water structure
    'basketballcourt': 'sports_facility',      # Basketball court is a sports facility
    'Expressway-Service-area': 'transportation_area', # Service areas are part of infrastructure
    'stadium': 'sports_facility',              # Stadium is a sports facility
    'airport': 'transportation_area',           # Airport is a transportation hub
    'baseballfield': 'sports_facility',        # Baseball field is a sports facility
    'bridge': 'transportation_area',                # Bridge is a type of infrastructure
    'windmill': 'energy_facility',             # Windmill is an energy facility
    'overpass': 'transportation_area',              # Overpass is a type of infrastructure
}

Data3_FAIR_1M = {
    'A220': 'aircraft',            # A220 is a type of aircraft
    'A321': 'aircraft',            # A321 is a type of aircraft
    'A330': 'aircraft',            # A330 is a type of aircraft
    'A350': 'aircraft',            # A350 is a type of aircraft
    'ARJ21': 'aircraft',           # ARJ21 is a type of aircraft
    'Boeing737': 'aircraft',       # Boeing737 is a type of aircraft
    'Boeing747': 'aircraft',       # Boeing747 is a type of aircraft
    'Boeing777': 'aircraft',       # Boeing777 is a type of aircraft
    'Boeing787': 'aircraft',       # Boeing787 is a type of aircraft
    'C919': 'aircraft',            # C919 is a type of aircraft
    'other-airplane': 'aircraft',  # General category for other types of airplanes

    'Baseball_Field': 'sports_facility',  # Baseball fields are sports facilities
    'Basketball_Court': 'sports_facility', # Basketball courts are sports facilities
    'Football_Field': 'sports_facility',   # Football fields are sports facilities
    'Tennis_Court': 'sports_facility',     # Tennis courts are sports facilities

    'Bridge': 'transportation_area',    # Bridge is a type of infrastructure
    'Roundabout': 'transportation_area', # Roundabout is part of infrastructure
    'Intersection': 'transportation_area', # Intersection is part of infrastructure

    'Bus': 'vehicle',             # Bus is a type of vehicle
    'Small_Car': 'vehicle',       # Small car is a type of vehicle
    'Cargo_Truck': 'vehicle',     # Cargo truck is a type of vehicle
    'Dump_Truck': 'vehicle',      # Dump truck is a type of vehicle
    'Excavator': 'vehicle',       # Excavator is a type of vehicle
    'Tractor': 'vehicle',         # Tractor is a type of vehicle
    'Trailer': 'vehicle',         # Trailer is a type of vehicle
    'Truck_Tractor': 'vehicle',   # Truck tractor is a type of vehicle
    'Van': 'vehicle',             # Van is a type of vehicle
    'other-vehicle': 'vehicle',   # General category for other types of vehicles

    'Dry_Cargo_Ship': 'ship',      # Dry cargo ships are types of ships
    'Liquid_Cargo_Ship': 'ship',   # Liquid cargo ships are types of ships
    'Engineering_Ship': 'ship',    # Engineering ships are types of ships
    'Fishing_Boat': 'ship',        # Fishing boats are types of ships
    'Motorboat': 'ship',           # Motorboats are types of ships
    'Passenger_Ship': 'ship',      # Passenger ships are types of ships
    'Tugboat': 'ship',             # Tugboats are types of ships
    'Warship': 'ship',             # Warships are types of ships
    'other-ship': 'ship',          # General category for other types of ships

    'Energy_Facility': 'energy_facility',  # 'Energy_Facility' is in the provided parent classes
}

Data4_HRRSD = {
    'Airplane': 'aircraft',               # Airplane is a type of aircraft
    'Storage_Tank': 'industrial_facility_and_structure', # Storage tanks are industrial structures
    'Bridge': 'transportation_area',           # Bridge is a type of infrastructure
    'Ground_Track_Field': 'sports_facility', # Ground track fields are sports facilities
    'Basketball_Court': 'sports_facility', # Basketball courts are sports facilities
    'Tennis_Court': 'sports_facility',     # Tennis courts are sports facilities
    'Ship': 'ship',                       # Ship is its own parent class
    'Baseball_Diamond': 'sports_facility', # Baseball diamonds are sports facilities
    'T_Junction': 'transportation_area',       # T-Junctions are part of infrastructure
    'Crossroad': 'transportation_area',        # Crossroads are part of infrastructure
    'Parking_Lot': 'transportation_area',  # Parking lots are typically part of transportation hubs
    'Harbor': 'water_structure',                 # Harbors fall under maritime
    'Vehicle': 'vehicle',                 # Vehicle is its own parent class
}

Data5_SpaceNet =  {
    'building': 'building',  # 'building' directly maps to 'building'
}

Data6_Xview = {
    'Aircraft_Hangar': 'building',                        # Aircraft hangars are buildings
    'Barge': 'ship',                                     # Barges are ships
    'Building': 'building',                              # Building is its own parent class
    'Bus': 'vehicle',                                   # Bus is a type of vehicle
    'Cargo_Container_Car': 'vehicle',                    # Cargo container cars are vehicles
    'Cargo_Truck': 'vehicle',                            # Cargo trucks are vehicles
    'Cement_Mixer': 'vehicle',                           # Cement mixers are vehicles
    'Construction_Site': 'industrial_facility_and_structure', # Construction sites are industrial facilities
    'Container_Crane': 'industrial_facility_and_structure', # Container cranes are industrial structures
    'Container_Ship': 'ship',                            # Container ships are ships
    'Crane_Truck': 'vehicle',                            # Crane trucks are vehicles
    'Damaged_Building': 'building',                       # Damaged buildings are buildings
    'Dump_Truck': 'vehicle',                             # Dump trucks are vehicles
    'Engineering_Vehicle': 'vehicle',                    # Engineering vehicles are vehicles
    'Excavator': 'vehicle',                              # Excavators are vehicles
    'Facility': 'industrial_facility_and_structure',      # Facilities are often industrial structures
    'Ferry': 'ship',                                    # Ferries are ships
    'Fishing_Vessel': 'ship',                            # Fishing vessels are ships
    'Fixed-wing_Aircraft': 'aircraft',                    # Fixed-wing aircraft are aircraft
    'Flat_Car': 'vehicle',                              # Flat cars are vehicles
    'Front_loader_Bulldozer': 'vehicle',                 # Front loader bulldozers are vehicles
    'Ground_Grader': 'vehicle',                          # Ground graders are vehicles
    'Haul_Truck': 'vehicle',                             # Haul trucks are vehicles
    'Helicopter': 'aircraft',                            # Helicopters are aircraft
    'Helipad': 'building',                              # Helipads are often part of buildings
    'Hut_Tent': 'building',                              # Hut tents are buildings
    'Locomotive': 'vehicle',                             # Locomotives are vehicles
    'Maritime_Vessel': 'ship',                           # Maritime vessels are ships
    'Mobile_Crane': 'vehicle',                           # Mobile cranes are vehicles
    'Motorboat': 'ship',                                # Motorboats are ships
    'Oil_Tanker': 'ship',                               # Oil tankers are ships
    'Passenger_Cargo_Plane': 'aircraft',                 # Passenger cargo planes are aircraft
    'Passenger_Car': 'vehicle',                          # Passenger cars are vehicles
    'Passenger_Vehicle': 'vehicle',                      # Passenger vehicles are vehicles
    'Pickup_Truck': 'vehicle',                           # Pickup trucks are vehicles
    'Pylon': 'transportation_area',                           # Pylons are part of infrastructure
    'Railway_Vehicle': 'vehicle',                        # Railway vehicles are vehicles
    'Reach_Stacker': 'vehicle',                          # Reach stackers are vehicles
    'Sailboat': 'ship',                                 # Sailboats are ships
    'Scraper_Tractor': 'vehicle',                        # Scraper tractors are vehicles
    'Shed': 'building',                                 # Sheds are buildings
    'Shipping_Container': 'industrial_facility_and_structure', # Shipping containers are industrial structures
    'Shipping_container_lot': 'industrial_facility_and_structure', # Shipping container lots are industrial structures
    'Small_Aircraft': 'aircraft',                        # Small aircraft are aircraft
    'Small_Car': 'vehicle',                              # Small cars are vehicles
    'Storage_Tank': 'industrial_facility_and_structure', # Storage tanks are industrial structures
    'Straddle_Carrier': 'vehicle',                       # Straddle carriers are vehicles
    'Tank_car': 'vehicle',                              # Tank cars are vehicles
    'Tower': 'building',                                # Towers are buildings
    'Tower_crane': 'industrial_facility_and_structure', # Tower cranes are industrial structures
    'Trailer': 'vehicle',                               # Trailers are vehicles
    'Truck': 'vehicle',                                # Trucks are vehicles
    'Truck_Tractor': 'vehicle',                         # Truck tractors are vehicles
    'Truck_Tractor_w__Box_Trailer': 'vehicle',           # Truck tractors with box trailers are vehicles
    'Truck_Tractor_w__Flatbed_Trailer': 'vehicle',       # Truck tractors with flatbed trailers are vehicles
    'Truck_Tractor_w__Liquid_Tank': 'vehicle',           # Truck tractors with liquid tanks are vehicles
    'Tugboat': 'ship',                                 # Tugboats are ships
    'Utility_Truck': 'vehicle',                         # Utility trucks are vehicles
    'Vehicle_Lot': 'transportation_area',                # Vehicle lots are part of transportation hubs
    'Yacht': 'ship',                                   # Yachts are ships
}

Data7_HRSC2016 = {
    'Arleigh_Burke': 'ship',                      # Arleigh Burke-class is a type of destroyer (ship)
    'Austen': 'ship',                             # Austen-class is a type of ship
    'Car_carrier': 'ship',                        # Car carriers are a type of ship
    'CntShip': 'ship',                           # Container ships are a type of ship
    'Container': 'ship',                       # Containers are used in industrial facilities
    'Cruise': 'ship',                            # Cruise ships are a type of ship
    'Enterprise': 'ship',                       # Enterprise (e.g., aircraft carrier) is an aircraft
    'Hovercraft': 'ship',                        # Hovercrafts are ships
    'Kuznetsov': 'ship',                         # Kuznetsov-class is a type of aircraft carrier (ship)
    'Medical': 'ship',                          # Medical facilities are generally classified as buildings
    'Midway_class': 'ship',                      # Midway-class is a type of aircraft carrier (ship)
    'Nimitz': 'ship',                           # Nimitz-class is a type of aircraft carrier (ship)
    'OXo': 'ship',                               # OXO class is a type of ship
    'Perry': 'ship',                             # Perry-class is a type of frigate (ship)
    'Sanantonio': 'ship',                        # San Antonio-class is a type of amphibious transport dock (ship)
    'Tarawa': 'ship',                           # Tarawa-class is a type of amphibious assault ship (ship)
    'Ticonderoga': 'ship',                       # Ticonderoga-class is a type of guided missile cruiser (ship)
    'WhidbeyIsland': 'ship',                     # Whidbey Island-class is a type of amphibious dock (ship)
    'aircraft_carrier': 'ship',                   # Aircraft carriers are a type of ship
    'lute': 'ship',                           # 'lute' does not clearly fit any provided parent class
    'merchant_ship': 'ship',                     # Merchant ships are a type of ship
    'ship': 'ship',                              # Generic term for ship
    'submarine': 'ship',                         # Submarines are a type of ship
    'warcraft': 'ship',                          # Warcraft generally refers to military ships
    'yacht': 'ship',                             # Yachts are a type of ship
}
Data8_GLH_Bridge =  {
    'bridge': 'transportation_area'
}

Data9_FMoW = {
    "airport": "transportation_area",  # 机场归入交通枢纽
    "airport_hangar": "transportation_area",  # 机场机库归入基础设施
    "airport_terminal": "transportation_area",  # 机场航站楼归入交通枢纽
    "amusement_park": "public_area",  # 游乐园归入休闲区
    # "aquaculture": "water_structure",  # 水产养殖归入水体结构
    "archaeological_site": "public_area",  # 考古遗址归入基础设施
    # "barn": "building",  # 谷仓归入建筑物
    "border_checkpoint": "transportation_area",  # 边界检查点归入交通枢纽
    "burial_site": "public_area",  # 埋葬地点归入基础设施
    "car_dealership": "building",  # 汽车经销商归入建筑物
    "construction_site": "industrial_facility_and_structure",  # 建筑工地归入工业设施和结构
    # "crop_field": "infrastructure",  # 农田归入基础设施
    "dam": "water_structure",  # 大坝归入水体结构
    # "debris_or_rubble": "infrastructure",  # 瓦砾或碎片归入基础设施
    "educational_institution": "building",  # 教育机构归入建筑物
    "electric_substation": "energy_facility",  # 电力变电站归入能源设施
    "factory_or_powerplant": "industrial_facility_and_structure",  # 工厂或发电厂归入工业设施和结构
    # "fire_station": "building",  # 消防站归入建筑物
    # "flooded_road": "infrastructure",  # 淹水道路归入基础设施
    "fountain": "recreational_area",  # 喷泉归入休闲区
    "gas_station": "building",  # 加油站归入建筑物
    "golf_course": "sports_facility",  # 高尔夫球场归入休闲区
    "ground_transportation_station": "transportation_area",  # 地面交通站点归入交通枢纽
    "helipad": "transportation_area",  # 直升机停机坪归入基础设施
    # "hospital": "building",  # 医院归入建筑物
    # "impoverished_settlement": "building",  # 贫困定居点归入建筑物
    "interchange": "transportation_area",  # 交汇点归入交通枢纽
    "lake_or_pond": "water_structure",  # 湖泊或池塘归入水体结构
    "lighthouse": "water_structure",  # 灯塔归入水体结构
    "military_facility": "military_facility",  # 军事设施归入军事设施
    # "multi-unit_residential": "building",  # 多单元住宅归入建筑物
    "nuclear_powerplant": "energy_facility",  # 核电站归入能源设施
    # "office_building": "building",  # 办公楼归入建筑物
    "oil_or_gas_facility": "energy_facility",  # 石油或天然气设施归入能源设施
    "park": "recreational_area",  # 公园归入休闲区
    "parking_lot_or_garage": "transportation_area",  # 停车场或车库归入基础设施
    # "place_of_worship": "building",  # 礼拜场所归入建筑物
    # "police_station": "building",  # 警察局归入建筑物
    "port": "transportation_area",  # 港口归入交通枢纽
    # "prison": "building",  # 监狱归入建筑物
    "race_track": "sports_facility",  # 赛道归入体育设施
    "railway_bridge": "transportation_area",  # 铁路桥归入基础设施
    # "recreational_facility": "recreational_area",  # 休闲设施归入休闲区
    "road_bridge": "transportation_area",  # 公路桥归入基础设施
    "runway": "transportation_area",  # 跑道归入交通枢纽
    "shipyard": "industrial_facility_and_structure",  # 造船厂归入工业设施和结构
    "shopping_mall": "building",  # 购物中心归入建筑物
    # "single-unit_residential": "building",  # 单元住宅归入建筑物
    "smokestack": "industrial_facility_and_structure",  # 烟囱归入工业设施和结构
    "solar_farm": "energy_facility",  # 太阳能农场归入能源设施
    "space_facility": "transportation_area",  # 太空设施归入基础设施
    "stadium": "sports_facility",  # 体育场归入体育设施
    # "storage_tank": "energy_facility",  # 储罐归入能源设施
    # "surface_mine": "industrial_facility_and_structure",  # 地面矿场归入工业设施和结构
    # "swimming_pool": "recreational_area",  # 游泳池归入休闲区
    # "toll_booth": "transportation_hub",  # 收费站归入交通枢纽
    # "tower": "infrastructure",  # 塔楼归入基础设施
    # "tunnel_opening": "infrastructure",  # 隧道开口归入基础设施
    # "waste_disposal": "industrial_facility_and_structure",  # 废物处理归入工业设施和结构
    # "water_treatment_facility": "energy_facility",  # 水处理设施归入能源设施
    # "wind_farm": "energy_facility",  # 风电场归入能源设施
    # "zoo": "recreational_area"  # 动物园归入休闲区
}

########################################################################