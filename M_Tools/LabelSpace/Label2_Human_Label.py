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
我总结了遥感图像中常见的目标，包含物体或者场景。这些目标通常具有较明确的边界，
可以进行目标检测，我需要你发挥想象力，补充可能的其他目标，
其他目标也需要具备较明确的边界，可以使用目标检测的方式来检测。
parent_classes_v2 = [
    # ---- 常见人造物体
    'aircraft',                     # 航空器：各类飞机
    'vehicle',                      # 车辆：各类汽车
    'ship',                         # 船只：各类船只
    'building',                     # 建筑物：各类建筑物
    # ---- 人类活动相关区域、设施
    'transportation_area',          # 交通枢纽：与交通相关的区域。机场、停车场、火车站、桥梁、十字路口、收费站
    'public_area',                  # 公共区域：与公共服务相关的区域。医院、公园、广场、停车场、学校、大型商城、历史遗迹
    'industrial_facility',          # 工业设施：与工业相关的区域。存储罐、集装箱、化工厂、造船厂、建筑工地、发电站、烟囱
    'water_area',                   # 水利结构：与水相关的区域。大坝、湖泊、灯塔、港口
    'sports_facility',              # 体育设施：与体育相关的区域。球类、田径、体育馆、高尔夫球场、游泳池
    'energy_facility',              # 能源设施：与能源相关的区域。风车、能源厂
    # ---- 其他潜在的可检测目标
    'communication_facility',       # 商业设施：写字楼、汽车经销商
    'agricultural_land',            # 农业用地：森林、农田、大棚
]

"""

parent_classes_v2 = [
    # ---- 常见人造物体
    'aircraft',                     # 航空器：各类飞机
    'vehicle',                      # 车辆：各类汽车
    'ship',                         # 船只：各类船只
    'building',                     # 建筑物：各类建筑物
    # ---- 人类活动相关区域、设施
    'transportation_area',          # 交通枢纽：与交通相关的区域。机场、停车场、火车站、桥梁、十字路口、收费站
    'public_area',                  # 公共区域：与公共服务相关的区域。医院、公园、广场、停车场、学校、大型商城、历史遗迹
    'industrial_facility',          # 工业设施：与工业相关的区域。存储罐、集装箱、化工厂、造船厂、建筑工地、发电站、烟囱
    'water_area',                   # 水利结构：与水相关的区域。大坝、湖泊、灯塔、港口
    'sports_facility',              # 体育设施：与体育相关的区域。球类、田径、体育馆、高尔夫球场、游泳池
    'energy_facility',              # 能源设施：与能源相关的区域。风车、能源厂
    # ---- 其他潜在的可检测目标
    'communication_facility',       # 商业设施：写字楼、汽车经销商
    'agricultural_land',            # 农业用地：森林、农田、大棚
]

"""
给定一些parent class列表，涵盖遥感图像中的场景或者物体。
你需要：
1. 将给定的class列表中的每一个元素归类到最恰当的parent class中，以python字典的形式输出。
2. 解释每个元素的归类原因，以注释的方式将解释写在1.中输出的字典中每个映射的后边。
如果无法归类，请给出理由
parent_classes_v2 = [
    # ---- 常见人造物体
    'aircraft',                     # 航空器：各类飞机
    'vehicle',                      # 车辆：各类汽车
    'ship',                         # 船只：各类船只
    'building',                     # 建筑物：各类建筑物
    # ---- 人类活动相关区域、设施
    'transportation_area',          # 交通枢纽：与交通相关的区域。机场、停车场、火车站、桥梁、十字路口、收费站
    'public_area',                  # 公共区域：与公共服务相关的区域。医院、公园、广场、停车场、学校、大型商城、历史遗迹
    'industrial_facility',          # 工业设施：与工业相关的区域。存储罐、集装箱、化工厂、造船厂、建筑工地、发电站、烟囱
    'water_area',                   # 水利结构：与水相关的区域。大坝、湖泊、灯塔、港口
    'sports_facility',              # 体育设施：与体育相关的区域。球类、田径、体育馆、高尔夫球场、游泳池
    'energy_facility',              # 能源设施：与能源相关的区域。风车、能源厂
    # ---- 其他潜在的可检测目标
    'communication_facility',       # 商业设施：写字楼、汽车经销商
    'agricultural_land',            # 农业用地：森林、农田、大棚
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
    'airport': 'transportation_area',         # 机场是与交通相关的区域，属于交通枢纽
    'baseball-diamond': 'sports_facility',    # 棒球场是与体育相关的区域，属于体育设施
    'basketball-court': 'sports_facility',    # 篮球场是与体育相关的区域，属于体育设施
    'bridge': 'transportation_area',          # 桥梁是与交通相关的区域，属于交通枢纽
    'container-crane': 'industrial_facility',  # 集装箱起重机用于工业作业，属于工业设施
    'ground-track-field': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施
    'harbor': 'water_area',                   # 港口是与水相关的区域，属于水利结构
    'helicopter': 'aircraft',                 # 直升机是航空器，属于航空器类别
    'helipad': 'transportation_area',         # 直升机停机坪是与交通相关的区域，属于交通枢纽
    'large-vehicle': 'vehicle',               # 大型车辆属于各种车辆的类别，归入车辆
    'plane': 'aircraft',                     # 飞机是航空器，属于航空器类别
    'roundabout': 'transportation_area',     # 环形交叉路口是与交通相关的区域，属于交通枢纽
    'ship': 'ship',                          # 船只是各种船只的类别，属于船只
    'small-vehicle': 'vehicle',              # 小型车辆属于各种车辆的类别，归入车辆
    'soccer-ball-field': 'sports_facility',  # 足球场是与体育相关的区域，属于体育设施
    'storage-tank': 'industrial_facility',    # 存储罐是与工业相关的设施，属于工业设施
    'swimming-pool': 'sports_facility',      # 游泳池是与体育相关的区域，属于体育设施
    'tennis-court': 'sports_facility'        # 网球场是与体育相关的区域，属于体育设施
}

Data2_DIOR = {
    'golffield': 'sports_facility',           # 高尔夫球场是与体育相关的区域，属于体育设施
    'vehicle': 'vehicle',                     # 车辆指各种汽车，属于车辆类别
    'Expressway-toll-station': 'transportation_area',  # 高速公路收费站是与交通相关的区域，属于交通枢纽
    'trainstation': 'transportation_area',    # 火车站是与交通相关的区域，属于交通枢纽
    'chimney': 'industrial_facility',         # 烟囱是与工业相关的设施，属于工业设施
    'storagetank': 'industrial_facility',     # 存储罐是与工业相关的设施，属于工业设施
    'ship': 'ship',                          # 船只是各种船只的类别，属于船只
    'harbor': 'water_area',                   # 港口是与水相关的区域，属于水利结构
    'airplane': 'aircraft',                  # 飞机是航空器，属于航空器类别
    'tenniscourt': 'sports_facility',        # 网球场是与体育相关的区域，属于体育设施
    'groundtrackfield': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施
    'dam': 'water_area',                     # 大坝是与水相关的区域，属于水利结构
    'basketballcourt': 'sports_facility',    # 篮球场是与体育相关的区域，属于体育设施
    'Expressway-Service-area': 'transportation_area',  # 高速公路服务区是与交通相关的区域，属于交通枢纽
    'stadium': 'sports_facility',            # 体育馆是与体育相关的区域，属于体育设施
    'airport': 'transportation_area',         # 机场是与交通相关的区域，属于交通枢纽
    'baseballfield': 'sports_facility',       # 棒球场是与体育相关的区域，属于体育设施
    'bridge': 'transportation_area',          # 桥梁是与交通相关的区域，属于交通枢纽
    'windmill': 'energy_facility',            # 风车是与能源相关的设施，属于能源设施
    'overpass': 'transportation_area'         # 高架桥是与交通相关的区域，属于交通枢纽
}


Data3_FAIR_1M = {
    'A220': 'aircraft',  # A220 是一种商业飞机，属于航空器。
    'A321': 'aircraft',  # A321 是一种商业飞机，属于航空器。
    'A330': 'aircraft',  # A330 是一种商业飞机，属于航空器。
    'A350': 'aircraft',  # A350 是一种商业飞机，属于航空器。
    'ARJ21': 'aircraft', # ARJ21 是一种中国制造的商业飞机，属于航空器。
    'Baseball_Field': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施。
    'Basketball_Court': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施。
    'Boeing737': 'aircraft',  # Boeing737 是一种商业飞机，属于航空器。
    'Boeing747': 'aircraft',  # Boeing747 是一种商业飞机，属于航空器。
    'Boeing777': 'aircraft',  # Boeing777 是一种商业飞机，属于航空器。
    'Boeing787': 'aircraft',  # Boeing787 是一种商业飞机，属于航空器。
    'Bridge': 'transportation_area',  # 桥梁是交通相关的结构，属于交通枢纽。
    'Bus': 'vehicle',  # 公交车是一种车辆，属于车辆类别。
    'C919': 'aircraft',  # C919 是一种中国制造的商业飞机，属于航空器。
    'Cargo_Truck': 'vehicle',  # 货车是一种车辆，属于车辆类别。
    'Dry_Cargo_Ship': 'ship',  # 干货船是一种船只，属于船只类别。
    'Dump_Truck': 'vehicle',  # 自卸车是一种车辆，属于车辆类别。
    'Engineering_Ship': 'ship',  # 工程船是一种船只，属于船只类别。
    'Excavator': 'vehicle',  # 挖掘机是一种车辆，属于车辆类别。
    'Fishing_Boat': 'ship',  # 渔船是一种船只，属于船只类别。
    'Football_Field': 'sports_facility',  # 足球场是与体育相关的区域，属于体育设施。
    'Intersection': 'transportation_area',  # 十字路口是交通相关的结构，属于交通枢纽。
    'Liquid_Cargo_Ship': 'ship',  # 液货船是一种船只，属于船只类别。
    'Motorboat': 'ship',  # 小型摩托艇是一种船只，属于船只类别。
    'Passenger_Ship': 'ship',  # 客船是一种船只，属于船只类别。
    'Roundabout': 'transportation_area',  # 环形交叉路口是交通相关的结构，属于交通枢纽。
    'Small_Car': 'vehicle',  # 小汽车是一种车辆，属于车辆类别。
    'Tennis_Court': 'sports_facility',  # 网球场是与体育相关的区域，属于体育设施。
    'Tractor': 'vehicle',  # 拖拉机是一种车辆，属于车辆类别。
    'Trailer': 'vehicle',  # 拖车是一种车辆，属于车辆类别。
    'Truck_Tractor': 'vehicle',  # 卡车拖头是一种车辆，属于车辆类别。
    'Tugboat': 'ship',  # 拖船是一种船只，属于船只类别。
    'Van': 'vehicle',  # 面包车是一种车辆，属于车辆类别.
    'Warship': 'ship',  # 战舰是一种船只，属于船只类别。
    'other-airplane': 'aircraft',  # 其他类型的飞机，归类为航空器。
    'other-ship': 'ship',  # 其他类型的船只，归类为船只。
    'other-vehicle': 'vehicle'  # 其他类型的车辆，归类为车辆。
}

Data4_HRRSD = {
    'airplane': 'aircraft',  # 飞机属于航空器。
    'storage_Tank': 'industrial_facility',  # 存储罐是与工业相关的设施，属于工业设施。
    'bridge': 'transportation_area',  # 桥梁是与交通相关的结构，属于交通枢纽。
    'ground_track_field': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施。
    'basketball_court': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施。
    'tennis_court': 'sports_facility',  # 网球场是与体育相关的区域，属于体育设施。
    'ship': 'ship',  # 船只属于船只类别。
    'baseball_diamond': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施。
    't_Junction': 'transportation_area',  # T型交叉路口是交通相关的结构，属于交通枢纽。
    'crossroad': 'transportation_area',  # 十字路口是交通相关的结构，属于交通枢纽。
    'parking_lot': 'public_area',  # 停车场是与公共服务相关的区域，属于公共区域。
    'harbor': 'water_area',  # 港口是与水相关的区域，属于水利结构。
    'vehicle': 'vehicle'  # 车辆属于车辆类别。
}

Data5_SpaceNet =  {
    'building': 'building',  # 'building' directly maps to 'building'
}

Data6_Xview = {
    'Aircraft_Hangar': 'building',  # 飞机库是一种建筑物，用于存放飞机。
    'Barge': 'ship',  # 驳船是一种船只，用于运输货物。
    'Building': 'building',  # 这显然是建筑物。
    'Bus': 'vehicle',  # 公交车是一种车辆。
    'Cargo_or_Container_Car': 'vehicle',  # 装载货物的汽车属于车辆。
    'Cargo_Truck': 'vehicle',  # 货车是一种车辆。
    'Cement_Mixer': 'vehicle',  # 水泥搅拌车是一种特殊用途的车辆。
    'Construction_Site': 'industrial_facility',  # 工地是与工业相关的区域。
    'Container_Crane': 'industrial_facility',  # 集装箱起重机是工业设施的一部分。
    'Container_Ship': 'ship',  # 集装箱船是一种船只。
    'Crane_Truck': 'vehicle',  # 起重机卡车是一种车辆。
    'Damaged_Building': 'building',  # 受损的建筑物仍然属于建筑物类别。
    'Dump_Truck': 'vehicle',  # 自卸车是一种车辆。
    'Engineering_Vehicle': 'vehicle',  # 工程车辆属于车辆类别。
    'Excavator': 'vehicle',  # 挖掘机是一种车辆。
    'Facility': 'building',  # 一般性的设施属于建筑物类别。
    'Ferry': 'ship',  # 渡船是一种船只。
    'Fishing_Vessel': 'ship',  # 渔船是一种船只。
    'Fixed-wing_Aircraft': 'aircraft',  # 固定翼飞机属于航空器。
    'Flat_Car': 'vehicle',  # 平车属于车辆类别。
    'Front_loader_or_Bulldozer': 'vehicle',  # 前铲式推土机属于车辆。
    'Ground_Grader': 'vehicle',  # 地面整平机属于车辆。
    'Haul_Truck': 'vehicle',  # 自卸卡车属于车辆。
    'Helicopter': 'aircraft',  # 直升机属于航空器。
    'Helipad': 'transportation_area',  # 停机坪是与交通相关的区域。
    'Hut_or_Tent': 'building',  # 小屋或帐篷属于建筑物。
    'Locomotive': 'vehicle',  # 机车属于车辆。
    'Maritime_Vessel': 'ship',  # 海洋船只属于船只。
    'Mobile_Crane': 'vehicle',  # 移动起重机属于车辆。
    'Motorboat': 'ship',  # 小型摩托艇是一种船只。
    'Oil_Tanker': 'ship',  # 油轮属于船只。
    'Passenger_or_Cargo_Plane': 'aircraft',  # 客货两用飞机属于航空器。
    'Passenger_Car': 'vehicle',  # 客车属于车辆。
    'Passenger_Vehicle': 'vehicle',  # 客运车辆属于车辆。
    'Pickup_Truck': 'vehicle',  # 皮卡车属于车辆。
    'Pylon': 'building',  # 塔架或电线杆属于建筑物类别。
    'Railway_Vehicle': 'vehicle',  # 铁路车辆属于车辆。
    'Reach_Stacker': 'vehicle',  # 起重堆垛机属于车辆。
    'Sailboat': 'ship',  # 帆船属于船只。
    'Scraper_or_Tractor': 'vehicle',  # 刮土机拖拉机属于车辆。
    'Shed': 'building',  # 棚子属于建筑物类别。
    'Shipping_Container': 'industrial_facility',  # 集装箱属于工业设施。
    'Shipping_container_lot': 'industrial_facility',  # 集装箱堆场属于工业设施。
    'Small_Aircraft': 'aircraft',  # 小型飞机属于航空器。
    'Small_Car': 'vehicle',  # 小汽车属于车辆。
    'Storage_Tank': 'industrial_facility',  # 存储罐属于工业设施。
    'Straddle_Carrier': 'vehicle',  # 跨运车属于车辆。
    'Tank_car': 'vehicle',  # 液罐车属于车辆。
    'Tower': 'building',  # 塔楼属于建筑物类别。
    'Tower_crane': 'industrial_facility',  # 塔式起重机属于工业设施。
    'Trailer': 'vehicle',  # 拖车属于车辆。
    'Truck': 'vehicle',  # 卡车属于车辆。
    'Truck_Tractor': 'vehicle',  # 卡车拖头属于车辆。
    'Truck_Tractor_with_Box_Trailer': 'vehicle',  # 卡车拖头带箱式拖车属于车辆。
    'Truck_Tractor_with_Flatbed_Trailer': 'vehicle',  # 卡车拖头带平板拖车属于车辆。
    'Truck_Tractor_with_Liquid_Tank': 'vehicle',  # 卡车拖头带液体罐属于车辆。
    'Tugboat': 'ship',  # 拖船属于船只。
    'Utility_Truck': 'vehicle',  # 工具车属于车辆。
    'Vehicle_Lot': 'transportation_area',  # 车辆停放场属于公共区域或者交通区域。
    'Yacht': 'ship'  # 游艇属于船只。
}

Data7_HRSC2016 = {
    'Arleigh_Burke': 'ship',  # Arleigh Burke级驱逐舰是一种军舰，属于船只类别。
    'Austen': 'ship',  # Austen号属于军舰或商业船只，归类为船只。
    'Car_carrier': 'ship',  # 车船是一种专用船只，用于运输汽车，属于船只类别。
    'CntShip': 'ship',  # CntShip是一种商用船只，归类为船只。
    'Container': 'ship',  #集装箱船是一种船只，用于存储和运输货物。
    'Cruise': 'ship',  # 游轮是一种客船，属于船只类别。
    'Enterprise': 'ship',  # Enterprise通常指航空母舰或军舰，属于船只。
    'Hovercraft': 'ship',  # 气垫船是一种特殊类型的船只，归类为船只。
    'Kuznetsov': 'ship',  # Kuznetsov是俄罗斯的航空母舰，属于船只类别。
    'Medical': 'ship',  # 医疗船是一种船只。
    'Midway_class': 'ship',  # Midway级航空母舰属于船只。
    'Nimitz': 'ship',  # Nimitz是美国的航空母舰，属于船只类别。
    'OXo': 'ship',  # OXO是一个船只名称，属于船只类别。
    'Perry': 'ship',  # Perry级驱逐舰属于船只类别。
    'Sanantonio': 'ship',  # San Antonio级两栖船只属于船只类别。
    'Tarawa': 'ship',  # Tarawa级两栖攻击舰属于船只。
    'Ticonderoga': 'ship',  # Ticonderoga级巡洋舰属于船只。
    'WhidbeyIsland': 'ship',  # Whidbey Island级两栖船只属于船只类别。
    'aircraft_carrier': 'ship',  # 航空母舰是一种军舰，属于船只。
    'lute': 'ship',  # Lute是船只的名称，归类为船只。
    'merchant_ship': 'ship',  # 商船属于船只类别。
    'ship': 'ship',  # 这个类别明确表示船只。
    'submarine': 'ship',  # 潜艇是一种船只。
    'warcraft': 'ship',  # 战舰或军舰属于船只类别。
    'yacht': 'ship'  # 游艇属于船只类别。
}

Data8_GLH_Bridge =  {
    'bridge': 'transportation_area'
}

Data9_FMoW = {
    'airport': 'transportation_area',  # 机场是与交通相关的区域。
    'airport_hangar': 'building',  # 机场机库属于建筑物类别。
    'airport_terminal': 'building',  # 机场航站楼属于建筑物类别。
    'amusement_park': 'public_area',  # 游乐园是公共区域的一部分。
    'archaeological_site': 'public_area',  # 考古遗址属于公共区域。
    'border_checkpoint': 'transportation_area',  # 边境检查站是与交通相关的区域。
    'burial_site': 'public_area',  # 埋葬地点属于公共区域。
    'car_dealership': 'communication_facility',  # 汽车经销商属于商业设施。
    'construction_site': 'industrial_facility',  # 建筑工地是与工业相关的区域。
    'dam': 'water_area',  # 水坝是与水相关的区域。
    'educational_institution': 'public_area',  # 教育机构（如学校）属于公共区域。
    'electric_substation': 'energy_facility',  # 电力变电站属于能源设施。
    'factory_or_powerplant': 'industrial_facility',  # 工厂或电厂属于工业设施。
    'fountain': 'water_area',  # 喷泉属于水域。
    'gas_station': 'transportation_area',  # 加油站是与交通相关的区域。
    'golf_course': 'sports_facility',  # 高尔夫球场属于体育设施。
    'ground_transportation_station': 'transportation_area',  # 地面交通站点（如公交车站、火车站）属于交通区域。
    'helipad': 'transportation_area',  # 停机坪是与交通相关的区域。
    'interchange': 'transportation_area',  # 交通交汇处属于交通区域。
    'lake_or_pond': 'water_area',  # 湖泊或池塘属于水利结构区域。
    'lighthouse': 'water_area',  # 灯塔属于与水相关的区域。
    'military_facility': 'public_area',  # 军事设施通常被视为公共区域的一部分。
    'nuclear_powerplant': 'energy_facility',  # 核电厂属于能源设施。
    'oil_or_gas_facility': 'industrial_facility',  # 石油或天然气设施属于工业设施。
    'park': 'public_area',  # 公园属于公共区域。
    'parking_lot_or_garage': 'transportation_area',  # 停车场或车库是与交通相关的区域。
    'port': 'water_area',  # 港口是与交通相关的区域，划分为水域。
    'race_track': 'sports_facility',  # 赛道属于体育设施。
    'railway_bridge': 'transportation_area',  # 铁路桥属于交通区域。
    'recreational_facility': 'public_area',  # 娱乐设施属于公共区域。
    'road_bridge': 'transportation_area',  # 公路桥属于交通区域。
    'runway': 'transportation_area',  # 跑道属于与交通相关的区域。
    'shipyard': 'industrial_facility',  # 造船厂属于工业设施。
    'shopping_mall': 'public_area',  # 购物中心属于公共区域。
    'smokestack': 'industrial_facility',  # 烟囱属于工业设施。
    'solar_farm': 'energy_facility',  # 太阳能农场属于能源设施。
    'space_facility': 'public_area',  # 太空设施通常被视为公共区域的一部分。
    'stadium': 'sports_facility'  # 体育场属于体育设施。
}


########################################################################