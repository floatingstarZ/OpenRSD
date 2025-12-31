import os
from pathlib import Path
from ctlib.os import *
from tqdm import tqdm
import numpy as np

parent_mapping = dict(
    # Mapping each category to its corresponding parent class
    Data1_DOTA2={
        'airport': 'transportation_area',  # 机场是与交通相关的区域，属于交通枢纽
        'baseball-diamond': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施
        'basketball-court': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施
        'bridge': 'transportation_area',  # 桥梁是与交通相关的区域，属于交通枢纽
        'container-crane': 'industrial_facility',  # 集装箱起重机用于工业作业，属于工业设施
        'ground-track-field': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施
        'harbor': 'water_area',  # 港口是与水相关的区域，属于水利结构
        'helicopter': 'aircraft',  # 直升机是航空器，属于航空器类别
        'helipad': 'transportation_area',  # 直升机停机坪是与交通相关的区域，属于交通枢纽
        'large-vehicle': 'vehicle',  # 大型车辆属于各种车辆的类别，归入车辆
        'plane': 'aircraft',  # 飞机是航空器，属于航空器类别
        'roundabout': 'transportation_area',  # 环形交叉路口是与交通相关的区域，属于交通枢纽
        'ship': 'ship',  # 船只是各种船只的类别，属于船只
        'small-vehicle': 'vehicle',  # 小型车辆属于各种车辆的类别，归入车辆
        'soccer-ball-field': 'sports_facility',  # 足球场是与体育相关的区域，属于体育设施
        'storage-tank': 'industrial_facility',  # 存储罐是与工业相关的设施，属于工业设施
        'swimming-pool': 'sports_facility',  # 游泳池是与体育相关的区域，属于体育设施
        'tennis-court': 'sports_facility'  # 网球场是与体育相关的区域，属于体育设施
    },

    Data2_DIOR_R={
        'golffield': 'sports_facility',  # 高尔夫球场是与体育相关的区域，属于体育设施
        'vehicle': 'vehicle',  # 车辆指各种汽车，属于车辆类别
        'Expressway-toll-station': 'transportation_area',  # 高速公路收费站是与交通相关的区域，属于交通枢纽
        'trainstation': 'transportation_area',  # 火车站是与交通相关的区域，属于交通枢纽
        'chimney': 'industrial_facility',  # 烟囱是与工业相关的设施，属于工业设施
        'storagetank': 'industrial_facility',  # 存储罐是与工业相关的设施，属于工业设施
        'ship': 'ship',  # 船只是各种船只的类别，属于船只
        'harbor': 'water_area',  # 港口是与水相关的区域，属于水利结构
        'airplane': 'aircraft',  # 飞机是航空器，属于航空器类别
        'tenniscourt': 'sports_facility',  # 网球场是与体育相关的区域，属于体育设施
        'groundtrackfield': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施
        'dam': 'water_area',  # 大坝是与水相关的区域，属于水利结构
        'basketballcourt': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施
        'Expressway-Service-area': 'transportation_area',  # 高速公路服务区是与交通相关的区域，属于交通枢纽
        'stadium': 'sports_facility',  # 体育馆是与体育相关的区域，属于体育设施
        'airport': 'transportation_area',  # 机场是与交通相关的区域，属于交通枢纽
        'baseballfield': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施
        'bridge': 'transportation_area',  # 桥梁是与交通相关的区域，属于交通枢纽
        'windmill': 'energy_facility',  # 风车是与能源相关的设施，属于能源设施
        'overpass': 'transportation_area'  # 高架桥是与交通相关的区域，属于交通枢纽
    },

    Data3_FAIR1M={
        'a220': 'aircraft',  # A220 是一种商业飞机，属于航空器。
        'a321': 'aircraft',  # A321 是一种商业飞机，属于航空器。
        'a330': 'aircraft',  # A330 是一种商业飞机，属于航空器。
        'a350': 'aircraft',  # A350 是一种商业飞机，属于航空器。
        'arj21': 'aircraft',  # ARJ21 是一种中国制造的商业飞机，属于航空器。
        'baseball_field': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施。
        'basketball_court': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施。
        'boeing737': 'aircraft',  # Boeing737 是一种商业飞机，属于航空器。
        'boeing747': 'aircraft',  # Boeing747 是一种商业飞机，属于航空器。
        'boeing777': 'aircraft',  # Boeing777 是一种商业飞机，属于航空器。
        'boeing787': 'aircraft',  # Boeing787 是一种商业飞机，属于航空器。
        'bridge': 'transportation_area',  # 桥梁是交通相关的结构，属于交通枢纽。
        'bus': 'vehicle',  # 公交车是一种车辆，属于车辆类别。
        'c919': 'aircraft',  # C919 是一种中国制造的商业飞机，属于航空器。
        'cargo_truck': 'vehicle',  # 货车是一种车辆，属于车辆类别。
        'dry_cargo_ship': 'ship',  # 干货船是一种船只，属于船只类别。
        'dump_truck': 'vehicle',  # 自卸车是一种车辆，属于车辆类别。
        'engineering_ship': 'ship',  # 工程船是一种船只，属于船只类别。
        'excavator': 'vehicle',  # 挖掘机是一种车辆，属于车辆类别。
        'fishing_boat': 'ship',  # 渔船是一种船只，属于船只类别。
        'football_field': 'sports_facility',  # 足球场是与体育相关的区域，属于体育设施。
        'intersection': 'transportation_area',  # 十字路口是交通相关的结构，属于交通枢纽。
        'liquid_cargo_ship': 'ship',  # 液货船是一种船只，属于船只类别。
        'motorboat': 'ship',  # 小型摩托艇是一种船只，属于船只类别。
        'passenger_ship': 'ship',  # 客船是一种船只，属于船只类别。
        'roundabout': 'transportation_area',  # 环形交叉路口是交通相关的结构，属于交通枢纽。
        'small_car': 'vehicle',  # 小汽车是一种车辆，属于车辆类别。
        'tennis_court': 'sports_facility',  # 网球场是与体育相关的区域，属于体育设施。
        'tractor': 'vehicle',  # 拖拉机是一种车辆，属于车辆类别。
        'trailer': 'vehicle',  # 拖车是一种车辆，属于车辆类别。
        'truck_tractor': 'vehicle',  # 卡车拖头是一种车辆，属于车辆类别。
        'tugboat': 'ship',  # 拖船是一种船只，属于船只类别。
        'van': 'vehicle',  # 面包车是一种车辆，属于车辆类别.
        'warship': 'ship',  # 战舰是一种船只，属于船只类别。
        'other-airplane': 'aircraft',  # 其他类型的飞机，归类为航空器。
        'other-ship': 'ship',  # 其他类型的船只，归类为船只。
        'other-vehicle': 'vehicle'  # 其他类型的车辆，归类为车辆。
    },

    Data4_HRRSD={
        'airplane': 'aircraft',  # 飞机属于航空器。
        'storage_tank': 'industrial_facility',  # 存储罐是与工业相关的设施，属于工业设施。
        'bridge': 'transportation_area',  # 桥梁是与交通相关的结构，属于交通枢纽。
        'ground_track_field': 'sports_facility',  # 田径场是与体育相关的区域，属于体育设施。
        'basketball_court': 'sports_facility',  # 篮球场是与体育相关的区域，属于体育设施。
        'tennis_court': 'sports_facility',  # 网球场是与体育相关的区域，属于体育设施。
        'ship': 'ship',  # 船只属于船只类别。
        'baseball_diamond': 'sports_facility',  # 棒球场是与体育相关的区域，属于体育设施。
        't_junction': 'transportation_area',  # T型交叉路口是交通相关的结构，属于交通枢纽。
        'crossroad': 'transportation_area',  # 十字路口是交通相关的结构，属于交通枢纽。
        'parking_lot': 'public_area',  # 停车场是与公共服务相关的区域，属于公共区域。
        'harbor': 'water_area',  # 港口是与水相关的区域，属于水利结构。
        'vehicle': 'vehicle'  # 车辆属于车辆类别。
    },

    Data5_SpaceNet={
        'building': 'building',  # 'building' directly maps to 'building'
    },

    Data6_Xview={
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
    },

    Data7_HRSC2016={
        'Arleigh_Burke': 'ship',  # Arleigh Burke级驱逐舰是一种军舰，属于船只类别。
        'Austen': 'ship',  # Austen号属于军舰或商业船只，归类为船只。
        'Car_carrier': 'ship',  # 车船是一种专用船只，用于运输汽车，属于船只类别。
        'CntShip': 'ship',  # CntShip是一种商用船只，归类为船只。
        'Container': 'ship',  # 集装箱船是一种船只，用于存储和运输货物。
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
    },

    Data8_GLH_Bridge={
        'bridge': 'transportation_area'
    },

    Data9_FMoW={
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
    },
    Data11_WHU_Mix={
        'building': 'building',  # 'building' directly maps to 'building'
    },
    Data12_ShipImageNet = {
    'aoe': 'ship',  # AOE (Fast Combat Support Ship) is a type of auxiliary ship, thus categorized under 'ship'.
    'arleigh_burke_dd': 'ship',  # Arleigh Burke-class destroyer, a warship, fits under 'ship'.
    'asagiri_dd': 'ship',  # Asagiri-class destroyer, another warship, is categorized as 'ship'.
    'atago_dd': 'ship',  # Atago-class destroyer, a type of destroyer, fits under 'ship'.
    'austin_ll': 'ship',  # Austin-class amphibious transport dock, a type of warship, fits under 'ship'.
    'barge': 'ship',  # A barge is a non-propelled vessel used for cargo transport, classified under 'ship'.
    'cargo': 'ship',  # Cargo ships are a common type of merchant ship, thus categorized under 'ship'.
    'commander': 'ship',  # 'Commander' might refer to a command vessel, which is also a type of ship.
    'container_ship': 'ship',  # Container ships are vessels designed to carry standardized containers, classified under 'ship'.
    'dock': 'transportation_area',  # A dock is part of a port or harbor where ships are loaded/unloaded, categorized as 'transportation_area'.
    'enterprise': 'ship',  # Likely referring to USS Enterprise, an aircraft carrier, classified under 'aircraft'.
    'epf': 'ship',  # Expeditionary Fast Transport, used for high-speed military logistics, is categorized under 'ship'.
    'ferry': 'ship',  # Ferries transport people and vehicles over water, so they are classified as 'ship'.
    'fishing_vessel': 'ship',  # Fishing vessels are watercraft used for fishing, categorized under 'ship'.
    'hatsuyuki_dd': 'ship',  # Hatsuyuki-class destroyer, another warship, fits under 'ship'.
    'hovercraft': 'ship',  # Hovercrafts are amphibious vehicles, capable of traveling over land and water, classified as 'vehicle'.
    'hyuga_dd': 'ship',  # Hyuga-class destroyer, a warship, categorized under 'ship'.
    'lha_ll': 'ship',  # LHA (Amphibious Assault Ship), designed for landing forces, is classified under 'ship'.
    'lsd_41_ll': 'ship',  # LSD-41 (Dock Landing Ship), a warship, fits under 'ship'.
    'masyuu_as': 'ship',  # Masyuu-class ammunition ship, used for logistical support, categorized as 'ship'.
    'medical_ship': 'ship',  # A medical ship provides hospital facilities at sea, categorized under 'ship'.
    'midway': 'ship',  # Likely referring to the Midway-class aircraft carrier, categorized under 'aircraft'.
    'motorboat': 'ship',  # A motorboat is a small powered watercraft, classified under 'ship'.
    'nimitz': 'ship',  # Likely referring to the Nimitz-class aircraft carrier, categorized under 'aircraft'.
    'oil_tanker': 'ship',  # Oil tankers are used for transporting crude oil, categorized under 'ship'.
    'osumi_ll': 'ship',  # Osumi-class landing ship, used for amphibious operations, categorized under 'ship'.
    'other_aircraft_carrier': 'ship',  # Other aircraft carriers are classified under 'aircraft' as they support aircraft operations.
    'other_auxiliary_ship': 'ship',  # Other auxiliary ships provide support to the fleet, classified under 'ship'.
    'other_destroyer': 'ship',  # Other destroyers, a type of warship, fit under 'ship'.
    'other_frigate': 'ship',  # Frigates, a type of warship, fit under 'ship'.
    'other_landing': 'ship',  # Other landing ships used for amphibious operations are classified under 'ship'.
    'other_merchant': 'ship',  # Other merchant vessels involved in commercial trade are classified under 'ship'.
    'other_ship': 'ship',  # General category for various ships, classified under 'ship'.
    'other_warship': 'ship',  # Other warships belong under the 'ship' category.
    'patrol': 'ship',  # Patrol boats, used for patrolling coastal areas, classified under 'ship'.
    'perry_ff': 'ship',  # Oliver Hazard Perry-class frigate, a warship, classified under 'ship'.
    'roro': 'ship',  # RoRo (Roll-on/roll-off) vessels used to carry wheeled cargo, classified under 'ship'.
    'sailboat': 'ship',  # Sailboats are small watercraft propelled by wind, classified under 'ship'.
    'sanantonio_as': 'ship',  # San Antonio-class amphibious transport dock, classified under 'ship'.
    'submarine': 'ship',  # Submarines operate underwater and are categorized under 'ship'.
    'test_ship': 'ship',  # Test ships are used for experimental purposes and are classified under 'ship'.
    'ticonderoga': 'ship',  # Ticonderoga-class cruiser, a warship, fits under 'ship'.
    'training_ship': 'ship',  # Training ships used for educating sailors, categorized under 'ship'.
    'tugboat': 'ship',  # Tugboats assist in maneuvering larger ships, classified under 'ship'.
    'wasp_ll': 'ship',  # Wasp-class amphibious assault ship, categorized under 'ship'.
    'yacht': 'ship',  # Yachts are recreational boats, typically classified under 'ship'.
    'yudao_ll': 'ship',  # Likely a specific class of amphibious landing ship, categorized under 'ship'.
    'yudeng_ll': 'ship',  # Likely a specific class of amphibious landing ship, categorized under 'ship'.
    'yuting_ll': 'ship',  # Likely a specific class of amphibious landing ship, categorized under 'ship'.
    'yuzhao_ll': 'ship',  # Likely a specific class of amphibious landing ship, categorized under 'ship'.
},
)
"""
1. 只保留other_dataset_names，去掉other_cls_names（数据集内的其他类别）
2. 如果负样本有building，则复制多份

"""

#### ------ 父类：[子类]
cls_tree = dict()
for data_name, pt_map in parent_mapping.items():
    for cls_name, pt_cls in pt_map.items():
        if pt_cls not in cls_tree.keys():
            cls_tree[pt_cls] = []
        cls_tree[pt_cls].append(cls_name)
for pt_cls, sub_classes in cls_tree.items():
    cls_tree[pt_cls] = sorted(list(set(sub_classes)))

#### ------ 选取负样本
neg_data_dict = dict()
for data_name, pt_map in parent_mapping.items():
    neg_dict = dict()
    in_data_cls_names = set(pt_map.keys())
    for cls_name, pt_cls in pt_map.items():
        # other_cls_names = list(in_data_cls_names - set([cls_name,]))
        other_dataset_names = []
        for p_cls, sub_classes in cls_tree.items():
            if p_cls == pt_cls:
                continue
            other_dataset_names.extend(sub_classes)
        other_names = other_dataset_names
        neg_dict[cls_name] = sorted(list(set(other_names)))
        # ----- 加赠更多building词汇
        if 'building' in neg_dict[cls_name]:
            neg_dict[cls_name].extend(['building',] * 20)

    neg_data_dict[data_name] = neg_dict

#### ------ 建立全局support字典
all_classes = []
for pt_cls, sub_classes in cls_tree.items():
    all_classes.extend(sub_classes)
all_classes = sorted(list(set(all_classes)))

os.chdir('/opt/data/nfs/huangziyue/Projects/MMRotate_AD')

from M_Tools.Base_Data_infos.data_infos import data_infos
all_support_infos = dict()
for data_name, data_info in data_infos.items():
    if data_name not in parent_mapping.keys():
        print(f'Pass, {data_name}')
        continue
    support_data = pklload(data_info['support_data'])
    for cls_name, support_info in support_data.items():
        if cls_name not in all_support_infos:
            all_support_infos[cls_name] = support_info
            continue
        else:
            if 'texts' not in all_support_infos[cls_name].keys():
                print(all_support_infos[cls_name])
                print(all_support_infos[cls_name])

            texts = all_support_infos[cls_name]['texts']
            text_embeds = all_support_infos[cls_name]['text_embeds']
            visual_embeds = all_support_infos[cls_name]['visual_embeds']
            all_support_infos[cls_name]['texts'] = texts + support_info['texts']
            all_support_infos[cls_name]['text_embeds'] = np.concatenate([text_embeds, support_info['text_embeds']])
            all_support_infos[cls_name]['visual_embeds'] = np.concatenate([visual_embeds, support_info['visual_embeds']])
Neg_supports = dict(
    all_support=all_support_infos,
    cls_tree=cls_tree,
    parent_mapping=parent_mapping,
    neg_dict=neg_data_dict,
)
pklsave(Neg_supports, './data/Neg_supports_v2.pkl')




