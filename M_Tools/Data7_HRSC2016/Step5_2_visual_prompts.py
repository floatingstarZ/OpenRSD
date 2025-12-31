"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 3 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['Arleigh_Burke', 'Austen',
           'Car_carrier', 'CntShip', 'Container', 'Cruise',
           'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
           'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
           'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
           'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['Arleigh_Burke', 'Austen',
           'Car_carrier', 'CntShip', 'Container', 'Cruise',
           'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
           'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
           'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
           'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht']

phrases = {
    'Arleigh_Burke': [
        'An Arleigh Burke-class destroyer is visible in the satellite image.',
        'The aerial view shows an Arleigh Burke-class ship navigating through the sea.',
        'In the remote sensing image, an Arleigh Burke-class destroyer is clearly identified.'
    ],
    'Austen': [
        'An Austen-class amphibious transport dock is shown in the aerial photo.',
        'The remote sensing image captures an Austen-class ship near the coast.',
        'In the satellite image, an Austen-class vessel is prominently displayed.'
    ],
    'Car_carrier': [
        'A car carrier ship is visible in the remote sensing image, loaded with vehicles.',
        'The aerial photograph reveals a car carrier navigating the harbor.',
        'In the satellite image, a large car carrier is seen transporting cars across the water.'
    ],
    'CntShip': [
        'A container ship is detected in the aerial view, with its cargo clearly visible.',
        'The remote sensing image shows a large container ship docked at the port.',
        'In the satellite photo, a container ship can be seen amidst the bustling harbor.'
    ],
    'Container': [
        'A container ship appears in the remote sensing image, filled with stacked containers.',
        'The aerial view reveals a container vessel maneuvering through the shipping lanes.',
        'In the satellite image, a ship carrying numerous containers is identifiable.'
    ],
    'Cruise': [
        'A luxury cruise ship is visible in the aerial image, cruising through calm waters.',
        'The remote sensing image shows a large cruise ship docked at a popular tourist port.',
        'In the satellite photo, a cruise liner is seen navigating the open sea.'
    ],
    'Enterprise': [
        'The remote sensing image features the USS Enterprise, an iconic aircraft carrier.',
        'An aerial photograph captures the USS Enterprise in a naval exercise.',
        'In the satellite view, the USS Enterprise is prominently displayed in the harbor.'
    ],
    'Hovercraft': [
        'A hovercraft is seen in the aerial image, gliding over the water surface.',
        'The remote sensing image shows a hovercraft navigating in shallow waters.',
        'In the satellite photo, a hovercraft is visible approaching the shore.'
    ],
    'Kuznetsov': [
        'The Russian aircraft carrier Kuznetsov is visible in the remote sensing image.',
        'An aerial view shows the Kuznetsov carrier conducting operations at sea.',
        'In the satellite photo, the Kuznetsov aircraft carrier is clearly identifiable.'
    ],
    'Medical': [
        'A medical evacuation ship is seen in the remote sensing image, stationed offshore.',
        'The aerial photograph reveals a medical ship providing assistance near the coast.',
        'In the satellite image, a medical vessel is visible in a disaster relief operation.'
    ],
    'Midway_class': [
        'A Midway-class aircraft carrier is captured in the remote sensing image, moored at port.',
        'The aerial view shows a Midway-class ship in the middle of a naval fleet.',
        'In the satellite photo, a Midway-class aircraft carrier is clearly distinguished.'
    ],
    'Nimitz': [
        'The Nimitz-class aircraft carrier is visible in the remote sensing image, sailing across the ocean.',
        'An aerial view reveals the Nimitz-class carrier conducting maneuvers at sea.',
        'In the satellite photo, the Nimitz-class aircraft carrier is prominently displayed.'
    ],
    'OXo': [
        'An OXO-class naval vessel is seen in the remote sensing image, docked at a naval base.',
        'The aerial photograph shows an OXO-class ship in a busy harbor.',
        'In the satellite image, an OXO-class vessel is identifiable in the water.'
    ],
    'Perry': [
        'A Perry-class frigate is visible in the aerial image, patrolling the waters.',
        'The remote sensing image captures a Perry-class ship in a naval exercise.',
        'In the satellite view, a Perry-class frigate is clearly seen near the coastline.'
    ],
    'Sanantonio': [
        'A San Antonio-class amphibious transport dock is detected in the remote sensing image.',
        'The aerial photograph shows a San Antonio-class ship docking at a military base.',
        'In the satellite photo, a San Antonio-class vessel is visible in the harbor.'
    ],
    'Tarawa': [
        'The Tarawa-class amphibious assault ship is visible in the remote sensing image, anchored offshore.',
        'An aerial view shows a Tarawa-class ship conducting operations near a coastal area.',
        'In the satellite image, a Tarawa-class vessel is seen in the marine environment.'
    ],
    'Ticonderoga': [
        'A Ticonderoga-class cruiser is visible in the remote sensing image, cruising through open waters.',
        'The aerial photograph shows a Ticonderoga-class ship in a naval formation.',
        'In the satellite view, a Ticonderoga-class cruiser is identifiable in the fleet.'
    ],
    'WhidbeyIsland': [
        'A Whidbey Island-class dock landing ship is seen in the aerial image, preparing for amphibious operations.',
        'The remote sensing image captures a Whidbey Island-class vessel at sea.',
        'In the satellite photo, a Whidbey Island-class ship is visible near the shore.'
    ],
    'aircraft_carrier': [
        'An aircraft carrier is visible in the remote sensing image, moving through the sea.',
        'The aerial photograph reveals an aircraft carrier docked at a naval base.',
        'In the satellite image, an aircraft carrier is clearly identified in the water.'
    ],
    'lute': [
        'A lute-class naval vessel is seen in the remote sensing image, positioned in a strategic location.',
        'The aerial view shows a lute-class ship maneuvering through the harbor.',
        'In the satellite photo, a lute-class vessel is visible at sea.'
    ],
    'merchant_ship': [
        'A merchant ship is visible in the remote sensing image, transporting cargo across the ocean.',
        'The aerial photograph captures a merchant vessel docked at a busy port.',
        'In the satellite view, a merchant ship is clearly identified in the shipping lane.'
    ],
    'ship': [
        'A ship is detected in the remote sensing image, sailing through open waters.',
        'The aerial view shows a ship approaching the harbor.',
        'In the satellite photo, a ship is visible amidst the marine landscape.'
    ],
    'submarine': [
        'A submarine is visible in the remote sensing image, submerged near the surface.',
        'The aerial photograph reveals a submarine surfaced during a naval exercise.',
        'In the satellite view, a submarine is identified beneath the water surface.'
    ],
    'warcraft': [
        'A warcraft is seen in the remote sensing image, participating in a naval maneuver.',
        'The aerial view shows a warcraft docked at a military port.',
        'In the satellite photo, a warcraft is clearly visible amidst other naval vessels.'
    ],
    'yacht': [
        'A luxury yacht is visible in the remote sensing image, cruising along the coastline.',
        'The aerial photograph captures a yacht docked in a scenic marina.',
        'In the satellite image, a yacht is seen sailing through tranquil waters.'
    ]
}
