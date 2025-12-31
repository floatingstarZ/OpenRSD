
"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 10 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['golffield', 'vehicle', 'Expressway-toll-station',
           'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
           'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
           'basketballcourt', 'Expressway-Service-area', 'stadium',
           'airport', 'baseballfield', 'bridge', 'windmill', 'overpass']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""

classes = ['golffield', 'vehicle', 'Expressway-toll-station',
           'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
           'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
           'basketballcourt', 'Expressway-Service-area', 'stadium',
           'airport', 'baseballfield', 'bridge', 'windmill', 'overpass']
phrases = {
    'golffield': [
        'A golf field visible in the satellite image.',
        'The aerial photo shows a sprawling golf field.',
        'Golf fields appear in the overhead shot.',
        'An expanse of golf field captured in the drone image.',
        'A lush golf field in the remote sensing image.',
        'Golf fields under a clear sky in the aerial image.',
        'A golf field surrounded by trees in the satellite picture.',
        'Aerial imagery captures the golf field.',
        'The remote sensing photo includes a golf field.',
        'A golf field during a sunny day in the satellite image.'
    ],
    'vehicle': [
        'Several vehicles seen in the aerial image.',
        'A car moving along the road in the drone image.',
        'A remote sensing image shows multiple vehicles.',
        'Vehicles scattered across the parking lot in the satellite image.',
        'A vehicle traveling in the overhead shot.',
        'The aerial photo captures a line of vehicles.',
        'Vehicles visible on the highway in the satellite image.',
        'A cluster of vehicles in the drone picture.',
        'A vehicle parked near the building in the aerial photo.',
        'Remote sensing captures a vehicle on the street.'
    ],
    'Expressway-toll-station': [
        'An expressway toll station seen in the satellite image.',
        'The aerial image shows an expressway toll station.',
        'A toll station on the expressway in the drone photo.',
        'Remote sensing captures the expressway toll station.',
        'An expressway toll station visible in the overhead shot.',
        'A line of cars at the toll station in the aerial image.',
        'The satellite picture includes an expressway toll station.',
        'A toll station on a busy expressway in the drone image.',
        'The expressway toll station seen in the remote sensing image.',
        'Aerial imagery shows an expressway toll station.'
    ],
    'trainstation': [
        'A train station visible in the aerial image.',
        'The drone photo shows a bustling train station.',
        'A remote sensing image captures the train station.',
        'A train station with several tracks in the satellite picture.',
        'The aerial shot includes a train station.',
        'A train station during rush hour in the overhead image.',
        'A train station surrounded by buildings in the drone picture.',
        'Remote sensing shows the train station clearly.',
        'The train station seen from above in the satellite image.',
        'A train station bustling with activity in the aerial photo.'
    ],
    'chimney': [
        'A chimney visible in the satellite image.',
        'The aerial image shows a tall chimney.',
        'A remote sensing image captures a smoking chimney.',
        'The drone picture includes an industrial chimney.',
        'A chimney releasing smoke in the overhead shot.',
        'The aerial photo captures a factory chimney.',
        'A chimney standing tall in the satellite picture.',
        'Remote sensing shows a chimney in an industrial area.',
        'A chimney with smoke in the aerial image.',
        'A tall chimney seen in the remote sensing image.'
    ],
    'storagetank': [
        'A storage tank seen in the aerial image.',
        'The drone photo shows a large storage tank.',
        'A remote sensing image captures several storage tanks.',
        'The satellite picture includes a storage tank.',
        'A storage tank in an industrial area in the overhead shot.',
        'The aerial image shows multiple storage tanks.',
        'A storage tank visible in the drone picture.',
        'Remote sensing captures a large storage tank.',
        'Aerial imagery shows a storage tank at a facility.',
        'A storage tank seen in the satellite image.'
    ],
    'ship': [
        'A ship visible in the aerial image.',
        'The drone photo captures a large ship.',
        'A remote sensing image shows a ship at sea.',
        'The satellite picture includes a ship near the harbor.',
        'A ship sailing in the overhead shot.',
        'A ship docked at the port in the aerial photo.',
        'The aerial image shows a cargo ship.',
        'Remote sensing captures a ship on the water.',
        'A ship seen from above in the satellite image.',
        'A large ship in the aerial image.'
    ],
    'harbor': [
        'A harbor visible in the satellite image.',
        'The aerial image shows a busy harbor.',
        'A remote sensing image captures the harbor.',
        'The drone picture includes a harbor with many ships.',
        'A harbor full of vessels in the overhead shot.',
        'A harbor seen in the aerial photo.',
        'The satellite image shows a bustling harbor.',
        'Remote sensing captures a harbor area.',
        'A harbor with boats in the aerial image.',
        'Aerial imagery shows a large harbor.'
    ],
    'airplane': [
        'An airplane visible in the aerial image.',
        'The drone photo captures an airplane on the runway.',
        'A remote sensing image shows an airplane in flight.',
        'The satellite picture includes an airplane at the airport.',
        'An airplane taking off in the overhead shot.',
        'An airplane on the tarmac in the aerial photo.',
        'The aerial image shows an airplane at the gate.',
        'Remote sensing captures an airplane in the sky.',
        'An airplane seen from above in the satellite image.',
        'An airplane parked in the aerial image.'
    ],
    'tenniscourt': [
        'A tennis court seen in the aerial image.',
        'The drone photo shows a tennis court.',
        'A remote sensing image captures several tennis courts.',
        'The satellite picture includes a tennis court.',
        'A tennis court in the overhead shot.',
        'The aerial image shows a tennis court in a park.',
        'A tennis court visible in the drone picture.',
        'Remote sensing captures a tennis court in use.',
        'Aerial imagery shows a tennis court with players.',
        'A tennis court seen in the satellite image.'
    ],
    'groundtrackfield': [
        'A ground track field visible in the satellite image.',
        'The aerial image shows a ground track field.',
        'A remote sensing image captures the ground track field.',
        'The drone picture includes a ground track field with runners.',
        'A ground track field in the overhead shot.',
        'The aerial photo captures a ground track field in use.',
        'A ground track field surrounded by spectators in the satellite picture.',
        'Remote sensing shows the ground track field clearly.',
        'A ground track field seen from above in the aerial image.',
        'A ground track field during a race in the satellite image.'
    ],
    'dam': [
        'A dam visible in the satellite image.',
        'The aerial image shows a large dam.',
        'A remote sensing image captures the dam.',
        'The drone picture includes a dam with water.',
        'A dam in the overhead shot.',
        'The aerial photo captures a dam in operation.',
        'A dam seen in the satellite picture.',
        'Remote sensing shows the dam clearly.',
        'A dam in the mountains in the aerial image.',
        'A large dam in the satellite image.'
    ],
    'basketballcourt': [
        'A basketball court seen in the aerial image.',
        'The drone photo shows a basketball court.',
        'A remote sensing image captures several basketball courts.',
        'The satellite picture includes a basketball court.',
        'A basketball court in the overhead shot.',
        'The aerial image shows a basketball court in a park.',
        'A basketball court visible in the drone picture.',
        'Remote sensing captures a basketball court in use.',
        'Aerial imagery shows a basketball court with players.',
        'A basketball court seen in the satellite image.'
    ],
    'Expressway-Service-area': [
        'An expressway service area seen in the aerial image.',
        'The drone photo shows an expressway service area.',
        'A remote sensing image captures the expressway service area.',
        'The satellite picture includes an expressway service area.',
        'An expressway service area in the overhead shot.',
        'The aerial image shows a busy expressway service area.',
        'An expressway service area visible in the drone picture.',
        'Remote sensing captures an expressway service area with facilities.',
        'Aerial imagery shows an expressway service area with vehicles.',
        'An expressway service area seen in the satellite image.'
    ],
    'stadium': [
        'A stadium visible in the satellite image.',
        'The aerial image shows a large stadium.',
        'A remote sensing image captures the stadium.',
        'The drone picture includes a stadium full of spectators.',
        'A stadium in the overhead shot.',
        'The aerial photo captures a stadium during an event.',
        'A stadium seen in the satellite picture.',
        'Remote sensing shows the stadium clearly.',
        'A stadium in the city in the aerial image.',
        'A large stadium in the satellite image.'
    ],
    'airport': [
        'An airport visible in the aerial image.',
        'The drone photo captures a busy airport.',
        'A remote sensing image shows an airport with multiple runways.',
        'The satellite picture includes an airport terminal.',
        'An airport seen in the overhead shot.',
        'The aerial image shows planes at the airport.',
        'An airport with many planes in the drone picture.',
        'Remote sensing captures an airport from above.',
        'An airport seen in the satellite image.',
        'A large airport in the aerial image.'
    ],
    'baseballfield': [
        'A baseball field seen in the aerial image.',
        'The drone photo shows a baseball field.',
        'A remote sensing image captures a baseball field with players.',
        'The satellite picture includes a baseball field.',
        'A baseball field in the overhead shot.',
        'The aerial image shows a baseball field in a park.',
        'A baseball field visible in the drone picture.',
        'Remote sensing captures a baseball field in use.',
        'Aerial imagery shows a baseball field with spectators.',
        'A baseball field seen in the satellite image.'
    ],
    'bridge': [
        'A bridge visible in the satellite image.',
        'The aerial image shows a long bridge.',
        'A remote sensing image captures the bridge over water.',
        'The drone picture includes a bridge with traffic.',
        'A bridge seen in the overhead shot.',
        'The aerial photo captures a bridge connecting two areas.',
        'A bridge seen in the satellite picture.',
        'Remote sensing shows the bridge clearly.',
        'A bridge crossing a river in the aerial image.',
        'A large bridge in the satellite image.'
    ],
    'windmill': [
        'A windmill seen in the aerial image.',
        'The drone photo shows a windmill farm.',
        'A remote sensing image captures several windmills.',
        'The satellite picture includes a windmill in a field.',
        'A windmill in the overhead shot.',
        'The aerial image shows a windmill generating power.',
        'A windmill visible in the drone picture.',
        'Remote sensing captures a windmill in operation.',
        'Aerial imagery shows a windmill on a hill.',
        'A windmill seen in the satellite image.'
    ],
    'overpass': [
        'An overpass visible in the aerial image.',
        'The drone photo shows a busy overpass.',
        'A remote sensing image captures an overpass with vehicles.',
        'The satellite picture includes an overpass connecting highways.',
        'An overpass seen in the overhead shot.',
        'The aerial image shows an overpass during rush hour.',
        'An overpass visible in the drone picture.',
        'Remote sensing captures an overpass in the city.',
        'Aerial imagery shows an overpass with traffic.',
        'An overpass seen in the satellite image.'
    ]
}




