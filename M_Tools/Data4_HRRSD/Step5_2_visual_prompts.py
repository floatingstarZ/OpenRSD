"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 7 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['airplane', 'storage', 'bridge', 'ground', 'basketball',
          'tennis', 'ship', 'baseball',
          'T', 'crossroad', 'parking', 'harbor', 'vehicle']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['airplane', 'storage', 'bridge', 'ground', 'basketball',
          'tennis', 'ship', 'baseball',
          'T', 'crossroad', 'parking', 'harbor', 'vehicle']
phrases = {
    'airplane': [
        'An airplane is visible in the remote sensing image.',
        'The aerial image shows an airplane on the runway.',
        'A jet appears in the satellite photo.',
        'An airplane flying over the landscape is captured in this image.',
        'The image reveals an airplane at an airport.',
        'A plane is seen soaring in the clear sky.',
        'An aircraft is depicted in the overhead shot.'
    ],
    'storage': [
        'Storage units are present in the aerial view.',
        'The image captures storage facilities in an industrial area.',
        'Aerial imagery shows a series of storage warehouses.',
        'Storage buildings are visible in the remote sensing image.',
        'An overhead shot reveals multiple storage structures.',
        'The photo highlights storage units amidst other buildings.',
        'Storage areas can be seen in the satellite image.'
    ],
    'bridge': [
        'A bridge spans across the river in the aerial image.',
        'The remote sensing image captures a bridge over the water.',
        'A bridge is visible connecting two landmasses.',
        'The image depicts a bridge with vehicles crossing it.',
        'An overhead view shows a long bridge.',
        'A bridge is captured in the image, arching over a canyon.',
        'The photo reveals a bridge amid a bustling cityscape.'
    ],
    'ground': [
        'Ground features are prominent in the satellite image.',
        'The aerial view highlights the ground terrain.',
        'An image shows various ground textures and surfaces.',
        'The ground is visible with different land uses in the photo.',
        'A remote sensing image displays diverse ground patterns.',
        'The ground is clearly seen in the rural area image.',
        'Ground details are captured in the urban landscape.'
    ],
    'basketball': [
        'A basketball court is visible in the aerial image.',
        'The photo shows a basketball game in progress on the court.',
        'An outdoor basketball court is depicted in the satellite image.',
        'The image captures a basketball court in a park.',
        'A basketball court is seen amidst a residential area.',
        'The aerial view shows people playing basketball.',
        'A basketball court is clearly marked in the school yard image.'
    ],
    'tennis': [
        'A tennis court is visible in the remote sensing image.',
        'The aerial view captures a tennis match in progress.',
        'An image shows multiple tennis courts in a sports complex.',
        'The satellite photo highlights a tennis court in a park.',
        'A tennis court is seen surrounded by greenery.',
        'The photo reveals players on a tennis court.',
        'An overhead view shows a tennis court with clear markings.'
    ],
    'ship': [
        'A ship is seen in the harbor in the aerial image.',
        'The remote sensing image shows a ship sailing in the ocean.',
        'An image captures a large ship docked at the port.',
        'The photo depicts a ship navigating through the sea.',
        'A ship is visible near the coastline in the satellite image.',
        'The aerial view reveals a cargo ship at sea.',
        'A ship is shown anchored in the bay.'
    ],
    'baseball': [
        'A baseball field is visible in the aerial image.',
        'The remote sensing image captures a baseball game in progress.',
        'An image shows a baseball diamond in a park.',
        'The photo depicts a baseball field with players on it.',
        'A baseball field is seen surrounded by bleachers.',
        'The aerial view shows a baseball field in a school yard.',
        'A baseball game is visible in the stadium image.'
    ],
    'T': [
        'A T-junction is visible in the aerial image.',
        'The remote sensing image captures a T-shaped intersection.',
        'An image shows a T-junction in an urban area.',
        'The photo depicts a T-intersection with traffic.',
        'A T-junction is seen amidst residential streets.',
        'The aerial view shows a T-intersection in a rural area.',
        'A T-shaped road junction is clearly visible in the image.'
    ],
    'crossroad': [
        'A crossroad is visible in the aerial image.',
        'The remote sensing image captures a busy intersection.',
        'An image shows a crossroad with multiple lanes.',
        'The photo depicts a crossroad in a bustling city.',
        'A crossroad is seen connecting four streets.',
        'The aerial view shows a crossroad with traffic signals.',
        'A major crossroad is clearly visible in the urban image.'
    ],
    'parking': [
        'A parking lot is visible in the aerial image.',
        'The remote sensing image captures a full parking area.',
        'An image shows a parking lot with many vehicles.',
        'The photo depicts a parking area next to a shopping mall.',
        'A parking lot is seen in the satellite image.',
        'The aerial view shows a parking lot in a busy area.',
        'A large parking area is visible in the industrial zone image.'
    ],
    'harbor': [
        'A harbor is visible in the aerial image.',
        'The remote sensing image captures ships docked in a harbor.',
        'An image shows a harbor with various boats and ships.',
        'The photo depicts a busy harbor with cargo ships.',
        'A harbor is seen in the coastal area image.',
        'The aerial view shows a harbor with fishing boats.',
        'A large harbor is clearly visible in the image.'
    ],
    'vehicle': [
        'Vehicles are visible in the aerial image.',
        'The remote sensing image captures cars on the highway.',
        'An image shows vehicles parked along the streets.',
        'The photo depicts vehicles moving through a city.',
        'Vehicles are seen in the image of a traffic intersection.',
        'The aerial view shows a row of parked vehicles.',
        'Various types of vehicles are visible in the urban image.'
    ]
}
