
"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 10 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['small-vehicle', 'storage-tank', 'large-vehicle',
           'plane', 'ship', 'harbor', 'tennis-court',
           'soccer-ball-field', 'swimming-pool', 'baseball-diamond',
           'ground-track-field', 'roundabout', 'basketball-court',
           'bridge', 'helicopter', 'airport', 'container-crane', 'helipad']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.


"""
classes = ['small-vehicle', 'storage-tank', 'large-vehicle',
           'plane', 'ship', 'harbor', 'tennis-court',
           'soccer-ball-field', 'swimming-pool', 'baseball-diamond',
           'ground-track-field', 'roundabout', 'basketball-court',
           'bridge', 'helicopter', 'airport', 'container-crane', 'helipad']
phrases = {
    'small-vehicle': [
        'A small vehicle is parked on the street in the aerial image.',
        'The aerial image captures a small vehicle traveling through a dense urban area.',
        'In the remote sensing image, a small vehicle can be seen near a busy intersection.',
        'A small vehicle is visible in the parking lot of a shopping center.',
        'The remote image features a small vehicle amidst a group of bicycles.',
        'A small vehicle is navigating through a foggy morning in the aerial view.',
        'In a sunny landscape, a small vehicle is observed alongside a rural road.',
        'A small vehicle is highlighted in the aerial image across from a large commercial building.',
        'The aerial photo shows a small vehicle parked under a tree.',
        'A small vehicle is spotted next to a small café in the detailed remote sensing image.'
    ],
    'storage-tank': [
        'A storage tank is prominently featured in the aerial image.',
        'In the remote sensing view, multiple storage tanks are arranged in a row.',
        'The image captures a large storage tank under clear skies.',
        'A storage tank can be seen beside a factory in the aerial photograph.',
        'In the aerial image, a green storage tank stands out against the industrial skyline.',
        'The remote sensing image includes a storage tank surrounded by agricultural fields.',
        'A storage tank is present in the image amidst a network of pipelines.',
        'The aerial view shows a massive storage tank next to a busy highway.',
        'In the remote sensing image, a weathered storage tank is located near a waterfront.',
        'A storage tank is depicted prominently in the remote aerial view of the port.'
    ],
    'large-vehicle': [
        'A large vehicle traverses the highway in the aerial image.',
        'In the remote sensing view, a large vehicle is loading goods at a distribution center.',
        'The image shows a large vehicle parked in front of a warehouse.',
        'A large vehicle is captured moving through a rural area in the aerial photograph.',
        'The aerial image displays a construction site with a large vehicle actively working.',
        'A large vehicle stands out against the snowy background in the remote view.',
        'In the image, a large vehicle is surrounded by crowding smaller cars.',
        'The remote sensing image includes a large vehicle stationed at a bus terminal.',
        'A large vehicle is depicted on a service road in the aerial view of the industrial sector.',
        'The image features a large vehicle standing still in the middle of a roundabout.'
    ],
    'plane': [
        'A plane is soaring through the sky in the aerial image.',
        'In the remote sensing image, a plane can be seen approaching the airport.',
        'The image captures a plane taking off against a backdrop of mountains.',
        'A plane is depicted flying over the ocean in the remote view.',
        'The aerial photograph shows a plane in the midst of a vibrant sunset.',
        'In the image, a plane is seen parked on the tarmac of a busy airport.',
        'A plane is approaching landing in the remote sensing view near populated areas.',
        'The aerial image captures a plane gliding above a thick layer of clouds.',
        'In the image, a small plane is flying over a schoolyard.',
        'The remote sensing image shows a plane in formation flying over the coast.'
    ],
    'ship': [
        'A ship is anchored in the harbor as seen in the remote sensing image.',
        'In the aerial image, a cargo ship is making its way through the busy shipping lanes.',
        'The remote image reveals a cruise ship gliding along the coastline.',
        'A ship can be seen departing from a port in the aerial photograph.',
        'The image showcases a fishing ship silhouetted against a colorful sunset.',
        'In the remote sensing view, a ship is docked at the wharf surrounded by containers.',
        'A ship is navigating through calm waters in the aerial image.',
        'The aerial photograph showcases a large ship passing under a bridge.',
        'In the image, a ship is shown encountering rough seas.',
        'The remote sensing view features a ship alongside a bustling harbor area.'
    ],
    'harbor': [
        'The harbor is bustling with activity in the aerial image.',
        'In the remote sensing view, a harbor is filled with various ships.',
        'The aerial photograph shows a beautiful harbor lined with palm trees.',
        'A harbor can be seen under clear blue skies in the aerial image.',
        'The remote sensing image captures a busy harbor during sunrise.',
        'Many vessels are docked in the harbor showcased in the aerial view.',
        'In the image, the harbor is bustling with cargo operations.',
        'The aerial view reveals a marina within the harbor filled with boats.',
        'A fishing harbor is displayed in the remote sensing image.',
        'The image depicts a calm harbor at sunset, reflecting the water surface.'
    ],
    'tennis-court': [
        'A tennis court is clearly marked in the aerial image.',
        'In the remote view, a tennis court is busy with players enjoying a sunny day.',
        'The image shows a bright green tennis court surrounded by trees.',
        'A tennis court is featured prominently in the aerial photograph of a sports facility.',
        'The remote image captures a tennis court under the lights during an evening match.',
        'In the aerial view, a tennis court is located next to a swimming pool.',
        'The image showcases multiple tennis courts side by side in a recreational area.',
        'A well-maintained tennis court is depicted in the remote sensing view.',
        'In the image, a tennis court is bustling with activity during a tournament.',
        'The aerial photograph features a tennis court nestled in a suburban neighborhood.'
    ],
    'soccer-ball-field': [
        'A soccer ball field is marked out in the aerial image.',
        'The remote sensing view captures a well-used soccer ball field on a vibrant day.',
        'In the image, a soccer ball field is surrounded by cheering fans.',
        'A soccer ball field is depicted clearly in the aerial view of a sports complex.',
        'The image showcases a soccer ball field with players in action during a match.',
        'In the remote view, a well-maintained soccer ball field is adjacent to a playground.',
        'A soccer ball field is visible under cloudy skies in the aerial photograph.',
        'The remote image captures a soccer ball field at dusk with floodlights on.',
        'In the image, a soccer ball field is bordered by trees and a walking path.',
        'The aerial view shows a community soccer ball field filled with young athletes.'
    ],
    'swimming-pool': [
        'A swimming pool is highlighted in the aerial image on a sunny day.',
        'The remote view captures a residential swimming pool surrounded by lounge chairs.',
        'In the image, a swimming pool is bustling with families enjoying their day.',
        'A swimming pool is visible in the aerial photograph of a hotel complex.',
        'The remote sensing image features a public swimming pool with a diving board.',
        'In the image, a swimming pool is located next to a well-landscaped garden.',
        'A swimming pool is depicted under a clear blue sky in the aerial view.',
        'The aerial image shows a swimming pool being cleaned early in the morning.',
        'In the remote view, a swimming pool is partially shaded by nearby trees.',
        'The image captures a swimming pool area during a summer BBQ event.'
    ],
    'baseball-diamond': [
        'A baseball diamond is clearly marked in the aerial image.',
        'In the remote view, a baseball diamond is hosting a youth game.',
        'The image shows a beautifully maintained baseball diamond from above.',
        'A baseball diamond is depicted in the aerial photograph during a sunny afternoon.',
        'The remote sensing image captures a baseball diamond with players warming up.',
        'In the image, a baseball diamond is surrounded by cheering spectators.',
        'The aerial view shows a baseball diamond near a public park.',
        'A baseball diamond is showcased in the remote view during a league match.',
        'The image features a home run hit just outside the boundaries of the baseball diamond.',
        'In the aerial image, bright colors mark the baseball diamond against green grass.'
    ],
    'ground-track-field': [
        'A ground track field is visible in the aerial image, bustling with athletes.',
        'In the remote view, a ground track field is hosting a community track event.',
        'The image captures a well-maintained ground track field under clear skies.',
        'A ground track field is depicted in the aerial photograph during a school sports day.',
        'The remote sensing image showcases runners on the ground track field.',
        'In the image, a ground track field is surrounded by cheering fans.',
        'A ground track field is visible next to a sports complex in the aerial view.',
        'The aerial photograph shows a ground track field marked with colorful lanes.',
        'In the remote image, a ground track field is being used for a relay race.',
        'The image captures silhouettes of athletes on the ground track field at dusk.'
    ],
    'roundabout': [
        'A roundabout is prominently featured in the remote sensing image.',
        'In the aerial photograph, a roundabout is busy with flowing traffic.',
        'The image shows a beautifully landscaped roundabout from above.',
        'A roundabout is depicted in the aerial view, surrounded by vehicles.',
        'The remote image captures a roundabout illuminated at night.',
        'A roundabout is visible in the image, flanked by pedestrian paths.',
        'In the aerial view, a roundabout connects several major roads.',
        'The image showcases a roundabout adorned with flowers in full bloom.',
        'A roundabout can be seen in the image, efficiently managing traffic flow.',
        'The remote sensing image features a roundabout in a vibrant urban area.'
    ],
    'basketball-court': [
        'A basketball court is vividly displayed in the aerial image.',
        'In the remote view, a basketball court is bustling with players practicing.',
        'The image showcases a neighborhood basketball court during a weekend game.',
        'A basketball court is depicted in the aerial photograph surrounded by trees.',
        'The remote image captures a basketball court at a community center.',
        'In the image, a basketball court is visible with spectators around its edges.',
        'A basketball court is featured prominently in the remote sensing view at dusk.',
        'The aerial view shows a basketball court marked by bright colors.',
        'In the remote view, a basketball court is home to a local tournament.',
        'The image captures the excitement of a basketball court during a final match.'
    ],
    'bridge': [
        'A bridge spans the river in the remote sensing image.',
        'In the aerial photograph, a large bridge connects two bustling cities.',
        'The image captures a beautiful arch bridge under a bright sky.',
        'A bridge is depicted in the remote view with vehicles crossing it.',
        'The aerial image shows a historic bridge surrounded by autumn foliage.',
        'In the picture, a bridge is highlighted amidst a dense urban landscape.',
        'The remote sensing image features a bridge over tranquil waters.',
        'The image shows traffic moving steadily across a bridge at dusk.',
        'A bridge is captured in the aerial image with a backdrop of rolling hills.',
        'In the remote view, a bridge is undergoing maintenance showing workers in action.'
    ],
    'helicopter': [
        'A helicopter is flying swiftly in the remote sensing image.',
        'In the aerial view, a helicopter is landing on a helipad.',
        'The remote image captures a helicopter hovering above a cityscape.',
        'A helicopter is depicted in a sunny setting, surveying the area below.',
        'The image shows a helicopter on a rescue mission near the coastline.',
        'In the remote view, a helicopter is transporting supplies to a remote area.',
        'A helicopter is featured prominently in the aerial image during a news event.',
        'The photograph captures a military helicopter in flight over a field.',
        'In the aerial image, a helicopter is shown circling around a wildfire.',
        'The remote sensing view displays a helicopter preparing for takeoff.'
    ],
    'airport': [
        'An airport is bustling with activity in the aerial image.',
        'In the remote sensing view, an airport can be seen with planes on the runway.',
        'The image captures a sprawling airport complex under clear skies.',
        'An airport is depicted in the aerial photograph with terminals filled with travelers.',
        'In the image, an airport is surrounded by roads and taxiing aircraft.',
        'The remote sensing image features an airport at dawn with planes taking off.',
        'A busy airport is visible in the image with various airlines represented.',
        'The aerial view shows cargo operations at the airport during the day.',
        'In the remote view, an airport is highlighted against the urban landscape.',
        'The image captures a night scene of an airport with lights twinkling.'
    ],
    'container-crane': [
        'A container crane is prominently featured in the remote sensing image.',
        'In the aerial view, a container crane is loading goods onto a ship.',
        'The image captures multiple container cranes working at the busy port.',
        'A large container crane stands tall in the aerial photograph against the skyline.',
        'In the remote view, a container crane is seen towering above cargo containers.',
        'The remote image showcases a container crane actively in use at a dock.',
        'A container crane is depicted amidst a busy harbor scene in the aerial image.',
        'The photograph highlights a container crane at sunrise, enhancing its silhouette.',
        'In the image, a container crane is positioned strategically on the waterfront.',
        'The aerial photograph shows a container crane as vital infrastructure for shipping.'
    ],
    'helipad': [
        'A helipad is clearly marked in the remote sensing image.',
        'In the aerial view, a helipad is situated on top of a hospital building.',
        'The image shows a helipad surrounded by urban development.',
        'A helipad is depicted in the aerial photograph with a helicopter landing.',
        'In the remote view, a helipad is visible on a remote mountain outpost.',
        'The image captures a helipad during sunset with shadows extending outward.',
        'A helipad is showcased in the aerial image amidst lush greenery.',
        'In the remote sensing view, a helipad is busy with incoming helicopter traffic.',
        'The photograph displays a helipad alongside a luxury villa overlooking the ocean.',
        'A helipad is prominently featured in the remote image, emphasizing its accessibility.'
    ]
}



