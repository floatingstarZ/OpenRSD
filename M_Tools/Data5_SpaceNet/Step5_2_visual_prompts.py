"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 20 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['buildings']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['building',]
categories = {
    'building': [
        'Building',
        'A building captured from the aerial image.',
        'A cluster of residential buildings in the aerial image.',
        'An aerial image showcasing commercial buildings in the city center.',
        'Industrial buildings surrounded by greenery in the remote sensing image.',
        'Institutional buildings aligned along the main road in the aerial photo.',
        'Educational buildings with large sports fields visible from above.',
        'Religious buildings with distinctive architecture in the aerial view.',
        'Transportation buildings with busy traffic around them in the remote sensing image.',
        'Agricultural buildings scattered across the rural landscape in the aerial image.',
        'Mixed-use buildings combining residential and commercial spaces in the city.',
    ]
}

