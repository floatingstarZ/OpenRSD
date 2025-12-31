"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 10 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['Bridge',]"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.
Imagine as many types of bridges as possible

"""
classes = ['bridge',]

phrases = {
    'bridge': [
        'A suspension bridge spans across the river in the aerial image.',
        'The remote sensing image shows a modern arch bridge crossing a valley.',
        'An aerial view reveals an old stone bridge over a narrow stream.',
        'The satellite image captures a highway bridge extending over a large lake.',
        'In the aerial photo, a bridge with multiple supports is seen crossing the bay.',
        'A long viaduct is prominently visible in the remote sensing image.',
        'The image from above shows a bridge with a unique design over a forested area.',
        'An aerial snapshot includes a bridge with a curved structure over a busy city street.',
        'The remote sensing image features a historic bridge in the midst of a rural landscape.',
        'A modern cable-stayed bridge is clearly visible in the satellite image.'
    ]
}
