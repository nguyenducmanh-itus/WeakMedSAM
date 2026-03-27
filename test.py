from PIL import Image, ImageChops

img_1 = Image.open("./data/FracAtlas/images/Fractured/IMG0000019.jpg")
img_2 = Image.open("./data/FracAtlas/images/Fractured/IMG0000025.jpg")
img_1 = img_1.resize((512, 512))
img_1.show()