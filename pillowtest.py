"""Pillow lib test"""

from PIL import Image, ImageDraw

im = Image.new("RGB", size=(1600, 900))
print(im.format, im.size, im.mode)
draw = ImageDraw.Draw(im);
draw.ellipse([(0, 0), (100, 100)], outline=(255, 0, 0))
im.save("newcreated.jpg")
