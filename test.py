from PIL import Image, ImageDraw

img = Image.open("data/BTXRD/images/IMG000023.jpeg")
img = img.convert("RGB")
print(img.size)
h, w = img.size
res_img = img.resize((512, 512))
bbx = [[
          181.6176470588235,
          2.941176470588235
        ],
        [
          727.409090909091,
          625.7272727272727
        ]] 
bbx[0][0] =  int(bbx[0][0] * (512 / w))
bbx[0][1] = int(bbx[0][1] * (512 / h))
bbx[1][0] =  int(bbx[1][0] * (512 / w))
bbx[1][1] = int(bbx[1][1] * (512 / h))
xy1 = tuple(bbx[0])
xy2 = tuple(bbx[1])
img1 = ImageDraw.Draw(res_img)
img1.rectangle([xy1, xy2], outline='red', width=2)
res_img.show()