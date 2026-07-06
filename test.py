import os

text = "String.txt"

text, ext = os.path.splitext(text)
print(text, ext)