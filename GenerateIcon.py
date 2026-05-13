from PIL import Image
import os

src = "Assets/fuzzyStride-logo.png"
dst = "Assets/fuzzyStride-logo.ico"
img = Image.open(src).convert("RGBA")
sizes = [(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]
img.save(dst, format="ICO", sizes=sizes)

print(f"Icon saved to: {os.path.abspath(dst)}")