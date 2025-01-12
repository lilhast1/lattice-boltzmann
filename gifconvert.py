from PIL import Image
import glob

folder_path = "sphere"
output_gif = "sphere.gif"
res = 50
# Get all files matching the pattern
all_files = glob.glob(f"{folder_path}/lbm3d_*.png")

# Filter to include only files with numbers that are multiples of 20

image_files = sorted(
    (file for file in all_files if int(file.split("_")[1].split(".")[0]) % res == 0),
    key=lambda x: int(x.split("_")[1].split(".")[0])  # Sort by number
)

# Open images and convert them to the same mode (optional)
images = [Image.open(img).convert("RGBA") for img in image_files]


images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],  # Add the rest of the images
    duration=200,  # Duration for each frame (ms)
    loop=0  # Infinite loop
)

print(f"GIF saved as {output_gif}")
