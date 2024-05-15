import subprocess
import pickle as pkl
import os
import re

# Change the working directory to 'figs'
os.chdir('figs')

filelist = []
for file in os.listdir():
    if not file.endswith('.png'):
        continue
    else:
        filelist.append(file)

# Define a key function that extracts the important number from a filename
def get_important_number(filename):
    match = re.search(r'_(-?\d+\.\d+e[+-]\d+)_', filename)
    if match:
        return float(match.group(1))
    else:
        return 0  # Default value if no match is found

# Sort the filelist using the key function
filelist.sort(key=get_important_number)

# Write the sorted file names to a text file
with open('filelist.txt', 'w') as f:
    for file in filelist:
        f.write(f"file '{file}'\n")
        f.write("duration 0.05\n")

# Use ffmpeg to create a video from the images
subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt', '-r', '2', '../output.mp4'])

# for file in os.listdir():
#     if not file.endswith('.pkl'):
#         continue

#     fig = pkl.load(open(file, 'rb'))
#     ax = fig.get_axes()[0]

#     ax.set_xlim(3.0, 3.5)
#     ax.set_ylim(-2.5, -0.3)

#     # Extract the amplitude strength from the file name
#     amp = float(file.split('_')[-1].replace('.pkl', ''))

#     # Format the amplitude to 5 significant digits
#     amp_formatted = "{:.5g}".format(amp)

#     # Construct the new file name
#     new_file_name = 'manifold_' + amp_formatted + '.png'

#     fig.savefig(new_file_name)