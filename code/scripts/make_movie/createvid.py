import subprocess
import pickle as pkl
import os

# Change the working directory to 'figs'
os.chdir('figs')

for file in os.listdir():
    if not file.endswith('.pkl'):
        continue

    fig = pkl.load(open(file, 'rb'))
    ax = fig.get_axes()[0]

    ax.set_xlim(3.0, 3.5)
    ax.set_ylim(-2.5, -0.3)

    # Extract the amplitude strength from the file name
    amp = float(file.split('_')[-1].replace('.pkl', ''))

    # Format the amplitude to 5 significant digits
    amp_formatted = "{:.5g}".format(amp)

    # Construct the new file name
    new_file_name = 'manifold_' + amp_formatted + '.png'

    fig.savefig(new_file_name)