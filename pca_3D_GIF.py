# Script to plot 3D PCA

## Inputs:
#			-X: feature set_xlabel
#			-y: target variables

## Outputs:
			- Gif of 3D video showing PCA plot 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

pca = PCA(n_components = 3)
result = pca.fit_transform(X)

import random, glob, os, IPython.display as IPdisplay
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
%matplotlib

save_folder = r'C:\Users\Administrator\Documents\Experiment1'

# set a filename, run the logistic model, and create the plot
gif_filename = 'e1kpcarbfvideo'
working_folder = '{}/{}'.format(save_folder, gif_filename)
if not os.path.exists(working_folder):
    os.makedirs(working_folder)


# Initialize figure
fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection = '3d')

# Plot PCA results
ax.scatter(result[y==1][0], result[y==1][1],result[y==1][2],label='Class 1',c='gold', s = 200)
ax.scatter(result[y==2][0], result[y==2][1],result[y==2][2],label='Class 2',c='seagreen', s = 200)
ax.scatter(result[y==3][0], result[y==3][1],result[y==3][2],label='Class 3',c='violet', s = 200)
ax.scatter(result[y==4][0], result[y==4][1],result[y==4][2],label='Class 4',c='navy', s = 200)

# make simple, bare axis lines through space:
xAxisLine = ((min(result[0]), max(result[0])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result[1]), max(result[1])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result[2]), max(result[2])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

# Legend
ax.legend()

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

#plt.show()

## Make Pics for video

plt.show()

# create frames (steps) for the animated gif
steps = 250

# a viewing perspective is composed of an elevation, distance, and azimuth
# define the range of values we'll cycle through for the distance of the viewing perspective
min_dist = 7.
max_dist = 9.
dist_range = np.arange(min_dist, max_dist, (max_dist-min_dist)/steps)

# define the range of values we'll cycle through for the elevation of the viewing perspective
min_elev = 0.
max_elev = 35.
elev_range = np.arange(max_elev, min_elev, (min_elev-max_elev)/steps)

# now create the individual frames that will be combined later into the animation
for azimuth in range(0, 360, int(360/steps)):
    
    # pan down, rotate around, and zoom out
    ax.azim = float(azimuth/3.)
    ax.elev = elev_range[int(azimuth/(360./steps))]
    ax.dist = dist_range[int(azimuth/(360./steps))]
    
    # set the figure title to the viewing perspective, and save each figure as a .png
    fig.suptitle('elev={:.1f}, azim={:.1f}, dist={:.1f}'.format(ax.elev, ax.azim, ax.dist))
    plt.savefig('{}/{}/img{:03d}.png'.format(save_folder, gif_filename, azimuth))
    

# load all the static images into a list then save as an animated gif
gif_filepath = '{}/{}.gif'.format(save_folder, gif_filename)
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(working_folder))]
gif = images[0]
gif.info['duration'] = 10 #milliseconds per frame
gif.info['loop'] = 0 #how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])
IPdisplay.Image(url=gif_filepath)