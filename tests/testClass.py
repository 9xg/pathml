import sys
sys.path.append("/home/gehrun01/Desktop")
from multiprocessing import Pool

from pathml import slide
import pyvips as pv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


left = 19000
top = 20000
width = 512
height = 512


demoSlidePath = '/media/gehrun01/work-io/cruk-phd-data/cytosponge/slides/BEST2/BEST2_CAM_0012/BEST2_CAM_0012_TFF3_1.svs'

demoImage = slide.Slide(demoSlidePath,verbose=True)
demoImage.setTileProperties(tileSize=100)
#demoImage.detectForeground()
#print(demoImage.getTileCount())
#print(demoImage.tileMetadata.keys())
#
for tile in tqdm(demoImage.iterateTiles()):
    pass


numbers = demoImage.tileMetadata.keys()
pool = Pool(processes=8)
print(pool.map(demoImage.getTile, numbers))

#print(demoImage.tileMetadata[(0,2)])


#pvImage = pv.Image.new_from_file(demoSlidePath)
#print(demoImage.slide.width)
#img = demoImage.slide.extract_area(left,top,width,height)

#np_3d = np.ndarray(buffer=img.write_to_memory(),
#                   dtype=format_to_dtype[img.format],
#                   shape=[img.height, img.width, img.bands])
#plt.imshow(np_3d)
#plt.show()
