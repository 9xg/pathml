from pathml import slide
import pyvips as pv
import matplotlib.pyplot as plt
import numpy as np


left = 19000
top = 20000
width = 512
height = 512

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

demoSlidePath = '/media/gehrun01/work-io/cruk-phd-data/cytosponge/slides/BEST2/BEST2_CAM_0012/BEST2_CAM_0012_TFF3_1.svs'

demoImage = slide.Slide(demoSlidePath,verbose=True)
demoImage.setTileProperties(tileSize=512)
demoImage.detectForeground()
print(demoImage.getTileCount())



print(demoImage.tileMetadata[(40,50)])

#print(demoImage.tileMetadata[(0,2)])


#pvImage = pv.Image.new_from_file(demoSlidePath)
#print(demoImage.slide.width)
#img = demoImage.slide.extract_area(left,top,width,height)

#np_3d = np.ndarray(buffer=img.write_to_memory(),
#                   dtype=format_to_dtype[img.format],
#                   shape=[img.height, img.width, img.bands])
#plt.imshow(np_3d)
#plt.show()