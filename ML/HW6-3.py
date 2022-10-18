#!/usr/bin/env python
# coding: utf-8

# In[6]:


#image processing
from PIL import Image
from io import BytesIO
import webcolors

# data analysis
import math
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from importlib import reload
from mpl_toolkits import mplot3d
import seaborn as sns

#modeling
from sklearn.cluster import KMeans


# In[2]:


pip install webcolors


# In[7]:


img = Image.open('cat.jpg')
w, h = img.size
ori_px = np.array(img.getdata()).reshape(h, w,-1)
plt.imshow(ori_px)


# In[8]:


ori_px


# In[49]:


X = np.array(img.getdata()).reshape(h*w,-1)
#ori_px = X.reshape(*ori_img.size, -1)
ori_px


# In[57]:


def imageByteSize(img):
    img_file = BytesIO()
    image = Image.fromarray(np.uint8(img))
    image.save(img_file, 'png')
    return img_file.tell()/1024
ori_img_size = imageByteSize(ori_img)
ori_img_n_colors = len(X)


# In[58]:


ori_img_total_var = sum(np.linalg.norm(X - np.mean(X, axis=0), axis=1)**2)


# In[100]:


kmeans = KMeans(n_clusters = 10, n_jobs = -1, random_state = 123).fit(X)
kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns = ['Red', 'Green', 'Blue'])


# In[101]:


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0])**2
        gd = (g_c - requested_color[1])**2
        bd = (b_c - requested_color[2])**2
        min_colors[(rd+gd+bd)] = name
    return min_colors[min(min_colors.keys())]
def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
    return closest_name
kmeans_df["Color Name"] = list(map(get_color_name, np.uint8(kmeans.cluster_centers_)))
kmeans_df


# In[102]:


def replaceWithCentroid(kmeans):
    new_pixels = []
    for label in kmeans.labels_:
        pixel_as_centroid = list(kmeans.cluster_centers_[label])
        new_pixels.append(pixel_as_centroid)
    new_pixels = np.array(new_pixels).reshape(*ori_img.size, -1)
    return new_pixels
new_pixels = replaceWithCentroid(kmeans)


# In[103]:


def plotImage(img_array, size):
    reload(plt)
    plt.imshow(np.array(img_array/255).reshape(*size))
    plt.axis('off')
    return plt
plotImage(new_pixels, ori_px.shape).show()


# In[ ]:





# In[ ]:





# In[ ]:




