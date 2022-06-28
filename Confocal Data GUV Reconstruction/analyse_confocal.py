#   python script for analysing guvs from confocal microscopy data
#   this script takes in czi files in 'TCZYX' format and fits an
#   ellipse on guvs in a semi-automatic fashion using user input
#   to create a bounding box around the interest area. make sure
#   necessary modules are installed. data will be saved in a csv
#   file with appropriate name. -jsama

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
from aicspylibczi import CziFile
from scipy.signal import find_peaks
from scipy import linalg
#input file name
file_name = input('\nPlease enter the name of the file to be \
analysed below ...\n')
#read czi file
czi = CziFile(file_name)
#reject if wrong format (must be 'TCZYX' and monochromatic)
if czi.dims != 'TCZYX' or czi.get_dims_shape()[0]['C'][1] != 1:
    print('\nFile format incorrect!\nformat must be TCZYX')
    exit()
#file to save data
data_file = open(file_name + '_analysis.csv', 'w')
data_file.write(file_name)
#define functions
#clahe object for filter
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3,3))
def apply_filters(array):
    #clahe to enhance contrast
    array = clahe.apply(array)
    #blur to smooth noise
    array = cv2.blur(array, ksize=(7,7))
    #gaussian for smoother peaks
    array = cv2.GaussianBlur(array, ksize=(7,7), sigmaX=5, sigmaY=5)
    #threshold to reject dim pixels
    _, array = cv2.threshold(array,55,255, cv2.THRESH_TOZERO)
    return array
def detect_peaks(image_array):
    #image center
    xi = image_array.shape[1]//2
    yi = image_array.shape[0]//2
    pixel_id = []
    #id each pixel
    for j in range(image_array.shape[0]): #scan down
        for i in range(image_array.shape[1]): #scan accross
            if image_array[j,i] != 0: #intensity not zero
                theta = np.angle((i - xi) + 1j*(j - yi)) #get angle
                #store coords, angle, intensity
                pixel_id.append(np.array([i, j,\
                    int(round(np.rad2deg(theta),0)), image_array[j,i]]))
    #convert to array
    pixel_id = np.array(pixel_id)
    peak_vl = [] #store peaks
    for i in np.arange(-180,181,1): #scan angles
        pixel_angle = pixel_id[pixel_id[:,2] == i,:] #at this particular angle
        #sort data with radial distance
        pixel_angle = pixel_angle[np.argsort(\
            np.sqrt((pixel_angle[:,1] - yi)**2 + (pixel_angle[:,0] - xi)**2))]
        #find signal peak
        peak_id, _ = find_peaks(pixel_angle[:,3], width=4)
        #store first peak coords, intensity
        if peak_id.size != 0 :
            peak = pixel_angle[peak_id[0]]
            peak_vl.append(np.array([peak[0],peak[1],peak[3]]))
    #output peak coords and intensity
    return np.array(peak_vl, dtype=int)
def direct_ellipse(xy):
    #direct implimentation of algorithm
    centroid = np.mean(xy, axis = 0)
    D1 = np.vstack([(xy[:,0]-centroid[0])**2, 
                    (xy[:,0]-centroid[0])*(xy[:,1]-centroid[1]), 
                    (xy[:,1]-centroid[1])**2]).T
    D2 = np.vstack([(xy[:,0]-centroid[0]), 
                    (xy[:,1]-centroid[1]),
                    np.ones(xy.shape[0])]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    _, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    a1 = eigvec[:, np.nonzero(con > 0)[0]]
    A =  np.concatenate((a1, T @ a1)).ravel()
    A3 = A[3]-2*A[0]*centroid[0]-A[1]*centroid[1]
    A4 = A[4]-2*A[2]*centroid[1]-A[1]*centroid[0]
    A5 = A[5]+A[0]*centroid[0]**2+A[2]*centroid[1]**2\
        +A[1]*centroid[0]*centroid[1]-A[3]*centroid[0]\
        -A[4]*centroid[1]
    A[3] = A3
    A[4] = A4
    A[5] = A5
    A = A/np.linalg.norm(A)
    #convert algebric parameters to polar
    a = A[0]
    b = A[1] / 2
    c = A[2]
    d = A[3] / 2
    f = A[4] / 2
    g = A[5]
    den = b**2 - a*c
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        phi += np.pi/2
    phi = phi % np.pi
    return x0, y0, ap, bp, phi
#main callback function
def main(event, x, y, flags, params):
    #global variables
    global ix, iy, jx, jy, draw, imgv, imgs, z, x0, y0, a, b, phi
    #draw rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        ix = x
        iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            imgv = imgc.copy()
            cv2.rectangle(imgv, pt1=(ix,iy), pt2=(x,y), color=(0,255,255),
            thickness=2)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        imgv = imgc.copy()
        cv2.rectangle(imgv, pt1=(ix,iy), pt2=(x,y), color=(0,255,255),
            thickness=2)
        jx, jy = x, y
    #draw elliptical fit
    elif event == cv2.EVENT_MBUTTONDOWN:
        imgv = imgc.copy()
        ix,jx = sorted((ix,jx))
        iy,jy = sorted((iy,jy))
        peak_vl = detect_peaks(apply_filters(imgs[iy:jy,ix:jx]))
        if peak_vl.shape[0] != 0:
            x = peak_vl[:,0]
            y = peak_vl[:,1]
            x0, y0, a, b, phi = direct_ellipse(np.vstack((x,y)).T)
            x0 += ix
            y0 += iy
            imgv = cv2.ellipse(imgv, (int(round(x0,0)),
                                        int(round(y0,0))), 
                                        (int(round(a,0)), 
                                        int(round(b,0))), 
                                        np.rad2deg(phi), 
                                        0, 
                                        360, 
                                        (0,0,255), 2)
    #save data
    elif event == cv2.EVENT_RBUTTONDOWN:
        imgv = cv2.putText(imgv, 
        "written frame : %i, x0 : %i, y0 : %i, a : %i, b : %i, phi : %i,\
            to memory"%(z+1, x0, y0, a, b, phi), 
        (X//10,Y//10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5,
        (0,255,0),
        1,
        cv2.LINE_AA)
        print('\nData saved for slice : ' + str(z))
        data_file.write("\n%i,%i,%i,%f,%f,%f"%(z+1, x0, y0, a, b, phi))
#parameters
shape = czi.get_dims_shape()[0]
Z = shape['Z'][1]
X = shape['X'][1]
Y = shape['Y'][1]
#start analysis
#write big guv data
print('\nAnalysing big guv')
data_file.write('\nbig guv')
data_file.write('\nframe,x0,y0,a,b,phi')
#iterate different frames
for z in range(Z):
    #sytax to read files
    img, _ = czi.read_image(Z = z)
    img = img[0,0,0,:,:]
    #make bitdepth 8bit
    imgs = np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX))
    imgc = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
    #initialize rectangle variables
    ix = -1
    iy = -1
    jx = -1
    jy = -1
    #initialize draw variable
    draw = False
    #copy image to display
    imgv = imgc.copy()
    #name window
    window_name = 'big guv select, slice : ' + str(z)
    cv2.namedWindow(window_name)
    #set callback
    cv2.setMouseCallback(window_name, main)
    #wait for escape key the kill windows
    while True:
        cv2.imshow(window_name, imgv)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()
#write small guv data
#comment out if not required
print('\nAnalysing small guv')
data_file.write('\nsmall guv')
data_file.write('\nframe,x0,y0,a,b,phi')
#iterate different frames
for z in range(Z):
    #sytax to read files
    img, _ = czi.read_image(Z = z)
    img = img[0,0,0,:,:]
    #make bitdepth 8bit
    imgs = np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX))
    imgc = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
    #initialize rectangle variables
    ix = -1
    iy = -1
    jx = -1
    jy = -1
    #initialize draw variable
    draw = False
    #copy image to display
    imgv = imgc.copy()
    #name window
    window_name = 'small guv select, slice : ' + str(z)
    cv2.namedWindow(window_name)
    #set callback
    cv2.setMouseCallback(window_name, main)
    #wait for escape key the kill windows
    while True:
        cv2.imshow(window_name, imgv)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()
#end program
data_file.close()
print('\nAnalysis finished!')