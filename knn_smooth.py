import numpy as np
from queue import PriorityQueue


class KNNSmooth:    
    # This function returns smoothend image by performing KNN smoothening provided image, window size, k
    def smooth(self, img, window, k):        
        m,n = img.shape        
        # checking if window size is greater than 1 and less than minimum of M X N
        if  window < 1 or window > min(m,n): 
            return 0
        
        # checking if k is less than window size and k is greater than 1
        if k < 1 or k > window*window:
            return 0
        
        # checking if window size is equal to 1 then k should be 0
        if window == 1 and k != 0:
            return 0
        
        # returning same image if window size = 1
        if window == 1:
            return img
        
        # zero padding
        pad_num = window//2
        pad_img = np.pad(img, (pad_num,), 'constant', constant_values=(0))        
        pad_m,pad_n = pad_img.shape
        img_n = np.zeros([m, n])

        for i in range(pad_num, pad_m-(pad_num)):
            for j in range(pad_num, pad_n-(pad_num)):
                temp_arr = []
                x = -1*(window//2)
                center_val = pad_img[i,j]
                while(x <= window//2):
                    y = -1*(window//2)
                    while(y <= window//2):
                        temp_arr.append(pad_img[i+x, j+y])
                        y += 1
                    x += 1
                knn_arr = []
                pix_val = pad_img[i,j]
                knn_arr.append(pix_val)
                pq = PriorityQueue()
                for i1 in range(len(temp_arr)):
                    pq.put((abs(temp_arr[i1]-pix_val), i1))
                for i1 in range(k):
                    p, pi = pq.get()
                    knn_arr.append(temp_arr[pi])
                img_n[i-pad_num, j-pad_num] = round(np.mean(knn_arr))

        return img_n