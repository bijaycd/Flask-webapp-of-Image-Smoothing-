import numpy as np

class SimpleAveragingSmooth:
    
    # This function returns smoothend image by performing Simple average smoothening provided image, window size
    def smooth(self, img, window):
        m, n = img.shape

        # checking if window size is greater than 1 and less than minimum of M X N
        if  window < 1 or window > min(m,n): 
            return 0
        
        # returning same image if window size = 1
        if window == 1:
            return img
        
        # zero padding
        pad_num = window//2
        pad_img = np.pad(img, (pad_num,), 'constant', constant_values=(0))
        
        pad_m,pad_n = pad_img.shape

        img_smoothened = np.zeros([m, n])

        for i in range(pad_num, pad_m-(pad_num)):
            for j in range(pad_num, pad_n-(pad_num)):
                k = -1*(window//2)
                temp = 0
                cnt = 0
                while(k <= window//2):
                    l = -1*(window//2)
                    while(l <= window//2):
                        temp += pad_img[i+k, j+l]
                        cnt += 1
                        l += 1
                    k += 1
                temp = round(temp / cnt)
                img_smoothened[i-pad_num, j-pad_num] = temp
                
        return img_smoothened