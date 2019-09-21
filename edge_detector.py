import cv2 as cv
import argparse
import numpy as np

# parser = argparse.ArgumentParser(
#         description='This sample shows how to define custom OpenCV deep learning layers in Python. '
#                     'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
#                     'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
# parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
# parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
# parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
# parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
# parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
# args = parser.parse_args()

#! [CropLayer]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
#! [CropLayer]

#! [Register]
cv.dnn_registerLayer('Crop', CropLayer)
#! [Register]

# Load the model.
net = cv.dnn.readNet(cv.samples.findFile(r"C:\Users\mail\PycharmProjects\captcha\hed-edge-detector-master\deploy.prototxt"), cv.samples.findFile(r"C:\Users\mail\PycharmProjects\captcha\hed-edge-detector-master\hed_pretrained_bsds.caffemodel"))



image=cv.imread(r"C:\Users\mail\Pictures\nafi\XLX6.png")
h_ori, w_ori, channels = image.shape
w = 500
h = 500
image=cv.resize(image,(h,w))

inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),

                           mean=(104.00698793, 116.66876762, 122.67891434),

                           swapRB=False, crop=False)

net.setInput(inp)
out = net.forward()
out = out[0, 0]

#out = cv.resize(out, (image.shape[1], image.shape[0]))
out = cv.resize(out, (w_ori, h_ori))

print(out.shape)
import matplotlib.pyplot as plt
plt.imshow(out)
out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)

out = 255 * out

out = out.astype(np.uint8)

#con=np.concatenate((image,out),axis=1)

cv.imwrite('out.jpg',out)
plt.imshow(out)
