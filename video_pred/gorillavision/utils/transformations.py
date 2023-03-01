from torchvision.transforms.functional import pad, resize

class FillSizePad(object):
    # Fill image to given size with padding
    def __init__(self, img_size, fill=0, padding_mode='constant', ):
        self.fill = fill
        self.padding_mode = padding_mode
        self.img_size = img_size
    
    def get_padding(self, img):
        imsize = img.size
        h_padding = (self.img_size[0] - imsize[0]) / 2
        v_padding = (self.img_size[1] - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        
        return (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

    def __call__(self, img):
        return pad(img, self.get_padding(img), self.fill, self.padding_mode)

class EnsureSize(object):
    def __init__(self, img_size, ):
        self.img_size = img_size

    def __call__(self, img):
        imsize = img.size
        if imsize[0] < self.img_size[0] and imsize[1] < self.img_size[1]:
            return resize(img, self.img_size)

        new_height = 0
        new_width = 0
        # if width  is longest side
        if imsize[0] > imsize[1]:
            new_width = self.img_size[0]
            factor = (imsize[0] - self.img_size[0]) / imsize[0]
            new_height = imsize[1] - factor*imsize[1]
        else:
            new_height = self.img_size[1]
            factor = (imsize[1] - self.img_size[1]) / imsize[1]
            new_width = imsize[0] - factor*imsize[0]

        return resize(img, (int(new_height), int(new_width)))
