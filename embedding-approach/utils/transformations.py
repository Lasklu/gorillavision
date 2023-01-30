from torchvision.transforms.functional import pad

class FillSizePad(object):
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