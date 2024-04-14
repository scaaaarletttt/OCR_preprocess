import subprocess
import numpy as np
import cv2
import imread from cv2
import OcropusNormalize as ocropus_norm
import PageDewarp as page_dewarp


class PreProcess:
    """Class for pre-processing the image """

    def __init__(self):
        pass

    def execute_command(self, command):
        """
        Executes the terminal command passed to it and return output and error messages shown in
         terminal output
        :param command: terminal command
        :return: terminal output and error text
        """
        sp = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        out, err = sp.communicate()
        return out, err

    def convert_pdf2image(self, pdf_file_path, out_img_dir, dpi=300):
        """
        converts the pdf file to image with the specified dpi
        :param pdf_file_path: path of input pdf file
        :param out_img_dir: path for saving the output images
        :param dpi: dpi of output image. Default value is 300
        :return:
        """
        cvt_command = "convert -density " + str(dpi) + " " + pdf_file_path + " " + out_img_dir + "my3-%04d.png"
        out, err = self.execute_command(command=cvt_command)
        return out, err

    def convert_img_dpi(self, in_img_path, out_img_path, dpi=300):
        """
        converts the input image to an image with the specified dpi
        :param in_img_path: path of input image file
        :param out_img_path: path of output image file
        :param dpi: dpi of output image. Default value is 300
        :return:
        """
        cvt_command = "convert -units PixelsPerInch " + in_img_path + " -density " + str(dpi) + " " + out_img_path
        out, err = self.execute_command(command=cvt_command)
        return out, err

    def apply_preprocess(self, in_img, apply_dewarp=False, apply_deskew=True):
        """
        applies Tesseract operations to the input image
        :param in_img: input image
        :param apply_dewarp: a flag for determining applying dewarp or not. Default value is False
        :param apply_deskew: a flag for determining applying deskewing or not. Default value is True.
        :return: preprocessed image
        """
        if apply_dewarp:
            page_dewarp_obj = page_dewarp.PageDewarp()
            in_img = page_dewarp_obj.apply_dewarp(in_img)
        lo = 5  # percentile for black estimation
        hi = 90  # percentile for white estimation
        if in_img.shape.__len__() == 3:
            in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        ocropus_norm_obj = ocropus_norm.OcropusNormalize()
        res_img = ocropus_norm_obj.process(in_img.astype(np.float64), lo, hi, apply_deskew)
        return res_img
