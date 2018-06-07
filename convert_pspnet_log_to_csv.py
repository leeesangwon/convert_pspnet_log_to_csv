"""
Convert pspnet log to csv file 
"""

import csv
from collections import namedtuple
import re

IoU_line = namedtuple("IoU_line", ["class_index", "class_name", "iou"])

def convert_pspnet_log_to_csv(pspnet_log_path, csv_path):
    with open(pspnet_log_path) as f:
        pspnet_log = f.read()
    iou_lines = get_iou_lines(pspnet_log)
    mean_iou = get_mean_iou(pspnet_log)
    pixel_wise_accuracy = get_pixel_wise_accuracy(pspnet_log)

    write_csv_file(csv_path, iou_lines, mean_iou, pixel_wise_accuracy)

def get_iou_lines(pspnet_log):    
    re_iou_line = re.compile(r'\n\s{0,2}\d{1,3}\s+[\w\s]+: 0\.\d{4}')

    lines = re_iou_line.finditer(pspnet_log)
    iou_lines = []
    for line in lines:
        line = line.group()
        class_index = get_class_index(line)
        class_name = get_class_name(line)
        iou = get_iou(line)
        iou_line = IoU_line(class_index=class_index, class_name=class_name, iou=iou)
        iou_lines.append(iou_line)
    
    return iou_lines

def get_class_index(line):
    re_class_index = re.compile(r'\n\s{0,2}\d{1,3}')
    return int(re_class_index.search(line).group().replace('\n', ''))

def get_class_name(line):
    re_class_name = re.compile(r'[a-zA-Z][a-zA-Z\s]*:')
    return re_class_name.search(line).group().replace(':', '')

def get_iou(line):
    re_iou = re.compile(r'[01]\.\d{4}')
    return float(re_iou.search(line).group())


def get_mean_iou(pspnet_log):
    re_miou = re.compile(r'Mean IoU over \d+ classes: [01]\.\d{4}')
    miou_line = re_miou.search(pspnet_log).group()
    mean_iou = get_iou(miou_line)
    return mean_iou

def get_pixel_wise_accuracy(pspnet_log):
    re_pixel_wise_accuracy_line = re.compile(r'Pixel-wise Accuracy: \d\d\.\d\d%')
    pixel_wise_accuracy_line = re_pixel_wise_accuracy_line.search(pspnet_log).group()
    re_pixel_wise_accuracy = re.compile(r'\d\d\.\d\d')
    pixel_wise_accuracy = float(re_pixel_wise_accuracy.search(pixel_wise_accuracy_line).group())
    return pixel_wise_accuracy

def write_csv_file(csv_path, iou_lines, mean_iou, pixel_wise_accuracy):
    title_row = ['mean_iou', 'pixel_wise_accuracy']
    value_row = [mean_iou, pixel_wise_accuracy]
    for iou_line in iou_lines:
        title_row.append(iou_line.class_name)
        value_row.append(iou_line.iou)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title_row)
        writer.writerow(value_row)

if __name__=='__main__':
    convert_pspnet_log_to_csv('pspnet50_ADE20K_1-150.log', csv_path='result.csv')