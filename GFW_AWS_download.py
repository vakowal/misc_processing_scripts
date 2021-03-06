"""Download Global Forest Watch data using Amazon Web Services.

Use this script to download the Global Forest Watch Aboveground Live
Woody Biomass Density dataset (
http://data.globalforestwatch.org/datasets/8f93a6f94a414f9588ce4657a39c59ff_1)
and Hansen forest loss dataset (
https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.5.html)
for coincidental areas to disk.
Ginger Kowal, March 2019, gkowal@stanford.edu

Prerequisites:
  - sign up for Amazon Web Services account
  - install boto3 and set credentials as described here:
  https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html
"""

import os
import re

import boto3
import urllib2
import pandas


# local disk location to store downloaded biomass rasters
alwbd_dir = 'C:/Users/ginge/Desktop/GFW_ALWBD_2000'
if not os.path.exists(alwbd_dir):
    os.makedirs(alwbd_dir)

# local disk location to store forest loss rasters
loss_dir = 'C:/Users/ginge/Desktop/Hansen_lossyear'
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


def download_aboveground_biomass_data():
    """Get biomass data from Amazon Web Services."""
    # I sussed these from the tile download links on GFW data portal
    GFW_bucket_name = 'gfw2-data'
    ALWBD_prefix = 'climate/WHRC_biomass/WHRC_V4/Processed'

    bucket_info = boto3.client('s3')
    download_resource = boto3.resource('s3')
    aws_objects = bucket_info.list_objects_v2(
        Bucket=GFW_bucket_name, Prefix=ALWBD_prefix)
    num_objects = len(aws_objects['Contents'])
    current_object = 1
    for obj in aws_objects['Contents'][0:20]:
        key = obj['Key']
        tile_name = os.path.basename(key)
        target_path = os.path.join(alwbd_dir, tile_name)
        if not os.path.exists(target_path) and len(tile_name) > 0:
            print "downloading object {} of {}".format(
                current_object, num_objects)
            download_resource.Bucket(
                GFW_bucket_name).download_file(key, target_path)
        current_object += 1


def get_forest_loss_data(hansen_url_csv_path):
    """Get forest loss data from Hansen dataset.

    Forest loss data available here:
    https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.5.html

    Retrieve only the granules that coincide with ALWBD data
    according to the file name.
    """
    alwbd_files = [f for f in os.listdir(alwbd_dir) if f.endswith('.tif')]
    hansen_url_list = pandas.read_csv(
        hansen_url_csv_path, header=None, names=['url'])
    hansen_base_url = 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2017-v1.5/Hansen_GFC-2017-v1.5_lossyear_<loc_string>.tif'
    current_object = 1
    downloaded = 0
    for biomass_file in alwbd_files:
        loc_string = biomass_file[:8]
        hansen_url = hansen_base_url.replace('<loc_string>', loc_string)
        if hansen_url in hansen_url_list['url'].values:
            print "grabbing object {} of 280".format(current_object)
            local_destination = os.path.join(
                loss_dir, hansen_url[72:])
            if os.path.isfile(local_destination):
                print "file {} already exists".format(os.path.basename(
                    local_destination))
                continue
            hansen_data = urllib2.urlopen(hansen_url).read()
            with open(local_destination, 'wb') as f:
                f.write(hansen_data)
            downloaded += 1
            current_object += 1
    print "downloaded {} files".format(downloaded)


if __name__ == '__main__':
    # I made this csv on my local machine, containing  the url paths of each
    # tile in the Hansen biomass loss dataset, from a txt file available at:
    # https://storage.googleapis.com/earthenginepartners-hansen/GFC-2017-v1.5/lossyear.txt
    hansen_url_csv_path = 'C:/Users/ginge/Desktop/lossyear.csv'

    download_aboveground_biomass_data()
    get_forest_loss_data(hansen_url_csv_path)
