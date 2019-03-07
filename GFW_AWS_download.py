"""Download Global Forest Watch data using Amazon Web Services.

Use this script to download the Global Forest Watch Aboveground Live
Woody Biomass Density dataset to disk.
Dataset described here:
http://data.globalforestwatch.org/datasets/8f93a6f94a414f9588ce4657a39c59ff_1
Ginger Kowal, March 2019, gkowal@stanford.edu

Prerequisites:
  - sign up for Amazon Web Services account
  - install boto3 and set credentials as described here:
  https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html
"""

import os

import boto3

# I sussed these from the tile download links on GFW data portal
GFW_bucket_name = 'gfw2-data'
ALWBD_prefix = 'climate/WHRC_biomass/WHRC_V4/Processed'

# local disk location to store downloaded data
local_directory = 'F:/GFW_ALWBD_2000'

if __name__ == '__main__':
    bucket_info = boto3.client('s3')
    download_resource = boto3.resource('s3')
    aws_objects = bucket_info.list_objects_v2(
        Bucket=GFW_bucket_name, Prefix=ALWBD_prefix)
    num_objects = len(aws_objects['Contents'])
    current_object = 1
    for obj in aws_objects['Contents']:
        key = obj['Key']
        tile_name = os.path.basename(key)
        target_path = os.path.join(local_directory, tile_name)
        if not os.path.exists(target_path) and len(tile_name) > 0:
            print "downloading object {} of {}".format(
                current_object, num_objects)
            download_resource.Bucket(
                GFW_bucket_name).download_file(key, target_path)
        current_object += 1
