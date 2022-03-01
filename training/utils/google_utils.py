# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage

import os
import time


# from google.cloud import storage

def gsutil_getsize(url=''):   # copy from yolov3
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(file, repo='ultralytics/yolov3'):   # copy from yolov3
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", '').lower())

    if not file.exists():
        # try:
        #     response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
        #     assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
        #     tag = response['tag_name']  # i.e. 'v1.0'
        # except:  # fallback plan
        assets = ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']
        tag = subprocess.check_output('git tag', shell=True).decode().split()[-1]

        name = file.name
        if name in assets:
            msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
            redundant = False  # second download option
            try:  # GitHub
                url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
                print(f'Downloading {url} to {file}...')
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1E6  # check
            except Exception as e:  # GCP
                print(f'Download error: {e}')
                assert redundant, 'No secondary mirror'
                url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                print(f'Downloading {url} to {file}...')
                os.system(f'curl -L {url} -o {file}')  # torch.hub.download_url_to_file(url, weights)
            finally:
                if not file.exists() or file.stat().st_size < 1E6:  # check
                    file.unlink(missing_ok=True)  # remove partial downloads
                    print(f'ERROR: Download failure: {msg}')
                print('')
                return


def gdrive_download(id='1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO', name='coco.zip'):
    # https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    if os.path.exists(name):  # remove existing
        os.remove(name)

    # Attempt large file download
    s = ["curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=%s\" > /dev/null" % id,
         "curl -Lb ./cookie -s \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (
             id, name),
         'rm ./cookie']
    r = sum([os.system(x) for x in s])  # run commands, get return zeros

    # Attempt small file download
    if not os.path.exists(name):  # file size < 40MB
        s = 'curl -f -L -o %s https://drive.google.com/uc?export=download&id=%s' % (name, id)
        r = os.system(s)

    # Error check
    if r != 0:
        os.system('rm ' + name)  # remove partial downloads
        print('ERROR: Download failure ')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    # Uploads a file to a bucket
    # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


def download_blob(bucket_name, source_blob_name, destination_file_name):
    # Uploads a blob from a bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
