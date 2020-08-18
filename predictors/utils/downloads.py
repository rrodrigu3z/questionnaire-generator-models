"""Utility functions for performing file downloads"""

import requests
import tqdm
import os.path


def download_file(url, filename=False, verbose=False):
    """Download file with progressbar

    Taken from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603

    Args:
        url: URL of the file to download.
        filename: Name for the output file (optional)
        verbose: Show file size info (optional).
    """
    if filename:
        local_filename = filename
    else:
        local_filename = os.path.join(".", url.split('/')[-1])

    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk = 1
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)

    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    with open(local_filename, 'wb') as fp:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size),
                               total=num_bars,
                               unit='KB',
                               desc=local_filename,
                               leave=True):
            fp.write(chunk)
    return
