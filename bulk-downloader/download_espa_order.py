#!/usr/bin/env python

"""
Purpose: A simple python client that will download all available (completed) scenes for
         a user order(s).

Requires: Standard Python installation. (can also use requests)

Changes:

23 May 2018: Adam J. Stewart suggestion/adds new CLI options for retries and directory control
31 Jan 2017: Updated HTTPS support for python 2.7 series (allow using requests library)
20 June 2017: Woodstonelee added option to download checksum and error handling on bad urls
30 June 2016: Guy Serbin added support for Python 3.x and download progress indicators.
24 August 2016: Guy Serbin added:
1. The downloads will now tell you which file number of all available scenes is being downloaded.
2. Added a try/except clause for cases where the remote server closes the connection during a download.
23 September 2016: Converted to using the ESPA API proper rather than relying on the RSS feed

"""
import argparse
import base64
import os
import random
import shutil
import sys
import platform
import time
import json
import hashlib
import logging
from getpass import getpass

if sys.version_info[0] == 3:
    import urllib.request as ul
else:
    import urllib2 as ul

try:
    import requests
    logging.getLogger("urllib3").setLevel(logging.WARNING)
except ImportError:
    requests = None

__version__ = '2.2.5'
LOGGER = logging.getLogger(__name__)
USERAGENT = ('EspaBulkDownloader/{v} ({s}) Python/{p}'
             .format(v=__version__, s=platform.platform(aliased=True),
                     p=platform.python_version()))


class HTTPSHandler(object):
    """ Python standard library TLS-secured HTTP/REST communications """
    def _set_ssl_context(self):
        try:
            from ssl import SSLContext, PROTOCOL_TLSv1_2
        except ImportError:
            msg = ('Cannot import SSL, HTTPS will not work. '
                   'Try `pip install requests` on Python < 2.7.9')
            raise ImportError(msg)
        else:
            self.context = SSLContext(PROTOCOL_TLSv1_2)

    def __init__(self, host=''):
        self.host = host
        self._set_ssl_context()
        self.handler = ul.HTTPSHandler(context=self.context)
        self.opener = ul.build_opener(self.handler)
        self.opener.addheaders = [('User-Agent', USERAGENT)]

    def auth(self, username, password):
        auth_handler = ul.HTTPBasicAuthHandler()
        auth_handler.add_password(realm='Authentication Required',
                                  uri=self.host, user=username,
                                  passwd=password)
        self.opener = ul.build_opener(auth_handler, self.handler)
        self.opener.addheaders = [('User-Agent', USERAGENT)]

    def get(self, uri, data=None):
        body = (json.dumps(data) if data else '').encode('ascii')
        request = ul.Request(self.host + uri)
        request.get_method = lambda: 'GET'
        response = self.opener.open(request, data=body)
        return json.loads(response.read().decode())

    def _download_bytes(self, full_url, first_byte, tmp_scene_path):
        request = ul.Request(full_url)
        request.headers['Range'] = 'bytes={}-'.format(first_byte)

        with open(tmp_scene_path, 'ab') as target:
            source = self.opener.open(request)
            shutil.copyfileobj(source, target)

        return os.path.getsize(tmp_scene_path)

    def download(self, uri, target_path, verbose=False):
        request = ul.Request(self.host + uri)
        request.get_method = lambda: 'HEAD'

        head = self.opener.open(request)

        file_size = int(head.headers['Content-Length'])

        first_byte, tmp_scene_path = 0, target_path + '.part'
        if os.path.exists(tmp_scene_path):
            first_byte = os.path.getsize(tmp_scene_path)

        while first_byte < file_size:
            first_byte = self._download_bytes(self.host + uri, first_byte, tmp_scene_path)

        if first_byte >= file_size:
            os.rename(tmp_scene_path, target_path)
        return target_path


class RequestsHandler(object):

    def __init__(self, host=''):
        self.host = host
        self.creds = None
        self.headers = {}

    def auth(self, username, password):
        basic = ('%s:%s' % (username, password)).encode('ascii')
        self.creds = (username, password)
        self.headers = {
                'User-Agent': USERAGENT + ' (Requests/%s)' % requests.__version__
        }

    def get(self, uri, data=None):
        response = requests.get(self.host+uri, json=data,
                                headers=self.headers, auth=self.creds)
        response.raise_for_status()
        results = response.json()
        return results

    def download(self, uri, target_path, verbose=False):
        fileurl = self.host + uri

        head = requests.head(fileurl)

        file_size = None
        if 'Content-Length' in head.headers:
            file_size = int(head.headers['Content-Length'])

        first_byte, tmp_scene_path = 0, target_path + '.part'
        if os.path.exists(tmp_scene_path):
            first_byte = os.path.getsize(tmp_scene_path)

        self.headers.update({'Range': 'bytes=%d-' % first_byte})
        sock = requests.get(fileurl, headers=self.headers, stream=True)

        f = open(tmp_scene_path, 'ab')
        bytes_in_mb = 1024*1024
        for block in sock.iter_content(chunk_size=bytes_in_mb):
            if block:
                f.write(block)
        f.close()

        if os.path.getsize(tmp_scene_path) >= file_size:
            os.rename(tmp_scene_path, target_path)
        return target_path

class Api(object):
    def __init__(self, username, password, host):
        if requests:
            self.handler = RequestsHandler(host)
        else:
            self.handler = HTTPSHandler(host)
        self.handler.auth(username, password)

    def api_request(self, endpoint, data=None):
        """
        Simple method to handle calls to a REST API that uses JSON

        args:
            endpoint - API endpoint URL
            data - Python dictionary to send as JSON to the API

        returns:
            Python dictionary representation of the API response
        """
        resp = self.handler.get(endpoint, data)
        if isinstance(resp, dict):
            messages = resp.pop('messages', dict())
            if messages.get('errors'):
                raise Exception('{}'.format(messages.get('errors')))
            if messages.get('warnings'):
                LOGGER.warning('{}'.format(messages.get('warnings')))
        return resp

    def get_completed_scenes(self, orderid):
        filters = {'status': 'complete'}
        resp = self.api_request('/api/v1/item-status/{0}'.format(orderid),
                                data=filters)
        if orderid not in resp:
            raise Exception('Order ID {} not found'.format(orderid))
        urls = [_.get('product_dload_url') for _ in resp[orderid]]
        return urls

    def retrieve_all_orders(self, email):
        filters = {'status': 'complete'}
        all_orders = self.api_request('/api/v1/list-orders/{0}'.format(email or ''),
                                      data=filters)

        return all_orders

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Scene(object):

    def __init__(self, srcurl):
        self.srcurl = srcurl

        parts = self.srcurl.split("/")
        self.orderid = parts[4]
        self.filename = parts[-1]
        self.name = self.filename.split('.tar.gz')[0]

    @classmethod
    def checksum(cls):
        cls.srcurl = str(cls.srcurl).replace('.tar.gz', '.md5')
        cls.filename = cls.filename.replace('.tar.gz', '.md5')
        cls.name = '%s MD5 checksum' % cls.name
        return cls


class LocalStorage(object):

    def __init__(self, basedir, no_order_directories=False, verbose=False):
        self.basedir = basedir
        self.no_order_directories = no_order_directories
        self.verbose = verbose
        if requests:
            self.handler = RequestsHandler()
        else:
            self.handler = HTTPSHandler()

    def directory_path(self, scene):
        if self.no_order_directories:
            path = self.basedir
        else:
            path = os.path.join(self.basedir, scene.orderid)
        if not os.path.exists(path):
            os.makedirs(path)
            LOGGER.debug("Created target_directory: %s " % path)
        return path

    def scene_path(self, scene):
        return os.path.join(self.directory_path(scene), scene.filename)

    def is_stored(self, scene):
        return os.path.exists(self.scene_path(scene))

    def store(self, scene, checksum=False, retry=0):
        if self.is_stored(scene):
            LOGGER.debug('Scene already exists on disk, skipping.')
            return

        for tries in range(0, retry+1):
            LOGGER.debug("Downloading %s, to: %s" % (scene.name, self.directory_path(scene)))
            try:
                self.handler.download(scene.srcurl, self.scene_path(scene), self.verbose)
                if checksum:
                    scene = scene.checksum()
                    self.handler.download(scene.srcurl, self.scene_path(scene), self.verbose)
                return
            except Exception as exc:
                LOGGER.error('Scene not reachable at %s (%s)', scene.srcurl, exc)
                time.sleep(random.randint(2, 30))


def main(username, email, order, target_directory, password=None, host=None, verbose=False,
         checksum=False, retry=0, no_order_directories=False):
    if not username:
        raise ValueError('Must supply valid username')
    if not password:
        password = getpass('Password: ')
    if not host:
        host = 'https://espa.cr.usgs.gov'

    storage = LocalStorage(target_directory, no_order_directories)

    with Api(username, password, host) as api:
        if order == 'ALL':
            orders = api.retrieve_all_orders(email)
        else:
            orders = [order]

        LOGGER.debug('Retrieving orders: {0}'.format(orders))

        for o in orders:
            scenes = api.get_completed_scenes(o)
            if len(scenes) < 1:
                LOGGER.warning('No scenes in "completed" state for order {}'.format(o))

            for s in range(len(scenes)):
                LOGGER.info('File {0} of {1} for order: {2}'.format(s + 1, len(scenes), o))

                scene = Scene(scenes[s])
                storage.store(scene, checksum, retry)


if __name__ == '__main__':
    epilog = ('ESPA Bulk Download Client Version 1.0.0. [Tested with Python 2.7]\n'
              'Retrieves all completed scenes for the user/order\n'
              'and places them into the target directory.\n'
              'Scenes are organized by order.\n\n'
              'It is safe to cancel and restart the client, as it will\n'
              'only download scenes one time (per directory)\n'
              ' \n'
              '*** Important ***\n'
              'If you intend to automate execution of this script,\n'
              'please take care to ensure only 1 instance runs at a time.\n'
              'Also please do not schedule execution more frequently than\n'
              'once per hour.\n'
              ' \n'
              '------------\n'
              'Examples:\n'
              '------------\n'
              'Linux/Mac: ./download_espa_order.py -e your_email@server.com -o ALL -d /some/directory/with/free/space\n\n'
              'Windows:   C:\python27\python download_espa_order.py -e your_email@server.com -o ALL -d C:\some\directory\with\\free\space'
              '\n ')

    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-e", "--email",
                        required=False,
                        help="email address for the user that submitted the order)")

    parser.add_argument("-o", "--order",
                        required=False, default='ALL',
                        help="which order to download (use ALL for every order)")

    parser.add_argument("-d", "--target_directory",
                        required=True,
                        help="where to store the downloaded scenes")

    parser.add_argument("-u", "--username",
                        required=True,
                        help="EE/ESPA account username")

    parser.add_argument("-p", "--password",
                        required=False,
                        help="EE/ESPA account password")

    parser.add_argument("-v", "--verbose",
                        required=False,
                        action='store_true',
                        help="be vocal about process")

    parser.add_argument("-i", "--host",
                        required=False)

    parser.add_argument("-c", '--checksum',
                        required=False,
                        action='store_true',
                        help="download additional MD5 checksum files (will warn if binaries do not match)")

    parser.add_argument("-r", '--retry',
                        required=False,
                        type=int,
                        choices=range(1, 11),
                        default=0,
                        help="number of retry attempts for any files which fail to download")

    parser.add_argument('-n', '--no-order-directories',
                        required=False,
                        action='store_true',
                        help='disable generation of order-prefixed directories')


    parsed_args = parser.parse_args()

    log_level = 'DEBUG' if parsed_args.verbose else 'INFO'
    logging.basicConfig(level=log_level, format='%(asctime)s| %(message)s')
    try:
        main(**vars(parsed_args))
    except KeyboardInterrupt:
        LOGGER.error('User killed process')
    except Exception as exc:
        LOGGER.critical('ERROR: %s' % exc, exc_info=os.getenv('DEBUG', False))
