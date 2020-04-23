import argparse
import os
import requests

def parse_mime_type(mime):
    return mime.split('/')[-1]

def create_output_dir(dir):
    if dir[-1] != '/':
        dir += '/'

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir

def download_image(url, directory_name, index):
    r = requests.get(url)
    image = r.content
    image_type = parse_mime_type(r.headers['Content-Type'])
    with open(f'{directory_name}/{index}.{image_type}', 'wb') as f:
        f.write(image)

def main():
    parser = argparse.ArgumentParser(description='Download imagse from a file containing a list of URLs')
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    dir = create_output_dir(args.output)

    with open(args.input) as f:
        urls = list(filter(lambda x: len(x.strip()) > 0, f.read().split('\n')))
        for i, url in enumerate(urls):
            download_image(url, dir, i)

if __name__ == '__main__':
    main()