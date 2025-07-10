# private_search_tech
This repo stores some source code which realize PIR search.

## File constructions

- Docker/
  - Dockerfile
- src/
  - pir_test.py ()
  - images/ (Save your image files in images/)
    - image1.[jpg/jpeg/png]
    - image2.[jpg/jpeg/png]
    - ...
  - texts/ (Save your text files in texts/)
    - text1.[txt/md]
    - text2.[txt/md]
    - ...
  - image2vec.py (test)
  - vec2enc.py (test)

## How to use

### Build

```bash
$ docker build -f docker/Dockerfile -t your-image-name .
$ docker run -it --rm your-image-name
```

### Usage
```bash
## Image
$ python3 pir_test.py --mode gendb --type image
# => Create index.json
$ python3 pir_test.py --mode search --type image --inp query.jpg --db index.json

## Text
$ python3 pir_test.py --mode gendb --type text
# => Create index.json
$ python3 pir_test.py --mode search --type text --inp query.txt --db index.json

```
