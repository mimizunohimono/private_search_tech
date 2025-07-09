# private_search_tech
This repo stores some source code which realize PIR search.

## File constructions

- Docker/
  - Dockerfile
- src/
  - pir_test.py ()
  - images/ (Save your image files in images/)
    - image1
    - image2
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
$ python3 pir_test.py --mode gendb
# Create index.json
$ python3 pir_test.py --mode search --inp query.jpg --db index.json
```
