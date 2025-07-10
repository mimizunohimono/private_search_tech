#!/usr/bin/env python3

## Import for FAISS
import os
import cv2
import numpy as np
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
# from torchvision.models import ResNet50_Weights
from sentence_transformers import SentenceTransformer
from PIL import Image

## Import for TenSEAL
import tenseal as ts

## Others
import argparse
import json
import ast
from typing import List, Tuple

## Parameters
IMAGE_DIR = "images"
TEXT_DIR = "texts"
JSON_PATH = "index.json"
TXT_EXT_LIST = (".txt", ".dat", ".md")
IMG_EXT_LIST = (".jpg", ".jpeg", ".png")

## Functions

def parse_args():
  parser = argparse.ArgumentParser(description="Options: --mode, --inp, --db --type")

  parser.add_argument(
    "--mode",
    required=True,
    choices=["gendb", "search"],
    help="Please choose a mode during gendb or search "
  )
  parser.add_argument("--inp", default="query.jpg", help="[gendb requirements] query input file")
  parser.add_argument("--db", default="index.json", help="[gendb requirements] database input file")
  parser.add_argument("--type", default="image", help="Data type: image(default) or text")

  args = parser.parse_args()


  if args.mode == "search":
    ## If --mode == search, you SHOULD set inp and db 
    if not args.inp or not args.db:
      parser.error("If --mode == search, you SHOULD set --inp and --db")  
    ## If...
    ##  -type == image and --inp == hoge.txt ==> Error
    ##  -type == text and --inp == hoge.jpg ==> Error
    
    if args.type == "image" and (os.path.splitext(args.inp)[-1] in TXT_EXT_LIST):
      parser.error("If --type == image, you SHOULD take a png/jpg/jpeg file as input in --inp")
    if args.type == "text" and (os.path.splitext(args.inp)[-1] in IMG_EXT_LIST):
      parser.error("If --type == text, you SHOULD take a txt/dat/md file as input in --inp")
    
  return args

def extract_features(model, transform, image_path):
  image = Image.open(image_path).convert('RGB')
  img_tensor = transform(image).unsqueeze(0)  
  # shape: (1, 3, 224, 224)
  with torch.no_grad():
    features = model(img_tensor).squeeze().numpy()  # shape: (2048,)
  return features / np.linalg.norm(features)  # Normalize

def save_data_index(model, transform, type):

  if type == "image":
    ## Image2Vec
    ## Input dir
    image_dir = IMAGE_DIR

    ## Output list
    features_list = []
    image_paths = []

    for fname in os.listdir(image_dir):
      if fname.lower().endswith(IMG_EXT_LIST):
        path = os.path.join(image_dir, fname)
        feat = extract_features(model, transform, path)
        feat_f32 = feat.astype('float32')
        features_list.append(feat_f32)
        image_paths.append(path)
        print(f"path:{path}, index:{feat_f32}")

    ## JSONnize
    json_data = [{"filename": f_name, "index": idx.tolist()} for f_name, idx in zip(image_paths, features_list)]
  
  else:
    ## Text2Vec
    files = []
    filename_list = []
    emb_list = []
    for fname in os.listdir(TEXT_DIR):
      path = os.path.join(TEXT_DIR, fname)
      if os.path.isfile(path) and os.path.splitext(fname)[-1] in TXT_EXT_LIST:
        with open(path, "r", encoding="utf-8") as f:
          content = f.read()
          emb = model.encode(content, convert_to_numpy=True, normalize_embeddings=True)
          filename_list.append(fname)
          emb_list.append(emb)
    ## JSONnize
    json_data = [{"filename": f_name, "index": idx.tolist()} for f_name, idx in zip(filename_list, emb_list)]
    
    ## For debug
    # for f, c in zip(filenames, contents):
    #   print(f, len(c))
  
  ## Save json file
  with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
  print(f"Saved in {JSON_PATH} !")

def search_from_db(model, transform, query, db):

  ## 1. Client Side
  ## Extract feature from query
  print()
  print("Step1. Client Side, Embedding and Encryption")
  q_feat = extract_features(model, transform, query)
  q_feat_f32 = q_feat.astype('float32')
  print(f"{query}'s index: {q_feat_f32}")

  ## Encrypt index
  def create_context():
    context = ts.context(
      ts.SCHEME_TYPE.CKKS,
      poly_modulus_degree=8192,
      coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context
  context = create_context()
  target_input = q_feat_f32
  enc_vec = ts.ckks_vector(context, target_input)
  print(f"Encrypted query: {enc_vec}")
  ## Save encrypted data
  with open("vector.tenseal", "wb") as f:
    f.write(enc_vec.serialize())
  print("Send this query to Server...")
  print()

  ## Server Side
  print("Step2. Server Side, Database construction")
  print("Already done.")
  print()

  print("Step3. Server Side, Query matching (Computing cossim)")
  ## Parse from db
  with open(db, "r", encoding="utf-8") as f:
    data = json.load(f)
  filenames = [entry["filename"] for entry in data]
  indexes = [entry["index"] for entry in data]
  print(f"Database: {filenames}")
  
  ## Matching
  enc_dot_list = []
  for idx in indexes:
    enc_dot = enc_vec.dot(idx)
    enc_dot_list.append(enc_dot)
  print(f"Encrypted dot product: {enc_dot_list}")
  print("Send this result to Client...")
  print()

  ## Decrypt
  dec_dot_dict = {}
  print("Step4. Client Side, Decryption and Sorting")
  for filename, enc_dot in zip(filenames, enc_dot_list):
    dec_dot = enc_dot.decrypt()
    dec_dot_dict[filename] = dec_dot
  dec_dot_dict_sorted =  dict(sorted(dec_dot_dict.items(), key=lambda x: x[1], reverse=True))
  for key, value in dec_dot_dict_sorted.items():
    print(f"file:{key}, sim:{value}")

def main():

  
  ## Argument
  args = parse_args()

  ## Model
  print("Loading model...")
  ## （ResNet50, excluded FC）
  ## The following expression is better, but it outputs the vector with the expornation nortation, so it leads to BUG now.
  # weights = ResNet50_Weights.DEFAULT
  # model = models.resnet50(weights=weights)
  if args.type == "image":
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Global average poolingの前まで
    model.eval()
  else:
    model = SentenceTransformer("all-MiniLM-L6-v2")
  print("Done.")

  # 前処理
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                        std=[0.229, 0.224, 0.225])
  ])

  ## Embedding and Save
  if args.mode == "gendb":
    save_data_index(model, transform, type=args.type)
  
  ## Search
  else:
    search_from_db(model, transform, query=args.inp, db=args.db)

if __name__ == "__main__":
  main()
