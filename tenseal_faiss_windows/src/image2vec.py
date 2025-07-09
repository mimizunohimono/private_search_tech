#!/usr/bin/env python3

import os
import cv2
import numpy as np
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 特徴量抽出モデル（ResNet50, 最後のFC層は除外）
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Global average poolingの前まで
model.eval()

# 前処理
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],\
             std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
  image = Image.open(image_path).convert('RGB')
  img_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
  with torch.no_grad():
    features = model(img_tensor).squeeze().numpy()  # shape: (2048,)
  return features / np.linalg.norm(features)  # 正規化

# 画像フォルダ
image_dir = "images"  # ここに検索対象の画像を入れておく

# 特徴量とファイル名の一覧作成
features_list = []
image_paths = []

for fname in os.listdir(image_dir):
  if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
    path = os.path.join(image_dir, fname)
    feat = extract_features(path)
    features_list.append(feat.astype('float32'))
    image_paths.append(path)

# FAISS index 作成（内積距離 or L2距離）
dim = 2048  # 特徴量の次元
index = faiss.IndexFlatL2(dim)
index.add(np.array(features_list))

# クエリ画像の特徴量を取得して検索
query_image_path = "query.jpg"
query_feat = extract_features(query_image_path).astype('float32').reshape(1, -1)

top_k = 3
result_dist, result_index = index.search(query_feat, top_k)

# 検索結果表示
print("Query image:", query_image_path)
print("Top matches:")
for i, idx in enumerate(result_index[0]):
  print(f"{i+1}: {image_paths[idx]} (Distance: {result_dist[0][i]:.4f})")
