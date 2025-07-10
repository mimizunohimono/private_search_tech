import tenseal as ts
import numpy as np

# CKKS コンテキストの作成
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # 適切な精度
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# ベクトルの準備
plain_vec = [1.0, 2.0, 3.0, 4.0]
enc_vec_data = [0.5, 1.0, 1.5, 2.0]

# 表示: 入力ベクトル
print("📌 平文ベクトル (plain_vec):", plain_vec)
print("📌 暗号化対象ベクトル (enc_vec_data):", enc_vec_data)

# 暗号文作成
context = create_context()
enc_vec = ts.ckks_vector(context, enc_vec_data)

# 内積計算（enc_vec is encrypted, plain_vec is plaintext）
enc_dot = enc_vec.dot(plain_vec)

# 表示: 暗号化された内積結果（中身は非表示のまま）
print("🔐 内積結果 (enc_dot):", enc_dot)

# 復号して結果確認
decrypted_result = enc_dot.decrypt()
print("🔓 復号後の内積結果 (decrypted_result):", decrypted_result)

# 検算
expected_result = np.dot(enc_vec_data, plain_vec)
print("✅ 検算結果 (plain inner product):", expected_result)
