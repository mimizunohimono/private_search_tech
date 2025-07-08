#!/usr/bin/env python3
"""
File: eva_polynomial_example.py
Example Python script for Programming with PyEVA (EVA v1.0.0).
This script defines a simple polynomial evaluation program, compiles it
for the CKKS scheme, generates encryption keys, encrypts inputs,
performs homomorphic evaluation, displays encrypted outputs vs decrypted,
decrypts outputs, and computes MSE against plaintext evaluation.
"""
from eva import EvaProgram, Input, Output, evaluate
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse

# 1. Define the program: evaluate 3*x^2 + 5*x - 2 over a vector of size 1024
vec_size = 1024
prog = EvaProgram('Polynomial', vec_size=vec_size)
with prog:
    x = Input('x')
    Output('y', 3 * x**2 + 5 * x - 2)

# 2. Set fixed-point scales and output ranges (30 bits)
prog.set_input_scales(30)
prog.set_output_ranges(30)

# 3. Compile for CKKS
compiler = CKKSCompiler()
compiled_poly, params, signature = compiler.compile(prog)
print("Compiled program DOT representation:")
print(compiled_poly.to_DOT())

# 4. Generate keys
public_ctx, secret_ctx = generate_keys(params)

# 5. Prepare plaintext inputs and encrypt
inputs = { 'x': [float(i) for i in range(vec_size)] }
enc_inputs = public_ctx.encrypt(inputs, signature)

# 6. Homomorphic evaluation
enc_outputs = public_ctx.execute(compiled_poly, enc_inputs)

# 7. Display encrypted outputs sample (first 5 ciphertexts)
sample_ciphertexts = enc_outputs['y'][:5]
print("Encrypted outputs sample (first 5 ciphertexts):", sample_ciphertexts)

# 8. Decrypt outputs
decrypted_outputs = secret_ctx.decrypt(enc_outputs, signature)
print(f"Decrypted outputs sample (first 10): {decrypted_outputs['y'][:10]}")

# 9. Plaintext evaluation and MSE computation
reference = evaluate(compiled_poly, inputs)
mse = valuation_mse(decrypted_outputs, reference)
print(f"Mean Squared Error: {mse}")

