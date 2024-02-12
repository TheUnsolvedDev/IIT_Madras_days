import pylops

a = pylops.Laplacian(dims=(10, 10), edge=True,
                     weights=(3, 3), dtype="float32")
print(a.__dict__)