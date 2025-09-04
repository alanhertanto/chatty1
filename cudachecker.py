import torch
print(torch.cuda.is_available())        # True kalau GPU bisa dipakai
# print(torch.cuda.get_device_name(0))    # Nama GPU
# print(torch.cuda.current_device())      # Index device aktif
