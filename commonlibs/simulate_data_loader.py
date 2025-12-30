gpu_num = 6
imgs_per_gpu = 20
samples_per_gpu = imgs_per_gpu
img_num = 5000
while img_num > 0:
    batches = []
    for i in range(gpu_num):
        batch = img_num - max(img_num - samples_per_gpu, 0)
        img_num = max(img_num - samples_per_gpu, 0)
        batches.append(batch)
    print(batches)


