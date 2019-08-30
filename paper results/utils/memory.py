import torch
import gc


def format_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def print_cuda_memory_usage(min_size=16):
    tensors = []
    tot = 0
    print("=" * 80)
    print("Tensors on GPU:")
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.device.type == "cuda":
            tot += obj.element_size() * obj.nelement()
            tensors.append((obj.element_size() * obj.nelement(), tuple(obj.size())))
    tensors.sort(key=lambda x: x[0], reverse=True)
    for t in tensors:
        if t[0] >= min_size:
            print(f"    Size: {format_size(t[0])}, Shape: {t[1]}")

    print(f"Total tensor size: {format_size(tot)}, Allocated memory: {format_size(torch.cuda.memory_allocated())}, Cached memory: {format_size(torch.cuda.memory_cached())}, Total GPU memory: {format_size(torch.cuda.get_device_properties(0).total_memory)}")
    print("=" * 80)

