import subprocess
import math

def get_gpu_specs():
    """使用 nvidia-smi 工具獲取 GPU 規格"""
    try:
        # 使用 nvidia-smi 命令獲取 GPU 信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,clocks.max.sm', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode('utf-8').strip()
        if not output:
            raise ValueError("No output from nvidia-smi")

        gpu_info = output.splitlines()[0].split(', ')

        gpu_name = gpu_info[0]
        total_memory = int(gpu_info[1]) * 1024 * 1024  # 轉換為 bytes
        max_sm_clock = int(gpu_info[2]) * 1000  # 轉換為 Hz

        # 假設 Tensor Core 的數量和 SM clock 相關
        tensor_cores = 64  # 此值可能需要根據具體的 GPU 型號進行調整

        return {
            'name': gpu_name,
            'total_memory': total_memory,
            'max_sm_clock': max_sm_clock,
            'tensor_cores': tensor_cores
        }

    except Exception as e:
        print(f"Error retrieving GPU specs: {e}")
        return None

def calculate_optimal_transformer_config(gpu_specs, context_length=512, training_factor=1.0, vocab_size=32000, reserved_vram=1 * 1024 * 1024 * 1024, batch_size=32):
    """根據 GPU 規格計算最佳的 Transformer 模型配置"""
    if gpu_specs is None:
        print("Error: GPU specs not available")
        return None

    total_memory = gpu_specs['total_memory'] - reserved_vram
    tensor_cores = gpu_specs['tensor_cores']

    # 定義一些合理的範圍和初始設置
    min_dim = 128
    max_dim = 4096
    min_layers = 2
    max_layers = 48

    optimal_config = None
    best_performance = float('-inf')

    for dim in range(min_dim, max_dim + 1, 128):
        # 保證 n_heads 是 3 的倍數
        n_heads = max(3, (dim // 64) // 3 * 3)
        for n_layers in range(min_layers, max_layers + 1):
            # 計算模型的參數量和所需的內存
            # 詞表嵌入內存
            vocab_embedding_memory = vocab_size * dim * 4  # 假設使用 float32
            # 模型參數內存（估計）
            model_memory = 2 * (dim * n_layers * (dim * 4 + dim * n_heads))  # 粗略估計模型大小

            # 計算注意力機制的上下文內存使用量
            attention_memory_usage = 2 * context_length * context_length * dim * n_layers * batch_size

            # 總的內存使用量
            total_memory_usage = (model_memory + vocab_embedding_memory + attention_memory_usage) * training_factor

            if total_memory_usage > total_memory:
                break

            # 假設性能與 Tensor Core 數量和模型大小相關
            performance = tensor_cores * (dim * n_heads) / math.log2(model_memory)

            if performance > best_performance:
                best_performance = performance
                optimal_config = {
                    'dim': dim,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'context_length': context_length,
                    'vocab_size': vocab_size,
                    'batch_size': batch_size
                }

    return optimal_config

def main():
    gpu_specs = get_gpu_specs()
    if gpu_specs is None:
        print("Unable to retrieve GPU specs.")
        return

    print(f"Detected GPU: {gpu_specs['name']}")
    print(f"Total Memory: {gpu_specs['total_memory'] / (1024 * 1024)} MB")
    print(f"Max SM Clock: {gpu_specs['max_sm_clock'] / 1e6} GHz")
    print(f"Tensor Cores: {gpu_specs['tensor_cores']}")

    # 設置上下文長度、詞匯表大小和批次大小
    context_length = 512
    vocab_size = 32000
    reserved_vram = 1 * 1024 * 1024 * 1024  # 預留 1GB
    batch_size = 32

    optimal_config = calculate_optimal_transformer_config(
        gpu_specs,
        context_length=context_length,
        training_factor=1.0,
        vocab_size=vocab_size,
        reserved_vram=reserved_vram,
        batch_size=batch_size
    )
    if optimal_config:
        print(f"Optimal Transformer Configuration: dim={optimal_config['dim']}, n_layers={optimal_config['n_layers']}, n_heads={optimal_config['n_heads']}, context_length={optimal_config['context_length']}, vocab_size={optimal_config['vocab_size']}, batch_size={optimal_config['batch_size']}")
    else:
        print("Unable to find an optimal configuration.")

if __name__ == "__main__":
    main()