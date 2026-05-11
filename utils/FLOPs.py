def calculate_flops_and_params(model, data):
    try:
        from thop import profile

        # 获取一个样本并确保有batch维度
        if isinstance(data, (list, tuple)):
            # 如果是三元组数据，取anchor部分
            dummy = data[0][0:1]  # data[0]是anchor, [0:1]保持batch维度
        else:
            dummy = data[0:1]
        
        # 检查维度，如果是3D则添加batch维度
        if dummy.dim() == 3:
            dummy = dummy.unsqueeze(0)  # 添加batch维度: (C, H, W) -> (1, C, H, W)
        
        print(f"Input shape for FLOPs calculation: {dummy.shape}")

        # 针对TripletNet的特殊处理
        if hasattr(model, 'embedding_net'):
            embedding_net = model.embedding_net
            flops_single, params = profile(embedding_net, inputs=(dummy,), verbose=False)
            total_flops = flops_single
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")
        else:
            total_flops, params = profile(model, inputs=(dummy,), verbose=False)
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")

        return total_flops, params
    except ImportError:
        print("THOP not installed. Please install it using 'pip install thop'")
        return None, None
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
