



# 一个使用onnx_graphsurgeon  的例子

融合 ONNX 模型中 `ConvTranspose` 和 `BatchNormalization (BN)` 层的函数，这是一个常见的模型优化操作

```python
import onnx
import numpy as np
import onnx_graphsurgeon as gs

def fuse_convtranspose_bn(onnx_model_path: str, output_path: str):
    # Load the ONNX model
    graph = gs.import_onnx(onnx.load(onnx_model_path))

    # Track nodes to remove
    nodes_to_remove = []

    # Iterate through nodes to find ConvTranspose -> BN patterns
    for node in graph.nodes:
        if node.op == "ConvTranspose":
            # Find the next BatchNormalization node
            bn_node = None
            for output in node.outputs:
                if len(output.outputs) == 1 and output.outputs[0].op == "BatchNormalization":
                    bn_node = output.outputs[0]
                    break
            
            if not bn_node:
                continue

  
            conv_weight_node = node.inputs[1].inputs[0].inputs[0].inputs[0].inputs[0]  # ConvTranspose weight is always 2nd input
            conv_weight = conv_weight_node.values
            out_channels = conv_weight_node.shape[1]

            # Extract BN parameters
            gamma = bn_node.inputs[1].values  # Scale
            beta = bn_node.inputs[2].values   # Bias
            mean = bn_node.inputs[3].values   # Mean
            var = bn_node.inputs[4].values    # Variance
            epsilon = bn_node.attrs.get("epsilon", 1e-5)

            # Compute fused weights and biases
            scale = gamma / np.sqrt(var + epsilon)
            fused_weight = scale.reshape(1, out_channels, 1, 1) * conv_weight
            conv_weight_node.values = fused_weight

            # Reroute connections to bypass BN


            for consumer in bn_node.outputs[0].outputs[:]:  # Iterate over a copy
                consumer.inputs = [node.outputs[0]]


            bn_node.outputs.clear()
            graph.nodes.remove(bn_node)


    # Cleanup orphaned nodes/tensors
    graph.cleanup()

    # Save the fused model
    onnx.save(gs.export_onnx(graph), output_path)



# Fuse ConvTranspose-BN
fused_model = fuse_convtranspose_bn("5278080_ec039e076f_seg_ptq_01.onnx", "fused_model.onnx")
```



# 另一个onnx_graphsurgeon的例子

- 删除输入后的第一个 `Reshape`
- 合并连续的 `Reshape` 节点
- 修改 `Resize` 属性和尺寸
- 修改输入输出 batch size

```python
import onnx
import onnx_graphsurgeon as gs
import os
import argparse

def modify_onnx(input_model, output_model=None):
    if output_model is None:
        filename, ext = os.path.splitext(input_model)
        output_model = f"{filename}_dla{ext}"
    
    graph = gs.import_onnx(onnx.load(input_model))
    
    # 处理第一个节点是Reshape的情况
    removed_reshape = False
    if len(graph.inputs) > 0 and len(graph.nodes) > 0:
        # 找到第一个使用图输入的节点
        first_node = None
        for node in graph.nodes:
            if graph.inputs[0] in node.inputs:
                first_node = node
                break
                
        # 检查该节点是否为Reshape
        if first_node is not None and first_node.op == "Reshape":
            reshape_output = first_node.outputs[0]
            
            # 创建新的输入变量，使用reshape输出的形状
            new_input = gs.Variable(
                name=graph.inputs[0].name,  # 保持原始输入名称
                dtype=reshape_output.dtype,
                shape=reshape_output.shape
            )
            
            # 将所有使用reshape输出的节点改为使用新输入
            for node in graph.nodes:
                for i, inp in enumerate(node.inputs):
                    if inp == reshape_output:
                        node.inputs[i] = new_input
            
            # 更新图的输入
            graph.inputs = [new_input] + graph.inputs[1:]
            
            # 删除Reshape节点
            graph.nodes.remove(first_node)
            removed_reshape = True
            
            print(f"已删除第一个Reshape节点，将输入shape设置为 {new_input.shape}")
    
    # 处理连续的Reshape节点
    consecutive_reshape_removed = 0
    reshape_nodes = [node for node in graph.nodes if node.op == "Reshape"]
    
    # 创建从Reshape输出到节点的映射，用于快速查找
    output_to_node = {}
    for node in graph.nodes:
        for inp in node.inputs:
            if inp.name:
                output_to_node[inp.name] = node
    
    nodes_to_remove = []
    for reshape1 in reshape_nodes:
        if reshape1 in nodes_to_remove:
            continue
            
        reshape1_output = reshape1.outputs[0]
        
        # 检查reshape1的输出是否连接到另一个reshape节点
        if reshape1_output.name in output_to_node:
            reshape2 = output_to_node[reshape1_output.name]
            if reshape2.op == "Reshape" and reshape2 not in nodes_to_remove:
                reshape2_output = reshape2.outputs[0]
                
                # 获取reshape1的输入和reshape2的输出
                reshape1_input = reshape1.inputs[0]
                
                # 检查reshape1的输入shape和reshape2的输出shape是否相同
                if reshape1_input.shape == reshape2_output.shape:
                    # 将所有使用reshape2输出的节点改为使用reshape1的输入
                    for node in graph.nodes:
                        for i, inp in enumerate(node.inputs):
                            if inp == reshape2_output:
                                node.inputs[i] = reshape1_input
                    
                    # 标记这两个reshape节点待删除
                    nodes_to_remove.extend([reshape1, reshape2])
                    consecutive_reshape_removed += 1
                    print(f"找到并移除连续Reshape节点: {reshape1.name} -> {reshape2.name}")
    
    # 从图中删除标记的节点
    for node in nodes_to_remove:
        graph.nodes.remove(node)
    
    if consecutive_reshape_removed > 0:
        print(f"总共移除了 {consecutive_reshape_removed} 对连续的Reshape节点")
    
    # 修改Resize节点
    resize_count = 0
    target_attrs = {
        "coordinate_transformation_mode": "asymmetric",
        "cubic_coeff_a": -0.75,
        "mode": "nearest",
        "nearest_mode": "floor"
    }
    
    for node in graph.nodes:
        if node.op == "Resize":
            resize_count += 1
            node.attrs.update(target_attrs)
            print(f"修改了Resize节点: {node.name}")
            
            # 修改Resize节点的目标尺寸
            # Resize节点通常有4个输入: X, roi, scales, sizes (有时只有前3个)
            if len(node.inputs) >= 4 and node.inputs[3] is not None:
                # 第4个输入是sizes
                sizes = node.inputs[3]
                
                # 如果sizes是常量，并且它的值可以被访问
                if isinstance(sizes, gs.Constant) and sizes.values is not None:
                    sizes_values = sizes.values
                    if len(sizes_values) > 0 and sizes_values[0] != 1:
                        # 修改第一个维度为1
                        old_size = int(sizes_values[0])
                        sizes_values[0] = 6
                        sizes.values = sizes_values
                        print(f"  修改了Resize节点 {node.name} 的目标尺寸batchsize从 {old_size} 到 6")
    
    print(f"总共修改的Resize节点数: {resize_count}")

    #haowang comment:  结尾的softmax + tranpose在DLA上run效率很低（FP16 softmax），这里尝试移除可以获得比较大的收益
    # # 查找输出连接到两个Slice节点的tensor
    # tensor_to_slices = {}
    # for node in graph.nodes:
    #     if node.op == "Slice":
    #         for input_tensor in node.inputs:
    #             if input_tensor.name:
    #                 if input_tensor.name not in tensor_to_slices:
    #                     tensor_to_slices[input_tensor.name] = []
    #                 tensor_to_slices[input_tensor.name].append(node)
    
    # found_output = False
    # for tensor_name, slice_nodes in tensor_to_slices.items():
    #     if len(slice_nodes) == 2:
    #         # 找到连接到恰好两个Slice节点的tensor
    #         for tensor in graph.tensors().values():
    #             if tensor.name == tensor_name:
    #                 # 设置该tensor为网络的唯一输出
    #                 graph.outputs = [tensor]
    #                 print(f"发现tensor '{tensor_name}' 连接到两个Slice节点，已将其设置为网络唯一输出")
    #                 found_output = True
    #                 break
    #         if found_output:
    #             break
    
    # if not found_output:
    #     graph.outputs = []
        
    # 修改输入和输出的batchsize为1
    batch_modified = False
    
    # 处理输入tensor
    for i, inp in enumerate(graph.inputs):
        if inp.shape is not None and len(inp.shape) > 0:
            # 复制原始shape并修改第一个维度为1
            new_shape = list(inp.shape)
            if new_shape[0] is not None and new_shape[0] != 6:
                old_shape = new_shape[0]
                new_shape[0] = 6
                inp.shape = tuple(new_shape)
                print(f"输入 '{inp.name}' 的batchsize从 {old_shape} 修改为 6")
                batch_modified = True
    
    # 如果发现了需要设为输出的tensor，就不再处理graph.outputs
    # if not found_output:
    #     # 处理输出tensor
    for i, out in enumerate(graph.outputs):
        if out.shape is not None and len(out.shape) > 0:
            # 复制原始shape并修改第一个维度为1
            new_shape = list(out.shape)
            if new_shape[0] is not None and new_shape[0] != 1:
                old_shape = new_shape[0]
                new_shape[0] = 1
                out.shape = tuple(new_shape)
                print(f"输出 '{out.name}' 的batchsize从 {old_shape} 修改为 6")
                batch_modified = True
    
    if batch_modified:
        print("已完成输入/输出batchsize修改为1")
        
    # 清理图
    graph.cleanup()
    
    # 保存修改后的模型
    output_onnx = gs.export_onnx(graph)
    onnx.save(output_onnx, output_model)
    print(f"修改后的模型保存至: {output_model}")
    
    return resize_count, removed_reshape, consecutive_reshape_removed, batch_modified

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修改ONNX模型")
    parser.add_argument("input_model", help="输入ONNX模型的路径")
    parser.add_argument("-o", "--output_model", help="保存修改后ONNX模型的路径", default=None)
    
    args = parser.parse_args()
    
    resize_count, removed_reshape, consecutive_reshape_removed, batch_modified = modify_onnx(args.input_model, args.output_model)
    
    if resize_count == 0 and not removed_reshape and consecutive_reshape_removed == 0 and not batch_modified:
        print("模型未进行任何修改。")
```

