# 1 简介

翻译自官网 https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/README.md



Polygraphy是一个工具包，旨在帮助在TensorRT和其他框架中运行和调试深度学习模型。它包括一个 [Python API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy) 和一个 [a command-line interface (CLI)](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/tools)）。

除其他外，使用Polygraphy，您可以：

* 在多个后端之间运行推理，如TensorRT和ONNX-Runtime，并比较结果(for example [API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/01_comparing_frameworks),[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/01_comparing_frameworks)).                              
* 将模型转换为各种格式，例如带有训练后量化的TensorRT引擎 (for example [API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/04_int8_calibration_in_tensorrt),[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/convert/01_int8_calibration_in_tensorrt)).                              
* 查看有关各种类型模型的信息 (for example [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/inspect))                              
* 在命令行上修改ONNX模型：
* 提取子图(for example [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/01_isolating_subgraphs)).                                    
* 简化和净化 (for example [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants)).                                    
* 隔离TensorRT中的错误策略 (for example [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics)).                              

有关更多详细信息，请参阅 [Polygraphy repository](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy).                     

# 2 安装

重要提示：Polygraphy仅支持Python 3.6及更高版本。在遵循以下说明之前，请确保您使用的是受支持的Python版本。

## Installing Prebuilt Wheels

```python
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

注意：在Linux上，默认情况下，命令行工具包通常安装到${HOME}/.local/bin。请确保将此目录添加到PATH环境变量中。

## 从源码包编译安装

### 使用Make Targets (Linux)

```bash
make install
```

### Using Powershell Script (Windows)

确保允许您在系统上执行脚本，然后运行：

```bash
.\install.ps1
```

### 手动构建

* 安装先决条件

  ```python
  python -m pip install wheel
  ```

* Build a wheel:

  ```python
  python setup.py bdist_wheel
  ```

* 从存储库手动安装wheel 

  On Linux, run:

  ```bash
  python -m pip install Polygraphy/dist/polygraphy-*-py2.py3-none-any.whl
  ```

  On Windows, using Powershell, run:

  ```bash
  $wheel_path = gci -Name Polygraphy\dist
  python -m pip install Polygraphy\dist\$wheel_path
  ```

  注：强烈建议为Polygraphy的彩色输出安装彩色模块，因为这可以大大提高可读性：

  ```bash
  python -m pip install colored
  ```



# 3 调试TensorRT精度问题

参考：https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md

TensorRT中的准确性问题，尤其是在大型网络中，调试起来可能很有挑战性。使它们易于管理的一种方法是缩小问题的规模或找出故障的根源。

本指南旨在提供这样做的一般方法；它的结构是一个扁平的流程图——在每个分支上，都提供了两个链接，因此您可以选择最适合您的情况的链接。

如果您使用的是ONNX型号，请在继续操作之前尝试[sanitizing it](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants) ，因为这在某些情况下可能会解决问题。

## 3.1 真实输入数据会产生影响吗？

某些模型可能对输入数据敏感。例如，实际输入可能比随机生成的输入具有更好的准确性。Polygraphy提供了多种提供真实输入数据的方法，如[`run` example 05](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/05_comparing_with_custom_input_data).所述。

使用真实的输入数据是否可以提高准确性？

* 是的，使用真实输入数据时，准确性是可以接受的。

  这可能意味着没有bug；相反，您的模型对输入数据很敏感。

* 不，即使使用真实的输入数据，我仍然会看到准确性问题。

  转至：间歇性或不间歇性？（下面3,2）

## 3.2 间歇性还是非间歇性？

这个问题在不同的引擎之间是间歇性的吗?

* 是的，有时当我重建引擎时，准确性问题就消失了。

  转到：调试间歇性精度问题（下面3,3）

* 不，我每次构建引擎时都会看到准确性问题。

  转到:分层是一个选项（下面3.4）

## 3.3 调试间歇性精度问题

由于引擎构建过程是不确定的，因此每次构建引擎时都可以选择不同的策略（即层实现）。当其中一种策略出现故障时，这可能表现为间歇性故障。Polygraphy包括一个调试构建子工具，可以帮助您找到这样的策略。

有关更多信息，请参阅[`debug` example 01](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics).。

你找到失败的策略了吗

* 是的，我知道哪种策略是错误的。

  转到：你有一个最小的失败案例！（下 3.8）

* 否，故障可能不是间歇性的。

  转到：分层是一种选择吗？（下3,4）

## 3.4 分层是一个选择吗

如果精度问题始终是可重复的，那么最好的下一步是找出导致故障的层。Polygraphy包括一种机制，将网络中的所有张量标记为输出，以便进行比较；然而，这可能会潜在地影响TensorRT的优化过程。因此，我们需要确定当所有输出张量都被标记时，我们是否仍然观察到准确性问题。

有关如何在继续之前比较每层输出的详细信息，请参阅 [this example](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/01_comparing_frameworks/README.md#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)。

在比较逐层输出时，您是否能够重现精度故障？

* 是的，即使我在网络中标记了其他输出，故障也会重新处理。

  转到：提取失败的子图（下 3.5）

* 不，标记其他输出会导致精度提高，或者当我标记其他输出时，我根本无法运行模型。

  转到：减少失败的Onnx模型（下 3.6）

## 3.5 提取失败子图

由于我们能够比较各层的输出，因此我们应该能够通过查看输出比较日志来确定哪个层首先引入了错误。一旦我们知道哪一层有问题，我们就可以从模型中提取它。

为了计算出所讨论的层的输入和输出张量，我们可以使用`polygraphy inspect model`检查模型。有关详细信息，请参阅其中一个示例：

* [TensorRT Networks](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/inspect/01_inspecting_a_tensorrt_network)
* [ONNX models](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/inspect/03_inspecting_an_onnx_model).

接下来，我们可以提取一个子图，只包括有问题的层。有关更多信息，请参阅[`surgeon` example 01](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/01_isolating_subgraphs).1。

这个孤立的子图是否再现了问题？

* 是的，子图也失败了。

  转到：你有一个最小的失败案例！（下 3.8）

* 不，子图工作得很好。

  转到：减少失败的Onnx模型（下 3,6）

## 3.6 减少失败的ONNX模型

当我们无法使用分层比较来确定故障源时，我们可以使用蛮力方法来减少ONNX模型——迭代生成越来越小的子图，以找到仍然失败的最小子图。`debug reduce`工具有助于实现这一过程的自动化。

有关更多信息，请参阅[`ebug` example 02](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/debug/02_reducing_failing_onnx_models)。

简化模型是否失败？

* 是的，精简模型失败了。

  转到：你有一个最小的失败案例！（下 3.8）

* 不，简化模型不会失败，或者以不同的方式失败。

  转到：重复检查您的减少选项（下 3.7）

## 3.7 重复检查您的减少选项

如果精简模型不再失败，或者以其他方式失败，请确保`--check`命令是正确的。您可能还想使用`--fail regex`来确保在减少模型时只考虑准确性故障（而不是其他无关的故障）。

* 再次尝试减少。

  转到：减少失败的Onnx模型（3.6）

## 3.8 你有一个最小的失败案例！

如果你已经做到了这一点，那么你现在就有了一个最小的失败案例！进一步的调试应该更容易。

如果您是TensorRT开发人员，那么此时您需要深入研究代码。如果没有，请报告您的错误！

## 3.9 run example

### example 01 比较框架

#### Introduction

您可以使用`run`子工具来比较不同框架之间的模型。在最简单的情况下，您可以提供一个模型和一个或多个框架标志。默认情况下，它将生成合成输入数据，使用指定的框架运行推理，然后比较指定框架的输出。

#### 运行示例

在本例中，我们将概述`run`子工具的各种常见用例：

* 比较TensorRT和ONNX运行时输出
* TensorRT精度比较
* 更改公差
* 更改比较度量
* ONNX Runtime和TensorRT的逐层输出比较

##### 比较TensorRT和ONNX运行时输出

要在Polygraphy中使用两个框架运行模型并执行输出比较：

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

`dynamic_identity.onnx`模型具有动态输入形状。默认情况下，`Polygraphy`会将模型中的任何动态输入尺寸覆盖为常量。DEFAULT_SHAPE_VALUE（定义为1）并警告您：

```bash
[W]     Input tensor: X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | No shapes provided; Will use shape: [1, 2, 1, 1] for min/opt/max in profile.
[W]     This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.
```

为了抑制此消息并明确向Polygraphy提供输入形状，请使用`--input-shapes`选项：

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

//TODO 待更新

### example 05 与自定义输入数据比较

#### Introduction

在某些情况下，我们可能希望使用自定义输入数据进行比较。Polygraphy提供了多种方法来做到这一点，下面将详细介绍[here](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/use_custom_input_data.md).。

在本例中，我们将演示两种不同的方法：

1. 通过在Python脚本`data_loader.py`中定义`load_data（）`函数来使用数据加载器脚本。Polygraphy将使用`load_data（`）在运行时生成输入。
2. 使用包含预先生成的输入的JSON文件。为了方便起见，我们将使用上面的脚本`data_loader.py`将`load_data（）`生成的输入保存到一个名为`custom_puts.json`的文件中。

提示：通常，在处理大量输入数据时，数据加载程序脚本是首选的，因为它可以避免写入磁盘。另一方面，JSON文件可能更具可移植性，并有助于确保再现性。

最后，我们将为`polygraphy run`提供自定义输入数据，并比较`ONNX Runtime`和`TensorRT`之间的输出。

由于我们的模型具有动态形状，我们需要设置一个TensorRT优化配置文件。有关如何通过命令行执行此操作的详细信息，请参见[`onvert` example 03](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/convert/03_dynamic_shapes_in_tensorrt).。为了简单起见，我们将创建一个配置文件，其中`min==opt==max`。

注意：重要的是，我们的优化配置文件与自定义数据加载程序提供的形状配合使用。在我们非常简单的情况下，数据加载器总是生成形状（1，2，28，28）的输入，所以我们只需要确保这在[min，max]内。

#### 运行示例

1. 运行脚本将输入数据保存到磁盘。注意：这仅适用于选项2。

   ```bash
   python3 data_loader.py
   ```

2. 使用自定义输入数据，使用TensorRT和ONNX Runtime运行模型：

   1. 选项1：使用数据加载程序脚本：

      ```bash
      polygraphy run dynamic_identity.onnx --trt --onnxrt \
          --trt-min-shapes X:[1,2,28,28] --trt-opt-shapes X:[1,2,28,28] --trt-max-shapes X:[1,2,28,28] \
          --data-loader-script data_loader.py
      ```

   2. 选项2：使用包含已保存输入的JSON文件：

      ```bash
      polygraphy run dynamic_identity.onnx --trt --onnxrt \
          --trt-min-shapes X:[1,2,28,28] --trt-opt-shapes X:[1,2,28,28] --trt-max-shapes X:[1,2,28,28] \
          --load-inputs custom_inputs.json
      ```

      

## 3.10 debug example

### example 01 调试Flaky TensorRT策略

重要提示：此示例不再适用于较新版本的`TensorRT`，因为它们所做的一些策略选择没有通过`IAlgorithSelector`接口公开。因此，下面概述的方法不能保证确定性的引擎构建。使用`TensorRT 8.`7及更新版本，您可以使用策略定时缓存（`Polygraphy中的--save timing cache和--load timing  cache`）来确保确定性，但这些文件是不透明的，因此无法通过检查差异策略进行解释

#### Introduction

有时，`TensorRT`中的策略可能会产生不正确的结果，或者有其他错误的行为。由于`TensorRT`构建器依赖于定时策略，因此引擎构建是不确定的，这可能会使策略错误表现为片状/间歇性故障。

解决这个问题的一种方法是多次运行生成器，保存每次运行的战术回放文件。一旦我们有了一套已知的好策略和已知的坏策略，我们就可以对它们进行比较，以确定哪种策略可能是错误的来源。

`debug build`工具允许您自动执行此过程。

有关`debug`工具如何工作的更多详细信息，请参阅帮助输出：`polygraph debug-h`和`polygraph debug-build-h`。

#### 运行示例

1. 从ONNX运行时生成golden 输出：

   ```bash
   polygraphy run identity.onnx --onnxrt \
       --save-outputs golden.json
   ```

2. 使用调试构建来重复构建TensorRT引擎，并将结果与golden 输出进行比较，每次保存一个策略重播文件

   ```bash
   polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
       --artifacts-dir replays --artifacts replay.json --until=10 \
       --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
   ```

   让我们来分解一下：

   与其他`debug`子工具一样，`debug build`在每次迭代中生成一个中间工件（默认情况下为`./polygraphy_debug.engine`）。本例中的这个工件是一个`TensorRT`引擎。

   提示：`debug build`支持其他工具支持的所有`TensorRT`构建器配置选项，如`convert`或`run`。

   为了`debug build`以确定每个引擎是否失败或通过，我们提供了`--check`命令。由于我们看到的是（虚假的）准确性问题，我们可以使用`polygraphy run`将引擎的输出与我们的黄金值进行比较。

   提示：与其他`debug`子工具一样，也支持交互式模式，只需省略`--check`参数即可使用该模式。

   与其他`debug`子工具不同，`debug build`没有自动终止条件，因此我们需要提供`--until`选项，以便工具知道何时停止。这可以是多次迭代，也可以是`good` or `bad`。在后一种情况下，工具将在分别找到第一次通过或失败的迭代后停止。

   由于我们最终想要比较好的和坏的战术回放，我们指定`--save-tactics`来保存每次迭代的策略重播文件，然后使用`--artifacts`来告诉调试构建来管理它们，这涉及到将它们排序到主`artifacts`目录下的`good`的和`bad`的子目录中，用`--artifacts-dir`指定。

3. 使用`inspect diff-tactic`来确定哪些策略可能不好：

   ```bash
   polygraphy inspect diff-tactics --dir replays
   ```

   注意：最后一步应该报告它无法确定潜在的糟糕策略，因为我们的糟糕目录此时应该是空的（否则请提交TensorRT问题！）：

   ```bash
   [I] Loaded 2 good tactic replays.
   [I] Loaded 0 bad tactic replays.
   [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
   ```

   有关调试工具的更多信息，以及适用于所有调试子工具的提示和技巧，请参阅调[how-to guide for `debug` subtools](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/use_debug_subtools_effectively.md).。

   

1. 1. 

   

# 附录

官方文档

*  https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/README.md
* [2.12. Polygraphy](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#polygraphy-ovr)

