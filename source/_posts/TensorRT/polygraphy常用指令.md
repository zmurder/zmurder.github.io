# 类似onnxsim

polygraphy可以实现类似onnxsim的功能，例如简化模型和常量折叠之类的，最重要的可以导出一个所有的节点都可以看到shape的onnx。即便是onnx中又自定义算子也没问题。

```bash
polygraphy surgeon sanitize ./1.onnx --fold-constants -o ./1_fold.onnx
```











<font color="white">这是一篇优秀的博客，必须推荐。</font>

