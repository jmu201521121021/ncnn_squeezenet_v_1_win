# ncnn_squeezenet_v_1_win

将 https://github.com/nihui/ncnn-android-squeezenet里的android平台代码在 PC上ncnn实现，可以熟悉ncnn的使用。

## 开发环境

- Windows10
- Visual Studio 2017
- ncnn （20191223）
- Opencv 3.0


## 推理

- 模型
  - **squeezenet_v_1 **：<https://github.com/nihui/ncnn-android-squeezenet>
  -  参考nihui大佬的android端的代码在pc上实现。


- main.cpp

  ```c++
  //  加载模型
  ncnn::Net net;
  net.load_param_bin("../model/squeezenet_v1.1.param.bin");
  net.load_model("../model/squeezenet_v1.1.bin");
  // forward
  ncnn::Extractor ex = net.create_extractor();
  ex.set_light_smode(true);
  ex.input(squeezenet_v1_1_param_id::BLOB_data, matIn);
  ex.extract(squeezenet_v1_1_param_id::BLOB_prob, matOut);
  ```

  ​

- 测试结果

  ![grand result](./images/result.jpg)

  ​