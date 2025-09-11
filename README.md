# model_conversion-quantization


This repository contains **model conversion & quantization** developed for ML model optimization on embedded and edge devices. 
An easy benchmark is also given in this tutorial.  
Originally inspired by an internship project at Advantech, this app has been reworked for **educational and research purposes** only.  

---

## ✨ Features
- 🔄 **Model conversion pipeline**: Keras/TF → ONNX → OpenVINO (FP32 / FP16 / INT8 quantization)
- ⚡ **Performance metrics**: latency
- 📦 Easy-to-use CLI for running benchmarks

---

## 📂 Repository Structure
```

model_conversion-quantization/
├── dataset
|   └── 
├── models
│   ├── IR
│   │   ├── FP16
│   │   │   ├── mobilenet_v1.bin
│   │   │   ├── mobilenet_v1.xml
│   │   │   ├── mobilenet_v2.bin
│   │   │   ├── mobilenet_v2.xml
│   │   │   ├── mobilenet_v3_Large.bin
│   │   │   ├── mobilenet_v3_Large.xml
│   │   │   ├── mobilenet_v3_Small.bin
│   │   │   └── mobilenet_v3_Small.xml
│   │   ├── FP32
│   │   │   ├── mobilenet_v1.bin
│   │   │   ├── mobilenet_v1.xml
│   │   │   ├── mobilenet_v2.bin
│   │   │   ├── mobilenet_v2.xml
│   │   │   ├── mobilenet_v3_Large.bin
│   │   │   ├── mobilenet_v3_Large.xml
│   │   │   ├── mobilenet_v3_Small.bin
│   │   │   └── mobilenet_v3_Small.xml
│   │   └── INT8
│   │       ├── mobilenet_v1.bin
│   │       ├── mobilenet_v1.xml
│   │       ├── mobilenet_v2.bin
│   │       ├── mobilenet_v2.xml
│   │       ├── mobilenet_v3_Large.bin
│   │       ├── mobilenet_v3_Large.xml
│   │       ├── mobilenet_v3_Small.bin
│   │       └── mobilenet_v3_Small.xml
│   ├── keras
│   │   ├── mobilenet_v1.weights.h5
│   │   ├── mobilenet_v2.weights.h5
│   │   ├── mobilenet_v3_Large.weights.h5
│   │   └── mobilenet_v3_Small.weights.h5
│   └── onnx
│       ├── mobilenet_v1.onnx
│       ├── mobilenet_v2.onnx
│       ├── mobilenet_v3_Large.onnx
│       └── mobilenet_v3_Small.onnx
├── notebook
│   ├── Convert and Quantize
│   │   ├── MobileNet_1.txt
│   │   ├── MobileNet_v2.txt
│   │   ├── MobileNet_v3_Large.txt
│   │   └── MobileNet_v3_Small.txt
│   ├── inference
│   │   ├── INT8
│   │   │   ├── MobileNet_v1.txt
│   │   │   ├── MobileNet_v2.txt
│   │   │   ├── MobileNet_v3_Large.txt
│   │   │   └── MobileNet_v3_Small.txt
│   │   ├── MobileNet_v1_FP16.txt
│   │   └── MobileNet_v1_FP32.txt
│   └── pip_installation.txt
├── README.md
├── requirements.txt
├── result_mobileNet.jpg
└── src
    ├── MobileNet_Convert_n_Quantize.py
    └── MobileNet_OV_inference.py


````

---

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/PatrickC26/model_conversion-quantization.git
cd model_conversion-quantization
pip install -r requirements.txt
````

---

## ▶️ Usage

Conver and quantize a model:
Noted that the model type should be changed in the py file 

```bash
python3 src/MobileNet_Convert_n_Quantize.py
```

Run a benchmark on a given model:

```bash
python3 src/MobileNet_OV_inference.py --model models/resnet50.onnx --precision INT8
```

---

## 📖 Documentation

See [`/docs`](./docs) for:

* Benchmarking methodology
* Analysis notebooks
* Example plots

---

## 📌 Notes

* 🔒 This repository is for **educational and research use only**.
* 🚫 No proprietary Advantech data is included.
* 🛠️ Contributions welcome (see Issues/PRs).

---

## 📄 License

This project is licensed under an **Educational Use License**.
Unauthorized commercial use is prohibited.


## ⚠️ Disclaimer
This project is provided for **educational and research purposes only**.  
It does not contain or intend to distribute any proprietary information from Advantech or other organizations.  

If you believe that any part of this repository violates copyright, licensing terms, or other rights,  
please open an issue or contact me, and I will address it promptly.

