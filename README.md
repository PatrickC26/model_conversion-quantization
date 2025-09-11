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

