## **Installation**

### **Prerequisites**
- Python 3.8+
- Conda

### **Setup**

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/GhM-Nada/table-OCR.git
   cd table-OCR
   ```

2. **Create and Activate Conda Environment**

   Create a new conda environment and activate it:

   ```bash
   conda create --name myenv python=3.12.7
   conda activate myenv
   ```

3. **Install PaddlePaddle**

   Install PaddlePaddle in the conda environment:

   ```bash
   python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   ```

4. **Install PaddleOCR**

   Install PaddleOCR:

   ```bash
   pip install paddleocr
   ```

5. **Install Additional Dependencies**

   Install other required packages:

   ```bash
   pip install -r requirements.txt

   ```
   ### **Usage**
Run :

```bash
python src/ocr_actif.py  
python src/ocr_passif.py  
```

