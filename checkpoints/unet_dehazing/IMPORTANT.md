Here’s a clean and professional snippet you can copy into your **README.md** 👇

---

### 🧠 Model Setup

This project requires a pre-trained model file (`model.pth`) which is not included in the repository due to size limitations.

#### 🔽 Step 1: Download the Model

Download the `model.pth` file from the [**Releases**](../../releases) section of this repository.

#### 📁 Step 2: Place the File

After downloading, place the file in the following directory:

```
project_root/
└── checkpoints/
    └── unet_dehazing
        └── unet_best.pth
```

> **Note:** If the `models` folder does not exist, create it manually.

#### ✅ Step 3: Verify

Once placed correctly, the project should automatically load the model when running the application.

---

Would you like me to adjust this for your exact folder structure (for example, if your model goes inside `src/models/` or somewhere else)?
