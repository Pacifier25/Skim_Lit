# 📄 SkimLit Classifier: Making Research Abstracts Easier to Read

SkimLit Classifier is an **NLP-powered tool** designed to break down complex and lengthy research abstracts into skimmable and easy-to-read content. This project leverages **state-of-the-art BERT models** to classify sentences in a research abstract into predefined categories, enabling faster understanding and better content organization.

---

## 🔍 Problem Statement

When exploring research abstracts, it's often challenging to quickly identify the role each sentence plays, such as **OBJECTIVE**, **METHODS**, **RESULTS**, **CONCLUSIONS**, or **BACKGROUND**. Traditional abstracts are dense and require readers to sift through large chunks of text.

The **SkimLit Classifier** simplifies this process by:

1. Splitting the abstract into sentences.
2. Automatically predicting the role (class) of each sentence.
3. Presenting the information in a structured and skimmable format.

---

## 🚀 Goal of the Project

The **SkimLit Classifier** aims to:
- Assist researchers and readers in **quickly skimming abstracts**.
- Improve the readability of dense text by **highlighting sentence roles**.
- Leverage modern NLP techniques to automate classification tasks.

In essence, the model helps to answer the question:  
*"What role does each sentence serve in a research abstract?"*

---

## 🌟 Features

- **Upload PDF or Enter Text**: The app supports both PDF uploads and direct text input for classification.
- **Automated Sentence Classification**: Automatically assigns labels to each sentence in the input.
- **Visually Enhanced Interface**: Aesthetic and easy-to-use UI built with **Streamlit**.
- **Predefined Categories**:
  - 📘 `BACKGROUND`
  - 🎯 `OBJECTIVE`
  - 🧪 `METHODS`
  - 📝 `RESULTS`
  - ✅ `CONCLUSIONS`

---

## 💻 Underlying Technology

- **Natural Language Processing (NLP)**:
  - Model: Fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)**.
  - Framework: Hugging Face Transformers with TensorFlow.

- **Streamlit**: Interactive web interface for user inputs and results display.

- **PyPDF2**: For extracting text from PDF files.

---

## 🖼 Example Workflow

### Input (Dense Abstract):
Mental illness, including depression, anxiety, and bipolar disorder, accounts for a significant proportion of global disability...


### Output (Classified Abstract):
- **BACKGROUND**: Mental illness, including depression, anxiety, and bipolar disorder, accounts for a significant proportion of global disability.
- **OBJECTIVE**: To investigate the efficacy of a new intervention in managing mental health disorders.
- **METHODS**: A randomized controlled trial was conducted with 200 participants over a 12-week period.
- **RESULTS**: The treatment group showed significant improvement compared to the control group.
- **CONCLUSIONS**: These findings suggest that the intervention is effective for managing mental health issues.

---

## 📂 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/skimlit-classifier.git
cd skimlit-classifier
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Pretrained Weights
```bash
Download skimlit_model.h5 and place it in the project directory.
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## 📊 Dataset
This project is inspired by the **PubMed 200k RCT dataset**, which contains labeled Randomized Controlled Trial (RCT) abstracts.
The dataset is used to explore how NLP models can classify sentences based on their role in an abstract.


---

## 🎨 App Design Highlights
* **Light and Friendly Background**: The app features a soothing light grayish-blue background for better readability.
* **Two-column Layout**: For seamless PDF uploads and text input.
* **Interactive Buttons and Outputs**: Provides real-time predictions with a clear display of results.

---


## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

---


## 🤝 Contributing
Contributions are always welcome! If you'd like to contribute, please:

1. Fork this repository.
2. Create a branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m "Add new feature").
4. Push to your branch (git push origin feature-branch).
5. Submit a Pull Request.

## ✉️ Contact
For questions, feedback, or collaborations, reach out via:

Email: Piyushbist10@gmail.com
GitHub: Pacifier25
