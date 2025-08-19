# 🐍 Python Projects Portfolio

Welcome to my Python projects repository!
This collection showcases my hands‑on work in **software development, data analysis, automation, and GUI/game development**.

> **What’s new (Aug 19, 2025):** README refreshed to include **new folders** (Snake/Pong/Turtle, GUI apps, bots), highlight **portfolio‑worthy projects**, and provide clean run instructions.

---

## ⭐ Featured Projects (Resume‑Ready)

These are the most impactful, unique, and fun projects that best showcase problem‑solving, code quality, and creativity.

### 1) 📩 Email Spam Classifier (NLP + ML)

Classifies emails as spam/ham using text preprocessing (tokenization, TF‑IDF) and Naive Bayes; includes evaluation metrics.

* **Skills:** NLP preprocessing, scikit‑learn pipelines, model evaluation
* **Tech:** Python, scikit‑learn, NLTK
* **Run:** `python email-spam-classifier.py`
* **Code:** [email-spam-classifier.py](email-spam-classifier.py)

### 2) 🎬 Netflix Rating Predictor (Regression)

Predicts movie ratings with feature engineering and regression models; includes train/test split and error analysis.

* **Skills:** EDA, feature prep, regression, metrics
* **Data:** [Netflix\_Dataset\_Movie.csv](Netflix_Dataset_Movie.csv)
* **Run:** `python Netflix_Rating_Predicto.py`
* **Code:** [Netflix\_Rating\_Predicto.py](Netflix_Rating_Predicto.py)

### 3) 🌍 Global Temperature Analysis (Data Storytelling)

Explores historical climate trends, warming rates, and seasonal patterns with clear visuals.

* **Skills:** Pandas, time‑series EDA, Matplotlib/Seaborn, insight narration
* **Outputs:** [global\_temperature\_trend.png](global_temperature_trend.png), [seasonal\_temperature\_patterns.png](seasonal_temperature_patterns.png), [temperature\_distribution\_by\_decade.png](temperature_distribution_by_decade.png), [temperature\_anomalies\_by\_region.png](temperature_anomalies_by_region.png)
* **Run:** `python global-temperature-analysis.py`
* **Code:** [global-temperature-analysis.py](global-temperature-analysis.py)

### 4) 📊 Car Sales Analysis (Business Insights)

Analyzes pricing & sales patterns; demonstrates data cleaning, grouping, and visualization.

* **Skills:** Pandas, aggregation, plotting, reporting
* **Data:** [Car\_sales.csv](Car_sales.csv)
* **Run:** `python car-sales-analysis.py`
* **Code:** [car-sales-analysis.py](car-sales-analysis.py)

### 5) 🖼️ Image Watermarking Desktop App (GUI + PIL)

Desktop tool to add watermarks to images (batch/single).

* **Skills:** Tkinter GUI, Pillow image ops, file dialogs
* **Folder:** [`image_watermark_desktop.app`](image_watermark_desktop.app)

### 6) 🐍🎮 Snake / Pong / Turtle Crossing — **Arcade Trio** (OOP + Animation)

Three polished classics showcasing OOP design, animation loops, and collision handling.

* **Snake:** [`snakegame1`](snakegame1), [`snakegame2`](snakegame2)
* **Pong:** [`build pong arcage game`](build%20pong%20arcage%20game)
* **Turtle Crossing:** [`turtlecrossingcapstone`](turtlecrossingcapstone)
* **Skills:** OOP, inheritance, composition, game loops, event handling (turtle)

### 7) 🧠 Flashcard App (Learning Tool, State & Persistence)

A GUI flashcard app for spaced practice.

* **Skills:** Tkinter, state management, timers, file I/O
* **Folder:** [`flashcardapp`](flashcardapp)

### 8) 🦖 Google Dinosaur Game Bot (Automation)

Automates Chrome Dino gameplay for high scores.

* **Skills:** Automation, image/key events, timing
* **Folder:** [`googledinogame`](googledinogame)

---

## 📚 Project Index

### 🎮 Games & GUI

* [`snakegame1`](snakegame1), [`snakegame2`](snakegame2) — Snake (Parts 1 & 2)
* [`build pong arcage game`](build%20pong%20arcage%20game) — Pong (arcade)
* [`turtlecrossingcapstone`](turtlecrossingcapstone) — Turtle Crossing Capstone
* [`turtle&gui`](turtle%26gui) — Turtle & GUI practice
* [`flashcardapp`](flashcardapp) — Flashcard learning app
* [`quiz`](quiz) — GUI Quiz App
* [`higherlower`](higherlower) — Higher/Lower (logic & state)
* [`blackjack`](blackjack) — Blackjack CLI/GUI

### 🤖 Automation & Bots

* [`googledinogame`](googledinogame) — Dino Game Bot
* [`coffeemachine`](coffeemachine) — Procedural coffee machine simulation
* [`coffeemakeradvance`](coffeemakeradvance) — OOP coffee machine (resources, menu, coins)

### 📈 Data & ML

* [email-spam-classifier.py](email-spam-classifier.py) — Email Spam Classifier (NLP)
* [Netflix\_Rating\_Predicto.py](Netflix_Rating_Predicto.py) — Netflix Rating Predictor
* [global-temperature-analysis.py](global-temperature-analysis.py) — Global Temperature EDA
* [car-sales-analysis.py](car-sales-analysis.py) — Car Sales EDA
* [datavisualized-using-csv.py](datavisualized-using-csv.py) — Generic CSV → charts

### 🧰 Utilities & CLI

* [todolist.py](todolist.py) — To‑Do (file persistence)
* [password.py](password.py) — Password Generator
* [calculator.py](calculator.py) / [calulator.py](calulator.py) — Calculators
* [currency-converter.py](currency-converter.py), [temperature-converter.py](temperature-converter.py)
* [student-grade-calculator.py](student-grade-calculator.py)
* [Number\_Guessin\_Game.py](Number_Guessin_Game.py), [hangman-game.py](hangman-game.py), [rockpaper.py](rockpaper.py), [tipcalculator.py](tipcalculator.py), [treasuregame.py](treasuregame.py)
* Pattern practice: [repetitive.py](repetitive.py), [anothermethodfor-repetativenumbers.py](anothermethodfor-repetativenumbers.py), [scopeandnumber](scopeandnumber)
* Fun generators: [bandname.py](bandname.py)

---

## 🛠 Tech Stack

* **Language:** Python
* **Core:** `os`, `time`, `random`, `math`, `string`
* **Data/ML:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`
* **GUI & Imaging:** `tkinter`, `turtle`, `Pillow`
* **Tools:** Jupyter, Git

---

## ▶️ How to Run (General)

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install common deps (only if project requires them)
pip install -r requirements.txt  # if provided
# or install selectively, e.g.
pip install pandas numpy matplotlib seaborn scikit-learn nltk pillow

# 3) Run a project
python path/to/script.py
```

> Some GUI/game projects live in folders. Open the folder for any additional assets or instructions.

---

## 🖼️ Visuals

* Climate EDA figures:
  [global\_temperature\_trend.png](global_temperature_trend.png) · [seasonal\_temperature\_patterns.png](seasonal_temperature_patterns.png) · [temperature\_distribution\_by\_decade.png](temperature_distribution_by_decade.png) · [temperature\_anomalies\_by\_region.png](temperature_anomalies_by_region.png)

---

## 📬 Connect

* **LinkedIn:** [https://www.linkedin.com/in/isra-didagur](https://www.linkedin.com/in/isra-didagur)
* **GitHub:** [https://github.com/Isra-didagur](https://github.com/Isra-didagur)

---

