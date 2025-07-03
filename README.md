# 🧠 Sales_Chat_SQL_Bot

Sales managers often struggle to explore and interpret sales data efficiently due to the technical complexity of SQL queries, limited self-service analytics tools, and the absence of interactive dashboards that adapt to natural language queries. This limits timely insights and hinders data-driven decision-making across teams.

> ⚡ Ask questions like:
> - *"Plot sales for INDOMIE PULL in NORTH 1 for February 2025"*
> - *"plot sales value for indomie in the branches NORTH 1 and EAST 3"*
> - *"Show top 3 SKUs by revenue"*
> - *"Show top 3 SKUs by Order Quantity"*
> - *"Show top 3 sales Person by revenue"*

---

## 🗂️ Project Structure
```
Sales_Chat_bot/
├── cbot/
│   ├── app_model_context2.py         # Streamlit app entry point
│   ├── chatbot_model_context3.py     # Core LLM interface logic
│   ├── cross_sell_model_context.py   # Context logic for cross-sell suggestions
│   ├── region_model_context.py       # Regional insights
│   ├── top_model_context.py          # Top-selling brands/products
│   ├── region_normalizer.py          # Preprocessing helper
│   ├── sales_forecaster.py           # Time series forecasting logic
│   ├── utils_helpers_model_context.py# Shared utilities
│   ├── sales_rt.csv                  # 🔽 (Download separately)
│   ├── sales_rt.parquet              # 🔽 (Download separately)
│   ├── dummy_sales_rt.parquet        # ✅ (Light sample for testing)
│   ├── metayb-logo.png               # Logo used in sidebar
│   └── requirements.txt              # Project dependencies
├── cbot_backup/                      # Backup of important files
└── README.md                         # Project documentation

```


---

## 📥 Download Required Data Files

This project uses large datasets which are not included in GitHub due to file size limits.

📦 Download the following files manually from this [Google Drive folder](https://drive.google.com/drive/folders/16M1jhAAlE9HgTqVnlYfDA69djVOIwVE7?usp=sharing):

- `sales_rt.csv` (~425MB)
- `sales_rt.parquet` (~52MB)

> After downloading, place them inside the `cbot/` directory.

---

## ⚙️ Setup Instructions

Follow the steps below to run the chatbot locally:

### 1 Clone the Repository

```bash
git clone https://github.com/subhaorku/Sales_Chat-BI_bot.git
cd Sales_Chat-BI_bot/cbot
```
### 2 Set Up a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Mac/Linux

```
### 3 Install Dependencies
```bash
pip install -r requirements.txt
```
### 4 Place the Data Files
```
cbot/
├── sales_rt.csv
└── sales_rt.parquet
```

### 5 Run the Streamlit Chatbot
From inside the cbot/ directory:
```bash
streamlit run app_model_context2.py
```
### 6  Visit: http://localhost:8501

### Example Prompts
"Show trend for INDOMIE PULL in NORTH 1"

"Forecast sales for DANO next 3 weeks"

"What are the top 3 brands in EAST last month?"

"Compare sales of COLGATE and DANO in SOUTH"

### Some Results
![image](https://github.com/user-attachments/assets/911d4947-38cf-4316-875d-977a55750a7a)

![image](https://github.com/user-attachments/assets/ba6735ca-d602-453c-ab48-b6d928f2733e)

![image](https://github.com/user-attachments/assets/09c86f4c-8832-4ec0-8587-4ad645b2ce29)
![image](https://github.com/user-attachments/assets/ee4ddfd4-3656-4316-8955-9a07708e3ae1)

![image](https://github.com/user-attachments/assets/43ab6e28-e3c2-4766-8ce4-4338aa143b3e)
