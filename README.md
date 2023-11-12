Link Streamlit Cloud : 
- [Dashboard(Single Data)](https://datasayens-dsw2023.streamlit.app)
- [Dashboard(Can Upload excel file for batch prediction)](https://datasayen-dsw2023s-batch-upload.streamlit.app)

# Setup environment

```
conda create --name main-ds python=3.10.12
conda activate main-ds
pip install -r requirements.txt
```

# If you want to skip Setup Environment
```
pip install -r requirements.txt
```

# Run Main steamlit app

```
streamlit dashboard_with_upload_excel.py
```

# Run Alternative steamlit app

```
streamlit run dashboard.py
```