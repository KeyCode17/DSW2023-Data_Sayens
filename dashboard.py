# Start Library
import os
import sys
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, classification_report
# End Library

dirloc = os.path.dirname(os.path.abspath(__file__))
heatdf = pd.read_excel(dirloc + "/dataset/heat.xlsx", index_col=0)
decdf = pd.read_excel(dirloc + "/dataset/decdf.xlsx")

with st.sidebar:
    # Title
    st.title("Team Data Sayens")
    # Using st.form to wrap checkboxes
    with st.form(key='location_form'):
        st.subheader("Choose Location")
        jakarta_selected = st.checkbox("Jakarta", value=True)
        bandung_selected = st.checkbox("Bandung", value=True)

        # Submit button to reload the page
        apply_button = st.form_submit_button(label='Apply')

    # Check if at least one checkbox is selected
    if not jakarta_selected and not bandung_selected:
        st.warning("Please select at least one location.")
    st.subheader("Jump to Analysis Result Section")
    st.markdown("[Call Center](#call-center-comparasion)")
    st.markdown("[Games Product](#games-product-uses-comparasion)")
    st.markdown("[Video Product](#video-product-uses-comparasion)")
    st.markdown("[Education Product](#education-product-uses-comparasion)")
    st.markdown("[Video Product](#video-product-uses-comparasion)")
    st.markdown("[Classification Report](#classification-random-forest-report)")
    st.markdown("[Predict Data](#manual-input-for-prediction)")


if not (jakarta_selected or bandung_selected):
    st.error("Please select at least one location.")
    sys.exit()

# Filter data based on selected locations
if jakarta_selected and bandung_selected:
    filtered_decdf = decdf
elif jakarta_selected:
    filtered_decdf = decdf[decdf['Location'] == 0]
elif bandung_selected:
    filtered_decdf = decdf[decdf['Location'] == 1]
else:
    filtered_decdf = pd.DataFrame()

st.title("Classification Model using Random Forest Algoritm to Predict Customer Decision by Customer Behavior")

# Header
st.header("Differential Of Churn")
# Filter data based on selected locations
if jakarta_selected and bandung_selected:
    st.subheader("Location: Jakarta and Bandung")
elif jakarta_selected:
    st.subheader("Location: Jakarta")
elif bandung_selected:
    st.subheader("Location: Bandung")

sizes = [len(filtered_decdf[filtered_decdf['Churn Label'] == 0]), len(filtered_decdf[filtered_decdf['Churn Label'] == 1])]
        
# Check for zero or negative values in sizes
if any(size <= 0 for size in sizes):
            raise ValueError("Invalid sizes: sizes should be positive and non-zero.")
        
labels = 'Loyal Client', 'Client Left'
colors = ['#003f5c', '#ffa600']
explode = (0.1, 0.2)

fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

# Check if the lengths of colors and explode match sizes
if len(colors) != len(sizes) or len(explode) != len(sizes):
    raise ValueError("Lengths of colors and explode should match the length of sizes.")
        
wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                   explode=explode, shadow=True, startangle=85,
                                   pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

ax_tes.axis('equal')

ax_tes.legend(loc='best', labels=labels)

st.pyplot(fig)

plt.clf()

# Count Plot
ax = sns.countplot(x='Location', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

label = {'No Churn', 'Churn'}

# Notation
for i, p in enumerate(ax.patches):
    if i < len(ax.patches)-len(label):
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                ha='center', fontsize=7, color='white')

# Xticks Label 
# Filter data based on selected locations
if jakarta_selected and bandung_selected:
    var = ['Jakarta', 'Bandung']
elif jakarta_selected:
    var = ['Jakarta']
elif bandung_selected:
    var = ['Bandung']
ax.set_xticklabels(var)

# Label
plt.xlabel('Location')
plt.ylabel('Jumlah Pengguna')
plt.title('Count Plot Location vs Churn Label')

# Legend
ax.legend(title='Churn Label', labels=label, loc='best')

st.pyplot(plt)

plt.clf()

st.header("Analysis Result")
st.subheader("Call Center Comparasion")
st.caption("It can be seen from the display data, that the service of the call center is not the cause of customers leaving the company, even when customers use call center services, it can reduce the number of customers leaving the company.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use Call Center]", "[Use Call Center]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Call Center', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Call Center')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Call Center'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Call Center'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Call Center'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Call Center'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Call Center_No', 'Call Center_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Call Center_No', 'Call Center_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

st.subheader("Games Product Uses Comparasion")
st.caption("It can be seen from the display data, that customers who use internet services on games product tend to have a significantly higher level of loyalty to the company than customers who do not use it as internet services on games product, so it can be estimated that customers are very satisfied with internet services in the use of games product.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use Games Product]", "[Use Games Product]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Games Product', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes', 'No internet service']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Games Product')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Games Product'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Games Product'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Games Product'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Games Product'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Games Product_No', 'Games Product_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Games Product_No', 'Games Product_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

st.subheader("Music Product Uses Comparasion")
st.caption("It can be seen from the display data, that customers who use internet services for music products have the most customers, so it can be estimated that for now, customers are quite satisfied with internet services and it can be said that the purpose of customers using internet services at the company is to use the music product.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use Music Product]", "[Use Music Product]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Music Product', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes', 'No internet service']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Music Product')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Music Product'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Music Product'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Music Product'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Music Product'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Music Product_No', 'Music Product_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Music Product_No', 'Music Product_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

plt.clf()

st.subheader("Education Product Uses Comparasion")
st.caption("It can be seen from the display data that customers who use internet service for educational products are among those who do not have a significant impact on churn rates.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use Education Product]", "[Use Education Product]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Education Product', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes', 'No internet service']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Education Product')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Education Product'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Education Product'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Education Product'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Education Product'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Education Product_No', 'Education Product_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Education Product_No', 'Education Product_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

st.subheader("Use MyApp Uses Comparasion")
st.caption("It can be seen from the display data, that customers who use internet service for Use MyApp, have the most impact on customers changing internet companies, so it is necessary to improve service to customers.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use MyApp]", "[Use MyApp]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Use MyApp', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes', 'No internet service']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Use MyApp')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Use MyApp'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Use MyApp'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Use MyApp'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Use MyApp'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Use MyApp_No', 'Use MyApp_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Use MyApp_No', 'Use MyApp_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

st.subheader("Video Product Uses Comparasion")
st.caption("Just like customers who use myapp, customers who use internet service for video have a strong impact that can make customers leave the company, so it is necessary to improve the video product.")

tab1, tab2, tab3, tab4 = st.tabs(["[Comparasion]", "[Does Not Use Video Product]", "[Use Video Product]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Video Product', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')

    # Xticks Label 
    var = ['No', 'Yes', 'No internet service']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Video Product')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Video Product'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Video Product'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Video Product'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Video Product'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Video Product_No', 'Video Product_Yes', 'Churn Label_No', 'Churn Label_Yes'], ['Video Product_No', 'Video Product_Yes', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")
    plt.title("Grafik Korelasi")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

st.subheader("Payment Method Comparasion")
st.caption("When viewed from each data, customers with pulsa payment methods have the highest impact on leaving the company.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["[Comparasion]", "[By Digital Wallet]", "[By Pulsa]", "[By Debit]", "[By Credit]", "[Correlation]"])
with tab1:
    # Count Plot
    ax = sns.countplot(x='Payment Method', hue='Churn Label', data=filtered_decdf, palette={1: '#ffa600', 0: '#003f5c'})

    label = ['No Churn', 'Churn']

    # Notation
    for i, p in enumerate(ax.patches):
            if i < len(ax.patches)-len(label):
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() / 100 + 50),
                        ha='center', fontsize=8, color='white')
    
    # Xticks Label 
    var = ['Digital Wallet', 'Pulsa', 'Debit', 'Credit']
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(var)

    # Label
    plt.xlabel('Video Product')
    plt.ylabel('Count')

    # Legend
    ax.legend(title='Churn Label', labels=label, loc='best')

    st.pyplot(plt)

    plt.clf()

with tab2:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Payment Method'] == 0)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Payment Method'] == 0)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab3:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Payment Method'] == 1)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Payment Method'] == 1)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab4:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Payment Method'] == 2)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Payment Method'] == 2)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab5:
    sizes = [len(filtered_decdf[(filtered_decdf['Churn Label'] == 0) & (filtered_decdf['Payment Method'] == 3)]),
             len(filtered_decdf[(filtered_decdf['Churn Label'] == 1) & (filtered_decdf['Payment Method'] == 3)])]

    # Check for zero or negative values in sizes
    if any(size <= 0 for size in sizes):
                raise ValueError("Invalid sizes: sizes should be positive and non-zero.")

    labels = 'Loyal Client', 'Client Left'
    colors = ['#003f5c', '#ffa600']
    explode = (0.1, 0.2)

    fig, ax_tes = plt.subplots()  # Use a different variable name for the axis

    # Check if the lengths of colors and explode match sizes
    if len(colors) != len(sizes) or len(explode) != len(sizes):
        raise ValueError("Lengths of colors and explode should match the length of sizes.")

    wedges, texts, autotexts = ax_tes.pie(sizes, autopct='%1.1f%%', colors=colors, 
                                       explode=explode, shadow=True, startangle=85,
                                       pctdistance=0.85, wedgeprops={'width': 1}, textprops={'color': 'white'})

    ax_tes.axis('equal')

    ax_tes.legend(loc='best', labels=labels)

    st.pyplot(fig)

    plt.clf()

with tab6:
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatdf.loc[['Payment Method_Digital Wallet', 'Payment Method_Pulsa', 'Payment Method_Debit', 'Payment Method_Credit', 'Churn Label_No', 'Churn Label_Yes'], ['Payment Method_Digital Wallet', 'Payment Method_Pulsa', 'Payment Method_Debit', 'Payment Method_Credit', 'Churn Label_No', 'Churn Label_Yes']], cmap='coolwarm', linewidths=.5, annot=True, fmt=".2f")

    # Miringkan label sumbu x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(plt)

    plt.clf()

nama_feature = ['Tenure Months', 'Device Class',
                'Games Product', 'Music Product',
                'Education Product', 'Use MyApp',
                'Video Product', 'Monthly Purchase (Thou. IDR)',
                'Call Center', 'CLTV (Predicted Thou. IDR)', 'Payment Method']

X = decdf[nama_feature].values
y = decdf['Churn Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load the saved model
model = joblib.load('random_forest_model.joblib')

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrik evaluasi untuk data latih
accuracy_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
roc_auc_train = roc_auc_score(y_train, y_train_pred)

# Metrik evaluasi untuk data tes
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred)

# Classification report untuk data tes
classification_report_test = classification_report(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, output_dict=True)

# Convert report to dataframe
df_classification_report = pd.DataFrame(report).transpose()

# Plot heatmap
st.header("Classification Random Forest Report")
plt.figure(figsize=(10, 5))
sns.heatmap(df_classification_report.iloc[:-1, :].T, annot=True, cmap="Blues")
st.pyplot(plt)

st.text("Data Train:")
st.caption(f"Train Accuracy: {accuracy_train:.2f}")
st.caption(f"Train F1 Score: {f1_train:.2f}")
st.caption(f"Train Precision: {precision_train:.2f}")
st.caption(f"Train Recall: {recall_train:.2f}")
st.caption(f"Train ROC AUC: {roc_auc_train:.2f}")

st.text("\nData Test:")
st.caption(f"Test Accuracy: {accuracy_test:.2f}")
st.caption(f"Test F1 Score: {f1_test:.2f}")
st.caption(f"Test Precision: {precision_test:.2f}")
st.caption(f"Test Recall: {recall_test:.2f}")
st.caption(f"Test ROC AUC: {roc_auc_test:.2f}")

st.header("Manual Input for Prediction")

with st.form(key='my_form'):
    # Add input fields for each feature
    tenure_months = st.number_input("Tenure Months", min_value=0, step=1, value=1)
    device_class = st.selectbox("Device Class", ['Low', 'Medium', 'High'], index=0)
    games_product = st.selectbox("Games Product", ['No', 'Yes', 'No internet service'], index=0)
    music_product = st.selectbox("Music Product", ['No', 'Yes', 'No internet service'], index=0)
    education_product = st.selectbox("Education Product", ['No', 'Yes', 'No internet service'], index=0)
    use_myapp = st.selectbox("Use MyApp", ['No', 'Yes', 'No internet service'], index=0)
    video_product = st.selectbox("Video Product", ['No', 'Yes', 'No internet service'], index=0)
    monthly_purchase = st.number_input("Monthly Purchase", min_value=0, step=1, value=1)
    call_center = st.selectbox("Call Center", ['No', 'Yes'], index=0),
    CLTV = st.number_input("CLTV (Predicted Thou. IDR)", min_value=0, step=1, value=1), 
    payment_method = st.selectbox("Payment Method", ['Digital Wallet', 'Pulsa', 'Debit', 'Credit'], index=0)

    # Submit button
    submitted = st.form_submit_button(label='Apply')

# Only execute the following code if the form is submitted
if submitted:
    # Create a feature vector based on user input
    user_input = [
        tenure_months,
        device_class,
        games_product,
        music_product,
        education_product,
        use_myapp,
        video_product,
        monthly_purchase,
        call_center,
        CLTV,
        payment_method
    ]

    # Convert categorical features to numerical values
    device_class_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    games_product_mapping = {'No': 0, 'Yes': 1, 'No internet service': 3}
    music_product_mapping = {'No': 0, 'Yes': 1, 'No internet service': 3}
    education_product_mapping = {'No': 0, 'Yes': 1, 'No internet service': 3}
    use_myapp_mapping = {'No': 0, 'Yes': 1, 'No internet service': 3}
    video_product_mapping = {'No': 0, 'Yes': 1, 'No internet service': 3}
    call_center_mapping = {'No': 0, 'Yes': 1}
    payment_method = {"Digital Wallet": 0, "Pulsa": 1, "Debit": 2, "Credit":3}
    user_input[1] = device_class_mapping[user_input[1]]
    user_input[2] = games_product_mapping[user_input[2]]
    user_input[3] = music_product_mapping[user_input[3]]
    user_input[4] = education_product_mapping[user_input[4]]
    user_input[5] = use_myapp_mapping[user_input[5]]
    user_input[6] = video_product_mapping[user_input[6]]
    user_input[8] = call_center_mapping[user_input[8]]
    user_input[10] = call_center_mapping[user_input[10]]

    # Make prediction
    prediction = model.predict([user_input])[0]

    st.header("Prediction Result")
    if prediction == 0:
        st.success("The model predicts 'No Churn'.")
    else:
        st.error("The model predicts 'Churn'.")